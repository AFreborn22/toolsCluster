from flask import Flask, render_template, request, send_file, url_for, Response
import pandas as pd
import os
import io
import base64
import uuid
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from models.randomForest import trainRandomForest
from models.logisticRegression import trainLogisticRegression
from models.svm import trainSVM
from models.decisionTree import trainDecisionTree
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from services.gcs import upload_to_gcs, read_csv_from_gcs, save_file_to_gcs, download_from_gcs
from config import Config
import matplotlib.pyplot as plt

load_dotenv()

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config.from_object(Config)

# Fungsi untuk preprocessing text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', error="No file uploaded for clustering.")

        try:
            # Simpan file dan proses CSV
            unique_id = str(uuid.uuid4())  
            file_extension = file.filename.split('.')[-1]
            filename = f"{unique_id}.{file_extension}"
            upload_to_gcs(file.stream, f"uploads/{file.filename}") 
            
            tiktok = read_csv_from_gcs(f"uploads/{file.filename}")
            tiktokData = tiktok[['uniqueId', 'text', 'createTimeISO']]
            tiktokData.dropna(inplace=True)
            tiktokData.drop_duplicates(inplace=True)

            # Preprocessing text
            tiktokData.loc[:, 'textClean'] = tiktokData['text'].apply(preprocess_text)

            # Stemming dengan caching
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()

            unique_words = set()
            tiktokData['textClean'].str.split().apply(unique_words.update)

            stemmed_words = {word: stemmer.stem(word) for word in unique_words}

            def stemWithDict(text):
                return ' '.join([stemmed_words.get(word, word) for word in text.split()])

            tiktokData.loc[:, 'textClean'] = tiktokData['text'].apply(stemWithDict)
            tiktokData.dropna(inplace=True)

            # Vectorization
            vectorData = tiktokData['textClean']
            vectorizer = TfidfVectorizer().fit_transform(vectorData)
            cosine_sim = cosine_similarity(vectorizer)

            cosineSim = pd.DataFrame(cosine_sim, columns=tiktokData.index, index=tiktokData.index)
            tiktokData['max_cosine_sim'] = cosineSim.apply(lambda row: row.drop(row.name).max(), axis=1)

            # Threshold for similar comments
            threshold = 0.7
            tiktokData['similar_comments_count'] = cosineSim.apply(lambda row: (row > threshold).sum() - 1, axis=1)

            # Menghitung jumlah komentar per akun
            tiktokData['comment_count'] = tiktokData.groupby('uniqueId')['uniqueId'].transform('count')

            # Mengonversi kolom 'createTimeISO' menjadi format datetime
            tiktokData['createTimeISO'] = pd.to_datetime(tiktokData['createTimeISO'])

            # Menghitung time_diff hanya untuk akun dengan lebih dari satu komentar
            tiktokData['time_diff'] = 0
            comment_counts = tiktokData['uniqueId'].value_counts()
            multiple_comment_accounts = comment_counts[comment_counts > 1].index

            tiktokData.loc[tiktokData['uniqueId'].isin(multiple_comment_accounts), 'time_diff'] = (
                tiktokData.groupby('uniqueId')['createTimeISO']
                .diff().dt.total_seconds()
                .fillna(0.1)
            )

            # Clustering
            features = ['max_cosine_sim', 'similar_comments_count', 'comment_count', 'time_diff']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(tiktokData[features])

            kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=42)
            tiktokData['cluster'] = kmeans.fit_predict(scaled_features)

            # tiktokData['cluster'] = tiktokData['cluster'].map({
            #     0: 'Natural Comment',
            #     1: 'Buzzer / Bot'
            # })
            cluster_summary = tiktokData.groupby('cluster')[features].mean()
            
            # Pemetaan dinamis berdasarkan pola data
            if cluster_summary.loc[0, 'similar_comments_count'] > cluster_summary.loc[1, 'similar_comments_count']:
                mapping = {0: 'Buzzer / Bot', 1: 'Natural Comment'}
            else:
                mapping = {0: 'Natural Comment', 1: 'Buzzer / Bot'}
            
            tiktokData['cluster'] = tiktokData['cluster'].map(mapping)

            # Menghitung jumlah komentar per cluster
            cluster_counts = tiktokData['cluster'].value_counts().to_dict()

            # Data untuk pie chart dan bar chart
            pie_data = [cluster_counts.get('Natural Comment', 0), cluster_counts.get('Buzzer / Bot', 0)]
            bar_data = [cluster_counts.get('Natural Comment', 0), cluster_counts.get('Buzzer / Bot', 0)]

            # Menyediakan contoh kalimat untuk tabel
            exampleNatural = tiktokData[tiktokData['cluster'] == 'Natural Comment'].sample(10)['text'].tolist()
            exampleBot = tiktokData[tiktokData['cluster'] == 'Buzzer / Bot'].sample(10)['text'].tolist()
            
            topComments = (
                tiktokData[tiktokData['cluster'] == 'Buzzer / Bot']  
                .groupby('textClean')
                .size()
                .reset_index(name='count')  
                .sort_values(by='count', ascending=False)
                .head()  
            )
            
            buzzer_data = tiktokData[tiktokData['cluster'] == 'Buzzer / Bot']
            
            topAccounts = (
                buzzer_data
                .groupby('uniqueId', as_index=False)  
                .agg({
                    'comment_count': 'sum',  
                    'text': 'first'  
                })
                .query('comment_count > 1')  
                .sort_values(by='comment_count', ascending=False)  
                .head()  
            )
                
            print("Top Comments:", topComments)
            print("Top Accounts:", topAccounts)
            
            topComments = topComments.to_records(index=False).tolist()
            topAccounts = topAccounts.to_records(index=False).tolist()
            
            # Simpan hasil klastering ke file CSV
            filename = f"{unique_id}_analysisResult.csv"
            save_file_to_gcs(tiktokData, f"results/{filename}")
            
            download_url = url_for('download_file', filename=f"{unique_id}_analysisResult.csv")
            
            return render_template(
                'index.html',
                download_url=download_url,
                exampleBot=exampleBot,
                exampleNatural=exampleNatural,
                pie_data=pie_data,
                bar_data=bar_data,
                topComments=topComments,
                topAccounts=topAccounts
            )
            
        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {e}")

    return render_template('index.html')

def plot_confusion_matrix(cm, model_name):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar()
    tick_marks = range(len(cm))

    # Menambahkan angka di dalam setiap sel matriks
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, format(cm[i, j], 'd'), 
                     horizontalalignment='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return image

@app.route('/modeling', methods=['GET', 'POST'])
def modeling():
    if request.method == 'POST':
        # Ambil file dari form
        file = request.files.get('clustered_file')
        if not file or file.filename == '' or not file.filename.endswith('.csv'):
            return render_template('modeling.html', error="Invalid file type. Please upload a CSV file.")

        try:
            # Generate unique ID dan simpan ke GCS
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}_clustered_data.csv"
            upload_to_gcs(file.stream, f"uploads/{filename}")

            # Baca file dari GCS
            tiktokData = read_csv_from_gcs(f"uploads/{filename}")

            # Validasi kolom yang diperlukan
            required_columns = ['max_cosine_sim', 'similar_comments_count', 'comment_count', 'time_diff', 'cluster']
            if not all(col in tiktokData.columns for col in required_columns):
                return render_template('modeling.html', error="Uploaded file does not contain the required columns.")

            # Ambil fitur dan label
            X = tiktokData[['max_cosine_sim', 'similar_comments_count', 'comment_count', 'time_diff']]
            y = tiktokData['cluster']

            # Normalisasi dan balancing data menggunakan SMOTE
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

            # Split data menjadi training dan testing
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.3, random_state=42)

            # Model yang tersedia
            models = {
                'Logistic Regression': trainLogisticRegression,
                'Random Forest': trainRandomForest,
                'SVM': trainSVM,
                'Decision Tree': trainDecisionTree
            }

            # Cek input model dari pengguna
            model_type = request.form['model_type']
            if model_type not in models and model_type != 'all':
                return render_template('modeling.html', error="Invalid model type selected.")

            results = []

            try:
                # Loop melalui model atau pilih satu model saja
                if model_type == 'all':  
                    for name, func in models.items():
                        model, accuracy, report, cv_report, cm = func(X_train, X_test, y_train, y_test)
                        cm_image = plot_confusion_matrix(cm, name)
                        model_filename = f"{unique_id}_{name.replace(' ', '_')}.pkl"
                        save_file_to_gcs(model, f"models/{model_filename}")
                        results.append({
                            'name': name,
                            'accuracy': accuracy,
                            'report': report,
                            'cv_report': cv_report,
                            'cm_image': cm_image,
                            'download_url': url_for('download_model', filename=model_filename)
                        })
                else:
                    func = models[model_type]
                    model, accuracy, report, cv_report, cm = func(X_train, X_test, y_train, y_test)
                    cm_image = plot_confusion_matrix(cm, model_type)
                    model_filename = f"{unique_id}_{model_type.replace(' ', '_')}.pkl"
                    save_file_to_gcs(model, f"models/{model_filename}")
                    results.append({
                        'name': model_type,
                        'accuracy': accuracy,
                        'report': report,
                        'cv_report': cv_report,
                        'cm_image': cm_image,
                        'download_url': url_for('download_model', filename=model_filename)
                    })

            except Exception as e:
                return render_template('modeling.html', error=f"An error occurred during model training: {e}")

            # Render hasil ke template HTML
            return render_template('modeling.html', results=results)

        except Exception as e:
            return render_template('modeling.html', error=f"Error processing file: {e}")

    # Tampilkan halaman modeling jika GET request
    return render_template('modeling.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        blob_name = f"results/{filename}"
        return download_from_gcs(blob_name)
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", error="File not found or download failed.")
    
@app.route('/download_model/<filename>')
def download_model(filename):
    try:
        blob_name = f"models/{filename}"
        return download_from_gcs(blob_name)
    except Exception as e:
        return render_template("modeling.html", error="File not found or download failed.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
from google.cloud import storage
from config import Config
import io
import pandas as pd
from flask import send_file
import pickle

storage_client = storage.Client()

def upload_to_gcs(file_obj, destination_blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_obj)  
    print(f"File uploaded to {destination_blob_name} in GCS.")

def read_csv_from_gcs(blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()  
    return pd.read_csv(io.StringIO(data))

def save_file_to_gcs(data, destination_blob_name, content_type):
    """
    Menyimpan data ke Google Cloud Storage.

    Args:
    - data: Data yang ingin disimpan (bisa berupa DataFrame atau model).
    - destination_blob_name (str): Nama file tujuan di bucket GCS.
    - content_type (str): Tipe konten file, seperti 'text/csv' atau 'application/octet-stream' untuk binary files.
    """
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    
    if content_type == 'text/csv':
        csv_data = data.to_csv(index=False)
        blob.upload_from_string(csv_data, content_type=content_type)
    elif content_type == 'application/octet-stream':  
        model_data = pickle.dumps(data)
        blob.upload_from_string(model_data, content_type=content_type)
    
    print(f"File '{destination_blob_name}' berhasil disimpan di GCS.")
    
def download_from_gcs(blob_name):
    """
    Mengunduh file dari Google Cloud Storage dan mengembalikannya sebagai respons Flask.
    """
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    # Unduh data sebagai bytes
    csv_data = blob.download_as_bytes()

    # Masukkan ke dalam BytesIO untuk send_file
    file_obj = io.BytesIO(csv_data)
    file_obj.seek(0)  # Pastikan pointer file ada di awal

    # Gunakan send_file untuk mengirim data
    return send_file(
        file_obj,
        mimetype='text/csv',
        as_attachment=True,
        download_name=blob_name.split("/")[-1]
    )
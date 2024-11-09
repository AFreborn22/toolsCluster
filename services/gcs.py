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

def save_analysis_to_gcs(tiktokData, destination_blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    csv_data = tiktokData.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')
    
def save_model_to_gcs(model, destination_blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    pickle_data = pickle.dumps(model)  
    blob.upload_from_string(pickle_data, content_type='application/octet-stream')
    print(f"Model uploaded to {destination_blob_name} in GCS.")
    
def download_from_gcs(blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    # Unduh data sebagai bytes
    csv_data = blob.download_as_bytes()

    # Masukkan ke dalam BytesIO untuk send_file
    file_obj = io.BytesIO(csv_data)
    file_obj.seek(0)  

    return send_file(
        file_obj,
        mimetype='text/csv',
        as_attachment=True,
        download_name=blob_name.split("/")[-1]
    )
    
def download_model_from_gcs(blob_name):
    """Download pickled model from GCS and return as BytesIO."""
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    model_data = blob.download_as_bytes()  
    return io.BytesIO(model_data) 
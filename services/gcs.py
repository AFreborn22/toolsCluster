from google.cloud import storage
from config import Config
import io
import pandas as pd
from flask import send_file

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
    """
    Menyimpan DataFrame ke Google Cloud Storage sebagai file CSV.
    
    Args:
    - tiktokData (pd.DataFrame): DataFrame hasil analisis yang ingin disimpan.
    - destination_blob_name (str): Nama file tujuan di bucket GCS.
    """
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    
    csv_data = tiktokData.to_csv(index=False)
    blob.upload_from_string(csv_data, content_type='text/csv')
    print(f"File '{destination_blob_name}' berhasil disimpan di GCS.")
    
def download_from_gcs(blob_name):
    """
    Mengunduh file dari Google Cloud Storage dan mengembalikannya sebagai respons Flask.
    
    Args:
    - blob_name (str): Nama file di bucket GCS.
    
    Returns:
    - send_file: File yang dapat diunduh oleh user.
    """
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    
    csv_data = blob.download_as_bytes()
    
    return send_file(
        io.BytesIO(csv_data),
        mimetype='text/csv',
        as_attachment=True,
        download_name=blob_name
    )
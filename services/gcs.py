from google.cloud import storage
from config import Config
import io
import pandas as pd

storage_client = storage.Client()

def upload_to_gcs(file_obj, destination_blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_obj)  # Mengunggah langsung dari stream
    print(f"File uploaded to {destination_blob_name} in GCS.")

def read_csv_from_gcs(blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()  # Mengunduh data sebagai teks
    return pd.read_csv(io.StringIO(data))

def download_from_gcs(source_blob_name, destination_file_path):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)
    print(f"Blob {source_blob_name} downloaded to {destination_file_path}.")
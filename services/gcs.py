from google.cloud import storage
from config import Config

storage_client = storage.Client()

def upload_to_gcs(source_file_path, destination_blob_name):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)
    print(f"File {source_file_path} uploaded to {destination_blob_name}.")

def download_from_gcs(source_blob_name, destination_file_path):
    bucket = storage_client.bucket(Config.GCS_BUCKET_NAME)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)
    print(f"Blob {source_blob_name} downloaded to {destination_file_path}.")
import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "secret_key")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "my-app-uploads-bucket")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "my-credentials.json")
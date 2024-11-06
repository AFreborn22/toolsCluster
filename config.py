import os

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "secret_key")
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "my-app-uploads-bucket")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "my-credentials.json")
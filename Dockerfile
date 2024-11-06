# Gunakan base image Python
FROM python:3.9-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin file dependencies dan install
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari direktori build ke dalam container
COPY . .

# Salin file kredensial dari konteks build ke container
COPY /workspace/my-credentials.json /app/my-credentials.json

# Set variabel lingkungan untuk file kredensial Google
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/my-credentials.json"

# Ekspos port 8080
EXPOSE 8080

# Tentukan perintah untuk menjalankan aplikasi
CMD ["python", "app.py"]
# Gunakan image dasar Python versi slim untuk ukuran image yang lebih kecil
FROM python:3.9-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Salin semua file proyek ke direktori kerja
COPY . .

# Salin file .env agar variabel lingkungan dapat diakses di dalam container
COPY .env .env

# Salin file kredensial Google Cloud ke container
COPY my-credentials.json /app/my-credentials.json

# Set variabel lingkungan untuk kredensial Google Cloud
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/my-credentials.json"

# Tentukan port yang akan digunakan (pastikan port ini sesuai dengan konfigurasi Flask)
EXPOSE 8080

# Tentukan command untuk menjalankan aplikasi
CMD ["python", "app.py"]    
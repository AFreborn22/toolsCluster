FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

COPY .env .env

COPY my-credentials.json /app/my-credentials.json

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/my-credentials.json"

EXPOSE 8080

# Tentukan command untuk menjalankan aplikasi
CMD ["python", "app.py"]    
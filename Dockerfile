# Use Python base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies file and install
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other files
COPY . .

# Copy the credentials file from build context to container
COPY my-credentials.json /app/my-credentials.json

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/my-credentials.json"

EXPOSE 8080

CMD ["python", "app.py"]
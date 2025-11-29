From python:3.10-slim

WORKDIR /app

# Copy the dependency list and install them
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the core application files
COPY app.py .
COPY Procfile .

# Copy configuration and model weights
COPY .env .
COPY yolov8n.pt .

# ... (Add other files and continue with the rest of your Dockerfile, like the CMD)
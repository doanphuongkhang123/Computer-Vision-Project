FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Force Python to send logs immediately to the terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app"

COPY FE/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "-u", "FE/app.py"]
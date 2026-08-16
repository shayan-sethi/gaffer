FROM python:3.10-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (FFmpeg and build essentials)
RUN apk add --no-cache \
    ffmpeg \
    gcc \
    musl-dev \
    linux-headers \
    libstdc++

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ingest.py"]

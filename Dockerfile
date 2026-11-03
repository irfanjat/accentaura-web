# Production Container for AccentAura (Docker & Hugging Face Spaces compatible)
FROM python:3.11-slim

# System audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure user 1000 for Hugging Face Spaces & security
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=user:user . .

# Ensure temp directory exists and is writable
RUN mkdir -p /tmp/accentaura_recordings && chown -R user:user /tmp/accentaura_recordings

USER user
ENV HOME=/home/user
ENV PORT=7860
ENV ACCENTAURA_TEMP_DIR=/tmp/accentaura_recordings
EXPOSE 7860

# Production WSGI server
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 120 --threads 2 backend.app:app"]

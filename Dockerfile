# 1. Base Image
FROM python:3.11-slim

# Set environment performance variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# 2. Set Working Directory
WORKDIR /app

# 3. Install System Dependencies & Update Pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Pre-cache Local FastEmbed ONNX Model (Prevents cold-start delays)
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# 6. Copy Application Source Code
COPY . .

# 7. Configure Non-Root Security User
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# 8. Expose Application Port
EXPOSE 8000

# 9. Healthcheck to verify the microservice is operational
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# 10. Start Production Uvicorn Server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
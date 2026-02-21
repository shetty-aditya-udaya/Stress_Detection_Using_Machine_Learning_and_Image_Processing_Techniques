# Production-grade ML Microservice
FROM python:3.10-slim

# Install system dependencies for OpenCV and ML
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
# Note: Ensure the heavy 'data' folder is ignored or handled separately if needed for training
COPY . .

# Environment variables for silent execution
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# Expose port (Render/Railway use env $PORT)
EXPOSE 8000

# Run with Uvicorn for single-process stability and low-resource overhead
CMD ["sh", "-c", "uvicorn api.backend:app --host 0.0.0.0 --port ${PORT:-8000}"]

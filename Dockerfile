FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the port used by Hugging Face Spaces
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
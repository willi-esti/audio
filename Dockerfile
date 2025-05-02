# Use a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - ffmpeg: for audio decoding
# - build-essential, pkg-config: often needed for building python packages
# - libsndfile1: often a dependency for audio libraries like soundfile (used by TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    pkg-config \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
# Consider using --extra-index-url for PyTorch if needed (e.g., for CUDA)
# Example for CUDA 11.8:
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 && \
#     pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the WebSocket port
EXPOSE 8000

# Command to run the application
CMD ["python", "audio.py"]
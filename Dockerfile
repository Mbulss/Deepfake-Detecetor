# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (CPU-only PyTorch to save space)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio && \
    pip install --no-cache-dir streamlit opencv-python pillow librosa soundfile timm pandas numpy scikit-learn matplotlib seaborn tqdm

# Copy application code and model files
COPY . .

# Ensure face detection files are present (they'll be auto-downloaded if missing, but better to include them)
# The COPY . . above should include:
# - haarcascade_frontalface_default.xml
# - deploy.prototxt (if exists)
# - res10_300x300_ssd_iter_140000.caffemodel
# - best_audio_xception.pth
# - epoch_007.pt

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# Use NVIDIA PyTorch base image with CUDA 11.8 support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgtk-3-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Complete OpenCV removal and clean reinstall
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true && \
    pip cache purge && \
    find /usr/local/lib/python3.10/dist-packages -name "*cv2*" -exec rm -rf {} + || true && \
    find /usr/local/lib/python3.10/dist-packages -name "*opencv*" -exec rm -rf {} + || true

# Install Python dependencies (OpenCV will be installed from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose common ports
EXPOSE 8888 6006

# Default command
CMD ["bash"]
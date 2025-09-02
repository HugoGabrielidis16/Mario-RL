FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Pre-configure timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install minimal system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    ffmpeg \
    # Build tools needed for nes-py
    build-essential \
    g++ \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in one layer with optimizations
RUN pip install -r requirements.txt

# Expose common ports
EXPOSE 8888 6006

# Default command
CMD ["bash"]
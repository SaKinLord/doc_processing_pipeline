# ==============================================================================
# BASE IMAGE
# We're using NVIDIA's official CUDA image. This provides an Ubuntu environment
# compatible with GPU drivers and CUDA libraries.
# We selected this image because our PyTorch version is compatible with CUDA 12.1.
# ==============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ==============================================================================
# ENVIRONMENT VARIABLES
# Prevents Python from creating .pyc files and sets UTF-8 as the
# default character set.
# ==============================================================================
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# ==============================================================================
# INSTALLING SYSTEM DEPENDENCIES
# Installing essential system tools such as Tesseract, Poppler (for PDF processing),
# and image libraries required by OpenCV.
# ==============================================================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tur \
    poppler-utils \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# SETTING WORKING DIRECTORY
# Setting the default working directory inside the container to /app.
# ==============================================================================
WORKDIR /app

# ==============================================================================
# INSTALLING PYTHON LIBRARIES
# First, we copy only the requirements.txt file. This way, when we make changes
# to the code, we don't have to reinstall libraries repeatedly thanks to
# Docker's layer caching mechanism.
# ==============================================================================
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ==============================================================================
# COPYING PROJECT FILES
# Copying all necessary project code into the /app directory inside the container.
# ==============================================================================
COPY . .

# ==============================================================================
# RUN COMMAND (ENTRYPOINT/CMD)
# Specifying the default command to run when the container starts.
# This command will launch our main pipeline.
# ==============================================================================
CMD ["python3", "main.py"]
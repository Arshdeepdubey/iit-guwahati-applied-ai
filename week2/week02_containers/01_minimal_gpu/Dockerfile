# ------------------------------------------------------------------
# Minimal GPU-enabled PyTorch container.
# Build:  docker build -t demo-torch:v1 .
# Run:    docker run --rm --gpus all demo-torch:v1
# ------------------------------------------------------------------

# Start from an NVIDIA CUDA base image. The ":runtime" flavour
# includes the CUDA runtime libraries but not the compiler/dev headers
# (which we don't need for running PyTorch wheels).
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# Standard hygiene: non-interactive apt, no bytecode writing,
# unbuffered stdout for clean log capture.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python + minimal system deps. Cleaning the apt lists in the same
# layer prevents the cache from bloating the image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch built against CUDA 12.1, matching the base image.
RUN pip install --break-system-packages --no-cache-dir torch
# Copy the smoke-test script.
COPY smoke_test.py /smoke_test.py

# Default command: verify CUDA is visible and show the device.
CMD ["python3", "/smoke_test.py"]

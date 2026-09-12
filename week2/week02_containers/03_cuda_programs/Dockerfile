# devel image = nvcc included (required for compilation)
# This is the 'toolkit level' in the 3-level probe (slide 7)
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install torch for smoke_test.py (CPU build fine for CI; GPU used if hardware present)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.4.0 \
    transformers==4.44.0

WORKDIR /src

# Copy source files
COPY . .

# Compile both CUDA programs at build time
RUN make all

CMD ["bash", "-c", "echo '=== vector_add ===' && ./vector_add && echo '' && echo '=== bandwidth_test ===' && ./bandwidth_test"]

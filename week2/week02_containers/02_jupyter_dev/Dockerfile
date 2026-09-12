# ------------------------------------------------------------------
# Jupyter development container with GPU access and pinned deps.
#
# Key techniques demonstrated:
#   - Multi-stage build (builder -> runtime) to keep final image small
#   - requirements.txt is compiled from requirements.in via pip-compile
#     for a fully-pinned, reproducible dependency set
#   - Layer ordering puts slowly-changing content on top so rebuilds
#     after code edits are fast
# ------------------------------------------------------------------

# -------- Stage 1: builder - has compilers if we need to build any
# native extensions; the artifacts are copied into the runtime stage.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Python — CUDA devel image has compilers but not Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment that we will copy wholesale
# into the runtime image. This keeps the final image free of build
# caches and apt metadata.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install dependencies from the fully-pinned requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
                --extra-index-url https://pypi.org/simple \
                -r /tmp/requirements.txt

# -------- Stage 2: runtime - smaller base, no compilers.
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Install Python in runtime stage — runtime image also lacks Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy only the installed virtual environment from the builder.
# No pip, no compilers, no build cache in the final image.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
EXPOSE 8888

# Run Jupyter Lab without a token for local dev convenience.
# DO NOT deploy this command line to production; it exposes an open
# Jupyter server.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password="]
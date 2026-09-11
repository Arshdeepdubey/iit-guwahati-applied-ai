#!/usr/bin/env bash
# Build and run the minimal GPU container smoke test.
#
# Prereqs on the host:
#   - NVIDIA driver installed (check with: nvidia-smi)
#   - nvidia-container-toolkit installed and Docker restarted
#     (check with: docker info | grep -i runtime)
#
# Usage:  ./run.sh
set -euo pipefail

IMAGE="demo-torch:v1"

echo "==> Building ${IMAGE}"
docker build -t "${IMAGE}" .

echo "==> Running smoke test with --gpus all"
docker run --rm --gpus all "${IMAGE}"

echo "==> If the above printed 'OK' you have a working GPU container."

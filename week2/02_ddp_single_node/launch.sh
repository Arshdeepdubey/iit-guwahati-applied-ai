#!/usr/bin/env bash
# Launch single-node DDP training. Picks up however many GPUs are
# visible via CUDA_VISIBLE_DEVICES.
#
# Examples:
#   ./launch.sh                         # use all GPUs on this node
#   CUDA_VISIBLE_DEVICES=0,1 ./launch.sh   # restrict to GPU 0 and 1
set -euo pipefail

NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}
echo "Launching DDP with nproc_per_node=${NPROC}"

# --standalone runs an in-process c10d rendezvous; fine for single-node.
torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    train_ddp.py "$@"

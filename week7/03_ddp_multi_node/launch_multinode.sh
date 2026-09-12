#!/usr/bin/env bash
# Launch multi-node DDP. Run this script on EVERY node; set NODE_RANK
# differently on each: NODE_RANK=0 on the master, 1 on the next, etc.
#
# Typical 2-node, 8-GPU-per-node setup:
#   on node0:  NODE_RANK=0 MASTER_ADDR=node0.example.com ./launch_multinode.sh
#   on node1:  NODE_RANK=1 MASTER_ADDR=node0.example.com ./launch_multinode.sh
#
# Environment knobs:
#   NNODES          total number of nodes (default 2)
#   NPROC_PER_NODE  GPUs per node (default = local GPU count)
#   MASTER_ADDR     resolvable hostname of node0 (required)
#   MASTER_PORT     any free TCP port (default 29500)
#   NODE_RANK       0-based node index (required)
#   JOB_ID          a unique identifier shared across nodes for rendezvous
set -euo pipefail

: "${MASTER_ADDR:?MASTER_ADDR must be set (resolvable hostname of node0)}"
: "${NODE_RANK:?NODE_RANK must be set (0 on master, 1+ on workers)}"

NNODES=${NNODES:-2}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
JOB_ID=${JOB_ID:-ddp-job-001}

echo "Launching rank ${NODE_RANK} of ${NNODES} with ${NPROC_PER_NODE} GPUs"
echo "Rendezvous @ ${MASTER_ADDR}:${MASTER_PORT} job_id=${JOB_ID}"

# Useful NCCL tuning env vars - uncomment per environment:
# export NCCL_DEBUG=INFO                       # verbose logs
# export NCCL_SOCKET_IFNAME=eth0               # pick the right NIC
# export NCCL_IB_HCA=mlx5_0,mlx5_1             # InfiniBand HCAs
# export NCCL_P2P_DISABLE=0                    # allow P2P intra-node
# export NCCL_ASYNC_ERROR_HANDLING=1           # crash on timeout

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_id="${JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=3 \
    ../02_ddp_single_node/train_ddp.py "$@"

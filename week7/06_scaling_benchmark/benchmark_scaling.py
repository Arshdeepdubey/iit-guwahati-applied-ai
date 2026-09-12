"""
Micro-benchmark for DDP scaling efficiency.

We measure samples/second at a given world_size by doing a fixed
number of training steps on a synthetic dataset (so data loading is
never the bottleneck). This isolates the compute + communication
cost of DDP itself.

Launch at different world sizes and compare:
    torchrun --standalone --nproc_per_node=1 benchmark_scaling.py
    torchrun --standalone --nproc_per_node=2 benchmark_scaling.py
    torchrun --standalone --nproc_per_node=4 benchmark_scaling.py
    torchrun --standalone --nproc_per_node=8 benchmark_scaling.py

Reporting (rank 0 only):
    world=1  samples/s = X
    world=2  samples/s = Y   scaling_eff = Y / (2*X)
    ...

A scaling efficiency < 0.85 typically points to:
    - interconnect limited (PCIe-only, small NVLink topology)
    - comm-dominated (model too small per step)
    - stragglers
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models


WARMUP_STEPS = 10
MEASURE_STEPS = 100
PER_GPU_BATCH = 64
INPUT_SHAPE = (3, 224, 224)


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = dist.get_rank()
    world      = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ResNet-50 is a reasonable mid-size model: non-trivial compute,
    # non-trivial gradient volume. ~25M params.
    model = models.resnet50(weights=None).to(device)
    model = DDP(model, device_ids=[local_rank])

    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic batch we reuse each step - no data loading cost.
    x = torch.randn(PER_GPU_BATCH, *INPUT_SHAPE, device=device)
    y = torch.randint(0, 1000, (PER_GPU_BATCH,), device=device)

    # Warm up: CUDA lazy init, NCCL channel setup, cuDNN selection.
    for _ in range(WARMUP_STEPS):
        optim.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(MEASURE_STEPS):
        optim.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # samples processed by this rank
    local_samples = PER_GPU_BATCH * MEASURE_STEPS
    # global throughput = world * local_samples / elapsed
    global_sps = torch.tensor([world * local_samples / elapsed], device=device)
    dist.all_reduce(global_sps, op=dist.ReduceOp.AVG)

    if rank == 0:
        print(f"world={world:>2}  per-gpu-batch={PER_GPU_BATCH}  "
              f"samples/s={global_sps.item():,.1f}  "
              f"step-time={elapsed/MEASURE_STEPS*1000:.1f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

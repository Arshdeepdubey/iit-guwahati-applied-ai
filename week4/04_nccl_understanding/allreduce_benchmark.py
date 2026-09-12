"""
Benchmark NCCL all-reduce bandwidth across the participating GPUs.

Purpose: build intuition for ring-all-reduce scaling by directly
measuring the effective bus bandwidth at different message sizes.

Launch:
    torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py
    torchrun --standalone --nproc_per_node=8 allreduce_benchmark.py

What you should see:
    - Per-message bandwidth rises with message size until it saturates
      near the hardware ceiling of the interconnect (NVLink / PCIe / IB).
    - Adding GPUs keeps per-GPU bandwidth ~constant, which is exactly
      the point of ring all-reduce.

Related theory:
    Each rank sends/receives ~ 2*(N-1)/N * T bytes, where T is the
    tensor size and N the world size. For N>=4 this is ~2T per rank
    regardless of N - why doubling the cluster doesn't double comm time.
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def bench(size_bytes: int, iters: int = 50, warmup: int = 5) -> float:
    """Return measured all-reduce bandwidth in GB/s per rank."""
    assert size_bytes % 4 == 0
    n_elems = size_bytes // 4
    x = torch.ones(n_elems, dtype=torch.float32, device="cuda")

    # Warm up to exclude first-call allocation / channel setup.
    for _ in range(warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # "Algorithm bandwidth" as reported by NCCL tests: bytes moved /
    # wall-clock. For all-reduce each rank does ~2*(N-1)/N * T bytes.
    world = dist.get_world_size()
    factor = 2 * (world - 1) / world
    effective = factor * size_bytes * iters
    return effective / elapsed / 1e9


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank  = dist.get_rank()
    world = dist.get_world_size()

    if rank == 0:
        print(f"NCCL all-reduce benchmark, world_size={world}")
        print(f"{'size':>12}  {'bandwidth':>14}")

    # Sweep message sizes from 4 KiB up to 256 MiB.
    for log2 in range(12, 29):   # 4 KiB .. 256 MiB
        size = 1 << log2
        gbps = bench(size)
        if rank == 0:
            print(f"{size:>12}  {gbps:>10.1f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

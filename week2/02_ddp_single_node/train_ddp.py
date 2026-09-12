"""
Single-node multi-GPU training with PyTorch DistributedDataParallel.

Launch (2 GPUs on one box):
    torchrun --standalone --nproc_per_node=2 train_ddp.py \
        --epochs 3 --per-gpu-batch-size 128

What torchrun does for us:
    - Spawns one process per GPU on this node.
    - Sets env vars: RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    - Coordinates the c10d rendezvous.

Anatomy of a DDP script (annotated below):
    1. init_process_group(backend="nccl")      <- collective channel opens
    2. set the CUDA device to LOCAL_RANK        <- each rank owns one GPU
    3. wrap model in DDP                        <- gradient all-reduce hook
    4. use DistributedSampler                   <- each rank sees its shard
    5. train loop as normal
    6. only rank 0 saves / logs

Common gotchas (called out inline):
    * forgetting sampler.set_epoch(epoch) => identical shuffle each epoch
    * saving model.state_dict() not model.module.state_dict()
    * BatchNorm with small per-GPU batch => SyncBatchNorm
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models


# --------------------------------------------------------------------
# Distributed setup / teardown
# --------------------------------------------------------------------

def ddp_setup() -> tuple[int, int, int]:
    """Initialise the process group and bind this process to its GPU.

    Returns (rank, local_rank, world_size).
    """
    # NCCL is the right backend for GPU-to-GPU collectives. Use "gloo"
    # only for CPU debugging.
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world      = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def ddp_cleanup() -> None:
    dist.destroy_process_group()


def is_rank_zero() -> bool:
    return dist.get_rank() == 0


def log0(msg: str) -> None:
    """Print only from rank 0 to avoid duplicated output."""
    if is_rank_zero():
        print(msg, flush=True)


# --------------------------------------------------------------------
# Model and data
# --------------------------------------------------------------------

def build_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # If you use BatchNorm in a model where per-GPU batch is small,
    # convert to SyncBatchNorm so statistics are averaged across ranks.
    # ResNet18 already uses BN; it's safe to wrap here.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def build_loaders(data_root: str, batch_size: int, world: int, rank: int,
                  num_workers: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Download only from rank 0 to avoid a race among processes all
    # writing to the same extraction directory.
    if is_rank_zero():
        datasets.CIFAR10(data_root, train=True, download=True)
        datasets.CIFAR10(data_root, train=False, download=True)
    dist.barrier()   # other ranks wait until rank 0 is done

    train_ds = datasets.CIFAR10(data_root, train=True, download=False, transform=tfm)
    val_ds   = datasets.CIFAR10(data_root, train=False, download=False, transform=tfm)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank,
                                       shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds, num_replicas=world, rank=rank,
                                       shuffle=False, drop_last=False)

    train_ld = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld, train_sampler


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    """Compute global validation loss and accuracy via all_reduce."""
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    correct  = torch.zeros(1, device=device)
    total    = torch.zeros(1, device=device)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y)
        correct  += (logits.argmax(dim=1) == y).sum()
        total    += y.size(0)

    # Sum across all ranks -> every rank sees the global total.
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct,  op=dist.ReduceOp.SUM)
    dist.all_reduce(total,    op=dist.ReduceOp.SUM)

    model.train()
    return (loss_sum / total).item(), (correct / total).item()


# --------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-gpu-batch-size", type=int, default=128)
    p.add_argument("--base-lr", type=float, default=1e-3,
                   help="LR at reference batch size; linearly scaled by world size.")
    p.add_argument("--reference-batch", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="./ckpt_ddp.pt")
    args = p.parse_args()

    # All ranks seed identically for identical weight init; DDP will
    # then broadcast weights from rank 0 at wrap time anyway.
    torch.manual_seed(args.seed)

    rank, local_rank, world = ddp_setup()
    device = torch.device(f"cuda:{local_rank}")
    log0(f"world_size={world}  per-gpu-batch={args.per_gpu_batch_size}  "
         f"global-batch={world * args.per_gpu_batch_size}")

    # Linear LR scaling rule (Goyal et al., 2017): LR scales with global batch.
    global_batch = args.per_gpu_batch_size * world
    scaled_lr = args.base_lr * (global_batch / args.reference_batch)
    log0(f"scaled LR = {scaled_lr:.5f} (base {args.base_lr} x "
         f"{global_batch}/{args.reference_batch})")

    model = build_model().to(device)
    model = DDP(model, device_ids=[local_rank])

    train_ld, val_ld, train_sampler = build_loaders(
        args.data_root, args.per_gpu_batch_size, world, rank, args.num_workers)

    optim = torch.optim.AdamW(model.parameters(), lr=scaled_lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # CRITICAL: advance the sampler's epoch so shuffling differs.
        train_sampler.set_epoch(epoch)

        t0 = time.time()
        for step, (x, y) in enumerate(train_ld):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            # forward + backward; gradient all-reduce happens
            # inside backward() via DDP's autograd hook.
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            if step % 50 == 0:
                log0(f"ep={epoch} step={step} loss={loss.item():.4f}")

        val_loss, val_acc = evaluate(model, val_ld, device)
        log0(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
             f"time={time.time()-t0:.1f}s")

    # Only rank 0 writes the checkpoint. Note the .module unwrap.
    if is_rank_zero():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.module.state_dict(), args.out)
        print(f"wrote {args.out}", flush=True)

    ddp_cleanup()


if __name__ == "__main__":
    main()

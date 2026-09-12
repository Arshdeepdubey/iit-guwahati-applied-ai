"""
DDP training with robust checkpointing.

Two strategies are demonstrated:
    - Rank-0 single-file checkpoints (torch.save on model.module). Simple;
      works for DDP where every replica is identical.
    - Distributed Checkpoint (DCP). Required for FSDP/ZeRO where state
      is sharded; works fine for DDP too and writes faster via parallel I/O.

Resumption is deterministic in the sense that given the same data order
and the same per-step RNG state, training resumes as if uninterrupted.
We save and restore:
    - model weights
    - optimizer state
    - LR scheduler state
    - current epoch and global step
    - Python / torch RNG state and CUDA RNG state

Launch:
    torchrun --standalone --nproc_per_node=2 train_with_checkpoint.py \
        --ckpt-dir ./ckpts --checkpoint-every 100

Resume:
    torchrun --standalone --nproc_per_node=2 train_with_checkpoint.py \
        --ckpt-dir ./ckpts --resume-from ./ckpts/step-100
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models


# --------------------------- helpers ----------------------------

def is_rank_zero() -> bool:
    return dist.get_rank() == 0


def log0(msg: str) -> None:
    if is_rank_zero():
        print(msg, flush=True)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def rng_state() -> dict:
    return {
        "torch":  torch.get_rng_state(),
        "cuda":   torch.cuda.get_rng_state_all(),
        "numpy":  np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


# --------------------------- checkpointing ----------------------------

def save_checkpoint_dcp(path: Path, model: DDP, optim, scheduler,
                        meta: dict) -> None:
    """Distributed checkpoint: each rank writes its shard; fast & scalable."""
    path.mkdir(parents=True, exist_ok=True)
    state = {
        "model":     model.module.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "meta":      meta,
    }
    dcp.save(state_dict=state, checkpoint_id=str(path))
    # RNG is small and rank-local; save via torch.save per rank.
    torch.save(rng_state(), path / f"rng_rank{dist.get_rank()}.pt")


def load_checkpoint_dcp(path: Path, model: DDP, optim, scheduler) -> dict:
    """Restore the DCP checkpoint and return the meta dict."""
    state = {
        "model":     model.module.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "meta":      {},
    }
    dcp.load(state_dict=state, checkpoint_id=str(path))
    model.module.load_state_dict(state["model"])
    optim.load_state_dict(state["optimizer"])
    if scheduler and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    rng = torch.load(path / f"rng_rank{dist.get_rank()}.pt")
    set_rng_state(rng)
    return state["meta"]


# ------------------------------ main ----------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-gpu-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt-dir", default="./ckpts")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_all(args.seed)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank])

    # data
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    if is_rank_zero():
        datasets.CIFAR10(args.data_root, train=True, download=True)
    dist.barrier()
    train_ds = datasets.CIFAR10(args.data_root, train=True, transform=tfm)
    sampler  = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    loader   = DataLoader(train_ds, batch_size=args.per_gpu_batch_size,
                          sampler=sampler, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        meta = load_checkpoint_dcp(Path(args.resume_from), model, optim, scheduler)
        start_epoch = meta.get("epoch", 0)
        global_step = meta.get("step", 0)
        log0(f"resumed from {args.resume_from}: epoch={start_epoch} step={global_step}")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        t0 = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            scheduler.step()
            global_step += 1

            if global_step % args.checkpoint_every == 0:
                ckpt = Path(args.ckpt_dir) / f"step-{global_step}"
                log0(f"checkpoint: {ckpt}")
                save_checkpoint_dcp(ckpt, model, optim, scheduler,
                                    meta={"epoch": epoch, "step": global_step})

        log0(f"epoch {epoch}: loss={loss.item():.4f} time={time.time()-t0:.1f}s")

    # final
    final = Path(args.ckpt_dir) / "final"
    save_checkpoint_dcp(final, model, optim, scheduler,
                        meta={"epoch": args.epochs, "step": global_step})
    log0(f"final checkpoint: {final}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
Demonstrates the linear LR scaling rule (Goyal et al., 2017).

Trains a simple CNN on MNIST at three global batch sizes:
    - 64   (small, baseline)
    - 512  (medium, scales OK with linear rule)
    - 4096 (large, fragile; needs warmup to converge)

Each configuration runs in three modes:
    (a) base-lr with no adjustment (control)
    (b) linearly-scaled LR, no warmup
    (c) linearly-scaled LR, linear warmup over first 10% of steps

The loss / val-accuracy curves illustrate:
    - Without LR scaling, large-batch training loses progress per epoch.
    - With scaling + warmup, large-batch approximates small-batch quality
      (up to the large-batch generalisation wall, well beyond 4096 here).

Run (single GPU is fine):
    python batch_size_study.py --out results.json
Plot:
    python batch_size_study.py --plot results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def lr_at_step(step: int, base_lr: float, total_steps: int,
               warmup_frac: float) -> float:
    """Linear warmup then cosine decay."""
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # cosine decay from peak to 0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_one(global_batch: int, ref_batch: int, base_lr: float,
              use_scaling: bool, warmup_frac: float,
              epochs: int, device: str) -> dict:
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    val   = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_ld = DataLoader(train, batch_size=global_batch, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val,   batch_size=512, shuffle=False)

    peak_lr = base_lr * (global_batch / ref_batch) if use_scaling else base_lr
    model = SmallCNN().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=peak_lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    total_steps = epochs * len(train_ld)

    step = 0
    losses, accs = [], []
    t0 = time.time()
    for ep in range(epochs):
        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lr_now = lr_at_step(step, peak_lr, total_steps, warmup_frac)
            for g in optim.param_groups:
                g["lr"] = lr_now
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            step += 1

        # Eval each epoch.
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        model.train()
        acc = correct / total
        losses.append(loss.item())
        accs.append(acc)

    return {
        "global_batch": global_batch,
        "use_scaling": use_scaling,
        "warmup_frac": warmup_frac,
        "peak_lr": peak_lr,
        "final_loss": losses[-1],
        "final_acc": accs[-1],
        "acc_by_epoch": accs,
        "wall_seconds": time.time() - t0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results.json")
    p.add_argument("--plot", help="path to results.json to plot instead of training")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; pip install matplotlib")
            return
        data = json.loads(Path(args.plot).read_text())
        for cfg in data:
            label = (f"bs={cfg['global_batch']} "
                     f"{'scale' if cfg['use_scaling'] else 'noscale'} "
                     f"warm={cfg['warmup_frac']}")
            plt.plot(cfg["acc_by_epoch"], label=label, marker="o")
        plt.xlabel("epoch"); plt.ylabel("val acc"); plt.legend(fontsize=7)
        plt.title("Effect of batch size, LR scaling, and warmup")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("scaling_study.png", dpi=120)
        print("wrote scaling_study.png")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    configs = [
        # (batch, use_scaling, warmup_frac)
        (64,   False, 0.0),
        (512,  False, 0.0),
        (512,  True,  0.0),
        (4096, False, 0.0),
        (4096, True,  0.0),
        (4096, True,  0.10),
    ]
    for (bs, scale, warm) in configs:
        print(f"\n=== bs={bs} scale={scale} warmup={warm} ===")
        r = train_one(global_batch=bs, ref_batch=64, base_lr=0.05,
                      use_scaling=scale, warmup_frac=warm,
                      epochs=args.epochs, device=device)
        print(r)
        results.append(r)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

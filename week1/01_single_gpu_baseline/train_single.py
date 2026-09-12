"""
Single-GPU baseline. Converted to DDP in ../02_ddp_single_node/train_ddp.py.

Trains a small ResNet on CIFAR-10 for a handful of epochs. Purpose is
pedagogical: the DDP version should produce an equivalent loss curve
when the effective global batch size and LR are matched.

Run:
    python train_single.py --data-root /tmp/cifar --epochs 3 --batch-size 128
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loaders(data_root: str, batch_size: int, num_workers: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
    val_ds   = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        correct  += (logits.argmax(dim=1) == y).sum().item()
        total    += y.size(0)
    model.train()
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="./ckpt_single.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    model = build_model().to(device)
    train_ld, val_ld = build_loaders(args.data_root, args.batch_size, args.num_workers)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_ld):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"ep={epoch} step={step} loss={loss.item():.4f}")

        val_loss, val_acc = evaluate(model, val_ld, device)
        dt = time.time() - t0
        print(f"epoch {epoch}: train_loss={running/len(train_ld):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} time={dt:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

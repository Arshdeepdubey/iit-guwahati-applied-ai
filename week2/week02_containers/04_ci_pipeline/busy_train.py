"""
busy_train.py — Week 2, Session 4, Lab 2
Runs a small CNN training loop that actually saturates the GPU.
Designed for live nvidia-smi dmon / pmon observation.

Usage (inside the gpu-hello container):
    python3 busy_train.py            # default: well-fed DataLoader (workers=4)
    python3 busy_train.py --starve   # workers=0 → input-starved, sm% collapses
"""

import argparse, time, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--starve", action="store_true",
                    help="Set DataLoader workers=0 to starve the GPU of data")
parser.add_argument("--epochs", type=int, default=999,
                    help="Number of epochs (default: run until Ctrl-C)")
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--size", type=int, default=64, help="Image spatial size")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[busy_train] device={DEVICE}  starve={args.starve}  batch={args.batch}")
if DEVICE == "cuda":
    print(f"[busy_train] GPU: {torch.cuda.get_device_name(0)}")

# ── Synthetic dataset (random images + labels) ──────────────────────────────
N = 8192
X = torch.randn(N, 3, args.size, args.size)
y = torch.randint(0, 10, (N,))
dataset = TensorDataset(X, y)
workers = 0 if args.starve else min(2, args.workers)
loader  = DataLoader(dataset, batch_size=args.batch,
                     shuffle=True, num_workers=workers, pin_memory=(DEVICE=="cuda"))

# ── Small CNN ────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(8),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 512), nn.ReLU(),
    nn.Linear(512, 10),
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

# ── Training loop ────────────────────────────────────────────────────────────
print("[busy_train] Training… press Ctrl-C to stop\n")
try:
    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | loss={total_loss/n_batches:.4f} | {elapsed:.1f}s/epoch")
except KeyboardInterrupt:
    print("\n[busy_train] stopped.")

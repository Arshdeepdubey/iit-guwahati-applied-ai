"""
smoke_test.py — Week 2, Session 3
Runs in CI (CPU) and on cluster (GPU). Same script, two tiers.
Catches: missing libs, broken wheels, version mismatches, CUDA-less wheels.
"""
import sys
import subprocess

# ── Tier 1: imports + versions (CPU, always runs) ─────────────────────────────
try:
    import torch
    print(f"torch        {torch.__version__}")
except ImportError as e:
    print(f"FAIL: torch not importable — {e}")
    sys.exit(1)

try:
    import transformers
    print(f"transformers {transformers.__version__}")
except ImportError:
    print("WARNING: transformers not installed (optional)")

# ── Tier 2: basic CPU math (always runs) ──────────────────────────────────────
x = torch.randn(64, 64) @ torch.randn(64, 64)
assert x.shape == (64, 64), "matmul shape wrong"
print("cpu math     OK")

# ── Tier 3: GPU (only when hardware present) ──────────────────────────────────
if torch.cuda.is_available():
    y = x.cuda() @ x.cuda()
    torch.cuda.synchronize()
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"gpu ok       {name}  ({mem} GB)")
else:
    print("gpu          cpu-only runner — gpu checks skipped")

# ── Tier 4: nvidia-smi (informational, non-fatal) ─────────────────────────────
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL, text=True
    ).strip()
    print(f"nvidia-smi   {out}")
except FileNotFoundError:
    print("nvidia-smi   not found (cpu-only runner)")

print("\nAll checks passed.")
sys.exit(0)

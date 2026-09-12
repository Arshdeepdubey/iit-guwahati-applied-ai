# Week 2 · Session 4 — Demo Files

## Files

| File | Purpose |
|---|---|
| `busy_train.py` | Small CNN training loop — run inside the `gpu-hello` container |
| `demo_script.md` | Full on-camera script: what to type, say, and expect for both labs |

## Prerequisites

The `gpu-hello` Docker image from Session 2 must be built. If not:
```powershell
# from week02_containers/02_jupyter_dev/
docker build -t gpu-hello .
```

## Lab 1 — MIG inspect (no extra files needed)

All commands run directly in PowerShell against the host nvidia-smi.
See `demo_script.md` → Lab 1.

## Lab 2 — busy_train

```powershell
# Copy busy_train.py to your demos folder, then:

# Well-fed (sm% ~97):
docker run --rm --gpus all -v C:\Users\chick\Desktop\demos:/demos gpu-hello python3 /demos/busy_train.py

# Starved (sm% ~8) — the teaching moment:
docker run --rm --gpus all -v C:\Users\chick\Desktop\demos:/demos gpu-hello python3 /demos/busy_train.py --starve
```

Watch `nvidia-smi dmon -s um -d 1` in a second terminal while training runs.

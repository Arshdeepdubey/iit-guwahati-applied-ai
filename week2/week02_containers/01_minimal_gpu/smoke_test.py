"""
GPU smoke test. Prints enough detail to diagnose most GPU-passthrough
failures at a glance.

Exit codes:
  0 - everything OK
  1 - CUDA not available in container
  2 - CUDA available but no devices visible
"""
from __future__ import annotations

import sys
import torch


def main() -> int:
    print("=" * 60)
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nFAILURE: torch.cuda.is_available() returned False.")
        print("Common causes:")
        print("  - forgot '--gpus all' on docker run")
        print("  - nvidia-container-toolkit not installed on host")
        print("  - host driver older than the CUDA version in the image")
        return 1

    device_count = torch.cuda.device_count()
    print(f"Device count    : {device_count}")
    if device_count == 0:
        print("\nFAILURE: CUDA reports available but 0 devices.")
        return 2

    print(f"CUDA runtime    : {torch.version.cuda}")
    print(f"cuDNN version   : {torch.backends.cudnn.version()}")
    print("-" * 60)

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"Device {i}: {props.name}")
        print(f"  Compute capability : {props.major}.{props.minor}")
        print(f"  Total memory       : {total / 1024**3:.1f} GiB")
        print(f"  Free memory        : {free  / 1024**3:.1f} GiB")
        print(f"  Multiprocessors    : {props.multi_processor_count}")

    # Quick arithmetic check: put a tensor on the GPU, do something,
    # bring it back. This catches errors like "driver OK but cuBLAS
    # can't initialise" that a mere is_available() misses.
    print("-" * 60)
    print("Running a tiny compute check on device 0...")
    a = torch.randn(1024, 1024, device="cuda:0")
    b = torch.randn(1024, 1024, device="cuda:0")
    c = (a @ b).sum().item()
    print(f"matmul+sum result: {c:.4f} (any finite number is success)")
    print("=" * 60)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

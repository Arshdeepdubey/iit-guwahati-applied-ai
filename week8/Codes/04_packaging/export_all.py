"""
Package a trained PyTorch model into three common deliverable formats:

    - safetensors: framework-native weights; safest for sharing
    - TorchScript:  self-contained Python-free module
    - ONNX:         graph-level interchange for non-PyTorch runtimes

Choice of format affects where and how you can serve:
    - vLLM / TGI / HF transformers    -> safetensors
    - ONNX Runtime, TensorRT          -> ONNX
    - Mobile / standalone C++         -> TorchScript or ONNX

(Note: the deep inference-acceleration view - TensorRT engines, INT8/FP16
quantisation, KV cache, batching strategies - is covered in Week 9.)

Usage:
    python export_all.py --ckpt ckpt_ddp.pt --out-dir ./exported
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_matching_model(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="trained state_dict file")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_matching_model(args.num_classes)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    example = torch.randn(1, 3, 224, 224)

    # 1. safetensors - just the weights, safe (no pickle), quick to load.
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), str(out / "model.safetensors"))
        print(f"  wrote {out / 'model.safetensors'}")
    except ImportError:
        print("  skipping safetensors (pip install safetensors)")

    # 2. TorchScript - traced module; shippable without source.
    scripted = torch.jit.trace(model, example)
    scripted.save(str(out / "model.ts"))
    print(f"  wrote {out / 'model.ts'}")

    # 3. ONNX - broadest interoperability.
    torch.onnx.export(
        model, example,
        str(out / "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"  wrote {out / 'model.onnx'}")

    # Bundle a small metadata file - every artifact ought to ship with
    # provenance so that 6 months from now you know what made it.
    meta = {
        "framework": "pytorch",
        "model_arch": "resnet18",
        "num_classes": args.num_classes,
        "source_checkpoint": str(Path(args.ckpt).resolve()),
        "example_input_shape": list(example.shape),
        "onnx_opset": args.opset,
    }
    import json
    (out / "model_card.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out / 'model_card.json'}")


if __name__ == "__main__":
    main()

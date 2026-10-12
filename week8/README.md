# Week 8: Advanced Topics - Learning Rate Scaling, Checkpointing & Packaging

## 🎯 Learning Objectives

By the end of Week 8, you will understand and be able to:
- ✅ Scale learning rates correctly for very large batches (>1000)
- ✅ Implement and use gradient checkpointing to reduce memory
- ✅ Save and load model checkpoints for training resumption
- ✅ Package models for production deployment
- ✅ Version models and manage artifacts
- ✅ Deploy trained models with FastAPI (inference service)
- ✅ Monitor model performance in production

---

## 📚 Key Concepts

### Learning Rate Scaling at Scale
- **Linear Scaling Rule**: LR = base_lr × (batch_size / 256)
- **Warmup Schedules**: Gradual LR increase at training start
- **Learning Rate Decay**: Cosine annealing, step decay
- **Limitations**: Linear scaling breaks at very large batch sizes

### Gradient Checkpointing
- **Trade-off**: Memory vs computation time
- **When to use**: Large models (ResNets, Transformers)
- **Implementation**: Wrap model blocks with checkpoint()
- **Memory savings**: 50-80% reduction with ~30% slowdown

### Checkpointing Training
- **Resumable training**: Save model, optimizer state, epoch number
- **Handling failures**: Restore and continue from last checkpoint
- **Distributed checkpointing**: Coordination across all ranks

### Model Packaging
- **Serialization**: Pickle, ONNX, TorchScript formats
- **Model versioning**: Semantic versioning, Git tags
- **Artifact storage**: Local filesystem, cloud storage (S3, GCS)
- **Reproducibility**: Recording hyperparameters, random seeds

### Production Deployment
- **Inference optimization**: Quantization, distillation, pruning
- **Serving framework**: FastAPI, TorchServe, Triton
- **Monitoring**: Model performance, latency, error rates
- **A/B testing**: Comparing model versions

---

## 📁 Folder Structure

```
week8/
├── README.md                          # This file
└── Codes/
    ├── 01_lr_scaling_study/
    │   └── batch_size_study.py       # LR scaling analysis
    ├── 02_checkpointing/
    │   └── train_with_checkpoint.py  # Checkpoint save/load
    ├── 03_model_optimization/        # Coming: quantization, pruning
    └── 04_packaging/
        ├── export_all.py             # Export models in multiple formats
        └── register_in_mlflow.py     # MLflow model registry
```

---

## 🔧 Setup & Requirements

### Install Additional Packages
```bash
pip install onnx torch-onnx  # Model export
pip install mlflow  # Experiment tracking and model registry
pip install pydantic  # API schemas
```

---

## 🚀 How to Run

### 1. Learning Rate Scaling Study

```bash
cd Codes/01_lr_scaling_study

# Run batch size study (measure convergence at different scales)
python batch_size_study.py \
    --batch-sizes 32 64 128 256 512 1024 \
    --epochs 30 \
    --output lr_scaling_results.csv

# Expected output:
# batch_size, base_lr, scaled_lr, final_acc, convergence_epoch
# 32,        0.1,    0.1,       82.1,      12
# 64,        0.1,    0.1,       81.9,      13
# 128,       0.1,    0.2,       81.8,      14
# 256,       0.1,    0.4,       80.5,      16
# 512,       0.1,    0.8,       78.2,      22
```

**Analysis:**
- Record where accuracy starts degrading
- Find "critical batch size" for your model
- Beyond this: need advanced LR scaling techniques (LARS, LAMB)

### 2. Training with Checkpointing

```bash
cd ../02_checkpointing

# Train with automatic checkpointing
python train_with_checkpoint.py \
    --epochs 30 \
    --checkpoint-dir ./checkpoints \
    --save-interval 5

# Output structure:
# checkpoints/
# ├── best_model.pth         # Best accuracy model
# ├── latest_checkpoint.pt   # Most recent
# └── epoch_5.pt, epoch_10.pt, ...
```

**Resume interrupted training:**
```bash
# Simulation: interrupt training (Ctrl+C at epoch 15)
python train_with_checkpoint.py --epochs 30 --resume ./checkpoints/latest_checkpoint.pt

# Training continues from epoch 15 automatically
```

**Checkpoint contents:**
```python
checkpoint = {
    'epoch': 14,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 1.23,
    'accuracy': 75.2,
    'hyperparams': {
        'learning_rate': 0.1,
        'batch_size': 128,
        'seed': 42
    }
}
```

### 3. Model Export and Packaging

```bash
cd ../04_packaging

# Export model in multiple formats
python export_all.py \
    --model-path ./checkpoints/best_model.pth \
    --output-dir ./exported_models

# Output:
# exported_models/
# ├── model.pth              # PyTorch format
# ├── model.onnx             # ONNX format (framework agnostic)
# ├── model.pt               # TorchScript (optimized)
# └── model_config.json      # Hyperparameters and metadata
```

**Model config example:**
```json
{
    "model": "ResNet18",
    "num_classes": 10,
    "input_size": 224,
    "accuracy": 82.1,
    "training_time_hours": 4.2,
    "framework": "PyTorch",
    "created_at": "2024-01-15T10:23:45Z",
    "hyperparameters": {
        "learning_rate": 0.1,
        "batch_size": 128,
        "epochs": 30,
        "optimizer": "SGD"
    }
}
```

### 4. MLflow Model Registry

```bash
# Register model in MLflow
python register_in_mlflow.py \
    --model-path ./exported_models/model.pth \
    --model-name cifar10-resnet18 \
    --version 1.0.0

# Start MLflow UI to view models
mlflow ui --port 5000
# Visit http://localhost:5000
```

### 5. Inference Service (Production)

```bash
cd ../01_inference_service

# Start FastAPI server with checkpointed model
python app.py --model-path ../week8/exported_models/model.pth

# Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ..., 1.6]}'  # 16 features for CIFAR-10
# Response: {"probs": [0.25, 0.35, 0.40], "latency_ms": 1.23}
```

---

## 📊 Understanding Results

### Learning Rate Scaling Analysis

**Expected findings:**
```
Batch Size | Optimal LR | Final Accuracy | Convergence Time
-----------|------------|----------------|------------------
32         | 0.1        | 82.2%          | 30 epochs
64         | 0.1        | 81.9%          | 30 epochs
128        | 0.2        | 81.8%          | 30 epochs (2x speedup)
256        | 0.4        | 80.5%          | 30 epochs (4x speedup)
512        | 0.8        | 78.2%          | 30 epochs (6x speedup)
1024       | 1.6        | 75.1%          | 30 epochs (low accuracy)
```

**Interpretation:**
- Linear scaling works up to batch_size=256
- Beyond 256: accuracy degrades despite LR scaling
- Solution: Use LARS optimizer for batch_size≥512

### Checkpoint Recovery

**Before checkpoint:**
```
Epoch 10: interrupted by system crash
All training progress lost
Must restart from epoch 0 (waste 33% of compute)
```

**With checkpoint:**
```
Epoch 10: crash happens
Resume from checkpoint
Training continues from epoch 10 (no wasted compute)
Total training time: 10 + 20 = 30 epochs (efficient)
```

**Memory overhead:**
- Checkpoint file size: ~100-500 MB (model dependent)
- Disk I/O time: 1-5 seconds per save
- Overhead: ~1% for saving every 5 epochs

### Model Packaging

**Format comparison:**

| Format | Size | Speed | Framework | Use Case |
|--------|------|-------|-----------|----------|
| PyTorch (.pth) | 50 MB | Fast | PyTorch | Training, research |
| ONNX (.onnx) | 50 MB | Medium | Any | Cross-framework inference |
| TorchScript (.pt) | 45 MB | Fastest | Any | Production, mobile |
| Serialized (.pickle) | 55 MB | Slow | Python only | Legacy systems |

---

## 🧪 Hands-On Experiments

### Experiment 1: Critical Batch Size Discovery
```bash
# Find the batch size where accuracy starts declining significantly
# Plot: Batch Size vs Final Accuracy

# Run batch size study and analyze inflection point
python batch_size_study.py --batch-sizes 32 64 128 256 512 1024 2048
```

### Experiment 2: Checkpoint Overhead
```bash
# Measure time cost of checkpointing
# Compare: training without checkpoints vs with checkpoints

# Without checkpoints: baseline time
python train_no_checkpoint.py --epochs 30

# With checkpoints (every 1, 5, 10 epochs)
python train_with_checkpoint.py --epochs 30 --save-interval 1
python train_with_checkpoint.py --epochs 30 --save-interval 5
python train_with_checkpoint.py --epochs 30 --save-interval 10

# Analyze: Overhead = (time_with - time_without) / time_without
```

### Experiment 3: Model Export Verification
```bash
# Verify exported models produce identical predictions
python verify_exports.py \
    --model-pth ./exported_models/model.pth \
    --model-onnx ./exported_models/model.onnx \
    --model-pt ./exported_models/model.pt \
    --num-samples 1000

# Expected: Max difference between outputs < 1e-5 (numerical precision)
```

### Experiment 4: Production Performance
```bash
# Load model variants and measure inference latency
# PyTorch → ONNX → TorchScript

# Benchmark each variant:
python benchmark_inference.py --format pytorch
python benchmark_inference.py --format onnx
python benchmark_inference.py --format torchscript

# Plot: Model Format vs Latency
# Expected: TorchScript fastest, PyTorch slowest
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory" with gradient checkpointing
**Cause**: Checkpointing trades memory for compute but still needs some memory

**Solution:**
```python
# More aggressive checkpointing
def checkpoint_sequential(module, *args, **kwargs):
    return checkpoint(module, *args, use_reentrant=False, **kwargs)

# Reduce batch size further
# Use `torch.cuda.empty_cache()` between batches
```

### Issue: Checkpoint file too large
**Cause**: Storing multiple checkpoints accumulates disk space

**Solution:**
```python
# Keep only recent N checkpoints
import os
checkpoints = sorted(glob.glob('checkpoints/*.pt'))
if len(checkpoints) > 5:
    os.remove(checkpoints[0])  # Remove oldest
```

### Issue: ONNX export fails
**Cause**: Model uses PyTorch-specific operations

**Solution:**
```python
# Use TorchScript instead of ONNX for complex models
model_scripted = torch.jit.script(model)
model_scripted.save('model.pt')
```

---

## 📚 Resources

### Learning Rate Scaling
- [LARS Optimizer Paper](https://arxiv.org/abs/1708.03888)
- [LAMB Optimizer](https://arxiv.org/abs/1904.00325)
- [Gradient Checkpointing Paper](https://arxiv.org/abs/1604.06174)

### Model Packaging
- [ONNX Format](https://onnx.ai/)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

### Production Deployment
- [TorchServe](https://pytorch.org/serve/)
- [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-center/products/triton-inference-server/)

---

## ✅ Verification Checklist

Before moving to Week 9, verify:
- [ ] You understand LR scaling laws and their limitations
- [ ] Gradient checkpointing reduces memory and you can measure the trade-off
- [ ] You can save and resume training from checkpoints
- [ ] Models can be exported to ONNX and TorchScript formats
- [ ] MLflow registry stores and tracks model versions
- [ ] Inference service serves exported models correctly
- [ ] Production performance is optimized (TorchScript fastest)

---

## 🎓 Concepts for Next Week

In **[Week 9: Foundational Language Models](../week9/README.md)**, you'll learn:
- Transformer architecture fundamentals
- Attention mechanisms and self-attention
- Why Transformers are better than RNNs for NLP
- Building blocks of modern LLMs

**Bridge**: Moving from vision (ResNet on CIFAR) → language (Transformers and LLMs)

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 7: Review & Consolidation](../week7/README.md)  
**Next Week**: [Week 9: Foundational Language Models →](../week9/README.md)

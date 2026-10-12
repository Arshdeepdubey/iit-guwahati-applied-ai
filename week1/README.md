# Week 1: Single GPU Foundations

## 🎯 Learning Objectives

By the end of Week 1, you will understand and be able to:
- ✅ Set up PyTorch and verify GPU availability
- ✅ Write optimized single-GPU training loops for deep learning models
- ✅ Use PyTorch's DataLoader for efficient data loading
- ✅ Implement model evaluation and inference
- ✅ Build and deploy a production-ready inference service with FastAPI
- ✅ Handle common GPU memory optimization techniques

---

## 📚 Key Concepts

### GPU Fundamentals
- **GPU Memory Hierarchy**: Global memory, shared memory, registers
- **Tensor Operations**: GPU-accelerated matrix operations via PyTorch
- **Batching**: Optimal batch sizes for memory and throughput
- **Pinned Memory**: Accelerating CPU-to-GPU transfers

### PyTorch Training Pipeline
- **Model Definition**: Creating neural networks with `nn.Module`
- **DataLoading**: Using `DataLoader` and `DistributedSampler` preparation
- **Training Loop**: Forward pass, backward pass, optimizer step
- **Evaluation**: Measuring accuracy and loss on validation data

### Inference Service
- **FastAPI**: Building HTTP endpoints for model serving
- **Lifespan Management**: Loading models at startup efficiently
- **Metrics**: Tracking request latency and error rates
- **Health Checks**: Readiness and liveness probes for orchestration

---

## 📁 Folder Structure

```
week1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── 01_single_gpu_baseline/            # Training script tutorial
│   ├── train_single.py               # Single-GPU ResNet on CIFAR-10
│   └── torch_init.py                 # GPU setup verification
└── 01_inference_service/              # FastAPI inference service
    └── app.py                        # Production-ready inference endpoint
```

### What's in Each Folder?

**`01_single_gpu_baseline/`**
- `train_single.py`: Trains a ResNet18 on CIFAR-10 dataset
  - **Purpose**: Pedagogical baseline for comparing with distributed training
  - **GPU Use**: Single GPU training with batch processing
  - **Output**: Training/validation loss curves over epochs
  
- `torch_init.py`: Utility to verify PyTorch and GPU setup
  - **Purpose**: Diagnostic tool to check environment
  - **Usage**: Run this first to ensure CUDA is available

**`01_inference_service/`**
- `app.py`: Production-ready FastAPI inference service
  - **Purpose**: Demonstrate model serving for real-world deployment
  - **Endpoints**: Health check, prediction, metrics
  - **GPU Use**: Model inference on GPU with batch predictions
  - **Ready for**: Kubernetes deployment, load balancing

---

## 🔧 Setup & Requirements

### System Prerequisites
- NVIDIA GPU with CUDA Compute Capability ≥ 3.5
- CUDA Toolkit 11.0+ (usually auto-installed via PyTorch)
- Python 3.8+

### Check Your Setup
```bash
# Verify CUDA availability
python torch_init.py

# Expected output:
# CUDA available: True
# CUDA Version: 12.1
# GPU Name: NVIDIA A100-SXM4-40GB
# PyTorch Version: 2.0.0
```

### Install Dependencies

From the `week1/` directory:
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `torch`, `torchvision`, `torchaudio` - PyTorch and computer vision
- `fastapi`, `uvicorn` - Web framework for inference service
- `pydantic` - Data validation for API requests

---

## 🚀 How to Run

### 1. Training a Model on CIFAR-10

**Quick Start (3 epochs for testing):**
```bash
cd 01_single_gpu_baseline

python train_single.py \
    --data-root /tmp/cifar \
    --epochs 3 \
    --batch-size 128 \
    --learning-rate 0.1
```

**Full Training (30 epochs for convergence):**
```bash
python train_single.py \
    --data-root /tmp/cifar-full \
    --epochs 30 \
    --batch-size 128 \
    --learning-rate 0.1 \
    --num-workers 4
```

**Parameter Explanations:**
- `--data-root`: Where to download/store CIFAR-10 dataset
- `--epochs`: Number of full passes through training data
- `--batch-size`: Samples per gradient update (higher = more GPU memory)
- `--learning-rate`: Step size for SGD optimizer
- `--num-workers`: Parallel processes for data loading (use CPU cores, not more than available)

### 2. Running the Inference Service

**In one terminal, start the service:**
```bash
cd 01_inference_service

# Run the FastAPI server
python app.py

# Expected output:
# [startup] loading model on device=cuda
# [startup] ready
# INFO:     Uvicorn running on http://0.0.0.0:8000
```

**In another terminal, test the endpoints:**

```bash
# Check readiness (before model loads, will return 503)
curl http://localhost:8000/health
# {"status": "ok", "device": "cuda"}

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}'

# Response:
# {"probs": [0.33, 0.33, 0.34], "latency_ms": 1.23}

# Get Prometheus metrics
curl http://localhost:8000/metrics
# inference_requests_total 1
# inference_errors_total 0
# inference_latency_ms_avg 1.23
```

**Test with Python:**
```python
import requests

# Make predictions
for i in range(5):
    resp = requests.post("http://localhost:8000/predict", json={
        "features": [0.1 * j for j in range(16)]
    })
    print(f"Request {i}: latency={resp.json()['latency_ms']:.2f}ms")
```

---

## 📊 Understanding Results

### Training Output Interpretation

**What you should see:**
```
Epoch 1/3
Train Loss: 2.3045 | Val Loss: 2.1203 | Val Acc: 22.5%
Epoch 2/3
Train Loss: 1.8932 | Val Loss: 1.7654 | Val Acc: 35.8%
Epoch 3/3
Train Loss: 1.5123 | Val Loss: 1.4521 | Val Acc: 48.2%
```

**Key Metrics:**
- **Training Loss**: Should decrease over time (indicates learning)
- **Validation Loss**: Should follow training loss (indicates generalization)
- **Validation Accuracy**: Should increase toward 100% (goal: >85% on CIFAR-10)

**What to watch for:**
- ⚠️ Loss not decreasing? → Learning rate too low or data problem
- ⚠️ Validation loss increasing while training loss decreases? → Overfitting (use regularization)
- ⚠️ GPU out of memory? → Reduce `--batch-size`
- ⚠️ Data loading slow? → Increase `--num-workers`

### Inference Service Output Interpretation

**Predictions:**
```json
{
  "probs": [0.25, 0.35, 0.40],
  "latency_ms": 1.45
}
```
- `probs`: Probability distribution over 3 classes (sums to ~1.0)
- `latency_ms`: How long the prediction took (measure of responsiveness)

**Metrics:**
```
inference_requests_total 100        # Total requests served
inference_errors_total 0             # Failures (should be 0!)
inference_latency_ms_avg 1.23        # Average response time
```

**Expected Performance:**
- **Latency**: < 5ms for batch=1 on GPU
- **Throughput**: > 1000 requests/sec with batching
- **Errors**: 0% (reliability is critical for production)

---

## 🔄 How to Compare Results

### Metric Comparison

| Metric | Target | Good | Acceptable | Bad |
|--------|--------|------|-----------|-----|
| **Final Val Acc** | >85% | >80% | 60-80% | <60% |
| **Training Time (30 epochs)** | <300s | <300s | 300-500s | >500s |
| **Inference Latency** | <2ms | <5ms | 5-20ms | >20ms |
| **Model Throughput** | >500 req/s | >500 | 100-500 | <100 |

### Reproducing Consistent Results

The training uses fixed random seeds for reproducibility:
```bash
# Same seed → Same loss curve
python train_single.py --data-root /tmp/cifar --epochs 3 --seed 42
python train_single.py --data-root /tmp/cifar --epochs 3 --seed 42
# Both should produce identical loss values
```

---

## 🧪 Hands-On Experiments

Try these experiments to deepen your understanding:

### Experiment 1: Batch Size Impact
```bash
# Small batch (slower, more noisy)
python train_single.py --batch-size 32 --epochs 3

# Large batch (faster, but may not converge as well)
python train_single.py --batch-size 256 --epochs 3

# Compare: Which converges faster? Which uses more GPU memory?
```

### Experiment 2: Learning Rate Sensitivity
```bash
# Conservative LR
python train_single.py --learning-rate 0.01 --epochs 3

# Aggressive LR (might diverge)
python train_single.py --learning-rate 1.0 --epochs 3

# Find the sweet spot: Which converges fastest?
```

### Experiment 3: Data Loader Workers
```bash
# Without parallel loading (num_workers=0)
python train_single.py --num-workers 0 --epochs 1

# With parallel loading (num_workers=4)
python train_single.py --num-workers 4 --epochs 1

# Compare: Which is faster? Check GPU utilization with nvidia-smi
```

### Experiment 4: Inference Batching
```bash
# Change app.py line ~120: uncomment batching code
# Send multiple requests while measuring throughput
# Measure: Does batching improve latency?
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory" Error
**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
1. **Reduce batch size**: `--batch-size 64` (instead of 128)
2. **Reduce model size**: Modify `train_single.py` line 24
3. **Enable gradient checkpointing**: See PyTorch docs
4. **Check GPU memory**: `nvidia-smi` shows current usage

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
# Reinstall PyTorch
pip install --upgrade torch torchvision torchaudio

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Issue: Inference Service Won't Start
**Problem:** Address already in use (port 8000 taken)

**Solutions:**
```bash
# Option 1: Use different port
python app.py --port 9000

# Option 2: Kill existing process
lsof -ti:8000 | xargs kill -9

# Option 3: Check what's using the port
netstat -tuln | grep 8000
```

### Issue: Training is Very Slow
**Diagnosis:**
```bash
# Check GPU utilization
nvidia-smi -l 1

# If GPU utilization < 50%:
#   - Increase batch size
#   - Increase num_workers (more CPU cores for data loading)
#   - Check disk I/O (first epoch loads dataset from disk)
```

---

## 📚 Resources

### In This Week's Folders
- **PDFs**: Presentation slides and theory
- **PPTx**: Interactive presentations
- Assignment solutions and reference materials

### PyTorch Documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [DataLoader Docs](https://pytorch.org/docs/stable/data.html)
- [Distributed Training](https://pytorch.org/docs/stable/distributed.html)

### FastAPI Documentation
- [FastAPI Starters](https://fastapi.tiangolo.com/)
- [Async/await for performance](https://fastapi.tiangolo.com/async-concurrency/)

### GPU Optimization
- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch GPU Performance](https://pytorch.org/docs/stable/notes/cuda.html)

---

## ✅ Verification Checklist

Before moving to Week 2, verify:
- [ ] You can run `python torch_init.py` and see GPU available
- [ ] CIFAR-10 dataset downloads and training runs
- [ ] Final validation accuracy > 40% after 3 epochs
- [ ] Inference service starts and responds to `/health` endpoint
- [ ] You can make predictions and get valid latency measurements
- [ ] You understand the difference between training and inference

---

## 🎓 Concepts for Next Week

In **[Week 2: Single-Node Multi-GPU (DDP)](../week2/README.md)**, you'll learn:
- How to scale training from 1 GPU → 2-8 GPUs on one machine
- Distributed Data Parallel (DDP) framework
- Using `torchrun` for process spawning and coordination
- Why distributed training needs synchronized batch norm and sampling

**Preview**: The training code from Week 1 will be adapted to use DDP. The convergence should match when you adjust batch sizes and learning rates proportionally.

---

## 📞 Getting Help

**For Week 1 specific issues:**
1. Check the Troubleshooting section above
2. Run `torch_init.py` to diagnose environment
3. Check the `requirements.txt` matches your installation
4. Look at the code comments for parameter explanations

**For general PyTorch questions:**
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)

---

**Last Updated**: October 2026  
**Next Week**: [Week 2: Single-Node Multi-GPU (DDP) →](../week2/README.md)

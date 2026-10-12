# Week 2: Single-Node Multi-GPU with Distributed Data Parallel (DDP)

## 🎯 Learning Objectives

By the end of Week 2, you will understand and be able to:
- ✅ Launch distributed training across multiple GPUs on one machine
- ✅ Use `torchrun` for automatic process spawning and coordination
- ✅ Understand Distributed Data Parallel (DDP) synchronization
- ✅ Adapt training code for distributed execution
- ✅ Scale batch size and learning rate for multiple GPUs
- ✅ Containerize and deploy distributed training in Docker
- ✅ Set up CI/CD pipelines for training automation

---

## 📚 Key Concepts

### Distributed Training Fundamentals
- **Process Group Initialization**: How DDP finds and coordinates processes
- **Rank & World Size**: Each process has a unique rank, world_size is total count
- **NCCL Backend**: NVIDIA Collective Communications Library for GPU-to-GPU sync
- **Gradient Synchronization**: All-reduce operations across GPUs

### torchrun Launcher
- **Process Spawning**: Automatically launches one process per GPU
- **Environment Variables**: Sets RANK, LOCAL_RANK, WORLD_SIZE automatically
- **Fault Tolerance**: Elastic training support for dynamic scaling
- **Debugging**: How to diagnose distributed communication failures

### Single-Node vs Multi-GPU
- **Data Parallelism**: Each GPU has the same model, processes different data
- **Synchronized Gradients**: Each step waits for all GPUs before updating
- **Effective Batch Size**: Per-GPU batch × world_size = total batch
- **Scaling Laws**: How to adjust learning rate when scaling

### Docker & Containerization
- **Layer Caching**: Docker build optimization for ML training
- **GPU in Containers**: Runtime flags for GPU access
- **Network Isolation**: Overlay networks for multi-node training

### CI/CD for Training
- **Automated Testing**: Smoke tests that verify distributed setup
- **Pipeline Stages**: Build → Test → Train
- **Artifact Management**: Saving models from training

---

## 📁 Folder Structure

```
week2/
├── README.md                          # This file
├── 02_ddp_single_node/                # DDP training implementation
│   ├── train_ddp.py                  # Single-node multi-GPU training
│   └── requirements.txt               # DDP-specific dependencies
├── 02_gpu_pod/                        # Kubernetes pod examples
│   └── gpu-training-pod.yaml         # K8s manifest for training
└── week02_containers/                 # Containerization examples
    ├── 01_minimal_gpu/               # Minimal GPU Dockerfile
    ├── 02_jupyter_dev/               # Development Jupyter environment
    ├── 03_cuda_programs/             # CUDA application examples
    └── 04_ci_pipeline/               # CI/CD pipeline examples
        ├── Dockerfile                # Production training container
        ├── ci_pipeline.yaml          # GitHub Actions / GitLab CI config
        └── busy_train.py             # Load test for pipeline
```

### What's in Each Folder?

**`02_ddp_single_node/`**
- `train_ddp.py`: Adapted version of Week 1's single-GPU training
  - **Differences from Week 1**: 
    - Calls `ddp_setup()` to initialize process group
    - Wraps model in `DistributedDataParallel`
    - Uses `DistributedSampler` for data sharding
    - Only rank-0 logs and saves checkpoints
  - **Launch**: Via `torchrun`, not direct `python`

**`week02_containers/01_minimal_gpu/`**
- Minimal Dockerfile showing GPU support
- Focus: Base image + PyTorch + entrypoint

**`week02_containers/02_jupyter_dev/`**
- Development environment with Jupyter notebook
- Interactive exploration and debugging

**`week02_containers/04_ci_pipeline/`**
- Production-grade CI pipeline
- `Dockerfile`: Multi-stage build for efficiency
- `ci_pipeline.yaml`: Automated test → build → train workflow

---

## 🔧 Setup & Requirements

### System Prerequisites
- Multiple NVIDIA GPUs visible to same machine (typically 2-8)
- NCCL library (comes with PyTorch)
- Docker (for containerization examples)

### Verify Multi-GPU Setup
```bash
cd 02_ddp_single_node

# Check available GPUs
python -c "import torch; print(torch.cuda.device_count())"
# Expected output: 2 (or 4, 8, etc.)

nvidia-smi
# Shows all GPUs and their memory
```

### Install Dependencies
```bash
# DDP-specific dependencies (same as Week 1 + distributed libraries)
pip install -r requirements.txt
```

**Additional packages vs Week 1:**
- `torch.distributed` (included with PyTorch)
- NCCL (comes with PyTorch CUDA build)

---

## 🚀 How to Run

### 1. Single-Node Multi-GPU Training

**Basic Launch (auto-detect 2 GPUs):**
```bash
cd 02_ddp_single_node

# Automatic: spawns 1 process per available GPU
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --epochs 3 \
    --per-gpu-batch-size 128
```

**Explicit GPU Count (test with 4 GPUs):**
```bash
torchrun --standalone --nproc_per_node=4 train_ddp.py \
    --epochs 3 \
    --per-gpu-batch-size 128 \
    --data-root /tmp/cifar
```

**Full Training Run (30 epochs, 8 GPUs):**
```bash
torchrun --standalone --nproc_per_node=8 train_ddp.py \
    --epochs 30 \
    --per-gpu-batch-size 64 \
    --learning-rate 0.8 \
    --data-root /tmp/cifar-full \
    --num-workers 8
```

**Parameter Explanations:**
- `--standalone`: Run on single machine (vs multi-node)
- `--nproc_per_node`: Number of processes (GPUs) per machine
- `--per-gpu-batch-size`: Batch size PER GPU (total = this × num_gpus)
- `--learning-rate`: Should be scaled with total batch size
- `--num-workers`: DataLoader workers per process (not global)

### 2. Understanding torchrun Output

**Expected log output from 2 GPUs:**
```
Rank 0 | GPU 0 | Epoch 1/3
Epoch 1/3, Rank=0, Step=100, Loss=2.15, LR=0.1
Rank 1 | GPU 1 | (gradients syncing...)
Epoch 1/3, Rank=0, Step=200, Loss=1.98, LR=0.1
...
All-reduce sync: 8.5ms
Final validation (rank 0 only): Loss=1.65, Acc=42.3%
```

**What's happening:**
- Both ranks train in parallel
- All-reduce waits for all GPUs before gradient update
- Only rank-0 prints and evaluates (avoids duplicate output)

### 3. Batch Size and Learning Rate Scaling

**Linear Scaling Rule**:
- **Week 1 (1 GPU)**: `--batch-size 128, --learning-rate 0.1`
- **Week 2 (2 GPUs)**: `--per-gpu-batch-size 128, --learning-rate 0.2` (LR × 2)
- **Week 2 (4 GPUs)**: `--per-gpu-batch-size 128, --learning-rate 0.4` (LR × 4)

**Why scale LR?** 
Larger effective batch → larger gradient magnitudes → need larger learning rate steps

**Test Scaling:**
```bash
# Run both and compare convergence
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --per-gpu-batch-size 128 --learning-rate 0.1 --epochs 3
# (Lower LR, might not converge well)

torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --per-gpu-batch-size 128 --learning-rate 0.2 --epochs 3
# (Scaled LR, should converge better)
```

### 4. Running in Docker

**Build Training Container:**
```bash
cd week02_containers/04_ci_pipeline

# Build image
docker build -t ml-training:latest .

# List layers to see caching
docker history ml-training:latest
```

**Run Training in Container:**
```bash
# Single GPU in container
docker run --gpus 1 ml-training:latest train_ddp.py --epochs 1

# Multi-GPU in container (2 GPUs)
docker run --gpus 2 ml-training:latest \
    torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 1
```

**Docker GPU Flags:**
- `--gpus 1`: Access 1 GPU
- `--gpus all`: Access all GPUs
- `--gpus device=0,1`: Specific GPUs

### 5. CI/CD Pipeline Example

**Automated Pipeline (GitHub Actions / GitLab CI):**
```yaml
stages:
  - build
  - test
  - train

build:
  script: docker build -t ml-training .

test:
  script: docker run ml-training python -m pytest tests/

train:
  script: docker run --gpus 2 ml-training \
          torchrun --nproc_per_node=2 train_ddp.py --epochs 3
```

---

## 📊 Understanding Results

### DDP Training Output Interpretation

**Expected output (2 GPUs):**
```
torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 3
```

```
[2024-01-15 10:23:45] Rank 0 initializing process group...
[2024-01-15 10:23:46] Rank 0 | Process group initialized (world_size=2)
[2024-01-15 10:23:47] Rank 1 | Loaded CIFAR-10 (50000 samples per rank)
[2024-01-15 10:23:48] Rank 0 | Epoch 1/3
[2024-01-15 10:23:50] Rank 0 | Step 100/391 | Loss: 2.30 | LR: 0.20
[2024-01-15 10:23:52] Rank 0 | Step 200/391 | Loss: 2.15 | LR: 0.20
...
[2024-01-15 10:24:30] Rank 0 | Val Loss: 2.10 | Val Acc: 24.5% | Time: 42.5s
[2024-01-15 10:24:32] Rank 0 | Epoch 2/3
...
[2024-01-15 10:26:45] Rank 0 | Training complete!
```

**Key Differences from Week 1:**
- **Ranks printing**: Both GPU processes log (but other ranks' data loading message shown)
- **Process group** initialization happens at startup
- **Validation runs only on Rank 0** (avoids duplicate compute)
- **Training time is roughly 1/N** of single-GPU time (N = number of GPUs)

### Performance Comparison

| Metric | 1 GPU | 2 GPU | 4 GPU | 8 GPU |
|--------|-------|-------|-------|-------|
| **Epoch Time** | 45s | 24s | 13s | 7s |
| **Speedup** | 1.0x | 1.87x | 3.4x | 6.4x |
| **Efficiency** | 100% | 93% | 85% | 80% |

**Efficiency = Speedup / GPU_count**
- Ideal: 100% (2 GPUs = 2x faster)
- Real: ~80-90% (communication overhead, load imbalance)

### Expected Accuracy Progression

With properly scaled learning rate:
```
Epoch 1: Loss 2.30 → 1.80 | Acc: 25% → 40%
Epoch 2: Loss 1.80 → 1.50 | Acc: 40% → 50%
Epoch 3: Loss 1.50 → 1.35 | Acc: 50% → 55%
```

⚠️ **If accuracy much lower than Week 1**: Check if learning rate was scaled correctly!

---

## 🔄 How to Compare Results

### Comparing with Week 1 (Single GPU)

**Same effective batch, same LR scaling:**
```bash
# Week 1 baseline (1 GPU)
cd week1/01_single_gpu_baseline
python train_single.py --batch-size 256 --learning-rate 0.1 --epochs 3

# Week 2 with 2 GPUs (same effective batch=256, scaled LR=0.2)
cd week2/02_ddp_single_node
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --per-gpu-batch-size 128 --learning-rate 0.2 --epochs 3
```

**Compare metrics:**
- ✅ Final accuracy should be nearly identical (within ±1%)
- ✅ Epoch time should be ~2x faster with 2 GPUs
- ✅ Training curves should have same shape

**If curves differ significantly:**
- 🔍 Check if learning rate was properly scaled
- 🔍 Verify all GPUs are actually being used (nvidia-smi during run)
- 🔍 Check for gradient synchronization issues in logs

### Communication Overhead Measurement

```bash
# Count all-reduce operations
# Add this to train_ddp.py after sync_gradients():
#   print(f"All-reduce time: {time.time() - t0:.3f}s")

# Expected:
# - 1-2ms per all-reduce (modern interconnect)
# - ~400 all-reduces per epoch (391 steps for CIFAR-10)
# - Total communication time: < 1 second per epoch
```

---

## 🧪 Hands-On Experiments

### Experiment 1: Scaling Law
```bash
# Run with 1, 2, 4 GPUs and plot epoch time
for n_gpu in 1 2 4; do
    torchrun --standalone --nproc_per_node=$n_gpu train_ddp.py --epochs 1
done
# Plot: GPUs vs Speedup (should be close to linear)
```

### Experiment 2: Learning Rate Sensitivity
```bash
# Test different LR scaling strategies
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --per-gpu-batch-size 128 --learning-rate 0.1 --epochs 3
# (Under-scaled LR - slower convergence)

torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --per-gpu-batch-size 128 --learning-rate 0.3 --epochs 3
# (Over-scaled LR - might diverge)
```

### Experiment 3: Communication Measurement
Modify train_ddp.py to measure all-reduce time:
```python
import time
t0 = time.time()
dist.all_reduce(loss_tensor)  # Synchronization point
comm_time = time.time() - t0
print(f"Rank {rank} | All-reduce: {comm_time*1000:.2f}ms")
```

### Experiment 4: GPU Utilization
```bash
# In separate terminal, monitor GPUs
watch -n 0.5 nvidia-smi

# Run training and observe:
# - Do all GPUs stay at ~100% utilization?
# - Does memory usage match per-GPU batch size?
# - Are there spikes in communication time?
```

---

## 🐛 Troubleshooting

### Issue: "RuntimeError: torch.cuda.is_available() is False"
**In DDP context**: One GPU process can't access GPU
```bash
# Check GPU visibility
nvidia-smi
torchrun --standalone --nproc_per_node=2 -c "import torch; print(torch.cuda.device_count())"
```

### Issue: "ProcessGroupNCCLError" or "All-Reduce Timeout"
**Symptom**: Training hangs after spawning processes

**Causes & Solutions**:
1. **Network misconfiguration**: Check MASTER_ADDR/PORT
   ```bash
   export MASTER_ADDR=127.0.0.1
   export MASTER_PORT=29500
   ```
2. **Incompatible NCCL version**: Verify with `python -c "import torch; print(torch.cuda.nccl.version())"`
3. **GPU not ready**: Try again (sometimes transient)

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce per-GPU batch size: `--per-gpu-batch-size 64`
2. Reduce num_workers: `--num-workers 2`
3. Use gradient checkpointing (advanced)

### Issue: "Cannot find torch.distributed module"
**Solution**: Ensure torch is installed correctly
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Only one GPU is used (despite setting nproc_per_node=2)
**Diagnosis:**
```bash
# Check if torchrun is actually spawning multiple processes
torchrun --standalone --nproc_per_node=2 -c "import os; print(os.environ.get('RANK'))"
# Should print: 0 \n 1

# Check nvidia-smi during training
nvidia-smi dmon -s p  # Monitor process memory
```

### Issue: Docker container can't see GPUs
```bash
# Ensure NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime nvidia-smi

# If not working, try:
docker run --rm --gpus all --runtime=nvidia nvidia/cuda:11.8.0-runtime nvidia-smi
```

---

## 📚 Resources

### In This Week's Folders
- **02_ddp_single_node/train_ddp.py**: Heavily commented DDP implementation
- **week02_containers/**: Complete Dockerfile examples
- **PDFs in pdfs/**: Detailed explanations and theory

### PyTorch Distributed Documentation
- [torch.distributed](https://pytorch.org/docs/stable/distributed.html)
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun Launcher](https://pytorch.org/docs/stable/elastic/quickstart.html)

### NCCL Documentation
- [NCCL Best Practices](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- [All-Reduce Optimization](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)

### Docker for ML
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Learning Rate Scaling
- [Linear Scaling Rule](https://arxiv.org/abs/1708.03888)
- [Batch Size Effects](https://arxiv.org/abs/1609.04836)

---

## ✅ Verification Checklist

Before moving to Week 3, verify:
- [ ] You can run `torchrun --standalone --nproc_per_node=2` successfully
- [ ] Both GPU processes initialize (check logs for "Rank 0" and "Rank 1")
- [ ] Training converges with 2+ GPUs (accuracy similar to Week 1)
- [ ] Epoch time scales roughly linearly with GPU count
- [ ] Learning rate scaling affects convergence (you understand why)
- [ ] Docker image builds and runs training successfully
- [ ] You can see GPU utilization during training (nvidia-smi)

---

## 🎓 Concepts for Next Week

In **[Week 3: Multi-Node Distributed Training](../week3/README.md)**, you'll learn:
- How to scale from 1 machine (Week 2) → multiple machines
- Network topology and latency-aware optimizations
- Fault tolerance and training resume
- Cloud deployment (AWS, GCP, Azure)

**Preview**: Same DDP code, but with `torchrun` pointing to multiple hosts and different communication backends (NCCL over Ethernet).

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 1: Single GPU Foundations](../week1/README.md)  
**Next Week**: [Week 3: Multi-Node Distributed Training →](../week3/README.md)

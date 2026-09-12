# Quick Reference Guide - Common Tasks

Quick answers for getting things done fast. For detailed explanations, see `NEW_JOINER_GUIDE.md`.

## 🏃 5-Minute Tasks

### Run Single GPU Training
```bash
cd week1/01_single_gpu_baseline/
python train_single.py --epochs 1 --batch-size 128 --data-root /tmp/cifar
```

### Run Inference Service Locally
```bash
cd week1/01_inference_service/
pip install fastapi uvicorn torch pydantic
python app.py
# Open browser: http://localhost:8000/docs
```

### Run DDP on Single Node (2 GPUs)
```bash
cd week2/02_ddp_single_node/
torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 1 --per-gpu-batch-size 128
```

### Check GPU Availability
```bash
nvidia-smi                    # See all GPUs
nvidia-smi -q                 # Detailed info
nvidia-smi dmon -s muc        # Real-time monitoring
```

### Build GPU Container
```bash
cd week2/week02_containers/01_minimal_gpu/
docker build -t gpu-test:latest .
docker run --gpus all -it gpu-test:latest nvidia-smi
```

---

## 📂 File Location Cheat Sheet

| What I Want | Where to Find | Start Here |
|------------|---------------|-----------|
| **Basic GPU training** | `week1/01_single_gpu_baseline/` | `train_single.py` |
| **Inference service** | `week1/01_inference_service/` | `app.py` + `Dockerfile` |
| **Multi-GPU training** | `week2/02_ddp_single_node/` | `train_ddp.py` |
| **CUDA examples** | `week2/week02_containers/03_cuda_programs/` | `vector_add.cu` |
| **Kubernetes specs** | `week3/03_networking_config/` | `07_statefulset.yaml` |
| **Service mesh** | `week4/04_istio/` | `10_virtualservice.yaml` |
| **NCCL benchmarks** | `week4/04_nccl_understanding/` | `allreduce_benchmark.py` |
| **Scaling study** | `week8/Codes/01_lr_scaling_study/` | Start here for advanced topics |

---

## 🐛 Debugging Common Issues

### Problem: `cuda runtime error(804)` or GPU OOM
**Solution**:
```python
# In your training script, check:
torch.cuda.empty_cache()  # Force cleanup
torch.cuda.memory_reserved()  # See reserved memory
torch.cuda.memory_allocated()  # See used memory

# Reduce batch size if OOM
python train_single.py --batch-size 64  # Instead of 128
```

### Problem: DDP hangs on `init_process_group()`
**Solution**: Ensure environment variables are set correctly
```bash
# Manual setup if torchrun doesn't work
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0

python -m torch.distributed.launch --nproc_per_node=2 train_ddp.py
```

### Problem: `torch.distributed.RuntimeError: Address already in use`
**Solution**: Free the port
```bash
lsof -ti:29500 | xargs kill -9  # Kill process on MASTER_PORT
# Then retry training
```

### Problem: Container can't see GPU
**Solution**: Check Docker GPU plugin
```bash
docker run --gpus all -it nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
# If fails, install NVIDIA container runtime
# See: https://github.com/NVIDIA/nvidia-docker
```

### Problem: Kubernetes pod can't access GPU
**Solution**: Verify GPU device plugin is installed
```bash
kubectl get nodes -o wide
kubectl describe nodes | grep nvidia  # Should show gpu: X

# If not showing, install:
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

### Problem: Training loss diverging with DDP
**Common cause**: Incorrect batch size or learning rate scaling

**Fix**: Match effective batch size
```python
# Week 1 (single GPU):
batch_size = 128
lr = 0.1

# Week 2 (2 GPUs with DDP):
per_gpu_batch_size = 64      # 64 * 2 GPUs = 128 effective batch size
lr = 0.1                      # Or scale: lr = 0.1 * sqrt(2) if using linear scaling rule

# Note: Different scaling rules exist (see week8 for details)
```

---

## 🔍 Code Inspection Patterns

### Find All Training Scripts
```bash
find . -name "*train*.py" -type f | head -20
```

### Find All YAML Files (Kubernetes)
```bash
find . -name "*.yaml" -o -name "*.yml" | sort
```

### Find All Dockerfiles
```bash
find . -name "Dockerfile" -exec dirname {} \;
```

### Count Lines of Code per Week
```bash
for week in week*/; do echo "$week: $(find $week -name '*.py' -type f -exec wc -l {} + | tail -1)"; done
```

---

## 📊 Performance Metrics You Should Track

### Single GPU (Week 1)
```python
# In training loop:
samples_per_sec = batch_size / batch_time
print(f"Throughput: {samples_per_sec:.2f} samples/sec")
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
```

### Multi-GPU DDP (Week 2-3)
```python
# Also track:
comm_time = measure_all_reduce_time()  # See week4/allreduce_benchmark.py
scaling_efficiency = (single_gpu_throughput * num_gpus) / ddp_throughput
print(f"Scaling Efficiency: {scaling_efficiency:.2%}")
```

### For Scaling Analysis (Week 6)
```
Strong Scaling: Fix total data, vary GPUs
- Ideal: 2x GPUs → 2x speedup (90%+ efficiency is good)

Weak Scaling: Scale data with GPUs
- Ideal: 2x GPUs, 2x data → same time (95%+ efficiency is great)
```

---

## 🚀 Quick Environment Setup

### On Ubuntu/Debian
```bash
# Install NVIDIA drivers
ubuntu-drivers autoinstall

# Install NVIDIA CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update && sudo apt install cuda

# Install cuDNN
# Download from nvidia.com, then:
sudo cp cudnn/lib/* /usr/local/cuda/lib64/
sudo cp cudnn/include/* /usr/local/cuda/include/
```

### Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Docker-Based Setup (Recommended)
```bash
cd week2/week02_containers/02_jupyter_dev/
docker-compose up
# Access Jupyter at http://localhost:8888
```

---

## 📖 Reading Order by Interest

### 🏃 "I Just Want to Run Code"
1. Week 1 → `train_single.py`
2. Week 2 → `train_ddp.py`
3. Week 2 → Docker setup
4. Done! You know the basics.

### 🎓 "I Want to Understand Everything"
1. NEW_JOINER_GUIDE.md (this file)
2. Week 1 → Train single GPU
3. Week 2 → Learn DDP architecture + launch script
4. Week 3 → Kubernetes networking
5. Week 4 → NCCL communication
6. Week 8 → Advanced techniques

### ☸️ "I'm Interested in Kubernetes"
1. Week 3 → Start with `03_networking_config/`
2. Read YAML files in order: ConfigMap → Pod → Service → StatefulSet
3. Week 4 → Istio for service mesh
4. Week 3 assignment PDF for hands-on tasks

### 💻 "I'm Interested in CUDA Programming"
1. Week 2 → `week02_containers/03_cuda_programs/vector_add.cu`
2. Week 2 → `bandwidth_test.cu`
3. Week 4 → `04_nccl_understanding/nccl_allreduce.cu`
4. Week 4 → `allreduce_benchmark.py` (Python wrapper)

### 🚀 "I'm Interested in Production Deployment"
1. Week 1 → Inference service (`01_inference_service/`)
2. Week 2 → Docker + CI/CD
3. Week 3 → Kubernetes basics
4. Week 4 → Istio service mesh
5. Week 8 → Deployment engineering

---

## ✅ Checklist: Getting Started

- [ ] Clone the repository
- [ ] Read `NEW_JOINER_GUIDE.md`
- [ ] Run `nvidia-smi` to verify GPU setup
- [ ] Run Week 1 single-GPU training
- [ ] Read Week 1's `train_single.py`
- [ ] Run Week 2 DDP training (if 2+ GPUs available)
- [ ] Read Week 2's `train_ddp.py` comments
- [ ] Build a GPU Docker container
- [ ] Pick a week based on your interest
- [ ] Work through the assignment PDFs

---

## 🔗 Key Concepts Map

```
GPU Basics (Week 1)
    ↓
Single GPU Training (Week 1)
    ↓
Multi-GPU Training: DDP (Week 2)
    ├─ Containerization (Week 2)
    │   ├─ Docker (Week 2)
    │   └─ CUDA Programming (Week 2)
    └─ Multi-Node Training (Week 3)
        ├─ Kubernetes Basics (Week 3)
        ├─ Pod Networking (Week 3)
        └─ Advanced Networking: Istio (Week 4)
              └─ NCCL Optimization (Week 4)
                  
TensorFlow Alternative (Week 5) ← Can go in parallel
Scaling Analysis (Week 6) ← Happens at any point
Advanced Topics (Week 8)
    ├─ LR Scaling
    ├─ Checkpointing
    └─ Packaging & Deployment
```

---

## 🎯 Success Metrics

After each week, you should be able to:

**Week 1**: ✓ Train ResNet on CIFAR-10, ✓ Run inference service  
**Week 2**: ✓ Train with 2+ GPUs using DDP, ✓ Build GPU containers  
**Week 3**: ✓ Deploy training pod to Kubernetes, ✓ Understand pod networking  
**Week 4**: ✓ Deploy with Istio routing, ✓ Understand NCCL communication  
**Week 8**: ✓ Implement LR scaling, ✓ Use gradient checkpointing  

---

## 💬 Common Questions While Reading Code

| Question | Answer | File |
|----------|--------|------|
| Where's the `torchrun` magic? | Environment setup in launcher | `launch.sh`, `launch_multinode.sh` |
| Why `model.module.state_dict()`? | DDP wraps model, need unwrap | `train_ddp.py` comments |
| What's `MASTER_ADDR`? | Rank 0's hostname (training coordinator) | Week 3 YAML files |
| How's gradient synchronized? | NCCL all-reduce after backward() | `train_ddp.py` + `allreduce_benchmark.py` |
| Why StatefulSet not Deployment? | Need stable pod names (pod-0, pod-1, ...) | `week3/03_networking_config/07_statefulset.yaml` |

---

## 🔐 Important Defaults in This Course

| Setting | Default | Where |
|---------|---------|-------|
| Model | ResNet18 | Week 1-3 |
| Dataset | CIFAR-10 | Week 1-3 |
| Batch Size | 128 | Week 1-3 |
| Optimizer | SGD | Week 1-3 |
| Learning Rate | 0.1 | Week 1-3 |
| Backend | NCCL (GPU) / gloo (CPU) | Week 2+ |
| GPU Framework | PyTorch | Week 1-4, 7-8 |
| Container Base | `nvidia/cuda:11.x.x-runtime` | Week 2 |
| Orchestration | Kubernetes | Week 3+ |

---

*Need more help? See NEW_JOINER_GUIDE.md for detailed explanations.*

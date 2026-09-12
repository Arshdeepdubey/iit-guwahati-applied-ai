# IIT Guwahati Applied Accelerated AI - New Joiner Guide

Welcome! This repository contains course materials and code for **Applied Accelerated AI**, covering GPU computing, distributed training, containerization, and Kubernetes orchestration.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Course Overview](#course-overview)
- [Key Technologies](#key-technologies)
- [Setup & Environment](#setup--environment)
- [Learning Path](#learning-path)
- [Important Concepts](#important-concepts)
- [FAQ](#faq)

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** - Core programming language
- **PyTorch** - Deep learning framework
- **Docker & Docker Compose** - Containerization
- **NVIDIA GPU** (optional) - CUDA-capable GPU recommended for full experience
- **kubectl** - Kubernetes command-line tool (for week 3+)

### First Steps
1. **Clone and explore**: You're already here!
2. **Read this guide** first (5 min read)
3. **Start with Week 1**: GPU basics and single-GPU training
4. **Progress sequentially**: Each week builds on previous concepts

---

## 📁 Repository Structure

```
iit-guwahati-applied-ai/
├── week1/          # GPU Fundamentals & Single-GPU Training
├── week2/          # Distributed Training (DDP) & Containerization
├── week3/          # Multi-Node Training & Kubernetes
├── week4/          # Service Mesh (Istio) & NCCL Optimization
├── week5/          # TensorFlow Acceleration
├── week6/          # Scaling & Benchmarking
├── week7/          # Consolidated Examples (weeks 1-4 recap)
├── week8/          # Advanced Topics (LR Scaling, Checkpointing)
├── week9-12/       # Additional Topics
└── README.md       # (Currently empty - see this guide!)
```

---

## 📚 Course Overview

### Week-by-Week Breakdown

#### **Week 1: GPU Fundamentals & Single-GPU Training**
**Focus**: Basic GPU programming and model training on a single GPU

**Key Directories**:
- `week1/01_single_gpu_baseline/` - ResNet18 training on CIFAR-10
- `week1/01_inference_service/` - FastAPI-based model inference service

**Key Files**:
- `train_single.py` - PyTorch training script for single GPU
- `app.py` - FastAPI inference server with health checks and metrics
- `Dockerfile` - Containerized inference service

**Concepts Covered**:
- GPU vs CPU computation
- CUDA fundamentals
- PyTorch training loops
- Model inference services
- Docker containerization basics

**What You'll Learn**:
```
GPU Hardware → PyTorch Setup → Single GPU Training → Inference Service
```

---

#### **Week 2: Distributed Training & Containerization**
**Focus**: Multi-GPU training on a single node + Docker deep dive

**Key Directories**:
- `week2/02_ddp_single_node/` - PyTorch DistributedDataParallel (DDP)
- `week2/02_gpu_pod/` - Kubernetes GPU pod specs
- `week2/week02_containers/` - CUDA programming and container techniques
  - `01_minimal_gpu/` - Minimal GPU container
  - `02_jupyter_dev/` - Jupyter + GPU development environment
  - `03_cuda_programs/` - CUDA kernels (vector addition, bandwidth test)
  - `04_ci_pipeline/` - CI/CD with GitHub Actions

**Key Files**:
- `train_ddp.py` - Multi-GPU training using DistributedDataParallel
- `launch.sh` - Script to launch DDP training
- `vector_add.cu` - CUDA kernel example (hello-world of CUDA)
- `bandwidth_test.cu` - GPU memory bandwidth benchmarking
- `build-and-test.yml` - GitHub Actions CI/CD pipeline

**Concepts Covered**:
- Distributed Data Parallel (DDP) principles
- Process groups and NCCL backend
- CUDA kernel programming
- Docker multi-stage builds
- GPU resource allocation in containers
- CI/CD pipelines with GPU support

**Critical DDP Patterns** (see `train_ddp.py`):
```python
# 1. Initialize process group
torch.distributed.init_process_group("nccl")

# 2. Wrap model in DDP
model = nn.DistributedDataParallel(model)

# 3. Use DistributedSampler to partition data
sampler = DistributedSampler(dataset)

# 4. Set epoch for correct shuffling
sampler.set_epoch(epoch)

# 5. Save only on rank 0
if rank == 0:
    torch.save(model.module.state_dict(), path)
```

---

#### **Week 3: Multi-Node Distributed Training & Kubernetes**
**Focus**: Scaling beyond single node + Kubernetes orchestration

**Key Directories**:
- `week3/03_ddp_multi_node/` - Multi-node DDP launcher
- `week3/03_networking_config/` - Kubernetes networking setup
  - ConfigMaps, Secrets, Training pods
  - Headless services, StatefulSets
  - LoadBalancer services

**Key Files**:
- `launch_multinode.sh` - Multi-node launch script
- `04_configmap.yaml` - Kubernetes ConfigMap for config sharing
- `05_training_pod.yaml` - Pod spec for distributed training
- `06_headless_svc.yaml` - Headless service for pod discovery
- `07_statefulset.yaml` - StatefulSet for ordered pod startup
- `08_loadbalancer_svc.yaml` - External load balancing

**Concepts Covered**:
- Multi-node communication via MPI
- Kubernetes pod networking
- Service discovery for distributed training
- StatefulSets vs Deployments
- ConfigMaps and Secrets management
- GPU resource requests in Kubernetes

**Kubernetes Networking Pattern**:
```yaml
# 1. Headless Service for pod discovery
Service (headless) → DNS: pod-0.svc.cluster.local

# 2. StatefulSet for stable pod identities
StatefulSet → Pod-0, Pod-1, ... (ordinal names)

# 3. ConfigMap for training config
ConfigMap → MASTER_ADDR=pod-0, MASTER_PORT=29500
```

---

#### **Week 4: Service Mesh (Istio) & NCCL Optimization**
**Focus**: Advanced networking and communication library optimization

**Key Directories**:
- `week4/04_istio/` - Istio service mesh configuration
- `week4/04_nccl_understanding/` - NCCL benchmark and optimization

**Key Files**:
- `llm_service_base.yaml` - Base service definition
- `09_istio_destinationrule.yaml` - Traffic routing rules
- `10_virtualservice.yaml` - Virtual service configuration
- `11_inference_v2.yaml` - Inference service with Istio
- `allreduce_benchmark.py` - NCCL collective communication benchmark
- `nccl_allreduce.cu` - CUDA implementation of all-reduce

**Concepts Covered**:
- Istio virtual services and destination rules
- Traffic management and canary deployments
- NCCL (NVIDIA Collective Communications Library)
- All-reduce optimization across GPUs
- Gradient synchronization patterns

---

#### **Week 5: TensorFlow Acceleration**
**Focus**: Accelerating TensorFlow training and inference

**Key Files**:
- `Week5_TF_Acceleration.ipynb` - Jupyter notebook with TensorFlow optimization techniques

**Concepts**:
- TensorFlow vs PyTorch comparison
- Graph execution optimization
- Mixed precision training
- Distributed TensorFlow strategies

---

#### **Week 6: Scaling & Benchmarking**
**Focus**: Performance measurement and scalability analysis

**Key Directories**:
- `week6/06_scaling_benchmark/` - Scaling benchmarks
- `week6/Code/` - Consolidated code from weeks 1-4

**Key Files**:
- `benchmark_scaling.py` - Script to benchmark scaling efficiency

**Concepts**:
- Strong vs weak scaling
- Speedup and efficiency metrics
- Communication overhead analysis

---

#### **Week 7: Consolidated Examples**
**Focus**: Unified implementations from weeks 1-4

**Contents**: Recreations of:
- Single GPU training
- Single-node DDP
- Multi-node DDP
- NCCL benchmarks

**Use**: Reference implementations and teaching examples

---

#### **Week 8: Advanced Topics**
**Focus**: Production-ready training techniques

**Key Directories**:
- `01_lr_scaling_study/` - Learning rate scaling for large batches
- `02_checkpointing/` - Gradient checkpointing for memory efficiency
- `04_packaging/` - Model packaging and versioning

**Concepts Covered**:
- Learning rate schedules for distributed training
- Activation checkpointing to reduce memory
- Model versioning and packaging
- Deployment engineering patterns

---

## 🛠️ Key Technologies

### Deep Learning Frameworks
| Technology | Purpose | Where Used |
|-----------|---------|-----------|
| **PyTorch** | Primary DL framework | Weeks 1-4, 7-8 |
| **TensorFlow** | Alternative DL framework | Week 5 |
| **torchvision** | Computer vision models | Weeks 1, 2 |

### GPU & Parallel Computing
| Technology | Purpose | Where Used |
|-----------|---------|-----------|
| **CUDA** | GPU programming | Weeks 2, 4 |
| **NCCL** | Collective communication | Weeks 2-4 |
| **cuDNN** | GPU-accelerated primitives | Implicit in PyTorch |
| **Apex/AMP** | Mixed precision training | Weeks 2, 8 |

### Containerization & Orchestration
| Technology | Purpose | Where Used |
|-----------|---------|-----------|
| **Docker** | Container images | Weeks 1-4, 8 |
| **Docker Compose** | Multi-container local dev | Week 2 |
| **Kubernetes** | Container orchestration | Weeks 3-4, 8 |
| **Istio** | Service mesh | Week 4 |

### Other Tools
| Technology | Purpose | Where Used |
|-----------|---------|-----------|
| **FastAPI** | Web framework for inference | Weeks 1, 3-4 |
| **Prometheus** | Metrics collection | Weeks 1, 8 |
| **PyYAML** | Configuration management | Weeks 3-4 |

---

## 🔧 Setup & Environment

### Local GPU Setup

**Check GPU availability**:
```bash
nvidia-smi
```

**Install PyTorch (CUDA-enabled)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify PyTorch GPU access**:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Docker Setup

**Build a GPU-enabled container** (example from week2):
```bash
cd week2/week02_containers/01_minimal_gpu/
docker build -t gpu-minimal:latest .
docker run --gpus all -it gpu-minimal:latest nvidia-smi
```

### Kubernetes Setup

**Prerequisites**:
- Kubernetes cluster with GPU node pool
- `kubectl` configured to access cluster
- Kubernetes GPU plugin (e.g., NVIDIA GPU device plugin)

**Verify GPU availability in cluster**:
```bash
kubectl get nodes -L nvidia.com/gpu
kubectl describe node <gpu-node>  # Look for nvidia.com/gpu
```

---

## 📖 Learning Path

### Recommended Progression

**Phase 1: Foundations (Weeks 1-2)** ⭐ START HERE
```
1. Week 1 → GPU Fundamentals
   └─ Run: python train_single.py --epochs 1 --batch-size 32
   └─ Build: Dockerfile, run inference service locally

2. Week 2 → Distributed Training (Single Node)
   └─ Read: train_ddp.py comments carefully
   └─ Run: torchrun --standalone --nproc_per_node=2 train_ddp.py
   └─ Build: GPU container, run GPU smoke tests
```

**Phase 2: Orchestration (Weeks 3-4)**
```
3. Week 3 → Kubernetes Basics
   └─ Setup: Local Kubernetes (minikube or Docker Desktop K8s)
   └─ Deploy: GPU pod specs, watch training via kubectl logs

4. Week 4 → Advanced Networking
   └─ Install: Istio service mesh (optional, requires cluster)
   └─ Understand: NCCL communication patterns
```

**Phase 3: Advanced Topics (Weeks 5-8)**
```
5. Week 5 → TensorFlow (if interested in TF)
6. Week 6 → Scalability Analysis
7. Week 8 → Production Techniques
```

### What to Read First
1. **This file** (you're reading it!) - 10 min
2. **Week 1 README** and `train_single.py` - 15 min
3. **Week 2 README** and `train_ddp.py` comments - 20 min
4. **Pick a week** and dive into code - 1-2 hours

---

## 💡 Important Concepts

### Distributed Data Parallel (DDP) - The Heart of Week 2-3

**What it is**: Each GPU gets a replica of the full model + a shard of the data.

**Training loop**:
```
1. Forward pass on each GPU (with its data shard)
2. Compute loss on each GPU
3. Backward pass computes gradients locally
4. All-reduce synchronizes gradients across GPUs (NCCL handles this)
5. Optimizer updates model on each GPU
6. Repeat
```

**Critical pattern from train_ddp.py**:
```python
# Wrong: identical data on all GPUs
loader = DataLoader(dataset, batch_size=128)

# Right: each GPU gets its shard
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, batch_size=128, sampler=sampler)
sampler.set_epoch(epoch)  # Reset shuffle each epoch!
```

### Gradient Synchronization Overhead

**Challenge**: Communication between GPUs slows training
- GPU-to-GPU via NVLink (Fastest ✓)
- GPU-to-GPU via PCIe (Fast)
- GPU-to-GPU via network (Slow ✗)

**Optimization techniques**:
- Gradient accumulation (reduce sync frequency)
- Gradient checkpointing (trade compute for memory)
- Overlapping communication with computation

### Scaling Laws

**Strong Scaling** (fixed problem Size)
```
Speedup = Original Time / Parallel Time
Ideal: Linear (2 GPUs = 2x faster)
Reality: Sub-linear due to communication overhead
```

**Weak Scaling** (problem size grows with GPU count)
```
Keeps per-GPU work constant
More realistic for production scenarios
```

### Kubernetes Networking for Training

**Key insight**: Distributed training needs stable pod addresses
- Regular Deployment: Pod names are random (breaks training)
- **StatefulSet**: Pod names are stable (pod-0, pod-1, ...) ✓

**Example from week3/07_statefulset.yaml**:
```yaml
StatefulSet train-job
├─ Pod: train-job-0 (MASTER, RANK=0)
├─ Pod: train-job-1 (RANK=1)
└─ Pod: train-job-2 (RANK=2)

DNS: train-job-0.headless-svc → 10.0.0.5 (stable)
```

---

## 🎯 Important Patterns & Gotchas

### Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| Forgetting `sampler.set_epoch(epoch)` | Identical shuffle each epoch | Add 1 line per epoch |
| Saving `model.state_dict()` in DDP | Only saves rank 0's replica | Use `model.module.state_dict()` |
| Using BatchNorm with small per-GPU batch | Statistics not representative | Use SyncBatchNorm |
| Not using DistributedSampler | Data duplication, wasted compute | Always use it with DDP |
| Not setting CUDA_VISIBLE_DEVICES | Multiple processes on same GPU | Let torchrun handle it |

### Performance Checkpoints

Track these metrics to understand scaling:
- **Samples/sec**: Throughput metric
- **Communication time**: % of epoch spent in all-reduce
- **GPU utilization**: nvidia-smi → should be >90%
- **Memory per GPU**: nvidia-smi → watch for memory leaks

---

## ❓ FAQ

### Q: Do I need a GPU to run this code?
**A**: Highly recommended for weeks 1-4. CPU-only training will be very slow. Options:
- Local NVIDIA GPU (Tesla T4, RTX series recommended)
- Google Colab (free limited GPU access)
- Cloud providers (AWS/GCP/Azure GPU instances)

### Q: Which week should I skip?
**A**: 
- **Skip Week 5** if not interested in TensorFlow
- **Skip Week 4** if not interested in service meshes (Istio)
- **Do Week 7** if you want consolidated examples

### Q: How do I know if I'm doing DDP correctly?
**A**: Check these signs:
- Training loss curve is identical to single-GPU baseline (same LR & batch size)
- GPU utilization >90% on all GPUs
- `nvidia-smi` shows the same model on each GPU
- Scaling efficiency >80% going from 2→4 GPUs

### Q: What's the difference between DDP and DistributedDataParallel?
**A**: They're the same! Full name is `torch.nn.parallel.DistributedDataParallel` (DDP).

### Q: How does NCCL differ from other backends?
**A**: NCCL is optimized for NVIDIA GPUs:
- **NCCL** (Week 4): GPU-native, fastest for GPU clusters
- **gloo**: CPU-friendly, works across heterogeneous hardware
- **MPI**: Traditional, works everywhere but slower

### Q: Can I run multi-node training locally?
**A**: Yes, using SSH or containers, but it's complex. Week 3 shows the patterns for cloud Kubernetes instead.

### Q: How do I debug DDP hangs?
**A**: Common causes:
1. Process didn't initialize correctly → Check `rank` and `world_size`
2. Processes on different code paths → Use conditional on `rank`
3. Collective operation mismatch → All ranks must call in same order

### Q: What's the deal with MASTER_ADDR and MASTER_PORT?
**A**: 
- **MASTER_ADDR**: Rank 0 node's hostname/IP (rank 0 coordinates initialization)
- **MASTER_PORT**: Any free port where rank 0 listens
- Set by `torchrun` automatically or manually in `launch_multinode.sh`

---

## 📚 Additional Resources

### Within This Repository
- Assignment PDFs in each week folder - Contains detailed problem sets
- Presentation slides (`.pptx` files) - Lecture materials
- Lab guides (`.pdf` files) - Step-by-step tutorials

### External References
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Lightning](https://www.pytorchlightning.ai/) - Production framework
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA CUDA Programming](https://developer.nvidia.com/cuda-learning-path)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 🤝 Contributing & Questions

This is an educational repository. For questions:
1. Check week-specific README or assignment PDFs
2. Review code comments (especially in Week 2's `train_ddp.py`)
3. Refer to external documentation linked above

---

## 📝 Key Files Reference

Quick lookup for important files across the repository:

### Training Scripts
- `week1/01_single_gpu_baseline/train_single.py` - Single GPU baseline
- `week2/02_ddp_single_node/train_ddp.py` - Single-node multi-GPU (DDP)
- `week3/03_ddp_multi_node/launch_multinode.sh` - Multi-node launcher
- `week8/Codes/01_lr_scaling_study/` - Learning rate scaling study

### Inference & Services
- `week1/01_inference_service/app.py` - FastAPI inference service
- `week4/04_istio/11_inference_v2.yaml` - Kubernetes inference deployment

### CUDA & GPU Programming
- `week2/week02_containers/03_cuda_programs/vector_add.cu` - CUDA kernel example
- `week2/week02_containers/03_cuda_programs/bandwidth_test.cu` - GPU bandwidth benchmark
- `week4/04_nccl_understanding/allreduce_benchmark.py` - NCCL communication benchmark

### Kubernetes Manifests
- `week3/03_networking_config/07_statefulset.yaml` - StatefulSet for distributed training
- `week3/03_networking_config/06_headless_svc.yaml` - Headless service for pod discovery
- `week4/04_istio/10_virtualservice.yaml` - Istio virtual service routing

### Docker Configuration
- `week1/01_inference_service/Dockerfile` - Inference service container
- `week2/week02_containers/01_minimal_gpu/Dockerfile` - Minimal GPU container
- `week2/week02_containers/02_jupyter_dev/docker-compose.yml` - Jupyter dev environment
- `week2/week02_containers/04_ci_pipeline/build-and-test.yml` - GitHub Actions CI/CD

---

## 🎓 Next Steps

1. ✅ Finished reading this guide
2. 👉 **Next**: Pick a week (start with Week 1) and read its README
3. 🧪 Clone/run the code examples
4. 📊 Modify parameters and observe changes
5. 🔬 Try extending examples with your own ideas

**Happy learning! 🚀**

---

*Last Updated: 2025 | IIT Guwahati Applied Accelerated AI Course*

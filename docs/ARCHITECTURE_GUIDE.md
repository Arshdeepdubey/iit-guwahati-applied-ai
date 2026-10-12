# Technical Architecture & Learning Flow

Deep dive into how components work together and the progression of concepts.

## 🏗️ Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLIED ACCELERATED AI COURSE                │
└─────────────────────────────────────────────────────────────────┘

WEEK 1: SINGLE GPU FOUNDATIONS
┌──────────────────────────────┐
│  GPU Hardware                │
│  ├─ Compute Cores            │
│  ├─ Memory Hierarchy          │
│  └─ Interconnect              │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  PyTorch Framework           │
│  ├─ Tensor Ops               │
│  ├─ Autograd                 │
│  └─ Model Training           │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  Training Script             │
│  ├─ Data Loading             │
│  ├─ Model Definition         │
│  ├─ Training Loop            │
│  └─ Evaluation               │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  Inference Service (FastAPI) │
│  ├─ Model Serving            │
│  ├─ HTTP Endpoints           │
│  └─ Health Checks            │
└──────────────────────────────┘

WEEK 2: SINGLE-NODE MULTI-GPU
┌──────────────────────────────┐
│  torchrun Launcher           │
│  ├─ Process Group Init       │
│  ├─ Rank Assignment          │
│  └─ GPU Mapping              │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  Distributed Data Parallel   │
│  ├─ Model Replica per GPU    │
│  ├─ Gradient Synchronization │
│  ├─ NCCL Backend             │
│  └─ All-Reduce Operations    │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  Containerization            │
│  ├─ Docker Image             │
│  ├─ CUDA Runtime             │
│  └─ Dependencies             │
└──────────────────────────────┘

WEEKS 3-4: MULTI-NODE & ORCHESTRATION
┌──────────────────────────────┐
│  Kubernetes Cluster          │
│  ├─ Master Node              │
│  ├─ Worker Nodes (w/ GPU)    │
│  └─ Networking               │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  StatefulSet Deployment      │
│  ├─ Pod 0 (Rank 0, Master)   │
│  ├─ Pod 1 (Rank 1)           │
│  └─ Pod N (Rank N)           │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  Service Mesh (Istio)        │
│  ├─ Virtual Services         │
│  ├─ Destination Rules        │
│  ├─ Traffic Management       │
│  └─ Observability            │
└──────────────────────────────┘
```

---

## 📊 Concept Progression Map

### Layer 1: GPU Computing Fundamentals
```
GPU Hardware
    ├─ CUDA Cores (Massive parallelism)
    ├─ Memory Types:
    │   ├─ Global Memory (Large, slow)
    │   ├─ Shared Memory (Fast, limited)
    │   ├─ L1/L2 Cache (Very fast)
    │   └─ Registers (Fastest, per-thread)
    ├─ Warp Execution (32 threads in lockstep)
    └─ Latency Hiding (Massive multithreading)
```

**Code Location**: `week2/week02_containers/03_cuda_programs/`

---

### Layer 2: Deep Learning Frameworks
```
PyTorch
    ├─ Computation Graph
    │   ├─ Define-by-run (Dynamic)
    │   └─ Forward → Backward (Autograd)
    ├─ Tensor Operations
    │   ├─ CPU operations
    │   └─ GPU operations (via CUDA)
    ├─ Optimizers
    │   └─ Parameter updates based on gradients
    └─ Models
        └─ Layers (Linear, Conv2d, etc.)
```

**Code Location**: `week1/01_single_gpu_baseline/train_single.py`

---

### Layer 3: Training at Scale
```
Training Loop
    ├─ Forward Pass
    │   └─ Input → Model → Loss
    ├─ Backward Pass
    │   └─ Loss → Gradients
    ├─ Synchronization (in DDP)
    │   └─ All-Reduce across GPUs
    └─ Optimizer Step
        └─ Update model parameters
```

**Code Location**: `week1/train_single.py` (single GPU)
**Code Location**: `week2/02_ddp_single_node/train_ddp.py` (multi-GPU)

---

### Layer 4: Distributed Training Architecture
```
Distributed Data Parallel (DDP)
    ├─ Setup Phase
    │   ├─ Rank assignment (which GPU am I?)
    │   ├─ Process group initialization
    │   └─ NCCL backend setup
    ├─ Training Phase
    │   ├─ Each GPU: forward + backward
    │   ├─ Gradient Communication
    │   │   └─ All-Reduce (NCCL) ← Synchronization point
    │   └─ Parameter Update (local)
    └─ Validation Phase
        └─ Only rank 0 saves checkpoints
```

**Key Pattern**:
```python
# Initialize
torch.distributed.init_process_group("nccl")
model = DDP(model.to(LOCAL_RANK), device_ids=[LOCAL_RANK])

# Train
for batch in dataloader:
    output = model(batch)              # Forward
    loss = criterion(output, targets)
    loss.backward()                    # Backward + All-Reduce (automatic via DDP)
    optimizer.step()                   # Update

# Save (only rank 0)
if rank == 0:
    torch.save(model.module.state_dict(), path)
```

**Code Location**: `week2/02_ddp_single_node/train_ddp.py`

---

### Layer 5: Communication Optimization
```
NCCL (NVIDIA Collective Communications Library)
    ├─ Collective Operations
    │   ├─ All-Reduce (Gradient synchronization)
    │   ├─ All-Gather (Combine data)
    │   └─ Broadcast (Send rank 0 to all)
    ├─ Optimization Levels
    │   ├─ Within GPU (NVLink) - Fastest
    │   ├─ Within Node (PCIe) - Fast
    │   ├─ Between Nodes (Network) - Slow
    │   └─ Ring topology for efficiency
    └─ Tuning Parameters
        ├─ NCCL_DEBUG (for debugging)
        ├─ NCCL_DEBUG_SUBSYS
        └─ NCCL algorithm selection
```

**Code Location**: `week4/04_nccl_understanding/allreduce_benchmark.py`
**Code Location**: `week4/04_nccl_understanding/nccl_allreduce.cu`

---

### Layer 6: Containerization
```
Docker for AI/ML
    ├─ Base Image
    │   └─ nvidia/cuda:11.x (includes CUDA runtime)
    ├─ Runtime Dependencies
    │   ├─ cuDNN (GPU acceleration library)
    │   ├─ PyTorch
    │   └─ Other ML libraries
    ├─ Application Code
    │   └─ Training or inference scripts
    ├─ Entrypoint
    │   └─ Default command to run
    └─ Best Practices
        ├─ Multi-stage builds
        ├─ Layer caching
        └─ Size optimization
```

**Code Location**: `week2/week02_containers/` (various Dockerfiles)

---

### Layer 7: Kubernetes Orchestration
```
Kubernetes for Distributed Training
    ├─ Cluster Architecture
    │   ├─ Control Plane (Master)
    │   │   └─ API Server, Scheduler, etcd
    │   └─ Worker Nodes (Compute)
    │       ├─ kubelet (Node agent)
    │       ├─ Container Runtime
    │       └─ kube-proxy
    ├─ Pod Networking
    │   ├─ Flat Pod network (all pods can reach each other)
    │   ├─ Service DNS (stable addresses)
    │   └─ CoreDNS (DNS resolution)
    ├─ StatefulSet (for distributed training)
    │   ├─ Stable pod names (pod-0, pod-1, ...)
    │   ├─ Stable storage
    │   └─ Ordered startup/shutdown
    ├─ ConfigMap & Secret
    │   ├─ Configuration (training hyperparams)
    │   └─ Secrets (credentials, API keys)
    └─ GPU Resources
        ├─ Resource requests
        ├─ Resource limits
        └─ Device plugin (nvidia-device-plugin)
```

**Code Location**: `week3/03_networking_config/` (all YAML files)

---

### Layer 8: Service Mesh (Advanced)
```
Istio Service Mesh
    ├─ Data Plane
    │   └─ Envoy Proxies (sidecar in each pod)
    ├─ Control Plane
    │   ├─ istiod (Central control)
    │   └─ Configuration distribution
    ├─ Traffic Management
    │   ├─ VirtualService (routing rules)
    │   ├─ DestinationRule (traffic policies)
    │   └─ Gateway (ingress/egress)
    ├─ Observability
    │   ├─ Metrics (Prometheus)
    │   ├─ Logs (Jaeger tracing)
    │   └─ Dashboards (Grafana)
    └─ Security
        ├─ mTLS (mutual authentication)
        └─ Authorization policies
```

**Code Location**: `week4/04_istio/` (Istio configuration)

---

## 🔄 Data Flow Through Training

### Single GPU (Week 1)
```
Input Data
    ↓
DataLoader (batching)
    ↓
Model.forward() [GPU]
    ↓
Loss Computation [GPU]
    ↓
Loss.backward() [GPU - gradient computation]
    ↓
Optimizer.step() [GPU - parameter update]
    ↓
Updated Model
```

**Where**: Single GPU, single process

---

### Multi-GPU Single Node (Week 2)
```
Input Data
    ↓
DistributedSampler (partition across GPUs)
    ├─ GPU 0: batch of 64 samples
    ├─ GPU 1: batch of 64 samples
    └─ GPU N: batch of 64 samples
    ↓
Model.forward() [Each GPU independent]
    ├─ GPU 0: forward on its 64 samples
    ├─ GPU 1: forward on its 64 samples
    └─ GPU N: forward on its 64 samples
    ↓
Loss Computation [Each GPU independent]
    ↓
Loss.backward() [Gradient computation + COMMUNICATION]
    ├─ GPU 0: computes gradients for its 64 samples
    ├─ GPU 1: computes gradients for its 64 samples
    └─ GPU N: computes gradients for its 64 samples
    ↓
    ‼️ ALL-REDUCE (NCCL synchronization)
    ├─ Gradients averaged across all GPUs
    └─ Result replicated to all GPUs
    ↓
Optimizer.step() [Each GPU applies same update]
    ├─ GPU 0: updates model with averaged gradients
    ├─ GPU 1: updates model with averaged gradients
    └─ GPU N: updates model with averaged gradients
    ↓
All GPUs have identical updated model
```

**Where**: Multi-GPU single node, multiple processes (torchrun)
**Communication**: Within-node via PCIe or NVLink

---

### Multi-Node (Week 3)
```
Node 1                              Node 2
├─ Pod 0 (RANK=0, Master)          ├─ Pod 2 (RANK=2)
│  ├─ GPU 0                        │  ├─ GPU 0
│  ├─ GPU 1                        │  ├─ GPU 1
│  └─ (2 processes)                │  └─ (2 processes)
│                                   │
└─ Pod 1 (RANK=1)                  └─ Pod 3 (RANK=3)
   ├─ GPU 0                           ├─ GPU 0
   ├─ GPU 1                           ├─ GPU 1
   └─ (2 processes)                   └─ (2 processes)

Training Data
    ↓
[Distributed] → Pod 0, Pod 1, Pod 2, Pod 3 (each gets shard)
    ↓
[Local training] → Each pod trains on its shard
    ↓
[Forward] → All processes compute forward locally
    ↓
[Backward] → All processes compute gradients locally
    ↓
‼️ [NETWORK ALL-REDUCE] (NCCL over TCP/IP)
   ├─ Pod 0 sends gradients to Pod 1, 2, 3
   ├─ Pod 1 sends gradients to Pod 0, 2, 3
   ├─ Pod 2 sends gradients to Pod 0, 1, 3
   └─ Pod 3 sends gradients to Pod 0, 1, 2
    ↓
[All pods receive averaged gradients]
    ↓
[Optimizer step] → Each pod updates model
    ↓
All pods have identical model
```

**Where**: Multiple nodes, potentially multiple GPUs per node
**Communication**: Between nodes via cluster network

---

## 🎓 Why Each Week Matters

### Week 1: Foundation
```
Why Single GPU?
├─ Establishes PyTorch baseline
├─ Enables debugging before adding distributed complexity
├─ Reference for performance comparison
└─ Validates data pipeline works

Key metric: Throughput (samples/sec)
Example: 500 samples/sec on single GPU
```

### Week 2: Scaling to Multiple GPUs
```
Why DDP?
├─ Enables ~N-GPU speedup (where N = number of GPUs)
├─ Data parallelism is simplest distribution strategy
├─ Sets up infrastructure for Week 3+ multi-node scaling
└─ Teaches gradient synchronization

Key metric: Scaling efficiency = (single_GPU_throughput * N) / DDP_throughput
Example: 2 GPUs @ 90% efficiency = 1.8x speedup (not perfect 2x)
```

### Week 3: Scaling to Hundreds of GPUs
```
Why Kubernetes?
├─ Manages resource allocation automatically
├─ Enables fault tolerance (pod restart)
├─ Simplifies deployment (no SSH scripting)
├─ Enables monitoring and logging at scale
└─ Prerequisite for production

Key metric: Availability (uptime %) and Reliability (fault recovery)
```

### Week 4: Production-Grade Networking
```
Why Istio + NCCL Optimization?
├─ Monitors inter-pod communication
├─ Enables gradual rollout (canary deployments)
├─ Provides security (mTLS)
├─ Tunes communication paths for performance
└─ Observability for debugging large clusters

Key metric: Communication time as % of training time
Target: <20% communication overhead
```

---

## 💾 Storage & Data Patterns

### Local Development (Week 1-2)
```
Disk
├─ /tmp/cifar/ (downloaded dataset)
├─ ./checkpoints/ (model saves)
└─ ./logs/ (training logs)

All local, easy to manage
```

### Kubernetes (Week 3+)
```
Node Local Volume (ephemeral)
├─ Pod gets empty storage
├─ Downloaded during container startup
├─ Lost when pod terminates

Persistent Volume (recommended for production)
├─ GCS / S3 / NFS mounted
├─ Shared across all pods
├─ Survives pod termination

Pattern:
1. Pod downloads dataset from persistent storage
2. Trains locally in /tmp/
3. Saves checkpoints to persistent storage
4. Pod terminates without data loss
```

---

## 🔌 Integration Points

### Data Flow
```
Data Source (CIFAR-10, ImageNet, custom)
    ↓
PyTorch DataLoader/DistributedSampler
    ↓
Model Training (single/multi-GPU/multi-node)
    ↓
Model Checkpoints (disk/cloud storage)
    ↓
Inference Service (serving predictions)
```

### Model Flow
```
Model Architecture Definition (resnet18, custom)
    ↓
Model Training (Week 1-3)
    ↓
Model Optimization (Week 4-8)
    ├─ Quantization
    ├─ Pruning
    ├─ Knowledge Distillation
    └─ Other techniques
    ↓
Model Serialization (torch.save)
    ↓
Model Deployment (Kubernetes + Istio)
    ↓
Model Serving (FastAPI inference)
```

### Monitoring Flow
```
Training/Inference Running
    ↓
Metrics Generated
    ├─ nvidia-smi (GPU stats)
    ├─ PyTorch profiler (time breakdown)
    └─ Custom metrics (throughput, loss)
    ↓
Metrics Collected (Prometheus, CloudLogging)
    ↓
Dashboards (Grafana, Cloud Console)
    ↓
Alerts triggered if anomalies
```

---

## 🔀 Decision Trees

### "Which GPU should I use?"
```
Do I have local GPU?
├─ YES → Use it (Week 1-3)
└─ NO
    ├─ Can I use cloud credits? → Use cloud GPU (GCP/AWS/Azure)
    ├─ Can I use Google Colab? → Use Colab (free, limited)
    └─ Can I only use CPU? → Use CPU (very slow, great for debugging)
```

### "Should I use DDP or multi-processing?"
```
Do I have 2+ GPUs on one machine?
├─ YES → Use DDP with torchrun (Week 2)
└─ NO
    ├─ Do I have 2+ machines? → Use DDP + multi-node (Week 3)
    ├─ Do I have 1 GPU? → Use single-GPU (Week 1)
    └─ Do I have only CPU? → Use PyTorch with CPU (slow)
```

### "Should I use PyTorch Lightning or raw PyTorch?"
```
This course uses:
├─ Raw PyTorch (Week 1-4) → Learn fundamentals
└─ Production code uses Lightning/frameworks

Recommendation:
├─ Weeks 1-4: Raw PyTorch (understand distributed patterns)
├─ Production: Use PyTorch Lightning or similar
└─ This teaches you what Lightning hides!
```

---

## 🚦 Performance Benchmarking Checklist

After each week, benchmark your code:

### Week 1 (Single GPU)
```
□ Throughput (samples/sec)
□ GPU memory usage (nvidia-smi)
□ GPU utilization (should be >90%)
□ Training time per epoch
□ Loss curve (sanity check)
```

### Week 2 (Multi-GPU)
```
□ Throughput with 2 GPUs
□ Scaling efficiency (target: >85%)
□ All-reduce time (nvidia-smi → Communication)
□ Gradient synchronization overhead
□ Per-GPU memory (should be ~same as single GPU)
□ Verify loss curve matches Week 1
```

### Week 3 (Multi-Node)
```
□ Throughput with N nodes
□ Scaling efficiency (target: >80% for inter-node)
□ Network bandwidth utilization
□ Communication time (bottleneck analysis)
□ Fault recovery time (kill pod, measure restart)
□ Checkpoint save/load time
```

### Week 4 (Service Mesh)
```
□ Throughput (should match Week 3)
□ Istio overhead (<5% acceptable)
□ Latency p50, p95, p99
□ Canary deployment switching time
□ mTLS handshake overhead
```

---

## 🎯 Success Criteria by Phase

### Phase 1: Local Development (Weeks 1-2)
```
✓ Code runs without errors
✓ Training loss decreases
✓ GPU utilization >90%
✓ Throughput reproducible
✓ Multi-GPU code produces same loss as single GPU
```

### Phase 2: Cluster Deployment (Weeks 3-4)
```
✓ Training runs on Kubernetes
✓ Pods communicate correctly
✓ Training loss curve matches local results
✓ Scaling efficiency >80%
✓ Pod failures trigger automatic recovery
```

### Phase 3: Production (Weeks 5-8)
```
✓ Inference service serving requests
✓ Monitoring alerts working
✓ Performance meets SLA
✓ Safe canary deployments possible
✓ Data pipeline scalable to production volume
```

---

*Use this guide to understand how concepts connect. Refer to NEW_JOINER_GUIDE.md for implementation details.*

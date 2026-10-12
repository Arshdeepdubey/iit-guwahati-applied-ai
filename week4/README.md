# Week 4: Communication & Optimization with Istio and NCCL

## 🎯 Learning Objectives

By the end of Week 4, you will understand and be able to:
- ✅ Profile NCCL all-reduce bandwidth across different message sizes
- ✅ Analyze collective communication patterns (ring vs tree algorithms)
- ✅ Measure communication overhead and identify bottlenecks
- ✅ Configure NCCL optimizations for your hardware
- ✅ Understand service mesh networking (Istio) for ML systems
- ✅ Monitor and visualize distributed communication patterns
- ✅ Optimize bandwidth utilization for large-scale training

---

## 📚 Key Concepts

### NCCL Collective Operations
- **All-Reduce**: Gradient synchronization (most common in DDP)
- **All-Gather**: Collect tensors from all ranks
- **Broadcast**: Send from one rank to all
- **Reduce-Scatter**: Inverse of all-gather
- **Ring Algorithm**: O(N) time, better for bandwidth saturation
- **Tree Algorithm**: O(log N) time, better for latency

### Bandwidth Analysis
- **Per-Rank Bandwidth**: Bytes transferred per rank per operation
- **Ring All-Reduce**: Each rank sends/receives ~2*(N-1)/N * message_size
- **Scaling Laws**: Why adding GPUs doesn't always proportionally speed up
- **Communication-Computation Overlap**: How to hide communication with computation

### Istio Service Mesh
- **Traffic Management**: Load balancing for inference services
- **Observability**: Metrics, logs, traces for debugging
- **Security**: mTLS for encrypted communication
- **Resilience**: Retries, timeouts, circuit breakers

### GPU Communication Details
- **NVLink**: Direct GPU-to-GPU (very fast)
- **PCIe**: GPU-to-CPU-to-GPU (slower)
- **Ethernet**: Node-to-node (slowest)
- **InfiniBand**: Specialized for HPC (faster than Ethernet)

---

## 📁 Folder Structure

```
week4/
├── README.md                          # This file
├── 04_nccl_understanding/             # NCCL benchmarking
│   ├── allreduce_benchmark.py        # Measure bandwidth at different sizes
│   └── nccl_analysis.py              # Analyze performance patterns
├── 04_istio/                          # Service mesh examples
│   ├── deployment.yaml               # Kubernetes deployment
│   ├── virtual-service.yaml          # Istio traffic management
│   └── monitoring.yaml               # Prometheus metrics collection
└── pdfs/                              # Performance analysis docs
```

---

## 🔧 Setup & Requirements

### Benchmarking Requirements
- Multiple GPUs on single or multiple machines
- PyTorch with NCCL support
- NCCL libraries (included in PyTorch CUDA build)

### Istio Requirements (optional, for service mesh)
- Kubernetes cluster
- Istio service mesh installed
- Prometheus for metrics collection

---

## 🚀 How to Run

### 1. NCCL All-Reduce Bandwidth Benchmarking

**Single Machine (2 GPUs):**
```bash
cd 04_nccl_understanding

torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py
```

**Expected output:**
```
Rank 0: benchmarking all-reduce
Message Size | Bandwidth (GB/s) | Effective Time (ms)
1 KB         | 0.5             | 0.002
1 MB         | 42.0            | 0.024
10 MB        | 45.5            | 0.220
100 MB       | 47.2            | 2.12
1 GB         | 47.3            | 21.2
```

**Key observations:**
- Bandwidth increases with message size (until saturation)
- Per-rank bandwidth saturates near interconnect ceiling
- Multiple GPUs: ring all-reduce keeps bandwidth ~constant

**Multi-Machine Benchmark (8 GPUs across 4 machines):**
```bash
# On each machine, same command
torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py

# Compare output with Week 3 (multi-node)
# Expect: network-limited, lower bandwidth than single-machine
```

### 2. Analyzing Communication Patterns

**Profile during training:**
```bash
# Enable detailed NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

cd ../week2/02_ddp_single_node
torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 1
```

**Parse and analyze logs:**
```bash
python 04_nccl_understanding/nccl_analysis.py --log-file nccl_debug.log
```

**Expected analysis:**
```
All-Reduce Statistics:
  Total calls: 391 (per epoch for CIFAR-10)
  Avg time: 1.24ms
  Min time: 1.01ms
  Max time: 2.34ms
  Total communication time: 485ms (12% of epoch)

Bandwidth utilization:
  Average: 45.3 GB/s (of 50 GB/s available)
  Efficiency: 90%
```

### 3. Running on Different Hardware

**CPU-side bandwidth (slower):**
```bash
# Compare GPU-to-GPU vs CPU-facilitated
export NCCL_DEBUG=INFO NCCL_ALGO=Ring  # Force ring algorithm
torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py
```

**With NVLink (very fast on A100/H100):**
```bash
# Output should show ~800 GB/s for NVLink-connected GPUs
torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py
```

**With InfiniBand (on HPC clusters):**
```bash
# Requires NCCL compiled with InfiniBand support
NCCL_IB_HCA=mlx5_0 torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py
```

### 4. Istio Service Mesh Deployment (Optional)

**Install Istio on Kubernetes:**
```bash
istioctl install --set profile=demo -y
```

**Deploy inference service with Istio:**
```bash
cd 04_istio

# Create namespace with sidecar injection
kubectl create namespace ml-inference
kubectl label namespace ml-inference istio-injection=enabled

# Deploy service
kubectl apply -f deployment.yaml
kubectl apply -f virtual-service.yaml

# Verify
kubectl get pods -n ml-inference
```

**Monitor traffic:**
```bash
# Port-forward Prometheus
kubectl port-forward -n istio-system svc/prometheus 9090:9090

# Visit http://localhost:9090
# Query: istio_request_duration_milliseconds_bucket
```

---

## 📊 Understanding Results

### All-Reduce Bandwidth Scaling Law

**Expected pattern:**
```
Message Size | Bandwidth Growth
1 KB        | 0.5 GB/s (latency dominated)
10 KB       | 2.0 GB/s
100 KB      | 8.0 GB/s
1 MB        | 25.0 GB/s
10 MB       | 42.0 GB/s (approaching saturation)
100 MB      | 47.0 GB/s (saturated)
1 GB        | 47.2 GB/s (saturated, limited by network)
```

**Why saturation occurs:**
- Latency is amortized over larger messages
- Bandwidth-limited by interconnect ceiling
- Can't send faster than physical hardware allows

### Communication vs Computation Ratio

**For typical DDP training:**
```
Computation Time (forward + backward): 45ms
All-Reduce Time (gradient synchronization): 1.5ms
Ratio: 45:1.5 = 30:1 (computation dominated)

Conclusion: Communication overhead is small (~3% of epoch time)
```

**If ratio is < 5:1:**
- Communication bottleneck (small models or models with large layers)
- Solution: Gradient accumulation, compression, or async communication

### Ring vs Tree Algorithm Performance

**From NCCL analysis:**
```
Ring Algorithm:
  - More stages: 2*(N-1) phases
  - Better bandwidth utilization
  - Higher latency per phase
  - Best for: large messages, high-bandwidth networks

Tree Algorithm:
  - Fewer stages: log(N) phases
  - Lower per-phase latency
  - Less bandwidth per phase
  - Best for: small messages, high-latency networks
```

### Istio Metrics

**Key metrics to monitor:**
```
istio_request_total: Total requests (by destination service)
istio_request_duration: Request latency (p50, p95, p99)
istio_requests_total{response_code="5xx"}: Error rate
```

**Example query (Prometheus):**
```
rate(istio_request_total[1m])  # Throughput
histogram_quantile(0.95, istio_request_duration)  # P95 latency
```

---

## 🔄 How to Compare Results

### Benchmark Comparison Across Hardware

| Hardware | Bandwidth | Notes |
|----------|-----------|-------|
| V100 (PCIe) | 30 GB/s | GPU-to-GPU via PCIe |
| A100 (PCIe) | 40 GB/s | Faster PCIe Gen4 |
| A100 (NVLink) | 400 GB/s | Direct GPU link |
| H100 (NVLink) | 900 GB/s | Newest architecture |
| Ethernet (1 Gbps) | 0.125 GB/s | Slow network |
| Ethernet (100 Gbps) | 12.5 GB/s | Typical data center |
| InfiniBand (200 Gbps) | 25 GB/s | HPC interconnect |

### Training Scaling Efficiency

**With communication profiling:**
```
Ideal scaling: Adding 2x GPUs → 2x speedup
Real scaling (A100 PCIe): Adding 2x GPUs → 1.9x speedup (95% efficiency)
Real scaling (Ethernet): Adding 2x GPUs → 1.7x speedup (85% efficiency)
Real scaling (8 nodes): → 6.3x speedup (78% efficiency)
```

**Efficiency formula:**
```
Efficiency = Actual Speedup / (Number of GPUs)
           = (Time_single / Time_multi) / N_gpus
```

---

## 🧪 Hands-On Experiments

### Experiment 1: Message Size Sensitivity
```bash
# Modify allreduce_benchmark.py to test these sizes:
sizes = [1, 10, 100, 1000, 10000, 100000]  # KB

# Plot: Message Size vs Bandwidth
# Observe: Where does saturation occur?
```

### Experiment 2: Number of GPUs Impact
```bash
# Run with 2, 4, 8 GPUs and measure:
# - Per-rank bandwidth (should stay constant)
# - Total time (should stay roughly constant for ring all-reduce)

for n_gpu in 2 4 8; do
    torchrun --nproc_per_node=$n_gpu allreduce_benchmark.py
done
```

### Experiment 3: Algorithm Comparison
```bash
# Force different NCCL algorithms and benchmark
for algo in Ring Tree; do
    export NCCL_ALGO=$algo
    torchrun --nproc_per_node=2 allreduce_benchmark.py
done
```

### Experiment 4: Gradient Accumulation
```bash
# Modify train_ddp.py to accumulate gradients before all-reduce:
# - Larger message sizes → better bandwidth utilization
# - Fewer synchronization points → faster training

# Compare: Gradient accumulation=1 vs 4
```

---

## 🐛 Troubleshooting

### Issue: "Low bandwidth" (< 10 GB/s)
**Potential causes:**
1. Using CPU interconnect instead of direct GPU path
2. Oversubscribed network
3. NCCL algorithm mismatch with hardware

**Diagnosis:**
```bash
NCCL_DEBUG=INFO python benchmark.py 2>&1 | grep "selected"
# Should show NCCL selecting appropriate algorithm
```

### Issue: Istio pod not starting
**Check logs:**
```bash
kubectl logs <pod> -c istio-proxy
kubectl describe pod <pod>
```

### Issue: "NCCL WARN / unhandled cuda error"
**Cause**: CUDA/NCCL version mismatch

**Solution:**
```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.nccl.version())"
# Reinstall with matching versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Resources

### NCCL & Communication
- [NCCL Algorithm Details](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/tuning.html)
- [Ring All-Reduce Algorithm](https://arxiv.org/abs/1711.00325)
- [NVIDIA GPU Communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/)

### Istio
- [Istio Documentation](https://istio.io/latest/docs/)
- [Service Mesh for ML](https://istio.io/latest/docs/setup/install/)

### Performance Analysis
- [PyTorch Profiling](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsys Profiling](https://developer.nvidia.com/nsys)

---

## ✅ Verification Checklist

Before moving to Week 5, verify:
- [ ] You can run all-reduce benchmarks on your hardware
- [ ] Bandwidth measurements are as expected
- [ ] You understand why bandwidth saturates at certain message sizes
- [ ] Ring vs tree algorithm trade-offs are clear
- [ ] Communication time is << computation time in your setup
- [ ] NCCL logs can be parsed and analyzed

---

## 🎓 Concepts for Next Week

In **[Week 5: TensorFlow Acceleration](../week5/README.md)**, you'll learn:
- TensorFlow distributed training (alternative to PyTorch)
- tf.distribute strategies (MirroredStrategy, TPUStrategy)
- Comparing PyTorch vs TensorFlow performance
- When to choose one framework over the other

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 3: Multi-Node Distributed Training](../week3/README.md)  
**Next Week**: [Week 5: TensorFlow Acceleration →](../week5/README.md)

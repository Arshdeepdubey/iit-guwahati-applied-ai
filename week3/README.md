# Week 3: Multi-Node Distributed Training

## 🎯 Learning Objectives

By the end of Week 3, you will understand and be able to:
- ✅ Scale training from single machine (Week 2) to multiple machines
- ✅ Configure network topology for distributed training
- ✅ Use elasticity to handle node failures gracefully
- ✅ Deploy distributed training on cloud infrastructure
- ✅ Monitor and debug multi-node communication
- ✅ Optimize bandwidth utilization across network links
- ✅ Handle clock synchronization and stragglers

---

## 📚 Key Concepts

### Network Topology
- **Bandwidth vs Latency**: How network interconnect affects all-reduce time
- **Collective Communication Patterns**: Ring vs tree for different message sizes
- **Bandwidth Saturation**: Point where adding more nodes doesn't help
- **Heterogeneous Networks**: Mixed network speeds (Ethernet + InfiniBand)

### Multi-Node DDP Setup
- **Process Group Initialization**: Discovery across multiple hosts
- **MASTER_ADDR/PORT**: How nodes find the coordinator
- **RANK assignment**: Global rank across all nodes
- **Local Rank vs Global Rank**: GPU mapping on multi-node

### Network Protocols
- **Ethernet (TCP)**: Standard, lower bandwidth (~100 Gbps)
- **InfiniBand (IB)**: High-speed, optimized for ML (~200 Gbps+)
- **NCCL over different transports**: Automatic selection
- **gRPC/RDMA**: Alternative communication layers

### Cloud Deployment
- **Kubernetes**: Orchestrating distributed training jobs
- **Auto-scaling**: Adding/removing nodes during training
- **Job Scheduling**: Ensuring efficient resource allocation
- **Spot Instances**: Cost optimization with interruptible compute

---

## 📁 Folder Structure

```
week3/
├── README.md                          # This file
├── 03_ddp_multi_node/                 # Multi-node training
│   ├── train_multi_node.py           # Updated DDP for multi-node
│   └── launch_multi_node.sh          # Launcher script for multiple hosts
├── 03_networking_config/              # Network optimization
│   ├── network_benchmark.py          # Measure bandwidth/latency
│   └── topology_analysis.py          # Analyze communication patterns
└── pdfs/                              # Theory and architecture docs
```

### What's in Each Folder?

**`03_ddp_multi_node/`**
- `train_multi_node.py`: Extends Week 2's train_ddp.py
  - **Changes**: Uses environment variables for multi-node coordination
  - **RANK**: Now across all machines globally
  - **Launch**: Via SSH or Kubernetes, not torchrun

**`03_networking_config/`**
- `network_benchmark.py`: Measures bandwidth at different message sizes
- `topology_analysis.py`: Visualizes which nodes communicate fastest

---

## 🔧 Setup & Requirements

### System Prerequisites
- Multiple machines with NVIDIA GPUs
- SSH access between all machines
- Shared filesystem (NFS) or object storage (S3, GCS)
- Synchronized clocks (NTP)
- Open firewall ports for NCCL communication

### Network Configuration

**For 2 machines with 1 GPU each:**
```bash
# Machine 0 (MASTER)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0

# Machine 1
export MASTER_ADDR=192.168.1.100  # Same as Machine 0
export MASTER_PORT=29500          # Same as Machine 0
export RANK=1
export WORLD_SIZE=2
export LOCAL_RANK=0
```

### Verify Network Connectivity
```bash
# From each machine, test network
ping $MASTER_ADDR
python -c "import socket; print(socket.gethostbyname('$(hostname)'))"

# Test NCCL communications (requires NCCL debugging enabled)
NCCL_DEBUG=INFO python -c "import torch; print(torch.cuda.nccl.version())"
```

---

## 🚀 How to Run

### 1. Multi-Node Training with SSH

**Setup (run once):**
```bash
# Create a hosts file
cat > hosts.txt << EOF
192.168.1.100 # Machine 0 (has 2 GPUs)
192.168.1.101 # Machine 1 (has 2 GPUs)
EOF

# Copy training script to all machines
for host in $(cat hosts.txt | grep -v '#'); do
    scp -r week3/03_ddp_multi_node user@$host:~/
done
```

**Launch training:**
```bash
# Simple: torchrun with --nnodes and --node_rank
# Machine 0 (primary):
torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_multi_node.py --epochs 3

# Machine 1 (secondary):
torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train_multi_node.py --epochs 3
```

**Expected output (both machines):**
```
[Machine 0] Rank 0 | GPU 0
[Machine 1] Rank 2 | GPU 0
[Machine 0] Rank 1 | GPU 1
[Machine 1] Rank 3 | GPU 1
All-reduce sync: 15.2ms (higher than single-node due to network)
```

### 2. Using a Launcher Script

**`launch_multi_node.sh` example:**
```bash
#!/bin/bash
NODES=(192.168.1.100 192.168.1.101)
NNODES=${#NODES[@]}

for i in "${!NODES[@]}"; do
    host=${NODES[$i]}
    if [ $i -eq 0 ]; then
        # Primary node
        ssh $host "cd ~/03_ddp_multi_node && \
            torchrun --nproc_per_node=2 --nnodes=$NNODES \
            --node_rank=$i --master_addr=${NODES[0]} \
            --master_port=29500 train_multi_node.py --epochs 3" &
    else
        # Secondary nodes
        ssh $host "cd ~/03_ddp_multi_node && \
            torchrun --nproc_per_node=2 --nnodes=$NNODES \
            --node_rank=$i --master_addr=${NODES[0]} \
            --master_port=29500 train_multi_node.py --epochs 3" &
    fi
done
wait
```

### 3. Network Benchmarking

**Measure bandwidth between nodes:**
```bash
cd 03_networking_config

# Run on two machines
python network_benchmark.py --master-addr 192.168.1.100 --rank 0
python network_benchmark.py --master-addr 192.168.1.100 --rank 1
```

**Expected output:**
```
Message Size | Bandwidth (GB/s) | Latency (us)
1 KB         | 0.5             | 150
1 MB         | 45.0            | 25
100 MB       | 47.3            | 2100
1 GB         | 47.5            | 21000
```

### 4. Kubernetes Deployment

**Create a training job manifest:**
```yaml
# distributed-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ddp-training-job
spec:
  parallelism: 2
  completions: 2
  template:
    spec:
      containers:
      - name: training
        image: ml-training:latest
        resources:
          limits:
            nvidia.com/gpu: 2
        env:
        - name: MASTER_ADDR
          value: "ddp-training-job-0"
        - name: WORLD_SIZE
          value: "4"
        command:
        - torchrun
        - --nproc_per_node=2
        - --nnodes=2
        - --node_rank=$(POD_ORDINAL)
        - train_multi_node.py
      restartPolicy: OnFailure
```

**Deploy:**
```bash
kubectl apply -f distributed-training-job.yaml

# Monitor
kubectl logs -f job/ddp-training-job
```

---

## 📊 Understanding Results

### Multi-Node Communication Overhead

**Comparison across topologies:**

| Setup | All-Reduce Time | Speedup | Efficiency |
|-------|--|--|--|
| 1 machine, 2 GPUs | 1.2ms | 1.87x | 93% |
| 2 machines, 2 GPUs each | 15.2ms | 3.5x | 87% |
| 4 machines, 2 GPUs each | 32.1ms | 6.3x | 78% |
| 8 machines, 2 GPUs each | 61.5ms | 11.2x | 70% |

**Key observation**: Communication time grows with machine count, reducing efficiency.

### Network Bandwidth Analysis

**From network_benchmark.py output:**
```
Small messages (< 1KB):
  - Latency dominated: ~100-200 microseconds
  - Bandwidth: low (~0.5-5 GB/s)
  - Use: Minimal for training (only scalar aggregations)

Medium messages (1 MB - 100 MB):
  - Bandwidth dominated: ~40-50 GB/s typical
  - Latency: 25-200 microseconds
  - Use: Most gradient all-reduces (typical layer sizes)

Large messages (> 100 MB):
  - Bandwidth saturated: close to network ceiling
  - Latency: proportional to message size
  - Use: Rare in modern architectures (large feature maps)
```

### Expected Convergence Across Nodes

**With proper setup:**
```
Epoch 1: Loss 2.30 | Acc: 25%
Epoch 2: Loss 1.80 | Acc: 40%
Epoch 3: Loss 1.50 | Acc: 50%
```

Should be **identical** to single-node training.

⚠️ **If different**: 
- Check if all nodes have same seed
- Verify synchronized clock (NTP)
- Check for dropped packets (network issue)

---

## 🔄 How to Compare Results

### Single-Node vs Multi-Node

**Run same training on:**
1. 1 machine with 4 GPUs (Week 2)
2. 2 machines with 2 GPUs each (Week 3)
3. Compare accuracy curves

**Expected:**
- Final accuracy: Identical (within ±0.5%)
- Training curves: Same shape
- Training time: ~2x faster (ignoring communication overhead)

### Bandwidth vs Computation Time

**Analyze bottleneck:**
```
Total training time = computation_time + communication_time

Communication-bound: increasing message batch size speeds up
Computation-bound: increasing batch size doesn't help
```

**Profile to check:**
```bash
NCCL_DEBUG=INFO python train_multi_node.py --epochs 1
# Look for: "All-reduce time" vs "Compute time" in logs
```

---

## 🧪 Hands-On Experiments

### Experiment 1: Network Topology Comparison
```bash
# Star topology (all nodes connect to master)
# vs Ring topology (N->N+1->...->N%size)
# Measure all-reduce time for each
python topology_analysis.py --topology star
python topology_analysis.py --topology ring
```

### Experiment 2: Gradual Scaling
```bash
# Run with increasing machine count
# 1 machine, 2 GPUs
# 2 machines, 2 GPUs each
# 4 machines, 2 GPUs each
# Plot: Speedup vs number of machines
```

### Experiment 3: Network Latency Injection
```bash
# Simulate high-latency network (e.g., cloud with latency)
# Use Linux tc (traffic control) to add delay
tc qdisc add dev eth0 root netem delay 10ms

# Re-run training and measure impact on convergence time
```

### Experiment 4: Fault Tolerance
```bash
# Kill a node mid-training (simulate failure)
# With checkpointing: training should resume
# Without: entire job fails

# Compare: checkpoint overhead vs ability to resume
```

---

## 🐛 Troubleshooting

### Issue: "Failed to initialize NCCL" or "Connection timeout"
**Causes:**
1. Firewall blocking NCCL port
2. Nodes can't resolve MASTER_ADDR
3. Different NCCL versions on nodes

**Solutions:**
```bash
# Test connectivity
telnet $MASTER_ADDR 29500

# Check NCCL library
python -c "import torch; print(torch.cuda.nccl.version())"

# Enable debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Issue: "Rank 0 and Rank 1 have different RANK values"
**Cause**: Inconsistent environment variables across nodes

**Solution:**
```bash
# Verify on each node
python -c "import os; print(os.environ['RANK'], os.environ['WORLD_SIZE'])"

# Should match the launch script configuration
```

### Issue: Training hangs (all ranks waiting)
**Probable cause**: One node's all-reduce is stuck

**Diagnosis:**
```bash
# Kill one process and observe others' behavior
# If others continue: network partition detected
# If all hang: synchronization deadlock

# Typical fix: check for NaN/Inf in gradients (causes NCCL hang)
```

### Issue: Slow all-reduce (>50ms)
**Causes:**
1. Congested network
2. Suboptimal NCCL settings
3. Too many machines for network topology

**Optimization:**
```bash
# Gradient compression (reduce communication volume)
# Ring all-reduce optimization (default in PyTorch)
# Use faster interconnect (InfiniBand if available)

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
```

---

## 📚 Resources

### Network & Communication
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- [PyTorch DDP Advanced](https://pytorch.org/docs/stable/distributed.html)
- [Ring All-Reduce Paper](https://arxiv.org/abs/1711.00325)

### Cloud Deployment
- [Kubernetes ML Training](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [AWS SageMaker Distributed Training](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
- [Google Cloud TPU Multi-Host](https://cloud.google.com/tpu/docs/distributed-training)

### Distributed Systems
- [Horovod: Another framework](https://horovod.ai/)
- [Fault Tolerance in Distributed ML](https://arxiv.org/abs/1604.00981)

---

## ✅ Verification Checklist

Before moving to Week 4, verify:
- [ ] You can configure MASTER_ADDR/PORT correctly
- [ ] Multi-node training starts without connection errors
- [ ] RANK assignment is correct across nodes
- [ ] Convergence is identical to single-node training
- [ ] Network bandwidth meets expectations
- [ ] All-reduce synchronization works without hangs
- [ ] You understand network topology impact

---

## 🎓 Concepts for Next Week

In **[Week 4: Communication & Optimization](../week4/README.md)**, you'll learn:
- Detailed NCCL benchmarking and profiling
- Collective operation optimization (all-reduce, all-gather, broadcast)
- Ring vs tree topologies and when to use each
- Service mesh networking (Istio) for production ML systems

**Preview**: Same DDP code, but with deep profiling to understand where time is spent and how to optimize it.

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 2: Single-Node Multi-GPU (DDP)](../week2/README.md)  
**Next Week**: [Week 4: Communication & Optimization →](../week4/README.md)

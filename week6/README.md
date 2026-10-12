# Week 6: Scaling Benchmarks & Exploratory Data Analysis

## 🎯 Learning Objectives

By the end of Week 6, you will understand and be able to:
- ✅ Design and run scaling benchmarks for distributed training
- ✅ Measure strong scaling and weak scaling properties
- ✅ Analyze batch size and learning rate effects
- ✅ Profile performance bottlenecks
- ✅ Perform exploratory data analysis on training metrics
- ✅ Visualize scaling laws and performance trends
- ✅ Identify optimal configurations for your hardware

---

## 📚 Key Concepts

### Scaling Benchmarks
- **Strong Scaling**: Fixed problem size, increasing compute resources
- **Weak Scaling**: Increasing problem size proportionally with resources
- **Speedup Formula**: T₁ / Tₙ (where n is number of GPUs)
- **Efficiency Formula**: Speedup / n (should be close to 1 for good scaling)
- **Amdahl's Law**: Upper bound on speedup from parallelization

### Batch Size Effects
- **Larger batches**: Faster training, but may hurt generalization
- **Learning rate scaling**: Must scale with effective batch size
- **Gradient noise**: Affects convergence, depends on batch size
- **Optimal batch size**: Often 32-256 per GPU for vision tasks

### Performance Analysis
- **Memory profiling**: Peak memory usage, memory efficiency
- **Throughput measurement**: Samples per second, batches per second
- **Communication overhead**: Percentage of time spent synchronizing
- **Utilization analysis**: GPU utilization, memory utilization

### Exploratory Data Analysis (EDA)
- **Loss curve analysis**: Convergence rate, final loss
- **Accuracy progression**: How accuracy improves over epochs
- **Correlation analysis**: Batch size vs learning rate effects
- **Visualization**: Plots showing scaling trends

---

## 📁 Folder Structure

```
week6/
├── README.md                          # This file
├── 06_scaling_benchmark/              # Scaling measurement tools
│   └── benchmark_scaling.py           # Run benchmarks at different scales
├── Code/                              # Supporting code
│   └── 01_inference_service/          # Inference service (from Week 1)
└── pdfs/                              # Detailed analysis and results
```

---

## 🔧 Setup & Requirements

### Prerequisites
- Completed weeks 1-5 material
- Multiple GPUs (preferably 2, 4, or 8 for meaningful scaling)
- Pandas and Matplotlib for analysis

### Install Analysis Tools
```bash
pip install pandas matplotlib seaborn numpy
pip install tensorboard  # Optional, for interactive analysis
```

---

## 🚀 How to Run

### 1. Run Scaling Benchmarks

```bash
cd 06_scaling_benchmark

# Benchmark with increasing GPU counts
for n_gpu in 1 2 4 8; do
    echo "=== Benchmarking with $n_gpu GPUs ==="
    torchrun --nproc_per_node=$n_gpu benchmark_scaling.py \
        --epochs 3 \
        --per-gpu-batch-size 128 \
        --output results_${n_gpu}gpu.csv
done
```

**Expected CSV output:**
```
epoch,gpu_count,batch_size,loss,accuracy,throughput_samples_sec,time_sec
1,1,128,2.30,25.5,1200,42.5
1,2,256,2.31,24.8,2180,46.2
1,4,512,2.29,25.2,3850,50.1
1,8,1024,2.28,25.5,6200,52.3
```

### 2. Analyze Scaling Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = {
    1: pd.read_csv('results_1gpu.csv'),
    2: pd.read_csv('results_2gpu.csv'),
    4: pd.read_csv('results_4gpu.csv'),
    8: pd.read_csv('results_8gpu.csv'),
}

# Calculate speedup
for n_gpu, df in results.items():
    speedup = results[1]['throughput_samples_sec'].mean() / df['throughput_samples_sec'].mean() * n_gpu
    print(f"{n_gpu} GPUs: {speedup:.2f}x speedup")

# Plot scaling curve
gpu_counts = list(results.keys())
speedups = [...]  # Calculate speedups

plt.plot(gpu_counts, speedups, 'o-', label='Measured')
plt.plot(gpu_counts, gpu_counts, '--', label='Ideal')
plt.xlabel('Number of GPUs')
plt.ylabel('Speedup')
plt.legend()
plt.savefig('scaling_curve.png')
```

### 3. Batch Size Sensitivity Analysis

```bash
# Run with different batch sizes
for batch_size in 32 64 128 256 512; do
    python ../week1/01_single_gpu_baseline/train_single.py \
        --batch-size $batch_size \
        --epochs 5 \
        --output results_bs${batch_size}.csv
done
```

### 4. Learning Rate Scaling Study

```bash
# Test different LR scaling strategies
cd 06_scaling_benchmark

for n_gpu in 1 2 4 8; do
    for lr_scale in 0.5 1.0 2.0; do
        base_lr = 0.1
        lr = base_lr * lr_scale * n_gpu  # Linear scaling
        
        torchrun --nproc_per_node=$n_gpu benchmark_scaling.py \
            --learning-rate $lr \
            --output results_${n_gpu}gpu_lr${lr_scale}.csv
    done
done
```

---

## 📊 Understanding Results

### Strong Scaling Results

**Expected pattern (ideal is linear):**
```
GPUs | Epoch Time | Speedup | Efficiency
-----|------------|---------|------------
  1  | 45.0s      | 1.0x    | 100%
  2  | 24.5s      | 1.84x   | 92%
  4  | 13.2s      | 3.41x   | 85%
  8  | 7.5s       | 6.0x    | 75%
```

**Analysis:**
- Good scaling if efficiency > 80%
- Efficiency decreases with more GPUs due to communication overhead
- Communication time becomes dominant at very high scale

### Weak Scaling Results

**Doubling GPUs and batch size together:**
```
GPUs | Batch Size | Epoch Time | Scaled Speedup
-----|------------|------------|---------------
  1  | 128        | 45.0s      | 1.0x
  2  | 256        | 45.5s      | ~1.0x (weak scaling)
  4  | 512        | 46.2s      | ~1.0x (weak scaling)
  8  | 1024       | 48.5s      | ~1.0x (weak scaling)
```

**Interpretation:**
- Consistent epoch time = perfect weak scaling
- Slight increase in time = communication overhead
- Expect: 5-15% increase per doubling

### Batch Size vs Accuracy Trade-off

**From sensitivity analysis:**
```
Batch Size | 32 | 64 | 128 | 256 | 512
-----------|----|----|-----|-----|-----
Time/epoch | 60s| 48s| 42s | 39s | 37s
Final Acc  | 82%| 81%| 80%| 78% | 75%
```

**Key insight:** 
- Larger batches = faster training but worse final accuracy
- Sweet spot usually around 128-256 per GPU
- Can mitigate accuracy loss with learning rate warmup

### Learning Rate Scaling Law

**Linear scaling rule verification:**
```
Batch Size | Effective LR | Convergence | Final Acc
-----------|--------------|-------------|----------
128        | 0.1          | ✓ Good      | 82%
256        | 0.2          | ✓ Good      | 82%
512        | 0.4          | ✓ Fair      | 80%
1024       | 0.8          | ✗ Diverges  | 65%
```

**Lesson:** 
- Linear scaling works well for moderate batch sizes
- Extreme scaling (>1000) requires more sophisticated techniques

---

## 🧪 Hands-On Experiments

### Experiment 1: Find Optimal Batch Size
```bash
# Binary search for best batch size
for bs in 64 128 256 512; do
    python train_single.py --batch-size $bs --epochs 30
    # Record final accuracy
done

# Plot: Batch Size vs Final Accuracy
# Find inflection point where accuracy starts declining
```

### Experiment 2: Scaling Law Exponent
```bash
# Measure speedup for different problem sizes
# Hypothesis: Speedup = GPU_count ^ alpha
# Find alpha (should be close to 1 for perfect scaling)

alpha = log(speedup_8gpu / speedup_4gpu) / log(8/4)
print(f"Scaling exponent: {alpha:.2f}")
```

### Experiment 3: Communication Ratio
```bash
# Measure what fraction of time is communication
# Add timing instrumentation to all-reduce operations
# Compare: communication time vs computation time

comm_ratio = communication_time / (communication_time + compute_time)
print(f"Communication ratio: {comm_ratio * 100:.1f}%")
```

### Experiment 4: Gradient Accumulation Impact
```bash
# Accumulate gradients over N steps, then sync
# Larger messages → better bandwidth utilization
# But: less frequent synchronization

for accumulation_steps in [1, 2, 4, 8]:
    # Run training and measure throughput
```

---

## 📚 Resources

### Benchmarking Papers
- [CIFAR-10 Scaling Study](https://arxiv.org/abs/1711.07971)
- [ImageNet Scaling Laws](https://arxiv.org/abs/1708.03888)
- [Large Batch Training](https://arxiv.org/abs/1609.04836)

### Performance Analysis
- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA Nsys](https://developer.nvidia.com/nsys)
- [TensorBoard for PyTorch](https://pytorch.org/docs/stable/tensorboard.html)

---

## ✅ Verification Checklist

Before moving to Week 7, verify:
- [ ] You can run scaling benchmarks and collect metrics
- [ ] Speedup measurements match your expectations
- [ ] You understand strong vs weak scaling
- [ ] Batch size sensitivity is documented
- [ ] Learning rate scaling is validated
- [ ] Scaling curves are plotted and analyzed

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 5: TensorFlow Acceleration](../week5/README.md)  
**Next Week**: [Week 7: Review & Consolidation →](../week7/README.md)

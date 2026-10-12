# Week 7: Review & Consolidation

## 🎯 Learning Objectives

By the end of Week 7, you will have:
- ✅ Reinforced foundational concepts from weeks 1-6
- ✅ Completed hands-on exercises consolidating distributed training
- ✅ Practiced debugging distributed systems
- ✅ Analyzed real performance data
- ✅ Optimized a full training pipeline end-to-end
- ✅ Documented your learnings
- ✅ Built confidence for advanced topics (weeks 8-12)

---

## 📚 Key Concepts

Week 7 is a review and consolidation week covering:

### From Week 1: Single GPU Foundations
- PyTorch fundamentals: tensors, autograd, training loops
- Data loading and batching strategies
- Model evaluation and inference

### From Week 2: DDP Single-Node
- torchrun launcher and process spawning
- Gradient synchronization and all-reduce
- Batch size and learning rate scaling

### From Week 3: Multi-Node Training
- Network configuration and coordination
- MASTER_ADDR/PORT environment setup
- Scaling to multiple machines

### From Week 4: Communication & Optimization
- NCCL profiling and bandwidth measurement
- Collective operations (ring vs tree algorithms)
- Identifying communication bottlenecks

### From Week 5: TensorFlow Acceleration
- Alternative framework comparison
- tf.distribute strategies
- PyTorch vs TensorFlow trade-offs

### From Week 6: Scaling Benchmarks
- Strong and weak scaling analysis
- Batch size sensitivity
- Learning rate scaling verification

---

## 📁 Folder Structure

```
week7/
├── README.md                          # This file
├── 01_single_gpu_baseline/            # Copies from Week 1
├── 02_ddp_single_node/                # Copies from Week 2
├── 03_ddp_multi_node/                 # Copies from Week 3
├── 04_nccl_understanding/             # Copies from Week 4
├── 06_scaling_benchmark/              # Copies from Week 6
└── exercises/                         # Week 7 specific exercises
    ├── 01_debug_training_curves.md
    ├── 02_profile_communication.md
    ├── 03_optimize_convergence.md
    └── 04_capstone_project.md
```

---

## 🔧 Setup & Requirements

All environments from weeks 1-6 are reused.

---

## 🚀 How to Run

### Exercise 1: Debug Training Curves

**Objective**: Identify and fix convergence issues

**Scenario**: A training job is converging slowly. Analyze the issue.

```bash
cd exercises

# Run the problematic training
python problematic_training.py --output curve_bad.csv

# Analyze the curve
python analyze_curve.py --csv curve_bad.csv

# Questions:
# 1. Is the loss decreasing monotonically?
# 2. Is there a plateau or divergence?
# 3. What's the validation accuracy trend?

# Fix: Adjust hyperparameters
python problematic_training.py --learning-rate 0.05 --output curve_good.csv

# Verify fix worked
python analyze_curve.py --csv curve_good.csv
```

### Exercise 2: Profile Communication

**Objective**: Measure and analyze NCCL communication

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

# Run DDP training and capture logs
cd ../02_ddp_single_node
torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 1 2>&1 | tee nccl.log

# Analyze communication patterns
cd ../exercises
python analyze_communication.py --log ../02_ddp_single_node/nccl.log

# Questions:
# 1. What's the average all-reduce time?
# 2. Is communication time consistent or variable?
# 3. Is communication a bottleneck (<10% of epoch)?
```

### Exercise 3: Optimize Convergence

**Objective**: Find optimal hyperparameters through systematic search

```bash
cd exercises

# Run hyperparameter search
python hyperparameter_search.py \
    --batch-sizes 64 128 256 \
    --learning-rates 0.01 0.05 0.1 0.2 \
    --output hparams_results.csv

# Analyze results
python plot_hparams.py --csv hparams_results.csv

# Select best configuration
python train_with_best_hparams.py --csv hparams_results.csv
```

### Exercise 4: Capstone Project

**Objective**: Full end-to-end training pipeline optimization

**Project Scope:**
1. Design a ResNet model
2. Train on CIFAR-10 with multiple GPUs
3. Measure scaling efficiency
4. Optimize hyperparameters
5. Document all findings

**Deliverables:**
- `capstone_report.md`: Summary of optimizations
- `final_model.pth`: Trained model weights
- `scaling_results.csv`: Benchmark data
- `performance_analysis.pdf`: Plots and analysis

**Suggested Outline:**
```markdown
# Capstone Project Report

## 1. Experiment Design
- Model architecture
- Dataset and preprocessing
- Hardware setup (GPUs, interconnect)

## 2. Baseline Results
- Single GPU training (30 epochs)
- Time per epoch
- Final accuracy

## 3. Scaling Analysis
- 2 GPU, 4 GPU results
- Speedup measurements
- Communication overhead

## 4. Hyperparameter Optimization
- Batch size tuning
- Learning rate scaling
- Convergence analysis

## 5. Final Optimized Training
- Best configuration discovered
- Final accuracy and convergence time
- Communication efficiency

## 6. Lessons Learned
- Key insights from experiments
- Surprising findings
- Recommendations for future work
```

---

## 📊 Understanding Results

### Training Curve Analysis

**Healthy convergence:**
```
Epoch 1: Loss 2.30, Acc 25%
Epoch 2: Loss 1.80, Acc 40%
Epoch 3: Loss 1.45, Acc 50%
Epoch 4: Loss 1.25, Acc 58%
...
Epoch 30: Loss 0.85, Acc 80%

Pattern: Smooth decrease, no spikes
```

**Red flags:**
```
Loss not decreasing: Learning rate too low
Loss diverging: Learning rate too high
Validation loss increasing while training decreases: Overfitting
Accuracy plateau: Model capacity insufficient or stuck local minimum
```

### Communication Profile

**Example NCCL analysis output:**
```
All-Reduce Statistics (Rank 0):
  Total calls: 391 (one per training step)
  Average time: 1.23ms
  Min: 0.98ms, Max: 2.14ms
  Total communication time: 482ms (11% of epoch)

Communication per step: 1.23ms
Computation per step: 113.2ms
Ratio: 1:92 (computation-dominated)

Conclusion: Communication is NOT a bottleneck
```

### Hyperparameter Search Results

**Example grid search findings:**
```
Batch Size: 64    | LR: 0.01 | Final Acc: 78.5%
Batch Size: 64    | LR: 0.05 | Final Acc: 80.2% ✓ BEST for BS=64
Batch Size: 128   | LR: 0.05 | Final Acc: 79.8%
Batch Size: 128   | LR: 0.10 | Final Acc: 80.1% ✓ BEST for BS=128
Batch Size: 256   | LR: 0.10 | Final Acc: 78.5%
Batch Size: 256   | LR: 0.20 | Final Acc: 79.2% ✓ BEST for BS=256
```

---

## 🧪 Hands-On Experiments

### Experiment 1: Temperature Check
```bash
# Verify you can reproduce results from weeks 1-6
cd week1/01_single_gpu_baseline
python train_single.py --batch-size 128 --epochs 3
# Expected: Final accuracy > 40%

cd ../../week2/02_ddp_single_node
torchrun --standalone --nproc_per_node=2 train_ddp.py --epochs 3
# Expected: Same convergence as Week 1

cd ../../week4/04_nccl_understanding
torchrun --standalone --nproc_per_node=2 allreduce_benchmark.py
# Expected: Bandwidth > 40 GB/s
```

### Experiment 2: Memory Profiling
```python
# Understand memory usage at different batch sizes
from torch.utils.checkpoint import checkpoint

def profile_memory(batch_size):
    torch.cuda.reset_peak_memory_stats()
    
    # Run training
    train_one_batch(batch_size)
    
    peak_memory = torch.cuda.max_memory_allocated()
    return peak_memory

for bs in [32, 64, 128, 256, 512]:
    peak = profile_memory(bs)
    print(f"Batch size {bs}: {peak / 1e9:.2f} GB")
```

### Experiment 3: Gradient Compression
```python
# Experiment with gradient compression to reduce communication
# Implement: quantization, sparsification, or other techniques

# Measure: Does it speed up training with same accuracy?
```

---

## 📚 Resources

### Self-Assessment
- [MLOps Maturity Model](https://ml-ops.systems/)
- [Distributed Training Checklist](https://horovod.ai/docs/tutorials/)

### Further Reading
- [Distributed Deep Learning Book](https://arxiv.org/abs/1708.03888)
- [ML Systems Design Patterns](https://mlsysdesign.com/)

---

## ✅ Verification Checklist

Before moving to Week 8, verify:
- [ ] You can reproduce results from weeks 1-6
- [ ] You understand debugging distributed training
- [ ] You can profile and analyze performance
- [ ] You can optimize hyperparameters systematically
- [ ] Capstone project is completed with documentation
- [ ] All code is commented and reproducible

---

## 🎓 Concepts for Next Week

In **[Week 8: Advanced Topics - LR Scaling, Checkpointing, Packaging](../week8/README.md)**, you'll learn:
- Learning rate scaling at massive scale (>1000 GPUs)
- Gradient checkpointing to save memory
- Model packaging and versioning for production
- Deployment best practices

**Bridge**: Consolidation week → Advanced optimization techniques

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 6: Scaling Benchmarks & EDA](../week6/README.md)  
**Next Week**: [Week 8: Advanced Topics →](../week8/README.md)

# Applied Accelerated AI: A Comprehensive Learning Journey

## 🚀 Program Overview

E&ICT, IIT Guwahati's **Applied Accelerated AI** course is a hands-on, industry-curated curriculum designed to prepare you for cutting-edge AI engineering. This program takes you from GPU fundamentals through distributed training, to LLM deployment—all with real, runnable code.

**🎯 What You'll Learn:**
- Single-GPU deep learning training optimization
- Distributed Data Parallel (DDP) training at scale
- Multi-node distributed systems and NCCL communication
- Containerization, CI/CD pipelines, and deployment
- Learning rate scaling, checkpointing, and model packaging
- Foundational language models and their architectures
- Production-ready AI inference services

---

## 📚 Course Structure

### **Foundation Phase (Weeks 1-4): GPU & Distributed Training**

| Week | Topic | Focus | Code Examples |
|------|-------|-------|---|
| **[Week 1](./week1/README.md)** | Single GPU Basics | PyTorch fundamentals, tensor operations, basic model training | ResNet on CIFAR-10, FastAPI inference service |
| **[Week 2](./week2/README.md)** | Single-Node Multi-GPU (DDP) | Distributed Data Parallel, torchrun launcher, synchronized training | DDP training, container setup |
| **[Week 3](./week3/README.md)** | Multi-Node Distributed Training | Cross-node communication, networking config, distributed setup | Multi-node DDP, networking protocols |
| **[Week 4](./week4/README.md)** | Communication & Optimization | NCCL benchmarking, Istio service mesh, collective operations | All-reduce bandwidth analysis |

### **Scaling & Optimization Phase (Weeks 5-8)**

| Week | Topic | Focus | Key Learnings |
|------|-------|-------|---|
| **[Week 5](./week5/README.md)** | TensorFlow Acceleration | Alternative framework deep-dive | TensorFlow GPU optimization |
| **[Week 6](./week6/README.md)** | Scaling Benchmarks & EDA | Performance measurement, data analysis | Benchmark scaling laws, visualization |
| **[Week 7](./week7/README.md)** | Review & Consolidation | Reinforce weeks 1-6 concepts | Hands-on project work |
| **[Week 8](./week8/README.md)** | Advanced Topics | LR scaling, checkpointing, packaging | Production-ready model versioning |

### **Language Models Phase (Weeks 9-12)**

| Week | Topic | Focus | Key Concepts |
|------|-------|-------|---|
| **[Week 9](./week9/README.md)** | LLM Foundations | Transformer architecture, attention mechanisms | Foundational language model concepts |
| **[Week 10](./week10/README.md)** | LLM Deep Dive | Model internals, training dynamics | Language model behavior analysis |
| **[Week 11](./week11/README.md)** | Advanced LLM Topics | Fine-tuning, prompt engineering, optimization | Practical LLM techniques |
| **[Week 12](./week12/README.md)** | LLM Deployment | Production serving, inference optimization | Deployment patterns and best practices |

---

## 🏗️ Quick Start

### Prerequisites
- Python 3.8+ (preferably 3.10+)
- NVIDIA GPU with CUDA support (for GPU code)
- git and pip

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd iit-guwahati-applied-ai

# Navigate to a week
cd week1

# Install dependencies
pip install -r requirements.txt
```

### Running Your First Example
```bash
cd week1/01_single_gpu_baseline
python train_single.py --data-root /tmp/cifar --epochs 3 --batch-size 128
```

See [Week 1 README](./week1/README.md) for detailed setup instructions.

---

## 📖 How to Use This Repository

1. **Start with [Week 1](./week1/README.md)** - Foundation in single-GPU training
2. **Follow the weekly progression** - Each week builds on the previous
3. **Read each week's README** - Contains objectives, setup steps, how to run code, and result interpretation
4. **Run the code** - Execute examples and observe the behavior
5. **Compare your results** - Use the expected outputs as benchmarks
6. **Move to the next week** - Each README has a "Next Steps" section

### For Different Learning Styles

**🔍 Visual Learners:** Each week folder contains PDFs, presentations, and notebooks  
**💻 Hands-On Learners:** Run the Python scripts and experiment with parameters  
**📊 Theory First:** Read the ARCHITECTURE_GUIDE.md (in docs/) for system design  
**🚀 Fast Track:** Jump to week of interest, but understand prerequisite concepts first

---

## 🗂️ Repository Structure

```
.
├── README.md                    # You are here
├── docs/                        # Reference documentation
│   ├── ARCHITECTURE_GUIDE.md   # System design & learning flow
│   └── QUICK_REFERENCE.md      # Command reference
├── week1/                       # Single GPU Basics
│   ├── README.md               # Week 1 guide
│   ├── requirements.txt
│   ├── 01_single_gpu_baseline/
│   └── 01_inference_service/
├── week2/                       # Single-Node Multi-GPU (DDP)
│   ├── README.md               # Week 2 guide
│   ├── 02_ddp_single_node/
│   └── week02_containers/
├── ... (weeks 3-12)
└── week12/                      # LLM Deployment
    └── README.md               # Week 12 guide
```

---

## 🎯 Learning Objectives by Phase

### Phase 1: Foundations (Weeks 1-4)
✅ Understand GPU memory and compute hierarchy  
✅ Write optimized PyTorch training scripts  
✅ Distribute training across multiple GPUs  
✅ Scale to multiple nodes with NCCL  
✅ Benchmark and measure communication efficiency

### Phase 2: Optimization (Weeks 5-8)
✅ Compare PyTorch and TensorFlow performance  
✅ Analyze scaling laws and batch size effects  
✅ Implement checkpointing and recovery  
✅ Package models for production deployment  
✅ Version and serve models reliably

### Phase 3: Language Models (Weeks 9-12)
✅ Understand transformer architecture  
✅ Analyze language model behavior  
✅ Fine-tune models for specific tasks  
✅ Optimize inference for production  
✅ Deploy LLMs at scale

---

## 💡 Key Features

- **Real Code, Real Hardware** - All examples run on actual GPUs
- **Progressive Complexity** - Start simple, build advanced concepts
- **Best Practices** - Production-ready patterns throughout
- **Reproducible Results** - Detailed setup and expected outputs
- **Multi-Framework** - PyTorch focus with TensorFlow comparison

---

## 🔧 Development Tips

### GPU Debugging
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor during training
nvidia-smi -l 1  # Refresh every 1 second
```

### Common Issues
- **GPU out of memory**: Reduce `--batch-size`
- **Slow data loading**: Increase `--num-workers`
- **DDP not finding processes**: Check network and MASTER_ADDR/PORT

See each week's README for week-specific troubleshooting.

---

## 📚 Reference Documentation

- **[ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md)** - Deep dive into system design and learning flow
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command cheatsheet and quick lookup
- **[NEW_JOINER_GUIDE.md](docs/NEW_JOINER_GUIDE.md)** - Getting started guide
- **Each Week's README** - Specific setup and execution guides

---

## 🎓 Course Philosophy

This course teaches **practical AI engineering**:
- Not just theory, but **runnable implementations**
- Not just single-GPU, but **distributed systems**
- Not just training, but **inference and deployment**
- Not just PyTorch, but **framework comparison**

Every week includes:
- 📝 Clear learning objectives
- 💾 Production-quality code
- 📊 Benchmark outputs
- 🔍 Result interpretation guides
- 🚀 Next steps and prerequisites

---

## 🤝 Contributing & Support

For questions about specific weeks, see that week's README.md for:
- Setup troubleshooting
- Expected behavior and common issues
- Resources and related documentation

---

## 📅 Last Updated
October 2026

---

**Ready to get started?** → [Jump to Week 1](./week1/README.md)

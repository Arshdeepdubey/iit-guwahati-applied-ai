# Week 11: Advanced LLM Topics

## 🎯 Learning Objectives

By the end of Week 11, you will understand and be able to:
- ✅ Fine-tune pretrained language models on custom tasks
- ✅ Implement and evaluate prompt engineering techniques
- ✅ Understand LoRA (Low-Rank Adaptation) for efficient fine-tuning
- ✅ Handle long-context sequences efficiently
- ✅ Implement instruction tuning and RLHF basics
- ✅ Measure and optimize model inference speed
- ✅ Handle edge cases and failure modes

---

## 📚 Key Concepts

### Fine-Tuning Strategies
- **Full Fine-Tuning**: Update all weights (expensive, most effective)
- **Parameter-Efficient**: LoRA, Prefix Tuning, Adapters
- **Prompt-Based**: In-context learning, few-shot learning
- **Domain Adaptation**: Tailoring to specific domain (medical, legal, code)

### Prompt Engineering
- **Instruction Design**: Clear, specific prompts yield better results
- **Chain-of-Thought**: Encouraging reasoning step-by-step
- **Few-Shot Learning**: Providing examples in prompt
- **Role Assignment**: "You are a..." context setting

### LoRA (Low-Rank Adaptation)
- **Parameter Reduction**: 0.1-1% of full fine-tuning parameters
- **Training Speed**: 2-3x faster than full fine-tuning
- **Memory Efficiency**: 80-90% memory savings
- **Adapter Composition**: Combining multiple LoRAs

### Long Context Handling
- **Attention Complexity**: O(n²) scales poorly
- **Flash Attention**: Fast approximate attention
- **Sparse Attention**: Only attend to k nearest neighbors
- **Paged Attention**: Efficient memory management

### Inference Optimization
- **KV-Cache**: Reusing computed keys/values
- **Batching**: Processing multiple requests together
- **Quantization**: 4-bit or 8-bit weights for speed
- **Speculative Decoding**: Parallel candidate generation

---

## 📁 Folder Structure

```
week11/
├── README.md
├── pptx/          # Presentations
├── videos/        # Video lectures
└── zip/          # Course materials
```

---

## 🔧 Setup & Requirements

```bash
pip install peft  # For LoRA
pip install bitsandbytes  # For quantization
pip install vllm  # For fast inference
pip install wandb  # For experiment tracking
```

---

## 🚀 How to Run

### 1. Fine-Tuning with LoRA

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
)

# Wrap with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ~2% of original

# Train normally
trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

### 2. Prompt Engineering

```python
# Basic prompt
prompt1 = "What is AI?"

# Better structured prompt
prompt2 = """Answer the following question concisely.
Question: What is AI?
Answer:"""

# Few-shot learning
prompt3 = """Examples:
Q: What is AI? A: Artificial Intelligence is...
Q: What is ML? A: Machine Learning is...
Q: What is DL? A: Deep Learning is..."""

# Chain-of-thought
prompt4 = """Think step by step.
Q: What is AI?
Step 1: Define Artificial
Step 2: Define Intelligence
Answer:"""
```

### 3. Inference Optimization

```python
from vllm import LLM, SamplingParams

# Load with quantization
llm = LLM(model="gpt2", quantization="awq", dtype="float16")

# Batch inference
prompts = ["What is AI?", "What is ML?", "What is DL?"]
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 📊 Understanding Results

### Fine-Tuning Comparison

| Method | Parameters | Speed | Memory | Accuracy |
|--------|-----------|-------|--------|----------|
| Full Fine-Tuning | 100% | 1.0x | 100% | 95% |
| LoRA (r=8) | 1% | 1.4x | 20% | 94.2% |
| LoRA (r=4) | 0.5% | 1.6x | 15% | 93.5% |
| Prefix Tuning | 0.5% | 1.5x | 10% | 92.1% |

**Interpretation:**
- LoRA achieves 94%+ performance with 1% parameters
- Speedup varies by hardware (less benefit on GPUs)
- Memory is dramatically reduced

### Prompt Engineering Impact

**Example task: Sentiment classification**
```
Bad prompt: "Classify"
Accuracy: 42%

Better prompt: "Classify the sentiment as positive or negative"
Accuracy: 71%

Best prompt: "Classify the sentiment as positive or negative. 
Positive example: I love this!
Negative example: This is terrible."
Accuracy: 88%
```

**Lesson**: Prompt design can match or exceed fine-tuning

### Inference Speed Optimization

**Latency comparison (ms per token):**
```
Full precision (FP32): 50ms
Mixed precision (FP16): 30ms
Quantized (INT8): 15ms
Quantized (INT4): 10ms
```

**Trade-off**: Accuracy loss is minimal (<1%) with INT8

---

## 🧪 Hands-On Experiments

### Experiment 1: LoRA Rank Sensitivity
```bash
# Train with different ranks and compare
for r in 4 8 16 32 64; do
    train_with_lora(rank=r)
    evaluate()
    # Plot: Rank vs Accuracy
done
```

### Experiment 2: Prompt Engineering A/B Test
```bash
# Test multiple prompts and measure quality
prompts_variants = [...]
for prompt in prompts_variants:
    responses = model.generate(prompt)
    quality_score = evaluate_responses(responses)
    # Find best prompt
```

### Experiment 3: Quantization Impact
```python
# Compare inference quality across quantization levels
for dtype in [torch.float32, torch.float16, torch.int8]:
    responses = model.generate(prompt, dtype=dtype)
    quality = evaluate_responses(responses)
    latency = measure_latency(model, dtype)
```

---

## 🐛 Troubleshooting

### Issue: LoRA fine-tuning diverges
**Solution:**
```python
# Reduce learning rate
training_args.learning_rate = 1e-4  # From 5e-4
# Increase warmup
training_args.warmup_steps = 500
```

### Issue: Inference speed is still slow
**Solution:**
```python
# Use batching
responses = model.generate(batch_size=32)  # Not single samples

# Use quantization
model = load_quantized_model("int4")
```

---

## 📚 Resources

- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09714)
- [vLLM Fast Inference](https://vllm.ai/)

---

## ✅ Verification Checklist

Before moving to Week 12, verify:
- [ ] You can fine-tune with LoRA and compare to full fine-tuning
- [ ] Prompt engineering improves output quality
- [ ] Inference optimization reduces latency
- [ ] Quantization maintains accuracy while speeding up
- [ ] You understand trade-offs between methods

---

**Last Updated**: October 2026  
**Next Week**: [Week 12: LLM Deployment →](../week12/README.md)

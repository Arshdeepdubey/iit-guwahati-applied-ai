# Week 9: Foundational Language Models

## 🎯 Learning Objectives

By the end of Week 9, you will understand and be able to:
- ✅ Understand Transformer architecture fundamentals
- ✅ Explain attention mechanisms and self-attention
- ✅ Understand why Transformers revolutionized NLP
- ✅ Implement basic attention layers
- ✅ Train a simple language model from scratch
- ✅ Understand positional encoding and its role
- ✅ Measure model perplexity and language modeling metrics

---

## 📚 Key Concepts

### Transformer Architecture
- **Encoder-Decoder Structure**: Processing and generation
- **Multi-Head Attention**: Parallel attention representations
- **Position-wise Feed-Forward**: Dense layers with ReLU
- **Layer Normalization**: Stabilizing deep networks
- **Residual Connections**: Skip connections for gradient flow

### Attention Mechanism
- **Scaled Dot-Product Attention**: Query-Key-Value computation
- **Attention Weights**: Softmax distribution over sequence
- **Self-Attention**: Attending to own sequence
- **Cross-Attention**: Attending to encoder output

### Language Modeling
- **Next Token Prediction**: Training objective
- **Perplexity**: Evaluation metric
- **Context Window**: Maximum sequence length
- **Tokenization**: Converting text to token IDs

### Embedding & Positional Encoding
- **Token Embeddings**: Dense representation per token
- **Positional Encodings**: Sine/cosine position encoding
- **Absolute vs Relative Positions**: Position encoding schemes
- **Rope (Rotary) Embeddings**: Modern alternative

---

## 📁 Folder Structure

```
week9/
├── README.md
└── Assignment materials and notebooks
```

---

## 🔧 Setup & Requirements

```bash
pip install transformers datasets tokenizers evaluate
pip install matplotlib seaborn  # For visualization
```

---

## 🚀 How to Run

### 1. Basic Attention Mechanism

```python
import torch
import torch.nn as nn

# Query, Key, Value projections
batch_size, seq_len, d_model = 4, 10, 64
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# Scaled dot-product attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
attention_weights = torch.softmax(scores, dim=-1)
output = torch.matmul(attention_weights, V)
```

### 2. Simple Language Model Training

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare data
train_dataset = TextDataset(tokenizer=tokenizer, file_path='data.txt', block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Train
training_args = TrainingArguments(
    output_dir='./models',
    num_train_epochs=3,
    per_device_train_batch_size=8,
)
trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)
trainer.train()
```

---

## 📊 Understanding Results

### Perplexity Analysis

**Expected metrics:**
```
Initial perplexity: ~1500
After 1 epoch: ~200
After 3 epochs: ~50
Final goal: < 20 on validation set
```

**Interpretation:**
- Lower perplexity = better predictions
- Perplexity = 2^(cross_entropy_loss)

### Training Dynamics

**Loss curve shape:**
```
Epoch 1: 6.5 → 4.2 (steep decline, learning quickly)
Epoch 2: 4.2 → 2.1 (moderate decline)
Epoch 3: 2.1 → 1.8 (plateau, approaching saturation)
```

---

## ✅ Verification Checklist

Before moving to Week 10, verify:
- [ ] You understand transformer architecture
- [ ] Self-attention mechanism is clear
- [ ] You can implement basic attention
- [ ] Language model training works
- [ ] You understand perplexity as a metric
- [ ] Positional encoding role is understood

---

**Last Updated**: October 2026  
**Next Week**: [Week 10: Language Model Deep Dive →](../week10/README.md)

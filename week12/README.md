# Week 12: LLM Deployment & Production Systems

## 🎯 Learning Objectives

By the end of Week 12, you will understand and be able to:
- ✅ Design and implement LLM serving systems
- ✅ Deploy models at scale with Kubernetes
- ✅ Implement caching and request batching
- ✅ Monitor LLM performance in production
- ✅ Handle failure modes and fallbacks
- ✅ Optimize cost and latency trade-offs
- ✅ Implement security and access control
- ✅ Set up observability (logging, metrics, tracing)

---

## 📚 Key Concepts

### LLM Serving Architecture
- **Request Batching**: Grouping requests for efficiency
- **KV-Cache Management**: Efficient token storage
- **Token Streaming**: Returning tokens as generated
- **Load Balancing**: Distributing across replicas

### Production Infrastructure
- **Container Orchestration**: Kubernetes for scaling
- **Service Mesh**: Istio for traffic management
- **Auto-Scaling**: HPA (Horizontal Pod Autoscaler)
- **Monitoring & Alerts**: Prometheus + Grafana

### Failure Handling
- **Timeouts**: Maximum response time
- **Retries**: Exponential backoff for transient failures
- **Circuit Breakers**: Prevent cascading failures
- **Fallback Models**: Degraded mode with smaller model

### Cost Optimization
- **Spot Instances**: Cheaper compute (interruptible)
- **Model Quantization**: Smaller models, faster inference
- **Caching**: Response cache for common queries
- **Mixture of Experts**: Selective model loading

### Security
- **Authentication**: API key, OAuth
- **Authorization**: Role-based access control
- **Rate Limiting**: Prevent abuse
- **Data Privacy**: PII redaction, logging control

### Observability
- **Metrics**: Latency, throughput, error rate
- **Logs**: Debugging and auditing
- **Traces**: Request flow across services
- **Alerts**: Automatic notifications on anomalies

---

## 📁 Folder Structure

```
week12/
├── README.md
├── pptx/          # Deployment guides
└── zip/          # Configuration files and scripts
```

---

## 🔧 Setup & Requirements

```bash
pip install fastapi uvicorn
pip install kubernetes  # For K8s integration
pip install prometheus-client  # For metrics
pip install opentelemetry-api  # For tracing
```

---

## 🚀 How to Run

### 1. Basic LLM API Server

```python
# server.py
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# Run: uvicorn server:app --reload
```

### 2. Production Features

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import time

app = FastAPI()

# Load model once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    yield
    # Shutdown - cleanup

app = FastAPI(lifespan=lifespan)

# Monitoring
from prometheus_client import Counter, Histogram
request_count = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')

@app.post("/generate")
async def generate(prompt: str):
    start = time.time()
    try:
        # ... inference logic ...
        request_count.labels(status="success").inc()
    except Exception as e:
        request_count.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request_latency.observe(time.time() - start)
    return response

# Metrics endpoint for Prometheus
from prometheus_client import generate_latest

@app.get("/metrics")
def metrics():
    return generate_latest()
```

### 3. Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-server
  template:
    metadata:
      labels:
        app: llm-server
    spec:
      containers:
      - name: llm
        image: llm-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            nvidia.com/gpu: 1
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
# hpa.yaml (Auto-scaling)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-server
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 4. Request Batching

```python
from collections import deque
import asyncio

class BatchProcessor:
    def __init__(self, batch_size=32, max_wait_ms=100):
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue = deque()
        self.results = {}
    
    async def process(self, request_id, prompt):
        self.queue.append((request_id, prompt))
        
        if len(self.queue) >= self.batch_size:
            await self._flush()
        else:
            # Wait briefly for more requests
            await asyncio.sleep(self.max_wait)
            if len(self.queue) > 0:
                await self._flush()
        
        # Return when ready
        while request_id not in self.results:
            await asyncio.sleep(0.01)
        return self.results.pop(request_id)
    
    async def _flush(self):
        batch = [self.queue.popleft() for _ in range(len(self.queue))]
        requests = [req[0] for req in batch]
        prompts = [req[1] for req in batch]
        
        # Batch inference
        outputs = model.generate(prompts, batch_size=len(batch))
        
        # Store results
        for req_id, output in zip(requests, outputs):
            self.results[req_id] = output
```

---

## 📊 Understanding Results

### Performance Metrics

**Expected production metrics:**
```
Metric              | Target      | Good        | Acceptable
--------------------|-------------|-------------|------------
P95 Latency         | <100ms      | <500ms      | <2s
P99 Latency         | <200ms      | <1s         | <5s
Throughput          | >100 req/s  | >50 req/s   | >10 req/s
Error Rate          | <0.1%       | <1%         | <5%
GPU Utilization     | >80%        | >60%        | >40%
Model Availability  | 99.9%       | 99%         | 95%
```

### Cost Analysis

**Example: Serving 1000 requests/sec**
```
Option 1: Single large GPU (A100)
  Cost: $2/hour
  Latency: 50ms
  
Option 2: Multiple smaller GPUs (4x T4)
  Cost: $1/hour
  Latency: 100ms (higher parallelism)
  
Option 3: Quantized model + CPUs
  Cost: $0.5/hour
  Latency: 200ms
  
Recommendation: Option 2 (balanced cost/performance)
```

---

## 🧪 Hands-On Experiments

### Experiment 1: Load Testing
```bash
# Test with different concurrent users
for concurrent in 10 50 100 500 1000; do
    ab -n 10000 -c $concurrent http://localhost:8000/generate
    # Record: latency, throughput, error rate
done
```

### Experiment 2: Auto-Scaling Verification
```bash
# Generate load and verify HPA scales
# Monitor pod count and latency during scaling
kubectl get pods -w
# Should see new pods starting when load increases
```

### Experiment 3: Failure Recovery
```bash
# Kill a pod and verify:
# 1. Requests rerouted to other pods
# 2. New pod restarts automatically
# 3. No request loss

kubectl delete pod llm-server-xyz123
kubectl get pods -w  # Watch recovery
```

---

## 🐛 Troubleshooting

### Issue: High latency during peak load
**Solutions:**
1. Enable request batching
2. Increase replica count
3. Use model quantization
4. Implement caching

### Issue: GPU memory errors
**Solutions:**
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce batch size
# Use smaller model variant
```

### Issue: Uneven request distribution
**Solution:**
```yaml
# Update K8s service to use round-robin
sessionAffinity: None
```

---

## 📚 Resources

### Deployment Patterns
- [vLLM Production Setup](https://docs.vllm.ai/en/latest/)
- [TorchServe](https://pytorch.org/serve/)
- [NVIDIA Triton Server](https://developer.nvidia.com/nvidia-triton-inference-server)

### Kubernetes for ML
- [KServe](https://kserve.github.io/website/)
- [Kubeflow](https://www.kubeflow.org/)

### Monitoring & Observability
- [Prometheus](https://prometheus.io/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Datadog for ML Monitoring](https://www.datadoghq.com/)

---

## ✅ Verification Checklist

Before completing the course, verify:
- [ ] LLM API server runs and responds correctly
- [ ] Production features (monitoring, health checks) work
- [ ] Kubernetes deployment and scaling work
- [ ] Request batching improves throughput
- [ ] Failure recovery is automatic
- [ ] Metrics are collected and queryable
- [ ] Cost/performance trade-offs are understood

---

## 🎓 Course Summary

**Weeks 1-4**: Distributed Training Foundations
- Single GPU → Multi-GPU → Multi-Node training
- Understanding communication and optimization

**Weeks 5-8**: Scaling & Advanced Topics
- TensorFlow comparison, benchmarking
- Learning rate scaling, checkpointing, packaging

**Weeks 9-12**: Language Models & Production
- Transformer architecture and LLMs
- Advanced techniques and production deployment

---

## 🚀 Next Steps After This Course

1. **Specialize**: Choose a domain (vision, NLP, multimodal)
2. **Research**: Dive into recent papers in your domain
3. **Projects**: Build end-to-end projects combining all weeks
4. **Industry**: Apply knowledge to real production systems
5. **Advanced**: Explore areas like retrieval (RAG), agents, RLHF

---

**Last Updated**: October 2026  
**Previous Week**: [← Week 11: Advanced LLM Topics](../week11/README.md)

🎉 **Congratulations on Completing the Course!**

---

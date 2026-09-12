"""
A minimal FastAPI inference service used throughout Week 3 and Week 8.

Endpoints:
  GET /health     - readiness/liveness probe target
  POST /predict   - receives a feature vector, returns a model output
  GET /metrics    - Prometheus-format metrics

In a real system the model would be loaded from an artifact store;
here we instantiate a tiny nn.Module at startup so the service has
realistic load-time characteristics.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------
# Model definition. Kept tiny so the container starts fast in demos.
# ----------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------
# Lightweight in-process metrics. For production use prometheus_client.
# ----------------------------------------------------------------

class Metrics:
    def __init__(self) -> None:
        self.requests_total = 0
        self.errors_total = 0
        self.latency_sum_ms = 0.0
        self.ready = False

    def record(self, latency_ms: float, error: bool = False) -> None:
        self.requests_total += 1
        self.latency_sum_ms += latency_ms
        if error:
            self.errors_total += 1

    def prometheus(self) -> str:
        avg = (self.latency_sum_ms / self.requests_total
               if self.requests_total else 0.0)
        return (
            f"# HELP inference_requests_total Total inference requests.\n"
            f"# TYPE inference_requests_total counter\n"
            f"inference_requests_total {self.requests_total}\n"
            f"# HELP inference_errors_total Inference requests that errored.\n"
            f"# TYPE inference_errors_total counter\n"
            f"inference_errors_total {self.errors_total}\n"
            f"# HELP inference_latency_ms_avg Average latency in ms.\n"
            f"# TYPE inference_latency_ms_avg gauge\n"
            f"inference_latency_ms_avg {avg:.3f}\n"
        )


metrics = Metrics()
model: TinyMLP | None = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------
# Application lifespan: load the model before accepting traffic.
# The readiness probe checks `metrics.ready`, so nothing is served
# until loading finishes.
# ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"[startup] loading model on device={device}")
    model = TinyMLP().to(device).eval()
    # Warm up: force any lazy CUDA init so the first real request is fast.
    with torch.no_grad():
        _ = model(torch.zeros(1, 16, device=device))
    metrics.ready = True
    print("[startup] ready")
    yield
    print("[shutdown] draining")


app = FastAPI(title="tiny-inference", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=16, max_length=16)


class PredictResponse(BaseModel):
    probs: list[float]
    latency_ms: float


@app.get("/health")
def health():
    """Used by Kubernetes readiness and liveness probes.

    Returns 503 until model load is complete. Once ready, returns 200
    and a small payload.
    """
    if not metrics.ready:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok", "device": device}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        x = torch.tensor([req.features], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    except Exception as exc:
        metrics.record(0.0, error=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - t0) * 1000.0
    metrics.record(latency_ms)
    return PredictResponse(probs=probs, latency_ms=latency_ms)


@app.get("/metrics", response_class=PlainTextResponse)
def prom_metrics():
    return metrics.prometheus()


if __name__ == "__main__":
    # For local runs only. In-cluster, uvicorn is the container CMD.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

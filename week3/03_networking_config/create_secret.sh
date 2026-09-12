#!/usr/bin/env bash
# Secrets are never committed as plain YAML with real values checked into
# Git -- create them imperatively instead (Session 2, slide 8).
#
# In this demo the values are dummy placeholders. In production, use
# External Secrets Operator (ESO) to sync AWS Secrets Manager / Vault
# into this Secret automatically instead of running this by hand
# (Session 2, slide 9 "Production tip").
set -euo pipefail

kubectl create secret generic ml-tokens \
  --from-literal=WANDB_API_KEY=demo-wandb-key-not-real \
  --from-literal=HF_TOKEN=demo-hf-token-not-real \
  --dry-run=client -o yaml | kubectl apply -f -

echo "==> Secret created. Verify (base64, NOT decrypted):"
kubectl get secret ml-tokens -o yaml

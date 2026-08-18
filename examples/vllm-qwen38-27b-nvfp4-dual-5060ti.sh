#!/usr/bin/env bash
# Alternative route: Qwen3.8-27B NVFP4 via Docker vLLM on dual RTX 5060 Ti (TP=2).
# Sanitized from seed. Adjust MODEL_DIR, CACHE_DIR, and host port for your env.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/path/to/unsloth/Qwen3.8-27B-NVFP4}"   # read-only model dir
CACHE_DIR="${CACHE_DIR:-/path/to/vllm-cache}"                   # persist JIT/cache
HOST_PORT="${HOST_PORT:-8081}"
NAME="${NAME:-local-llm-vllm}"

# NOTE: do NOT enable vLLM MTP here. On these no-P2P dual cards it is a net
# negative (31.7 -> 15.7 tok/s) because each draft round crosses PCIe.
docker rm -f "$NAME" 2>/dev/null || true

exec docker run --rm --gpus all \
  --name "$NAME" \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/root/.cache/vllm" \
  -p "$HOST_PORT:8000" \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name local-llm \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1}' \
  --mm-processor-kwargs '{"max_pixels":802816}' \
  --reasoning-parser qwen3 \
  "$@"

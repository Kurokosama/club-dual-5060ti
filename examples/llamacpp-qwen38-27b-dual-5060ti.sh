#!/usr/bin/env bash
# Qwen3.8 27B UD-Q5_K_XL on dual RTX 5060 Ti (llama.cpp, tensor-split + MTP).
# Sanitized seed preset. Adjust MODEL, CTX, and port for your environment.
set -euo pipefail

MODEL="${MODEL:-/path/to/qwen3.8-27b-ud-q5_k_xl.gguf}"
CTX="${CTX:-192000}"
PORT="${PORT:-8081}"
TPSPLIT="${TPSPLIT:-50/50}"

# NOTE: do NOT expect a speedup from -p-min here (measured net negative).
exec llama-server \
  -m "$MODEL" \
  --ctx-size "$CTX" \
  --tensor-split "$TPSPLIT" \
  --split-mode layer \
  -ngl 999 \
  --no-mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --mtp 3 \
  --reasoning-effort low \
  --host 127.0.0.1 --port "$PORT" \
  "$@"

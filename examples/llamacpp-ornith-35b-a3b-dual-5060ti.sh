#!/bin/bash
# Ornith-1.5-35B-A3B (Q4_K_M) on dual RTX 5060 Ti 16GB — 256K native ctx + vision, NO speculative decoding.
#
# Measured on MY box (Ryzen 5 4500 PCIe3-only, 2x RTX 5060 Ti, no P2P):
#   pure generation ~109 tok/s  (plain / no MTP)
#   MTP was a NEGATIVE here (~73 tok/s best), see README "8/19 update" + llama.cpp issue #26750.
# Numbers may differ on other hardware — re-measure before trusting them.
#
# Note: official llama.cpp releases are CUDA-build; to use BOTH GPUs in tensor-split you
# must self-compile the CUDA build (Blackwell -> -DCMAKE_CUDA_ARCHITECTURES=120).

MODEL=/path/to/ornith-1.5-35b-q4_k_m.gguf
MMPROJ=/path/to/mmproj-ornith-1.5-35b-bf16.gguf

llama-server \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --image-min-tokens 1024 \
  --ctx-size 256000 \
  --flash-attn on \
  -ngl 999 \
  --tensor-split 1,1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --reasoning-effort low \
  --host 127.0.0.1 --port 8081

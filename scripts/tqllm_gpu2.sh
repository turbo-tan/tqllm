#!/usr/bin/env bash
# tqLLM server on GPU 2 (RTX 3090), 256K context, DFlash2 drafter, OpenAI API on 127.0.0.1:8086.
# Runs the NATIVE build (build/tqserve) so local source patches apply, e.g. TQLLM_DEBUG=reqlog.
# Rebuild after editing source:  cmake --build build --target tqserve -j$(nproc)
# Managed by pm2 app tqllm-gpu2 (tqllm_gpu2.config.js): pm2 start/stop/restart tqllm-gpu2.
set -euo pipefail
cd /home/green-gpu/tqllm
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
# prefill: per-chunk progress | decode: progress line every ~2.5 s while decoding | reqlog: one line per request | profile: spec round phases
export TQLLM_DEBUG=prefill,decode,reqlog,profile
exec ./build/tqserve \
  --model models/Qwen3.8-27B-TQ3_4S-v2.gguf \
  -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  --draft-vocab models/mtp-vocab/atx_65536.txt \
  --host 127.0.0.1 --port 8086 --context 262144 --np 1 --max-tokens 16384 \
  --reasoning on --reasoning-effort xhigh --kv tq3

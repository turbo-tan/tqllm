#!/usr/bin/env bash
# tqLLM serving PAW-27B-X3.1 (trellis) on GPU 2 (RTX 3090), 256K context, DFlash drafter + --draft-retrieve 32.
# OpenAI API on 127.0.0.1:8087. Managed by pm2 app tqllama (tqllama.config.js).
# Uses the RAM-disk copy in /dev/shm/x3 when present (fast load), else the HDD copy via models/.
set -euo pipefail
cd /home/green-gpu/tqllm
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
export TQLLM_DEBUG=prefill,decode,reqlog   # progress lines like truss: prefill %, decode tok/s + draft acceptance, one line per request
MODEL=/dev/shm/x3/PAW-27B-X3.1.gguf
[ -f "$MODEL" ] || MODEL=models/PAW-27B-X3.1.gguf
exec ./build/tqserve \
  --model "$MODEL" \
  -md models/drafters/df2_stock_q4km.gguf --draft-retrieve 32 \
  --host 127.0.0.1 --port 8087 --context 262144 --np 1 --max-tokens 16384 \
  --reasoning on --reasoning-effort xhigh --kv tq3 -ub 4096 \
  --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.0 \
  --prompt-cache on --reasoning-budget 12288
# sampling: Qwen thinking settings when the client sends none (greedy thinking looped to max_tokens);
# presence penalty 1.0 (Qwen suggests 1.5; lower because the agent writes code) breaks phrase-level thinking loops
# prompt cache: an agent step re-reads only its new tokens, not the whole conversation (57K = 42 s each step)
# reasoning budget: thinking is closed at 12288 tokens (24576 let the agent circle for 25K tokens) and 8192 more are kept for the answer (no "response incomplete")

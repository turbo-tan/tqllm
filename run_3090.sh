#!/bin/bash
# tqLLM on the idle RTX 3090 (GPU1), port 8086. 262,144 context with q4/q4 KV.
cd /mnt/data_2t_3/github_repos/tqllm
M=models/Qwen3.8-27B-TQ3_4S-v2.gguf
D=models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
V=models/mtp-vocab/atx_65536.txt
for f in "$M" "$D" "$V"; do [ -f "$f" ] || { echo "missing $f - run: python3 get_models.py"; exit 1; }; done
CUDA_VISIBLE_DEVICES=1 exec ./build/tqserve \
  --model "$M" -md "$D" --draft-vocab "$V" \
  --context 262144 --kv q4/q4 --prompt-cache on \
  --reasoning on --max-tokens 4096 --host 0.0.0.0 --port 8086

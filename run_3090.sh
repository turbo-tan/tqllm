#!/usr/bin/env bash
# Serve one Qwen3.8-27B file on a single 24 GB card (RTX 3090 / 4090) with the
# DFlash2 drafter. Every path is relative to this script, so it runs from any
# checkout on any machine.
#
#   ./run_3090.sh                          TQ3_4S-v2, GPU 0, port 8086, 262K context
#   GPU=1 ./run_3090.sh                    choose the card
#   PORT=8090 ./run_3090.sh                choose the port
#   CONTEXT=131072 ./run_3090.sh           if weights + KV do not fit in 24 GB
#   KV=tq3 ./run_3090.sh                  KV cache format: q4/q4 (default), tq3, q8, bf16
#   MODEL=models/Qwen3.8-27B-TQ_L.gguf TEMPLATE=models/chat_template_v22.jinja ./run_3090.sh
#                                          any other file; TEMPLATE is required for
#                                          the TQ_XS/S/M/L builds (unfixed template)
#
# Env: MODEL DRAFTER DRAFT_VOCAB TEMPLATE GPU PORT CONTEXT KV MAX_TOKENS
# Weights:  python3 get_models.py --type TQ3_4S-v2      Health: curl -s localhost:8086/health
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-models/Qwen3.8-27B-TQ3_4S-v2.gguf}"
DRAFTER="${DRAFTER:-models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf}"
DRAFT_VOCAB="${DRAFT_VOCAB:-models/mtp-vocab/atx_65536.txt}"
TEMPLATE="${TEMPLATE:-}"                       # set for builds with the unfixed chat template
GPU="${GPU:-0}"
PORT="${PORT:-8086}"
CONTEXT="${CONTEXT:-262144}"
KV="${KV:-q4/q4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

[ -x build/tqserve ] || { echo "build/tqserve missing: cmake --build build -j$(nproc)" >&2; exit 1; }
for f in "$MODEL" "$DRAFTER" "$DRAFT_VOCAB"; do
  [ -f "$f" ] || { echo "missing $f - fetch it with: python3 get_models.py --type TQ3_4S-v2" >&2; exit 1; }
done

args=(--model "$MODEL" -md "$DRAFTER" --draft-vocab "$DRAFT_VOCAB"
      --context "$CONTEXT" --kv "$KV" --prompt-cache on
      --reasoning on --max-tokens "$MAX_TOKENS" --host 0.0.0.0 --port "$PORT")
[ -n "$TEMPLATE" ] && args+=(--chat-template-file "$TEMPLATE")

echo "tqserve: $(basename "$MODEL") on GPU $GPU, port $PORT, context $CONTEXT, kv $KV"
exec env CUDA_VISIBLE_DEVICES="$GPU" ./build/tqserve "${args[@]}"

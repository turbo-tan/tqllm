#!/usr/bin/env bash
# run_3090.sh preset for the TQ_L build (TQ4 body, 4.31 bpw, 14.72 GB, closest
# to BF16). Sets MODEL, applies the fixed chat template when present, then
# hands off to run_3090.sh - which owns all the serving logic and env vars.
#
#   ./run_3090_TQ_L.sh                    TQ_L, GPU 0, port 8086, 262K context
#   GPU=1 ./run_3090_TQ_L.sh              choose the card
#   CONTEXT=131072 ./run_3090_TQ_L.sh     if 262K does not fit (TQ_L is 230 MB
#                                         heavier than TQ3_4S-v2)
#   TEMPLATE= ./run_3090_TQ_L.sh          use the file's own official template
#   MODEL=models/Qwen3.8-27B-TQ_M.gguf ./run_3090_TQ_L.sh    any TQ_* build
#
# Weights:  python3 get_models.py --type TQ_L
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export MODEL="${MODEL:-models/Qwen3.8-27B-TQ_L.gguf}"
if [ -z "${TEMPLATE:-}" ]; then
  [ -f models/chat_template_v22.jinja ] && TEMPLATE=models/chat_template_v22.jinja || TEMPLATE=""
fi
export TEMPLATE
exec ./run_3090.sh "$@"

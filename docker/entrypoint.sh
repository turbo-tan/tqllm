#!/bin/sh
set -eu
model=${MODEL_FILE:-/models/model.gguf}
if [ ! -r "$model" ]; then
    echo "Required model missing: $model" >&2
    exit 2
fi
# GGUF models carry their tokenizer; TOKENIZER_FILE only overrides it (or is required for non-GGUF checkpoints).
if [ -n "${TOKENIZER_FILE:-}" ]; then
    if [ ! -r "$TOKENIZER_FILE" ]; then
        echo "Tokenizer missing: $TOKENIZER_FILE" >&2
        exit 2
    fi
    set -- --tokenizer "$TOKENIZER_FILE" "$@"
fi
# Optional container settings use the same CLI options as native launches.
if [ -n "${DRAFT_MODEL_FILE:-}" ]; then
    if [ ! -r "$DRAFT_MODEL_FILE" ]; then
        echo "Draft model missing: $DRAFT_MODEL_FILE" >&2
        exit 2
    fi
    set -- -md "$DRAFT_MODEL_FILE" "$@"
fi
if [ -n "${DRAFT_VOCAB_FILE:-}" ]; then
    if [ ! -r "$DRAFT_VOCAB_FILE" ]; then
        echo "Draft vocabulary missing: $DRAFT_VOCAB_FILE" >&2
        exit 2
    fi
    set -- --draft-vocab "$DRAFT_VOCAB_FILE" "$@"
fi
if [ -n "${KV_FORMAT:-}" ]; then
    set -- --kv "$KV_FORMAT" "$@"
fi
exec tqserve --model "$model" --host 0.0.0.0 --port 8086 \
    --context "${CONTEXT:-16384}" --np "${PARALLEL:-1}" --max-tokens "${MAX_TOKENS:-1024}" \
    --reasoning "${REASONING:-off}" "$@"

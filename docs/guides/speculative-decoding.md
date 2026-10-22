# DFlash2 and native MTP

[Documentation](../README.md) / [Project home](../../README.md)

## Fast decoding: about 250 tok/s on an RTX 3090

tqLLM speculates with a DFlash2 draft model: the drafter proposes several tokens, the 27B model checks them in one pass,
and commits the ones it would have produced itself. Greedy output is identical, token for token, to decoding without
the drafter: plain decoding and every verify (tree, chain, MTP) share one attention arithmetic, checked by
`test-tree`, `test-dflash-exact` and `test-native-mtp` and on long prompts up to 51K tokens. Download into `./models`:

| file | Hugging Face repo |
|---|---|
| `Qwen3.8-27B-TQ3_4S-v2.gguf`, the 27B model (14.5 GB) | `YTan2000/Qwen3.8-27B-TQ3_4S` |
| `drafters/Qwen3.8-27B-DFlash2-TQ3.gguf`, the DFlash2 drafter (0.79 GB) | `YTan2000/Qwen3.8-27B-TQ` |
| `mtp-vocab/atx_65536.txt`, the drafter's token list | `YTan2000/Qwen3.8-27B-TQ` |

```sh
hf download YTan2000/Qwen3.8-27B-TQ3_4S Qwen3.8-27B-TQ3_4S-v2.gguf --local-dir models
hf download YTan2000/Qwen3.8-27B-TQ drafters/Qwen3.8-27B-DFlash2-TQ3.gguf mtp-vocab/atx_65536.txt --local-dir models
```

RTX 3090, 32K context, thinking on:

```sh
docker run --rm --gpus all -p 127.0.0.1:8086:8086 -v "$PWD/models:/models:ro" \
  -e MODEL_FILE=/models/Qwen3.8-27B-TQ3_4S-v2.gguf \
  -e DRAFT_MODEL_FILE=/models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
  -e DRAFT_VOCAB_FILE=/models/mtp-vocab/atx_65536.txt \
  -e CONTEXT=32768 -e REASONING=on -e MAX_TOKENS=4096 \
  ghcr.io/turbo-tan/tqllm:1.2.3
```

With Compose, set the same names: `DRAFT_NAME=drafters/Qwen3.8-27B-DFlash2-TQ3.gguf
VOCAB_NAME=mtp-vocab/atx_65536.txt MODEL_NAME=Qwen3.8-27B-TQ3_4S-v2.gguf CONTEXT=32768 REASONING=on docker compose up`.

What to expect: on the LocalMaxxing reasoning prompt (32K context, 320 output tokens) an RTX 3090 runs at 240-270 tok/s
with this drafter (TQ3 and Q4_K_M builds measure the same). The server reports tok/s for every
request. Speed depends on how often the drafter's guesses are accepted, so it varies with the text: predictable text
(reasoning, prose) runs fastest, code and long contexts slower.

Long prompts: `--draft-retrieve 32` lets the drafter also read 32 far passages of the prompt (RTX 3090, 64K context:
31K / 40K / 51K-token prompts 158 / 143 / 137 -> 176 / 156 / 148 tok/s, short prompts -2 %). It keeps the drafter's
cache for the whole context (about 20 KB per token), so it is off by default.

No drafter file: `--mtp` drafts with the model's own prediction head (one user, greedy; same output as plain decoding):
155 tok/s at 1.6K and 125 tok/s at 51K tokens on an RTX 3090, against 57 and 47 without drafting.

### Recommended settings

| GPU | model | `CONTEXT` | `KV_FORMAT` | drafter | measured decode (tok/s) |
|---|---|---|---|---|---|
| RTX 3090 / 4090 (24 GB) | `Qwen3.8-27B-TQ3_4S-v2.gguf` | 32768 | default (bf16) | yes | 131-155 on summaries, 256 on LocalMaxxing |
| RTX 5060 Ti / 4060 Ti (16 GB) | `Qwen3.8-27B-TQ3_4S-v2.gguf` | 16384 | `tq3` | yes | 77-95 on summaries |
| GB10 / DGX Spark (128 GB) | `Qwen3.8-27B-TQ3_4S-v2.gguf` | 131072 or more | default (bf16) | yes | 43-54 on summaries, 79 on LocalMaxxing |

- `KV_FORMAT=tq3` stores the KV cache at about 3 bits per value instead of 16. Use it when memory is tight or the context
  is long; on 24 GB at 32K the default is fine.
- The memory fit runs at start: if `CONTEXT` does not fit, tqserve moves the token embeddings and LM head to host
  memory first, then shrinks the context, and logs each step. Pass `-fitt N` to keep N MiB of the GPU free (default
  768, 256 on WSL).
- `REASONING=on` for thinking tasks, with `MAX_TOKENS` large enough for the thinking plus the answer (4096 or more).
- `PARALLEL=N` serves N requests at once; the context is split evenly between them.

# Choose a setup for your hardware

[Documentation](../README.md) / [Project home](../../README.md)

Find your GPU memory, pick the row for what you do most, and start `tqserve` with that line. Everything else (context,
buffers, what moves to system RAM) the memory fit sizes at start and logs. Numbers are decode tokens per second for one
request, measured on the card named, with the build and date given under each table.

Files: [YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) (TQ_XS / TQ_S / TQ_M / TQ_L, the `qsa/`
files and `drafters/Qwen3.8-27B-DFlash2-TQ3.gguf`), [YTan2000/Qwen3.8-Flash-Next-tqLLM](https://huggingface.co/YTan2000/Qwen3.8-Flash-Next-tqLLM).

## Quick pick

| Your hardware | Best for chats and documents | Best for very long documents (100K+) |
|---|---|---|
| 16 GB NVIDIA (RTX 5060 Ti, 4060 Ti 16 GB, 4080) | 27B TQ_XS + DFlash2, `--kv tq3` | 27B TQ_XS-QSA + DFlash2, `--kv tq3 --qsa-from 49152` |
| 24 GB NVIDIA (RTX 3090, 4090) | 27B TQ3_4S-v2 + DFlash2 | 27B TQ_L-QSA + DFlash2, `--kv turbo4 --qsa-from 49152` |
| 24 GB NVIDIA + 64 GB RAM, wants a bigger model | Qwen3.8-Flash-Next (125B MoE) | Flash-Next |
| 32 GB NVIDIA (RTX 5090) | 27B TQ_L + DFlash2, or Flash-Next | Flash-Next |
| DGX Spark / GB10 (128 GB) | 27B TQ3_4S-v2 + DFlash2 | 27B, `--context 131072` or more |
| Mac (Apple Silicon) | see the [Mac guide](mac.md) | see the [Mac guide](mac.md) |

The DFlash2 drafter (`-md Qwen3.8-27B-DFlash2-TQ3.gguf`) is the biggest single speed-up on every card: 3-4x on chats.
Its output is the same as without it (greedy).

## 16 GB (RTX 5060 Ti 16 GB)

```sh
hf download YTan2000/Qwen3.8-27B-TQ Qwen3.8-27B-TQ_XS.gguf qsa/Qwen3.8-27B-TQ_XS-QSA.gguf \
  drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --local-dir models
# chats, code, documents up to about 64K tokens
tqserve --model models/Qwen3.8-27B-TQ_XS.gguf -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --kv tq3 --context 65536
# documents of 100K-200K tokens (approximate attention past 48K keys)
tqserve --model models/qsa/Qwen3.8-27B-TQ_XS-QSA.gguf -md models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf --kv tq3 \
  --context 204800 --qsa-from 49152
```

| setup | chats | 4K document | 16K document | 48K document | 123K document | 200K document |
|---|---|---|---|---|---|---|
| TQ_XS + DFlash2, `--kv tq3`, context 64K | 80-152 (mean 121) | 78.1 | 90.8 | 76.7 | | |
| TQ_XS-QSA + DFlash2, `--kv tq3`, context 200K, QSA on | | | | | 124.1 | 109.5 |
| same, QSA off (exact) | | | | | 100.4 | 35.4 |
| TQ_XS + DFlash2, `--kv tq3`, context 200K (no QSA file) | | | | | | 43.7 |
| TQ_M + DFlash2, `--kv tq3`, context 64K | 76-146 (mean 110) | 82.1 | 75.9 | 64.5 | | |
| TQ_M, no drafter, `--kv tq3`, context 64K | 30.8 | 30.5 | 29.7 | 27.7 | | |
| TQ_M, no drafter, `-ctk q8_0 -ctv q8_0`, context 64K | 30.8 | 30.3 | 29.4 | 8.1 | | |

Time to first token on this card: 3.2 s at 4K, 12.2 s at 16K, 44 s at 48K, 188 s at 123K, 364 s at 200K (prefill about
1,150 tok/s up to 16K, 550 tok/s at 200K).

- Use `--kv tq3` on 16 GB. With `q8_0` K/V the fit keeps most of a long conversation in system RAM and reads it back
  every token: a 48K document decodes at 8 tok/s on TQ_M (5 on TQ_L) instead of 28.
- TQ_XS leaves the most room for the drafter and the context; TQ_S / TQ_M / TQ_L are a little better quality and a little
  slower (chats 32.0 / 30.8 / 28.0 tok/s without a drafter; TQ_L + DFlash2 does not fit on 16 GB).
- QSA is approximate: past 48K keys each answer token reads only the key blocks the file's indexer picks. At 123K its
  answers agree with the exact path on 87-92 % of tokens (RTX 3090 check); use `--qsa-from off` when every token must
  match.

Measured 11-17 on an RTX 5060 Ti 16 GB (Linux, 31 GB RAM), a tqLLM 1.7.0 development build, one request, greedy. Chats: 5
short questions, 512 tokens; documents: WikiText-2 cut to length, 64-256 tokens out; 123K / 200K: the C++ ladder prompt,
up to 512 tokens out.

## 24 GB (RTX 3090 / 4090)

```sh
tqserve --model Qwen3.8-27B-TQ3_4S-v2.gguf -md Qwen3.8-27B-DFlash2-TQ3.gguf --context 32768
tqserve --model Qwen3.8-27B-TQ_L-QSA.gguf -md Qwen3.8-27B-DFlash2-TQ3.gguf --kv turbo4 --context 262144 --qsa-from 49152
```

| setup | 2K | 8K | 32K | 128K | 256K |
|---|---|---|---|---|---|
| TQ3_4S-v2 + DFlash2, context 32K (prefill 2,150-2,430 tok/s) | 155.2 | 149.9 | 131.0 | | |
| TQ3_4S-v2, no drafter | 55.5 | 54.5 | 50.6 | | |
| TQ_L-QSA + DFlash2, `--kv turbo4`, QSA on | | | | 182.2 | 170.5 |
| same, QSA off (exact) | | | | 143.6 | 119.5 |

Reasoning-style prompts accept more drafts: 240-270 tok/s on the LocalMaxxing prompt with the first setup. Sources:
[Benchmarks](../BENCHMARKS.md) (engine de8e3c8) and the 1.6.0 release notes (QSA rows, mean of 3 prompts).

### Flash-Next on 24 GB + 64 GB RAM

The 125B MoE keeps its experts in system RAM and a cache of them on the GPU. RTX 3090: short chats 91.6 tok/s, decode
76.4 / 85.1 / 77.1 tok/s after 32K / 64K / 128K prompts, first token 15 s at 32K and 57 s at 128K
([Benchmarks](../BENCHMARKS.md)). Setup: the model card above.

## 32 GB (RTX 5090)

27B TQ_L + DFlash2: chat 227-244, JSON 481, 32K 242, 64K 200 tok/s. Flash-Next: chats 137-168, 32K / 64K 134, 128K
83-127 tok/s, prefill 4,000-5,200 tok/s. Details: [RTX 5090](rtx-5090.md).

## DGX Spark / GB10

27B TQ3_4S-v2 + DFlash2, context 131072 or more: 43-54 tok/s on summaries, 79 on the LocalMaxxing prompt
([Speculative decoding](speculative-decoding.md)). Flash-Next NVFP4 and two linked Sparks: [Multi-GPU](multi-gpu.md).
Bigger MoE models on one GB10: [guide](gb10-big-moe.md).

## 12 GB and smaller

The 27B tiers need about 12 GB for the weights alone, so a 12 GB card runs short contexts only; serve one or two users
(`--np 2`: a 4-user copy-engine fault on 12 GB is still open). No measured table yet.

## Check your own card

Every request's prefill and decode rate is in the monitor at `http://localhost:8086/`, and the server logs what the
memory fit chose at start (`tqllm: fit: ...` and `tqllm: KV: ... positions in host memory`). If the log shows many K/V
positions in host memory, switch to `--kv tq3` or a smaller context.

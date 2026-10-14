# Changelog

All notable changes to tqLLM. Images: `ytan2000/tqllm:<version>` and `ghcr.io/turbo-tan/tqllm:<version>` (multi-arch:
x86-64 RTX 30 / 40 / 50 and arm64 GB10 / DGX Spark). `latest` is the newest release.

## Unreleased (next: 1.0.1)

### Added
- MoE: `--moe-cache-mb C` keeps recently used host-memory experts in a VRAM cache, managed on the GPU inside the decode
  graph. On a 16 GB budget, MiniCPM-V 4.7 35B-A3B Q5_K_M goes from 71.9 / 112.5 / 54.9 to 96.2 / 146.3 / 85.9 tok/s
  (RTX 3090 capped to 16 GB; llama.cpp `-ncmoe 19`: 62.6 / 62.8 / 62.8). Output identical.
- In progress: native OpenAI Responses API (`/v1/responses`) so Codex connects directly; native Anthropic Messages API
  (`/v1/messages`) for Claude Code.

### Changed
- MoE decode runs the shared expert inside the routed expert launches. MiniCPM-V 4.7 35B-A3B on one RTX 3090:
  IQ3_S 207.8 -> 220.1 tok/s, Q5_K_M (2000 MiB experts in host memory) 176.5 -> 187.1 tok/s. Output identical.

### Known issues
- `tqllm launch codex` fails with Codex CLI 0.16x: Codex now requires the Responses API (`wire_api = "responses"`).
  Fixed by the native `/v1/responses` above.

## 1.0.0 - 2026-10-14

First public release.

### Models
- Qwen3.8 / Qwen3.5 dense models (hybrid attention + Gated DeltaNet), e.g. Qwen3.8-27B in the TQ release files
  (`YTan2000/Qwen3.8-27B-TQ`: TQ_XS 11.85 GB, TQ_S 12.55 GB, TQ_M 13.14 GB, TQ_L 14.72 GB).
- Mixture-of-experts `qwen35moe` models, e.g. MiniCPM-V 4.7 35B-A3B (text). `--moe-host-mb N` keeps the least-routed
  experts in pinned system RAM when the file is larger than the GPU, placed from a routing profile
  (`TQLLM_MOE_PROFILE`); host experts run on a side stream, staged through VRAM.
- GGUF tensor types: TQ3, TQ4, TQ3_4S, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q4_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, Q2_K, IQ2_S,
  IQ2_XS, IQ2_XXS, IQ1_S, F16, BF16 (Q4_K_M, Q5_K_M, Q6_K, Q8_0 and 2-bit Unsloth UD files load).

### Speculative decoding (exact)
- DFlash2 drafter with tree verify, chain verify (`--no-tree`) and the model's own MTP head (`--mtp`): greedy output is
  identical to plain decoding in every mode (one attention arithmetic for plain and verify blocks).
- DSpark drafters (openbmb) for MiniCPM-V 4.7, chain verify by default.
- `--draft-retrieve N`: the drafter also reads far passages of long prompts.

### Memory and context
- Automatic memory fit: picks the KV format, context and prefill buffers for the GPU and prints what it chose.
- Compressed KV caches: tq3, turbo2/3/4, trellis-coded tcq2/tcq3, q4 and more (`--list-kv`).
- TqCache: the oldest KV positions live in system RAM when the context does not fit on the GPU, with identical output.
- Up to 256K context on one RTX 3090 (measured: 140 tok/s at 256K with TQ_L, tq3 KV, DFlash2); YaRN RoPE scaling
  (`--rope-scaling yarn`) beyond the trained length.

### Speed
- Prefill GEMMs in int8 on RTX 30 (Ampere) and FP8 on RTX 40 / 50 and GB10; prefill padding rows zeroed so the first
  token is stable across server starts.
- Qwen3.8-27B on one RTX 3090: 273.1 tok/s (LocalMaxxing reasoning-v1, approved) / 199.5 (code-v1).
- MiniCPM-V 4.7 35B-A3B on one RTX 3090: IQ3_S 202-208 tok/s vs llama.cpp 136-147; Q5_K_M 168-182 vs 123-125
  (`-ncmoe 2`). DGX Spark with DSpark: 145-165 vs 106-115 on code prompts, 88 vs 72 on prose.

### Server and tools
- OpenAI-compatible HTTP server: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/admin/slots`;
  tool calls, reasoning on / off with effort and budget, Jinja chat templates from the GGUF.
- `--np N` parallel slots with batched verify.
- `tools/launch/tqllm.py`: `launch opencode | claude` starts the server (Docker or native) and the agent; model
  catalog with Hugging Face download and sha256 check.
- Docker image for x86-64 and arm64 in one tag; Linux and Windows (WSL 2).

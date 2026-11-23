# Inkling-2-Mini on a Mac (MLX / Metal)

Thinking Machines' Inkling-2-Mini (26.4B parameters, 3.4B active, a looped MoE: 24 layers with layers 6-17 run three
times, so 48 layer visits a token) runs on Apple Silicon through tqLLM's MLX backend, from the original checkpoint
(NVFP4 / MXFP8 experts, BF16 dense weights). Code: `src/models/inkling2/loop/inkling2_mlx*.{cpp,hpp}` and
`inkling2_mlx_ops.hpp` (the Metal kernels), the engine in `src/backends/metal/engine_inkling2_mlx.cpp`.

## Run

```sh
tqserve --model <Inkling-2-Mini folder> --jinja --np 1 --context 32768          # exact, plain decode
tqserve --model <folder> --jinja --np 1 --context 32768 --mtp                    # + DSpark drafter (mtp.safetensors)
tqserve --model <folder> --jinja --np 1 --context 32768 --mtp --exact off        # 8-bit attention + LM head
tqserve --model <folder> --jinja --np 1 --mtp -md dspark-Inkling-2-Mini-Q8_0.gguf  # drafter from Unsloth's GGUF
```

- `--exact on` (the default) keeps the checkpoint's numerics. `--exact off` stores the attention projections (q, k, v,
  output, output gate) and the LM head in 8 bits (affine, group 64); greedy tokens still come from the BF16 head.
- `--mtp` drafts with the model's own DSpark head and verifies each draft block in one forward. It is exact against the
  verify path: the speculative output is the greedy decode of the verify forward (MLX's multi-row products are not
  bit-identical to its matrix-vector ones, so speculative and plain output can differ at near-ties). The engine
  chooses the drafts per round from measured acceptance and round time; `--draft-max N` fixes it.
- Run with `MLX_ENABLE_TF32=0` (the gates below were measured with it).
- Memory: ~19 GB of weights, +0.35 GB for the bounded greedy head, +0.4 GB with `--mtp`. On 48 GB, 256K of context
  peaks at 30.7 GB. The verify experts' matrix-unit copy (+~11 GB) is only made when it fits within 70 % of memory.

## Speed (Apple M5 Pro, 48 GB, 2026-11-23, commit 5bb112d)

Same machine and session as MTPLX 2.13.1 (`youssofal/MTPLX` cf39588), chat API, greedy, reasoning effort none.

Prefill, time to first token on long-document prompts (tok/s):

| | 1K | 8K | 32K | 64K | 256K (check tool) |
|---|---|---|---|---|---|
| tqLLM | **1428** | **1694** | **1554** | **1367** | 789 |
| MTPLX 2.13.1 (Optimized-Quality) | 1053 | 1507 | 1458 | 1291 | |

Decode, 5 chat / code prompts x 256 tokens, end to end (whole request), mean (median) tok/s:

| | plain | with drafter |
|---|---|---|
| tqLLM exact | **40.4** (40.5) | **53.3** (57.6) |
| MTPLX Optimized-Quality | 35.3 (35.5) | 48.5 (50.3) |
| tqLLM `--exact off` | **48.3** (48.3) | 57.1 (**61.1**) |
| MTPLX Optimized-Speed (8-bit dense) | 45.3 (45.7) | **58.8** (57.9) |

Long context, exact (check tool): plain decode 41 / 39 / 33 / 27 / 20 / 13 tok/s at 1K / 8K / 32K / 64K / 128K / 256K;
`--mtp` 45 tok/s at 128K and 29 at 256K. Against the starting point (97f28b97): plain decode 10.8 -> 40.4 tok/s,
prefill 132 -> 1428 tok/s (1K).

Open gap: 8-bit speculative decoding on chat text, where the drafter's acceptance is lowest (MTPLX ~3 % ahead on the
mean, tqLLM ahead on the median). Rounds are GPU-bound; see "Next" below.

## Quality gates (5bb112d)

`tools/inkling2_mlx_check.cpp` against the HF reference (golden data):

| gate | exact | `--exact off` |
|---|---|---|
| argmax flips, 5 prompts x 64 (limit 4) | 2 / 320 | 3 / 320 |
| full-vocabulary KLD vs BF16, 8,768 positions | 0.0024443, top-1 98.31 %, PPL 5.669 | 0.0027348, top-1 97.92 %, PPL 5.663 |
| speculative vs reference (128 tokens) | agrees at least as long as plain on every prompt | |

`--exact off` sits just under its 98 % top-1 line (KLD and PPL pass); it has been at 97.9-98.0 % since d3158e9.
`tests/mlx_inkling2.cpp` (`test-mlx-inkling2`) checks every custom kernel against MLX-op or host references.

```sh
M=<model folder>
MLX_ENABLE_TF32=0 build/inkling2-mlx-check $M $M/golden/bin --flips --chunk 64
MLX_ENABLE_TF32=0 build/inkling2-mlx-check $M $M/golden/mt --mt bf16 --chunk 64 [--exact off]
MLX_ENABLE_TF32=0 build/inkling2-mlx-check $M $M/golden/bin --spec 0 --decode 128 --chunk 64
MLX_ENABLE_TF32=0 build/inkling2-mlx-check $M $M/golden/bin --bench 8192 --chunk 4096 --runs 2 [--spec 0] [--exact off]
```

## How it is fast

Decode (one token) is one lazily built graph a step, evaluated with all state, pipelined with the next step.
- Fused Metal kernels for the per-visit glue: depth attention (with the following RMSNorm), RMSNorm, L2 clip (with the
  log-position scale), MoE router, static convolutions (with the expert combine and the residual add), the attention
  mask. Row counts are kernel inputs, not template arguments, so a new chunk length builds no new kernels.
- Exact greedy head from a bounded 6-bit copy: every row is scored with the copy, and only rows whose error bound lets
  them win (about 2 a token) are recomputed in BF16; 0.35 GB read a token instead of 0.92 GB, same token.

Prefill (prompt chunks of 4096 rows, one graph a chunk):
- Attention on the matrix unit with the relative bias, mask and sink applied in place: no mask tensor, only the key
  tiles a row can see, the KV cache read through its strides (`inkling2_prompt_attention`).
- Key / value convolution with its clip or norm in one kernel, message gates (merge + norm) in one kernel, the
  expert combine reading the grouped products through their restore order.

Speculative decoding (DSpark, `--mtp`):
- Verify experts on the matrix unit from a fragment-ordered copy (each expert read once for the block's rows), the
  routed pairs sorted in one kernel, the rollback of rejected rows one kernel per convolution history.
- The drafts and the first verify visits are queued while the rest of the round is built.

## Next

- 2..8-row 8-bit products near the one-row bandwidth: MLX's small-matrix path costs ~2 ms a verify round over the
  one-row read (a scalar and a matrix-unit attempt were both slower than MLX so far).
- The drafter's head over the 200K vocabulary (~2.6 ms a round): a frequency shortlist would cut it.
- Deciding the round on the GPU (MTPLX's approach): estimated ~1 ms a round, a larger change.
- Long-context prompt attention: ~19 TFLOP/s, limited by staging K / V (64-key tiles need more than 32 KB).

Measurement scripts and full logs for the numbers above were kept with the benchmark workspace (`RESULTS.md`).

# Inkling-2-Mini

Thinking Machines' Inkling-2-Mini (26.4B parameters, 3.4B active; a looped mixture of experts: 24 layers with layers
6-17 run three times, so 48 layer visits a token). tqLLM loads the official checkpoint folder as it is (NVFP4 / MXFP8
experts, BF16 dense weights) on NVIDIA GPUs and on Apple Silicon. Output is exact by default: the engine reproduces the
reference model's numerics (KLD 0.0023 against the BF16 reference on an RTX 3090, 0.0024 on an M5 Pro).

## Download

```sh
pip install -U huggingface_hub
hf download thinkingmachines/Inkling-2-Mini --local-dir models/Inkling-2-Mini
```

## Run on NVIDIA (RTX 30 / 40 / 50 with 24 GB, DGX Spark / GB10)

```sh
docker run --rm --gpus all -p 127.0.0.1:8086:8086 -v "$PWD/models:/models:ro" --entrypoint tqserve \
  ghcr.io/turbo-tan/tqllm:1.7.0 --model /models/Inkling-2-Mini --jinja --host 0.0.0.0 --port 8086
```

- `--jinja` uses the model's own chat template (tool calls, the thinking split).
- `--mtp` drafts with the model's own MTP head (greedy requests).
- `--exact off` trades a little accuracy for speed: INT8 dense weights on every card, and on Blackwell / DGX Spark the
  prompt experts on the FP4 tensor cores (KLD 0.0096, top-1 96.0 % against BF16; the exact path is 0.0023 / 98.4 %).
- 16 GB cards: the 20 GB checkpoint does not fit, so part of the experts stream from system RAM each token; decode drops
  to about 23 tok/s at 8K on an RTX 5060 Ti. A smaller file for 16 GB cards is planned.

## Speed against other engines (NVIDIA, 2026-11-23, tqLLM 1.7.0)

One stream, plain decoding, same machine, one engine at a time. Strata 0.2.20 (strata-inkling 0.1.66) runs its SQ-20
pack, its bit-exact repack of the same weights, at its fastest K/V setting for each length. Each engine runs its own
benchmark prompt. tok/s.

DGX Spark (GB10), prompt processing, exact against exact:

| prompt | 8K | 32K | 64K | 128K | 256K |
|---|---|---|---|---|---|
| tqLLM | **3,028** | **2,781** | **2,438** | **1,997** | **1,451** |
| Strata 0.2.20 | 1,422 | 1,119 | 879 | 617 | 385 |
| ratio | 2.1x | 2.5x | 2.7x | 3.2x | 3.7x |

DGX Spark, answers (decode): tqLLM exact 32.2 / 28.6 / 24.9 at 8K / 32K / 64K, `--exact off` 39.5 / 34.2 / 29.2;
Strata 46.9 / 33.2 / 18.8. Strata leads short prompts; tqLLM leads from 32K (`--exact off`) and 64K (exact).
Against vLLM's Inkling image (recorded 11-21): answers 41.1 / 39.5 / 34.2 against 36.9 / 35.5 / 31.3 tok/s at 1K / 8K
/ 32K (`--exact off`), prompts 3,835 against 1,790 at 8K.

RTX 3090 24 GB, answers (two runs each, means):

| answers | 8K | 64K | 128K | 256K |
|---|---|---|---|---|
| tqLLM exact | 77.9 | **65.3** | **55.0** | **31.2** |
| tqLLM `--exact off` | 88.4 | **72.1** | **60.6** | **36.0** |
| Strata 0.2.20 | **93.4** | 34.9 | 6.4 * | 2.1 * |

\* On a 24 GB card Strata runs out of VRAM before 128K with its K/V on the card ("VRAM exhausted growing elastic
global KV cache": fp8 at about 85K tokens, k8v4 / k8v8 at about 115K); these rows keep its K/V in host RAM.
Prompts on the 3090: tqLLM 2,720 / 2,479 / 1,492 tok/s at 8K / 64K / 128K, Strata 1,500 / 1,234 / 928.
Against llama.cpp (IQ4_NL GGUF) on the 3090: answers 1.23-1.30x, prompts 1.3-4.1x.

## Quality

Full-vocabulary KLD and top-1 agreement against the BF16 reference, 8,768 positions of held-out text:

| path | KLD | top-1 | PPL |
|---|---|---|---|
| RTX 3090, exact (default) | 0.0023 | 98.35 % | 5.662 |
| RTX 3090, `--exact off` | 0.0029 | 98.06 % | |
| DGX Spark, `--exact off` (FP4 prompt experts) | 0.0096 | 96.04 % | 5.703 |
| M5 Pro, exact | 0.0024 | 98.31 % | 5.669 |

## Mac (MLX / Metal)

Thinking Machines' Inkling-2-Mini (26.4B parameters, 3.4B active, a looped MoE: 24 layers with layers 6-17 run three
times, so 48 layer visits a token) runs on Apple Silicon through tqLLM's MLX backend, from the original checkpoint
(NVFP4 / MXFP8 experts, BF16 dense weights). Code: `src/models/inkling2/loop/inkling2_mlx*.{cpp,hpp}` and
`inkling2_mlx_ops.hpp` (the Metal kernels), the engine in `src/backends/metal/engine_inkling2_mlx.cpp`.

### Run

```sh
tqserve --model <Inkling-2-Mini folder> --jinja --np 1 --context 32768          # exact, plain decode
tqserve --model <folder> --jinja --np 1 --context 32768 --mtp                    # + DSpark drafter (mtp.safetensors)
tqserve --model <folder> --jinja --np 1 --context 32768 --mtp --exact off        # 8-bit attention + LM head
tqserve --model <folder> --jinja --np 1 --mtp -md dspark-Inkling-2-Mini-Q8_0.gguf  # drafter from Unsloth's GGUF
tqserve --model <folder> --jinja --np 1 --context 32768 -md <Inkling-2-Mini-DFlash2 folder>  # DFlash2 drafter (incoai, Apache-2.0)
```

- `--exact on` (the default) keeps the checkpoint's numerics. `--exact off` stores the attention projections (q, k, v,
  output, output gate) and the LM head in 8 bits (affine, group 64); greedy tokens still come from the BF16 head.
- `--mtp` drafts with the model's own DSpark head; `-md FOLDER` with incoai's DFlash2 drafter (config.json +
  model.safetensors of `incoai/Inkling-2-Mini-DFlash2`: a five-layer block drafter fed by the same taps, ~0.9 more
  tokens a round). Each draft block is verified in one forward, and the speculative output is the plain greedy output
  bit for bit: a decode step and a verify block (1..8 rows) run the same row-invariant products
  (`src/models/inkling2/loop/inkling2_rows.hpp`: dense on the 8 x 8 matrix unit, routed experts read once per run of
  rows, attention over keys in absolute-position blocks), so row j of a block equals the j-th decode step
  (`inkling2-mlx-check --rowcheck 8`, `--spec 0`: spec == plain 5/5, exit 1 otherwise). The engine chooses the drafts
  per round from measured acceptance and round time; `--draft-max N` fixes it.
- Turn reuse: the state after a prompt's prefill is kept; a next request whose prompt starts with that prompt and
  continues at a message start (a chat follow-up) resumes there and prefills only the new messages, with the tokens a
  fresh server gives (prefill chunks restart at message starts past 1024; `inkling2-mlx-check --turns N`).
- Run with `MLX_ENABLE_TF32=0` (the gates below were measured with it).
- Memory: ~19 GB of weights, +0.35 GB for the bounded greedy head, +0.4 GB with `--mtp`. On 48 GB, 256K of context
  peaks at 30.7 GB. The DFlash2 drafter adds ~0.6 GB (8-bit projections + its selector tables).

### Speed (Apple M5 Pro, 48 GB, 2026-11-23, commit 5bb112d)

These rows predate the row-invariant products and DFlash2 (1.7.0's MLX products, speculative output not equal to
plain). Same-session card A/B of the row-invariant build (2026-11-24, commit d73914f6): exact + DFlash2 57.1 tok/s
(DSpark on 1.7.0's products 52.1), `--exact off` + DFlash2 66.8 (56.3), exact plain 36.4 (40.2).

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

### Quality gates (5bb112d)

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

### How it is fast

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

### Next

- 2..8-row 8-bit products near the one-row bandwidth: MLX's small-matrix path costs ~2 ms a verify round over the
  one-row read (a scalar and a matrix-unit attempt were both slower than MLX so far).
- The drafter's head over the 200K vocabulary (~2.6 ms a round): a frequency shortlist would cut it.
- Deciding the round on the GPU (MTPLX's approach): estimated ~1 ms a round, a larger change.
- Long-context prompt attention: ~19 TFLOP/s, limited by staging K / V (64-key tiles need more than 32 KB).

Measurement scripts and full logs for the numbers above were kept with the benchmark workspace (`RESULTS.md`).

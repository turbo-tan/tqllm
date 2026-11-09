# DSpine drafter (experimental)

This branch adds Tencent's independent-write DSpine variant to the DFlash loader, CUDA and
Metal paths. The local CUDA exactness and performance gates passed on an RTX 3090 on
2026-11-08 (results below). The Metal runtime gate and independent review are still pending.
The 1.5.0 candidate is not changed until those checks and review pass.

The published [Qwen3-8B-DSpine-b16 checkpoint](https://huggingface.co/tencent/Qwen3-8B-DSpine-b16)
is a drafter for **Qwen/Qwen3-8B**, not a standalone language model or a Qwen3.8-27B drafter.
It borrows the matching target's embedding and output head. Architecture, hidden width,
vocabulary, layer count and feature taps are validated before loading its bridge tensors.
A compatible trained checkpoint is required for every other target.

## Conversion

Use an existing Python environment with NumPy and gguf installed. The converter reads
safetensors bytes directly; it does not execute checkpoint-provided Python code.
Check disk space before downloading: the drafter is about 2.5 GB, and conversion needs a
second copy. The Qwen3-8B target is separate.

```sh
hf download tencent/Qwen3-8B-DSpine-b16 --local-dir models/dspine-8b
python3 tools/convert/dflash/dspine_hf_to_gguf.py models/dspine-8b models/dspine-8b-bf16.gguf
```

For the Qwen3.8-27B target there is `tencent/Qwen3.8-27B-DSpine-b16` (published 2026-11-07).
The engine guards drafter/target pairing with `dflash.target_architecture`; the checkpoint's
`model_type` names the drafter's own block type, so pass the target's engine arch explicitly:

```sh
hf download tencent/Qwen3.8-27B-DSpine-b16 --local-dir models/dspine-27b
python3 tools/convert/dflash/dspine_hf_to_gguf.py models/dspine-27b models/dspine-27b-bf16.gguf --target-arch qwen35
```

The converter checks the format markers, disabled terminal projection and every source
tensor's shape and dtype. Stacked per-layer reads/writes become `blk.N.dspine_read.weight`
and `blk.N.dspine_write.weight`. The codebook and gate tensors use the upstream
`dspine_codebook`, `dspine_gate`, `dspine_pred_gate` and `dspine_prev` names. Bridge weights
stay BF16; the NAX drafter converter also leaves them untouched.

## Candidate validation command

After building this branch, the following is the initial CUDA smoke configuration. It is
a validation recipe, not a passed benchmark or a published release command:

```sh
build/tqserve --model models/Qwen3-8B-Q4_K_M.gguf \
  -md models/dspine-8b-bf16.gguf --kv bf16 --context 8192 --draft-max 15 --np 1
```

CUDA refinement currently reads the target's Q5_K/Q6_K output-head copy, so the initial
target is the standard Q4_K_M file with a Q6_K head. Qwen3 target support uses BF16 KV and
chain verification; arbitrary tree masks for its 128-dimensional CUDA attention are not
implemented. The existing multi-slot scheduler can serve several chain requests. This
RC builds with 12 verify columns, so `--draft-max 15` is capped at 11 proposals plus the
anchor; the trained draft block still has 16 rows. A 16-column build can verify 15 proposals.
This experimental Qwen3 registration does not use automatic memory fitting: choose an explicit
context that fits. Registered Qwen3 engines report that the exactness gate is pending.

On Metal, Qwen3 also uses BF16 KV. The drafter keeps full context in a context-sized ring,
with causal masks within the draft block; the existing device tree consumes dense conditional
scores. The final refinement currently evaluates the shared proposal head for every refined
row. Its runtime and memory cost have not been measured. Metal continues to serve one slot.

## Local CUDA gate results (2026-11-08, RTX 3090, .188)

Build: Release sm_86, 12 verify columns; `tqserve` SHA256
`62e4bf40ae74fcd6f4c38fb5b4bd4311c77111ae22c2df5912c4630c269d21fd`.
Target `Qwen3-8B-Q4_K_M.gguf` SHA256 `120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4`,
converted drafter SHA256 `f693104ed1d105d451c22bf0b4dd810e786f5ad81382324499b31dc04e698b9a`.
Evidence: gate, ctest, cache and benchmark JSON + logs kept in the task work directory
recorded in the steering plan (PLAN-dspine-codex-20261108).

- Exactness gate: 20/20 comparisons with identical greedy token IDs vs target-only, at
  `--draft-max 15` (capped to 11 proposals): four prompts, repeated requests on a live server,
  two concurrent slots, a 512-token context boundary and a 12-row micro-batch (`-ub 12`).
- Full CTest: 158 passed, 0 failed, 1 hardware skip (`int8_features`); includes the CUDA
  bridge/refinement oracle and the exact causal 128-wide attention regression.
- Prompt cache: reused-prefix hit on both arms (prefill about 526 ms -> 65 ms), token IDs equal.
- Sampling smoke (T=0.8, seed 42, two concurrent slots): valid completions.
- Quiet ABBA decode benchmark, `--draft-max 7`, median-stable repeats, exact token IDs:
  prose 331 vs 118 tok/s, code 548-554 vs 117, arithmetic 394 vs 117, long (1025-token prompt)
  147 vs 115 (DSpine vs target-only). Acceptance on the long prompt is low (63/259); the
  short-prompt arms accept 100-110 of 133-217 proposals.

These are single-machine local results, not a release claim. Metal runtime numbers and an
independent review are still required before incorporation.

### Qwen3.8-27B same-target results (2026-11-08, RTX 3090)

Target `Qwen3.8-27B-TQ3_4S-v2.gguf`, drafter `tencent/Qwen3.8-27B-DSpine-b16` converted with
`--target-arch qwen35` (SHA256 `dc86c705d283f451e5bceeff7714f8c1d44e280eed20993c47e571b0c2add7a0`).
Same binary and harness as above.

- Exactness gate: 20/20 identical greedy token IDs vs target-only (np2 needs `--context 8192`
  so each slot keeps 4096 positions; the 27B + BF16 drafter does not fit np2 at 4096 total).
- Quiet ABBA decode, `--draft-max 7`, DSpine vs the release DFlash2-TQ3 drafter on the same
  target: prose 96-100 vs 124-125 tok/s, code 177-179 vs 238, arithmetic 125-131 vs 154-174,
  long (1.1K prompt) 80-82 vs 86-92. At the 11-proposal cap DSpine measures 104 / 187 / 134 / 82.
  DFlash2 is currently faster on this target; DSpine's trained 16-row block is capped by the
  12-column verify width, and its exact-predecessor refinement costs more per step.
- Cross-server note: the load-time prefill crossover calibration occasionally selects the
  batched instead of the native prefill path (noisy dequant cost measurement). The two paths
  round differently, so near-tie greedy tokens can differ between servers with either
  drafter or none; within one server, speculative output stays exactly equal to target-only
  (the gate above). This predates the DSpine branch.

## Correctness contract and gates

Every draft layer adds a gated predecessor-code message. The last bridge write is recomputed
for each exact predecessor candidate, producing a predecessor-major 16 x 16 score table.
The target still verifies proposed tokens before committing them. Saved target caches and
request scheduling use the existing mechanisms.

Host tests cover the schema, actual converter round trip, BF16 bridge arithmetic, causal and
slot isolation, and dense-score tree selection. `dspine-cuda` compares the CUDA bridge,
predecessor gather and refinement primitives against host calculations. `mlx_dspine_bridge`
compares MLX CPU/GPU bridge outputs against the same BF16 CPU reference. GPU tests must run
on the intended platform; compiling them does not establish correctness.

Before release, compare target-only and speculative token IDs on the real model, including
rejections, context boundaries, prompt reuse and concurrent CUDA slots. Record real decode
time, accepted tokens per step, VRAM and the exact build. Compare drafter acceptance only
on the same trained target and prompt set. No speedup or distribution-equivalence result
is claimed by this implementation alone.

Algorithm references: [Tencent's bridge](https://github.com/tencent/dspine/blob/main/dspine_mlx/dspine_mlx/bridge.py),
[llama.cpp adaptation d493f841b2](https://github.com/Ajay9o9/llama.cpp/commit/d493f841b20241606c47a4e0537d3c786b5cf50b).
The checkpoint is Apache-2.0 licensed; retain its license when distributing its weights.

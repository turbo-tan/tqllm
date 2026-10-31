# Adding a model architecture

[Documentation](../README.md) / [Project home](../../README.md)

A new architecture (Gemma, DeepSeek, ...) is one folder of model code over shared components. Open an issue first
(CONTRIBUTING.md), so the work is not duplicated.

## What you write

| Piece | Where | Notes |
|---|---|---|
| Config + tensor schema check | `src/models/<family>/<arch>/` | read the GGUF / HF config, check every tensor name and shape before allocating |
| CPU reference forward for one token | same folder | the ground truth the GPU forward is tested against |
| GPU forward and the `Engine` subclass | same folder (`.cu` next to its host `.cpp`) | implement `src/engine/engine.hpp`; only what the model needs (see below) |
| Registration | `src/backends/<platform>/builtin_<platform>.cpp` | one `register_engine({name, matcher, make, exact_path})` line; first match wins, so put specific matchers before general ones |
| Converter (if the source checkpoint needs one) | `tools/convert/<arch>/` | |
| Tests | `tests/`, registered in `CMakeLists.txt` | schema check on a mock file; GPU greedy tokens equal the CPU reference; a serve smoke test |
| Docs | `docs/models/<arch>.md`, and the model folder in the AGENTS.md source map | `tests/test_docs_map.py` fails until the folder is in the map |

The engine interface has defaults for everything optional. An engine that serves one request at a time implements
`synchronize`, `layers`, `hidden`, `vocab`, `eos`, `context`, plus either `step` (one token, the server drives drafting)
or `spec_begin` / `spec_advance` / `spec_finish` (the engine drafts itself; return `incremental() == true`). Batched
serving across slots adds `slots`, `select_slot` and `spec_step`. Images add `set_prompt_images`.

## What you get without writing it

| Need | Component | Entry point |
|---|---|---|
| Weight formats (K-quants, IQ, TQ3 / TQ4, FP8, ...) | `src/formats/weight/` | `weight_format(type)->load(...)` gives a `Mat` with `matvec`, the expert GEMMs and `dequant_h`; every format passes `tests/format_contract.cu` |
| K/V cache formats, host tail (TqCache), paged pool | `src/formats/kv/`, `src/kv/` | `KvStore` / `KvMemory` (rule 4: models do not allocate K/V themselves) |
| MoE routing, grouped expert GEMM, expert caches (VRAM / host / mixed) | `src/experts/` | shared by every MoE model; there is no MoE model folder |
| Sampling (temperature, top-k / top-p, presence, speculative accept) | `src/spec/token_sampling.hpp` | `top_dist`, `draw`, `accept_drafts` over the head's top-16 |
| Drafting (DFlash, token trees) | `src/spec/` | |
| Vision encoders | `src/vision/` | `--mmproj` |
| GGUF / safetensors readers, tokenizer, options, CUDA helpers | `src/core/` | |
| `--exact on` semantics | `src/engine/model.cpp` | set `exact_path = false` in the registry entry until the engine has an exact path; the server then warns |

When something is missing, extend the component; do not copy it into the model folder (AGENTS.md rules 2, 4-6).

## Before the first PR

- `ctest` passes, including `layering` (a model never includes another model or a backend) and `docs_map`.
- Greedy text identical across two runs, and GPU tokens equal the CPU reference on the test prompts.
- New environment variables or options are in `config/flags.yaml` (`python3 tools/gen_flags.py --check`).

Kernels are per model today: the SwiGLU / RoPE / RMS-norm kernels of the existing models are fused variants, not
copies. A new model may use one of them as a starting point; a kernel two models share byte for byte moves to
`src/core`.

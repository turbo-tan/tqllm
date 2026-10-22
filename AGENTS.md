# AGENTS.md: how tqLLM is organised

Read this before adding or moving code. It applies to people and to coding agents alike.

tqLLM is one engine with shared components and one folder per model architecture. Host and device code of a
component live in the same folder. Dependencies point down only, and `tests/test_layering.py` checks that in CI.

```
  tools/                     entry points only: tqserve, tqbench, tqkld, tqrun, tqverify, ... (arguments, main)
  tools/serve/               HTTP / OpenAI API, scheduler, sessions, chat templates
  tools/convert/<arch>/      checkpoint converters
        |
        v
  src/engine/                Model (opens a file, fits memory, picks the engine), engine interface (engine.hpp)
        |
        v
  src/models/<family>/<arch>/      one folder per architecture
        qwen/qwen35          Qwen3.5 / Qwen3.8 dense and qwen35moe: one engine, MoE is a block type
        qwen/qwen4exp        Flash-Next
        qwen/qwen2           the draft-model forward
        deepseek/ nemotron/ glm/ ...   next architectures, same shape
  src/backends/<name>/       another runtime for the same models (e.g. mlx/: Apple MLX / Metal); model level,
                             may use components and core, never a model folder or another backend
        |   a model may use any component below; it never includes another model
        v
  src/formats/weight/        weight formats: host codec + kernels + registry (WeightFormat)
  src/formats/kv/            KV-cache formats: record layout + attention + registry (KvFormat)
  src/kv/                    KV store: device / host tiers (KvMemory, CUDA virtual memory), TqCache
  src/experts/               routing, grouped expert GEMM, expert caches (VRAM / host / mixed): every MoE model
  src/spec/                  drafting and verify: DFlash, token trees
  src/vision/                vision encoders (--mmproj)
        |
        v
  src/core/                  CUDA helpers, int8 / fp8 GEMM, graph capture, GGUF / safetensors readers,
                             tokenizer, options, runtime     (+ include/tqllm/: the public API)
        |
        v
  third_party/               vendored code; its licence is in licenses/ and named in NOTICE
```

## Rules

1. Include downward only. `src/core` includes nothing above it. Components include core and other components.
   Models include components and core. Nothing includes `tools/`.
2. A model never includes another model. Code two models need moves down into a component or core.
3. No engine code in `tools/`. Tools call the `Model` / `Engine` interface. A model registers its GGUF
   `general.architecture` string with the model dispatch in `src/engine/`.
4. K/V memory goes through `src/kv` (KvStore / KvMemory). A model does not allocate its own K/V buffers, so KV
   formats and TqCache work for every architecture.
5. Weights go through the WeightFormat registry. A new format is its files in `src/formats/weight/` (a folder of its
   own when it has several) plus one line in `registry.cpp`, and it passes the format tests.
6. Expert routing, expert GEMM and expert caches live in `src/experts` and are shared by every MoE architecture.
   There is no MoE model folder.
7. Device code (`.cu`, `.cuh`) sits next to the host code of the same component.
8. Includes name the file only (`#include "kv_store.hpp"`); CMake puts every source folder on the search path, so
   file names must stay unique across `src/`.
9. Every environment flag and option is in `config/flags.yaml` with an owner and an expiry.
10. Existing ctest names do not change. A new test goes next to the closest existing one and is registered in
    `CMakeLists.txt`.
11. No new top-level folder, component or layer without the maintainers' agreement.

## Adding a model architecture

- `src/models/<family>/<arch>/`: config + tensor schema check, a CPU reference forward for one token, the CUDA
  forward, and the dispatch entry in `src/engine/`.
- Use `src/kv`, `src/formats`, `src/experts` and `src/spec`. Extend them when something is missing; do not copy them
  into the model folder.
- Tests: the schema check on a mock file; CUDA greedy tokens identical to the CPU reference; a serve smoke test.
- Converters go in `tools/convert/<arch>/`. One page in `docs/models/<arch>.md`.

## Known exceptions

`tests/layering_allowlist.txt` lists the violations that exist today. The list may only shrink: the test fails on a
new violation and on an entry that no longer applies, so remove an entry in the PR that fixes it. The main open items:

- Shared kernel primitives (norms, GEMV, attention helpers) still live in `qwen35_kernels.cuh` and `qwen2_kernels.cuh`,
  so several formats include a model header. They move to `src/core`.
- The Flash-Next engine lives in its `fn_*` components (the old `tqfn_stage.cpp` monolith is in `archive/`, never rebuilt), but it still keeps its
  own K/V buffers (rule 4) and its own projection dispatch instead of the `WeightFormat` registry. Moving K/V onto
  `src/kv` and the projections onto the registry are the next steps.

`docs/dev/layout-map-20261017.tsv` maps every path from before this layout to its new place, including paths that
only exist on open branches; use it to move a branch onto this layout.

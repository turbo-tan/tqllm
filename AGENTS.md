# AGENTS.md

Two parts: **Setting tqLLM up for a user** (for coding agents asked to install and run tqLLM) and **How tqLLM is
organised** (for anyone changing the code).

## Setting tqLLM up for a user

tqLLM is an OpenAI-compatible inference server for Qwen3.8 models on NVIDIA GPUs (RTX 30 / 40 / 50, GB10 / DGX
Spark), Linux or Windows with WSL 2. Steps, in order:

1. **Check the hardware.** Run `nvidia-smi` and note the GPU and its memory. Docker needs the NVIDIA Container
   Toolkit (`docker run --rm --gpus all ubuntu nvidia-smi` must print the GPU).
2. **Pick a model file** from [YTan2000/Qwen3.8-27B-TQ](https://huggingface.co/YTan2000/Qwen3.8-27B-TQ) by GPU memory:

   | GPU memory | File | Size |
   |---|---|---|
   | 16 GB | `Qwen3.8-27B-TQ_XS.gguf` (most room for context) or `Qwen3.8-27B-TQ_S.gguf` | 11.85 / 12.55 GB |
   | 16-24 GB | `Qwen3.8-27B-TQ_M.gguf` | 13.14 GB |
   | 24 GB and GB10 | `Qwen3.8-27B-TQ_L.gguf` (closest to the original model) | 14.72 GB |

   Also download the drafter and its vocabulary (exact speculative decoding: same output, faster):
   ```bash
   hf download YTan2000/Qwen3.8-27B-TQ Qwen3.8-27B-TQ_L.gguf drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
     mtp-vocab/atx_65536.txt --local-dir models
   ```
3. **Run the server** (Docker, x86-64 and arm64 in one tag):
   ```bash
   docker run --rm --gpus all -p 127.0.0.1:8086:8086 -v "$PWD/models:/models:ro" \
     -e MODEL_FILE=/models/Qwen3.8-27B-TQ_L.gguf \
     -e DRAFT_MODEL_FILE=/models/drafters/Qwen3.8-27B-DFlash2-TQ3.gguf \
     -e DRAFT_VOCAB_FILE=/models/mtp-vocab/atx_65536.txt \
     -e CONTEXT=32768 -e REASONING=on -e MAX_TOKENS=4096 \
     ytan2000/tqllm:1.2.0
   ```
   The server fits the KV cache and context to the free memory by itself and prints what it chose. On a 16 GB GPU
   use `-e CONTEXT=16384 -e KV_FORMAT=tq3`. Without Docker: build `tqserve` from source (README, "Build").
4. **Check it.** `curl -s http://localhost:8086/health` must answer `{"status":"ok",...}`; then
   ```bash
   curl http://localhost:8086/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":64}'
   ```
5. **Connect the user's tools.** Any OpenAI-compatible client: base URL `http://localhost:8086/v1`, model `tqllm`, any
   API key. Codex uses the Responses API (`wire_api = "responses"`). Or let the launcher start the server and the
   agent together: `python3 tools/launch/tqllm.py launch opencode` (also `claude`, `codex`, `pi`, `dsh`;
   `--list` shows the model catalog and downloads with a sha256 check).

If it fails: no GPU in the container -> install the NVIDIA Container Toolkit; out of memory -> a smaller file or a
smaller `CONTEXT`; a GPU older than RTX 30 is not supported. Qwen3.8 Flash-Next (125B MoE) on one 24 GB GPU needs
extra files beside the GGUF; its setup guide is not published yet, so do not improvise one.

## How tqLLM is organised

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

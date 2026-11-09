# AGENTS.md

Two parts: **Setting tqLLM up for a user** (for coding agents asked to install and run tqLLM) and **How tqLLM is
organised** (for anyone changing the code).

## Setting tqLLM up for a user

tqLLM is an OpenAI-compatible inference server for Qwen3.8 models on NVIDIA GPUs (RTX 30 / 40 / 50, GB10 / DGX
Spark), Linux or Windows with WSL 2, and on AMD GPUs through ROCm/HIP (see the AMD guide; single GPU only in this
release). Steps, in order:

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
     ytan2000/tqllm:1.4.2
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
smaller `CONTEXT`; a GPU older than RTX 30 is not supported.

Qwen3.8 Flash-Next (125B MoE) on one 24 GB GPU (RTX 30 or newer) or a GB10: needs 64 GB of system RAM or more (tested
with 128 GB) and image 1.4.2 or newer. Follow the run steps on https://huggingface.co/YTan2000/Qwen3.8-Flash-Next-tqLLM exactly: the ISTA-DASLab
IQ3_S GGUF plus the small files there in one folder, then `tqserve --model <folder>/flashnext.tqfn`. Start-up takes
about two minutes (the experts are arranged in RAM for the GPU cache); wait for `/health` before the first request.

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
  src/engine/                Model (opens a file, asks the registry for the engine, forwards every call): backend
                             neutral, no CUDA / Metal include; engine interface (engine.hpp) and registry (registry.hpp)
  src/backends/<platform>/   one folder per platform: cuda/, hip/, metal/, android/. builtin_<platform>.cpp lists the engines
                             that platform serves (register_builtin_engines), beside its loaders and its engine code.
                             Engine level: may use models, components and core; no model includes a backend.
        |
        v
  src/models/<family>/<arch>/      one folder per architecture
        qwen/qwen35          Qwen3.5 / Qwen3.8 dense and qwen35moe: one engine, MoE is a block type
        qwen/qwen4exp        Flash-Next
        qwen/qwen2           the draft-model forward
        glm/glm5next         GLM-5.3-Flash (DSA attention, MoE)
        minimax/m3           MiniMax-M3.1-Flash (GLM kernels through src/experts)
        deepseek/ gemma/ ...   next architectures, same shape
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
3. No engine code in `tools/`. Tools call the `Model` / `Engine` interface. An engine is added to its platform's
   `register_builtin_engines()` (`src/backends/<platform>/builtin_<platform>.cpp`) with a matcher for its model files;
   `src/engine/model.cpp` never names an engine.
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
12. **A backend is a registration + its own kernels, never a copy of shared code.** `src/backends/<platform>/` holds
    the platform shim or compatibility headers, that platform's own kernels, and stubs for what it does not implement
    (factories that refuse by name). The engine list is shared (`src/engine/builtin_engines.cpp`), and so are the
    device primitives' names: each platform implements them in `src/core/platform/<platform>_device.cuh` with its
    capability macros (`TQLLM_DEV_*`), selected in one place, `src/core/common.cuh`. Anything that is not platform specific (the memory fit, engine
    assembly, `Model` methods) is compiled for every backend from one file: `src/engine/load_fitted.cpp`,
    `src/engine/qwen2_engine.cpp`, `src/engine/model_capture.cpp`. A `git diff` between two `src/backends/*` folders
    must not show a near-identical `.cpp`. `CMakeLists.txt` has one `TQLLM_ENGINE_SOURCES` list; a backend block adds
    only its registration and its exclusions, so a new engine file is added once.
13. **Model and component code does not branch on the backend.** Platform differences go into a helper next to the
    thing that differs and branch on capabilities: `TQLLM_DEV_PTX` / `TQLLM_DEV_CP_ASYNC` / `TQLLM_DEV_LDS_64K` (device,
    `src/core/platform/`), `src/core/backend_caps.hpp` (host), `src/core/cub_compat.hpp` (CUB / hipCUB),
    `src/core/blas_compat.hpp` (cuBLAS / hipBLAS), `src/core/copy_compat.hpp` (batched copies). A
    `#if defined(TQLLM_WITH_<backend>)` in `src/models/` or in a component is a review blocker, and
    `tests/test_layering.py` (BackendNames) fails on one; the same holds for `#if` in a tool when a single `TQLLM_WITH_GPU` would do.
14. **Unsupported means an error, not a fallback.** A stub for a path a backend does not implement throws with the
    reason (and a device-side `__trap`/assert where a throw is not possible); it never returns silently, never returns
    zeros, and never switches the caller to a different code path behind its back. Capability questions are answered by
    `supported()`-style predicates the caller consults, not by catching exceptions.
15. **Detection stays exact on every backend.** GGUF/HF model detection is host metadata code, so
    `is_<model>_gguf()` / matchers are shared, not stubbed: an engine a backend cannot run must still be *recognised*
    and refused by name, never fall through to another architecture's loader. Keep the list in
    `docs/guides/adding-a-backend.md` ("what a backend must implement") current.
16. **Report your own mistakes, in the same message.** If you get something wrong - a push to the wrong repo, a number
    taken from a stale tree, a claim you did not verify, an action taken in someone else's repository - say what happened,
    what it did to shared state (what was pushed where, what was reverted, what is still dirty) and what you did about it.
    Do not quietly re-edit and let the reader believe it never happened: silent self-correction destroys the evidence the
    next person needs, and that is worse than the error. Unsure whether an action is allowed? Ask *before*, not after;
    writes to a contributor's repository, force-pushes, deletes and public pushes are ask-first, always. Wording shared
    with the fleet: `agent.md` rule 3.

## Tests a pull request must bring

Reviewers ask for these before merge; a PR that changes device math, formats, backends or the request path without them
is not reviewable.

1. **A test that fails on the old code.** A bug fix gets a unit test that reproduces it (or a ctest that exercises the
   fixed path); "tested by hand on my GPU" is a note, not a test.
2. **The invariant tests for the area you touched.** `tree` (one attention arithmetic: every token tree node equals its
   chain), `dflash_exact` / `native_mtp` (greedy with a drafter equals greedy without), `fp_formats` (a column's bits do
   not depend on batch width), the format contract and the kernel oracles. If your change touches attention, drafting,
   the memory fit or a format, say in the PR which of these you ran and what they printed.
3. **A new backend or platform runs the shared engine tests**, not just its own kernels: `tree` is registered in ctest
   for every backend (`TQLLM_ENGINE_TESTS_SYNTHETIC`), and `dflash_exact` / `native_mtp` are built as targets on every
   backend (`TQLLM_ENGINE_TESTS_MANUAL`, they need a model on the command line). `docs/guides/adding-a-backend.md` lists
   the full set.
4. **Ctest names do not change** (rule 10) and the test is registered in `CMakeLists.txt`, in the block that builds it.
   A test that needs a GPU model path takes it as an argument and stays a manual target; a test that can be synthetic
   must be a ctest entry, because CI only runs the synthetic suite.
5. **The gate for a numeric change**: ctest 100 % plus byte-identical texts on the 8 gate prompts for every engine the
   change can reach, with the build and the machine named in the PR. A PTX-level argument ("the compiler already
   contracts this") is evidence only when you show it (e.g. `nvcc -ptx` per-kernel counts before/after).
6. **Every platform you touched builds.** Both backends are compiled on every PR by `.github/workflows/gpu-compile.yml`
   (CUDA and HIP compile-only, no GPU needed); a change to shared code that only one of them compiles is not done.

## Adding a model architecture

Checklist (what you write, what you get from the components): [docs/guides/adding-a-model.md](docs/guides/adding-a-model.md).

- `src/models/<family>/<arch>/`: config + tensor schema check, a CPU reference forward for one token, the CUDA
  forward; the registry entry goes in `src/backends/<platform>/builtin_<platform>.cpp`.
- Use `src/kv`, `src/formats`, `src/experts` and `src/spec`. Extend them when something is missing; do not copy them
  into the model folder.
- Tests: the schema check on a mock file; CUDA greedy tokens identical to the CPU reference; a serve smoke test.
- Converters go in `tools/convert/<arch>/`. One page in `docs/models/<arch>.md`.

## Branches, rebases and PR size (how not to end up in rebase hell)

llama.cpp lives with permanent rebase pain because big branches outlive the code they were written against. The rules
that keep tqLLM out of it:

1. **Small PRs, short life.** A PR is merged, split or closed within ~2 weeks of opening; anything larger than ~800
   changed lines, or touching more than one of {engine, format, backend, server, docs}, needs a design note and a
   maintainer's agreement *before* the code.
2. **Rebase onto `main` weekly, and before review starts**, not once at the end. A branch that has not been rebased for
   two weeks is treated as stale: it is closed with a note and reopened as a fresh PR when someone actually wants it.
3. **A rebase is a moment to drop, not only to fix.** When a conflict shows a file that is now a near-copy of another
   (the AMD `load_fitted.cpp` case), the resolution is "delete the copy", not "keep both sides compiling".
4. **Land in the right layer the first time.** A short-lived branch that needs the same file in two `src/backends/*`
   folders is a sign the file belongs in `src/engine`, `src/models/...` or `src/core` (rules 12-13).
5. **Nothing is reviewed from a stale head.** Reviews happen against a rebased head with the required CI green; a head
   that is CONFLICTING is not reviewable, and the reviewer says so instead of reading the old diff.
6. **No long-lived platform branches.** A backend lands in one PR against `main`; further platform work is small PRs on
   top. We do not carry an "amd" branch that is rebased forever.
7. **Prefer a branch in this repo over a fork for work we drive.** A pull request from a fork needs a maintainer to
   approve *every* workflow run before it starts (`action_required`), and approvals do not survive a push, so the checks
   stop being a gate and become a queue. A branch in `turbo-tan/tqllm` runs the same checks with no approval step. Fork
   pull requests stay welcome; work that we own and keep pushing goes in-repo.

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

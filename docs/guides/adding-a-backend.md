# Adding a GPU backend

Checklist for a new platform (AMD/ROCm today, Vulkan/CPU-Vector/another vendor tomorrow). Read
[AGENTS.md](../../AGENTS.md) rules 1-15 first: a backend is a *registration plus its own kernels*, never a copy of
shared code. The AMD backend (`src/backends/hip/`) is the reference implementation.

## What you write

| Piece | Where | Notes |
|---|---|---|
| Platform registration | `src/backends/<platform>/builtin_<platform>.cpp` | `register_builtin_engines()`: the engines this platform serves, matcher + factory, in match order. Nothing else in the tree learns your platform's name. |
| Shim / compatibility headers | `src/backends/<platform>/shim/` (or equivalent) | Only the host API names that differ, as compile-time mappings. Keep it thin: if the shim needs to *implement* behaviour, that behaviour belongs in `src/core`. |
| Platform device kernels | `src/backends/<platform>/` | The code that genuinely has no portable form. Every kernel you port keeps the documented arithmetic of the shared one (same online-softmax, same accumulation order where the format tests pin it). |
| Unsupported path stubs | `src/backends/<platform>/unsupported_<platform>.cpp` | One file, each function either throws with the reason or returns a capability answer. No silent no-ops, no host round-trips standing in for a kernel, no "falls back to the dense path" surprises (rule 14). |
| Build block | `CMakeLists.txt` | The platform block adds `${TQLLM_ENGINE_SOURCES}` (shared) + its registration + its exclusions. Never a second copy of the list. |
| CI job | `.github/workflows/gpu-compile.yml` | A compile-only job for your platform: GitHub runners have no GPU, but a compiler does not need one. This is what keeps the backend from rotting between releases. |
| Docs | `docs/guides/<platform>.md` + a row in `docs/README.md` | State what works, what refuses, and the exact build commands. A number in the docs carries the machine and the build. |

## What you implement, because it is platform-neutral and compiled once

Do **not** copy these; they are built for every backend from one file:

- `src/engine/load_fitted.cpp` - the `-fitt` memory fit (placement steps, `fit_slots`, host-tail measurement).
- `src/engine/qwen2_engine.cpp` - engine assembly for the Qwen2 drafter.
- `src/engine/model_capture.cpp` - `Model::capture`.
- `src/spec/dflash.cu` - drafting and verify, shared by every backend.

If one of them does not compile on your platform, the fix is in the shim or in `src/core`, not a copy in your
folder: a copy drifts the first time someone changes the original (the AMD copy lost the integrated-GPU memory
branch within hours of the rebase).

## Platform differences go into `src/core`

| Difference | Helper |
|---|---|
| Device math (tensor-core MMA, `ldmatrix`, `cp.async`, packed f16, dot products) | a new `src/core/platform/<platform>_device.cuh` with the same `tqllm_dev` names and the `TQLLM_DEV_*` capability macros; one `#include` line in `src/core/common.cuh` selects it |
| What the platform can do, for host code (fragment GEMMs, dense solver, batched copies) | `src/core/backend_caps.hpp` |
| Batched host-to-device copies | `src/core/copy_compat.hpp` |
| CUB vs hipCUB | `src/core/cub_compat.hpp` |
| cuBLAS vs hipBLAS (math mode, device capability checks) | `src/core/blas_compat.hpp` |
| A device API name that only differs in spelling | your platform's shim |

A `#if defined(TQLLM_WITH_<platform>)` inside `src/models/` or a component is a review blocker (rule 13), enforced by
`tests/test_layering.py`. Kernels test capabilities instead (`#if TQLLM_DEV_LDS_64K`, `if constexpr (tqllm::caps::...)`),
so a third platform with the same capability takes the same path without touching the kernel.

## What a backend must pass before it is called supported

1. `ctest` on the platform, including the shared invariants: `tree` (one attention arithmetic), `fp_formats` (bits do
   not depend on batch width), `format_contract`, the kernel oracles, `quant`, `tq3_rows`, `tq_gemm`.
2. `dflash_exact` and `native_mtp` built as targets and run once with a real model and drafter (greedy with a drafter
   equals greedy without). They need a model on the command line, so they stay manual on every backend.
3. A greedy-vs-reference check of your own: the NVIDIA gate prompts are not evidence for hardware with different
   rounding. Record the AMD goldens in the platform doc when they are first taken.
4. `tqserve` serving a real request, with the number in the PR (an end-to-end token rate, not a kernel microbench).

Until (1)-(4) run, the platform is *experimental*: say so in the docs, keep it behind an opt-in CMake option, and do not
advertise models it cannot run.

## The registry is where capability is decided

A platform that cannot run a format or a prefill path reports it from the strategy (`has_gemm()`, `supported()`), so the
engine stays free of platform branches (rule 5). Two rules for those answers:

- Report the capability, do not disable the feature permanently: `return true; // AMD uses dequant + hipBLAS` in
  `gemm_off()` silently removes the native GEMM forever, including after someone ports it.
- A registry that skips a format also owns the message: an X3 file on AMD should say "X3 is not built for AMD", not
  "unknown format".
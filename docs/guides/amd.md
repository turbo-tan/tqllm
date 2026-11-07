# AMD GPUs (ROCm / HIP)

tqLLM builds an AMD backend from the same sources as the CUDA one. The compatibility layer is
compile-time only: it maps `<cuda_runtime.h>` and friends onto HIP headers, and every piece of
device math that CUDA expresses in PTX gets a native AMD implementation next to it. There is no
CUDA-PTX interpreter or per-call emulation in the hot paths.

Tested on: **Radeon Pro W7800 (gfx1100, RDNA3)**, ROCm 7.13, Ubuntu 24.04. Other RDNA3 boards
should behave the same. CDNA (gfx90a/942/950) is expected to work but is not tested by this PR.

## Build

Requirements: ROCm 7.x with HIP and hipBLAS, CMake 3.24 or newer, a C++20 host compiler.

```sh
cmake -S . -B build-hip \
  -DTQLLM_CUDA=OFF -DTQLLM_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake
cmake --build build-hip -j
```

`CMAKE_HIP_ARCHITECTURES` defaults to the GPU in the machine (`native`). Use an explicit target
(e.g. `gfx1100`) when packaging for another board. If hipBLAS lives outside the compiler's default
search path, point `CMAKE_PREFIX_PATH` at the ROCm tree's `lib/cmake` directory (the build does
this automatically from `hipconfig --rocmpath` when it is on `PATH`).

Run the same way as the CUDA build:

```sh
./build-hip/tqserve --model models/model.gguf --port 8086
```

## What works in this backend

| Area | Status |
|---|---|
| Qwen3.8 dense and MoE (`qwen35`, `qwen35moe`) | yes |
| TQ4 / TQ3 weight formats | yes (decode GEMV and prefill) |
| TQ3 KV cache (`tq3`) | yes |
| Qwen2 drafter, DFlash speculative decoding | yes |
| BF16 prefill GEMM | hipBLAS |
| Split-KV decode attention, sliding window, ring buffers | yes |

Not built on HIP in this PR, each with an explicit error instead of a silent fallback:

- Flash-Next (`qwen4exp`) engine: compiles on HIP (qwen4exp.cu, qwen4exp_qsa.cu, fn_*.cpp all
  build; CUB calls ported to hipCUB). Not runnable: the expert GEMM uses sparse MMAs
  (`mma.sp.sync`) with no RDNA3 equivalent, QSA attention is stubbed, and the model's GGUF uses
  custom tensor type 42 (Flash-Next dialect) not yet handled by the HIP GGUF reader. The model
  also needs 64 GB RAM + 24 GB VRAM (this box has 30 GB). A full port requires: sparse→dense
  v_wmma rewrite, QSA attention port, GGUF type 42 support, and a machine with 64+ GB RAM.
- GLM-5.3-Flash and MiniMax-M3 engines (no AMD kernels yet; model probes return no match).
- Vision / multimodal input (`--mmproj`).
- FP8 and INT8 prefill GEMM (`Fp8Gemm` / `Int8Gemm` report `supported() == false`; prefill uses
  BF16 instead).
- TqCache compaction (`--kv-compact`): the stand-in solve uses cuSOLVER; the option logs that it is
  off on AMD.
- Tensor parallelism (NCCL/RCCL): single GPU only. `TP` setup paths that include `<nccl.h>` fail at
  compile time with a message naming RCCL.
- X3 (PAW-27B trellis) weight format: its device code is raw PTX and is not registered on AMD.

## How the port is organised

`src/backends/hip/`

- `shim/`: compile-time name mapping for the CUDA host API (`cuda_runtime.h`, `cuda.h`,
  `cublas_v2.h`, bf16/fp16/fp8 headers, math constants) and the 32-bit-mask warp-sync set. Nothing
  here changes arithmetic.
- `builtin_hip.cpp`, `load_fitted.cpp`, `qwen2_engine.cpp`, `model_capture.cpp`: engine glue,
  mirrored from `src/backends/cuda/` with the vendor checks adjusted (no `sm_XX` test on AMD).

Device primitives live next to the kernels they serve:

- `src/core/common.cuh` keeps the CUDA `mma`/`ldmatrix` contract and supplies native AMD versions:
  `v_dot4` (`__dp4a`) for the int8 dot products, `v_dot2_f32_f16` / `_bf16` for the 16-bit dots,
  packed `v_pk_*` f16 arithmetic for the half2 helpers, and native LDS reads with the ldmatrix lane
  mapping. Kernels that keep the CUDA register layout therefore run the same math on RDNA3's packed
  ALUs, and the hot kernels additionally get AMD-fit tilings (below).
- Decode is bandwidth-bound on these models: the GEMV kernels are plain SIMT and already run at
  memory speed, so they need no matrix-core rewrite.
- Prefill runs the existing dequantise-to-BF16 path on hipBLAS (rocBLAS matrix cores). The custom
  `tq4_gemm_h` kernels are CUDA-fragment code and are disabled on HIP (`TQLLM_DEBUG=tq_gemm` has no
  effect); `Fp8Gemm`/`Int8Gemm` are off for the same reason.
- Attention keeps the CUDA kernels' online-softmax math and partial-output layout, with AMD tile
  sizes that fit 64 KB of LDS. The `v_wmma` fragment layouts are verified on gfx1100 (for
  `v_wmma_f32_16x16x16_f16`: even lanes hold one A row of 16 halves, lanes 0..15 one B column of 16
  halves, 16..31 duplicate, accumulator `lane = (m/8)*16 + n`, `register = m%8`); moving the prefill
  attention kernels onto them is the next performance step, not part of this PR.

RDNA3 limits a workgroup to **64 KB of LDS** (NVIDIA parts allow 99-164 KB). The CUDA attention
kernels were sized for the larger budget, so the AMD build uses reduced query-row and K/V tile
counts and single-buffered staging where a kernel asked for more than 64 KB. The math and the
partial-output layout are unchanged; only the tiling differs.

## Validation

```sh
ctest --test-dir build-hip -R "format_contract|kernels|tq_gemm|tq3_rows|moe_device|dflash|graph_capture|kv_mem_count|fp_formats|quant|matvec_wide|kv_turbo_group|kv_tcq|dflash_retrieve" --output-on-failure
```

These are the same device tests the CUDA build runs: they compare kernels against the CPU reference,
check that batched forms match per-column forms bit for bit, and exercise the KV codecs, MoE,
DFlash and graph capture. `kv_mem_count` and `kv_host` cover the CUDA-driver virtual-memory KV store
through the HIP aliases, including the host tier.

On the W7800 with ROCm 7.13 the full suite is green: `ctest --test-dir build-hip` reports 78/78
(63 CPU + 15 GPU tests). `test-dflash` covers the multi-query, tree/chain, Turbo3 and Turbo4
attention paths; `test-kv-turbo-group` checks that batched verify equals sequential decode for every
KV codec.

## Measured on the tested card

Radeon Pro W7800 (gfx1100), ROCm 7.13, 24 GB, `Qwen3.8-27B` GGUF, `--context 1024`, `--max-tokens 8`:

- prompt: 34.4 tok/s (15 tokens, 436 ms)
- decode: 10.0 tok/s (8 tokens)

Decode is memory-bound and runs the native GEMV kernels. Prefill runs the dequant + hipBLAS path; the
custom `tq4_gemm_h` kernel (the CUDA fast path) is still CUDA-fragment code and is disabled on HIP,
so prefill is the main performance item for a follow-up.

## Troubleshooting

- `cannot find -lhipblas`: the hipBLAS CMake package exports `-lhipblas` without a path on some
  ROCm installs. The build adds `${hipconfig --rocmpath}/lib` to the link path; if you moved the
  ROCm tree, pass `-DCMAKE_PREFIX_PATH=<rocm>/lib/cmake`.
- `language not recognized: ' hip'` from a host file: an old ROCm `hip-config` adds `-x hip` to
  every language. The build re-scopes those flags to HIP sources; use ROCm 6.4+ or 7.x.
- Kernel launch fails with "invalid argument" right after start: check the requested dynamic shared
  memory against `sharedMemPerBlockOptin` (64 KB on RDNA3). The attention tile sizes pick a budget
  at compile time; do not raise them on a 64 KB part.
- `--kv-compact` logs `off`: expected on the AMD backend until hipsolver is linked.

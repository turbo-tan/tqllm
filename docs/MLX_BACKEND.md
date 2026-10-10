# MLX backend implementation lane

Branch: feat/mlx-backend. Initial base: release main 776e257.
This is a native C++ backend project; MLX inference is not implemented yet.

Keep the existing numerical quant formats. Custom MLX/Metal operators must
preserve each format's codes, scales, codebook and activation basis. Container
conversion or lossless runtime repacking is allowed; affine requantization is
a separately labelled quality experiment.

## Stages and gates

1. Portable build boundaries: separate CUDA-only diagnostics from shared
   server, template and request tests. Build with CUDA disabled on a host
   without CUDA headers. Preserve the CUDA build and test targets.
2. MLX C++ engine factory: persistent weights and sequence state; backend-safe
   logits access, sampling, evaluation and synchronization. Load tiny fixtures
   and complete real HTTP requests on Apple Silicon.
3. Existing TQ4 operators: rotation, packed-code reconstruction, decode and
   multi-column verify against independent CPU oracles. Bounded prefill GEMM
   scratch; no dense expansion of the whole model.
4. Real 27B correctness: teacher-forced logits/KLD, consecutive-request
   isolation, reset, EOS and sampling. Measure prefill and AR at 2K/8K/32K
   with exact model/build/input provenance.
5. MTP/DFlash2: exact greedy verification, speculative state rollback,
   acceptance and verify cost. Compare competitors only after correctness.
6. Add the actual winning TQ3 and mixed-recipe tensor types. Reject unsupported
   tensors explicitly; do not silently substitute another quant.

## Initial baseline, 2026-10-10

`cmake -S . -B build-host -DTQLLM_CUDA=OFF -DCMAKE_BUILD_TYPE=Release`
fails at generation: test-kv-mem-count unconditionally links CUDA::cudart.
The shared request header also includes CUDA and the CUDA vision interface,
while tqjinja is defined inside the CUDA block. These are portability blockers,
not evidence that an MLX backend has failed an inference test.

Mac experiments already exist in private research as Python MLX + custom TQ4
Metal operators. Reuse verified math and tests, but keep public source free of
machine addresses, credentials, logs and private model recipes. Check current
Mac lane ownership before any hardware run; one loaded model at a time.

## Experimental operators

The portable host build and tests run on Apple Silicon. Configure
`TQLLM_CUDA=OFF`, `TQLLM_MLX=ON` and point `CMAKE_PREFIX_PATH` at an installed
MLX C++ package (MLX 0.32 or later). This builds `tqmlx` and `test-mlx-tq4`.
It does not yet build a complete MLX inference engine or server.

The first operators preserve the existing 128-point signed rotation and
nonuniform TQ4 codebook: uint8 packed codes, FP16 scales per 128 columns,
and FP32 accumulation. They cover the existing safetensors layout, not all
GGUF quant types. The test compares the rotation and matrix outputs to
independent CPU calculations across tile sizes and partial output rows.

Use one output row per SIMD group initially. A wider tile is not an assumed
optimization; select tiles only after whole-model timing and correctness.
Integer-level expansion into native MLX affine 8-bit storage is a diagnostic
experiment, not an enabled backend: some batch sizes failed its precision
gate, and it increases weight storage.

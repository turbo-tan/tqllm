# MLX backend implementation lane

Branch: feat/mlx-backend. Initial base: release main 776e257.
This is an experimental native C++ backend lane. Dense qwen35 forward exists;
production Model/session dispatch and the HTTP server are not wired yet.

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
It also builds `tqmlx-probe`, a noninteractive dense qwen35 forward harness.
It does not yet build an MLX HTTP server.

The first operators preserve the existing 128-point signed rotation and
nonuniform TQ4 codebook: uint8 packed codes, FP16 scales per 128 columns,
and FP32 accumulation. They cover the existing safetensors layout and native GGUF300/301,
Q6_K, Q8_0 and Q4_K embedding blocks. GGUF scales use integer codebook
levels; the older safetensors layout uses normalized levels. The test compares the rotation and matrix outputs to
independent CPU calculations across tile sizes and partial output rows.

Use one output row per SIMD group initially. A wider tile is not an assumed
optimization; select tiles only after whole-model timing and correctness.
Integer-level expansion into native MLX affine 8-bit storage is a diagnostic
experiment, not an enabled backend: some batch sizes failed its precision
gate, and it increases weight storage.

## GGUF matrix boundary

`src/mlx/weights.hpp` dispatches native compressed matrices and reverses GGUF
column-first dimensions into MLX row-major dimensions. TQ matrices apply their
128-point input rotation exactly once. Plain F32/F16/BF16 tensors retain their
stored values. Unsupported types and malformed lengths fail explicitly.
Q4_K embeddings read only requested rows; the vocabulary is never expanded
into a second dense table.

`test-mlx-gguf <model.gguf>` checks representative real-file rows against CPU
calculations for every matrix type present, including batches1/8/64 and repeated
embedding indices. It is an arithmetic gate, not a model quality test.

Strict FP32 oracle tests use `MLX_ENABLE_TF32=0`. MLX defaults can use reduced
precision for FP32 matrix multiplication, so batched dense matrices must not
be assumed to match the FP32 oracle under the default environment. Keep the
precision setting in benchmark provenance.

## Experimental dense forward

`Qwen35` loads compressed GGUF weights and owns independent attention,
convolution and DeltaNet caches. GGUF value-head ordering is tiled: value head
h uses key head h % key_heads. The MLX recurrence receives explicitly expanded
key/query heads in that order. Reset clears every sequence cache.

`tqmlx-probe --model file.gguf --prompt text --generate 32 --repeat 2` loads one
model, produces greedy text and rejects unequal token output across repeated
resets. `--tokens-file` accepts whitespace-separated exact token IDs;
`--ignore-eos` is only for fixed-length throughput probes. Decode reports
(output_tokens - 1) / elapsed time after the first token. Precision, warmup,
repeats and prefill chunk must be matched and recorded in comparisons.
`--all-logits --logits file` dumps FP32 prompt logits for teacher-forced gates.

This path is experimental, without speculative decode, vision, MoE, tied
embeddings or RoPE scaling. It requires native Q4_K embeddings. Coherent text
and reset equality are smoke gates; they do not establish model quality or a
speed win. Keep it out of release dispatch until the many-token quality and
performance gates pass.

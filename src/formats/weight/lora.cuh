#pragma once
// Runtime LoRA on the device (lora.hpp): y += scale * B (A x) after the base matmul of an adapted matrix.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace tqllm_cuda {

constexpr int kLoraMaxRank = 64;
constexpr int kLoraSlice = 512;  // input columns per block of the A x kernel
inline size_t lora_scratch_floats(int n, int cols, int rank) {
    return size_t(n) * size_t(rank) * size_t((cols + kLoraSlice - 1) / kLoraSlice);
}

// a [rank][cols] FP32 (in the basis of x), b [rows][rank] FP32 (alpha / rank folded in), x [n][cols] FP16, y [n][ldy] BF16
// (first `rows` of each column updated). scale: device float, read when the kernels run (captured CUDA graphs follow
// changes); 0 leaves y bit-identical. t: device scratch of lora_scratch_floats(n, cols, rank) floats. cols even,
// rank <= kLoraMaxRank. Deterministic: A x is summed per column slice, the slices in a fixed order (no atomics).
void lora_apply(const float * a, const float * b, int rows, int cols, int rank, const __half * x, __nv_bfloat16 * y, int n,
                int ldy, const float * scale, float * t, cudaStream_t s);

// prompt chunks (n > kLoraGemmCols): the same y += scale * B (A x) as two cuBLAS GEMMs on tensor cores, t = A x in FP32 from
// an FP16 copy of A, then y += B (scale t) from a BF16 copy of B (scale applied to t on the device, so 0 adds exact zeros).
// t: n * rank floats, tb: n * rank BF16. The per-column kernels above do the same work on CUDA cores, ~3x slower than the
// whole prompt chunk on a GB10 with 256 adapted matrices.
constexpr int kLoraGemmCols = 64;
void lora_apply_gemm(cublasHandle_t h, const __half * a16, const __nv_bfloat16 * b16, int rows, int cols, int rank, const __half * x,
                     __nv_bfloat16 * y, int n, int ldy, const float * scale, float * t, __nv_bfloat16 * tb, cudaStream_t s);

}  // namespace tqllm_cuda

#pragma once
// Runtime LoRA on the device (lora.hpp): y += scale * B (A x) after the base matmul of an adapted matrix.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {

constexpr int kLoraMaxRank = 64;

// a [rank][cols] FP32 (in the basis of x), b [rows][rank] FP32 (alpha / rank folded in), x [n][cols] FP16, y [n][ldy] BF16
// (first `rows` of each column updated). scale: device float, read when the kernels run (captured CUDA graphs follow
// changes); 0 leaves y bit-identical. t: device scratch of n * rank floats. cols even, rank <= kLoraMaxRank.
void lora_apply(const float * a, const float * b, int rows, int cols, int rank, const __half * x, __nv_bfloat16 * y, int n,
                int ldy, const float * scale, float * t, cudaStream_t s);

}  // namespace tqllm_cuda

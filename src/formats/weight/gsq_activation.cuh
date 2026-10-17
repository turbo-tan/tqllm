#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda {
void gsq_q8k_round_h(const float * x, __half * y, int64_t count, cudaStream_t stream);
void gsq_q8_round_h(const float * x, __half * y, int64_t count, cudaStream_t stream);
}

#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda {
void dspine_read_norm(const __nv_bfloat16* x, __nv_bfloat16* normalized, float* rms,
                      int hidden, int rows, float eps, cudaStream_t);
void dspine_codes(__nv_bfloat16* codes, const __nv_bfloat16* codebook, const uint32_t* tokens,
                  int code, int block, int rows, float eps, cudaStream_t);
void dspine_message(const __nv_bfloat16* packed, const float* bias, __nv_bfloat16* message,
                    int rank, int block, int rows, cudaStream_t);
void dspine_add(__nv_bfloat16* x, const __nv_bfloat16* delta, const float* rms,
                int hidden, int block, int rows, cudaStream_t);
void dspine_predecessor_codes(const uint32_t* candidates, const uint32_t* tokens,
                              const __nv_bfloat16* codebook, __nv_bfloat16* out,
                              int code, int block, int rows, cudaStream_t);
void dspine_refine_message(const __nv_bfloat16* own, const __nv_bfloat16* predecessor,
                           const float* bias, __nv_bfloat16* message, int rank, int rows, cudaStream_t);
void dspine_refine_norm(const __nv_bfloat16* raw, const __nv_bfloat16* delta, const float* rms,
                        const float* norm, __nv_bfloat16* out, int hidden, int rows, float eps, cudaStream_t);
void dspine_score(const int8_t* head, const __half* scales, const __nv_bfloat16* refined,
                   const uint32_t* candidates, float* scores, int hidden, int rows, cudaStream_t);
}

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tqllm_cuda {
void gsq_q2_matvec(const void *, const __half *, __nv_bfloat16 *, int rows, int cols, int n, cudaStream_t);
void gsq_q2_matvec_f32(const void *, const __half *, float *, int rows, int cols, int n, cudaStream_t);
void gsq_q2_expert_matvec(const void *, const __half *, __nv_bfloat16 *, int rows, int cols,
                          int experts, const int * ids, int n, int topk, bool per_expert, cudaStream_t);
void gsq_q2_dequant(const void *, __half *, int rows, int cols, cudaStream_t);
// experts grouped by moe_group (counts [experts], groups [experts][capacity] of assignment ids), y [assignments][rows]
void gsq_q2_expert_grouped(const void *, const __half *, __nv_bfloat16 *, int rows, int cols, const int * counts, const int * groups,
                           int capacity, int experts, int topk, bool per_expert, cudaStream_t);
// dedup map (quant_kernels.cuh kDedupStride layout): each group's row blocks read once for its columns; per column the
// products and their order are those of gsq_q2_expert_matvec (bit-identical)
void gsq_q2_expert_dedup(const void *, const __half *, __nv_bfloat16 *, int rows, int cols, const int * dmap, int groups, cudaStream_t);
}

#pragma once
// FNT trellis weight format kernels (fnt.cu): kb = stream bits per weight pair (4..7); w = streams [rows][cols / 128][kb * 8 B],
// sc = FP16 scales [rows][cols / 640]; inputs and outputs in the plain basis (the kernels rotate the inputs themselves);
// mix: FNT-kX (the 5 blocks of each 640-group also mixed, fnt.cu mix5).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tqllm_cuda {
// stacked experts [experts * rows][cols], ids [n * topk] (null: expert 0), y [n * topk][rows]
void fnt_expert_matvec(bool mix, int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                       int experts, const int * ids, int n, int topk, bool per_expert, cudaStream_t);
void fnt_matvec_f32(bool mix, int kb, const void * w, const __half * sc, const __half * x, float * y, int rows, int cols, int n, cudaStream_t);
// dedup map (quant_kernels.cuh kDedupStride layout): per column the numbers of fnt_expert_matvec (bit-identical)
void fnt_expert_dedup(bool mix, int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                      const int * dmap, int groups, cudaStream_t);
// moe_group lists (counts [experts], groups [experts][capacity]); per column the numbers of fnt_expert_matvec
void fnt_expert_grouped(bool mix, int kb, const void * w, const __half * sc, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                        const int * counts, const int * groups, int capacity, int experts, int topk, bool per_expert, cudaStream_t);
// FP16 [rows][cols] in the plain basis
void fnt_dequant_h(bool mix, int kb, const void * w, const __half * sc, __half * out, int rows, int cols, cudaStream_t);
}

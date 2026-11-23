// Multi-query attention with per-query key ranges and an optional sliding window / ring cache (drafters: DFlash2,
// DSpark), head_dim 128 or 256. Moved from src/models/qwen/qwen35/attention.cu so components (src/spec) can use it.
#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdexcept>

#include "attention_gen.cuh"

#include "attention_gen_kernels.cuh"
using namespace tqllm_attn_gen;

namespace tqllm_cuda {

int attention_gen_partial_floats(int heads, int context, int nq) {
    return nq * heads * ((context + gen_chunk<128>() - 1) / gen_chunk<128>()) * (128 + 2);
}

void attention_gen128(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, const int * dpos, int nq, int max_seq, int h, int kh, float scale, bool causal,
                      int window, cudaStream_t s, int ring) {
    if (h % kh || h / kh > GEN_MAXG) throw std::invalid_argument("attention_gen128: <= 8 query heads per KV head");
    constexpr int CH = gen_chunk<128>();
    if (ring && (window <= 0 || ring % CH || ring < window + nq + CH))
        throw std::invalid_argument("attention_gen128: ring needs a window, ring % 64 == 0, ring >= window + nq + 64");
    const int nsplit = ring ? (window + nq + CH - 1) / CH + 1 : (max_seq + CH - 1) / CH;
    // Causal target decoding and verification must use the same reduction at every batch width.
    if (!causal && (h / kh) * nq <= 32)
        attn_gen_mq_k<128><<<dim3(kh, nsplit), 128, 0, s>>>(q, kc, vc, part, dpos, nq, h, kh, scale, nsplit,
                                                            causal ? 1 : 0, window, ring);
    else
        attn_gen_split_k<128><<<dim3(kh, nsplit, nq), 128, 0, s>>>(q, kc, vc, part, dpos, nq, h, kh, scale, nsplit,
                                                                   causal ? 1 : 0, window, ring);
    attn_gen_merge_k<128><<<dim3(h, nq), 128, 0, s>>>(part, y, nsplit, h);
}

void attention_gen256(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, const int * dpos, int nq, int max_seq, int h, int kh, float scale, bool causal,
                      cudaStream_t s) {
    if (h % kh || h / kh > GEN_MAXG) throw std::invalid_argument("attention_gen256: <= 8 query heads per KV head");
    const int nsplit = (max_seq + gen_chunk<256>() - 1) / gen_chunk<256>();
    attn_gen_split_k<256><<<dim3(kh, nsplit, nq), 128, 0, s>>>(q, kc, vc, part, dpos, nq, h, kh, scale, nsplit,
                                                               causal ? 1 : 0, 0, 0);
    attn_gen_merge_k<256><<<dim3(h, nq), 256, 0, s>>>(part, y, nsplit, h);
}

int attention_gen256_partial_floats(int heads, int context, int nq) {
    return nq * heads * ((context + gen_chunk<256>() - 1) / gen_chunk<256>()) * (256 + 2);
}

void attention_merge256(const float * part, __nv_bfloat16 * y, int nsplit, int heads, int nq, cudaStream_t s) {
    attn_gen_merge_k<256><<<dim3(heads, nq), 256, 0, s>>>(part, y, nsplit, heads);
}

}  // namespace tqllm_cuda

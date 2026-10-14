#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace tqllm_cuda {
void gather(const __nv_bfloat16 *, __nv_bfloat16 *, int, int, cudaStream_t);
void rms(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, float, cudaStream_t);
void bias(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, cudaStream_t);
void rope(__nv_bfloat16 *, int, int, int, float, cudaStream_t);
void cache_write(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, int, int, int, cudaStream_t);
void scores(const __nv_bfloat16 *, const __nv_bfloat16 *, float *, int, int, int, int, int, float, cudaStream_t);
void softmax(float *, int, int, int, cudaStream_t);
void attend(const float *, const __nv_bfloat16 *, __nv_bfloat16 *, int, int, int, int, int, cudaStream_t);
void residual(__nv_bfloat16 *, const __nv_bfloat16 *, int, cudaStream_t);
void swiglu(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, cudaStream_t);
// Fused split-KV decode attention (head_dim 64 or 256, <= 8 query heads per KV head). `part` needs
// attention_partial_floats(heads, context, dim) floats.
int attention_partial_floats(int heads, int context, int dim);
void attention_decode(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, int seq, int heads, int kvheads, int dim, float scale, cudaStream_t);
// nq queries (q, y: [nq][heads * dim]) at consecutive positions: query z attends to keys [0, seq0 + z). `part` needs
// nq * attention_partial_floats(heads, seq0 + nq - 1, dim) floats.
void attention_decode_multi(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                            __nv_bfloat16 * y, float * part, int seq0, int nq, int heads, int kvheads, int dim, float scale,
                            cudaStream_t);
// Same with the first query's sequence length read from the device (*dseq0) and the grid sized for max_seq
// (the longest sequence any query can have), so the launch can live in a CUDA graph.
void attention_decode_multi_dev(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                                __nv_bfloat16 * y, float * part, const int * dseq0, int max_seq, int nq, int heads,
                                int kvheads, int dim, float scale, cudaStream_t);
// head_dim 128 multi-query attention with per-query key ranges (see attention.cu): nq queries at positions
// *dpos + z; causal or block-non-causal; sliding window (0 = none). `part`: attention_gen_partial_floats floats.
// ring > 0 (needs a window): the cache holds position p at slot p % ring (ring a multiple of 64, >= window + nq + 64)
// and only the window is scanned, so cost and memory do not grow with the context.
int attention_gen_partial_floats(int heads, int context, int nq);
void attention_gen128(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, const int * dpos, int nq, int max_seq, int heads, int kvheads, float scale,
                      bool causal, int window, cudaStream_t, int ring = 0);
// The same for head_dim 256 (DSpark drafter), full context (no window / ring); part: attention_gen256_partial_floats.
void attention_gen256(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, const int * dpos, int nq, int max_seq, int heads, int kvheads, float scale,
                      bool causal, cudaStream_t);
int attention_gen256_partial_floats(int heads, int context, int nq);
// Multi-query attention for head_dim 256: query z sees keys [0, *dseq0 + z). Linear blocks are split to fit 48 rows.
// K/V tiles are
// loaded once per block for all queries; fixed split count (no context-dependent grid). part: attention_mq_partial_floats.
int attention_mq_partial_floats(int heads, int nq, int nsplit);
// par (device, nq entries, parent column or -1): two-part tree attention (BF16 K/V; partials need nsplit + 1 splits)
void attention_mq256(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                     float * part, const int * dseq0, int nq, int heads, int kvheads, float scale, int nsplit,
                     cudaStream_t, int query_offset = 0, const int * par = nullptr,
                     const int * dep = nullptr);  // dep: node depths (tree), or null for a chain
// Prefill chunk (any nq): the verify kernel's 8-query groups, many per launch (grid.z) as part_floats allows; per query
// the same arithmetic as attention_mq256 with query_offset
void attention_mq256_prefill(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                             __nv_bfloat16 * y, float * part, size_t part_floats, const int * dseq0, int nq, int heads,
                             int kvheads, float scale, int nsplit, cudaStream_t);
// Same with an int8 cache and one FP16 scale per 32 values: ksc / vsc [pos][kv head][8] (cache_write_multi_q8).
void attention_mq256_q8(const __nv_bfloat16 * q, const int8_t * kc, const int8_t * vc, const __half * ksc,
                        const __half * vsc, __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int heads,
                        int kvheads, float scale, int nsplit, cudaStream_t, int query_offset = 0);
void argmax(const float *, int, uint32_t *, cudaStream_t);
}

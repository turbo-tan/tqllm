#pragma once
// Multi-query attention with per-query key ranges for drafters (src/core/attention_gen.cu).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {
// head_dim 128 multi-query attention with per-query key ranges (see attention_gen.cu): nq queries at positions
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
// merge of attention_gen256-style partials (also used by the TQ3 KV verify path)
void attention_merge256(const float * part, __nv_bfloat16 * y, int nsplit, int heads, int nq, cudaStream_t);
}  // namespace tqllm_cuda

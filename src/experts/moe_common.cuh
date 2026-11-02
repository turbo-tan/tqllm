#pragma once
// MoE pieces shared by model engines (moved from the GLM-5.3 engine, 11-02, now used by GLM-5.3 and MiniMax-M3): sigmoid
// routing with a selection bias (plain, and fused with the router's split-k sum) and the weighted sum of the routed and
// shared expert outputs.
#include "common_kernels.cuh"

namespace tqllm_cuda::moe_common {
// sigmoid routing: p = sigmoid(logits); select top-k of p + bias; weights = p[sel] (normalised if `norm`, times `scale`
// unless it is 0 or 1). ids / w [k]
// n tokens: logits [n][experts], ids / w [n][k]
void route(const float * logits, const float * bias, int * ids, float * w, int experts, int k, bool norm, float scale,
               cudaStream_t s, int n = 1);
// the same routing over the split-k partials of a router matvec instead of over summed logits: partial
// [(t * experts + e) * splits + sp] is what a split-k router pass writes (mm_mv_f32_splitk). The block sums its
// expert's splits in the ascending order router_sum_k used and runs the same top-k walk, so logits / ids / w are the
// bits glm_route of those logits would give, with one launch fewer per layer. `logits` may be null.
void route_partial(const float * partial, float * logits, const float * bias, int * ids, float * w, int experts,
                       int k, bool norm, float scale, cudaStream_t s, int n, int splits);
// out[j] = shared[j] + sum_i w[i] y[i][j]   y BF16 [k][hidden]
// n tokens: y [n][k][hidden], w [n][k], shared / out [n][hidden]
void moe_sum(const __nv_bfloat16 * y, const float * w, const float * shared, float * out, int k, int hidden,
                 cudaStream_t s, int n = 1);
}  // namespace tqllm_cuda::moe_common

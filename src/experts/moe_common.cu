// Shared MoE kernels (moe_common.cuh): the GLM-5.3 engine's originals, unchanged.
#include "moe_common.cuh"
#include <cfloat>
#include <stdexcept>
#include <string>

namespace tqllm_cuda::moe_common {
using common::warp_sum; using common::block_sum; using common::block_max; using common::sigm;
namespace {
void check_launch(const char * what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + cudaGetErrorString(e));
}
int blocks(int n, int t) { return (n + t - 1) / t; }

// one block per token; experts <= 1024 (288 for GLM-5.3-Flash). Stable top-k: larger biased prob first, then lower
// index; each round is a block-wide argmax over (value, -index).
__global__ void route_k(const float * logits, const float * bias, int * ids, float * w, int experts, int k, bool norm,
                        float scale) {
    __shared__ float p[1024], sel[1024];
    __shared__ int chosen[64];
    __shared__ float rv[32];
    __shared__ int ri[32];
    logits += size_t(blockIdx.x) * experts; ids += blockIdx.x * k; w += blockIdx.x * k;
    for (int i = threadIdx.x; i < experts; i += blockDim.x) {
        p[i] = sigm(logits[i]);
        sel[i] = p[i] + (bias ? bias[i] : 0.f);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -FLT_MAX;
        int bi = 0x7fffffff;
        for (int i = threadIdx.x; i < experts; i += blockDim.x)
            if (sel[i] > bv || (sel[i] == bv && i < bi)) { bv = sel[i]; bi = i; }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = rv[0];
            int ix = ri[0];
            for (int q = 1; q < nw; ++q)
                if (rv[q] > v || (rv[q] == v && ri[q] < ix)) { v = rv[q]; ix = ri[q]; }
            chosen[r] = ix;
            sel[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += p[chosen[r]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            float v = p[chosen[r]];
            if (norm) v /= sum;
            if (scale != 0.f && scale != 1.f) v *= scale;
            ids[r] = chosen[r];
            w[r] = v;
        }
    }
}

// route_k over the split-k partials: the same sum router_sum_k did (the splits of one expert, ascending) and the same
// top-k walk, so the logits, the ids and the weights are the bits of the two-launch path with one launch fewer.
__global__ void route_partial_k(const float * __restrict__ partial, float * __restrict__ logits,
                                const float * __restrict__ bias, int * __restrict__ ids, float * __restrict__ w,
                                int experts, int k, bool norm, float scale, int splits) {
    __shared__ float p[1024], sel[1024];
    __shared__ int chosen[64];
    __shared__ float rv[32];
    __shared__ int ri[32];
    const float * pp = partial + size_t(blockIdx.x) * experts * splits;
    ids += blockIdx.x * k; w += blockIdx.x * k;
    if (logits) logits += size_t(blockIdx.x) * experts;
    for (int i = threadIdx.x; i < experts; i += blockDim.x) {
        float v = 0.f;
        for (int sp = 0; sp < splits; ++sp) v += pp[size_t(i) * splits + sp];
        if (logits) logits[i] = v;
        p[i] = sigm(v);
        sel[i] = p[i] + (bias ? bias[i] : 0.f);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -FLT_MAX;
        int bi = 0x7fffffff;
        for (int i = threadIdx.x; i < experts; i += blockDim.x)
            if (sel[i] > bv || (sel[i] == bv && i < bi)) { bv = sel[i]; bi = i; }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = rv[0];
            int ix = ri[0];
            for (int q = 1; q < nw; ++q)
                if (rv[q] > v || (rv[q] == v && ri[q] < ix)) { v = rv[q]; ix = ri[q]; }
            chosen[r] = ix;
            sel[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += p[chosen[r]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            float v = p[chosen[r]];
            if (norm) v /= sum;
            if (scale != 0.f && scale != 1.f) v *= scale;
            ids[r] = chosen[r];
            w[r] = v;
        }
    }
}

// route_partial_k with one warp per token and the top-k walk in registers. The block-per-token walk is latency, not
// work: at the decode router's shape (128 experts, k 8) it costs ~8 us per call, 0.5 ms of a 147 ms step and flat in n,
// because a round is a scan, a warp tree, two __syncthreads and thread 0's serial pick over the warp partials, with the
// other 255 threads idle in it. Here a lane owns ceil(experts / 32) candidates in registers (biased value, sigmoid,
// index) and a round is one warp reduction on the order the block version scans by - larger biased value first, ties to
// the lower index. That order is total, so the tree's max is the scan's max: the same expert wins every round, removing
// it is the same `sel = -FLT_MAX`, and the split sums (ascending), the sigmoids, the normalisation sum over the chosen
// in round order and the ids / weights are the block version's bits.
template <int MAXE>
__global__ void route_partial_warp_k(const float * __restrict__ partial, float * __restrict__ logits,
                                     const float * __restrict__ bias, int * __restrict__ ids, float * __restrict__ w,
                                     int experts, int k, bool norm, float scale, int splits, int tokens) {
    __shared__ int chosen[8][64];
    __shared__ float pv[8][64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tok = blockIdx.x * 8 + warp;
    if (tok >= tokens) return;
    const float * pp = partial + size_t(tok) * experts * splits;
    float sel[MAXE], p[MAXE];
    int ix[MAXE];
    for (int j = 0; j < MAXE; ++j) {
        const int e = lane + j * 32;
        if (e < experts) {
            float v = 0.f;
            for (int sp = 0; sp < splits; ++sp) v += pp[size_t(e) * splits + sp];
            if (logits) logits[size_t(tok) * experts + e] = v;
            p[j] = sigm(v);
            sel[j] = p[j] + (bias ? bias[e] : 0.f);
            ix[j] = e;
        } else { sel[j] = -FLT_MAX; p[j] = 0.f; ix[j] = 0x7fffffff; }
    }
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float bv = -FLT_MAX;
        int bi = 0x7fffffff;
        for (int j = 0; j < MAXE; ++j)
            if (sel[j] > bv || (sel[j] == bv && ix[j] < bi)) { bv = sel[j]; bi = ix[j]; }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float mine = 0.f;
        for (int j = 0; j < MAXE; ++j)
            if (ix[j] == bi) { mine = p[j]; sel[j] = -FLT_MAX; }
        const float got = __shfl_sync(0xffffffffu, mine, bi & 31);   // the lane that owns the winner holds its sigmoid
        sum += got;
        if (lane == 0) { chosen[warp][r] = bi; pv[warp][r] = got; }
    }
    if (lane == 0) {
        ids += tok * k; w += tok * k;
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            float v = pv[warp][r];
            if (norm) v /= sum;
            if (scale != 0.f && scale != 1.f) v *= scale;
            ids[r] = chosen[warp][r];
            w[r] = v;
        }
    }
}

__global__ void moe_sum_k(const __nv_bfloat16 * y, const float * w, const float * shared, float * out, int k, int hidden) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (j >= hidden) return;
    y += size_t(t) * k * hidden; w += t * k; out += size_t(t) * hidden;
    if (shared) shared += size_t(t) * hidden;
    float v = shared ? shared[j] : 0.f;
    for (int i = 0; i < k; ++i) v += w[i] * __bfloat162float(y[size_t(i) * hidden + j]);
    out[j] = v;
}
}  // namespace

void route(const float * logits, const float * bias, int * ids, float * w, int experts, int k, bool norm, float scale,
               cudaStream_t s, int n) {
    if (experts > 1024 || k > 64 || k > experts) throw std::invalid_argument("glm_route: experts <= 1024, k <= 64");
    route_k<<<n, 256, 0, s>>>(logits, bias, ids, w, experts, k, norm, scale);
    check_launch("moe_common kernel: ");
}
void route_partial(const float * partial, float * logits, const float * bias, int * ids, float * w, int experts,
                       int k, bool norm, float scale, cudaStream_t s, int n, int splits) {
    if (experts > 1024 || k > 64 || k > experts) throw std::invalid_argument("glm_route_partial: experts <= 1024, k <= 64");
    if (experts <= 128) {   // the decode router: eight tokens per block, one warp each, no barrier in the walk
        route_partial_warp_k<4><<<(n + 7) / 8, 256, 0, s>>>(partial, logits, bias, ids, w, experts, k, norm, scale, splits, n);
    } else {
        route_partial_k<<<n, 256, 0, s>>>(partial, logits, bias, ids, w, experts, k, norm, scale, splits);
    }
    check_launch("moe_common kernel: ");
}
void moe_sum(const __nv_bfloat16 * y, const float * w, const float * shared, float * out, int k, int hidden,
                 cudaStream_t s, int n) {
    moe_sum_k<<<dim3(blocks(hidden, 256), n), 256, 0, s>>>(y, w, shared, out, k, hidden);
    check_launch("moe_common kernel: ");
}
}  // namespace tqllm_cuda::moe_common

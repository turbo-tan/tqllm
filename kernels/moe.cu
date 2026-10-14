#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "moe_kernels.cuh"
#include <stdexcept>
#include <math_constants.h>
// TQFN_* switches that won their A/Bs are on by default; set NAME=0 to turn one off
static inline bool tqfn_default_on(const char * n) { const char * v = std::getenv(n); return !v || (v[0] && v[0] != '0'); }

namespace tqllm_cuda {
namespace {
__device__ float router_value(float value) { return value; }
__device__ float router_value(__half value) { return __half2float(value); }
template<class Input> __global__ void router_k(const float * w, const float * shared, const Input * x, float * logits,
                         int h, int e) {
    const int row = blockIdx.x, token = blockIdx.y;
    const float * wr = row == e ? shared : w + int64_t(row) * h;
    const Input * xr = x + int64_t(token) * h;
    float sum = 0.f;
    for (int j = threadIdx.x; j < h; j += blockDim.x) sum += wr[j] * router_value(xr[j]);
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    __shared__ float partial[8];
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0.f;
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (threadIdx.x == 0) logits[int64_t(token) * (e + 1) + row] = sum;
    }
}

__global__ void topk_k(const float * logits, int * ids, float * weights, float * shared_weights,
                       int e, int k, float scale) {
    // 256 threads. Reductions use warp shuffles; the softmax sum keeps the same pairing order as a 256-wide tree
    // (128, 64, 32 in shared memory, then 16..1 within warp 0), so every probability is bit-identical. Max and argmax
    // (ties -> lower index) do not depend on the order.
    const int token = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float * x = logits + int64_t(token) * (e + 1);
    extern __shared__ float probs[];
    __shared__ float reduce[256];
    __shared__ int indices[8];
    __shared__ float selected_sum;
    float m = -CUDART_INF_F;
    for (int j = tid; j < e; j += blockDim.x) m = fmaxf(m, x[j]);
    for (int d = 16; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    if (!lane) reduce[warp] = m;
    __syncthreads();
    m = reduce[0]; for (int w = 1; w < 8; ++w) m = fmaxf(m, reduce[w]);
    __syncthreads();
    float sum = 0.f;
    for (int j = tid; j < e; j += blockDim.x) { probs[j] = expf(x[j] - m); sum += probs[j]; }
    reduce[tid] = sum;
    __syncthreads();
    for (int d = 128; d >= 32; d >>= 1) { if (tid < d) reduce[tid] += reduce[tid + d]; __syncthreads(); }
    if (warp == 0) {
        sum = reduce[tid];
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (!lane) reduce[0] = sum;
    }
    __syncthreads();
    sum = reduce[0];
    for (int j = tid; j < e; j += blockDim.x) probs[j] /= sum;
    if (!tid) selected_sum = 0.f;
    __syncthreads();
    for (int i = 0; i < k; ++i) {
        float value = -CUDART_INF_F; int best = e;
        for (int j = tid; j < e; j += blockDim.x)
            if (probs[j] > value || (probs[j] == value && j < best)) { value = probs[j]; best = j; }
        for (int d = 16; d; d >>= 1) {
            const float v = __shfl_down_sync(0xffffffffu, value, d); const int b = __shfl_down_sync(0xffffffffu, best, d);
            if (v > value || (v == value && b < best)) { value = v; best = b; }
        }
        if (!lane) { reduce[warp] = value; indices[warp] = best; }
        __syncthreads();
        if (!tid) {
            value = reduce[0]; best = indices[0];
            for (int w = 1; w < 8; ++w) if (reduce[w] > value || (reduce[w] == value && indices[w] < best)) { value = reduce[w]; best = indices[w]; }
            ids[int64_t(token) * k + i] = best;
            weights[int64_t(token) * k + i] = probs[best];
            selected_sum += probs[best];
            probs[best] = -1.f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float total = fmaxf(selected_sum, 6.103515625e-5f);
        const float factor = (scale == 0.f ? 1.f : scale) / total;
        for (int i = 0; i < k; ++i) weights[int64_t(token) * k + i] *= factor;
        shared_weights[token] = 1.f / (1.f + expf(-x[e]));
    }
}

// One warp per token, bit-identical to topk_k: lane l plays the 256-thread block's threads l + 32 q (q < 8), with the
// same per-thread sums (j = t, t + 256, ...), the same 128/64/32 tree and the same final warp reduction; the argmax rule
// (larger value, ties -> lower index) has one answer whatever the order. No block barriers.
template <int R>
__global__ void topk_warp_k(const float * logits, int * ids, float * weights, float * shared_weights, int e, int k, float scale) {
    const int token = blockIdx.x, lane = threadIdx.x;
    const float * x = logits + int64_t(token) * (e + 1);
    float p[8][R];
    float m = -CUDART_INF_F;
#pragma unroll
    for (int q = 0; q < 8; ++q)
#pragma unroll
        for (int r = 0; r < R; ++r) { const int j = lane + 32 * q + 256 * r; p[q][r] = j < e ? x[j] : -CUDART_INF_F; m = fmaxf(m, p[q][r]); }
    for (int d = 16; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    float s[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        s[q] = 0.f;
#pragma unroll
        for (int r = 0; r < R; ++r) { const int j = lane + 32 * q + 256 * r; if (j < e) { p[q][r] = expf(p[q][r] - m); s[q] += p[q][r]; } }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) s[q] += s[q + 4];
#pragma unroll
    for (int q = 0; q < 2; ++q) s[q] += s[q + 2];
    float sum = s[0] + s[1];
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    sum = __shfl_sync(0xffffffffu, sum, 0);
#pragma unroll
    for (int q = 0; q < 8; ++q)
#pragma unroll
        for (int r = 0; r < R; ++r) { const int j = lane + 32 * q + 256 * r; p[q][r] = j < e ? p[q][r] / sum : -2.f; }
    float selected = 0.f;
    for (int i = 0; i < k; ++i) {
        float value = -CUDART_INF_F; int best = e;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int j = lane + 32 * q + 256 * r;
                if (j < e && (p[q][r] > value || (p[q][r] == value && j < best))) { value = p[q][r]; best = j; }
            }
        for (int d = 16; d; d >>= 1) {
            const float v = __shfl_xor_sync(0xffffffffu, value, d); const int b = __shfl_xor_sync(0xffffffffu, best, d);
            if (v > value || (v == value && b < best)) { value = v; best = b; }
        }
        if (!lane) { ids[int64_t(token) * k + i] = best; weights[int64_t(token) * k + i] = value; }
        selected += value;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int q = 0; q < 8; ++q) if (lane + 32 * q + 256 * r == best) p[q][r] = -1.f;
    }
    if (!lane) {
        const float total = fmaxf(selected, 6.103515625e-5f);
        const float factor = (scale == 0.f ? 1.f : scale) / total;
        for (int i = 0; i < k; ++i) weights[int64_t(token) * k + i] *= factor;
        shared_weights[token] = 1.f / (1.f + expf(-x[e]));
    }
}

__global__ void group_k(const int * ids, int * counts, int * groups, int n, int k) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group < n * k) {
        const int expert = ids[group];
        if (expert < 0) return;
        const int slot = atomicAdd(counts + expert, 1);
        groups[int64_t(expert) * n + slot] = group;
    }
}

__global__ void split_ids_k(const int * ids, const int * hot_slot, const int * cold_slot, int * hot_ids, int * cold_ids,
                            int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { const int e = ids[i]; hot_ids[i] = hot_slot[e]; cold_ids[i] = cold_slot[e]; }
}
__global__ void count_k(const int * ids, unsigned long long * counts, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) atomicAdd(counts + ids[i], 1ull);
}

__global__ void swiglu_k(const __nv_bfloat16 * gate, const __nv_bfloat16 * up, __half * act, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float g = __bfloat162float(gate[i]), u = __bfloat162float(up[i]);
        const auto v = __float2bfloat16_rn(g / (1.f + expf(-g)) * u);
        act[i] = __float2half_rn(__bfloat162float(v));
    }
}

__global__ void sum_k(const __nv_bfloat16 * routed, const __nv_bfloat16 * shared, const float * weights,
                      const float * shared_weights, __nv_bfloat16 * out, int h, int k, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int token = i / h, j = i % h;
        float sum = 0.f;
        for (int s = 0; s < k; ++s)
            sum += weights[int64_t(token) * k + s] * __bfloat162float(routed[(int64_t(token) * k + s) * h + j]);
        sum += shared_weights[token] * __bfloat162float(shared[i]);
        out[i] = __float2bfloat16_rn(sum);
    }
}
}  // namespace

void moe_router(const float * router, const float * shared, const __half * x, float * logits,
                int h, int e, int n, cudaStream_t stream) {
    if (h <= 0 || e <= 0 || n <= 0) throw std::invalid_argument("moe_router: invalid shape");
    router_k<<<dim3(unsigned(e + 1), unsigned(n)), 256, 0, stream>>>(router, shared, x, logits, h, e);
}
void moe_router_f32(const float * router, const float * shared, const float * x, float * logits,
                    int h, int e, int n, cudaStream_t stream) {
    if (h <= 0 || e <= 0 || n <= 0 || !router || !shared || !x || !logits)
        throw std::invalid_argument("moe_router_f32: invalid shape or pointer");
    router_k<<<dim3(unsigned(e + 1), unsigned(n)), 256, 0, stream>>>(router, shared, x, logits, h, e);
}
void moe_topk(const float * logits, int * ids, float * weights, float * shared_weights,
              int e, int k, int n, float scale, cudaStream_t stream) {
    if (e <= 0 || e > 4096 || k <= 0 || k > e || n <= 0)
        throw std::invalid_argument("moe_topk: invalid shape");
    moe_topk_variant(logits, ids, weights, shared_weights, e, k, n, scale, true, stream);  // one warp per token
}
void moe_topk_variant(const float * logits, int * ids, float * weights, float * shared_weights,
                      int e, int k, int n, float scale, bool warp, cudaStream_t stream) {
    if (e <= 0 || e > 4096 || k <= 0 || k > e || n <= 0) throw std::invalid_argument("moe_topk: invalid shape");
    if (warp && e <= 512) topk_warp_k<2><<<n, 32, 0, stream>>>(logits, ids, weights, shared_weights, e, k, scale);
    else if (warp && e <= 1024) topk_warp_k<4><<<n, 32, 0, stream>>>(logits, ids, weights, shared_weights, e, k, scale);
    else topk_k<<<n, 256, size_t(e) * sizeof(float), stream>>>(logits, ids, weights, shared_weights, e, k, scale);
}
void moe_group(const int * ids, int * counts, int * groups, int e, int n, int k, cudaStream_t stream) {
    if (e <= 0 || n <= 0 || k <= 0) throw std::invalid_argument("moe_group: invalid shape");  // k > e is fine: ids outside the set are -1
    const auto rc = cudaMemsetAsync(counts, 0, size_t(e) * sizeof(int), stream);
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
    group_k<<<(n * k + 255) / 256, 256, 0, stream>>>(ids, counts, groups, n, k);
}
void moe_split_ids(const int * ids, const int * hot_slot, const int * cold_slot, int * hot_ids, int * cold_ids,
                   int count, cudaStream_t stream) {
    if (count <= 0) return;
    split_ids_k<<<(count + 255) / 256, 256, 0, stream>>>(ids, hot_slot, cold_slot, hot_ids, cold_ids, count);
}
void moe_count(const int * ids, unsigned long long * counts, int count, cudaStream_t stream) {
    if (count <= 0) return;
    count_k<<<(count + 255) / 256, 256, 0, stream>>>(ids, counts, count);
}
void moe_swiglu(const __nv_bfloat16 * gate, const __nv_bfloat16 * up, __half * act, int count, cudaStream_t stream) {
    swiglu_k<<<(count + 255) / 256, 256, 0, stream>>>(gate, up, act, count);
}
void moe_sum(const __nv_bfloat16 * routed, const __nv_bfloat16 * shared, const float * weights,
             const float * shared_weights, __nv_bfloat16 * out, int h, int k, int n, cudaStream_t stream) {
    sum_k<<<(n * h + 255) / 256, 256, 0, stream>>>(routed, shared, weights, shared_weights, out, h, k, n * h);
}

struct ScatterBatch { int n; ScatterItem it[kScatterMax]; };
// grid (n, 8): 8 blocks share one piece; 16-byte copies when source, destination and size allow, bytes otherwise
__global__ void scatter_copy_k(ScatterBatch b) {
    const ScatterItem e = b.it[blockIdx.x];
    const unsigned long long part = blockIdx.y, parts = gridDim.y;
    const bool vec = ((reinterpret_cast<uintptr_t>(e.src) | reinterpret_cast<uintptr_t>(e.dst) | e.bytes) & 15) == 0;
    if (vec) {
        const uint4 * s = reinterpret_cast<const uint4 *>(e.src);
        uint4 * d = reinterpret_cast<uint4 *>(e.dst);
        const unsigned long long n16 = e.bytes / 16;
        for (unsigned long long i = part * blockDim.x + threadIdx.x; i < n16; i += parts * blockDim.x) d[i] = s[i];
    } else {
        for (unsigned long long i = part * blockDim.x + threadIdx.x; i < e.bytes; i += parts * blockDim.x) e.dst[i] = e.src[i];
    }
}
void scatter_copy(const ScatterItem * items, int n, cudaStream_t stream, int parts) {
    for (int off = 0; off < n; off += kScatterMax) {
        ScatterBatch b;
        b.n = std::min(kScatterMax, n - off);
        for (int i = 0; i < b.n; ++i) b.it[i] = items[off + i];
        scatter_copy_k<<<dim3(unsigned(b.n), unsigned(parts)), 256, 0, stream>>>(b);
    }
}
// Small host data (slot and dedup maps) carried in the kernel parameters: no read of host memory, no copy engine.
template <int BYTES> struct ParamBlob { unsigned n; uint4 w[BYTES / 16]; };
template <int BYTES> __global__ void param_copy_k(ParamBlob<BYTES> b, uint4 * dst) { for (unsigned i = threadIdx.x; i < b.n; i += blockDim.x) dst[i] = b.w[i]; }
template <int BYTES> static void param_launch(const char * src, size_t len, uint4 * dst, cudaStream_t stream) {
    ParamBlob<BYTES> b; b.n = unsigned(len / 16); std::memcpy(b.w, src, len);
    param_copy_k<BYTES><<<1, 256, 0, stream>>>(b, dst);
}
void param_copy(const void * host, size_t bytes, void * dst, cudaStream_t stream) {
    if (bytes % 16 || reinterpret_cast<uintptr_t>(dst) % 16) throw std::invalid_argument("param_copy: 16-byte multiples only");
    for (size_t off = 0; off < bytes; off += kParamCopyMax) {
        const size_t len = std::min(bytes - off, size_t(kParamCopyMax));
        const char * src = static_cast<const char *>(host) + off; uint4 * d = reinterpret_cast<uint4 *>(static_cast<char *>(dst) + off);
        if (len <= 1024) param_launch<1024>(src, len, d, stream);
        else if (len <= 2048) param_launch<2048>(src, len, d, stream);
        else if (len <= 4096) param_launch<4096>(src, len, d, stream);
        else param_launch<kParamCopyMax>(src, len, d, stream);
    }
}
}  // namespace tqllm_cuda

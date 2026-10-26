// GLM-5.3 DSA k-pool indexer and sparse MLA (glm5next_kernels.cuh). A token's indexer key (LayerNorm of attn_k x) and
// gate logits go to a per-layer FP16 cache; every r consecutive tokens form a pool whose key is the channel-wise
// softmax(gate + ape) mix of its members' keys, built when the pool's last member is written. A query at position q
// scores the pools that end at or before q, sum_h w_h relu(q_h . K_pool), keeps the top `pools` of them (all when fewer)
// and attends to their member cells plus the trailing incomplete pool [floor((q + 1) / r) r, q].
#include "glm5next_kernels.cuh"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

#include <cfloat>
#include <stdexcept>
#include <vector>

namespace tqllm_cuda {
namespace {

void check_launch() {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm5next dsa kernel: ") + cudaGetErrorString(e));
}
__device__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    return warp_sum(t);
}
__device__ float block_max(float v, float * red) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : -FLT_MAX;
    for (int o = 16; o; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    return t;
}

// one block per token, D threads: LayerNorm(k) and g as FP16 into cache row *d_pos + t ([k | g], 2D halves)
__global__ void kg_store_k(const float * k, const float * g, const float * w, const float * b, __half * kg, int D, float eps,
                           const int * d_pos, int pos0, int ring) {
    __shared__ float red[32];
    const int t = blockIdx.x, j = threadIdx.x;
    const float x = j < D ? k[size_t(t) * D + j] : 0.f;
    const float mean = block_sum(x, red) / float(D);
    const float d = j < D ? x - mean : 0.f;
    const float var = block_sum(d * d, red) / float(D);
    if (j >= D) return;
    const size_t row = size_t((d_pos ? *d_pos : pos0) + t) % size_t(ring);
    kg[row * 2 * D + j] = __float2half(d * rsqrtf(var + eps) * w[j] + b[j]);
    kg[row * 2 * D + D + j] = __float2half(g[size_t(t) * D + j]);
}

// one block per token, D threads: a token that completes a pool writes the pool key
__global__ void pool_k(const __half * kg, const float * ape, float * pooled, int D, int r, const int * d_pos, int pos0, int ring) {
    const int p = (d_pos ? *d_pos : pos0) + int(blockIdx.x), c = threadIdx.x;
    if (p % r != r - 1 || c >= D) return;
    const int b = p / r;
    float logit[8], m = -FLT_MAX;
    for (int i = 0; i < r; ++i) {
        logit[i] = __half2float(kg[((size_t(b) * r + i) % size_t(ring)) * 2 * D + D + c]) + ape[size_t(i) * D + c];
        m = fmaxf(m, logit[i]);
    }
    float z = 0.f, acc = 0.f;
    for (int i = 0; i < r; ++i) {
        const float e = expf(logit[i] - m);
        z += e;
        acc += e * __half2float(kg[((size_t(b) * r + i) % size_t(ring)) * 2 * D + c]);
    }
    pooled[size_t(b) * D + c] = acc / z;
}

// grid (ceil(P / 8), tokens), 8 warps; warp = pool, lane = head (32 heads): score[t][b] = sum_h w_h relu(q_h . K_b),
// -inf for a pool the token may not pick
__global__ void score_k(const float * q, const float * w, const float * pooled, float * score, int D, int r, int P,
                        int pos0, float wscale) {
    extern __shared__ float qs[];  // [32][D + 1]
    const int t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < 32 * D; i += blockDim.x) qs[(i / D) * (D + 1) + i % D] = q[size_t(t) * 32 * D + i];
    __syncthreads();
    const int b = blockIdx.x * 8 + warp;
    if (b >= P) return;
    const int pick = (pos0 + t + 1) / r;
    float s = -INFINITY;
    if (b < pick) {
        const float * kb = pooled + size_t(b) * D;
        const float * qh = qs + lane * (D + 1);
        float acc = 0.f;
        for (int d = 0; d < D; ++d) acc += qh[d] * kb[d];
        s = warp_sum(w[size_t(t) * 32 + lane] * wscale * fmaxf(acc, 0.f));
    }
    if (lane == 0) score[size_t(t) * P + b] = s;
}

__global__ void iota_k(int * v, int P, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) v[i] = i % P;
}
__global__ void offsets_k(int * off, int n, int P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) off[i] = i * P;
}

// one block per token: cells = the members of the first min(pickable, pools) sorted pools, then the tail
__global__ void cells_k(const int * order, int * cells, int * count, int r, int P, int pools, int pos0, int stride) {
    const int t = blockIdx.x, q = pos0 + t, pick = (q + 1) / r, k = min(pick, pools);
    int * out = cells + size_t(t) * stride;
    for (int i = threadIdx.x; i < k * r; i += blockDim.x) out[i] = order[size_t(t) * P + i / r] * r + i % r;
    const int tail0 = pick * r;
    for (int p = tail0 + int(threadIdx.x); p <= q; p += blockDim.x) out[k * r + (p - tail0)] = p;
    if (threadIdx.x == 0) count[t] = k * r + (q - tail0 + 1);
}

// grid (heads, tokens): softmax over the token's cells of qabs . latent, then the latent mix (mla_attn_k on a cell list)
__device__ __forceinline__ float lat_f(float x) { return x; }
__device__ __forceinline__ float lat_f(__half x) { return __half2float(x); }
template <class LT>
__global__ void mla_sparse_k(const float * qabs, const LT * lat, float * kv, int R, const int * cells, const int * count,
                             int stride, float scale) {
    extern __shared__ float sh[];
    __shared__ float red[32];
    float * qa = sh, * p = sh + R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int h = blockIdx.x + blockIdx.y * gridDim.x, t = blockIdx.y, n = count[t];
    const int * cl = cells + size_t(t) * stride;
    for (int j = threadIdx.x; j < R; j += blockDim.x) qa[j] = qabs[size_t(h) * R + j];
    __syncthreads();
    for (int i = warp; i < n; i += nw) {
        const LT * row = lat + size_t(cl[i]) * R;
        float acc = 0.f;
        for (int j = lane; j < R; j += 32) acc += qa[j] * lat_f(row[j]);
        acc = warp_sum(acc);
        if (lane == 0) p[i] = acc * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, p[i]);
    m = block_max(m, red);
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += (p[i] = expf(p[i] - m));
    z = block_sum(z, red);
    __syncthreads();
    for (int j = threadIdx.x; j < R; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n; ++i) acc += p[i] * lat_f(lat[size_t(cl[i]) * R + j]);
        kv[size_t(h) * R + j] = acc / z;
    }
}

}  // namespace

void glm_dsa_store(const float * k, const float * g, const float * ln_w, const float * ln_b, __half * kg, const float * ape,
                   float * pooled, int D, int r, float eps, int n, const int * d_pos, int pos0, int ring, cudaStream_t s) {
    if (D > 1024 || r > 8 || D % 32) throw std::invalid_argument("glm_dsa_store: unsupported indexer shape");
    if (ring < n + r) throw std::invalid_argument("glm_dsa_store: key ring shorter than a call plus a pool");
    kg_store_k<<<n, D, 0, s>>>(k, g, ln_w, ln_b, kg, D, eps, d_pos, pos0, ring);
    check_launch();
    pool_k<<<n, D, 0, s>>>(kg, ape, pooled, D, r, d_pos, pos0, ring);
    check_launch();
}

size_t glm_dsa_select_bytes(int tokens, int P) {
    size_t temp = 0;
    cub::DeviceSegmentedSort::StableSortPairsDescending(nullptr, temp, static_cast<const float *>(nullptr),
        static_cast<float *>(nullptr), static_cast<const int *>(nullptr), static_cast<int *>(nullptr), tokens * P, tokens,
        static_cast<const int *>(nullptr), static_cast<const int *>(nullptr) + 1);
    // scores, sorted scores, pool ids in / out, segment offsets, then cub's own scratch
    return size_t(tokens) * P * 16 + size_t(tokens + 1) * 4 + temp + 1024;
}

void glm_dsa_select(const float * q, const float * w, const float * pooled, int D, int r, int pools, int tokens, int pos0,
                    void * scratch, size_t scratch_bytes, int * cells, int * count, int stride, cudaStream_t s) {
    const int P = (pos0 + tokens) / r;  // pools any token of the group may pick
    if (P <= 0 || tokens <= 0) throw std::invalid_argument("glm_dsa_select: nothing to select from");
    if (stride < pools * r + r - 1) throw std::invalid_argument("glm_dsa_select: cell stride too small");
    char * base = static_cast<char *>(scratch);
    float * score = reinterpret_cast<float *>(base);
    float * sorted = score + size_t(tokens) * P;
    int * ids = reinterpret_cast<int *>(sorted + size_t(tokens) * P);
    int * order = ids + size_t(tokens) * P;
    int * off = order + size_t(tokens) * P;
    void * temp = off + tokens + 1;
    const size_t used = static_cast<size_t>(static_cast<char *>(temp) - base);
    if (used > scratch_bytes) throw std::invalid_argument("glm_dsa_select: scratch too small");
    size_t temp_bytes = scratch_bytes - used;
    score_k<<<dim3(unsigned((P + 7) / 8), unsigned(tokens)), 256, size_t(32) * (D + 1) * sizeof(float), s>>>(
        q, w, pooled, score, D, r, P, pos0, 1.f / sqrtf(float(D) * 32.f));
    check_launch();
    iota_k<<<unsigned((size_t(tokens) * P + 255) / 256), 256, 0, s>>>(ids, P, tokens * P);
    offsets_k<<<unsigned((tokens + 256) / 256), 256, 0, s>>>(off, tokens, P);
    check_launch();
    if (cub::DeviceSegmentedSort::StableSortPairsDescending(temp, temp_bytes, score, sorted, ids, order, tokens * P, tokens,
                                                            off, off + 1, s) != cudaSuccess)
        throw std::runtime_error("glm_dsa_select: segmented sort failed");
    cells_k<<<tokens, 256, 0, s>>>(order, cells, count, r, P, pools, pos0, stride);
    check_launch();
}

template <class LT>
void mla_sparse(const float * qabs, const LT * lat, float * kv, int heads, int R, const int * cells, const int * count, int stride,
                float scale, int tokens, cudaStream_t s) {
    const size_t sh = size_t(R + stride) * sizeof(float);
    if (sh > 48 * 1024) throw std::invalid_argument("glm_mla_attn_sparse: cell list beyond shared memory");
    mla_sparse_k<LT><<<dim3(unsigned(heads), unsigned(tokens)), 256, sh, s>>>(qabs, lat, kv, R, cells, count, stride, scale);
    check_launch();
}
void glm_mla_attn_sparse(const float * qabs, const float * lat, float * kv, int heads, int R, const int * cells,
                         const int * count, int stride, float scale, int tokens, cudaStream_t s) {
    mla_sparse(qabs, lat, kv, heads, R, cells, count, stride, scale, tokens, s);
}
void glm_mla_attn_sparse(const float * qabs, const __half * lat, float * kv, int heads, int R, const int * cells,
                         const int * count, int stride, float scale, int tokens, cudaStream_t s) {
    mla_sparse(qabs, lat, kv, heads, R, cells, count, stride, scale, tokens, s);
}

}  // namespace tqllm_cuda

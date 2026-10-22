#include <cstdlib>
#include "dflash_kernels.cuh"

#include <math_constants.h>

namespace {

__constant__ float Q_SIGNS[32] = {+1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
                                  -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1};
__constant__ float Q_CB[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                              0.230106f,  0.725222f,  1.277503f,  1.988943f};
__constant__ float Q_BOUND[7] = {-1.644041f, -1.015870f, -0.493924f, -0.008701f, 0.477664f, 1.001362f, 1.633223f};

__device__ __forceinline__ float grp8_sum(float v) {  // sum over the 8 lanes of an aligned group
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ uint32_t choose(float v) {
    uint32_t idx = 0;
#pragma unroll
    for (int b = 0; b < 7; ++b) idx += v > Q_BOUND[b] ? 1u : 0u;
    return idx;
}

__device__ uint32_t encode_e3m5(float val) {
    if (!(val > 0.f)) return 0;
    int e = int(floorf(log2f(val))) + 9;
    e = e < 0 ? 0 : (e > 7 ? 7 : e);
    const float mant = val / ldexpf(1.f, e - 9) - 1.f;
    int m = int(rintf(mant * 32.f));
    m = m < 0 ? 0 : (m > 31 ? 31 : m);
    return uint32_t((e << 5) | m);
}

// One warp per 32-weight block, lane = element.
__global__ void tq3_quantize_k(const float * x, uint4 * out, int64_t nblocks) {
    const int64_t b = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (b >= nblocks) return;
    float v = x[b * 32 + lane] * Q_SIGNS[lane];
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, s);
        v = (lane & s) ? o - v : v + o;
    }
    v *= 0.17677669529663687f;
    float scale = fmaxf(sqrtf(grp8_sum(v * v) / 8.f), 1e-10f);
    uint32_t idx = 0;
    for (int it = 0; it < 4; ++it) {
        idx = choose(v / fmaxf(scale, 1e-10f));
        const float c = Q_CB[idx];
        const float numer = grp8_sum(v * c), denom = grp8_sum(c * c);
        if (denom > 1e-12f) scale = fmaxf(numer / denom, 1e-10f);
    }
    // the reference keeps the indices of the last refinement pass (not re-chosen for the final scale)
    const int g = lane / 8, j = lane % 8;
    uint32_t word = idx << (3 * j);  // 24-bit group word
    word |= __shfl_xor_sync(0xffffffffu, word, 1);
    word |= __shfl_xor_sync(0xffffffffu, word, 2);
    word |= __shfl_xor_sync(0xffffffffu, word, 4);
    const uint32_t sb = encode_e3m5(scale) << (8 * g);
    const uint32_t g0 = __shfl_sync(0xffffffffu, word, 0), g1 = __shfl_sync(0xffffffffu, word, 8);
    const uint32_t g2 = __shfl_sync(0xffffffffu, word, 16), g3 = __shfl_sync(0xffffffffu, word, 24);
    const uint32_t scales = __shfl_sync(0xffffffffu, sb, 0) | __shfl_sync(0xffffffffu, sb, 8) |
                            __shfl_sync(0xffffffffu, sb, 16) | __shfl_sync(0xffffffffu, sb, 24);
    if (lane == 0) out[b] = make_uint4(scales, g0 | (g1 << 24), (g1 >> 8) | (g2 << 16), (g2 >> 16) | (g3 << 8));
}

__global__ void dflash_conv_k(const __nv_bfloat16 * h, const __nv_bfloat16 * dyn, int ds, const float * base, int side,
                              __nv_bfloat16 * y, int C, int group) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int G = C / group, gi = c / group;
    float acc = 0.f;
#pragma unroll
    for (int tap = 0; tap < 2; ++tap) {
        if (t - tap < 0) continue;
        const int ci = tap + 2 * side;
        const float w = __bfloat162float(dyn[int64_t(t) * ds + gi + G * ci]) + base[c + int64_t(C) * ci];
        acc += w * __bfloat162float(h[int64_t(t - tap) * C + c]);
    }
    y[int64_t(t) * C + c] = __float2bfloat16_rn(acc);
}

// One block per row; 16 selection rounds, each an argmax over values strictly below the previous pick.
__global__ void topk16_k(const float * x, int len, float * vals, uint32_t * idx) {
    __shared__ float sv[8];
    __shared__ uint32_t si[8];
    __shared__ float last_v;
    __shared__ uint32_t last_i;
    const float * xr = x + int64_t(blockIdx.x) * len;
    if (threadIdx.x == 0) {
        last_v = CUDART_INF_F;
        last_i = 0xffffffffu;
    }
    __syncthreads();
    for (int r = 0; r < 16; ++r) {
        float bv = -CUDART_INF_F;
        uint32_t bi = 0xffffffffu;
        const float lv = last_v;
        const uint32_t li = last_i;
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            const float v = xr[i];
            const bool below = v < lv || (v == lv && uint32_t(i) > li);  // strictly after the last pick
            if (below && (v > bv || (v == bv && uint32_t(i) < bi))) {
                bv = v;
                bi = i;
            }
        }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (threadIdx.x % 32 == 0) {
            sv[threadIdx.x / 32] = bv;
            si[threadIdx.x / 32] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < 8; ++w)
                if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
                    bv = sv[w];
                    bi = si[w];
                }
            vals[blockIdx.x * 16 + r] = bv;
            idx[blockIdx.x * 16 + r] = bi;
            last_v = bv;
            last_i = bi;
        }
        __syncthreads();
    }
}

// grid (n - 1 positions, 16 predecessors), block = rank threads (256). Writes score[i][j][k] for positions i >= 1.
__global__ void dflash_lattice_k(const uint32_t * cand, const float * unary, const __nv_bfloat16 * gate, int gs,
                                 const __nv_bfloat16 * next, const __nv_bfloat16 * prev, uint32_t anchor,
                                 const uint32_t * anchor_dev, int rank,
                                 float * score) {
    __shared__ float red[32];
    const int i = blockIdx.x + 1, j = blockIdx.y, r = threadIdx.x;
    const uint32_t p = i == 1 ? (anchor_dev ? *anchor_dev : anchor) : cand[(i - 1) * 16 + j];
    const float cond = __bfloat162float(prev[int64_t(p) * rank + r]) * __bfloat162float(gate[int64_t(i) * gs + r]);
    for (int k = 0; k < 16; ++k) {
        float v = __bfloat162float(next[int64_t(cand[i * 16 + k]) * rank + r]) * cond;
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (r % 32 == 0) red[r / 32] = v;
        __syncthreads();
        if (r == 0) {
            float t = 0.f;
            for (int w = 0; w < rank / 32; ++w) t += red[w];
            score[(int64_t(i) * 16 + j) * 16 + k] = t + unary[i * 16 + k];
        }
        __syncthreads();
    }
}

} // namespace

namespace {
__global__ void gather_rows_k(const uint8_t * src, uint8_t * dst, const uint32_t * ids, int64_t row_bytes) {
    const int64_t r = blockIdx.x;
    const uint8_t * s = src + int64_t(ids[r]) * row_bytes;
    uint8_t * d = dst + r * row_bytes;
    for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) d[i] = s[i];
}
__global__ void map_ids_k(uint32_t * idx, const uint32_t * ids, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) idx[i] = ids[idx[i]];
}
} // namespace

namespace tqllm_cuda {

void tq3_quantize(const float * x, void * out, int64_t nblocks, cudaStream_t s) {
    tq3_quantize_k<<<int((nblocks * 32 + 255) / 256), 256, 0, s>>>(x, static_cast<uint4 *>(out), nblocks);
}

void dflash_conv(const __nv_bfloat16 * h, const __nv_bfloat16 * dyn, int dyn_stride, const float * base, int side,
                 __nv_bfloat16 * y, int n, int channels, int group, cudaStream_t s) {
    dflash_conv_k<<<dim3((channels + 255) / 256, n), 256, 0, s>>>(h, dyn, dyn_stride, base, side, y, channels, group);
}

void topk16(const float * x, int len, int rows, float * vals, uint32_t * idx, cudaStream_t s) {
    topk16_k<<<rows, 256, 0, s>>>(x, len, vals, idx);
}

void dflash_lattice(const uint32_t * cand, const float * unary, const __nv_bfloat16 * gate, int gate_stride,
                    const __nv_bfloat16 * next, const __nv_bfloat16 * prev, uint32_t anchor, int n, int rank,
                    float * score, cudaStream_t s, const uint32_t * anchor_dev) {
    if (n < 2) return;
    dflash_lattice_k<<<dim3(n - 1, 16), rank, 0, s>>>(cand, unary, gate, gate_stride, next, prev, anchor, anchor_dev, rank,
                                                      score);
}

namespace {
// One block per descriptor: row d <- row cols[d] for d = 1 .. k - 1 in order (cols[d] >= d, so no source row is
// overwritten before it is read); rows start at base + (use_pos ? pos : 0) * row_bytes.
__global__ void gather_path_k(const tqllm_cuda::PathGather * g, const int * cols, int k, int pos) {
    const auto & dsc = g[blockIdx.x];
    uint8_t * base = static_cast<uint8_t *>(dsc.base) + (dsc.use_pos ? int64_t(pos) * dsc.row_bytes : 0);
    for (int d = 1; d < k; ++d) {
        const int c = cols[d];
        if (c != d) {
            const uint4 * s = reinterpret_cast<const uint4 *>(base + int64_t(c) * dsc.row_bytes);
            uint4 * t = reinterpret_cast<uint4 *>(base + int64_t(d) * dsc.row_bytes);
            for (int64_t i = threadIdx.x; i < dsc.row_bytes / 16; i += blockDim.x) t[i] = s[i];
        }
        __syncthreads();
    }
}
}  // namespace

namespace {
// Best-first token tree over the DFlash2 lattice, on the device (one thread): node = (depth d, candidate c, parent);
// log p(c | parent's candidate) = softmax of score[d][parent cand][.] (d == 1: the anchor row 0). Writes the verify
// block: tok[0] = anchor, then budget nodes parent-before-child, with parents and depths.
__global__ void dflash_tree_k(const uint32_t * idx, const float * score, const uint32_t * anchor, int B, int budget,
                              uint32_t * tok, int * par, int * dep, float inv_temp) {
    if (threadIdx.x || blockIdx.x) return;
    constexpr int MAXN = 16, POOL = 256;
    float nlp[MAXN], plp[POOL];
    int ncand[MAXN], ndep[MAXN], pd[POOL], pc[POOL], pp[POOL], np = 0, n = 1;
    tok[0] = *anchor, par[0] = -1, dep[0] = 0, nlp[0] = 0.f, ncand[0] = 0, ndep[0] = 0;
    auto push = [&](int node) {
        const int d = ndep[node] + 1;
        if (d >= B) return;
        const float * row = score + (int64_t(d) * 16 + (d == 1 ? 0 : ncand[node])) * 16;
        float mx = row[0];
        for (int k = 1; k < 16; ++k) mx = fmaxf(mx, row[k]);
        float z = 0.f;
        for (int k = 0; k < 16; ++k) z += expf((row[k] - mx) * inv_temp);
        const float lz = logf(z);
        for (int k = 0; k < 16 && np < POOL; ++k, ++np) {
            plp[np] = nlp[node] + (row[k] - mx) * inv_temp - lz;
            pd[np] = d, pc[np] = k, pp[np] = node;
        }
    };
    push(0);
    while (n < budget + 1 && n < MAXN && np > 0) {
        int b = 0;
        for (int i = 1; i < np; ++i)
            if (plp[i] > plp[b]) b = i;
        tok[n] = idx[pd[b] * 16 + pc[b]], par[n] = pp[b], dep[n] = pd[b];
        nlp[n] = plp[b], ncand[n] = pc[b], ndep[n] = pd[b];
        --np, plp[b] = plp[np], pd[b] = pd[np], pc[b] = pc[np], pp[b] = pp[np];
        push(n++);
    }
}
// Greedy chain through the DFlash2 lattice (dflash_draft_batch's host chain, on the device): tok[i - 1], i = 1..want,
// is the first highest-scoring candidate in the row of the previous pick.
__global__ void dflash_chain_k(const uint32_t * idx, const float * score, int want, uint32_t * tok) {
    if (threadIdx.x || blockIdx.x) return;
    int pred = 0;
    for (int i = 1; i <= want; ++i) {
        const float * row = score + (int64_t(i) * 16 + pred) * 16;
        int b = 0;
        for (int k = 1; k < 16; ++k)
            if (row[b] < row[k]) b = k;
        pred = b;
        tok[i - 1] = idx[i * 16 + b];
    }
}
}  // namespace

void dflash_chain(const uint32_t * idx, const float * score, int want, uint32_t * tok, cudaStream_t s) {
    if (want > 0) dflash_chain_k<<<1, 32, 0, s>>>(idx, score, want, tok);
}

void dflash_tree(const uint32_t * idx, const float * score, const uint32_t * anchor, int B, int budget, uint32_t * tok,
                 int * par, int * dep, cudaStream_t s) {
    dflash_tree_k<<<1, 32, 0, s>>>(idx, score, anchor, B, budget, tok, par, dep, 1.f);
}

void gather_path(const PathGather * descs, int n, const int * cols, int k, int pos, cudaStream_t s) {
    if (n > 0 && k > 1) gather_path_k<<<n, 256, 0, s>>>(descs, cols, k, pos);
}

void gather_rows(const void * src, void * dst, const uint32_t * ids, int n, int64_t row_bytes, cudaStream_t s) {
    gather_rows_k<<<n, 256, 0, s>>>(static_cast<const uint8_t *>(src), static_cast<uint8_t *>(dst), ids, row_bytes);
}

void map_ids(uint32_t * idx, const uint32_t * ids, int n, cudaStream_t s) {
    map_ids_k<<<(n + 255) / 256, 256, 0, s>>>(idx, ids, n);
}

} // namespace tqllm_cuda

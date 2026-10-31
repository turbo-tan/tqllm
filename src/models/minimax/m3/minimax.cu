#include "minimax_kernels.cuh"

#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <stdexcept>

namespace tqllm_cuda {
namespace {
int blocks(long n, int t) { return int((n + t - 1) / t); }

// KV cache element: FP16, or FP8 E4M3 (saturating) when the requested context only fits at half the bytes
__device__ __forceinline__ float kvf(__half x) { return __half2float(x); }
__device__ __forceinline__ float kvf(__nv_fp8_e4m3 x) { return float(x); }
template <class T> __device__ __forceinline__ T kv_of(float x);
template <> __device__ __forceinline__ __half kv_of<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_fp8_e4m3 kv_of<__nv_fp8_e4m3>(float x) { return __nv_fp8_e4m3(x); }

__device__ __forceinline__ float oai(float g, float u, float alpha, float limit) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    return (u + 1.f) * g / (1.f + __expf(-alpha * g));
}
__global__ void swigluoai_f32_k(const float * g, const float * u, __half * out, int n, float alpha, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(oai(g[i], u[i], alpha, limit));
}
__global__ void swigluoai_bf16_k(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * out, int n, float alpha, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(oai(__bfloat162float(g[i]), __bfloat162float(u[i]), alpha, limit));
}
__global__ void rope_k(float * x, int heads, int dim, int rot, int pos0, float theta) {
    const int t = blockIdx.y, h = blockIdx.x, half = rot / 2;
    float * r = x + (size_t(t) * heads + h) * dim;
    const float pos = float(pos0 + t);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float f = pos * powf(theta, -2.f * float(i) / float(rot));
        float sn, cs;
        sincosf(f, &sn, &cs);
        const float a = r[i], b = r[i + half];
        r[i] = a * cs - b * sn;
        r[i + half] = b * cs + a * sn;
    }
}
__global__ void put_rows_k(const float * src, float * dst, long n, int width, int row0) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[size_t(row0) * width + i] = src[i];
}
// one block of 4 warps per (token, q head); each lane holds 4 of the 128 dims; warps take keys round robin with an online
// softmax each, then the 4 partial results merge in shared memory
__global__ void attn_k(const float * q, const float * k, const float * v, float * out, int hq, int hk, int pos0, float scale) {
    constexpr int D = 128, W = 4;
    const int t = blockIdx.y, h = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int kh = h / (hq / hk), rows = pos0 + t + 1;
    const float * qr = q + (size_t(t) * hq + h) * D;
    float qv[4], acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) qv[i] = qr[lane * 4 + i];
    float m = -INFINITY, l = 0.f;
    for (int j = warp; j < rows; j += W) {
        const float * kr = k + (size_t(j) * hk + kh) * D + lane * 4;
        float s = qv[0] * kr[0] + qv[1] * kr[1] + qv[2] * kr[2] + qv[3] * kr[3];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        s *= scale;
        const float mn = fmaxf(m, s), a = __expf(m - mn), b = __expf(s - mn);
        const float * vr = v + (size_t(j) * hk + kh) * D + lane * 4;
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * a + b * vr[i];
        l = l * a + b;
        m = mn;
    }
    __shared__ float sm[W], sl[W], sacc[W][D];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int i = 0; i < 4; ++i) sacc[warp][lane * 4 + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < W; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f, o[4] = {0, 0, 0, 0};
        for (int w = 0; w < W; ++w) {
            if (sm[w] == -INFINITY) continue;
            const float f = __expf(sm[w] - M);
            L += sl[w] * f;
            for (int i = 0; i < 4; ++i) o[i] += sacc[w][lane * 4 + i] * f;
        }
        float * orow = out + (size_t(t) * hq + h) * D;
        for (int i = 0; i < 4; ++i) orow[lane * 4 + i] = o[i] / L;
    }
}
template <class T>
__global__ void put_rows_h_k(const float * src, T * dst, long n, int width, int row0) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[size_t(row0) * width + i] = kv_of<T>(src[i]);
}
// one block per (token, key block): 4 warps, a lane holds 4 of the 128 dims of every index head
template <class T>
__global__ void index_scores_k(const float * iq, const T * ik, float * out, int heads, int pos0, int block, int nblocks,
                               int local) {
    constexpr int D = 128;
    const int t = blockIdx.y, b = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int pos = pos0 + t, qb = pos / block;
    float best = -INFINITY;
    if (b <= qb) {
        for (int j = b * block + warp; j < (b + 1) * block && j <= pos; j += 4) {
            const T * kr = ik + size_t(j) * D + lane * 4;
            const float k0 = kvf(kr[0]), k1 = kvf(kr[1]), k2 = kvf(kr[2]), k3 = kvf(kr[3]);
            for (int h = 0; h < heads; ++h) {
                const float * qr = iq + (size_t(t) * heads + h) * D + lane * 4;
                float d = qr[0] * k0 + qr[1] * k1 + qr[2] * k2 + qr[3] * k3;
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                best = fmaxf(best, d);
            }
        }
    }
    __shared__ float sm[4];
    if (lane == 0) sm[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
        if (b <= qb && b > qb - local) m = INFINITY;
        out[size_t(t) * nblocks + b] = m;
    }
}
// one block per token: k rounds of a block-wide argmax (lowest index on ties), each winner taken out
__global__ void topk_blocks_k(const float * scores, int * ids, int nblocks, int k, float * work) {
    const int t = blockIdx.x;
    float * s = work + size_t(t) * nblocks;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) s[i] = scores[size_t(t) * nblocks + i];
    __syncthreads();
    __shared__ float bv[256];
    __shared__ int bi[256];
    for (int r = 0; r < k; ++r) {
        float v = -INFINITY; int at = -1;
        for (int i = threadIdx.x; i < nblocks; i += blockDim.x)
            if (s[i] > v || (s[i] == v && at >= 0 && i < at && s[i] != -INFINITY)) { v = s[i]; at = i; }
        bv[threadIdx.x] = v; bi[threadIdx.x] = at;
        __syncthreads();
        for (int w = blockDim.x / 2; w > 0; w >>= 1) {
            if (threadIdx.x < w) {
                const float ov = bv[threadIdx.x + w]; const int oi = bi[threadIdx.x + w];
                if (oi >= 0 && (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && (bi[threadIdx.x] < 0 || oi < bi[threadIdx.x])))) {
                    bv[threadIdx.x] = ov; bi[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const bool ok = bi[0] >= 0 && bv[0] != -INFINITY;
            ids[size_t(t) * k + r] = ok ? bi[0] : -1;
            if (ok) s[bi[0]] = -INFINITY;
        }
        __syncthreads();
    }
}
template <class T>
__global__ void attn_h_k(const float * q, const T * k, const T * v, float * out, int hq, int hk, int pos0, float scale,
                         const int * ids, int topk, int block) {
    constexpr int D = 128, W = 4;
    const int t = blockIdx.y, h = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int kh = h / (hq / hk), pos = pos0 + t;
    const float * qr = q + (size_t(t) * hq + h) * D;
    float qv[4], acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) qv[i] = qr[lane * 4 + i];
    float m = -INFINITY, l = 0.f;
    auto key = [&](int j) {
        const T * kr = k + (size_t(j) * hk + kh) * D + lane * 4;
        float s = qv[0] * kvf(kr[0]) + qv[1] * kvf(kr[1]) + qv[2] * kvf(kr[2]) + qv[3] * kvf(kr[3]);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        s *= scale;
        const float mn = fmaxf(m, s), a = __expf(m - mn), b = __expf(s - mn);
        const T * vr = v + (size_t(j) * hk + kh) * D + lane * 4;
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * a + b * kvf(vr[i]);
        l = l * a + b;
        m = mn;
    };
    if (!ids) {
        for (int j = warp; j <= pos; j += W) key(j);
    } else {
        // keys in ascending block order so the sum order does not depend on the ranking
        int sel[32]; int ns = 0;
        for (int r = 0; r < topk && r < 32; ++r) { const int b = ids[size_t(t) * topk + r]; if (b < 0) break; sel[ns++] = b; }
        for (int a = 1; a < ns; ++a) { const int x = sel[a]; int c = a - 1; while (c >= 0 && sel[c] > x) { sel[c + 1] = sel[c]; --c; } sel[c + 1] = x; }
        int idx = 0;
        for (int r = 0; r < ns; ++r)
            for (int j = sel[r] * block; j < (sel[r] + 1) * block && j <= pos; ++j, ++idx)
                if (idx % W == warp) key(j);
    }
    __shared__ float sm[W], sl[W], sacc[W][D];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int i = 0; i < 4; ++i) sacc[warp][lane * 4 + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < W; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f, o[4] = {0, 0, 0, 0};
        for (int w = 0; w < W; ++w) {
            if (sm[w] == -INFINITY) continue;
            const float f = __expf(sm[w] - M);
            L += sl[w] * f;
            for (int i = 0; i < 4; ++i) o[i] += sacc[w][lane * 4 + i] * f;
        }
        float * orow = out + (size_t(t) * hq + h) * D;
        for (int i = 0; i < 4; ++i) orow[lane * 4 + i] = o[i] / L;
    }
}
// Prompt-chunk attention: one block per (KV head, token), one warp per query head of the group (hq / hk <= 32). The
// group's keys / values are staged once per tile in shared memory, so the G query heads stop reading them G times;
// same keys and math as attn_h_k (online softmax per query head, sparse blocks in ascending order).
template <class E>
__global__ void attn_hg_k(const float * q, const E * k, const E * v, float * out, int hq, int hk, int pos0, float scale,
                          const int * ids, int topk, int block) {
    constexpr int D = 128, T = 32, C = D * int(sizeof(E)) / 16;  // 16-byte chunks per row
    const int t = blockIdx.y, kh = blockIdx.x, G = hq / hk, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int h = kh * G + warp, pos = pos0 + t;
    __shared__ uint4 ks[T][C], vs[T][C];
    __shared__ int sel[32], cnt[33], nsel;
    if (threadIdx.x == 0) {
        int ns = 0;
        if (ids) {
            for (int r = 0; r < topk && r < 32; ++r) { const int b = ids[size_t(t) * topk + r]; if (b < 0) break; sel[ns++] = b; }
            for (int a = 1; a < ns; ++a) { const int x = sel[a]; int c = a - 1; while (c >= 0 && sel[c] > x) { sel[c + 1] = sel[c]; --c; } sel[c + 1] = x; }
            cnt[0] = 0;
            for (int r = 0; r < ns; ++r) cnt[r + 1] = cnt[r] + max(0, min(block, pos - sel[r] * block + 1));
        }
        nsel = ns;
    }
    __syncthreads();
    const int keys = ids ? cnt[nsel] : pos + 1;
    float qv[4], acc[4] = {0, 0, 0, 0}, m = -INFINITY, l = 0.f;
    const float * qr = q + (size_t(t) * hq + h) * D;
    for (int i = 0; i < 4; ++i) qv[i] = qr[lane * 4 + i];
    for (int base = 0; base < keys; base += T) {
        const int tn = min(T, keys - base);
        for (int e = threadIdx.x; e < tn * C; e += blockDim.x) {
            const int r = e / C, c = e % C, idx = base + r;
            int j = idx;
            if (ids) { int s = 0; while (cnt[s + 1] <= idx) ++s; j = sel[s] * block + (idx - cnt[s]); }
            ks[r][c] = reinterpret_cast<const uint4 *>(k + (size_t(j) * hk + kh) * D)[c];
            vs[r][c] = reinterpret_cast<const uint4 *>(v + (size_t(j) * hk + kh) * D)[c];
        }
        __syncthreads();
        if (warp < G)
            for (int r = 0; r < tn; ++r) {
                const E * kr = reinterpret_cast<const E *>(ks[r]) + lane * 4, * vr = reinterpret_cast<const E *>(vs[r]) + lane * 4;
                float s = qv[0] * kvf(kr[0]) + qv[1] * kvf(kr[1]) + qv[2] * kvf(kr[2]) + qv[3] * kvf(kr[3]);
                for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
                s *= scale;
                const float mn = fmaxf(m, s), a = __expf(m - mn), b = __expf(s - mn);
                for (int i = 0; i < 4; ++i) acc[i] = acc[i] * a + b * kvf(vr[i]);
                l = l * a + b;
                m = mn;
            }
        __syncthreads();
    }
    if (warp < G) {
        float * orow = out + (size_t(t) * hq + h) * D;
        for (int i = 0; i < 4; ++i) orow[lane * 4 + i] = acc[i] / l;
    }
}
// Prompt-chunk attention on tensor cores, for 16 query heads per KV head: one block per (KV head, 8 consecutive
// tokens), one warp per token holding its 16 heads as the M of mma.m16n8k16 (FP16 inputs, FP32 accumulate). The
// block walks the union of its tokens' key blocks in ascending order, staging each 64-key tile once for the 8
// tokens (neighbours mostly pick the same blocks); a warp computes only the blocks its token selected, so a token's
// result does not depend on its neighbours.
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void stage8(__half * dst, uint4 raw, const __half *) { *reinterpret_cast<uint4 *>(dst) = raw; }
__device__ __forceinline__ void stage8(__half * dst, uint2 raw, const __nv_fp8_e4m3 *) {
    const __nv_fp8_e4m3 * e = reinterpret_cast<const __nv_fp8_e4m3 *>(&raw);
    for (int i = 0; i < 8; ++i) dst[i] = __float2half(float(e[i]));
}
constexpr int MMA_TT = 8, MMA_KT = 64, MMA_LD = 128 + 8;  // tokens per block, keys per tile, padded smem row (halves)
template <class E>
__global__ void __launch_bounds__(256) attn_mma_k(const float * q, const E * k, const E * v, float * out, int n, int hq,
                                                  int hk, int pos0, float scale, const int * ids, int topk, int block) {
    constexpr int D = 128, TT = MMA_TT, KT = MMA_KT, LD = MMA_LD;
    using Raw = std::conditional_t<sizeof(E) == 2, uint4, uint2>;  // 8 elements
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half * ks = reinterpret_cast<__half *>(smem_raw);  // [KT][LD]
    __half * vt = ks + KT * LD;                          // [D][KT + 8] (V transposed)
    constexpr int VLD = KT + 8;
    __shared__ int sel[TT][32], ns[TT], uni[TT * 32], nuni;
    __shared__ uint32_t bits[64];
    const int kh = blockIdx.x, t0 = blockIdx.y * TT, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane >> 2, tq = lane & 3, t = t0 + warp, pos = pos0 + t;
    const bool live = t < n;
    const int last = pos0 + min(n, t0 + TT) - 1;  // the group's furthest position
    if (ids) {
        for (int i = threadIdx.x; i < 64; i += blockDim.x) bits[i] = 0;
        __syncthreads();
        if (lane == 0) {
            int c = 0;
            if (live)
                for (int r = 0; r < topk && r < 32; ++r) { const int b = ids[size_t(t) * topk + r]; if (b < 0) break; sel[warp][c++] = b; }
            for (int a = 1; a < c; ++a) { const int x = sel[warp][a]; int j = a - 1; while (j >= 0 && sel[warp][j] > x) { sel[warp][j + 1] = sel[warp][j]; --j; } sel[warp][j + 1] = x; }
            ns[warp] = c;
            for (int r = 0; r < c; ++r) atomicOr(&bits[sel[warp][r] >> 5], 1u << (sel[warp][r] & 31));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int c = 0;
            for (int w = 0; w < 64; ++w) for (uint32_t m = bits[w]; m; m &= m - 1) uni[c++] = w * 32 + __ffs(m) - 1;
            nuni = c;
        }
    } else if (threadIdx.x == 0) {
        nuni = last / block + 1;
    }
    __syncthreads();
    // Q fragments of this warp's token: rows = the KV head's 16 query heads, k = head dims
    uint32_t qa[8][4];
    if (live) {
        const float * qr = q + (size_t(t) * hq + kh * 16) * D;
        for (int kk = 0; kk < 8; ++kk) {
            const int c0 = kk * 16 + tq * 2;
            qa[kk][0] = pack_h2(qr[size_t(g) * D + c0], qr[size_t(g) * D + c0 + 1]);
            qa[kk][1] = pack_h2(qr[size_t(g + 8) * D + c0], qr[size_t(g + 8) * D + c0 + 1]);
            qa[kk][2] = pack_h2(qr[size_t(g) * D + c0 + 8], qr[size_t(g) * D + c0 + 9]);
            qa[kk][3] = pack_h2(qr[size_t(g + 8) * D + c0 + 8], qr[size_t(g + 8) * D + c0 + 9]);
        }
    }
    float o[16][4] = {}, m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const float sl2 = scale * 1.4426950408889634f;
    int p = 0;  // next entry of this token's sorted selection
    for (int u = 0; u < nuni; ++u) {
        const int b = ids ? uni[u] : u;
        bool use = live && b * block <= pos;
        if (use && ids) { use = p < ns[warp] && sel[warp][p] == b; if (use) ++p; }
        for (int k0 = 0; k0 < block; k0 += KT) {
            const int j0 = b * block + k0;
            if (j0 > last) break;  // uniform across the block
            __syncthreads();
            for (int e = threadIdx.x; e < KT * (D / 8); e += blockDim.x) {
                const int r = e / (D / 8), c = (e % (D / 8)) * 8, j = j0 + r;
                if (j <= last) {
                    const Raw kr = *reinterpret_cast<const Raw *>(k + (size_t(j) * hk + kh) * D + c);
                    const Raw vr = *reinterpret_cast<const Raw *>(v + (size_t(j) * hk + kh) * D + c);
                    stage8(ks + r * LD + c, kr, k);
                    __half tmp[8];
                    stage8(tmp, vr, v);
                    for (int i = 0; i < 8; ++i) vt[(c + i) * VLD + r] = tmp[i];
                } else {
                    for (int i = 0; i < 8; ++i) { ks[r * LD + c + i] = __float2half(0.f); vt[(c + i) * VLD + r] = __float2half(0.f); }
                }
            }
            __syncthreads();
            if (!use || j0 > pos) continue;
            // S = Q K^T: 16 heads x 64 keys
            float s[8][4];
            for (int nt = 0; nt < 8; ++nt) {
                s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
                const __half * kr = ks + (nt * 8 + g) * LD + tq * 2;
                for (int kk = 0; kk < 8; ++kk)
                    mma16816(s[nt], qa[kk], *reinterpret_cast<const uint32_t *>(kr + kk * 16),
                             *reinterpret_cast<const uint32_t *>(kr + kk * 16 + 8));
            }
            float mx0 = m0, mx1 = m1;
            for (int nt = 0; nt < 8; ++nt)
                for (int i = 0; i < 2; ++i) {
                    const int j = j0 + nt * 8 + tq * 2 + i;
                    const bool ok = j <= pos;
                    s[nt][i] = ok ? s[nt][i] * sl2 : -INFINITY;
                    s[nt][2 + i] = ok ? s[nt][2 + i] * sl2 : -INFINITY;
                    mx0 = fmaxf(mx0, s[nt][i]); mx1 = fmaxf(mx1, s[nt][2 + i]);
                }
            for (int x = 1; x < 4; x <<= 1) { mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, x)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, x)); }
            const float a0 = exp2f(m0 - mx0), a1 = exp2f(m1 - mx1);
            m0 = mx0; m1 = mx1;
            float r0 = 0.f, r1 = 0.f;
            for (int nt = 0; nt < 8; ++nt)
                for (int i = 0; i < 2; ++i) {
                    s[nt][i] = exp2f(s[nt][i] - m0); r0 += s[nt][i];
                    s[nt][2 + i] = exp2f(s[nt][2 + i] - m1); r1 += s[nt][2 + i];
                }
            for (int x = 1; x < 4; x <<= 1) { r0 += __shfl_xor_sync(0xffffffffu, r0, x); r1 += __shfl_xor_sync(0xffffffffu, r1, x); }
            l0 = l0 * a0 + r0; l1 = l1 * a1 + r1;
            for (int dt = 0; dt < 16; ++dt) { o[dt][0] *= a0; o[dt][1] *= a0; o[dt][2] *= a1; o[dt][3] *= a1; }
            // O += P V: k = 64 keys in 4 steps of 16, n = 128 dims in 16 tiles of 8
            for (int kk = 0; kk < 4; ++kk) {
                const uint32_t pa[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                        pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
                for (int dt = 0; dt < 16; ++dt) {
                    const __half * vr = vt + (dt * 8 + g) * VLD + kk * 16 + tq * 2;
                    mma16816(o[dt], pa, *reinterpret_cast<const uint32_t *>(vr), *reinterpret_cast<const uint32_t *>(vr + 8));
                }
            }
        }
    }
    if (live) {
        float * orow = out + (size_t(t) * hq + kh * 16) * D;
        const float i0 = 1.f / l0, i1 = 1.f / l1;
        for (int dt = 0; dt < 16; ++dt) {
            const int c = dt * 8 + tq * 2;
            orow[size_t(g) * D + c] = o[dt][0] * i0; orow[size_t(g) * D + c + 1] = o[dt][1] * i0;
            orow[size_t(g + 8) * D + c] = o[dt][2] * i1; orow[size_t(g + 8) * D + c + 1] = o[dt][3] * i1;
        }
    }
}
// Prompt-chunk indexer scores on tensor cores (4 index heads): one block per (key block, 16 tokens), one warp per 4
// tokens whose 16 (token, head) rows are the MMA M; the block's 128 keys staged once in shared memory (FP16) for the
// 16 tokens. Score = max over the token's keys <= pos and the heads of q . k (FP16 products, FP32 accumulate), the
// local blocks +inf, blocks past the token -inf: the rules of index_scores_k.
template <class T>
__global__ void __launch_bounds__(128) index_mma_k(const float * iq, const T * ik, float * out, int n, int pos0, int block,
                                                   int nblocks, int local) {
    constexpr int D = 128, KB = 128, LD = D + 8;
    using Raw = std::conditional_t<sizeof(T) == 2, uint4, uint2>;
    __shared__ __align__(16) __half ks[KB * LD];
    const int b = blockIdx.x, t0 = blockIdx.y * 16, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane >> 2, tq = lane & 3;
    const int last = pos0 + min(n, t0 + 16) - 1;
    const int j0 = b * block;
    const bool any = j0 <= last;  // uniform: some token of the block reaches this key block
    if (any) {
        for (int e = threadIdx.x; e < KB * (D / 8); e += blockDim.x) {
            const int r = e / (D / 8), c = (e % (D / 8)) * 8, j = j0 + r;
            if (r < block && j <= last) stage8(ks + r * LD + c, *reinterpret_cast<const Raw *>(ik + size_t(j) * D + c), ik);
            else for (int i = 0; i < 8; ++i) ks[r * LD + c + i] = __float2half(0.f);
        }
    }
    __syncthreads();
    // rows: (token tw * 4 + h); this lane holds rows g and g + 8 -> tokens g / 4 and 2 + g / 4 of the warp, head g % 4
    const int tw = t0 + warp * 4;
    const int ta = tw + g / 4, tb = tw + 2 + g / 4;
    float ma = -INFINITY, mb = -INFINITY;
    if (any && tw < n) {
        uint32_t qa[8][4];
        const float * qr0 = iq + (size_t(tw) * 4 + g) * D, * qr1 = iq + (size_t(tw) * 4 + g + 8) * D;
        const bool ra = ta < n, rb = tb < n;
        for (int kk = 0; kk < 8; ++kk) {
            const int c0 = kk * 16 + tq * 2;
            qa[kk][0] = ra ? pack_h2(qr0[c0], qr0[c0 + 1]) : 0u;
            qa[kk][1] = rb ? pack_h2(qr1[c0], qr1[c0 + 1]) : 0u;
            qa[kk][2] = ra ? pack_h2(qr0[c0 + 8], qr0[c0 + 9]) : 0u;
            qa[kk][3] = rb ? pack_h2(qr1[c0 + 8], qr1[c0 + 9]) : 0u;
        }
        const int pa = pos0 + ta, pb = pos0 + tb;
        for (int nt = 0; nt < min(block, KB) / 8; ++nt) {
            if (j0 + nt * 8 > pos0 + min(n - 1, tw + 3)) break;
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            const __half * kr = ks + (nt * 8 + g) * LD + tq * 2;
            for (int kk = 0; kk < 8; ++kk)
                mma16816(c, qa[kk], *reinterpret_cast<const uint32_t *>(kr + kk * 16), *reinterpret_cast<const uint32_t *>(kr + kk * 16 + 8));
            for (int i = 0; i < 2; ++i) {
                const int j = j0 + nt * 8 + tq * 2 + i;
                if (j <= pa) ma = fmaxf(ma, c[i]);
                if (j <= pb) mb = fmaxf(mb, c[2 + i]);
            }
        }
    }
    // max over the quad (keys) and over the 4 heads (g % 4: lane bits 2, 3)
    for (int x = 1; x < 16; x <<= 1) { ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, x)); mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, x)); }
    if ((lane & 15) == 0) {
        for (int k = 0; k < 2; ++k) {
            const int t = (k ? tb : ta);
            if (t >= n) continue;
            const int pos = pos0 + t, qb = pos / block;
            float m = b <= qb ? (k ? mb : ma) : -INFINITY;
            if (b <= qb && b > qb - local) m = INFINITY;
            out[size_t(t) * nblocks + b] = m;
        }
    }
}
template <class T>
__global__ void attn_draft_k(const float * q, const T * kc, const T * vc, const float * kn, const float * vn, float * out,
                             int B, int hq, int hk, int anchor, int window, float scale) {
    constexpr int D = 128, W = 4;
    const int o = blockIdx.y, h = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int kh = h / (hq / hk);
    const float * qr = q + (size_t(o) * hq + h) * D;
    float qv[4], acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) qv[i] = qr[lane * 4 + i];
    float m = -INFINITY, l = 0.f;
    auto key = [&](float k0, float k1, float k2, float k3, const float vv[4]) {
        float s = qv[0] * k0 + qv[1] * k1 + qv[2] * k2 + qv[3] * k3;
        for (int d = 16; d > 0; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
        s *= scale;
        const float mn = fmaxf(m, s), a = __expf(m - mn), b = __expf(s - mn);
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * a + b * vv[i];
        l = l * a + b;
        m = mn;
    };
    const int first = max(0, anchor + o - window + 1);
    for (int j = first + warp; j < anchor; j += W) {
        const T * kr = kc + (size_t(j) * hk + kh) * D + lane * 4;
        const T * vr = vc + (size_t(j) * hk + kh) * D + lane * 4;
        const float vv[4] = {kvf(vr[0]), kvf(vr[1]), kvf(vr[2]), kvf(vr[3])};
        key(kvf(kr[0]), kvf(kr[1]), kvf(kr[2]), kvf(kr[3]), vv);
    }
    for (int j = warp; j < B; j += W) {
        const float * kr = kn + (size_t(j) * hk + kh) * D + lane * 4;
        const float * vr = vn + (size_t(j) * hk + kh) * D + lane * 4;
        const float vv[4] = {vr[0], vr[1], vr[2], vr[3]};
        key(kr[0], kr[1], kr[2], kr[3], vv);
    }
    __shared__ float sm[W], sl[W], sacc[W][D];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int i = 0; i < 4; ++i) sacc[warp][lane * 4 + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < W; ++w) M = fmaxf(M, sm[w]);
        float L = 0.f, r[4] = {0, 0, 0, 0};
        for (int w = 0; w < W; ++w) {
            if (sm[w] == -INFINITY) continue;
            const float f = __expf(sm[w] - M);
            L += sl[w] * f;
            for (int i = 0; i < 4; ++i) r[i] += sacc[w][lane * 4 + i] * f;
        }
        float * orow = out + (size_t(o) * hq + h) * D;
        for (int i = 0; i < 4; ++i) orow[lane * 4 + i] = r[i] / L;
    }
}
__global__ void gather_row_k(const float * table, const int * id, float * out, int width) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < width) out[j] = table[size_t(*id) * width + j];
}
__global__ void add_k(float * a, const float * b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
}  // namespace

void mm_attn_draft(const float * q, const void * kc, const void * vc, const float * kn, const float * vn, float * out,
                   int B, int hq, int hk, int dim, int anchor, int window, float scale, cudaStream_t s, bool fp8) {
    if (dim != 128 || hq % hk) throw std::invalid_argument("mm_attn_draft: head_dim 128 and hq a multiple of hk only");
    const dim3 g = dim3(unsigned(hq), unsigned(B));
    if (fp8) attn_draft_k<<<g, 128, 0, s>>>(q, static_cast<const __nv_fp8_e4m3 *>(kc), static_cast<const __nv_fp8_e4m3 *>(vc), kn, vn, out, B, hq, hk, anchor, window, scale);
    else attn_draft_k<<<g, 128, 0, s>>>(q, static_cast<const __half *>(kc), static_cast<const __half *>(vc), kn, vn, out, B, hq, hk, anchor, window, scale);
}
void mm_gather_row(const float * table, const int * id, float * out, int width, cudaStream_t s) {
    gather_row_k<<<blocks(width, 256), 256, 0, s>>>(table, id, out, width);
}
void mm_add(float * a, const float * b, int n, cudaStream_t s) { add_k<<<blocks(n, 256), 256, 0, s>>>(a, b, n); }

void mm_put_rows_h(const float * src, void * dst, int width, int n, int row0, cudaStream_t s, bool fp8) {
    const long e = long(n) * width;
    if (fp8) put_rows_h_k<<<blocks(e, 256), 256, 0, s>>>(src, static_cast<__nv_fp8_e4m3 *>(dst), e, width, row0);
    else put_rows_h_k<<<blocks(e, 256), 256, 0, s>>>(src, static_cast<__half *>(dst), e, width, row0);
}
void mm_index_scores_mma(const float * iq, const void * ik, float * out, int n, int heads, int dim, int pos0, int block,
                         int nblocks, int local, cudaStream_t s, bool fp8) {
    if (dim != 128 || heads != 4 || block > 128 || block % 8)
        throw std::invalid_argument("mm_index_scores_mma: index head_dim 128, 4 index heads, block <= 128 (multiple of 8)");
    const dim3 g = dim3(unsigned(nblocks), unsigned((n + 15) / 16));
    if (fp8) index_mma_k<<<g, 128, 0, s>>>(iq, static_cast<const __nv_fp8_e4m3 *>(ik), out, n, pos0, block, nblocks, local);
    else index_mma_k<<<g, 128, 0, s>>>(iq, static_cast<const __half *>(ik), out, n, pos0, block, nblocks, local);
}
void mm_index_scores(const float * iq, const void * ik, float * out, int n, int heads, int dim, int pos0, int block,
                     int nblocks, int local, cudaStream_t s, bool fp8) {
    if (dim != 128) throw std::invalid_argument("mm_index_scores: index head_dim 128 only");
    const dim3 g = dim3(unsigned(nblocks), unsigned(n));
    if (fp8) index_scores_k<<<g, 128, 0, s>>>(iq, static_cast<const __nv_fp8_e4m3 *>(ik), out, heads, pos0, block, nblocks, local);
    else index_scores_k<<<g, 128, 0, s>>>(iq, static_cast<const __half *>(ik), out, heads, pos0, block, nblocks, local);
}
void mm_topk_blocks(const float * scores, int * ids, int n, int nblocks, int k, cudaStream_t s) {
    static float * work = nullptr; static size_t cap = 0;
    const size_t need = size_t(n) * nblocks * sizeof(float);
    if (need > cap) { if (work) cudaFree(work); if (cudaMalloc(&work, need) != cudaSuccess) throw std::runtime_error("mm_topk_blocks: alloc"); cap = need; }
    topk_blocks_k<<<n, 256, 0, s>>>(scores, ids, nblocks, k, work);
}
void mm_attn_h(const float * q, const void * k, const void * v, float * out, int n, int hq, int hk, int dim, int pos0,
               float scale, const int * ids, int topk, int block, cudaStream_t s, bool fp8) {
    if (dim != 128 || hq % hk || topk > 32) throw std::invalid_argument("mm_attn_h: head_dim 128, hq a multiple of hk, top-k <= 32");
    const dim3 g = dim3(unsigned(hq), unsigned(n));
    if (fp8) attn_h_k<<<g, 128, 0, s>>>(q, static_cast<const __nv_fp8_e4m3 *>(k), static_cast<const __nv_fp8_e4m3 *>(v), out, hq, hk, pos0, scale, ids, topk, block);
    else attn_h_k<<<g, 128, 0, s>>>(q, static_cast<const __half *>(k), static_cast<const __half *>(v), out, hq, hk, pos0, scale, ids, topk, block);
}
void mm_attn_hg(const float * q, const void * k, const void * v, float * out, int n, int hq, int hk, int dim, int pos0,
                float scale, const int * ids, int topk, int block, cudaStream_t s, bool fp8) {
    if (dim != 128 || hq % hk || hq / hk > 32 || topk > 32)
        throw std::invalid_argument("mm_attn_hg: head_dim 128, hq a multiple of hk, at most 32 query heads per KV head, top-k <= 32");
    const dim3 g = dim3(unsigned(hk), unsigned(n));
    const unsigned th = 32u * unsigned(hq / hk);
    if (fp8) attn_hg_k<<<g, th, 0, s>>>(q, static_cast<const __nv_fp8_e4m3 *>(k), static_cast<const __nv_fp8_e4m3 *>(v), out, hq, hk, pos0, scale, ids, topk, block);
    else attn_hg_k<<<g, th, 0, s>>>(q, static_cast<const __half *>(k), static_cast<const __half *>(v), out, hq, hk, pos0, scale, ids, topk, block);
}

void mm_attn_mma(const float * q, const void * k, const void * v, float * out, int n, int hq, int hk, int dim, int pos0,
                 float scale, const int * ids, int topk, int block, cudaStream_t s, bool fp8) {
    if (dim != 128 || hq != 16 * hk || topk > 32 || block % MMA_KT || (pos0 + n + block - 1) / block > 64 * 32)
        throw std::invalid_argument("mm_attn_mma: head_dim 128, 16 query heads per KV head, top-k <= 32, block a multiple of 64, <= 2048 blocks");
    const dim3 g = dim3(unsigned(hk), unsigned((n + MMA_TT - 1) / MMA_TT));
    const size_t sm = size_t(MMA_KT) * MMA_LD * sizeof(__half) + size_t(128) * (MMA_KT + 8) * sizeof(__half);
    if (fp8) attn_mma_k<<<g, 32 * MMA_TT, sm, s>>>(q, static_cast<const __nv_fp8_e4m3 *>(k), static_cast<const __nv_fp8_e4m3 *>(v), out, n, hq, hk, pos0, scale, ids, topk, block);
    else attn_mma_k<<<g, 32 * MMA_TT, sm, s>>>(q, static_cast<const __half *>(k), static_cast<const __half *>(v), out, n, hq, hk, pos0, scale, ids, topk, block);
}

void mm_swigluoai_f32(const float * g, const float * u, __half * out, int n, float alpha, float limit, cudaStream_t s) {
    swigluoai_f32_k<<<blocks(n, 256), 256, 0, s>>>(g, u, out, n, alpha, limit);
}
void mm_swigluoai_bf16(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * out, int n, float alpha, float limit,
                       cudaStream_t s) {
    swigluoai_bf16_k<<<blocks(n, 256), 256, 0, s>>>(g, u, out, n, alpha, limit);
}
void mm_rope(float * x, int tokens, int heads, int dim, int rot, int pos0, float theta, cudaStream_t s) {
    rope_k<<<dim3(unsigned(heads), unsigned(tokens)), 32, 0, s>>>(x, heads, dim, rot, pos0, theta);
}
void mm_put_rows(const float * src, float * dst, int width, int n, int row0, cudaStream_t s) {
    const long e = long(n) * width;
    put_rows_k<<<blocks(e, 256), 256, 0, s>>>(src, dst, e, width, row0);
}
void mm_attn(const float * q, const float * k, const float * v, float * out, int n, int hq, int hk, int dim, int pos0,
             float scale, cudaStream_t s) {
    if (dim != 128 || hq % hk) throw std::invalid_argument("mm_attn: head_dim 128 and hq a multiple of hk only");
    attn_k<<<dim3(unsigned(hq), unsigned(n)), 128, 0, s>>>(q, k, v, out, hq, hk, pos0, scale);
}
}  // namespace tqllm_cuda

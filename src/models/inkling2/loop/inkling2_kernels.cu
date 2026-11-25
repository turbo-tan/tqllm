// Inkling-2-Mini small kernels (see inkling2_kernels.cuh). Correctness first: simple one-block-per-token shapes.
#include "inkling2_kernels.cuh"
#include "inkling2_attention_body.cuh"
#include "tqt_decode.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tqllm_cuda::inkling2 {
namespace {

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// sum over a (virtual) block of 256 threads (8 warps); every thread gets the result
__device__ float block_sum(float v, float * sh, const Blk & b) {
    const int lane = b.t % 32, warp = b.t / 32;
    v = warp_sum(v);
    b.sync();
    if (lane == 0) sh[warp] = v;
    b.sync();
    float t = 0.0f;
    for (int w = 0; w < b.nt / 32; ++w) t += sh[w];
    return t;
}
// activation loads: read-only-cache loads, or (P) plain coherent loads for data an earlier part of the same launch wrote
template <bool P> __device__ __forceinline__ uint4 ldx(const uint4 * p) {
    if constexpr (P) return *p;
    else return __ldg(p);
}

__device__ __forceinline__ void rms_norm_body(const Blk & b, const float * x, const float * w, float * y32, __nv_bfloat16 * y16, int D, float eps,
                                              float * sh) {
    const size_t o = size_t(b.x) * D;
    float ss = 0.0f;
    for (int c = b.t; c < D; c += b.nt) ss += x[o + c] * x[o + c];
    const float r = rsqrtf(block_sum(ss, sh, b) / float(D) + eps);
    for (int c = b.t; c < D; c += b.nt) {
        const __nv_bfloat16 v = __float2bfloat16(w[c] * (x[o + c] * r));
        if (y32) y32[o + c] = __bfloat162float(v);
        if (y16) y16[o + c] = v;
    }
}
__global__ void rms_norm_k(const float * x, const float * w, float * y32, __nv_bfloat16 * y16, int D, float eps) {
    __shared__ float sh[8];
    rms_norm_body(blk_of_launch(), x, w, y32, y16, D, eps, sh);
}

__device__ __forceinline__ void embed_body(const Blk & b, const __nv_bfloat16 * table, const int * ids, const float * w, float * y, int D, float eps,
                                           float * sh) {
    const __nv_bfloat16 * row = table + size_t(ids[b.x]) * D;
    float ss = 0.0f;
    for (int c = b.t; c < D; c += b.nt) { const float v = __bfloat162float(row[c]); ss += v * v; }
    const float r = rsqrtf(block_sum(ss, sh, b) / float(D) + eps);
    for (int c = b.t; c < D; c += b.nt) y[size_t(b.x) * D + c] = bfr(w[c] * (__bfloat162float(row[c]) * r));
}
__global__ void embed_k(const __nv_bfloat16 * table, const int * ids, const float * w, float * y, int D, float eps) {
    __shared__ float sh[8];
    embed_body(blk_of_launch(), table, ids, w, y, D, eps, sh);
}

__global__ void to_bf16_k(const float * x, __nv_bfloat16 * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] = __float2bfloat16(x[i]);
}
__global__ void to_f32_k(const __nv_bfloat16 * x, float * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] = __bfloat162float(x[i]);
}

__device__ __forceinline__ void conv_static_body(const Blk & b, const __nv_bfloat16 * x, const float * w, float * state, float * y32,
                                                 __nv_bfloat16 * y16, int n, int C, const float * add, bool round_sum, __nv_bfloat16 * hist) {
    const int c = b.x * b.nt + b.t;
    if (c >= C) return;
    float h0 = state[c * 3], h1 = state[c * 3 + 1], h2 = state[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const float sc = rsqrtf(5.0f);
    for (int t = 0; t < n; ++t) {
        const float xt = __bfloat162float(x[size_t(t) * C + c]);
        if (hist) hist[size_t(t) * C + c] = x[size_t(t) * C + c];
        const float y = (w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt + xt) * sc;
        const __nv_bfloat16 b = __float2bfloat16(y);
        if (y32) {
            float v = __bfloat162float(b);
            if (add) { v = add[size_t(t) * C + c] + v; if (round_sum) v = bfr(v); }
            y32[size_t(t) * C + c] = v;
        }
        if (y16) y16[size_t(t) * C + c] = b;
        h0 = h1; h1 = h2; h2 = xt;
    }
    state[c * 3] = h0; state[c * 3 + 1] = h1; state[c * 3 + 2] = h2;
}
__global__ void conv_static_k(const __nv_bfloat16 * x, const float * w, float * state, float * y32, __nv_bfloat16 * y16, int n, int C,
                              const float * add, bool round_sum, __nv_bfloat16 * hist) {
    conv_static_body(blk_of_launch(), x, w, state, y32, y16, n, C, add, round_sum, hist);
}

__device__ __forceinline__ void conv_dynamic_body(const Blk & b, const __nv_bfloat16 * x, const float * stat, const __nv_bfloat16 * dyn, float * state,
                                                  __nv_bfloat16 * y16, int n, int C, __nv_bfloat16 * hist) {
    const int c = b.x * b.nt + b.t;
    if (c >= C) return;
    const int g = c / 128, heads4 = (C / 128) * 4;
    float h[3] = {state[c * 3], state[c * 3 + 1], state[c * 3 + 2]};  // x_{t-3}, x_{t-2}, x_{t-1}
    for (int t = 0; t < n; ++t) {
        const float xt = __bfloat162float(x[size_t(t) * C + c]);
        if (hist) hist[size_t(t) * C + c] = x[size_t(t) * C + c];
        const __nv_bfloat16 * d = dyn + size_t(t) * heads4 + g * 4;
        float y = 0.0f;
        const float xs[4] = {xt, h[2], h[1], h[0]};  // lag 0..3
        for (int lag = 0; lag < 4; ++lag) y += (stat[lag * C + c] + __bfloat162float(d[lag])) * xs[lag];
        y16[size_t(t) * C + c] = __float2bfloat16((y + xt) / 3.0f);
        h[0] = h[1]; h[1] = h[2]; h[2] = xt;
    }
    state[c * 3] = h[0]; state[c * 3 + 1] = h[1]; state[c * 3 + 2] = h[2];
}
__global__ void conv_dynamic_k(const __nv_bfloat16 * x, const float * stat, const __nv_bfloat16 * dyn, float * state, __nv_bfloat16 * y16, int n, int C,
                               __nv_bfloat16 * hist) {
    conv_dynamic_body(blk_of_launch(), x, stat, dyn, state, y16, n, C, hist);
}

constexpr int kMgiThreads = 256, kMgiPer = 10;  // D <= 2560
__device__ __forceinline__ void msg_gate_in_body(const Blk & bk, const float * h, const float * m, __nv_bfloat16 * z, int D, float eps, float * sh) {
    const size_t o = size_t(bk.x) * D;
    float hh[kMgiPer], mm[kMgiPer];
    float a = 0.0f, b = 0.0f;
#pragma unroll
    for (int u = 0; u < kMgiPer; ++u) {
        const int c = bk.t + u * kMgiThreads;
        hh[u] = c < D ? h[o + c] : 0.0f;
        mm[u] = c < D ? m[o + c] : 0.0f;
        a = fmaf(hh[u], hh[u], a);
        b = fmaf(mm[u], mm[u], b);
    }
    const float ra = rsqrtf(block_sum(a, sh, bk) / float(D) + eps);
    const float rb = rsqrtf(block_sum(b, sh, bk) / float(D) + eps);
#pragma unroll
    for (int u = 0; u < kMgiPer; ++u) {
        const int c = bk.t + u * kMgiThreads;
        if (c >= D) break;
        z[2 * o + c] = __float2bfloat16(hh[u] * ra);
        z[2 * o + D + c] = __float2bfloat16(mm[u] * rb);
    }
}
__global__ void __launch_bounds__(kMgiThreads) msg_gate_in_k(const float * h, const float * m, __nv_bfloat16 * z, int D, float eps) {
    __shared__ float sh[8];
    msg_gate_in_body(blk_of_launch(), h, m, z, D, eps, sh);
}

__device__ __forceinline__ void msg_gate_out_body(const Blk & b, const __nv_bfloat16 * g, const __nv_bfloat16 * FT, const __nv_bfloat16 * GT, const float * h,
                                                  const float * m, float * y, int D, int rank, float * gs) {
    const int t = b.y, c = b.x * b.nt + b.t;
    for (int r = b.t; r < rank; r += b.nt) gs[r] = __bfloat162float(g[size_t(t) * rank + r]);
    b.sync();
    if (c >= D) return;
    float f = 0.0f, a = 0.0f;
    for (int r0 = 0; r0 < rank; r0 += 16) {  // 16 rows' loads in flight, then the sums in r order
        float fv[16], av[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            fv[j] = r0 + j < rank ? __bfloat162float(FT[size_t(r0 + j) * D + c]) : 0.0f;
            av[j] = r0 + j < rank ? __bfloat162float(GT[size_t(r0 + j) * D + c]) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (r0 + j >= rank) break;
            f = fmaf(fv[j], gs[r0 + j], f);
            a = fmaf(av[j], gs[r0 + j], a);
        }
    }
    const size_t o = size_t(t) * D + c;
    y[o] = (1.0f - tanhf(f)) * h[o] + tanhf(a) * m[o];
}
__global__ void msg_gate_out_k(const __nv_bfloat16 * g, const __nv_bfloat16 * FT, const __nv_bfloat16 * GT, const float * h, const float * m, float * y,
                               int D, int rank) {
    extern __shared__ float gs[];
    msg_gate_out_body(blk_of_launch(), g, FT, GT, h, m, y, D, rank, gs);
}

// prompt chunks: FT / GT are the traffic of this kernel (rank x D read once per token above). A block takes kMgoChan channels for
// kMgoTok tokens: each thread keeps its channel's weight column in registers (r chunks of 8) and its tokens' f / a in registers,
// so the weights are read once per kMgoTok tokens. Per-token sums keep the r order of the kernel above (bitwise equal for any n).
constexpr int kMgoChan = 128, kMgoTok = 8;
__global__ void __launch_bounds__(kMgoChan) msg_gate_out_tile_k(const __nv_bfloat16 * g, const __nv_bfloat16 * FT, const __nv_bfloat16 * GT,
                                                                const float * h, const float * m, float * y, int n, int D, int rank) {
    const int c = blockIdx.x * kMgoChan + threadIdx.x, t0 = blockIdx.y * kMgoTok;
    if (c >= D) return;
    extern __shared__ float gs[];  // gs[kMgoTok][rank]
    const int nt = kMgoTok < n - t0 ? kMgoTok : n - t0;
    for (int t = 0; t < nt; ++t)
        for (int r = threadIdx.x; r < rank; r += kMgoChan) gs[t * rank + r] = __bfloat162float(g[size_t(t0 + t) * rank + r]);
    __syncthreads();
    float f[kMgoTok], a[kMgoTok];
#pragma unroll
    for (int t = 0; t < kMgoTok; ++t) { f[t] = 0.0f; a[t] = 0.0f; }
    for (int r0 = 0; r0 < rank; r0 += 8) {
        float fv[8], av[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r0 + r >= rank) break;
            fv[r] = __bfloat162float(FT[size_t(r0 + r) * D + c]);
            av[r] = __bfloat162float(GT[size_t(r0 + r) * D + c]);
        }
#pragma unroll
        for (int t = 0; t < kMgoTok; ++t) {
            if (t >= nt) break;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r0 + r >= rank) break;
                const float gv = gs[t * rank + r0 + r];
                f[t] = fmaf(fv[r], gv, f[t]);
                a[t] = fmaf(av[r], gv, a[t]);
            }
        }
    }
    for (int t = 0; t < nt; ++t) {
        const size_t o = size_t(t0 + t) * D + c;
        y[o] = (1.0f - tanhf(f[t])) * h[o] + tanhf(a[t]) * m[o];
    }
}

__global__ void add_k(const float * a, const float * b, float * y, size_t n, bool r) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float v = a[i] + b[i];
        y[i] = r ? bfr(v) : v;
    }
}

__device__ __forceinline__ float e4m3(uint8_t b) {
    __nv_fp8_e4m3 v; v.__x = b; return float(v);
}
__device__ __forceinline__ float e8m0(uint8_t s) { return s ? __int_as_float(int(s) << 23) : 5.877472e-39f; }

__global__ void moe_combine_k(const __nv_bfloat16 * y, const float * wts, __nv_bfloat16 * out, int D, int topk) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float s = 0.0f;
    for (int k = 0; k < topk; ++k) s = fmaf(__bfloat162float(y[(size_t(t) * topk + k) * D + c]), wts[t * topk + k], s);
    out[size_t(t) * D + c] = __float2bfloat16(s);
}
__global__ void inv_order_k(const int * order, int A, int * inv) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < A) inv[order[j]] = j;
}
__global__ void moe_combine_grouped_k(const __nv_bfloat16 * yg, const int * inv, const float * wts, __nv_bfloat16 * out, int D, int topk) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float s = 0.0f;
    for (int k = 0; k < topk; ++k) s = fmaf(__bfloat162float(yg[size_t(inv[t * topk + k]) * D + c]), wts[t * topk + k], s);
    out[size_t(t) * D + c] = __float2bfloat16(s);
}

__device__ __forceinline__ float softcap1(float x, float c) { return c * tanhf(x / c); }
__global__ void softcap_k(float * x, size_t n, float c) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) x[i] = softcap1(x[i], c);
}

// ---- few-token path ----

__device__ __forceinline__ void unpack8(const uint4 & u, float * f) {
    const __nv_bfloat162 * b = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 v = __bfloat1622float2(b[j]); f[2 * j] = v.x; f[2 * j + 1] = v.y; }
}
__device__ __forceinline__ uint4 ld_stream(const uint4 * p) { return __ldcs(p); }

// warp = two consecutive rows of one segment; lane = every 32nd group of 8 columns; per token one F32 sum per row
struct GemvArgs { GemvSeg seg[kMaxSegs]; int warp0[kMaxSegs + 1]; int nseg; };
template <int N, bool P>
__device__ __forceinline__ void gemv_warp(const GemvArgs & a, int w, int lane) {
    GemvSeg sg = a.seg[0];
    int base = 0;
#pragma unroll
    for (int k = 1; k < kMaxSegs; ++k)
        if (k < a.nseg && w >= a.warp0[k]) { sg = a.seg[k]; base = a.warp0[k]; }
    const int r0 = (w - base) * 2, nc = sg.K / 8;
    const bool two = r0 + 1 < sg.rows;
    if (sg.scale) {  // INT8 rows: 16 weights per 16-byte load, the row scale after the sum
        const int n16 = sg.K / 16;
        const uint4 * q0 = reinterpret_cast<const uint4 *>(reinterpret_cast<const int8_t *>(sg.W) + size_t(r0) * sg.K);
        const uint4 * q1 = two ? q0 + n16 : q0;
        float acc[2][N];
#pragma unroll
        for (int t = 0; t < N; ++t) acc[0][t] = acc[1][t] = 0.0f;
#pragma unroll 2
        for (int c = lane; c < n16; c += 32) {
            const uint4 ua = ld_stream(q0 + c), ub = ld_stream(q1 + c);
            const uint32_t wa[4] = {ua.x, ua.y, ua.z, ua.w}, wb[4] = {ub.x, ub.y, ub.z, ub.w};
            float fa[16], fb[16];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    fa[j * 4 + b] = float(int(wa[j] << (24 - 8 * b)) >> 24);
                    fb[j * 4 + b] = float(int(wb[j] << (24 - 8 * b)) >> 24);
                }
#pragma unroll
            for (int t = 0; t < N; ++t) {
                float xv[16];
                const uint4 * xp = reinterpret_cast<const uint4 *>(sg.x + size_t(t) * sg.ldx) + 2 * c;
                unpack8(ldx<P>(xp), xv); unpack8(ldx<P>(xp + 1), xv + 8);
                float pa = 0.0f, pb = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j) { pa = fmaf(fa[j], xv[j], pa); pb = fmaf(fb[j], xv[j], pb); }
                acc[0][t] += pa; acc[1][t] += pb;
            }
        }
#pragma unroll
        for (int t = 0; t < N; ++t)
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float v = warp_sum(acc[r][t]);
                if (lane == 0 && (r == 0 || two)) {
                    const float y = v * sg.scale[r0 + r];
                    const size_t o = size_t(t) * sg.ldy + r0 + r;
                    if (sg.f32) static_cast<float *>(sg.y)[o] = y;
                    else static_cast<__nv_bfloat16 *>(sg.y)[o] = __float2bfloat16(y);
                }
            }
        return;
    }
    const uint4 * w0 = reinterpret_cast<const uint4 *>(sg.W + size_t(r0) * sg.K);
    const uint4 * w1 = two ? w0 + nc : w0;
    float acc[2][N];
#pragma unroll
    for (int t = 0; t < N; ++t) acc[0][t] = acc[1][t] = 0.0f;
#pragma unroll 4
    for (int c = lane; c < nc; c += 32) {
        float fa[8], fb[8];
        unpack8(ld_stream(w0 + c), fa);
        unpack8(ld_stream(w1 + c), fb);
#pragma unroll
        for (int t = 0; t < N; ++t) {
            float xv[8];
            unpack8(ldx<P>(reinterpret_cast<const uint4 *>(sg.x + size_t(t) * sg.ldx) + c), xv);
            float pa = 0.0f, pb = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) { pa = fmaf(fa[j], xv[j], pa); pb = fmaf(fb[j], xv[j], pb); }
            acc[0][t] += pa; acc[1][t] += pb;
        }
    }
#pragma unroll
    for (int t = 0; t < N; ++t)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0 && (r == 0 || two)) {
                const size_t o = size_t(t) * sg.ldy + r0 + r;
                if (sg.f32) static_cast<float *>(sg.y)[o] = v;
                else static_cast<__nv_bfloat16 *>(sg.y)[o] = __float2bfloat16(v);
            }
        }
}
template <int N>
__global__ void __launch_bounds__(256) gemv_k(GemvArgs a) {
    const int w = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (w >= a.warp0[a.nseg]) return;
    gemv_warp<N, false>(a, w, lane);
}

// ---- exact greedy LM head (decode): INT8 per-row copy + error bound, BF16 rescoring of the rows that can still win ----
// one BF16 row dotted with x with gemv_warp<1>'s per-lane order (8 columns a step, lane = every 32nd step), then warp_sum
__device__ __forceinline__ float bf16_row_dot(const __nv_bfloat16 * W, const __nv_bfloat16 * x, int K, int lane) {
    const uint4 * w0 = reinterpret_cast<const uint4 *>(W);
    const int nc = K / 8;
    float acc = 0.0f;
#pragma unroll 4
    for (int c = lane; c < nc; c += 32) {
        float fa[8], xv[8];
        unpack8(ld_stream(w0 + c), fa);
        unpack8(__ldg(reinterpret_cast<const uint4 *>(x) + c), xv);
        float pa = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) pa = fmaf(fa[j], xv[j], pa);
        acc += pa;
    }
    return warp_sum(acc);
}
// orderable key of (value, index): larger value first, then the lower index (std::max_element's choice)
__device__ __forceinline__ unsigned long long argmax_key(float v, int i) {
    unsigned u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | (0xffffffffu - unsigned(i));
}
__device__ __forceinline__ void block_argmax_commit(unsigned long long key, unsigned long long * cell) {
    for (int o = 16; o; o >>= 1) { const unsigned long long k2 = __shfl_xor_sync(0xffffffffu, key, o); key = k2 > key ? k2 : key; }
    __shared__ unsigned long long wk[32];
    if (threadIdx.x % 32 == 0) wk[threadIdx.x / 32] = key;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < int(blockDim.x / 32); ++w) key = wk[w] > key ? wk[w] : key;
        atomicMax(cell, key);
    }
}
// cell = max key of x[0 .. n)
__global__ void __launch_bounds__(256) argmax_cell_k(const float * x, size_t n, unsigned long long * cell) {
    unsigned long long key = 0;
    for (size_t i = blockIdx.x * size_t(256) + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) { const unsigned long long k = argmax_key(x[i], int(i)); key = k > key ? k : key; }
    block_argmax_commit(key, cell);
}
// bound_r = |w_r - s_r q_r| (1 + 1e-4) + 3e-4 |w_r|, a warp per row
__global__ void __launch_bounds__(256) head_bound_k(const __nv_bfloat16 * w, const int8_t * q, const float * scale, float * bound, int rows, int K) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    double e = 0, n = 0;
    for (int c = lane; c < K; c += 32) {
        const double wv = __bfloat162float(w[size_t(r) * K + c]), d = wv - double(float(q[size_t(r) * K + c]) * scale[r]);
        e += d * d; n += wv * wv;
    }
    for (int o = 16; o; o >>= 1) { e += __shfl_xor_sync(0xffffffffu, e, o); n += __shfl_xor_sync(0xffffffffu, n, o); }
    if (lane == 0) bound[r] = float(sqrt(e) * (1.0 + 1e-4) + 3e-4 * sqrt(n)) * (1.0f + 1e-6f);
}
// a warp per row: rows whose approximate score plus bound reaches the best row's floor are rescored exactly (BF16 row,
// the gemv's sum, bf16, softcap) into cells[1]; every block computes |x| itself (same order everywhere)
__global__ void __launch_bounds__(256) head_select_k(const __nv_bfloat16 * w, const float * approx, const float * bound, const __nv_bfloat16 * x, int V,
                                                     int K, float cap, unsigned long long * cells) {
    __shared__ float sh[8];
    float ss = 0.0f;
    for (int c = threadIdx.x; c < K; c += 256) { const float v = __bfloat162float(x[c]); ss = fmaf(v, v, ss); }
    const Blk bk = blk_of_launch();
    const float nx = sqrtf(block_sum(ss, sh, bk)) * (1.0f + 1e-5f);
    const int b = int(0xffffffffu - unsigned(cells[0] & 0xffffffffu));
    const float A = approx[b], margin = nx * bound[b];
    const float floor_ = A - margin - (fabsf(A) + margin + 1.0f) * (1.0f / 64.0f) - 1e-5f;
    // past the softcap's saturation distinct logits can share one softcapped value: a row is ruled out only when its
    // softcapped upper bound is strictly below the best row's softcapped lower bound (each padded past BF16 rounding)
    const float lo = A - margin, s_lo = softcap1(lo - (fabsf(lo) + 1.0f) * (1.0f / 128.0f), cap);
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    unsigned long long key = 0;
    const float up = r < V ? approx[r] + nx * bound[r] + 1e-5f : 0.0f;
    if (r < V && (up >= floor_ || softcap1(up + (fabsf(up) + 1.0f) * (1.0f / 128.0f), cap) >= s_lo)) {
        const float v = bf16_row_dot(w + size_t(r) * K, x, K, lane);
        key = argmax_key(softcap1(__bfloat162float(__float2bfloat16(v)), cap), r);
    }
    block_argmax_commit(key, cells + 1);
}
// token from cells[c]; both cells cleared for the next step
__global__ void argmax_take_k(unsigned long long * cells, int c, int * token, int * token2) {
    const int t = int(0xffffffffu - unsigned(cells[c] & 0xffffffffu));
    *token = t;
    if (token2) *token2 = t;
    cells[0] = 0; cells[1] = 0;
}
// FP4 / FP8 decode: the hardware conversions on sm_100+, else a shared-memory table (sm_86 / sm_89 convert in software,
// which made the expert kernels compute-bound: 11-16 % of the 3090's bandwidth). Same exact values either way.
struct DecodeLut { float2 fp4[256]; float fp8[256]; };
__device__ __constant__ float kE2M1v[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
__device__ __forceinline__ void lut_init(DecodeLut * t) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1000
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        t->fp4[i] = make_float2(kE2M1v[i & 15], kE2M1v[i >> 4]);
        t->fp8[i] = (i & 0x7f) == 0x7f ? 0.0f : e4m3(uint8_t(i));
    }
    __syncthreads();
#endif
}
// e2m1 magnitude (m = sign-less 3-bit code, [e1 e0 m]) as exact F16 bits: e = m >> 1, mantissa = m & 1;
// e > 0 -> 2^(e - 1) (1 + m/2), e == 0 -> 0 or 0.5
__device__ __forceinline__ unsigned e2m1_mag_h(unsigned m) {
    const unsigned e = (m >> 1) & 3, mm = m & 1;
    return (e || mm) ? 0x3800u + (e << 10) + (e ? mm << 9 : 0) : 0u;
}
__device__ __forceinline__ float2 fp4x2(const DecodeLut * t, uint32_t byte) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t(byte & 0xff), __NV_E2M1)));
#else
    return t->fp4[byte & 0xff];
#endif
}
__device__ __forceinline__ float2 fp8x2(const DecodeLut * t, uint32_t two) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(two & 0xffff), __NV_E4M3)));
#else
    return make_float2(t->fp8[two & 0xff], t->fp8[(two >> 8) & 0xff]);
#endif
}

// one routed-expert row dotted with x (F32 sums), weights as the reference dequantises them:
//   FMT 1 MXFP8 rows (cols E4M3, then cols / 32 E8M0): e4m3 x 2^e8m0, exact in BF16; 4 columns per lane per step
//   FMT 2 NVF4 planes: codes [cols / 2] (low nibble first), aux [cols / 16 E4M3 scales][F32 global]:
//         e2m1 x (scale x global) per 16 codes; 32 codes (one 16-byte load) per lane per step
// NC = 16-byte steps per row when known at compile time (all loads issue before the arithmetic), 0 = runtime cols.
template <int FMT, int NC, bool P = false>
__device__ __forceinline__ float row_dot(const uint8_t * row, const uint8_t * aux, int cols, const __nv_bfloat16 * x, int lane, const DecodeLut * lut) {
    float acc = 0.0f;
    if constexpr (FMT == 2) {
        const int nc = NC ? NC : cols / 32;
        const float g = __ldg(reinterpret_cast<const float *>(aux + cols / 16));
#pragma unroll
        for (int i0 = 0; i0 < (NC ? (NC + 31) / 32 : 1); ++i0) {
            for (int c = (NC ? i0 * 32 : 0) + lane; c < (NC ? min(i0 * 32 + 32, NC) : nc); c += 32) {
                const uint4 q = __ldcs(reinterpret_cast<const uint4 *>(row) + c);
                const uint32_t s2 = __ldg(reinterpret_cast<const uint16_t *>(aux) + c);
                const float sc[2] = {e4m3(uint8_t(s2 & 0xff)), e4m3(uint8_t(s2 >> 8))};
                const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
                float part = 0.0f;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    if (w == 2) { acc = fmaf(part, sc[0] * g, acc); part = 0.0f; }
                    float xv[8];
                    unpack8(ldx<P>(reinterpret_cast<const uint4 *>(x + c * 32) + w), xv);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 e = fp4x2(lut, qw[w] >> (8 * j));
                        part = fmaf(e.x, xv[2 * j], part);
                        part = fmaf(e.y, xv[2 * j + 1], part);
                    }
                }
                acc = fmaf(part, sc[1] * g, acc);
            }
        }
    } else {
        const int ni = NC ? NC : cols / 128;
#pragma unroll
        for (int i0 = 0; i0 < (NC ? NC : 1); ++i0) {
            for (int c0 = (NC ? i0 * 128 : 0) + lane * 4; c0 < (NC ? i0 * 128 + 128 : ni * 128); c0 += 128) {
                const uint32_t q = __ldg(reinterpret_cast<const uint32_t *>(row + c0));
                const float s = e8m0(__ldg(row + cols + c0 / 32));
                const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(x + c0));
                const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(x + c0 + 2));
                const float2 a = fp8x2(lut, q), b = fp8x2(lut, q >> 16);
                float part = a.x * x01.x;
                part = fmaf(a.y, x01.y, part);
                part = fmaf(b.x, x23.x, part);
                part = fmaf(b.y, x23.y, part);
                acc = fmaf(part, s, acc);
            }
        }
    }
    return acc;
}
// e2m1 pair (one byte, low nibble first) as FP16 bit patterns: magnitude bits e1 e0 m at FP16 bits 11..9, sign at 15;
// the value is exactly the e2m1 value / 16384 (0.5 lands on the FP16 subnormal 2^-15), so callers fold 16384 into the scale
__device__ __forceinline__ float2 e2m1x2_scaled(uint32_t b) {
    const uint32_t lo = ((b & 0x7u) << 9) | ((b & 0x8u) << 12), hi = ((b & 0x70u) << 5) | ((b & 0x80u) << 8);
    return __half22float2(__halves2half2(__ushort_as_half(uint16_t(lo)), __ushort_as_half(uint16_t(hi))));
}

// NVF4 rows sharing one activation vector: each lane loads its x slice once per 16-byte step and uses it for all R rows
// (row_dot loads it once per row). Per row the operations and their order are row_dot's, so every sum is bitwise equal.
// XG: rows per activation vector (row r dots xs[r / XG]); the per-row operations do not depend on R or XG
template <int NC, int R, bool P, int XG>
__device__ __forceinline__ void rows_dot_nvf4_x(const uint8_t * const * row, const uint8_t * const * aux, int cols, const __nv_bfloat16 * const * xs,
                                                int lane, float * acc) {
    constexpr int NG = R / XG;
    static_assert(R % XG == 0, "row groups");
    float g[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r] = 0.0f; g[r] = __ldg(reinterpret_cast<const float *>(aux[r] + cols / 16)) * 16384.0f; }  // e2m1x2_scaled is the value / 2^14
#pragma unroll
    for (int i0 = 0; i0 < (NC + 31) / 32; ++i0) {
        const int c = i0 * 32 + lane;
        if (c >= NC) break;
        uint4 q[R]; uint32_t s2[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q[r] = __ldcs(reinterpret_cast<const uint4 *>(row[r]) + c);
            s2[r] = __ldg(reinterpret_cast<const uint16_t *>(aux[r]) + c);
        }
        float part[R];
#pragma unroll
        for (int r = 0; r < R; ++r) part[r] = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int gi = 0; gi < NG; ++gi) {  // one group's 8 activations live at a time
                float xv[8];
                unpack8(ldx<P>(reinterpret_cast<const uint4 *>(xs[gi] + c * 32) + w), xv);
#pragma unroll
                for (int r = gi * XG; r < gi * XG + XG; ++r) {
                    if (w == 2) { acc[r] = fmaf(part[r], e4m3(uint8_t(s2[r] & 0xff)) * g[r], acc[r]); part[r] = 0.0f; }
                    const uint32_t qw = w == 0 ? q[r].x : w == 1 ? q[r].y : w == 2 ? q[r].z : q[r].w;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 e = e2m1x2_scaled(qw >> (8 * j));
                        part[r] = fmaf(e.x, xv[2 * j], part[r]);
                        part[r] = fmaf(e.y, xv[2 * j + 1], part[r]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = fmaf(part[r], e4m3(uint8_t(s2[r] >> 8)) * g[r], acc[r]);
    }
}
template <int NC, int R, bool P = false>
__device__ __forceinline__ void rows_dot_nvf4(const uint8_t * const * row, const uint8_t * const * aux, int cols, const __nv_bfloat16 * x,
                                              int lane, const DecodeLut *, float * acc) {
    const __nv_bfloat16 * xs[1] = {x};
    rows_dot_nvf4_x<NC, R, P, R>(row, aux, cols, xs, lane, acc);
}

// a row's data / aux pointers inside expert e
template <int FMT>
__device__ __forceinline__ const uint8_t * row_ptr(const ExpertMat & m, int e, int row, int cols) {
    return (m.wp ? m.wp[e] : m.w + size_t(e) * m.stride) + size_t(row) * (FMT == 5 ? size_t(cols) * 3 / 8 : FMT == 4 ? size_t(cols / 4) : FMT >= 2 ? size_t(cols / 2) : size_t(cols + cols / 32));
}
template <int FMT>
__device__ __forceinline__ const uint8_t * aux_ptr(const ExpertMat & m, int e, int row, int cols) {
    if constexpr (FMT >= 3) return (m.ap ? m.ap[e] : m.aux + size_t(e) * m.aux_stride) + size_t(row) * (cols / 128) * 2;
    return FMT == 2 ? (m.ap ? m.ap[e] : m.aux + size_t(e) * m.aux_stride) + size_t(row) * (cols / 16 + 4) : nullptr;
}

// warp (assignment a, row): row of the gate and the up matrix of assignment a
template <int FMT, int NC, bool P>
__device__ __forceinline__ void expert_gate_up_warp(const ExpertMat & mg, const ExpertMat & mu, int rows, int cols, const __nv_bfloat16 * x, const int * ids,
                                                    __nv_bfloat16 * act, int topk, float gc, float uc, int a, int row, int lane, const DecodeLut * lut) {
    const int e = ids[a];
    const __nv_bfloat16 * xr = x + size_t(a / topk) * cols;
    const float g0 = row_dot<FMT, NC, P>(row_ptr<FMT>(mg, e, row, cols), aux_ptr<FMT>(mg, e, row, cols), cols, xr, lane, lut);
    const float u0 = row_dot<FMT, NC, P>(row_ptr<FMT>(mu, e, row, cols), aux_ptr<FMT>(mu, e, row, cols), cols, xr, lane, lut);
    const float g = bfr(warp_sum(g0)), u = bfr(warp_sum(u0));
    if (lane == 0) act[size_t(a) * rows + row] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
}
// grid (rows / 8, n * topk): warp = row i of the gate and the up matrix of assignment blockIdx.y
template <int FMT, int NC>
__global__ void __launch_bounds__(256) expert_gate_up_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x, const int * ids,
                                                        __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int a = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (row >= rows) return;
    expert_gate_up_warp<FMT, NC, false>(mg, mu, rows, cols, x, ids, act, topk, gc, uc, a, row, lane, &lut);
}

// NVF4, R rows per warp (row0 = first): the warp's R gate rows and R up rows share the x loads
template <int NC, int R, bool P>
__device__ __forceinline__ void expert_gate_up_nvf4_warp(const ExpertMat & mg, const ExpertMat & mu, int rows, int cols, const __nv_bfloat16 * x,
                                                         const int * ids, __nv_bfloat16 * act, int topk, float gc, float uc, int a, int row0, int lane,
                                                         const DecodeLut * lut) {
    const int e = ids[a];
    const uint8_t * rp[2 * R]; const uint8_t * ap[2 * R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, rows - 1);
        rp[r] = row_ptr<2>(mg, e, row, cols); ap[r] = aux_ptr<2>(mg, e, row, cols);
        rp[R + r] = row_ptr<2>(mu, e, row, cols); ap[R + r] = aux_ptr<2>(mu, e, row, cols);
    }
    float acc[2 * R];
    rows_dot_nvf4<NC, 2 * R, P>(rp, ap, cols, x + size_t(a / topk) * cols, lane, lut, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float g = bfr(warp_sum(acc[r])), u = bfr(warp_sum(acc[R + r]));
        if (lane == 0 && row0 + r < rows)
            act[size_t(a) * rows + row0 + r] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
    }
}
// grid (rows / (8 R), n * topk)
template <int NC, int R>
__global__ void __launch_bounds__(256) expert_gate_up_nvf4_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x,
                                                             const int * ids, __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int a = blockIdx.y, row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    if (row0 >= rows) return;
    expert_gate_up_nvf4_warp<NC, R, false>(mg, mu, rows, cols, x, ids, act, topk, gc, uc, a, row0, lane, nullptr);  // NVF4 decodes without the table
}
// grid (rows / R, n), one warp per routed expert k (topk warps): warp k's R row sums, then the combine over k in order
// (the per-row sums of a warp that reads the 8 experts one after another, spread over 8 warps)
template <int NC, int R>
__global__ void __launch_bounds__(256) expert_down_nvf4_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids,
                                                          const float * wts, __nv_bfloat16 * out, int topk) {
    const int t = blockIdx.y, row0 = blockIdx.x * R, k = threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ float ps[16][R];
    {
        const int a = t * topk + k, e = ids[a];
        const uint8_t * rp[R]; const uint8_t * ap[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { const int row = min(row0 + r, rows - 1); rp[r] = row_ptr<2>(md, e, row, cols); ap[r] = aux_ptr<2>(md, e, row, cols); }
        float p[R];
        rows_dot_nvf4<NC, R>(rp, ap, cols, act + size_t(a) * cols, lane, nullptr, p);
#pragma unroll
        for (int r = 0; r < R; ++r) { const float v = bfr(warp_sum(p[r])); if (lane == 0) ps[k][r] = v; }
    }
    __syncthreads();
    if (int(threadIdx.x) < R && row0 + int(threadIdx.x) < rows) {
        const int r = threadIdx.x;
        float y = 0.0f;
        for (int kk = 0; kk < topk; ++kk) y = fmaf(ps[kk][r], wts[t * topk + kk], y);
        out[size_t(t) * rows + row0 + r] = __float2bfloat16(y);
    }
}

// warp per 128 columns: signs S1, a 128-point Walsh-Hadamard transform (4 values per lane + shuffles), / sqrt(128), FP16
__constant__ uint32_t kRotS1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
__global__ void rotate_h_k(const __nv_bfloat16 * x, __half * xh, int k) {
    const int col = blockIdx.y, lane = threadIdx.x % 32, grp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (grp * 128 >= k) return;
    const size_t base = size_t(col) * k + size_t(grp) * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float xv = __bfloat162float(x[base + j]);
        v[j] = ((kRotS1[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) ? -xv : xv;
    }
    float a = v[0], b = v[1];
    v[0] = a + b, v[1] = a - b;
    a = v[2], b = v[3];
    v[2] = a + b, v[3] = a - b;
    a = v[0], b = v[2];
    v[0] = a + b, v[2] = a - b;
    a = v[1], b = v[3];
    v[1] = a + b, v[3] = a - b;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? other - v[j] : v[j] + other;
        }
#pragma unroll
    for (int j = 0; j < 4; ++j) xh[base + j] = __float2half_rn(v[j] * 0.08838834764831845f);
}
// TQT row (fmt 3) dotted with the rotated FP16 activation: a lane takes whole 32-weight segments (16 stream bytes + the
// first word of the next segment of its block), the code's FP16 pair values and F32 products, one FP16 scale per block
__device__ __forceinline__ float row_dot_tqt(const uint8_t * row, const uint8_t * aux, int cols, const __half * xh, int lane) {
    const uint4 * w4 = reinterpret_cast<const uint4 *>(row);
    const __half * sc = reinterpret_cast<const __half *>(aux);
    float acc = 0.0f;
    for (int sg = lane; sg < cols / 32; sg += 32) {
        const uint4 v = __ldcs(w4 + sg);
        const uint32_t nx = __ldg(reinterpret_cast<const uint32_t *>(w4 + ((sg & ~3) | ((sg + 1) & 3))));
        const uint32_t E[5] = {v.x, v.y, v.z, v.w, nx};
        const float scl = __half2float(__ldg(sc + sg / 4));
        const uint4 * xp = reinterpret_cast<const uint4 *>(xh + size_t(sg) * 32);
        float part = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 xv = __ldg(xp + q);
            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t wp = tqllm_cuda::tqtk::pair(E, q * 4 + j);
                const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wp));
                const float2 xf = __half22float2(*reinterpret_cast<const __half2 *>(&xw[j]));
                part = fmaf(wf.x, xf.x, part);
                part = fmaf(wf.y, xf.y, part);
            }
        }
        acc = fmaf(scl, part, acc);
    }
    return acc;
}
// TQT2 row (fmt 4): a lane takes a 32-weight segment = 8 stream bytes (+ the first word of the next segment of its block for the
// windows of the last pairs), pair p reads the 16 bits at bit 4p
__device__ __forceinline__ float row_dot_tqt2(const uint8_t * row, const uint8_t * aux, int cols, const __half * xh, int lane) {
    const uint2 * w2 = reinterpret_cast<const uint2 *>(row);
    const __half * sc = reinterpret_cast<const __half *>(aux);
    float acc = 0.0f;
    for (int sg = lane; sg < cols / 32; sg += 32) {
        const uint2 v = __ldcs(w2 + sg);
        const uint32_t nx = __ldg(reinterpret_cast<const uint32_t *>(w2 + ((sg & ~3) | ((sg + 1) & 3))));
        const uint32_t E[3] = {v.x, v.y, nx};
        const float scl = __half2float(__ldg(sc + sg / 4));
        const uint4 * xp = reinterpret_cast<const uint4 *>(xh + size_t(sg) * 32);
        float part = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 xv = __ldg(xp + q);
            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int p = q * 4 + j, i = p / 8, o = 4 * (p % 8);
                const uint32_t wp = tqllm_cuda::tqtk::pair_win16(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], o));
                const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wp));
                const float2 xf = __half22float2(*reinterpret_cast<const __half2 *>(&xw[j]));
                part = fmaf(wf.x, xf.x, part);
                part = fmaf(wf.y, xf.y, part);
            }
        }
        acc = fmaf(scl, part, acc);
    }
    return acc;
}
// TQT3 row (fmt 5): a lane takes a 32-weight segment = 12 stream bytes (+ the first word of the next segment of its block), pair p
// reads the 18 bits at bit 6 p
__device__ __forceinline__ float row_dot_tqt3(const uint8_t * row, const uint8_t * aux, int cols, const __half * xh, int lane) {
    const uint32_t * w3 = reinterpret_cast<const uint32_t *>(row);
    const __half * sc = reinterpret_cast<const __half *>(aux);
    float acc = 0.0f;
    for (int sg = lane; sg < cols / 32; sg += 32) {
        const uint32_t E[4] = {__ldcs(w3 + 3 * sg), __ldcs(w3 + 3 * sg + 1), __ldcs(w3 + 3 * sg + 2), __ldg(w3 + 3 * ((sg & ~3) | ((sg + 1) & 3)))};
        const float scl = __half2float(__ldg(sc + sg / 4));
        const uint4 * xp = reinterpret_cast<const uint4 *>(xh + size_t(sg) * 32);
        float part = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 xv = __ldg(xp + q);
            const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int p = q * 4 + j, i = (6 * p) / 32, o = (6 * p) % 32;
                const uint32_t wp = tqllm_cuda::tqtk::pair_win(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], o));
                const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wp));
                const float2 xf = __half22float2(*reinterpret_cast<const __half2 *>(&xw[j]));
                part = fmaf(wf.x, xf.x, part);
                part = fmaf(wf.y, xf.y, part);
            }
        }
        acc = fmaf(scl, part, acc);
    }
    return acc;
}
template <int FMT> __device__ __forceinline__ float row_dot_trellis(const uint8_t * row, const uint8_t * aux, int cols, const __half * xh, int lane) {
    if constexpr (FMT == 5) return row_dot_tqt3(row, aux, cols, xh, lane);
    else if constexpr (FMT == 4) return row_dot_tqt2(row, aux, cols, xh, lane);
    else return row_dot_tqt(row, aux, cols, xh, lane);
}
// a TQT / TQT2 / TQT3 row against the M members of a dedup group: row_dot_trellis's arithmetic per member (the same
// segments per lane, the same FP16 pair values, F32 products and per-block scale step), so each member's sum equals the
// one-token kernel's bitwise; the stream words are read and decoded once for all members
template <int FMT, int M>
__device__ __forceinline__ void row_dot_trellis_members(const uint8_t * row, const uint8_t * aux, int cols, const __half * const * xs, int lane,
                                                        float (&acc)[M]) {
    const __half * sc = reinterpret_cast<const __half *>(aux);
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    for (int sg = lane; sg < cols / 32; sg += 32) {
        uint32_t E[5];
        if constexpr (FMT == 5) {
            const uint32_t * w3 = reinterpret_cast<const uint32_t *>(row);
            E[0] = __ldcs(w3 + 3 * sg); E[1] = __ldcs(w3 + 3 * sg + 1); E[2] = __ldcs(w3 + 3 * sg + 2); E[3] = __ldg(w3 + 3 * ((sg & ~3) | ((sg + 1) & 3)));
            E[4] = 0;
        } else if constexpr (FMT == 4) {
            const uint2 * w2 = reinterpret_cast<const uint2 *>(row);
            const uint2 v = __ldcs(w2 + sg);
            E[0] = v.x; E[1] = v.y; E[2] = __ldg(reinterpret_cast<const uint32_t *>(w2 + ((sg & ~3) | ((sg + 1) & 3)))); E[3] = E[4] = 0;
        } else {
            const uint4 * w4 = reinterpret_cast<const uint4 *>(row);
            const uint4 v = __ldcs(w4 + sg);
            E[0] = v.x; E[1] = v.y; E[2] = v.z; E[3] = v.w; E[4] = __ldg(reinterpret_cast<const uint32_t *>(w4 + ((sg & ~3) | ((sg + 1) & 3))));
        }
        const float scl = __half2float(__ldg(sc + sg / 4));
        float part[M];
#pragma unroll
        for (int m = 0; m < M; ++m) part[m] = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint32_t xw[M][4];
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4 xv = __ldg(reinterpret_cast<const uint4 *>(xs[m] + size_t(sg) * 32) + q);
                xw[m][0] = xv.x; xw[m][1] = xv.y; xw[m][2] = xv.z; xw[m][3] = xv.w;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int p = q * 4 + j;
                uint32_t wp;
                if constexpr (FMT == 5) { const int i = (6 * p) / 32, o = (6 * p) % 32; wp = tqllm_cuda::tqtk::pair_win(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], o)); }
                else if constexpr (FMT == 4) { const int i = p / 8, o = 4 * (p % 8); wp = tqllm_cuda::tqtk::pair_win16(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], o)); }
                else wp = tqllm_cuda::tqtk::pair(E, p);
                const float2 wf = __half22float2(*reinterpret_cast<const __half2 *>(&wp));
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const float2 xf = __half22float2(*reinterpret_cast<const __half2 *>(&xw[m][j]));
                    part[m] = fmaf(wf.x, xf.x, part[m]);
                    part[m] = fmaf(wf.y, xf.y, part[m]);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = fmaf(scl, part[m], acc[m]);
    }
}
template <int FMT, int M>
__device__ __forceinline__ void dedup_gate_up_tqt_body(const ExpertMat & mg, const ExpertMat & mu, int rows, int cols, const __half * xh, int e,
                                                       const int * members, __nv_bfloat16 * act, int topk, float gc, float uc, int row, int lane) {
    const __half * xs[M];
#pragma unroll
    for (int m = 0; m < M; ++m) xs[m] = xh + size_t(members[m] / topk) * cols;
    float ga[M], ua[M];
    row_dot_trellis_members<FMT, M>(row_ptr<FMT>(mg, e, row, cols), aux_ptr<FMT>(mg, e, row, cols), cols, xs, lane, ga);
    row_dot_trellis_members<FMT, M>(row_ptr<FMT>(mu, e, row, cols), aux_ptr<FMT>(mu, e, row, cols), cols, xs, lane, ua);
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float g = bfr(warp_sum(ga[m])), u = bfr(warp_sum(ua[m]));
        if (lane == 0) act[size_t(members[m]) * rows + row] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
    }
}
template <int FMT, int M>
__device__ __forceinline__ void dedup_down_tqt_body(const ExpertMat & md, int rows, int cols, const __half * acth, int e, const int * members,
                                                    __nv_bfloat16 * y, int row, int lane) {
    const __half * xs[M];
#pragma unroll
    for (int m = 0; m < M; ++m) xs[m] = acth + size_t(members[m]) * cols;
    float acc[M];
    row_dot_trellis_members<FMT, M>(row_ptr<FMT>(md, e, row, cols), aux_ptr<FMT>(md, e, row, cols), cols, xs, lane, acc);
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float v = warp_sum(acc[m]);
        if (lane == 0) y[size_t(members[m]) * rows + row] = __float2bfloat16(v);
    }
}
#define TQT_DEDUP_SWITCH(CALL)                                                                                                      \
    switch (nm) {                                                                                                                   \
        case 1: CALL(1); break; case 2: CALL(2); break; case 3: CALL(3); break; case 4: CALL(4); break;                             \
        case 5: CALL(5); break; case 6: CALL(6); break; case 7: CALL(7); break; default: CALL(8); break;                            \
    }
// grid (rows / 8, n * topk): block y = dedup group (groups from dedup_groups_k), a warp per row for every member
template <int FMT>
__global__ void __launch_bounds__(256) dedup_gate_up_tqt_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __half * xh, const int * ids,
                                                           const int * groups, __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int gi = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (gi >= groups[0] || row >= rows) return;
    const int * gr = groups + 1 + gi * (1 + 8);
    const int nm = gr[0], e = ids[gr[1]];
#define TQT_CALL(M) dedup_gate_up_tqt_body<FMT, M>(mg, mu, rows, cols, xh, e, gr + 1, act, topk, gc, uc, row, lane)
    TQT_DEDUP_SWITCH(TQT_CALL)
#undef TQT_CALL
}
template <int FMT>
__global__ void __launch_bounds__(256) dedup_down_tqt_k(ExpertMat md, int rows, int cols, const __half * acth, const int * ids, const int * groups,
                                                        __nv_bfloat16 * y) {
    const int gi = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (gi >= groups[0] || row >= rows) return;
    const int * gr = groups + 1 + gi * (1 + 8);
    const int nm = gr[0], e = ids[gr[1]];
#define TQT_CALL(M) dedup_down_tqt_body<FMT, M>(md, rows, cols, acth, e, gr + 1, y, row, lane)
    TQT_DEDUP_SWITCH(TQT_CALL)
#undef TQT_CALL
}
#undef TQT_DEDUP_SWITCH
// grid (rows / 8, n * topk): as expert_gate_up_k, xh = x rotated per 128 (FP16)
template <int FMT>
__global__ void __launch_bounds__(256) expert_gate_up_tqt_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __half * xh, const int * ids,
                                                            __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int a = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const int e = ids[a];
    const __half * xr = xh + size_t(a / topk) * cols;
    const float g0 = row_dot_trellis<FMT>(row_ptr<FMT>(mg, e, row, cols), aux_ptr<FMT>(mg, e, row, cols), cols, xr, lane);
    const float u0 = row_dot_trellis<FMT>(row_ptr<FMT>(mu, e, row, cols), aux_ptr<FMT>(mu, e, row, cols), cols, xr, lane);
    const float g = bfr(warp_sum(g0)), u = bfr(warp_sum(u0));
    if (lane == 0) act[size_t(a) * rows + row] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
}
// grid (rows / 8, n): as expert_down_k, acth = act rotated per 128 (FP16 [n * topk][cols])
template <int FMT>
__global__ void __launch_bounds__(256) expert_down_tqt_k(ExpertMat md, int rows, int cols, const __half * acth, const int * ids, const float * wts,
                                                         __nv_bfloat16 * out, int topk) {
    const int t = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    float p[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        const int a = t * topk + k, e = ids[a];
        p[k] = row_dot_trellis<FMT>(row_ptr<FMT>(md, e, row, cols), aux_ptr<FMT>(md, e, row, cols), cols, acth + size_t(a) * cols, lane);
    }
    float y = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        y = fmaf(bfr(warp_sum(p[k])), wts[t * topk + k], y);
    }
    if (lane == 0) out[size_t(t) * rows + row] = __float2bfloat16(y);
}

// warp (token t, row): output row over the token's topk experts
template <int FMT, int NC, bool P>
__device__ __forceinline__ void expert_down_warp(const ExpertMat & md, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const float * wts,
                                                 __nv_bfloat16 * out, int topk, int t, int row, int lane, const DecodeLut * lut) {
    float p[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        const int a = t * topk + k, e = ids[a];
        p[k] = row_dot<FMT, NC, P>(row_ptr<FMT>(md, e, row, cols), aux_ptr<FMT>(md, e, row, cols), cols, act + size_t(a) * cols, lane, lut);
    }
    float y = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        y = fmaf(bfr(warp_sum(p[k])), wts[t * topk + k], y);
    }
    if (lane == 0) out[size_t(t) * rows + row] = __float2bfloat16(y);
}
// grid (rows / 8, n): warp = output row of token blockIdx.y over its topk experts
template <int FMT, int NC>
__global__ void __launch_bounds__(256) expert_down_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const float * wts,
                                                     __nv_bfloat16 * out, int topk) {
    const int t = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (row >= rows) return;
    expert_down_warp<FMT, NC, false>(md, rows, cols, act, ids, wts, out, topk, t, row, lane, &lut);
}

// one warp per token: s = sigmoid(logit); top-k of s + bias (lowest index on ties); w = s / sum * scale
__device__ __forceinline__ void route_topk_warp(const float * logits, const float * bias, int * ids, float * wts, int E, int topk, float scale, int t,
                                                int lane, float * s, float * sel) {
    for (int e = lane; e < E; e += 32) {
        const float d = logits[size_t(t) * E + e];
        s[e] = 1.0f / (1.0f + expf(-d));
        sel[e] = s[e] + bias[e];
    }
    __syncwarp();
    float sum = 0.0f;
    int chosen[16];
    for (int k = 0; k < topk; ++k) {
        float best = -INFINITY; int bi = 0x7fffffff;
        for (int e = lane; e < E; e += 32) if (sel[e] > best || (sel[e] == best && e < bi)) { best = sel[e]; bi = e; }
        for (int o = 16; o; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        chosen[k] = bi;
        sum += s[bi];
        __syncwarp();
        if (lane == 0) sel[bi] = -INFINITY;
        __syncwarp();
    }
    if (lane == 0)
        for (int k = 0; k < topk; ++k) { ids[t * topk + k] = chosen[k]; wts[t * topk + k] = s[chosen[k]] / (sum + 1e-20f) * scale; }
}
__global__ void route_topk_k(const float * logits, const float * bias, int * ids, float * wts, int E, int topk, float scale, int * counts) {
    __shared__ float s[256], sel[256];
    route_topk_warp(logits, bias, ids, wts, E, topk, scale, blockIdx.x, threadIdx.x, s, sel);
    __syncwarp();  // lane 0's ids are visible to the warp
    if (counts && int(threadIdx.x) < topk) atomicAdd(counts + ids[blockIdx.x * topk + threadIdx.x], 1);
}

// one block (512 threads) per token: depth attention for nq streams (F32 out, optional) and, per stream with a weight,
// y16 = bf16(w * rms(stream)) as rms_norm()
struct NormOut { const float * w[4]; __nv_bfloat16 * y[4]; };
constexpr int kArnThreads = 512, kArnPer = 5;  // D <= 2560
// NQ >= nq sizes the per-thread accumulators (the FFN call, nq = 1, is half the registers and reductions of nq = 3)
template <int NQ>
__global__ void __launch_bounds__(kArnThreads) attn_res_norm_k(States st, const float * __restrict__ q, int nq, float * out, NormOut no, int n, int D,
                                                               float eps) {
    const int t = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int NV = kMaxStates * (1 + NQ), NW = kArnThreads / 32, NS = 1 + NQ;
    __shared__ float part[NW][NV], tot[NV];
    float v[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) v[k] = 0.0f;
#pragma unroll
    for (int u = 0; u < kArnPer; ++u) {
        const int c = threadIdx.x + u * kArnThreads;
        if (c >= D) break;
        float qc[NQ];
#pragma unroll
        for (int i = 0; i < NQ; ++i) qc[i] = i < nq ? q[size_t(i) * D + c] : 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) {
            if (j >= st.count) break;
            const float x = st.s[j][size_t(t) * D + c];
            v[j * NS] = fmaf(x, x, v[j * NS]);
#pragma unroll
            for (int i = 0; i < NQ; ++i) if (i < nq) v[j * NS + 1 + i] = fmaf(qc[i], x, v[j * NS + 1 + i]);
        }
    }
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) {
        if (j >= st.count) break;
#pragma unroll
        for (int i = 0; i <= NQ; ++i) {
            if (i > nq) break;
            const float r = warp_sum(v[j * NS + i]);
            if (lane == 0) part[warp][j * NS + i] = r;
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < st.count * NS; k += kArnThreads) {
        float r = 0.0f;
        for (int w = 0; w < NW; ++w) r += part[w][k];
        tot[k] = r;
    }
    __syncthreads();
    float inv[kMaxStates];
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) inv[j] = j < st.count ? rsqrtf(tot[j * NS] / float(D) + 1e-6f) : 0.0f;
    const float sd = rsqrtf(float(D));
    float ys[NQ][kArnPer], ss[NQ];
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        ss[i] = 0.0f;
        if (i >= nq) continue;
        float p[kMaxStates], m = -INFINITY;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) if (j < st.count) { p[j] = tot[j * NS + 1 + i] * inv[j] * sd; m = fmaxf(m, p[j]); }
        float l = 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) if (j < st.count) l += expf(p[j] - m);
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) if (j < st.count) p[j] = expf(p[j] - m) / l;
#pragma unroll
        for (int u = 0; u < kArnPer; ++u) {
            const int c = threadIdx.x + u * kArnThreads;
            float y = 0.0f;
            if (c < D) {
#pragma unroll
                for (int j = 0; j < kMaxStates; ++j) if (j < st.count) y += p[j] * st.s[j][size_t(t) * D + c] * inv[j];
                if (out) out[(size_t(i) * n + t) * D + c] = y;
                ss[i] = fmaf(y, y, ss[i]);
            }
            ys[i][u] = y;
        }
    }
    __syncthreads();  // part[] is reused
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        if (i >= nq || !no.w[i]) continue;
        const float r = warp_sum(ss[i]);
        if (lane == 0) part[warp][i] = r;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        if (i >= nq || !no.w[i]) continue;
        float tsum = 0.0f;
        for (int w = 0; w < NW; ++w) tsum += part[w][i];
        const float r = rsqrtf(tsum / float(D) + eps);
#pragma unroll
        for (int u = 0; u < kArnPer; ++u) {
            const int c = threadIdx.x + u * kArnThreads;
            if (c < D) no.y[i][size_t(t) * D + c] = __float2bfloat16(no.w[i][c] * (ys[i][u] * r));
        }
    }
}

// decode: one block per token reads every state twice through a single SM, which caps the kernel at ~10 GB/s of DRAM
// (one SM's share); split D over kArnSplit blocks per token. Two kernels: partial per-(state, stream) sums, then the
// softmax mix + norms reading the states back from L2. scratch: [n][kArnSplit][kMaxStates * 5] partials + n counters
// (zeroed once at allocation; the last block of the stats kernel resets its token's counter). The sum order differs from
// attn_res_norm_k's (per-block partials, ascending); the same order for any n, so n-token == 1-token stays bitwise.
constexpr int kArnSplit = 8, kArnSplitThreads = 288;
constexpr int kArnNV = kMaxStates * 5, kArnNW = kArnSplitThreads / 32;
// the per-channel terms of the depth-attention statistics (channel c < D of token t): v[j * 5] = x_j^2, v[j * 5 + 1 + i] = q_i x_j
__device__ __forceinline__ void arn_stats_vals(const States & st, const float * q, int nq, int t, int c, int D, float * v) {
#pragma unroll
    for (int k = 0; k < kArnNV; ++k) v[k] = 0.0f;
    if (c < D) {
        float qc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) qc[i] = i < nq ? q[size_t(i) * D + c] : 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxStates; ++j) {
            if (j >= st.count) break;
            const float x = st.s[j][size_t(t) * D + c];
            v[j * 5] = fmaf(x, x, v[j * 5]);
#pragma unroll
            for (int i = 0; i < 4; ++i) if (i < nq) v[j * 5 + 1 + i] = fmaf(qc[i], x, v[j * 5 + 1 + i]);
        }
    }
}
// stream i of the depth attention: the softmax weights over the states (the same for every channel), then channel c of token t
__device__ __forceinline__ void arn_mix_p(const States & st, const float * tot, const float * sp, float sd, int i, float * p) {
    float m = -INFINITY;
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) if (j < st.count) { p[j] = tot[j * 5 + 1 + i] * sp[j] * sd; m = fmaxf(m, p[j]); }
    float l = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) if (j < st.count) l += expf(p[j] - m);
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) if (j < st.count) p[j] = expf(p[j] - m) / l;
}
__device__ __forceinline__ float arn_mix_dot(const States & st, const float * p, const float * sp, int t, int c, int D) {
    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxStates; ++j) if (j < st.count) y += p[j] * st.s[j][size_t(t) * D + c] * sp[j];
    return y;
}
__device__ __forceinline__ float arn_mix_val(const States & st, const float * tot, const float * sp, float sd, int i, int t, int c, int D) {
    float p[kMaxStates];
    arn_mix_p(st, tot, sp, sd, i, p);
    return arn_mix_dot(st, p, sp, t, c, D);
}
__device__ __forceinline__ void attn_res_stats_body(const Blk & bk, const States & st, const float * q, int nq, float * part, int D, float (*sh)[kArnNV]) {
    const int t = bk.y, b = bk.x, lane = bk.t % 32, warp = bk.t / 32;
    constexpr int NV = kArnNV, NW = kArnNW;
    const int c = b * kArnSplitThreads + bk.t;
    float v[NV];
    arn_stats_vals(st, q, nq, t, c, D, v);
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        if (k / 5 >= st.count || (k % 5) > nq) continue;
        const float r = warp_sum(v[k]);
        if (lane == 0) sh[warp][k] = r;
    }
    bk.sync();
    if (bk.t < NV && bk.t / 5 < st.count && bk.t % 5 <= nq) {
        float r = 0.0f;
        for (int w = 0; w < NW; ++w) r += sh[w][bk.t];
        part[(size_t(t) * kArnSplit + b) * NV + bk.t] = r;
    }
}
__device__ __forceinline__ void attn_res_mix_body(const Blk & bk, const States & st, int nq, const float * part, float * out, int n, int D, float * tot,
                                                  float * sp) {
    const int t = bk.y, b = bk.x;
    constexpr int NV = kArnNV;
    if (bk.t < NV && bk.t / 5 < st.count && bk.t % 5 <= nq) {
        float r = 0.0f;
        for (int bb = 0; bb < kArnSplit; ++bb) r += part[(size_t(t) * kArnSplit + bb) * NV + bk.t];
        tot[bk.t] = r;
    }
    bk.sync();
    if (bk.t < kMaxStates) sp[bk.t] = bk.t < st.count ? rsqrtf(tot[bk.t * 5] / float(D) + 1e-6f) : 0.0f;
    bk.sync();
    const float sd = rsqrtf(float(D));
    const int c = b * kArnSplitThreads + bk.t;
    if (c < D)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i >= nq) continue;
            const float y = arn_mix_val(st, tot, sp, sd, i, t, c, D);
            if (out) out[(size_t(i) * n + t) * D + c] = y;
        }
}
// prompt chunks: every (token, channel) in parallel (the conv reads inputs only), then the state takes the last 3 inputs;
// the same expressions as the per-channel kernels above
__device__ __forceinline__ float conv_in(const __nv_bfloat16 * x, const float * state, int t, int c, int C) {
    return t >= 0 ? __bfloat162float(x[size_t(t) * C + c]) : state[c * 3 + 3 + t];
}
__global__ void conv_static_par_k(const __nv_bfloat16 * x, const float * w, const float * state, float * y32, __nv_bfloat16 * y16, int n, int C,
                                  const float * add, bool round_sum) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = conv_in(x, state, t - 3, c, C), h1 = conv_in(x, state, t - 2, c, C), h2 = conv_in(x, state, t - 1, c, C);
    const float xt = __bfloat162float(x[size_t(t) * C + c]);
    const float y = (w[c * 4] * h0 + w[c * 4 + 1] * h1 + w[c * 4 + 2] * h2 + w[c * 4 + 3] * xt + xt) * rsqrtf(5.0f);
    const __nv_bfloat16 b = __float2bfloat16(y);
    if (y32) {
        float v = __bfloat162float(b);
        if (add) { v = add[size_t(t) * C + c] + v; if (round_sum) v = bfr(v); }
        y32[size_t(t) * C + c] = v;
    }
    if (y16) y16[size_t(t) * C + c] = b;
}
__global__ void conv_dynamic_par_k(const __nv_bfloat16 * x, const float * stat, const __nv_bfloat16 * dyn, const float * state, __nv_bfloat16 * y16,
                                   int n, int C) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int g = c / 128, heads4 = (C / 128) * 4;
    const float xt = __bfloat162float(x[size_t(t) * C + c]);
    const float xs[4] = {xt, conv_in(x, state, t - 1, c, C), conv_in(x, state, t - 2, c, C), conv_in(x, state, t - 3, c, C)};
    const __nv_bfloat16 * d = dyn + size_t(t) * heads4 + g * 4;
    float y = 0.0f;
    for (int lag = 0; lag < 4; ++lag) y += (stat[lag * C + c] + __bfloat162float(d[lag])) * xs[lag];
    y16[size_t(t) * C + c] = __float2bfloat16((y + xt) / 3.0f);
}
__global__ void conv_state_k(const __nv_bfloat16 * x, float * state, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float a = conv_in(x, state, n - 3, c, C), b = conv_in(x, state, n - 2, c, C), d = conv_in(x, state, n - 1, c, C);
    state[c * 3] = a; state[c * 3 + 1] = b; state[c * 3 + 2] = d;
}

// ---- verify blocks: each distinct routed expert read once for all its assignments ----
// one block, A <= kMaxAssign threads: groups of assignments by expert (leaders in assignment order, members ascending)
constexpr int kMaxAssign = kSmallN * 8, kMaxMembers = kSmallN;
__global__ void dedup_groups_k(const int * ids, int A, int * groups) {  // groups: [0] count, then [g][1 + kMaxMembers]: n, members
    const int a = threadIdx.x;
    __shared__ int lead[kMaxAssign];
    const int e = a < A ? ids[a] : -1;
    int first = a;
    for (int b = 0; b < a; ++b) if (ids[b] == e) { first = b; break; }
    lead[a] = a < A && first == a;
    __syncthreads();
    if (a < A && lead[a]) {
        int g = 0;
        for (int b = 0; b < a; ++b) g += lead[b];
        int * gr = groups + 1 + g * (1 + kMaxMembers), cnt = 0;
        for (int b = a; b < A; ++b) if (ids[b] == e && cnt < kMaxMembers) gr[1 + cnt++] = b;
        gr[0] = cnt;
    }
    __syncthreads();
    if (a == 0) { int c = 0; for (int b = 0; b < A; ++b) c += lead[b]; groups[0] = c; }
}

// one row of an expert against up to kMaxMembers inputs, each with row_dot's exact per-input arithmetic
template <int FMT>
__device__ __forceinline__ void row_dot_multi(const uint8_t * row, const uint8_t * aux, int cols, const __nv_bfloat16 * const * xs, int nm, int lane,
                                              float * acc, const DecodeLut * lut) {
    for (int m = 0; m < kMaxMembers; ++m) acc[m] = 0.0f;
    if constexpr (FMT == 2) {
        const int nc = cols / 32;
        const float g = __ldg(reinterpret_cast<const float *>(aux + cols / 16));
        for (int c = lane; c < nc; c += 32) {
            const uint4 q = __ldcs(reinterpret_cast<const uint4 *>(row) + c);
            const uint32_t s2 = __ldg(reinterpret_cast<const uint16_t *>(aux) + c);
            const float sc[2] = {e4m3(uint8_t(s2 & 0xff)), e4m3(uint8_t(s2 >> 8))};
            const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
            float wv[32];
#pragma unroll
            for (int w = 0; w < 4; ++w)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 e = fp4x2(lut, qw[w] >> (8 * j));
                    wv[w * 8 + 2 * j] = e.x;
                    wv[w * 8 + 2 * j + 1] = e.y;
                }
#pragma unroll
            for (int m = 0; m < kMaxMembers; ++m) {
                if (m >= nm) break;
                float part = 0.0f;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    if (w == 2) { acc[m] = fmaf(part, sc[0] * g, acc[m]); part = 0.0f; }
                    float xv[8];
                    unpack8(__ldg(reinterpret_cast<const uint4 *>(xs[m] + c * 32) + w), xv);
#pragma unroll
                    for (int j = 0; j < 8; ++j) part = fmaf(wv[w * 8 + j], xv[j], part);
                }
                acc[m] = fmaf(part, sc[1] * g, acc[m]);
            }
        }
    } else {
        for (int c0 = lane * 4; c0 < cols; c0 += 128) {
            const uint32_t q = __ldg(reinterpret_cast<const uint32_t *>(row + c0));
            const float s = e8m0(__ldg(row + cols + c0 / 32));
            const float2 a = fp8x2(lut, q), b = fp8x2(lut, q >> 16);
#pragma unroll
            for (int m = 0; m < kMaxMembers; ++m) {
                if (m >= nm) break;
                const float2 x01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(xs[m] + c0));
                const float2 x23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(xs[m] + c0 + 2));
                float part = a.x * x01.x;
                part = fmaf(a.y, x01.y, part);
                part = fmaf(b.x, x23.x, part);
                part = fmaf(b.y, x23.y, part);
                acc[m] = fmaf(part, s, acc[m]);
            }
        }
    }
}

// grid (rows / 8, n * topk): block y = group; gate / up rows of its expert for every member -> act[a][row]
template <int FMT>
__global__ void __launch_bounds__(256) dedup_gate_up_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x, const int * ids,
                                                       const int * groups, __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int gi = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (gi >= groups[0] || row >= rows) return;
    const int * gr = groups + 1 + gi * (1 + kMaxMembers);
    const int nm = gr[0], e = ids[gr[1]];
    const __nv_bfloat16 * xs[kMaxMembers];
#pragma unroll
    for (int m = 0; m < kMaxMembers; ++m) xs[m] = x + size_t(gr[1 + (m < nm ? m : 0)] / topk) * cols;
    float ga[kMaxMembers], ua[kMaxMembers];
    row_dot_multi<FMT>(row_ptr<FMT>(mg, e, row, cols), aux_ptr<FMT>(mg, e, row, cols), cols, xs, nm, lane, ga, &lut);
    row_dot_multi<FMT>(row_ptr<FMT>(mu, e, row, cols), aux_ptr<FMT>(mu, e, row, cols), cols, xs, nm, lane, ua, &lut);
#pragma unroll
    for (int m = 0; m < kMaxMembers; ++m) {
        if (m >= nm) break;
        const float g = bfr(warp_sum(ga[m])), u = bfr(warp_sum(ua[m]));
        if (lane == 0) act[size_t(gr[1 + m]) * rows + row] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
    }
}
// down rows of the group's expert for every member -> y[a][row] = bf16(dot) (then moe_combine)
template <int FMT>
__global__ void __launch_bounds__(256) dedup_down_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const int * groups,
                                                    __nv_bfloat16 * y) {
    const int gi = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (gi >= groups[0] || row >= rows) return;
    const int * gr = groups + 1 + gi * (1 + kMaxMembers);
    const int nm = gr[0], e = ids[gr[1]];
    const __nv_bfloat16 * xs[kMaxMembers];
#pragma unroll
    for (int m = 0; m < kMaxMembers; ++m) xs[m] = act + size_t(gr[1 + (m < nm ? m : 0)]) * cols;
    float acc[kMaxMembers];
    row_dot_multi<FMT>(row_ptr<FMT>(md, e, row, cols), aux_ptr<FMT>(md, e, row, cols), cols, xs, nm, lane, acc, &lut);
#pragma unroll
    for (int m = 0; m < kMaxMembers; ++m) {
        if (m >= nm) break;
        const float v = warp_sum(acc[m]);
        if (lane == 0) y[size_t(gr[1 + m]) * rows + row] = __float2bfloat16(v);
    }
}

// NVF4 rows against the M members of a dedup group: rows_dot_nvf4_x's arithmetic per (row, member), so a member's results
// equal the one-token kernels bitwise; each weight chunk is read once for all members, each activation chunk once per R rows
template <int NC, int R, int M>
__device__ __forceinline__ void rows_dot_nvf4_members(const uint8_t * const * row, const uint8_t * const * aux, int cols, const __nv_bfloat16 * const * xs,
                                                      int lane, float (&acc)[R][M]) {
    float g[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        g[r] = __ldg(reinterpret_cast<const float *>(aux[r] + cols / 16)) * 16384.0f;
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    }
#pragma unroll
    for (int i0 = 0; i0 < (NC + 31) / 32; ++i0) {
        const int c = i0 * 32 + lane;
        if (c >= NC) break;
        uint4 q[R]; uint32_t s2[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q[r] = __ldcs(reinterpret_cast<const uint4 *>(row[r]) + c);
            s2[r] = __ldg(reinterpret_cast<const uint16_t *>(aux[r]) + c);
        }
        float part[R][M];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) part[r][m] = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xv[8];
                unpack8(__ldg(reinterpret_cast<const uint4 *>(xs[m] + c * 32) + w), xv);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (w == 2) { acc[r][m] = fmaf(part[r][m], e4m3(uint8_t(s2[r] & 0xff)) * g[r], acc[r][m]); part[r][m] = 0.0f; }
                    const uint32_t qw = w == 0 ? q[r].x : w == 1 ? q[r].y : w == 2 ? q[r].z : q[r].w;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 e = e2m1x2_scaled(qw >> (8 * j));
                        part[r][m] = fmaf(e.x, xv[2 * j], part[r][m]);
                        part[r][m] = fmaf(e.y, xv[2 * j + 1], part[r][m]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) acc[r][m] = fmaf(part[r][m], e4m3(uint8_t(s2[r] >> 8)) * g[r], acc[r][m]);
    }
}
template <int NC, int R, int M>
__device__ __forceinline__ void dedup_gate_up_nvf4_body(const ExpertMat & mg, const ExpertMat & mu, int rows, int cols, const __nv_bfloat16 * x, int e,
                                                        const int * members, __nv_bfloat16 * act, int topk, float gc, float uc, int row0, int lane) {
    const uint8_t * rp[2 * R]; const uint8_t * ap[2 * R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, rows - 1);
        rp[r] = row_ptr<2>(mg, e, row, cols); ap[r] = aux_ptr<2>(mg, e, row, cols);
        rp[R + r] = row_ptr<2>(mu, e, row, cols); ap[R + r] = aux_ptr<2>(mu, e, row, cols);
    }
    const __nv_bfloat16 * xs[M];
#pragma unroll
    for (int m = 0; m < M; ++m) xs[m] = x + size_t(members[m] / topk) * cols;
    float acc[2 * R][M];
    rows_dot_nvf4_members<NC, 2 * R, M>(rp, ap, cols, xs, lane, acc);
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float g = bfr(warp_sum(acc[r][m])), u = bfr(warp_sum(acc[R + r][m]));
            if (lane == 0 && row0 + r < rows)
                act[size_t(members[m]) * rows + row0 + r] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
        }
}
template <int NC, int R, int M>
__device__ __forceinline__ void dedup_down_nvf4_body(const ExpertMat & md, int rows, int cols, const __nv_bfloat16 * act, int e, const int * members,
                                                     __nv_bfloat16 * y, int row0, int lane) {
    const uint8_t * rp[R]; const uint8_t * ap[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { const int row = min(row0 + r, rows - 1); rp[r] = row_ptr<2>(md, e, row, cols); ap[r] = aux_ptr<2>(md, e, row, cols); }
    const __nv_bfloat16 * xs[M];
#pragma unroll
    for (int m = 0; m < M; ++m) xs[m] = act + size_t(members[m]) * cols;
    float acc[R][M];
    rows_dot_nvf4_members<NC, R, M>(rp, ap, cols, xs, lane, acc);
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_sum(acc[r][m]);
            if (lane == 0 && row0 + r < rows) y[size_t(members[m]) * rows + row0 + r] = __float2bfloat16(v);
        }
}
#define TQ_DEDUP_SWITCH(CALL)                                                                                                       \
    switch (nm) {                                                                                                                   \
        case 1: CALL(1); break; case 2: CALL(2); break; case 3: CALL(3); break; case 4: CALL(4); break;                             \
        case 5: CALL(5); break; case 6: CALL(6); break; case 7: CALL(7); break; default: CALL(8); break;                            \
    }
static_assert(kMaxMembers == 8, "dedup member switch");
// grid (rows / (8 R), n * topk): block y = group, a warp's R gate and R up rows for every member
template <int NC, int R>
__global__ void __launch_bounds__(256) dedup_gate_up_nvf4_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x, const int * ids,
                                                            const int * groups, __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int gi = blockIdx.y, row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    if (gi >= groups[0] || row0 >= rows) return;
    const int * gr = groups + 1 + gi * (1 + kMaxMembers);
    const int nm = gr[0], e = ids[gr[1]];
#define TQ_CALL(M) dedup_gate_up_nvf4_body<NC, R, M>(mg, mu, rows, cols, x, e, gr + 1, act, topk, gc, uc, row0, lane)
    TQ_DEDUP_SWITCH(TQ_CALL)
#undef TQ_CALL
}
template <int NC, int R>
__global__ void __launch_bounds__(256) dedup_down_nvf4_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const int * groups,
                                                         __nv_bfloat16 * y) {
    const int gi = blockIdx.y, row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    if (gi >= groups[0] || row0 >= rows) return;
    const int * gr = groups + 1 + gi * (1 + kMaxMembers);
    const int nm = gr[0], e = ids[gr[1]];
#define TQ_CALL(M) dedup_down_nvf4_body<NC, R, M>(md, rows, cols, act, e, gr + 1, y, row0, lane)
    TQ_DEDUP_SWITCH(TQ_CALL)
#undef TQ_CALL
}
#undef TQ_DEDUP_SWITCH

// ---- routed experts on tensor cores (decode and verify, 1 .. 8 tokens): one m16n8k16 BF16 MMA tile = 16 rows of one expert x
// the (up to 8) assignments routed to it (dedup group members as MMA columns). Weights decode exactly into BF16 (NVF4:
// e2m1 x e4m3 block scale; MXFP8: e4m3 x 2^e8m0), the F32 global scale after the sum. A thread holds 16 consecutive columns
// of a 64-column chunk; the MMA's k order is a fixed permutation of them (the same for weights and activations). Every
// output element depends only on its own row and column data and a k order fixed by the shapes, so a token's results do
// not depend on the other tokens, the group sizes or n: verify rows == single steps bitwise.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}
// one row's CPT consecutive columns of a chunk (4 CPT columns: the row's 4 threads side by side), decoded on demand into BF16
// pairs: pair p = columns 2p, 2p + 1 of the thread's slice
template <int FMT, int CPT> struct MmaRow;
template <int CPT> struct MmaRow<2, CPT> {  // NVF4: CPT / 2 bytes of codes (low nibble first), one E4M3 scale per 16 columns
    uint32_t w[CPT / 8]; float sc[CPT / 16];
    __device__ __forceinline__ void load(const uint8_t * row, const uint8_t * aux, int cols, int c, int t) {
        const uint4 * q = reinterpret_cast<const uint4 *>(row + (4 * c + t) * (CPT / 2));
#pragma unroll
        for (int i = 0; i < CPT / 32; ++i) { const uint4 v = __ldcs(q + i); w[4 * i] = v.x; w[4 * i + 1] = v.y; w[4 * i + 2] = v.z; w[4 * i + 3] = v.w; }
        const uint8_t * sp = aux + (4 * c + t) * (CPT / 16);
#pragma unroll
        for (int i = 0; i < CPT / 16; ++i) sc[i] = e4m3(__ldg(sp + i)) * 16384.0f;  // e2m1x2_scaled is the value / 2^14
    }
    __device__ __forceinline__ uint32_t pair(int p) const {
        const float2 e = e2m1x2_scaled(w[p / 4] >> (8 * (p & 3)));
        return pack_bf16x2(e.x * sc[p / 8], e.y * sc[p / 8]);
    }
};
template <int CPT> struct MmaRow<1, CPT> {  // MXFP8: CPT E4M3 bytes (rows are 4-byte aligned), one E8M0 scale per 32 columns
    uint32_t w[CPT / 4]; float sc[CPT / 32];
    __device__ __forceinline__ void load(const uint8_t * row, const uint8_t *, int cols, int c, int t) {
        const uint32_t * q = reinterpret_cast<const uint32_t *>(row + (4 * c + t) * CPT);
#pragma unroll
        for (int i = 0; i < CPT / 4; ++i) w[i] = __ldcs(q + i);
#pragma unroll
        for (int i = 0; i < CPT / 32; ++i) sc[i] = e8m0(__ldg(row + cols + (4 * c + t) * (CPT / 32) + i));
    }
    __device__ __forceinline__ uint32_t pair(int p) const {
        const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t((w[p / 2] >> (16 * (p & 1))) & 0xffff), __NV_E4M3)));
        return pack_bf16x2(v.x * sc[p / 16], v.y * sc[p / 16]);
    }
};
// TQT rows (fmt 3 / 4 / 5, 32 columns = one segment per thread, a 4-thread chunk = one 128-column block with one FP16 scale):
// the decoded FP16 pairs as stored, FP16 MMA against the rotated FP16 activations, the block scale on the chunk's F32 sum
template <int FMT> struct MmaRowTqt {
    uint32_t E[FMT == 3 ? 5 : FMT == 4 ? 3 : 4]; float sc;
    __device__ __forceinline__ void load(const uint8_t * row, const uint8_t * aux, int, int c, int t) {
        const int sg = 4 * c + t, nx = 4 * c + ((t + 1) & 3);
        if constexpr (FMT == 3) {
            const uint4 v = __ldcs(reinterpret_cast<const uint4 *>(row) + sg);
            E[0] = v.x; E[1] = v.y; E[2] = v.z; E[3] = v.w; E[4] = __ldg(reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint4 *>(row) + nx));
        } else if constexpr (FMT == 4) {
            const uint2 v = __ldcs(reinterpret_cast<const uint2 *>(row) + sg);
            E[0] = v.x; E[1] = v.y; E[2] = __ldg(reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint2 *>(row) + nx));
        } else {
            const uint32_t * w3 = reinterpret_cast<const uint32_t *>(row);
            E[0] = __ldcs(w3 + 3 * sg); E[1] = __ldcs(w3 + 3 * sg + 1); E[2] = __ldcs(w3 + 3 * sg + 2); E[3] = __ldg(w3 + 3 * nx);
        }
        sc = __half2float(__ldg(reinterpret_cast<const __half *>(aux) + c));
    }
    __device__ __forceinline__ uint32_t pair(int p) const {
        if constexpr (FMT == 3) return tqllm_cuda::tqtk::pair(E, p);
        else if constexpr (FMT == 4) {
            const int i = p / 8, o = 4 * (p % 8);
            return tqllm_cuda::tqtk::pair_win16(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], o));
        } else {
            const int i = (6 * p) / 32, o = (6 * p) % 32;
            return tqllm_cuda::tqtk::pair_win(o == 0 ? E[i] : __funnelshift_r(E[i], E[i + 1], o));
        }
    }
};
template <int CPT> struct MmaRow<3, CPT> : MmaRowTqt<3> { static_assert(CPT == 32, "TQT: one segment per thread"); };
template <int CPT> struct MmaRow<4, CPT> : MmaRowTqt<4> { static_assert(CPT == 32, "TQT: one segment per thread"); };
template <int CPT> struct MmaRow<5, CPT> : MmaRowTqt<5> { static_assert(CPT == 32, "TQT: one segment per thread"); };
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// the CPT / 4 MMAs of one chunk: rows (lo = tile row g, hi = g + 8) against xb (this thread's CPT columns of token g as BF16
// pairs); MMA j takes the thread's columns 4j .. 4j + 3 as its k 2t, 2t + 1, 2t + 8, 2t + 9
template <int FMT, int CPT>
__device__ __forceinline__ void mma_chunk(float (&c)[4], const MmaRow<FMT, CPT> & lo, const MmaRow<FMT, CPT> & hi, const uint32_t (&xb)[CPT / 2]) {
    if constexpr (FMT >= 3) {  // TQT: the chunk is one scale block; its F32 sum times the row's scale
        float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < CPT / 4; ++j) {
            const uint32_t a[4] = {lo.pair(2 * j), hi.pair(2 * j), lo.pair(2 * j + 1), hi.pair(2 * j + 1)};
            mma_f16(p, a, xb[2 * j], xb[2 * j + 1]);
        }
        c[0] = fmaf(p[0], lo.sc, c[0]); c[1] = fmaf(p[1], lo.sc, c[1]);
        c[2] = fmaf(p[2], hi.sc, c[2]); c[3] = fmaf(p[3], hi.sc, c[3]);
        return;
    }
#pragma unroll
    for (int j = 0; j < CPT / 4; ++j) {
        const uint32_t a[4] = {lo.pair(2 * j), hi.pair(2 * j), lo.pair(2 * j + 1), hi.pair(2 * j + 1)};
        mma_bf16(c, a, xb[2 * j], xb[2 * j + 1]);
    }
}
template <int CPT>
__device__ __forceinline__ void load_x(const __nv_bfloat16 * x, int c, int t, uint32_t (&xb)[CPT / 2]) {
    if (!x) {
#pragma unroll
        for (int i = 0; i < CPT / 2; ++i) xb[i] = 0u;
        return;
    }
    const uint4 * p = reinterpret_cast<const uint4 *>(x + (4 * c + t) * CPT);
#pragma unroll
    for (int i = 0; i < CPT / 8; ++i) { const uint4 v = __ldg(p + i); xb[4 * i] = v.x; xb[4 * i + 1] = v.y; xb[4 * i + 2] = v.z; xb[4 * i + 3] = v.w; }
}
__device__ __forceinline__ void load_x16(const __nv_bfloat16 * x, int c, int t, uint32_t (&xb)[8]) { load_x<16>(x, c, t, xb); }
template <int FMT>
__device__ __forceinline__ float mma_global_scale(const uint8_t * aux, int cols) {
    if constexpr (FMT == 2) return __ldg(reinterpret_cast<const float *>(aux + cols / 16));
    return 1.0f;
}
// split-K warps per 16-row tile and columns per thread (a chunk = 4 CPT columns), fixed by the shapes: the k order of every sum
constexpr int kMmaGuWarps = 6, kMmaGuCpt = 32, kMmaDnWarps = 3, kMmaDnCpt = 32;
// grid (rows / 16, n * topk), block kMmaGuWarps warps: gate and up tiles of group blockIdx.y, k split over the warps
template <int FMT>
__global__ void __launch_bounds__(32 * kMmaGuWarps) mma_gate_up_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x,
                                                                  const int * ids, const int * groups, __nv_bfloat16 * act, int topk, float gc,
                                                                  float uc) {
    const int gi = blockIdx.y;
    if (groups && gi >= groups[0]) return;
    // groups == nullptr: one token, assignment blockIdx.y alone (the same arithmetic as a group of one)
    const int one[2] = {1, gi};
    const int * gr = groups ? groups + 1 + gi * (1 + kMaxMembers) : one;
    const int nm = gr[0], e = ids[gr[1]], warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int r0 = blockIdx.x * 16 + g, r1 = r0 + 8;
    const uint8_t * gw0 = row_ptr<FMT>(mg, e, r0, cols), * gw1 = row_ptr<FMT>(mg, e, r1, cols);
    const uint8_t * uw0 = row_ptr<FMT>(mu, e, r0, cols), * uw1 = row_ptr<FMT>(mu, e, r1, cols);
    const uint8_t * ga0 = aux_ptr<FMT>(mg, e, r0, cols), * ga1 = aux_ptr<FMT>(mg, e, r1, cols);
    const uint8_t * ua0 = aux_ptr<FMT>(mu, e, r0, cols), * ua1 = aux_ptr<FMT>(mu, e, r1, cols);
    const __nv_bfloat16 * xr = g < nm ? x + size_t(gr[1 + g] / topk) * cols : nullptr;
    constexpr int CPT = kMmaGuCpt;
    const int nch = cols / (4 * CPT), c0 = warp * nch / kMmaGuWarps, c1 = (warp + 1) * nch / kMmaGuWarps;
    float cg[4] = {0.0f, 0.0f, 0.0f, 0.0f}, cu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = c0; c < c1; ++c) {
        MmaRow<FMT, CPT> g0, g1, u0, u1;
        g0.load(gw0, ga0, cols, c, t); g1.load(gw1, ga1, cols, c, t); u0.load(uw0, ua0, cols, c, t); u1.load(uw1, ua1, cols, c, t);
        uint32_t xb[CPT / 2];
        load_x<CPT>(xr, c, t, xb);
        mma_chunk<FMT, CPT>(cg, g0, g1, xb);
        mma_chunk<FMT, CPT>(cu, u0, u1, xb);
    }
    __shared__ float red[kMmaGuWarps][32][8];
#pragma unroll
    for (int i = 0; i < 4; ++i) { red[warp][lane][i] = cg[i]; red[warp][lane][4 + i] = cu[i]; }
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = red[0][lane][i];
#pragma unroll
        for (int w = 1; w < kMmaGuWarps; ++w) v += red[w][lane][i];
        (i < 4 ? cg[i] : cu[i - 4]) = v;
    }
    const float sg[2] = {mma_global_scale<FMT>(ga0, cols), mma_global_scale<FMT>(ga1, cols)};
    const float su[2] = {mma_global_scale<FMT>(ua0, cols), mma_global_scale<FMT>(ua1, cols)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // c[i]: row r0 (i < 2) or r1, column (member) 2t + (i & 1)
        const int m = 2 * t + (i & 1), h = i / 2, row = h ? r1 : r0;
        if (m >= nm || row >= rows) continue;
        const float gv = bfr(cg[i] * sg[h]), uv = bfr(cu[i] * su[h]);
        act[size_t(gr[1 + m]) * rows + row] = __float2bfloat16(gc * tanhf(gv / gc) * (1.0f / (1.0f + expf(-gv))) * (uc * tanhf(uv / uc)));
    }
}
// grid (rows / 16, n * topk), block kMmaDnWarps warps: y[a][row] = bf16(W_down[e][row] . act[a]) for the group's members
template <int FMT>
__global__ void __launch_bounds__(32 * kMmaDnWarps) mma_down_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids,
                                                               const int * groups, __nv_bfloat16 * y) {
    const int gi = blockIdx.y;
    if (gi >= groups[0]) return;
    const int * gr = groups + 1 + gi * (1 + kMaxMembers);
    const int nm = gr[0], e = ids[gr[1]], warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int r0 = blockIdx.x * 16 + g, r1 = r0 + 8;
    const uint8_t * w0 = row_ptr<FMT>(md, e, r0, cols), * w1 = row_ptr<FMT>(md, e, r1, cols);
    const uint8_t * a0 = aux_ptr<FMT>(md, e, r0, cols), * a1 = aux_ptr<FMT>(md, e, r1, cols);
    const __nv_bfloat16 * xr = g < nm ? act + size_t(gr[1 + g]) * cols : nullptr;
    constexpr int CPT = kMmaDnCpt;
    const int nch = cols / (4 * CPT), c0 = warp * nch / kMmaDnWarps, c1 = (warp + 1) * nch / kMmaDnWarps;
    float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 3
    for (int c = c0; c < c1; ++c) {
        MmaRow<FMT, CPT> lo, hi;
        lo.load(w0, a0, cols, c, t); hi.load(w1, a1, cols, c, t);
        uint32_t xb[CPT / 2];
        load_x<CPT>(xr, c, t, xb);
        mma_chunk<FMT, CPT>(cc, lo, hi, xb);
    }
    __shared__ float red[kMmaDnWarps][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = cc[i];
    __syncthreads();
    if (warp) return;
    const float sc[2] = {mma_global_scale<FMT>(a0, cols), mma_global_scale<FMT>(a1, cols)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v = red[0][lane][i];
#pragma unroll
        for (int w = 1; w < kMmaDnWarps; ++w) v += red[w][lane][i];
        const int m = 2 * t + (i & 1), h = i / 2, row = h ? r1 : r0;
        if (m < nm && row < rows) y[size_t(gr[1 + m]) * rows + row] = __float2bfloat16(v * sc[h]);
    }
}

// ---- dense BF16 projections on tensor cores (1 .. 8 tokens): the expert kernels' tiles with BF16 rows as stored; the k
// split and order are fixed by K, so a token's sums do not depend on n (verify rows == single steps bitwise)
struct MmaGemvArgs { GemvSeg seg[kMaxSegs]; int nseg, n, tile0[kMaxSegs + 1]; };
constexpr int kMmaGemvWarps = 8;
__global__ void __launch_bounds__(32 * kMmaGemvWarps) mma_gemv_k(MmaGemvArgs a) {
    int si = 0;
#pragma unroll
    for (int k = 1; k < kMaxSegs; ++k) if (k < a.nseg && int(blockIdx.x) >= a.tile0[k]) si = k;
    const GemvSeg & sg = a.seg[si];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    const int r0 = (int(blockIdx.x) - a.tile0[si]) * 16 + g, r1 = r0 + 8;
    const __nv_bfloat16 * w0 = sg.W + size_t(min(r0, sg.rows - 1)) * sg.K, * w1 = sg.W + size_t(min(r1, sg.rows - 1)) * sg.K;
    const __nv_bfloat16 * xr = g < a.n ? sg.x + size_t(g) * sg.ldx : nullptr;
    const int nch = sg.K / 64, c0 = warp * nch / kMmaGemvWarps, c1 = (warp + 1) * nch / kMmaGemvWarps;
    float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 2
    for (int c = c0; c < c1; ++c) {
        const uint4 * p0 = reinterpret_cast<const uint4 *>(w0 + 64 * c + 16 * t), * p1 = reinterpret_cast<const uint4 *>(w1 + 64 * c + 16 * t);
        const uint4 l0 = ld_stream(p0), l1 = ld_stream(p0 + 1), h0 = ld_stream(p1), h1 = ld_stream(p1 + 1);
        const uint32_t lo[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w}, hi[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        uint32_t xb[8];
        load_x16(xr, c, t, xb);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t af[4] = {lo[2 * j], hi[2 * j], lo[2 * j + 1], hi[2 * j + 1]};
            mma_bf16(cc, af, xb[2 * j], xb[2 * j + 1]);
        }
    }
    __shared__ float red[kMmaGemvWarps][32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = cc[i];
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v = red[0][lane][i];
#pragma unroll
        for (int w = 1; w < kMmaGemvWarps; ++w) v += red[w][lane][i];
        const int tok = 2 * t + (i & 1), row = i < 2 ? r0 : r1;
        if (tok >= a.n || row >= sg.rows) continue;
        const size_t o = size_t(tok) * sg.ldy + row;
        if (sg.f32) static_cast<float *>(sg.y)[o] = v;
        else static_cast<__nv_bfloat16 *>(sg.y)[o] = __float2bfloat16(v);
    }
}

// one token: down tiles of all topk experts in one block (warp = (k, k-split part)), then the combine over k in order:
// the arithmetic of mma_down_k + moe_combine_k, without the groups kernel, the yd round trip and the combine launch
template <int FMT>
__global__ void __launch_bounds__(32 * kMmaDnWarps * 8) mma_down_combine1_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids,
                                                                             const float * wts, __nv_bfloat16 * out, int topk) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4, k = warp / kMmaDnWarps, part = warp % kMmaDnWarps;
    const int r0 = blockIdx.x * 16 + g, r1 = r0 + 8;
    __shared__ float red[8][kMmaDnWarps][32][4];
    __shared__ float scl[8][2][8];
    if (k < topk) {
        const int e = ids[k];
        const uint8_t * w0 = row_ptr<FMT>(md, e, r0, cols), * w1 = row_ptr<FMT>(md, e, r1, cols);
        const uint8_t * a0 = aux_ptr<FMT>(md, e, r0, cols), * a1 = aux_ptr<FMT>(md, e, r1, cols);
        const __nv_bfloat16 * xr = g == 0 ? act + size_t(k) * cols : nullptr;
        constexpr int CPT = kMmaDnCpt;
        const int nch = cols / (4 * CPT), c0 = part * nch / kMmaDnWarps, c1 = (part + 1) * nch / kMmaDnWarps;
        float cc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 3
        for (int c = c0; c < c1; ++c) {
            MmaRow<FMT, CPT> lo, hi;
            lo.load(w0, a0, cols, c, t); hi.load(w1, a1, cols, c, t);
            uint32_t xb[CPT / 2];
            load_x<CPT>(xr, c, t, xb);
            mma_chunk<FMT, CPT>(cc, lo, hi, xb);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) red[k][part][lane][i] = cc[i];
        if (part == 0 && t == 0) { scl[k][0][g] = mma_global_scale<FMT>(a0, cols); scl[k][1][g] = mma_global_scale<FMT>(a1, cols); }
    }
    __syncthreads();
    // column 0 (the token) of c[0] (row r0) and c[2] (row r1) lives in lanes with t == 0: 16 rows, one thread each
    if (threadIdx.x >= 16) return;
    const int gg = threadIdx.x % 8, h = threadIdx.x / 8, row = blockIdx.x * 16 + gg + 8 * h, ln = gg * 4;
    if (row >= rows) return;
    float y = 0.0f;
    for (int kk = 0; kk < topk; ++kk) {
        float v = red[kk][0][ln][2 * h];
#pragma unroll
        for (int w = 1; w < kMmaDnWarps; ++w) v += red[kk][w][ln][2 * h];
        y = fmaf(__bfloat162float(__float2bfloat16(v * scl[kk][h][gg])), wts[kk], y);
    }
    out[row] = __float2bfloat16(y);
}

// one warp per row: max |w| -> scale = max / 127; q = round(w / scale)
__global__ void quantize_rows_i8_k(const __nv_bfloat16 * w, int8_t * q, float * scale, int rows, int K) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const __nv_bfloat16 * wr = w + size_t(r) * K;
    float m = 0.0f;
    for (int c = lane; c < K; c += 32) m = fmaxf(m, fabsf(__bfloat162float(wr[c])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float sc = m > 0.0f ? m / 127.0f : 1.0f;
    for (int c = lane; c < K; c += 32) q[size_t(r) * K + c] = int8_t(__float2int_rn(__bfloat162float(wr[c]) / sc));
    if (lane == 0) scale[r] = sc;
}
__global__ void dequant_rows_i8_k(const int8_t * q, const float * scale, __nv_bfloat16 * w, size_t n, int K) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        w[i] = __float2bfloat16(float(q[i]) * scale[i / K]);
}


__device__ __forceinline__ unsigned ld_acquire(const unsigned * p) {
    unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
// decode / verify depth attention in one launch: grid (kArnSplit, n) blocks of kArnSplitThreads. Each block's split
// statistics, then the token's 8 blocks wait for each other on a counter (8 blocks are always resident together), the
// mix, and block j < (norms) runs stream j's rms_norm (rms_norm_body, 256 threads) once every mix is in. The bodies and
// their order are those of attn_res_stats_k, attn_res_mix_k and rms_norm_k. cnt: 3 zeroed words per token, zero again
// after the launch.
__device__ __forceinline__ void spin_until(const unsigned * p, unsigned v) {
    while (ld_acquire(p) < v) {}
    __threadfence();
}
__global__ void __launch_bounds__(kArnSplitThreads) attn_res_fused_k(States st, const float * q, int nq, float * part, unsigned * cnt, float * out,
                                                                    NormOut no, int n, int D, float eps) {
    __shared__ float sh[kArnNW][kArnNV];
    __shared__ float tot[kArnNV], sp[kMaxStates], nsh[8];
    const Blk b = blk_of_launch();
    const int t = blockIdx.y;
    unsigned * c = cnt + 3 * t;
    attn_res_stats_body(b, st, q, nq, part, D, sh);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(&c[0], 1u); spin_until(&c[0], kArnSplit); }
    __syncthreads();
    attn_res_mix_body(b, st, nq, part, out, n, D, tot, sp);
    int nn = 0, si = -1;
    for (int i = 0; i < nq; ++i) if (no.y[i]) { if (nn == int(blockIdx.x)) si = i; ++nn; }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned d = atomicAdd(&c[1], 1u);
        if (nn == 0 && d == kArnSplit - 1) { c[0] = 0u; c[1] = 0u; }
        if (si >= 0) spin_until(&c[1], kArnSplit);
    }
    if (si < 0) return;
    __syncthreads();
    if (threadIdx.x < 256)
        rms_norm_body(Blk{t, 0, 0, int(threadIdx.x), 256, 1}, out + size_t(si) * n * D, no.w[si], nullptr, no.y[si], D, eps, nsh);
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(&c[2], 1u) == unsigned(nn - 1)) { c[0] = 0u; c[1] = 0u; c[2] = 0u; }
}

int grid_of(size_t n) { return int(std::min<size_t>((n + 255) / 256, 4096)); }
}  // namespace

void attn_res(const States & st, const float * q, int nq, float * out, int n, int D, float * scratch, cudaStream_t s) {
    attn_res_norm(st, q, nq, out, nullptr, nullptr, n, D, 0.0f, scratch, s);
}
void rms_norm(const float * x, const float * w, float * y32, __nv_bfloat16 * y16, int n, int D, float eps, cudaStream_t s) {
    if (n <= 0) return;
    rms_norm_k<<<n, 256, 0, s>>>(x, w, y32, y16, D, eps);
}
void to_bf16(const float * x, __nv_bfloat16 * y, size_t n, cudaStream_t s) {
    if (n) to_bf16_k<<<grid_of(n), 256, 0, s>>>(x, y, n); }
void to_f32(const __nv_bfloat16 * x, float * y, size_t n, cudaStream_t s) {
    if (n) to_f32_k<<<grid_of(n), 256, 0, s>>>(x, y, n); }
void embed(const __nv_bfloat16 * table, const int * ids, const float * w, float * y, int n, int D, float eps, cudaStream_t s) {
    if (n <= 0) return;
    embed_k<<<n, 256, 0, s>>>(table, ids, w, y, D, eps);
}
void conv_static(const __nv_bfloat16 * x, const float * w, float * state, float * y32, __nv_bfloat16 * y16, int n, int C, cudaStream_t s,
                 const float * add, bool round_sum, __nv_bfloat16 * hist) {
    if (n <= 0) return;
    if (n <= kSmallN) { conv_static_k<<<(C + 255) / 256, 256, 0, s>>>(x, w, state, y32, y16, n, C, add, round_sum, hist); return; }
    if (hist) throw std::runtime_error("inkling2 conv: input history only for few-token calls");
    conv_static_par_k<<<dim3((C + 255) / 256, n), 256, 0, s>>>(x, w, state, y32, y16, n, C, add, round_sum);
    conv_state_k<<<(C + 255) / 256, 256, 0, s>>>(x, state, n, C);
}
void conv_dynamic(const __nv_bfloat16 * x, const float * stat, const __nv_bfloat16 * dyn, float * state, __nv_bfloat16 * y16, int n, int C, cudaStream_t s,
                  __nv_bfloat16 * hist) {
    if (n <= 0) return;
    if (n <= kSmallN) { conv_dynamic_k<<<(C + 255) / 256, 256, 0, s>>>(x, stat, dyn, state, y16, n, C, hist); return; }
    if (hist) throw std::runtime_error("inkling2 conv: input history only for few-token calls");
    conv_dynamic_par_k<<<dim3((C + 255) / 256, n), 256, 0, s>>>(x, stat, dyn, state, y16, n, C);
    conv_state_k<<<(C + 255) / 256, 256, 0, s>>>(x, state, n, C);
}
void msg_gate_in(const float * h, const float * m, __nv_bfloat16 * z, int n, int D, float eps, cudaStream_t s) {
    if (D > kMgiThreads * kMgiPer) throw std::runtime_error("inkling2 msg_gate_in: D too large");
    if (n <= 0) return;
    msg_gate_in_k<<<n, kMgiThreads, 0, s>>>(h, m, z, D, eps);
}
void msg_gate_out(const __nv_bfloat16 * g, const __nv_bfloat16 * FT, const __nv_bfloat16 * GT, const float * h, const float * m, float * y, int n, int D,
                  int rank, cudaStream_t s) {
    if (n <= 0) return;
    if (n >= kMgoTok)
        msg_gate_out_tile_k<<<dim3((D + kMgoChan - 1) / kMgoChan, (n + kMgoTok - 1) / kMgoTok), kMgoChan, kMgoTok * rank * int(sizeof(float)), s>>>(
            g, FT, GT, h, m, y, n, D, rank);
    else
        msg_gate_out_k<<<dim3((D + 255) / 256, n), 256, rank * sizeof(float), s>>>(g, FT, GT, h, m, y, D, rank);
}
__global__ void msg_gate_combine_k(const float * fa, const float * h, const float * m, float * y, int n, int D) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(n) * D) return;
    const size_t t = i / D, c = i % D;
    y[i] = (1.0f - tanhf(fa[t * 2 * D + c])) * h[i] + tanhf(fa[t * 2 * D + D + c]) * m[i];
}
void msg_gate_combine(const float * fa, const float * h, const float * m, float * y, int n, int D, cudaStream_t s) {
    if (n > 0) msg_gate_combine_k<<<grid_of(size_t(n) * D), 256, 0, s>>>(fa, h, m, y, n, D);
}
void add(const float * a, const float * b, float * y, size_t n, bool round_bf16, cudaStream_t s) {
    if (n) add_k<<<grid_of(n), 256, 0, s>>>(a, b, y, n, round_bf16); }
void moe_combine(const __nv_bfloat16 * y, const float * wts, __nv_bfloat16 * out, int n, int D, int topk, cudaStream_t s) {
    if (n > 0) moe_combine_k<<<dim3((D + 255) / 256, n), 256, 0, s>>>(y, wts, out, D, topk);
}
void moe_combine_grouped(const __nv_bfloat16 * yg, const int * order, int A, const float * wts, __nv_bfloat16 * out, int n, int D, int topk, int * inv,
                         cudaStream_t s) {
    if (n <= 0) return;
    inv_order_k<<<(A + 255) / 256, 256, 0, s>>>(order, A, inv);
    moe_combine_grouped_k<<<dim3((D + 255) / 256, n), 256, 0, s>>>(yg, inv, wts, out, D, topk);
}
void softcap(float * x, size_t n, float c, cudaStream_t s) {
    if (n) softcap_k<<<grid_of(n), 256, 0, s>>>(x, n, c); }

void attn_res_norm(const States & st, const float * queries, int nq, float * out, const float * const * w, __nv_bfloat16 * const * y16, int n, int D,
                   float eps, float * scratch, cudaStream_t s) {
    if (n <= 0) return;
    if (D > kArnThreads * kArnPer || nq > 4) throw std::runtime_error("inkling2 attn_res_norm: D too large or nq > 4");
    NormOut no{};
    for (int i = 0; i < nq; ++i) { no.w[i] = w ? w[i] : nullptr; no.y[i] = y16 ? y16[i] : nullptr; }
    if (n <= kSmallN) {  // decode / verify block: split D over blocks, then the per-stream norms from the F32 out
        if (!scratch) throw std::runtime_error("inkling2 attn_res_norm: null scratch");
        // counters first at a fixed place (3 per token slot, whatever n), then the partials, then the attention result
        unsigned * cnt = reinterpret_cast<unsigned *>(scratch);
        float * part = scratch + 3 * kSmallN;
        bool need_norm = false;
        for (int i = 0; i < nq; ++i) need_norm = need_norm || no.y[i];
        // with no out buffer and norms requested, the attention result goes to the tail of the scratch
        float * dst = out ? out : (need_norm ? part + size_t(n) * kArnSplit * kMaxStates * 5 : nullptr);
        if (need_norm && !dst) throw std::runtime_error("inkling2 attn_res_norm: null out");
        // one launch: statistics, mix, norms
        for (int i = 0; i < nq; ++i) if (no.y[i] && !no.w[i]) throw std::runtime_error("inkling2 attn_res_norm: norm weight without w");
        attn_res_fused_k<<<dim3(kArnSplit, n), kArnSplitThreads, 0, s>>>(st, queries, nq, part, cnt, dst, no, n, D, eps);
        return;
    }
    switch (nq) {
        case 1: attn_res_norm_k<1><<<n, kArnThreads, 0, s>>>(st, queries, nq, out, no, n, D, eps); break;
        case 2: attn_res_norm_k<2><<<n, kArnThreads, 0, s>>>(st, queries, nq, out, no, n, D, eps); break;
        case 3: attn_res_norm_k<3><<<n, kArnThreads, 0, s>>>(st, queries, nq, out, no, n, D, eps); break;
        default: attn_res_norm_k<4><<<n, kArnThreads, 0, s>>>(st, queries, nq, out, no, n, D, eps); break;
    }
}

// split path: [n][kArnSplit][kMaxStates * 5] partials + n counters + up to 4 * n * D floats of attention output
int attn_res_norm_scratch_words(int n, int D) {
    return n > 0 && n <= kSmallN ? 3 * kSmallN + n * kArnSplit * kMaxStates * 5 + 4 * n * D : 0;
}

void gemv_impl(const GemvSeg * segs, int nseg, int n, cudaStream_t s, bool allow_mma);
namespace { bool g_gemv_mma = true; }
void set_gemv_mma(bool on) { g_gemv_mma = on; }
void gemv(const GemvSeg * segs, int nseg, int n, cudaStream_t s) { gemv_impl(segs, nseg, n, s, g_gemv_mma); }
void gemv_rows(const GemvSeg * segs, int nseg, int n, cudaStream_t s) { gemv_impl(segs, nseg, n, s, false); }
void gemv_impl(const GemvSeg * segs, int nseg, int n, cudaStream_t s, bool allow_mma) {
    if (n <= 0 || nseg <= 0) return;
    if (n > kSmallN || nseg > kMaxSegs) throw std::runtime_error("inkling2 gemv: too many tokens / segments");
    // BF16 rows: tensor cores (gemv_rows: the LM head, whose bounded greedy rescoring sums like gemv_k<1>)
    bool mma = allow_mma;
    for (int i = 0; i < nseg; ++i)
        mma = mma && !segs[i].scale && segs[i].K % 64 == 0 && segs[i].ldx % 8 == 0 &&
              reinterpret_cast<uintptr_t>(segs[i].W) % 16 == 0 && reinterpret_cast<uintptr_t>(segs[i].x) % 16 == 0;
    if (mma) {
        MmaGemvArgs m{};
        m.nseg = nseg; m.n = n;
        for (int i = 0; i < nseg; ++i) { m.seg[i] = segs[i]; m.tile0[i + 1] = m.tile0[i] + (segs[i].rows + 15) / 16; }
        mma_gemv_k<<<m.tile0[nseg], 32 * kMmaGemvWarps, 0, s>>>(m);
        return;
    }
    GemvArgs a{};
    a.nseg = nseg;
    for (int i = 0; i < nseg; ++i) {
        if (segs[i].K % (segs[i].scale ? 16 : 8)) throw std::runtime_error("inkling2 gemv: K % 8 (INT8: % 16)");
        a.seg[i] = segs[i];
        a.warp0[i + 1] = a.warp0[i] + (segs[i].rows + 1) / 2;
    }
    const int blocks = (a.warp0[nseg] + 7) / 8;
    switch (n) {
        case 1: gemv_k<1><<<blocks, 256, 0, s>>>(a); break;
        case 2: gemv_k<2><<<blocks, 256, 0, s>>>(a); break;
        case 3: gemv_k<3><<<blocks, 256, 0, s>>>(a); break;
        case 4: gemv_k<4><<<blocks, 256, 0, s>>>(a); break;
        case 5: gemv_k<5><<<blocks, 256, 0, s>>>(a); break;
        case 6: gemv_k<6><<<blocks, 256, 0, s>>>(a); break;
        case 7: gemv_k<7><<<blocks, 256, 0, s>>>(a); break;
        default: gemv_k<8><<<blocks, 256, 0, s>>>(a); break;
    }
}
template <int FMT, int NC>
void gate_up_launch(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const __nv_bfloat16 * x, const int * ids, __nv_bfloat16 * act, int n,
                    int topk, float gc, float uc, cudaStream_t s) {
    if constexpr (FMT == 2 && NC > 0) {
        constexpr int R = 2;
        expert_gate_up_nvf4_k<NC, R><<<dim3((rows + 8 * R - 1) / (8 * R), n * topk), 256, 0, s>>>(g, u, rows, cols, x, ids, act, topk, gc, uc);
    } else {
        expert_gate_up_k<FMT, NC><<<dim3((rows + 7) / 8, n * topk), 256, 0, s>>>(g, u, rows, cols, x, ids, act, topk, gc, uc);
    }
}
template <int FMT, int NC>
void down_launch(const ExpertMat & d, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const float * wts, __nv_bfloat16 * out, int n,
                 int topk, cudaStream_t s) {
    if constexpr (FMT == 2 && NC > 0) {
        constexpr int R = 2;
        expert_down_nvf4_k<NC, R><<<dim3((rows + R - 1) / R, n), topk * 32, 0, s>>>(d, rows, cols, act, ids, wts, out, topk);
    } else {
        expert_down_k<FMT, NC><<<dim3((rows + 7) / 8, n), 256, 0, s>>>(d, rows, cols, act, ids, wts, out, topk);
    }
}
void expert_gate_up(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const __nv_bfloat16 * x, const int * ids, __nv_bfloat16 * act, int n,
                    int topk, float gc, float uc, cudaStream_t s, const __half * xh) {
    if (n <= 0) return;
    if (cols % 128 || g.fmt != u.fmt) throw std::runtime_error("inkling2 experts: cols % 128 or mixed formats");
    if (g.fmt >= 3) {
        if (!xh) throw std::runtime_error("inkling2 experts: TQT needs the rotated activation");
        if (g.fmt == 5) expert_gate_up_tqt_k<5><<<dim3((rows + 7) / 8, n * topk), 256, 0, s>>>(g, u, rows, cols, xh, ids, act, topk, gc, uc);
        else if (g.fmt == 4) expert_gate_up_tqt_k<4><<<dim3((rows + 7) / 8, n * topk), 256, 0, s>>>(g, u, rows, cols, xh, ids, act, topk, gc, uc);
        else expert_gate_up_tqt_k<3><<<dim3((rows + 7) / 8, n * topk), 256, 0, s>>>(g, u, rows, cols, xh, ids, act, topk, gc, uc);
    } else if (g.fmt == 2) {
        if (cols == 2304) gate_up_launch<2, 72>(g, u, rows, cols, x, ids, act, n, topk, gc, uc, s);
        else gate_up_launch<2, 0>(g, u, rows, cols, x, ids, act, n, topk, gc, uc, s);
    } else if (g.fmt == 1) {
        if (cols == 2304) gate_up_launch<1, 18>(g, u, rows, cols, x, ids, act, n, topk, gc, uc, s);
        else gate_up_launch<1, 0>(g, u, rows, cols, x, ids, act, n, topk, gc, uc, s);
    } else {
        throw std::runtime_error("inkling2 experts: format");
    }
}
void expert_down(const ExpertMat & d, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const float * wts, __nv_bfloat16 * out, int n,
                 int topk, cudaStream_t s, const __half * acth) {
    if (n <= 0) return;
    if (topk > 16 || cols % 128) throw std::runtime_error("inkling2 expert_down: topk > 16 or cols % 128");
    if (d.fmt >= 3) {
        if (!acth) throw std::runtime_error("inkling2 experts: TQT needs the rotated activation");
        if (d.fmt == 5) expert_down_tqt_k<5><<<dim3((rows + 7) / 8, n), 256, 0, s>>>(d, rows, cols, acth, ids, wts, out, topk);
        else if (d.fmt == 4) expert_down_tqt_k<4><<<dim3((rows + 7) / 8, n), 256, 0, s>>>(d, rows, cols, acth, ids, wts, out, topk);
        else expert_down_tqt_k<3><<<dim3((rows + 7) / 8, n), 256, 0, s>>>(d, rows, cols, acth, ids, wts, out, topk);
    } else if (d.fmt == 2) {
        if (cols == 1152) down_launch<2, 36>(d, rows, cols, act, ids, wts, out, n, topk, s);
        else down_launch<2, 0>(d, rows, cols, act, ids, wts, out, n, topk, s);
    } else if (d.fmt == 1) {
        if (cols == 1152) down_launch<1, 9>(d, rows, cols, act, ids, wts, out, n, topk, s);
        else down_launch<1, 0>(d, rows, cols, act, ids, wts, out, n, topk, s);
    } else {
        throw std::runtime_error("inkling2 experts: format");
    }
}
// state = the last 3 of (snap's 3 inputs, then hist[0 .. keep - 1]), per channel
__global__ void conv_rollback_k(const float * snap, const __nv_bfloat16 * hist, float * state, int keep, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int j = 0; j < 3; ++j) {
        const int i = keep - 3 + j;  // index into hist; negative: into the snapshot
        v[j] = i >= 0 ? __bfloat162float(hist[size_t(i) * C + c]) : snap[c * 3 + 3 + i];
    }
    state[c * 3] = v[0]; state[c * 3 + 1] = v[1]; state[c * 3 + 2] = v[2];
}
void conv_rollback(const float * snap, const __nv_bfloat16 * hist, float * state, int keep, int C, cudaStream_t s) {
    conv_rollback_k<<<(C + 255) / 256, 256, 0, s>>>(snap, hist, state, keep, C);
}
__global__ void count_routes_k(const int * ids, int A, int * counts) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < A && ids[a] >= 0) atomicAdd(counts + ids[a], 1);
}
void count_routes(const int * ids, int A, int * counts, cudaStream_t s) {
    if (A > 0) count_routes_k<<<(A + 255) / 256, 256, 0, s>>>(ids, A, counts);
}
void expert_dedup(const ExpertMat & g, const ExpertMat & u, const ExpertMat & d, int I, int D, const __nv_bfloat16 * x, const int * ids, const float * wts,
                  int * groups, __nv_bfloat16 * act, __nv_bfloat16 * yd, __nv_bfloat16 * out, int n, int topk, float gc, float uc, cudaStream_t s) {
    const int A = n * topk;
    if (g.fmt >= 3) throw std::runtime_error("inkling2 expert_dedup: TQT experts take the per-assignment kernels");
    if (A > kMaxAssign || n > kMaxMembers || g.fmt != d.fmt) throw std::runtime_error("inkling2 expert_dedup: block too wide");
    dedup_groups_k<<<1, kMaxAssign, 0, s>>>(ids, A, groups);
    if (g.fmt == 2 && D == 72 * 32 && I == 36 * 32) {  // Inkling-2-Mini's shapes: the one-token kernels' arithmetic, weights read once per group
        constexpr int RG = 1, RD = 1;  // rows per warp: the member accumulators stay within ~100 registers (2 blocks / SM)
        dedup_gate_up_nvf4_k<72, RG><<<dim3((I + 8 * RG - 1) / (8 * RG), A), 256, 0, s>>>(g, u, I, D, x, ids, groups, act, topk, gc, uc);
        dedup_down_nvf4_k<36, RD><<<dim3((D + 8 * RD - 1) / (8 * RD), A), 256, 0, s>>>(d, D, I, act, ids, groups, yd);
    } else if (g.fmt == 2) {
        dedup_gate_up_k<2><<<dim3((I + 7) / 8, A), 256, 0, s>>>(g, u, I, D, x, ids, groups, act, topk, gc, uc);
        dedup_down_k<2><<<dim3((D + 7) / 8, A), 256, 0, s>>>(d, D, I, act, ids, groups, yd);
    } else {
        dedup_gate_up_k<1><<<dim3((I + 7) / 8, A), 256, 0, s>>>(g, u, I, D, x, ids, groups, act, topk, gc, uc);
        dedup_down_k<1><<<dim3((D + 7) / 8, A), 256, 0, s>>>(d, D, I, act, ids, groups, yd);
    }
    moe_combine(yd, wts, out, n, D, topk, s);
}

bool expert_mma_ok(const ExpertMat & g, const ExpertMat & d, int I, int D) {
    const bool tqt = g.fmt >= 3 && g.fmt <= 5;  // one 32-column segment per thread, a chunk = one 128-column scale block
    return (g.fmt == 1 || g.fmt == 2 || (tqt && kMmaGuCpt == 32 && kMmaDnCpt == 32)) && d.fmt == g.fmt && D % (4 * kMmaGuCpt) == 0 &&
           I % (4 * kMmaDnCpt) == 0 && I % 16 == 0 && D % 16 == 0;
}
void expert_mma_groups(const int * ids, int A, int * groups, cudaStream_t s) {
    if (A > kMaxAssign) throw std::runtime_error("inkling2 expert_mma: block too wide");
    dedup_groups_k<<<1, kMaxAssign, 0, s>>>(ids, A, groups);
}
template <class F> void by_fmt(int fmt, F && f) {
    switch (fmt) {
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        default: throw std::runtime_error("inkling2 expert_mma: format");
    }
}
// x / act: BF16, or for TQT (fmt >= 3) the rotated FP16 copies (same element size)
void expert_mma_gate_up(const ExpertMat & g, const ExpertMat & u, int I, int D, const __nv_bfloat16 * x, const int * ids, const int * groups,
                        __nv_bfloat16 * act, int A, int topk, float gc, float uc, cudaStream_t s) {
    const dim3 grid(I / 16, A);
    by_fmt(g.fmt, [&](auto F) { mma_gate_up_k<decltype(F)::value><<<grid, 32 * kMmaGuWarps, 0, s>>>(g, u, I, D, x, ids, groups, act, topk, gc, uc); });
}
void expert_mma_down(const ExpertMat & d, int D, int I, const __nv_bfloat16 * act, const int * ids, const int * groups, __nv_bfloat16 * yd, int A,
                     cudaStream_t s) {
    const dim3 grid(D / 16, A);
    by_fmt(d.fmt, [&](auto F) { mma_down_k<decltype(F)::value><<<grid, 32 * kMmaDnWarps, 0, s>>>(d, D, I, act, ids, groups, yd); });
}
void expert_mma(const ExpertMat & g, const ExpertMat & u, const ExpertMat & d, int I, int D, const __nv_bfloat16 * x, const int * ids, const float * wts,
                int * groups, __nv_bfloat16 * act, __nv_bfloat16 * yd, __nv_bfloat16 * out, int n, int topk, float gc, float uc, cudaStream_t s,
                __half * xh, __half * acth) {
    if (!expert_mma_ok(g, d, I, D) || n > kMaxMembers) throw std::runtime_error("inkling2 expert_mma: format / shape / block");
    const bool tqt = g.fmt >= 3;
    if (tqt && (!xh || !acth)) throw std::runtime_error("inkling2 expert_mma: TQT needs the rotated activation scratch");
    const int A = n * topk;
    const __nv_bfloat16 * xin = x, * ain = act;
    if (tqt) {  // TQT weights live in TQ4's rotated basis: the activations rotate per 128 (FP16)
        rotate_h(x, xh, D, n, s);
        xin = reinterpret_cast<const __nv_bfloat16 *>(xh);
        ain = reinterpret_cast<const __nv_bfloat16 *>(acth);
    }
    if (n == 1) {  // the same bits without the groups kernel, the yd round trip and the combine launch
        if (topk > 8) throw std::runtime_error("inkling2 expert_mma: topk > 8");
        expert_mma_gate_up(g, u, I, D, xin, ids, nullptr, act, A, topk, gc, uc, s);
        if (tqt) rotate_h(act, acth, I, A, s);
        by_fmt(d.fmt, [&](auto F) { mma_down_combine1_k<decltype(F)::value><<<D / 16, 32 * kMmaDnWarps * 8, 0, s>>>(d, D, I, ain, ids, wts, out, topk); });
        return;
    }
    expert_mma_groups(ids, A, groups, s);
    expert_mma_gate_up(g, u, I, D, xin, ids, groups, act, A, topk, gc, uc, s);
    if (tqt) rotate_h(act, acth, I, A, s);
    expert_mma_down(d, D, I, ain, ids, groups, yd, A, s);
    moe_combine(yd, wts, out, n, D, topk, s);
}

void expert_dedup_tqt(const ExpertMat & g, const ExpertMat & u, const ExpertMat & d, int I, int D, const __half * xh, const int * ids, const float * wts,
                      int * groups, __nv_bfloat16 * act, __half * acth, __nv_bfloat16 * yd, __nv_bfloat16 * out, int n, int topk, float gc, float uc,
                      cudaStream_t s) {
    const int A = n * topk;
    if (g.fmt < 3 || g.fmt != u.fmt) throw std::runtime_error("inkling2 expert_dedup_tqt: format");
    if (A > kMaxAssign || n > kMaxMembers) throw std::runtime_error("inkling2 expert_dedup_tqt: block too wide");
    if (D % 128 || I % 128) throw std::runtime_error("inkling2 expert_dedup_tqt: cols % 128");
    dedup_groups_k<<<1, kMaxAssign, 0, s>>>(ids, A, groups);
    const dim3 gg((I + 7) / 8, A);
    if (g.fmt == 5) dedup_gate_up_tqt_k<5><<<gg, 256, 0, s>>>(g, u, I, D, xh, ids, groups, act, topk, gc, uc);
    else if (g.fmt == 4) dedup_gate_up_tqt_k<4><<<gg, 256, 0, s>>>(g, u, I, D, xh, ids, groups, act, topk, gc, uc);
    else dedup_gate_up_tqt_k<3><<<gg, 256, 0, s>>>(g, u, I, D, xh, ids, groups, act, topk, gc, uc);
    rotate_h(act, acth, I, A, s);
    const dim3 gd((D + 7) / 8, A);
    if (d.fmt == 5) dedup_down_tqt_k<5><<<gd, 256, 0, s>>>(d, D, I, acth, ids, groups, yd);
    else if (d.fmt == 4) dedup_down_tqt_k<4><<<gd, 256, 0, s>>>(d, D, I, acth, ids, groups, yd);
    else if (d.fmt == 3) dedup_down_tqt_k<3><<<gd, 256, 0, s>>>(d, D, I, acth, ids, groups, yd);
    else throw std::runtime_error("inkling2 expert_dedup_tqt: down format");
    moe_combine(yd, wts, out, n, D, topk, s);
}
void quantize_rows_i8(const __nv_bfloat16 * w, int8_t * q, float * scale, int rows, int K, cudaStream_t s) {
    if (rows > 0) quantize_rows_i8_k<<<(rows + 7) / 8, 256, 0, s>>>(w, q, scale, rows, K);
}
void dequant_rows_i8(const int8_t * q, const float * scale, __nv_bfloat16 * w, int rows, int K, cudaStream_t s) {
    const size_t n = size_t(rows) * K;
    if (n) dequant_rows_i8_k<<<grid_of(n), 256, 0, s>>>(q, scale, w, n, K);
}
void rotate_h(const __nv_bfloat16 * x, __half * xh, int k, int ncols, cudaStream_t s) {
    if (ncols > 0 && k > 0) rotate_h_k<<<dim3((k / 128 + 7) / 8, ncols), 256, 0, s>>>(x, xh, k);
}
void route_topk(const float * logits, const float * bias, int * ids, float * wts, int n, int E, int topk, float scale, cudaStream_t s, int * counts) {
    if (n <= 0) return;
    if (E > 256 || topk > 32) throw std::runtime_error("inkling2 route_topk: E > 256 or topk > 32");
    route_topk_k<<<n, 32, 0, s>>>(logits, bias, ids, wts, E, topk, scale, counts);
}
// one block of 128 per KV head: its k and v channels' convolutions over the n tokens, then kv_finish of each token
__global__ void __launch_bounds__(128) conv_kv_k(ConvKvA a, int n) {
    __shared__ float sm[8];
    const Blk b = blk_of_launch();
    for (int s2 = 0; s2 < 2; ++s2) {
        if (a.dyn[s2]) { const auto & d = a.d[s2]; conv_dynamic_body(b, d.x, d.stat, d.dyn, d.state, d.y16, d.n, d.C, d.hist); }
        else { const auto & c = a.s[s2]; conv_static_body(b, c.x, c.w, c.state, c.y32, c.y16, c.n, c.C, c.add, c.round_sum != 0, c.hist); }
    }
    const auto & k = a.kv;
    for (int t = 0; t < n; ++t) {
        kv_finish_body(Blk{t * k.kvheads + b.x, 0, 0, b.t, 128, 0}, k.k, k.v, k.kc, k.vc, k.kvheads, k.clip, k.eps, k.dpos, k.ring, sm);
        __syncthreads();  // sm is reused by the next token
    }
}
void conv_kv_finish(const ConvKvA & a, int n, cudaStream_t s) {
    if (n <= 0) return;
    if (n > kSmallN || a.s[0].C != a.kv.kvheads * 128) throw std::runtime_error("inkling2 conv_kv_finish: shape");
    conv_kv_k<<<a.kv.kvheads, 128, 0, s>>>(a, n);
}

void head_bound_build(const __nv_bfloat16 * w, const int8_t * q, const float * scale, float * bound, int rows, int K, cudaStream_t s) {
    if (rows > 0) head_bound_k<<<(rows + 7) / 8, 256, 0, s>>>(w, q, scale, bound, rows, K);
}
void head_greedy(const HeadBound & h, const __nv_bfloat16 * x, int n, float cap, float * approx, unsigned long long * cells, int * token, int * token2,
                 cudaStream_t s) {
    if (h.K % 16 || h.V > h.rows || n < 1 || n > kSmallN) throw std::runtime_error("inkling2 head_greedy: shape");
    const GemvSeg sg{reinterpret_cast<const __nv_bfloat16 *>(h.q), x, approx, h.V, h.K, h.K, h.V, 1, h.scale};
    gemv(&sg, 1, n, s);  // the INT8 copy read once for the n rows
    for (int i = 0; i < n; ++i) {
        argmax_cell_k<<<160, 256, 0, s>>>(approx + size_t(i) * h.V, size_t(h.V), cells);
        head_select_k<<<(h.V + 7) / 8, 256, 0, s>>>(h.w, approx + size_t(i) * h.V, h.bound, x + size_t(i) * h.K, h.V, h.K, cap, cells);
        argmax_take_k<<<1, 1, 0, s>>>(cells, 1, token + i, token2 ? token2 + i : nullptr);
    }
}
void argmax_greedy(const float * x, size_t n, unsigned long long * cells, int * token, int * token2, cudaStream_t s) {
    argmax_cell_k<<<160, 256, 0, s>>>(x, n, cells);
    argmax_take_k<<<1, 1, 0, s>>>(cells, 0, token, token2);
}
}  // namespace tqllm_cuda::inkling2

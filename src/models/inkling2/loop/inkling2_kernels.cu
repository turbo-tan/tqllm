// Inkling-2-Mini small kernels (see inkling2_kernels.cuh). Correctness first: simple one-block-per-token shapes.
#include "inkling2_kernels.cuh"
#include "tqt_decode.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <cmath>
#include <stdexcept>

namespace tqllm_cuda::inkling2 {
namespace {

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// sum over a block of 256 threads (8 warps); every thread gets the result
__device__ float block_sum(float v, float * sh) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < int(blockDim.x / 32); ++w) t += sh[w];
    return t;
}

__global__ void rms_norm_k(const float * x, const float * w, float * y32, __nv_bfloat16 * y16, int D, float eps) {
    const size_t o = size_t(blockIdx.x) * D;
    __shared__ float sh[8];
    float ss = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) ss += x[o + c] * x[o + c];
    const float r = rsqrtf(block_sum(ss, sh) / float(D) + eps);
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const __nv_bfloat16 v = __float2bfloat16(w[c] * (x[o + c] * r));
        if (y32) y32[o + c] = __bfloat162float(v);
        if (y16) y16[o + c] = v;
    }
}

__global__ void embed_k(const __nv_bfloat16 * table, const int * ids, const float * w, float * y, int D, float eps) {
    const __nv_bfloat16 * row = table + size_t(ids[blockIdx.x]) * D;
    __shared__ float sh[8];
    float ss = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) { const float v = __bfloat162float(row[c]); ss += v * v; }
    const float r = rsqrtf(block_sum(ss, sh) / float(D) + eps);
    for (int c = threadIdx.x; c < D; c += blockDim.x) y[size_t(blockIdx.x) * D + c] = bfr(w[c] * (__bfloat162float(row[c]) * r));
}

__global__ void to_bf16_k(const float * x, __nv_bfloat16 * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] = __float2bfloat16(x[i]);
}
__global__ void to_f32_k(const __nv_bfloat16 * x, float * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) y[i] = __bfloat162float(x[i]);
}

__global__ void conv_static_k(const __nv_bfloat16 * x, const float * w, float * state, float * y32, __nv_bfloat16 * y16, int n, int C,
                              const float * add, bool round_sum, __nv_bfloat16 * hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void conv_dynamic_k(const __nv_bfloat16 * x, const float * stat, const __nv_bfloat16 * dyn, float * state, __nv_bfloat16 * y16, int n, int C,
                               __nv_bfloat16 * hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
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

constexpr int kMgiThreads = 256, kMgiPer = 10;  // D <= 2560
__global__ void __launch_bounds__(kMgiThreads) msg_gate_in_k(const float * h, const float * m, __nv_bfloat16 * z, int D, float eps) {
    const size_t o = size_t(blockIdx.x) * D;
    __shared__ float sh[8];
    float hh[kMgiPer], mm[kMgiPer];
    float a = 0.0f, b = 0.0f;
#pragma unroll
    for (int u = 0; u < kMgiPer; ++u) {
        const int c = threadIdx.x + u * kMgiThreads;
        hh[u] = c < D ? h[o + c] : 0.0f;
        mm[u] = c < D ? m[o + c] : 0.0f;
        a = fmaf(hh[u], hh[u], a);
        b = fmaf(mm[u], mm[u], b);
    }
    const float ra = rsqrtf(block_sum(a, sh) / float(D) + eps);
    const float rb = rsqrtf(block_sum(b, sh) / float(D) + eps);
#pragma unroll
    for (int u = 0; u < kMgiPer; ++u) {
        const int c = threadIdx.x + u * kMgiThreads;
        if (c >= D) break;
        z[2 * o + c] = __float2bfloat16(hh[u] * ra);
        z[2 * o + D + c] = __float2bfloat16(mm[u] * rb);
    }
}

__global__ void msg_gate_out_k(const __nv_bfloat16 * g, const __nv_bfloat16 * FT, const __nv_bfloat16 * GT, const float * h, const float * m, float * y,
                               int D, int rank) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float gs[];
    for (int r = threadIdx.x; r < rank; r += blockDim.x) gs[r] = __bfloat162float(g[size_t(t) * rank + r]);
    __syncthreads();
    if (c >= D) return;
    float f = 0.0f, a = 0.0f;
    for (int r = 0; r < rank; ++r) {
        f = fmaf(__bfloat162float(FT[size_t(r) * D + c]), gs[r], f);
        a = fmaf(__bfloat162float(GT[size_t(r) * D + c]), gs[r], a);
    }
    const size_t o = size_t(t) * D + c;
    y[o] = (1.0f - tanhf(f)) * h[o] + tanhf(a) * m[o];
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

__global__ void softcap_k(float * x, size_t n, float c) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) x[i] = c * tanhf(x[i] / c);
}

// ---- few-token path ----
struct GemvArgs { GemvSeg seg[kMaxSegs]; int warp0[kMaxSegs + 1]; int nseg; };

__device__ __forceinline__ void unpack8(const uint4 & u, float * f) {
    const __nv_bfloat162 * b = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 v = __bfloat1622float2(b[j]); f[2 * j] = v.x; f[2 * j + 1] = v.y; }
}
__device__ __forceinline__ uint4 ld_stream(const uint4 * p) { return __ldcs(p); }

// warp = two consecutive rows of one segment; lane = every 32nd group of 8 columns; per token one F32 sum per row
template <int N>
__global__ void __launch_bounds__(256) gemv_k(GemvArgs a) {
    const int w = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (w >= a.warp0[a.nseg]) return;
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
                unpack8(__ldg(xp), xv); unpack8(__ldg(xp + 1), xv + 8);
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
            unpack8(__ldg(reinterpret_cast<const uint4 *>(sg.x + size_t(t) * sg.ldx) + c), xv);
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
template <int FMT, int NC>
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
                    unpack8(__ldg(reinterpret_cast<const uint4 *>(x + c * 32) + w), xv);
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
template <int NC, int R>
__device__ __forceinline__ void rows_dot_nvf4(const uint8_t * const * row, const uint8_t * const * aux, int cols, const __nv_bfloat16 * x,
                                              int lane, const DecodeLut * lut, float * acc) {
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
            float xv[8];
            unpack8(__ldg(reinterpret_cast<const uint4 *>(x + c * 32) + w), xv);
#pragma unroll
            for (int r = 0; r < R; ++r) {
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
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = fmaf(part[r], e4m3(uint8_t(s2[r] >> 8)) * g[r], acc[r]);
    }
}

// a row's data / aux pointers inside expert e
template <int FMT>
__device__ __forceinline__ const uint8_t * row_ptr(const ExpertMat & m, int e, int row, int cols) {
    return (m.wp ? m.wp[e] : m.w + size_t(e) * m.stride) + size_t(row) * (FMT >= 2 ? size_t(cols / 2) : size_t(cols + cols / 32));
}
template <int FMT>
__device__ __forceinline__ const uint8_t * aux_ptr(const ExpertMat & m, int e, int row, int cols) {
    if constexpr (FMT == 3) return (m.ap ? m.ap[e] : m.aux + size_t(e) * m.aux_stride) + size_t(row) * (cols / 128) * 2;
    return FMT == 2 ? (m.ap ? m.ap[e] : m.aux + size_t(e) * m.aux_stride) + size_t(row) * (cols / 16 + 4) : nullptr;
}

// grid (rows / 8, n * topk): warp = row i of the gate and the up matrix of assignment blockIdx.y
template <int FMT, int NC>
__global__ void __launch_bounds__(256) expert_gate_up_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x, const int * ids,
                                                        __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int a = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (row >= rows) return;
    const int e = ids[a];
    const __nv_bfloat16 * xr = x + size_t(a / topk) * cols;
    const float g0 = row_dot<FMT, NC>(row_ptr<FMT>(mg, e, row, cols), aux_ptr<FMT>(mg, e, row, cols), cols, xr, lane, &lut);
    const float u0 = row_dot<FMT, NC>(row_ptr<FMT>(mu, e, row, cols), aux_ptr<FMT>(mu, e, row, cols), cols, xr, lane, &lut);
    const float g = bfr(warp_sum(g0)), u = bfr(warp_sum(u0));
    if (lane == 0) act[size_t(a) * rows + row] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
}

// NVF4, R rows per warp: grid (rows / (8 R), n * topk); the warp's R gate rows and R up rows share the x loads
template <int NC, int R>
__global__ void __launch_bounds__(256) expert_gate_up_nvf4_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __nv_bfloat16 * x,
                                                             const int * ids, __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int a = blockIdx.y, row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (row0 >= rows) return;
    const int e = ids[a];
    const uint8_t * rp[2 * R]; const uint8_t * ap[2 * R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, rows - 1);
        rp[r] = row_ptr<2>(mg, e, row, cols); ap[r] = aux_ptr<2>(mg, e, row, cols);
        rp[R + r] = row_ptr<2>(mu, e, row, cols); ap[R + r] = aux_ptr<2>(mu, e, row, cols);
    }
    float acc[2 * R];
    rows_dot_nvf4<NC, 2 * R>(rp, ap, cols, x + size_t(a / topk) * cols, lane, &lut, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float g = bfr(warp_sum(acc[r])), u = bfr(warp_sum(acc[R + r]));
        if (lane == 0 && row0 + r < rows)
            act[size_t(a) * rows + row0 + r] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
    }
}
// NVF4, R rows per warp: grid (rows / (8 R), n); per expert k the warp's R rows share the act loads
template <int NC, int R>
__global__ void __launch_bounds__(256) expert_down_nvf4_k(ExpertMat md, int rows, int cols, const __nv_bfloat16 * act, const int * ids,
                                                          const float * wts, __nv_bfloat16 * out, int topk) {
    const int t = blockIdx.y, row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    __shared__ DecodeLut lut;
    lut_init(&lut);
    if (row0 >= rows) return;
    float y[R];
#pragma unroll
    for (int r = 0; r < R; ++r) y[r] = 0.0f;
    for (int k = 0; k < topk; ++k) {
        const int a = t * topk + k, e = ids[a];
        const uint8_t * rp[R]; const uint8_t * ap[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { const int row = min(row0 + r, rows - 1); rp[r] = row_ptr<2>(md, e, row, cols); ap[r] = aux_ptr<2>(md, e, row, cols); }
        float p[R];
        rows_dot_nvf4<NC, R>(rp, ap, cols, act + size_t(a) * cols, lane, &lut, p);
        const float w = wts[a];
#pragma unroll
        for (int r = 0; r < R; ++r) y[r] = fmaf(bfr(warp_sum(p[r])), w, y[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) if (lane == 0 && row0 + r < rows) out[size_t(t) * rows + row0 + r] = __float2bfloat16(y[r]);
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
// grid (rows / 8, n * topk): as expert_gate_up_k, xh = x rotated per 128 (FP16)
__global__ void __launch_bounds__(256) expert_gate_up_tqt_k(ExpertMat mg, ExpertMat mu, int rows, int cols, const __half * xh, const int * ids,
                                                            __nv_bfloat16 * act, int topk, float gc, float uc) {
    const int a = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const int e = ids[a];
    const __half * xr = xh + size_t(a / topk) * cols;
    const float g0 = row_dot_tqt(row_ptr<3>(mg, e, row, cols), aux_ptr<3>(mg, e, row, cols), cols, xr, lane);
    const float u0 = row_dot_tqt(row_ptr<3>(mu, e, row, cols), aux_ptr<3>(mu, e, row, cols), cols, xr, lane);
    const float g = bfr(warp_sum(g0)), u = bfr(warp_sum(u0));
    if (lane == 0) act[size_t(a) * rows + row] = __float2bfloat16(gc * tanhf(g / gc) * (1.0f / (1.0f + expf(-g))) * (uc * tanhf(u / uc)));
}
// grid (rows / 8, n): as expert_down_k, acth = act rotated per 128 (FP16 [n * topk][cols])
__global__ void __launch_bounds__(256) expert_down_tqt_k(ExpertMat md, int rows, int cols, const __half * acth, const int * ids, const float * wts,
                                                         __nv_bfloat16 * out, int topk) {
    const int t = blockIdx.y, row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    float p[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        const int a = t * topk + k, e = ids[a];
        p[k] = row_dot_tqt(row_ptr<3>(md, e, row, cols), aux_ptr<3>(md, e, row, cols), cols, acth + size_t(a) * cols, lane);
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
    float p[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        const int a = t * topk + k, e = ids[a];
        p[k] = row_dot<FMT, NC>(row_ptr<FMT>(md, e, row, cols), aux_ptr<FMT>(md, e, row, cols), cols, act + size_t(a) * cols, lane, &lut);
    }
    float y = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k >= topk) break;
        y = fmaf(bfr(warp_sum(p[k])), wts[t * topk + k], y);
    }
    if (lane == 0) out[size_t(t) * rows + row] = __float2bfloat16(y);
}

// one warp per token: s = sigmoid(logit); top-k of s + bias (lowest index on ties); w = s / sum * scale
__global__ void route_topk_k(const float * logits, const float * bias, int * ids, float * wts, int E, int topk, float scale) {
    const int t = blockIdx.x, lane = threadIdx.x;
    __shared__ float s[256], sel[256];
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
__global__ void __launch_bounds__(kArnSplitThreads) attn_res_stats_k(States st, const float * __restrict__ q, int nq, float * part, unsigned * cnt,
                                                                     int D) {
    const int t = blockIdx.y, b = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int NV = kMaxStates * 5, NW = kArnSplitThreads / 32;
    __shared__ float sh[NW][NV];
    const int c = b * kArnSplitThreads + threadIdx.x;
    float v[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) v[k] = 0.0f;
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
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        if (k / 5 >= st.count || (k % 5) > nq) continue;
        const float r = warp_sum(v[k]);
        if (lane == 0) sh[warp][k] = r;
    }
    __syncthreads();
    if (threadIdx.x < NV && threadIdx.x / 5 < st.count && threadIdx.x % 5 <= nq) {
        float r = 0.0f;
        for (int w = 0; w < NW; ++w) r += sh[w][threadIdx.x];
        part[(size_t(t) * kArnSplit + b) * NV + threadIdx.x] = r;
    }
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(&cnt[t], 1u);
        if (old == kArnSplit - 1) { cnt[t] = 0u; __threadfence(); }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kArnSplitThreads) attn_res_mix_k(States st, int nq, float * part, float * out, int n, int D) {
    const int t = blockIdx.y, b = blockIdx.x;
    constexpr int NV = kMaxStates * 5;
    __shared__ float tot[NV];
    __shared__ float sp[kMaxStates];  // inv[j]
    if (threadIdx.x < NV && threadIdx.x / 5 < st.count && threadIdx.x % 5 <= nq) {
        float r = 0.0f;
        for (int bb = 0; bb < kArnSplit; ++bb) r += part[(size_t(t) * kArnSplit + bb) * NV + threadIdx.x];
        tot[threadIdx.x] = r;
    }
    __syncthreads();
    if (threadIdx.x < kMaxStates) sp[threadIdx.x] = threadIdx.x < st.count ? rsqrtf(tot[threadIdx.x * 5] / float(D) + 1e-6f) : 0.0f;
    __syncthreads();
    const float sd = rsqrtf(float(D));
    const int c = b * kArnSplitThreads + threadIdx.x;
    if (c < D)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i >= nq) continue;
            float p[kMaxStates], m = -INFINITY;
#pragma unroll
            for (int j = 0; j < kMaxStates; ++j) if (j < st.count) { p[j] = tot[j * 5 + 1 + i] * sp[j] * sd; m = fmaxf(m, p[j]); }
            float l = 0.0f;
#pragma unroll
            for (int j = 0; j < kMaxStates; ++j) if (j < st.count) l += expf(p[j] - m);
#pragma unroll
            for (int j = 0; j < kMaxStates; ++j) if (j < st.count) p[j] = expf(p[j] - m) / l;
            float y = 0.0f;
#pragma unroll
            for (int j = 0; j < kMaxStates; ++j) if (j < st.count) y += p[j] * st.s[j][size_t(t) * D + c] * sp[j];
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

int grid_of(size_t n) { return int(std::min<size_t>((n + 255) / 256, 4096)); }
}  // namespace

void attn_res(const States & st, const float * q, int nq, float * out, int n, int D, float * scratch, cudaStream_t s) {
    attn_res_norm(st, q, nq, out, nullptr, nullptr, n, D, 0.0f, scratch, s);
}
void rms_norm(const float * x, const float * w, float * y32, __nv_bfloat16 * y16, int n, int D, float eps, cudaStream_t s) {
    if (n > 0) rms_norm_k<<<n, 256, 0, s>>>(x, w, y32, y16, D, eps);
}
void to_bf16(const float * x, __nv_bfloat16 * y, size_t n, cudaStream_t s) { if (n) to_bf16_k<<<grid_of(n), 256, 0, s>>>(x, y, n); }
void to_f32(const __nv_bfloat16 * x, float * y, size_t n, cudaStream_t s) { if (n) to_f32_k<<<grid_of(n), 256, 0, s>>>(x, y, n); }
void embed(const __nv_bfloat16 * table, const int * ids, const float * w, float * y, int n, int D, float eps, cudaStream_t s) {
    if (n > 0) embed_k<<<n, 256, 0, s>>>(table, ids, w, y, D, eps);
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
    if (n > 0) msg_gate_in_k<<<n, kMgiThreads, 0, s>>>(h, m, z, D, eps);
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
void add(const float * a, const float * b, float * y, size_t n, bool round_bf16, cudaStream_t s) { if (n) add_k<<<grid_of(n), 256, 0, s>>>(a, b, y, n, round_bf16); }
void moe_combine(const __nv_bfloat16 * y, const float * wts, __nv_bfloat16 * out, int n, int D, int topk, cudaStream_t s) {
    if (n > 0) moe_combine_k<<<dim3((D + 255) / 256, n), 256, 0, s>>>(y, wts, out, D, topk);
}
void moe_combine_grouped(const __nv_bfloat16 * yg, const int * order, int A, const float * wts, __nv_bfloat16 * out, int n, int D, int topk, int * inv,
                         cudaStream_t s) {
    if (n <= 0) return;
    inv_order_k<<<(A + 255) / 256, 256, 0, s>>>(order, A, inv);
    moe_combine_grouped_k<<<dim3((D + 255) / 256, n), 256, 0, s>>>(yg, inv, wts, out, D, topk);
}
void softcap(float * x, size_t n, float c, cudaStream_t s) { if (n) softcap_k<<<grid_of(n), 256, 0, s>>>(x, n, c); }

void attn_res_norm(const States & st, const float * queries, int nq, float * out, const float * const * w, __nv_bfloat16 * const * y16, int n, int D,
                   float eps, float * scratch, cudaStream_t s) {
    if (n <= 0) return;
    if (D > kArnThreads * kArnPer || nq > 4) throw std::runtime_error("inkling2 attn_res_norm: D too large or nq > 4");
    NormOut no{};
    for (int i = 0; i < nq; ++i) { no.w[i] = w ? w[i] : nullptr; no.y[i] = y16 ? y16[i] : nullptr; }
    if (n <= kSmallN) {  // decode / verify block: split D over blocks, then the per-stream norms from the F32 out
        if (!scratch) throw std::runtime_error("inkling2 attn_res_norm: null scratch");
        float * part = scratch;
        bool need_norm = false;
        for (int i = 0; i < nq; ++i) need_norm = need_norm || no.y[i];
        // with no out buffer and norms requested, the attention result goes to the tail of the scratch
        float * dst = out ? out : (need_norm ? part + size_t(n) * (kArnSplit * kMaxStates * 5 + 1) : nullptr);
        if (need_norm && !dst) throw std::runtime_error("inkling2 attn_res_norm: null out");
        attn_res_stats_k<<<dim3(kArnSplit, n), kArnSplitThreads, 0, s>>>(st, queries, nq, part, reinterpret_cast<unsigned *>(part + size_t(n) * kArnSplit * kMaxStates * 5), D);
        attn_res_mix_k<<<dim3(kArnSplit, n), kArnSplitThreads, 0, s>>>(st, nq, part, dst, n, D);
        for (int i = 0; i < nq; ++i) {
            if (!no.y[i]) continue;
            if (!no.w[i]) throw std::runtime_error("inkling2 attn_res_norm: norm weight without w");
            rms_norm_k<<<n, 256, 0, s>>>(dst + size_t(i) * n * D, no.w[i], nullptr, no.y[i], D, eps);
        }
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
    return n > 0 && n <= kSmallN ? n * (kArnSplit * kMaxStates * 5 + 1) + 4 * n * D : 0;
}

void gemv(const GemvSeg * segs, int nseg, int n, cudaStream_t s) {
    if (n <= 0 || nseg <= 0) return;
    if (n > kSmallN || nseg > kMaxSegs) throw std::runtime_error("inkling2 gemv: too many tokens / segments");
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
        expert_down_nvf4_k<NC, R><<<dim3((rows + 8 * R - 1) / (8 * R), n), 256, 0, s>>>(d, rows, cols, act, ids, wts, out, topk);
    } else {
        expert_down_k<FMT, NC><<<dim3((rows + 7) / 8, n), 256, 0, s>>>(d, rows, cols, act, ids, wts, out, topk);
    }
}
void expert_gate_up(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const __nv_bfloat16 * x, const int * ids, __nv_bfloat16 * act, int n,
                    int topk, float gc, float uc, cudaStream_t s, const __half * xh) {
    if (n <= 0) return;
    if (cols % 128 || g.fmt != u.fmt) throw std::runtime_error("inkling2 experts: cols % 128 or mixed formats");
    if (g.fmt == 3) {
        if (!xh) throw std::runtime_error("inkling2 experts: TQT needs the rotated activation");
        expert_gate_up_tqt_k<<<dim3((rows + 7) / 8, n * topk), 256, 0, s>>>(g, u, rows, cols, xh, ids, act, topk, gc, uc);
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
    if (d.fmt == 3) {
        if (!acth) throw std::runtime_error("inkling2 experts: TQT needs the rotated activation");
        expert_down_tqt_k<<<dim3((rows + 7) / 8, n), 256, 0, s>>>(d, rows, cols, acth, ids, wts, out, topk);
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
    if (g.fmt == 3) throw std::runtime_error("inkling2 expert_dedup: TQT experts take the per-assignment kernels");
    if (A > kMaxAssign || n > kMaxMembers || g.fmt != d.fmt) throw std::runtime_error("inkling2 expert_dedup: block too wide");
    dedup_groups_k<<<1, kMaxAssign, 0, s>>>(ids, A, groups);
    if (g.fmt == 2) {
        dedup_gate_up_k<2><<<dim3((I + 7) / 8, A), 256, 0, s>>>(g, u, I, D, x, ids, groups, act, topk, gc, uc);
        dedup_down_k<2><<<dim3((D + 7) / 8, A), 256, 0, s>>>(d, D, I, act, ids, groups, yd);
    } else {
        dedup_gate_up_k<1><<<dim3((I + 7) / 8, A), 256, 0, s>>>(g, u, I, D, x, ids, groups, act, topk, gc, uc);
        dedup_down_k<1><<<dim3((D + 7) / 8, A), 256, 0, s>>>(d, D, I, act, ids, groups, yd);
    }
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
void route_topk(const float * logits, const float * bias, int * ids, float * wts, int n, int E, int topk, float scale, cudaStream_t s) {
    if (n > 0) route_topk_k<<<n, 32, 0, s>>>(logits, bias, ids, wts, E, topk, scale);
}

}  // namespace tqllm_cuda::inkling2

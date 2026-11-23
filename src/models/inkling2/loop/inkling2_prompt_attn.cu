// Inkling-2-Mini attention for prompt chunks on tensor cores (see inkling2_attention.cuh): flash-attention style, one
// block per (64 queries, head), 4 warps of 16 query rows, 64-key tiles. Scores Q.K^T and P.V run as BF16 mma.sync
// m16n8k16 with F32 accumulators (q, k are BF16 as the reference stores them; P.V runs in FP16, P rounded to FP16 and V
// converted exactly up to FP16's range); the relative bias (distances < rel_extent), the causal /
// window mask, the sink column (the softmax starts from it) and the online softmax run in F32 on the fragments.
#include "inkling2_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <stdexcept>

namespace tqllm_cuda {
namespace {
constexpr int HD = 128, DR = 16, BQ = 128, BK = 64, LD = HD + 8, NP = BQ + BK, NT = BQ / 16 * 32;

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float tau(float alpha, float floor_, int pos) { return 1.0f + alpha * logf(fmaxf(float(pos + 1) / floor_, 1.0f)); }
__device__ __forceinline__ uint32_t smem_u32(const void * p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void ldsm_x4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma(float * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_h(float * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}
// 8 BF16 -> 8 FP16 (V is RMS-normalised: |v| well inside the FP16 range)
__device__ __forceinline__ uint4 bf16x8_to_h(uint4 u) {
    const __nv_bfloat162 * b = reinterpret_cast<const __nv_bfloat162 *>(&u);
    uint4 r;
    uint32_t * o = reinterpret_cast<uint32_t *>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(b[j]); o[j] = pack_h(f.x, f.y); }
    return r;
}

// q -> clip(q * q_scale) (* tau) as BF16, rs -> clip(rs * rel_scale) (* tau), BF16-valued F32; grid (nq, heads), 128 threads
__global__ void __launch_bounds__(128) prompt_prep_k(Inkling2AttnArgs a, __nv_bfloat16 * qp, float * rp) {
    const int i = blockIdx.x, h = blockIdx.y, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int pos = *a.dpos + i;
    const float tp = tau(a.log_alpha, a.log_floor, pos);
    __shared__ float red[4];
    float x = __bfloat162float(a.q[(size_t(i) * a.heads + h) * HD + t]) * __bfloat162float(a.q_scale[h * HD + t]);
    float ss = x * x;
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    x = bf16r(x * fminf(1.0f, a.qk_clip / (sqrtf(red[0] + red[1] + red[2] + red[3]) + 1e-6f)));
    if (a.log_scale) x = bf16r(x * tp);
    qp[(size_t(i) * a.heads + h) * HD + t] = __float2bfloat16(x);
    if (warp == 0) {
        float r = lane < DR ? __bfloat162float(a.rs[(size_t(i) * a.heads + h) * DR + lane]) * __bfloat162float(a.rel_scale[h * DR + lane]) : 0.0f;
        float rn = r * r;
        for (int o = 16; o; o >>= 1) rn += __shfl_xor_sync(0xffffffffu, rn, o);
        r = bf16r(r * fminf(1.0f, a.rel_clip / (sqrtf(rn) + 1e-6f)));
        if (a.log_scale) r = bf16r(r * tp);
        if (lane < DR) rp[(size_t(i) * a.heads + h) * DR + lane] = r;
    }
}

__global__ void __launch_bounds__(NT) prompt_attn_k(Inkling2AttnArgs a, const __nv_bfloat16 * qp, const float * rp) {
    const int qt = blockIdx.x, h = blockIdx.y, t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int g = h / (a.heads / a.kvheads), pos0 = *a.dpos, i0 = qt * BQ;
    __shared__ __align__(16) __nv_bfloat16 sK[BK][LD], sV[BK][LD];
    __shared__ float sP[DR][NP];
    // Q tile through sK into fragments (BK rows at a time: the tile exceeds the 48 KB static smem budget)
    uint32_t qa[HD / 16][4];
    for (int half = 0; half < BQ / BK; ++half) {
        for (int c = t; c < BK * (HD / 8); c += NT) {
            const int r = c / (HD / 8), k = c % (HD / 8), q = half * BK + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (i0 + q < a.nq) v = *reinterpret_cast<const uint4 *>(qp + (size_t(i0 + q) * a.heads + h) * HD + k * 8);
            *reinterpret_cast<uint4 *>(&sK[r][k * 8]) = v;
        }
        __syncthreads();
        if (warp >= half * (BK / 16) && warp < (half + 1) * (BK / 16))
#pragma unroll
            for (int ks = 0; ks < HD / 16; ++ks) ldsm_x4(qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], &sK[(warp - half * (BK / 16)) * 16 + lane % 16][ks * 16 + (lane / 16) * 8]);
        __syncthreads();
    }
    // this thread's two query rows
    const int rl[2] = {warp * 16 + lane / 4, warp * 16 + lane / 4 + 8};
    bool valid[2];
    int qpos[2];
    float rsq[2][DR], m[2], l[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        valid[j] = i0 + rl[j] < a.nq;
        qpos[j] = pos0 + i0 + rl[j];
#pragma unroll
        for (int r = 0; r < DR; ++r) rsq[j][r] = valid[j] ? rp[(size_t(i0 + rl[j]) * a.heads + h) * DR + r] : 0.0f;
        m[j] = valid[j] ? __bfloat162float(a.sink[size_t(i0 + rl[j]) * a.heads + h]) : 0.0f;  // the sink column: exp(sink - m) = 1
        l[j] = lane % 4 == 0 ? 1.0f : 0.0f;  // the sink's exp(0), counted once per quad
    }
    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    const int qmin = pos0 + i0, qmax = pos0 + min(i0 + BQ, a.nq) - 1;
    const int klo = a.window > 0 ? max(0, qmin - a.window + 1) : 0;
    const float scale = rsqrtf(float(HD));
    for (int k0 = (klo / BK) * BK; k0 <= qmax; k0 += BK) {
        for (int c = t; c < BK * (HD / 8); c += NT) {
            const int r = c / (HD / 8), k = c % (HD / 8), key = k0 + r;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
            if (key <= qmax) {
                const size_t row = (size_t(a.ring ? key % a.ring : key) * a.kvheads + g) * HD + k * 8;
                kv = *reinterpret_cast<const uint4 *>(a.kc + row);
                vv = *reinterpret_cast<const uint4 *>(a.vc + row);
            }
            *reinterpret_cast<uint4 *>(&sK[r][k * 8]) = kv;
            *reinterpret_cast<uint4 *>(&sV[r][k * 8]) = bf16x8_to_h(vv);
        }
        const int dmin = qmin - (k0 + BK - 1);  // smallest distance in this tile (may be negative: masked)
        const bool bias = dmin < a.rel_extent;
        if (bias)
            for (int c = t; c < DR * NP; c += NT) {
                const int r = c / NP, d = dmin + c % NP;
                sP[r][c % NP] = d >= 0 && d < a.rel_extent ? __bfloat162float(a.rel_proj[size_t(r) * a.rel_extent + d]) : 0.0f;
            }
        __syncthreads();
        float s[BK / 8][4];
#pragma unroll
        for (int n = 0; n < BK / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int p = 0; p < BK / 16; ++p)
#pragma unroll
            for (int ks = 0; ks < HD / 16; ++ks) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &sK[p * 16 + (lane / 16) * 8 + lane % 8][ks * 16 + ((lane / 8) % 2) * 8]);
                mma(s[2 * p], qa[ks], b0, b1);
                mma(s[2 * p + 1], qa[ks], b2, b3);
            }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < BK / 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = e / 2, key = k0 + n * 8 + 2 * (lane % 4) + (e % 2), d = qpos[j] - key;
                float v = -INFINITY;
                if (valid[j] && d >= 0 && (a.window <= 0 || d < a.window)) {
                    v = s[n][e] * scale;
                    if (bias && d < a.rel_extent) {
                        float b = 0.0f;
#pragma unroll
                        for (int r = 0; r < DR; ++r) b += rsq[j][r] * sP[r][d - dmin];
                        v += b;
                    }
                }
                s[n][e] = v;
                mx[j] = fmaxf(mx[j], v);
            }
        float corr[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            mx[j] = fmaxf(mx[j], __shfl_xor_sync(0xffffffffu, mx[j], 1));
            mx[j] = fmaxf(mx[j], __shfl_xor_sync(0xffffffffu, mx[j], 2));
            const float mn = fmaxf(m[j], mx[j]);
            corr[j] = __expf(m[j] - mn);
            m[j] = mn;
            l[j] *= corr[j];
        }
#pragma unroll
        for (int n = 0; n < HD / 8; ++n) { o[n][0] *= corr[0]; o[n][1] *= corr[0]; o[n][2] *= corr[1]; o[n][3] *= corr[1]; }
        uint32_t pa[BK / 16][4];
#pragma unroll
        for (int n = 0; n < BK / 8; ++n) {
            float p[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) { p[e] = __expf(s[n][e] - m[e / 2]); l[e / 2] += p[e]; }
            pa[n / 2][(n % 2) * 2] = pack_h(p[0], p[1]);
            pa[n / 2][(n % 2) * 2 + 1] = pack_h(p[2], p[3]);
        }
#pragma unroll
        for (int kc = 0; kc < BK / 16; ++kc)
#pragma unroll
            for (int dp = 0; dp < HD / 16; ++dp) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, &sV[kc * 16 + lane % 16][dp * 16 + (lane / 16) * 8]);
                mma_h(o[2 * dp], pa[kc], b0, b1);
                mma_h(o[2 * dp + 1], pa[kc], b2, b3);
            }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        l[j] += __shfl_xor_sync(0xffffffffu, l[j], 1);
        l[j] += __shfl_xor_sync(0xffffffffu, l[j], 2);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (!valid[j]) continue;
        const size_t base = (size_t(i0 + rl[j]) * a.heads + h) * HD;
        const float inv = 1.0f / l[j];
#pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            const int d = n * 8 + 2 * (lane % 4);
            const float g0 = __bfloat162float(a.gate[base + d]), g1 = __bfloat162float(a.gate[base + d + 1]);
            a.out[base + d] = __float2bfloat16(o[n][2 * j] * inv * (1.0f / (1.0f + __expf(-g0))));
            a.out[base + d + 1] = __float2bfloat16(o[n][2 * j + 1] * inv * (1.0f / (1.0f + __expf(-g1))));
        }
    }
}
}  // namespace

void inkling2_attention_prompt(const Inkling2AttnArgs & a, __nv_bfloat16 * qbuf, float * rbuf, cudaStream_t s) {
    if (a.nq <= 0) return;
    if (!a.dpos || a.hf_bf16) throw std::runtime_error("inkling2 prompt attention: needs dpos, F32 mode");
    prompt_prep_k<<<dim3(a.nq, a.heads), 128, 0, s>>>(a, qbuf, rbuf);
    prompt_attn_k<<<dim3((a.nq + BQ - 1) / BQ, a.heads), NT, 0, s>>>(a, qbuf, rbuf);
}

}  // namespace tqllm_cuda

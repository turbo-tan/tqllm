#pragma once
// Kernels of the multi-query drafter attention (src/core/attention_gen.cu); the merge kernel is shared with the 27B's
// verify paths (src/models/qwen/qwen35/attention_*.inc), so the templates live here, in their own namespace.
#include <cuda_bf16.h>
#include <math_constants.h>
#include "common.cuh"

namespace tqllm_attn_gen {
constexpr int GEN_MAXG = 8;
template <int DIM> __host__ __device__ constexpr int gen_chunk() { return DIM <= 64 ? 128 : 64; }

// ---- general multi-query attention (drafter): per-query key range [lo, hi) with an optional sliding window ----
// Query z sits at position p = *dpos + z. Causal: hi = p + 1; non-causal (DFlash block): hi = *dpos + nq for every
// query. Window w > 0: lo = max(0, p - w + 1). Splits are fixed-size chunks of the whole context; chunks outside
// [lo, hi) write an empty partial (max -inf, sum 0).
template <int DIM>
__global__ void attn_gen_split_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                                 float * part, const int * dpos, int nq, int heads, int kvheads, float scale,
                                 int nsplit, int causal, int window, int ring) {
    constexpr int CHUNK = gen_chunk<DIM>();
    __shared__ float qs[GEN_MAXG][DIM];
    __shared__ float sc[GEN_MAXG][CHUNK];
    __shared__ __align__(16) __nv_bfloat16 vs[CHUNK][DIM];
    __shared__ float mx[GEN_MAXG], sm[GEN_MAXG];
    const int kh = blockIdx.x, split = blockIdx.y, z = blockIdx.z, t = threadIdx.x;
    const int g = heads / kvheads;
    const int p = *dpos + z, hi = causal ? p + 1 : *dpos + nq, lo = window > 0 ? max(0, p - window + 1) : 0;
    const int base = ring ? max(0, *dpos - window + 1) / CHUNK * CHUNK : 0;  // ring: scan the window only
    const int start = base + split * CHUNK, a = max(start, lo), bnd = min(start + CHUNK, hi);
    const int slot0 = ring ? start % ring : start;  // chunk-aligned, ring % CHUNK == 0 -> no wrap inside a chunk
    float * out0 = part + (int64_t(z) * heads + kh * g) * nsplit * (DIM + 2);
    if (a >= bnd) {  // empty chunk for this query
        if (t < g) {
            float * o = out0 + (int64_t(t) * nsplit + split) * (DIM + 2);
            o[DIM] = -CUDART_INF_F;
            o[DIM + 1] = 0.f;
        }
        return;
    }
    const int n = bnd - start;  // positions start .. bnd-1 loaded; those below `a` are masked
    for (int i = t; i < g * DIM; i += blockDim.x)
        qs[i / DIM][i % DIM] = __bfloat162float(q[(int64_t(z) * heads + kh * g + i / DIM) * DIM + i % DIM]);
    for (int i = t; i < n * (DIM / 8); i += blockDim.x) {
        const int pp = i / (DIM / 8), c = i % (DIM / 8);
        reinterpret_cast<uint4 *>(&vs[pp][0])[c] =
            reinterpret_cast<const uint4 *>(vc + (int64_t(slot0 + pp) * kvheads + kh) * DIM)[c];
    }
    __syncthreads();
    if (t < CHUNK) {
        float sacc[GEN_MAXG];
        for (int h = 0; h < GEN_MAXG; ++h) sacc[h] = 0.f;
        const bool live = t < n && start + t >= a;
        if (live) {
            const uint4 * kr = reinterpret_cast<const uint4 *>(kc + (int64_t(slot0 + t) * kvheads + kh) * DIM);
            for (int c = 0; c < DIM / 8; ++c) {
                const uint4 raw = kr[c];
                const __nv_bfloat16 * kv = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float kf = __bfloat162float(kv[e]);
                    for (int h = 0; h < g; ++h) sacc[h] += qs[h][c * 8 + e] * kf;
                }
            }
        }
        for (int h = 0; h < g; ++h) sc[h][t] = live ? sacc[h] * scale : -CUDART_INF_F;
    }
    __syncthreads();
    const int warp = t / 32, lane = t % 32, nwarps = blockDim.x / 32;
    for (int h = warp; h < g; h += nwarps) {
        float m = -CUDART_INF_F;
        for (int pp = lane; pp < CHUNK; pp += 32) m = fmaxf(m, sc[h][pp]);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float zz = 0.f;
        for (int pp = lane; pp < CHUNK; pp += 32) {
            const float e = sc[h][pp] > -CUDART_INF_F ? expf(sc[h][pp] - m) : 0.f;
            sc[h][pp] = e;
            zz += e;
        }
        for (int o = 16; o; o >>= 1) zz += __shfl_xor_sync(0xffffffffu, zz, o);
        if (lane == 0) {
            mx[h] = m;
            sm[h] = zz;
        }
    }
    __syncthreads();
    for (int i = t; i < g * DIM; i += blockDim.x) {
        const int h = i / DIM, d = i % DIM;
        float acc = 0.f;
        for (int pp = 0; pp < n; ++pp) acc += sc[h][pp] * __bfloat162float(vs[pp][d]);
        float * o = out0 + (int64_t(h) * nsplit + split) * (DIM + 2);
        o[d] = acc;
        if (d == 0) {
            o[DIM] = mx[h];
            o[DIM + 1] = sm[h];
        }
    }
}

// Same partials as attn_gen_split_k, but one block per (kv head, chunk) serves all nq * g query rows (<= 32), so each
// K/V chunk is read once instead of once per query. QK: two threads per key (64 dims each, K row in registers).
template <int DIM>
__global__ void __launch_bounds__(128) attn_gen_mq_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc,
                                                     const __nv_bfloat16 * vc, float * part, const int * dpos, int nq,
                                                     int heads, int kvheads, float scale, int nsplit, int causal,
                                                     int window, int ring) {
    constexpr int CHUNK = 64, MAXR = 32, HD = DIM / 2, SCS = CHUNK + 4;
    static_assert(DIM == 128, "attn_gen_mq_k: head_dim 128");
    __shared__ __align__(16) float qs[MAXR][DIM];
    __shared__ __align__(16) float sc[MAXR][SCS];
    __shared__ __align__(16) __nv_bfloat16 vs[CHUNK][DIM];
    const int kh = blockIdx.x, split = blockIdx.y, t = threadIdx.x;
    const int g = heads / kvheads, rows = nq * g, p0 = *dpos;
    const int lo_min = window > 0 ? max(0, p0 - window + 1) : 0, hi_max = p0 + nq;
    const int base = ring ? lo_min / CHUNK * CHUNK : 0;
    const int start = base + split * CHUNK, a = max(start, lo_min), bnd = min(start + CHUNK, hi_max);
    const int slot0 = ring ? start % ring : start;
    auto out = [&](int r) { return part + ((int64_t(r / g) * heads + kh * g + r % g) * nsplit + split) * (DIM + 2); };
    if (a >= bnd) {
        for (int r = t; r < rows; r += blockDim.x) out(r)[DIM] = -CUDART_INF_F, out(r)[DIM + 1] = 0.f;
        return;
    }
    const int n = bnd - start;
    for (int i = t; i < rows * DIM; i += blockDim.x) {
        const int r = i / DIM;
        qs[r][i % DIM] = __bfloat162float(q[(int64_t(r / g) * heads + kh * g + r % g) * DIM + i % DIM]);
    }
    for (int i = t; i < n * (DIM / 8); i += blockDim.x) {
        const int pp = i / (DIM / 8), c = i % (DIM / 8);
        reinterpret_cast<uint4 *>(&vs[pp][0])[c] =
            reinterpret_cast<const uint4 *>(vc + (int64_t(slot0 + pp) * kvheads + kh) * DIM)[c];
    }
    for (int i = n * (DIM / 8) + t; i < CHUNK * (DIM / 8); i += blockDim.x)  // P is 0 there; 0 * stale smem may be NaN
        reinterpret_cast<uint4 *>(&vs[i / (DIM / 8)][0])[i % (DIM / 8)] = make_uint4(0u, 0u, 0u, 0u);
    const int key = t / 2, half = t % 2;
    float kf[HD];
    if (key < n) {
        const uint4 * kr = reinterpret_cast<const uint4 *>(kc + (int64_t(slot0 + key) * kvheads + kh) * DIM + half * HD);
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
            const uint4 raw = kr[c];
            const __nv_bfloat16 * kv = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
            for (int e = 0; e < 8; ++e) kf[c * 8 + e] = __bfloat162float(kv[e]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < HD; ++i) kf[i] = 0.f;
    }
    __syncthreads();
    for (int r = 0; r < rows; ++r) {
        const float4 * qr = reinterpret_cast<const float4 *>(&qs[r][half * HD]);
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < HD / 4; ++c) {
            const float4 qv = qr[c];
            s += qv.x * kf[4 * c] + qv.y * kf[4 * c + 1] + qv.z * kf[4 * c + 2] + qv.w * kf[4 * c + 3];
        }
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        if (half == 0) {
            const int p = p0 + r / g, kpos = start + key;
            const int lo = window > 0 ? max(0, p - window + 1) : 0, hi = causal ? p + 1 : hi_max;
            sc[r][key] = key < n && kpos >= lo && kpos < hi ? s * scale : -CUDART_INF_F;
        }
    }
    __syncthreads();
    __shared__ float mx[MAXR], sm[MAXR];
    const int warp = t / 32, lane = t % 32;
    for (int r = warp; r < rows; r += blockDim.x / 32) {
        const float v0 = sc[r][lane], v1 = sc[r][lane + 32];
        float m = fmaxf(v0, v1);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float e0 = v0 > -CUDART_INF_F ? expf(v0 - m) : 0.f, e1 = v1 > -CUDART_INF_F ? expf(v1 - m) : 0.f;
        sc[r][lane] = e0, sc[r][lane + 32] = e1;
        float zz = e0 + e1;
        for (int o = 16; o; o >>= 1) zz += __shfl_xor_sync(0xffffffffu, zz, o);
        if (lane == 0) mx[r] = m, sm[r] = zz;
    }
    __syncthreads();
    // PV: thread owns dim d for every row; P read 4 keys at a time
    const int d = t;
    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r) acc[r] = 0.f;
    for (int pp = 0; pp < CHUNK; pp += 4) {
        const float v0 = __bfloat162float(vs[pp][d]), v1 = __bfloat162float(vs[pp + 1][d]);
        const float v2 = __bfloat162float(vs[pp + 2][d]), v3 = __bfloat162float(vs[pp + 3][d]);
#pragma unroll
        for (int r = 0; r < MAXR; ++r) {
            if (r >= rows) break;
            const float4 p = *reinterpret_cast<const float4 *>(&sc[r][pp]);
            acc[r] += p.x * v0 + p.y * v1 + p.z * v2 + p.w * v3;
        }
    }
#pragma unroll
    for (int r = 0; r < MAXR; ++r) {
        if (r >= rows) break;
        float * o = out(r);
        o[d] = acc[r];
        if (d == 0) o[DIM] = mx[r], o[DIM + 1] = sm[r];
    }
}

// qsa_seq0 (QSA rows of a chain block): only the rows whose split point is >= qsa_from are merged
template <int DIM> __global__ void attn_gen_merge_k(const float * part, __nv_bfloat16 * y, int nsplit, int heads,
                                                    const int * qsa_seq0 = nullptr, int qsa_from = 0) {
    const int h = blockIdx.x, d = threadIdx.x, qz = blockIdx.y;
    if (qsa_seq0 && tqllm_dev::tq_split_point(*qsa_seq0 - 1 + qz) < qsa_from) return;
    const float * base = part + (int64_t(qz) * heads + h) * nsplit * (DIM + 2);
    y += int64_t(qz) * heads * DIM;
    float m = -CUDART_INF_F;
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, base[s * (DIM + 2) + DIM]);
    float zz = 0.f, acc = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const float * pp = base + s * (DIM + 2);
        if (!(pp[DIM + 1] > 0.f)) continue;
        const float w = expf(pp[DIM] - m);
        zz += pp[DIM + 1] * w;
        acc += pp[d] * w;
    }
    y[h * DIM + d] = __float2bfloat16_rn(acc / zz);
}

}  // namespace tqllm_attn_gen

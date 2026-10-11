// Split-KV decode attention (flash-decoding) for one query token, GQA, head_dim 64 or 256.
// Pass 1: block (kv_head, split) stages one chunk of K/V in shared memory once and serves every query head of the
// GQA group, writing an unnormalised partial output plus its max and sum. Pass 2 merges the splits.
#include <cuda_fp16.h>
#include "common.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <math_constants.h>
#include <cstdlib>
#include <stdexcept>

namespace {

constexpr int ATT_MAXG = 8;

template <int DIM> __host__ __device__ constexpr int att_chunk() { return DIM <= 64 ? 128 : 64; }

template <int DIM>
__global__ void attn_split_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, float * part,
                             int seq0, int heads, int kvheads, float scale, int nsplit, const int * dseq0 = nullptr) {
    if (dseq0) seq0 = *dseq0;
    const int z = blockIdx.z, seq = seq0 + z;  // query z of a verify batch sees seq0 + z keys (causal)
    if (int(blockIdx.y) * att_chunk<DIM>() >= seq) return;
    q += int64_t(z) * heads * DIM;
    part += int64_t(z) * heads * nsplit * (DIM + 2);
    constexpr int CHUNK = att_chunk<DIM>();
    __shared__ float qs[ATT_MAXG][DIM];
    __shared__ float sc[ATT_MAXG][CHUNK];
    __shared__ __align__(16) __nv_bfloat16 vs[CHUNK][DIM];
    __shared__ float mx[ATT_MAXG], sm[ATT_MAXG];
    const int kh = blockIdx.x, split = blockIdx.y, t = threadIdx.x;
    const int g = heads / kvheads, start = split * CHUNK, n = min(CHUNK, seq - start);

    for (int i = t; i < g * DIM; i += blockDim.x)
        qs[i / DIM][i % DIM] = __bfloat162float(q[(kh * g + i / DIM) * DIM + i % DIM]);
    for (int i = t; i < n * (DIM / 8); i += blockDim.x) {
        const int p = i / (DIM / 8), c = i % (DIM / 8);
        reinterpret_cast<uint4 *>(&vs[p][0])[c] =
            reinterpret_cast<const uint4 *>(vc + (int64_t(start + p) * kvheads + kh) * DIM)[c];
    }
    __syncthreads();

    if (t < CHUNK) {
        float s[ATT_MAXG];
        for (int h = 0; h < ATT_MAXG; ++h) s[h] = 0.f;
        if (t < n) {
            const uint4 * kr = reinterpret_cast<const uint4 *>(kc + (int64_t(start + t) * kvheads + kh) * DIM);
            for (int c = 0; c < DIM / 8; ++c) {
                const uint4 raw = kr[c];
                const __nv_bfloat16 * kv = reinterpret_cast<const __nv_bfloat16 *>(&raw);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float kf = __bfloat162float(kv[e]);
                    for (int h = 0; h < g; ++h) s[h] += qs[h][c * 8 + e] * kf;
                }
            }
        }
        for (int h = 0; h < g; ++h) sc[h][t] = t < n ? s[h] * scale : -CUDART_INF_F;
    }
    __syncthreads();

    const int warp = t / 32, lane = t % 32, nwarps = blockDim.x / 32;
    for (int h = warp; h < g; h += nwarps) {
        float m = -CUDART_INF_F;
        for (int p = lane; p < CHUNK; p += 32) m = fmaxf(m, sc[h][p]);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float z = 0.f;
        for (int p = lane; p < CHUNK; p += 32) {
            const float e = p < n ? expf(sc[h][p] - m) : 0.f;
            sc[h][p] = e;
            z += e;
        }
        for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
        if (lane == 0) {
            mx[h] = m;
            sm[h] = z;
        }
    }
    __syncthreads();

    // partial layout: [heads][nsplit][DIM + 2] = unnormalised output, max, sum
    for (int i = t; i < g * DIM; i += blockDim.x) {
        const int h = i / DIM, d = i % DIM;
        float acc = 0.f;
        for (int p = 0; p < n; ++p) acc += sc[h][p] * __bfloat162float(vs[p][d]);
        float * out = part + (int64_t(kh * g + h) * nsplit + split) * (DIM + 2);
        out[d] = acc;
        if (d == 0) {
            out[DIM] = mx[h];
            out[DIM + 1] = sm[h];
        }
    }
}

template <int DIM>
__global__ void attn_merge_k(const float * part, __nv_bfloat16 * y, int nsplit_max, int seq0, int heads,
                             const int * dseq0 = nullptr) {
    if (dseq0) seq0 = *dseq0;
    const int h = blockIdx.x, d = threadIdx.x, qz = blockIdx.y;
    const int nsplit = (seq0 + qz + att_chunk<DIM>() - 1) / att_chunk<DIM>();
    const float * base = part + (int64_t(qz) * heads + h) * nsplit_max * (DIM + 2);
    y += int64_t(qz) * heads * DIM;
    float m = -CUDART_INF_F;
    for (int s = 0; s < nsplit; ++s) m = fmaxf(m, base[s * (DIM + 2) + DIM]);
    float z = 0.f, acc = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const float * p = base + s * (DIM + 2);
        const float w = expf(p[DIM] - m);
        z += p[DIM + 1] * w;
        acc += p[d] * w;
    }
    y[h * DIM + d] = __float2bfloat16_rn(acc / z);
}

template <int DIM>
void launch(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y, float * part,
            int seq0, int nq, int h, int kh, float scale, cudaStream_t s) {
    const int nsplit = (seq0 + nq - 1 + att_chunk<DIM>() - 1) / att_chunk<DIM>();  // for the longest query
    attn_split_k<DIM><<<dim3(kh, nsplit, nq), 128, 0, s>>>(q, kc, vc, part, seq0, h, kh, scale, nsplit);
    attn_merge_k<DIM><<<dim3(h, nq), DIM, 0, s>>>(part, y, nsplit, seq0, h);
}

// ---- general multi-query attention (drafter): per-query key range [lo, hi) with an optional sliding window ----
// Query z sits at position p = *dpos + z. Causal: hi = p + 1; non-causal (DFlash block): hi = *dpos + nq for every
// query. Window w > 0: lo = max(0, p - w + 1). Splits are fixed-size chunks of the whole context; chunks outside
// [lo, hi) write an empty partial (max -inf, sum 0).
template <int DIM>
__global__ void attn_gen_split_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                                 float * part, const int * dpos, int nq, int heads, int kvheads, float scale,
                                 int nsplit, int causal, int window, int ring) {
    constexpr int CHUNK = att_chunk<DIM>();
    __shared__ float qs[ATT_MAXG][DIM];
    __shared__ float sc[ATT_MAXG][CHUNK];
    __shared__ __align__(16) __nv_bfloat16 vs[CHUNK][DIM];
    __shared__ float mx[ATT_MAXG], sm[ATT_MAXG];
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
        float sacc[ATT_MAXG];
        for (int h = 0; h < ATT_MAXG; ++h) sacc[h] = 0.f;
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

template <int DIM> __global__ void attn_gen_merge_k(const float * part, __nv_bfloat16 * y, int nsplit, int heads) {
    const int h = blockIdx.x, d = threadIdx.x, qz = blockIdx.y;
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

// ---- multi-query verify attention (head_dim 256): one block per (kv head, split) serves every query position ----
// nq query positions at seq0 - 1 .. seq0 + nq - 2 (query z sees keys [0, seq0 + z)). Fixed split count: block s walks
// its key range in tiles of MQ_TILE, staging K and V once in shared memory for all nq * g query rows, with an online
// softmax per row; each thread owns one head dimension d and keeps the rows' accumulators in registers.
constexpr int MQ_TILE = 32, MQ_MAXROWS = 48;
using tqllm_dev::tq_split_point;  // two-part verify attention split (common.cuh)

static __device__ __constant__ float turbo3_attn_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static __device__ __constant__ float turbo3_attn_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};
static __device__ __constant__ float turbo3_attn_centroids[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f, 0.021663f, 0.066822f, 0.118786f, 0.190207f
};

__global__ void turbo3_inverse_multi_k(__nv_bfloat16 * data, int heads, int dim) {
    const int d = threadIdx.x, sub = blockIdx.x % (dim / 128), h = (blockIdx.x / (dim / 128)) % heads;
    const int q = blockIdx.x / (heads * (dim / 128));
    const int64_t base = (int64_t(q) * heads + h) * dim + sub * 128;
    __shared__ float x[128];
    x[d] = __bfloat162float(data[base + d]) * turbo3_attn_s2[d];
    for (int width = 1; width < 128; width <<= 1) {
        __syncthreads();
        const int group = (d / (2 * width)) * (2 * width), lane = d % (2 * width);
        if (lane < width) {
            const float a = x[group + lane], b = x[group + lane + width];
            x[group + lane] = a + b;
            x[group + lane + width] = a - b;
        }
    }
    __syncthreads();
    data[base + d] = __float2bfloat16_rn(x[d] * (0.08838834764831845f * turbo3_attn_s1[d]));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 x, __nv_bfloat16 y) {
    const __nv_bfloat162 h(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void cp_async16(void * dst, const void * src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src));
}

// Block = 8 warps (DIM == 256), rows = nq * group <= 48 (3 m-tiles). BF16 tensor cores (m16n8k16), work balanced
// over the 4 SM sub-partitions (warps w and w + 4 share one):
//   S = Q K^T: warp w -> key n-tile w % 4, m-tiles {0, 1} (w < 4) or {2} (w >= 4); fragments via ldmatrix.
//   O += P V:  warp w -> dims 32 w .. 32 w + 31, all m-tiles; V fragments via ldmatrix.trans.
// K/V tiles of 32 keys are double-buffered with cp.async.
// Q8: the cache is int8 with one FP16 scale per 32 values ([pos][kv head][DIM / 32]); int8 tiles are double-buffered
// with cp.async and converted to BF16 in shared memory before the (unchanged) tensor-core phases.
template <int DIM, bool Q8, bool TURBO3V>
__global__ void __launch_bounds__(DIM) attn_mq_k(const __nv_bfloat16 * q, const void * kc_, const void * vc_,
                                                 const __half * ksc, const __half * vsc, float * part,
                                                 const int * dseq0, int nq, int heads, int kvheads, float scale,
                                                 int nsplit, int query_offset, int pstride, int blk,
                                                 const int * dep = nullptr) {
    static_assert(DIM == 256 && MQ_TILE == 32 && MQ_MAXROWS == 48, "warp layout assumes 8 warps, 3 m-tiles, 4 n-tiles");
    constexpr int MT = MQ_MAXROWS / 16;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int KS = DIM + 8;        // row stride (bf16): 528 B -> conflict-free ldmatrix rows
    constexpr int SCS = MQ_TILE + 4;   // score row stride (floats): rows start on different banks
    __nv_bfloat16 * qs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);            // [MAXROWS][KS]
    constexpr int NB = Q8 ? 1 : 2;                                               // BF16 tile buffers
    static_assert(!TURBO3V, "Turbo3 V is served by attn_ctv_k");
    constexpr bool Q8V = Q8 && !TURBO3V;                                         // int8 V (else Turbo3 V)
    constexpr int V3B = (DIM / 128) * 50;                                        // Turbo3 V bytes per key
    __nv_bfloat16 * kbuf = qs + MQ_MAXROWS * KS;                                 // [NB][TILE][KS]
    __nv_bfloat16 * vbuf = kbuf + NB * MQ_TILE * KS;                             // [NB][TILE][KS]
    int8_t * k8 = reinterpret_cast<int8_t *>(vbuf + NB * MQ_TILE * KS);          // Q8: [2][TILE][DIM]
    int8_t * v8 = k8 + (Q8 ? 2 * MQ_TILE * DIM : 0);                             // Q8V: [2][TILE][DIM]
    __half * kss = reinterpret_cast<__half *>(v8 + (Q8V ? 2 * MQ_TILE * DIM : 0)); // Q8: [2][TILE][DIM / 32]
    __half * vss = kss + (Q8 ? 2 * MQ_TILE * (DIM / 32) : 0);                     // Q8V: [2][TILE][DIM / 32]
    uint8_t * v3s = reinterpret_cast<uint8_t *>(vss + (Q8V ? 2 * MQ_TILE * (DIM / 32) : 0)); // T3V: [2][TILE][V3B]
    float * lut = reinterpret_cast<float *>(v3s + (TURBO3V ? (2 * MQ_TILE * V3B + 15) / 16 * 16 : 0));
    float * sc = lut + (TURBO3V ? 8 : 0);                                        // [MAXROWS][SCS]
    const auto * kc = static_cast<const __nv_bfloat16 *>(kc_), * vc = static_cast<const __nv_bfloat16 *>(vc_);
    const auto * kq = static_cast<const int8_t *>(kc_), * vq = static_cast<const int8_t *>(vc_);
    const auto * v3 = static_cast<const uint8_t *>(vc_);
    // per-row running max / sum / rescale live in the 4 padding columns of each score row
    auto mrow_ = [&](int r) -> float & { return sc[r * SCS + MQ_TILE]; };
    auto lrow_ = [&](int r) -> float & { return sc[r * SCS + MQ_TILE + 1]; };
    auto arow_ = [&](int r) -> float & { return sc[r * SCS + MQ_TILE + 2]; };
    const int kh = blockIdx.x, split = blockIdx.y, t = threadIdx.x;
    const int warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g, mt = (rows + 15) / 16;
    const int seq0 = *dseq0 + query_offset, seq_last = seq0 + nq - 1;
    // tiles interleaved over the splits (tile i -> split i % nsplit): a key's split and summation order depend only on
    // its position, not on nq, so a query's result is the same for any verify width
    // blk: keys of the current block (slots seq0 - 1 ..) are left to attn_blk_k, so this pass covers [0, seq0 - 1)
    const int stride = nsplit * MQ_TILE, beg = split * MQ_TILE, end = blk ? seq0 - 1 : seq_last;
    // blk (two-part verify attention): row r (query z = r / g at position qpos = seq0 - 1 + depth) takes keys below
    // tq_split_point(qpos) here; attn_blk_k does the rest. The split depends on qpos alone, never on where the block starts.
    __shared__ int lim[MQ_MAXROWS];
    if (t < MQ_MAXROWS) {
        const int z = t / g;
        lim[t] = blk ? tq_split_point(seq0 - 1 + (dep ? dep[min(z, nq - 1)] : z)) : seq0 + z;
    }
    auto sh = [](const void * p) { return uint32_t(__cvta_generic_to_shared(p)); };

    auto stage = [&](int buf, int t0) {
        const int n = min(MQ_TILE, end - t0);
        if constexpr (Q8) {
            for (int i = t; i < MQ_TILE * (DIM / 16); i += blockDim.x) {
                const int p = i / (DIM / 16), c = i % (DIM / 16);
                if (p < n) {
                    const int64_t src = (int64_t(t0 + p) * kvheads + kh) * DIM + c * 16;
                    cp_async16(k8 + (buf * MQ_TILE + p) * DIM + c * 16, kq + src);
                    if constexpr (Q8V) cp_async16(v8 + (buf * MQ_TILE + p) * DIM + c * 16, vq + src);
                }
            }
            if (t < n) {  // per-32 scales: DIM / 32 halves = 16 B per key
                const int64_t si = (int64_t(t0 + t) * kvheads + kh) * (DIM / 32);
                cp_async16(kss + (buf * MQ_TILE + t) * (DIM / 32), ksc + si);
                if constexpr (Q8V) cp_async16(vss + (buf * MQ_TILE + t) * (DIM / 32), vsc + si);
            }
            asm volatile("cp.async.commit_group;\n" ::);
            return;
        }
        for (int i = t; i < MQ_TILE * (DIM / 8); i += blockDim.x) {
            const int p = i / (DIM / 8), c = i % (DIM / 8);
            __nv_bfloat16 * dk = kbuf + (buf * MQ_TILE + p) * KS + c * 8, * dv = vbuf + (buf * MQ_TILE + p) * KS + c * 8;
            if (p < n) {
                const int64_t src = (int64_t(t0 + p) * kvheads + kh) * DIM + c * 8;
                cp_async16(dk, kc + src);
                cp_async16(dv, vc + src);
            } else {  // zero tail: 0 * stale NaN in the MMA
                *reinterpret_cast<uint4 *>(dk) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4 *>(dv) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (beg < end) stage(0, beg);

    for (int i = t; i < MQ_MAXROWS * (DIM / 8); i += blockDim.x) {
        const int r = i / (DIM / 8), c = i % (DIM / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < rows) v = *reinterpret_cast<const uint4 *>(q + (int64_t(r / g) * heads + kh * g + r % g) * DIM + c * 8);
        *reinterpret_cast<uint4 *>(qs + r * KS + c * 8) = v;
    }
    for (int r = t; r < MQ_MAXROWS; r += blockDim.x) {
        mrow_(r) = -CUDART_INF_F;
        lrow_(r) = 0.f;
        arow_(r) = 0.f;
    }
    const int nt = warp % 4, m_lo = warp < 4 ? 0 : 2, m_n = warp < 4 ? 2 : 1; // QK ownership
    const int d0 = warp * 32;                                                   // PV ownership
    float o[MT][4][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[m][j][0] = o[m][j][1] = o[m][j][2] = o[m][j][3] = 0.f;

    int buf = 0;
    for (int t0 = beg; t0 < end; t0 += stride, buf ^= 1) {
        const int n = min(MQ_TILE, end - t0);
        if (t0 + stride < end) {
            stage(buf ^ 1, t0 + stride);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        if constexpr (Q8) {  // int8 x per-32 scale -> BF16 tiles (zero tail rows: 0 * stale NaN in the MMA)
            auto conv = [&](const int8_t * q8, const __half * ss, __nv_bfloat16 * dstbuf) {
                for (int i = t; i < MQ_TILE * (DIM / 16); i += blockDim.x) {
                    const int p = i / (DIM / 16), c = i % (DIM / 16);
                    uint4 o0 = make_uint4(0, 0, 0, 0), o1 = o0;
                    if (p < n) {
                        const float f = __half2float(ss[(buf * MQ_TILE + p) * (DIM / 32) + c / 2]);
                        const uint4 raw = *reinterpret_cast<const uint4 *>(q8 + (buf * MQ_TILE + p) * DIM + c * 16);
                        const int8_t * bb = reinterpret_cast<const int8_t *>(&raw);
                        uint32_t w[8];
#pragma unroll
                        for (int e = 0; e < 8; ++e) w[e] = pack_bf16(float(bb[2 * e]) * f, float(bb[2 * e + 1]) * f);
                        o0 = make_uint4(w[0], w[1], w[2], w[3]);
                        o1 = make_uint4(w[4], w[5], w[6], w[7]);
                    }
                    uint4 * dst = reinterpret_cast<uint4 *>(dstbuf + p * KS + c * 16);
                    dst[0] = o0;
                    dst[1] = o1;
                }
            };
            conv(k8, kss, kbuf);
            if constexpr (Q8V) conv(v8, vss, vbuf);
            __syncthreads();
        }
        const __nv_bfloat16 * ks = kbuf + (Q8 ? 0 : buf) * MQ_TILE * KS, * vs = vbuf + (Q8 ? 0 : buf) * MQ_TILE * KS;
        {
        float c[2][4] = {};
#pragma unroll
            for (int kk = 0; kk < DIM / 16; kk += 2) {
                uint32_t b[4];  // keys nt*8 .. +7, dims kk*16 .. kk*16+31
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                             : "r"(sh(ks + (nt * 8 + lane % 8) * KS + kk * 16 + (lane / 8) * 8)));
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    if (mm >= m_n || m_lo + mm >= mt) break;
                    const int m = m_lo + mm;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        uint32_t a[4];
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                     : "r"(sh(qs + (m * 16 + lane % 16) * KS + (kk + h) * 16 + (lane / 16) * 8)));
                        mma_bf16(c[mm], a[0], a[1], a[2], a[3], b[2 * h], b[2 * h + 1]);
                    }
                }
            }
#pragma unroll
            for (int mm = 0; mm < 2; ++mm) {
                if (mm >= m_n || m_lo + mm >= mt) break;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (m_lo + mm) * 16 + gid + (i >= 2 ? 8 : 0), p = nt * 8 + tig * 2 + (i & 1);
                    sc[r * SCS + p] = r < rows && p < n && t0 + p < lim[min(r, MQ_MAXROWS - 1)] ? c[mm][i] * scale : -CUDART_INF_F;
                }
            }
        }
        __syncthreads();
        if (t < rows * 4) {  // 4 threads per row, keys t % 4 + 4 i (quads are warp-aligned)
            const int r = t / 4;
            const unsigned qm4 = 0xFu << (lane & 28);  // this row's quad: warps can be partly active (rows * 4 < 256)
            float * row = sc + r * SCS + t % 4;
            float v[8], mx = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < 8; ++i) mx = fmaxf(mx, v[i] = row[4 * i]);
            mx = fmaxf(mx, __shfl_xor_sync(qm4, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(qm4, mx, 2));
            const float mold = mrow_(r), mnew = fmaxf(mold, mx);
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float e = v[i] > -CUDART_INF_F ? __expf(v[i] - mnew) : 0.f;
                row[4 * i] = e;
                sum += e;
            }
            sum += __shfl_xor_sync(qm4, sum, 1);
            sum += __shfl_xor_sync(qm4, sum, 2);
            __syncwarp(qm4);
            if (t % 4 == 0) {
                const float alpha = mold > -CUDART_INF_F ? __expf(mold - mnew) : 0.f;
                arow_(r) = alpha;
                lrow_(r) = lrow_(r) * alpha + sum;
                mrow_(r) = mnew;
            }
        }
        __syncthreads();
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= mt) break;
            const float s0 = arow_(m * 16 + gid), s1 = arow_(m * 16 + gid + 8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[m][j][0] *= s0;
                o[m][j][1] *= s0;
                o[m][j][2] *= s1;
                o[m][j][3] *= s1;
            }
        }
#pragma unroll
        for (int kk = 0; kk < MQ_TILE / 16; ++kk) {
            uint32_t b[2][4];  // ldmatrix.trans: V[key][dim] -> B fragments of n-tiles (0,1) and (2,3)
#pragma unroll
            for (int jp = 0; jp < 2; ++jp)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b[jp][0]), "=r"(b[jp][1]), "=r"(b[jp][2]), "=r"(b[jp][3])
                             : "r"(sh(vs + (kk * 16 + lane % 16) * KS + d0 + jp * 16 + (lane / 16) * 8)));
            const int p = kk * 16 + tig * 2;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= mt) break;
                const float * p0 = sc + (m * 16 + gid) * SCS + p, * p1 = p0 + 8 * SCS;
                const uint32_t a0 = pack_bf16(p0[0], p0[1]), a1 = pack_bf16(p1[0], p1[1]);
                const uint32_t a2 = pack_bf16(p0[8], p0[9]), a3 = pack_bf16(p1[8], p1[9]);
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(o[m][j], a0, a1, a2, a3, b[j / 2][2 * (j % 2)], b[j / 2][2 * (j % 2) + 1]);
            }
        }
        __syncthreads();
    }
    // partial layout (shared with attn_gen_merge_k): [z][head][split][DIM + 2]
    auto out = [&](int r) { return part + ((int64_t(r / g) * heads + kh * g + r % g) * pstride + split) * (DIM + 2); };
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = m * 16 + gid + (i >= 2 ? 8 : 0);
                if (r < rows) out(r)[d0 + j * 8 + tig * 2 + (i & 1)] = o[m][j][i];
            }
    if (t < rows) {
        out(t)[DIM] = mrow_(t);
        out(t)[DIM + 1] = lrow_(t);
    }
}

} // namespace

namespace tqllm_cuda {


int attention_partial_floats(int heads, int context, int dim) {
    const int chunk = dim <= 64 ? att_chunk<64>() : att_chunk<256>();
    return heads * ((context + chunk - 1) / chunk) * (dim + 2);
}

void attention_decode(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, int seq, int h, int kh, int d, float scale, cudaStream_t s) {
    if (h % kh || h / kh > ATT_MAXG) throw std::invalid_argument("attention_decode: <= 8 query heads per KV head");
    if (d == 64) launch<64>(q, kc, vc, y, part, seq, 1, h, kh, scale, s);
    else if (d == 256) launch<256>(q, kc, vc, y, part, seq, 1, h, kh, scale, s);
    else throw std::invalid_argument("attention_decode: head_dim must be 64 or 256");
}

void attention_decode_multi_dev(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                                __nv_bfloat16 * y, float * part, const int * dseq0, int max_seq, int nq, int h, int kh,
                                int d, float scale, cudaStream_t s) {
    if (d != 256 || h % kh || h / kh > ATT_MAXG) throw std::invalid_argument("attention_decode_multi_dev: head_dim 256");
    const int nsplit = (max_seq + att_chunk<256>() - 1) / att_chunk<256>();
    attn_split_k<256><<<dim3(kh, nsplit, nq), 128, 0, s>>>(q, kc, vc, part, 0, h, kh, scale, nsplit, dseq0);
    attn_merge_k<256><<<dim3(h, nq), 256, 0, s>>>(part, y, nsplit, 0, h, dseq0);
}

void attention_decode_multi(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                            __nv_bfloat16 * y, float * part, int seq0, int nq, int h, int kh, int d, float scale,
                            cudaStream_t s) {
    if (h % kh || h / kh > ATT_MAXG) throw std::invalid_argument("attention_decode: <= 8 query heads per KV head");
    if (d == 64) launch<64>(q, kc, vc, y, part, seq0, nq, h, kh, scale, s);
    else if (d == 256) launch<256>(q, kc, vc, y, part, seq0, nq, h, kh, scale, s);
    else throw std::invalid_argument("attention_decode: head_dim must be 64 or 256");
}

int attention_gen_partial_floats(int heads, int context, int nq) {
    return nq * heads * ((context + att_chunk<128>() - 1) / att_chunk<128>()) * (128 + 2);
}

void attention_gen128(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                      float * part, const int * dpos, int nq, int max_seq, int h, int kh, float scale, bool causal,
                      int window, cudaStream_t s, int ring) {
    if (h % kh || h / kh > ATT_MAXG) throw std::invalid_argument("attention_gen128: <= 8 query heads per KV head");
    constexpr int CH = att_chunk<128>();
    if (ring && (window <= 0 || ring % CH || ring < window + nq + CH))
        throw std::invalid_argument("attention_gen128: ring needs a window, ring % 64 == 0, ring >= window + nq + 64");
    const int nsplit = ring ? (window + nq + CH - 1) / CH + 1 : (max_seq + CH - 1) / CH;
    if ((h / kh) * nq <= 32)
        attn_gen_mq_k<128><<<dim3(kh, nsplit), 128, 0, s>>>(q, kc, vc, part, dpos, nq, h, kh, scale, nsplit,
                                                            causal ? 1 : 0, window, ring);
    else
        attn_gen_split_k<128><<<dim3(kh, nsplit, nq), 128, 0, s>>>(q, kc, vc, part, dpos, nq, h, kh, scale, nsplit,
                                                                   causal ? 1 : 0, window, ring);
    attn_gen_merge_k<128><<<dim3(h, nq), 128, 0, s>>>(part, y, nsplit, h);
}

void attention_merge256(const float * part, __nv_bfloat16 * y, int nsplit, int heads, int nq, cudaStream_t s) {
    attn_gen_merge_k<256><<<dim3(heads, nq), 256, 0, s>>>(part, y, nsplit, heads);
}

int attention_mq_partial_floats(int heads, int nq, int nsplit) { return nq * heads * nsplit * (256 + 2); }

// CTV verify attention (int8 K with FP16 scale per 32 + Turbo3 V), DIM 256, rows = nq * group <= 48, 8 warps,
// sized for 2 blocks per SM (~46 KB shared) so one block's barrier phases overlap the other's.
//   S = Q K^T on int8 tensor cores (m16n8k32): Q is quantised per (row, 32 dims) in the prologue; each 32-dim block's
//     int32 product is rescaled by q_scale * k_scale. Warp w -> key n-tile w % 4, m-tiles {0, 1} (w < 4) or {2}.
//   O += P V on FP16 tensor cores (FP16 accumulation per 32-key tile, then FP32): V fragments are decoded from the Turbo3 blocks in registers (rotated basis;
//     the caller applies the inverse WHT). Warp w -> dims 32 w .. 32 w + 31, all m-tiles.
namespace ctv {
constexpr int DIM = 256, TILE = 32, MAXR = 48, RS = DIM + 16;  // int8 row stride 272 B: conflict-free 32-bit loads
constexpr int V3B = 100, SCS = TILE + 4, NBK = DIM / 32;  // V3B = sizeof(Turbo3V256)
constexpr int K4B = 132;  // sizeof(Turbo4K256)
// K region per stage: Q8 K = int8 rows (RS) + FP16 scales per 32; Turbo4 K = 132-byte records
__host__ __device__ constexpr size_t kbytes(bool t4k) { return t4k ? size_t(TILE) * K4B : size_t(TILE) * RS + size_t(TILE) * NBK * 2; }
__host__ __device__ constexpr size_t vbytes(bool t4v) { return size_t(TILE) * (t4v ? K4B : V3B); }
__host__ __device__ constexpr size_t smem(bool t4k, bool t4v = false) {
    return size_t(MAXR) * RS + size_t(MAXR) * NBK * 4 + 2 * kbytes(t4k) + (2 * vbytes(t4v) + 15) / 16 * 16 + 8 * 4 +
           2 * size_t(MAXR) * SCS * 4 + 2 * size_t(MAXR) * 4;
}
__device__ __constant__ uint32_t S1BITS[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};  // turbo3 sign flips (1 = -)
__device__ __constant__ uint32_t S2BITS[4] = {0x16ACEE90u, 0x3F628FDCu, 0xB7A5357Au, 0xBEA56562u};
}

// 4 Turbo4 K nibbles ((negative << 3) | rank) -> 4 int8 levels +-{6,18,31,44,58,75,96,127}
__device__ __forceinline__ uint32_t t4k_expand(uint32_t c) {
    const uint32_t pm = __byte_perm(0x2C1F1206u, 0x7F604B3Au, c & 0x7777u);
    uint32_t sg = (c >> 3) & 0x1111u;
    sg = (sg & 0x0011u) | ((sg & 0x1100u) << 8);         // nibble sign bits -> bits 0, 4, 16, 20
    sg = (sg & 0x00010001u) | ((sg & 0x00100010u) << 4); // -> bits 0, 8, 16, 24 (one per byte)
    return __vadd4(pm ^ (sg * 0xFFu), sg);               // two's-complement negate the flagged bytes
}

__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t hadd2_u(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void mma_f16acc(uint32_t (&c)[2], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// T4K: Turbo4 K (Q rotated per 128 by the same transform, int8 per (row, 128); scores rescaled twice per head)
// T4V: Turbo4 V records (4-bit, PV layout) instead of Turbo3 V.
template <bool T4K, bool T4V>
__global__ void __launch_bounds__(256, 2) attn_ctv_k(const __nv_bfloat16 * q, const void * kq_, const uint8_t * v3,
                                                     const __half * ksc, float * part, const int * dseq0, int nq,
                                                     int heads, int kvheads, float scale, int nsplit, int query_offset) {
    using namespace ctv;
    constexpr int MT = MAXR / 16;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);                          // [MAXR][RS]
    float * qsc = reinterpret_cast<float *>(q8 + MAXR * RS);                     // [MAXR][NBK] (T4K: 2 used)
    int8_t * k8 = reinterpret_cast<int8_t *>(qsc + MAXR * NBK);                  // Q8: [2][TILE][RS]; T4K: [2][TILE][K4B]
    __half * kss = reinterpret_cast<__half *>(k8 + 2 * TILE * RS);               // Q8: [2][TILE][NBK]
    uint8_t * v3s = reinterpret_cast<uint8_t *>(k8) + 2 * kbytes(T4K);           // [2][TILE][V3B]
    const auto * kq = static_cast<const int8_t *>(kq_);
    const auto * k4g = static_cast<const uint8_t *>(kq_);
    constexpr int VB = T4V ? K4B : V3B;  // V record bytes per key
    float * lut = reinterpret_cast<float *>(v3s + (2 * vbytes(T4V) + 15) / 16 * 16);
    float * scb = lut + 8;                                                       // [2][MAXR][SCS], col 32 = alpha
    float * mrun = scb + 2 * MAXR * SCS, * lrun = mrun + MAXR;                   // running max / sum per row
    const int kh = blockIdx.x, split = blockIdx.y, t = threadIdx.x;
    const int warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g, mt = (rows + 15) / 16;
    const int seq0 = *dseq0 + query_offset, seq_last = seq0 + nq - 1;
    // tiles interleaved over the splits (tile i -> split i % nsplit): a query's result does not depend on nq
    const int stride = nsplit * TILE, beg = split * TILE, end = seq_last;

    auto stage_k = [&](int buf, int t0) {
        const int n = min(TILE, end - t0);
        if constexpr (T4K) {
            for (int i = t; i < TILE * (K4B / 4); i += blockDim.x) {
                const int p = i / (K4B / 4), w = i % (K4B / 4);
                if (p < n)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(
                                     uint32_t(__cvta_generic_to_shared(k8 + (buf * TILE + p) * K4B + w * 4))),
                                 "l"(k4g + (int64_t(t0 + p) * kvheads + kh) * K4B + w * 4));
            }
        } else {
            for (int i = t; i < TILE * (DIM / 16); i += blockDim.x) {
                const int p = i / (DIM / 16), c = i % (DIM / 16);
                if (p < n) cp_async16(k8 + (buf * TILE + p) * RS + c * 16, kq + (int64_t(t0 + p) * kvheads + kh) * DIM + c * 16);
            }
            if (t < n) cp_async16(kss + (buf * TILE + t) * NBK, ksc + (int64_t(t0 + t) * kvheads + kh) * NBK);
        }
    };
    auto stage_v = [&](int buf, int t0) {
        const int n = min(TILE, end - t0);
        for (int i = t; i < TILE * (VB / 4); i += blockDim.x) {
            const int p = i / (VB / 4), w = i % (VB / 4);
            if (p < n)
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(
                                 uint32_t(__cvta_generic_to_shared(v3s + (buf * TILE + p) * VB + w * 4))),
                             "l"(v3 + (int64_t(t0 + p) * kvheads + kh) * VB + w * 4));
        }
    };
    auto commit = [] { asm volatile("cp.async.commit_group;\n" ::); };
    const int ntile = beg < end ? (end - beg + stride - 1) / stride : 0;
    for (int u = 0; u < 2 && u < ntile; ++u) {
        stage_k(u, beg + u * stride);
        stage_v(u, beg + u * stride);
    }
    commit();

    if constexpr (T4K)  // Q -> rotated (same transform as K) -> int8 per (row, 128): warp per (row, half), 4 values/lane
        for (int pr = warp; pr < MAXR * 2; pr += 8) {
            const int r = pr / 2, h = pr % 2;
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            if (r < rows) {
                const __nv_bfloat16 * qr = q + (int64_t(r / g) * heads + kh * g + r % g) * DIM + h * 128 + lane * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(qr[j]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((S1BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
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
            float mx = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] *= 0.08838834764831845f;
                if ((S2BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
                mx = fmaxf(mx, fabsf(v[j]));
            }
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float sq = mx / 127.f;
            uint32_t pk = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) pk |= (uint32_t(sq > 0.f ? __float2int_rn(v[j] / sq) : 0) & 0xFFu) << (8 * j);
            *reinterpret_cast<uint32_t *>(q8 + r * RS + h * 128 + lane * 4) = pk;
            if (lane == 0) qsc[r * NBK + h] = sq;
        }
    else
    // Q -> int8 per (row, 32 dims): warp handles (row, block) pairs, lane = element
    for (int pr = warp; pr < MAXR * NBK; pr += 8) {
        const int r = pr / NBK, b = pr % NBK;
        float v = 0.f;
        if (r < rows) v = __bfloat162float(q[(int64_t(r / g) * heads + kh * g + r % g) * DIM + b * 32 + lane]);
        float m = fabsf(v);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float s = m / 127.f;
        q8[r * RS + b * 32 + lane] = int8_t(s > 0.f ? __float2int_rn(v / s) : 0);
        if (lane == 0) qsc[r * NBK + b] = s;
    }
    if (t < 8) lut[t] = turbo3_attn_centroids[t];
    for (int r = t; r < MAXR; r += blockDim.x) {
        mrun[r] = -CUDART_INF_F;
        lrun[r] = 0.f;
    }
    const int nt = warp % 4, m_lo = warp < 4 ? 0 : 2, m_n = warp < 4 ? 2 : 1;  // QK ownership
    const int d0 = warp * 32, vsub = d0 / 128, vj0 = d0 % 128;                 // PV ownership (one Turbo3 block)
    float o[MT][4][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[m][j][0] = o[m][j][1] = o[m][j][2] = o[m][j][3] = 0.f;

    // Software pipeline, 2 barriers per tile: phase A = QK(tile u + 1) and PV(tile u) (independent tensor work);
    // phase B = softmax(tile u + 1). Scores are double-buffered; K(u + 2) is prefetched in phase A into the slot QK(u)
    // freed, V(u + 2) in phase B into the slot PV(u) freed.
    auto qk_tile = [&](int buf, int t0, float * sc) {
        const int n = min(TILE, end - t0);
        if constexpr (T4K) {  // S = Q K^T in the rotated basis, int32 per 128-dim half, 2 rescales
            int ci[2][2][4] = {};
            const uint8_t * krec = reinterpret_cast<const uint8_t *>(k8) + (buf * TILE + nt * 8 + gid) * K4B;
#pragma unroll
            for (int b = 0; b < NBK; ++b) {
                const uint32_t cw = *reinterpret_cast<const uint32_t *>(krec + 4 + (b * 4 + tig) * 4);
                const uint32_t b0 = t4k_expand(cw & 0xFFFFu), b1 = t4k_expand(cw >> 16);
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    if (mm >= m_n || m_lo + mm >= mt) break;
                    const int8_t * qa = q8 + ((m_lo + mm) * 16 + gid) * RS + b * 32 + 4 * tig;
                    mma_s8(ci[mm][b / 4], *reinterpret_cast<const uint32_t *>(qa),
                           *reinterpret_cast<const uint32_t *>(qa + 8 * RS), *reinterpret_cast<const uint32_t *>(qa + 16),
                           *reinterpret_cast<const uint32_t *>(qa + 8 * RS + 16), b0, b1);
                }
            }
            const __half * sk0 = reinterpret_cast<const __half *>(reinterpret_cast<const uint8_t *>(k8) +
                                                                  (buf * TILE + nt * 8 + 2 * tig) * K4B);
            const __half * sk1 = reinterpret_cast<const __half *>(reinterpret_cast<const uint8_t *>(sk0) + K4B);
#pragma unroll
            for (int mm = 0; mm < 2; ++mm) {
                if (mm >= m_n || m_lo + mm >= mt) break;
                const int r0 = (m_lo + mm) * 16 + gid;
                float cf[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float q0s = qsc[r0 * NBK + h], q1s = qsc[(r0 + 8) * NBK + h];
                    const float k0s = __half2float(sk0[h]), k1s = __half2float(sk1[h]);
                    cf[0] += float(ci[mm][h][0]) * q0s * k0s;
                    cf[1] += float(ci[mm][h][1]) * q0s * k1s;
                    cf[2] += float(ci[mm][h][2]) * q1s * k0s;
                    cf[3] += float(ci[mm][h][3]) * q1s * k1s;
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = r0 + (i >= 2 ? 8 : 0), p = nt * 8 + tig * 2 + (i & 1);
                    sc[r * SCS + p] = r < rows && p < n && t0 + p < seq0 + r / g ? cf[i] * scale : -CUDART_INF_F;
                }
            }
        } else {  // S = Q K^T
            float cf[2][4] = {};
            const int8_t * kb = k8 + (buf * TILE + nt * 8 + gid) * RS;
            const __half * ks0 = kss + (buf * TILE + nt * 8 + 2 * tig) * NBK;
#pragma unroll
            for (int b = 0; b < NBK; ++b) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t *>(kb + b * 32 + 4 * tig);
                const uint32_t b1 = *reinterpret_cast<const uint32_t *>(kb + b * 32 + 16 + 4 * tig);
                const float k0s = __half2float(ks0[b]), k1s = __half2float(ks0[NBK + b]);
#pragma unroll
                for (int mm = 0; mm < 2; ++mm) {
                    if (mm >= m_n || m_lo + mm >= mt) break;
                    const int r0 = (m_lo + mm) * 16 + gid;
                    const int8_t * qa = q8 + r0 * RS + b * 32 + 4 * tig;
                    int ci[4] = {0, 0, 0, 0};
                    mma_s8(ci, *reinterpret_cast<const uint32_t *>(qa), *reinterpret_cast<const uint32_t *>(qa + 8 * RS),
                           *reinterpret_cast<const uint32_t *>(qa + 16), *reinterpret_cast<const uint32_t *>(qa + 8 * RS + 16),
                           b0, b1);
                    const float q0s = qsc[r0 * NBK + b], q1s = qsc[(r0 + 8) * NBK + b];
                    cf[mm][0] += float(ci[0]) * q0s * k0s;
                    cf[mm][1] += float(ci[1]) * q0s * k1s;
                    cf[mm][2] += float(ci[2]) * q1s * k0s;
                    cf[mm][3] += float(ci[3]) * q1s * k1s;
                }
            }
#pragma unroll
            for (int mm = 0; mm < 2; ++mm) {
                if (mm >= m_n || m_lo + mm >= mt) break;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (m_lo + mm) * 16 + gid + (i >= 2 ? 8 : 0), p = nt * 8 + tig * 2 + (i & 1);
                    sc[r * SCS + p] = r < rows && p < n && t0 + p < seq0 + r / g ? cf[mm][i] * scale : -CUDART_INF_F;
                }
            }
        }
    };
    auto softmax_tile = [&](float * sc) {
        if (t < rows * 4) {  // 4 threads per row; thread q owns key pairs (8i + 2q, 8i + 2q + 1), i = 0..3
            const int r = t / 4, q = t % 4;
            const unsigned qm4 = 0xFu << (lane & 28);
            float * row = sc + r * SCS;
            float v[8], mx = -CUDART_INF_F;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                mx = fmaxf(mx, v[2 * i] = row[8 * i + 2 * q]);
                mx = fmaxf(mx, v[2 * i + 1] = row[8 * i + 2 * q + 1]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(qm4, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(qm4, mx, 2));  // all quad reads of the row are done past this point
            const float mold = mrun[r], mnew = fmaxf(mold, mx);
            float sum = 0.f;
            uint32_t * ph = reinterpret_cast<uint32_t *>(row);  // P as FP16 pairs: word j = keys (2j, 2j + 1)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float e0 = v[2 * i] > -CUDART_INF_F ? __expf(v[2 * i] - mnew) : 0.f;
                const float e1 = v[2 * i + 1] > -CUDART_INF_F ? __expf(v[2 * i + 1] - mnew) : 0.f;
                ph[4 * i + q] = pack_h2(e0, e1);
                sum += e0 + e1;
            }
            sum += __shfl_xor_sync(qm4, sum, 1);
            sum += __shfl_xor_sync(qm4, sum, 2);
            if (q == 0) {
                const float alpha = mold > -CUDART_INF_F ? __expf(mold - mnew) : 0.f;
                row[TILE] = alpha;
                lrun[r] = lrun[r] * alpha + sum;
                mrun[r] = mnew;
            }
        }
    };
    auto pv_tile = [&](int buf, int t0, const float * sc) {
        const int n = min(TILE, end - t0);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= mt) break;
            const float s0 = sc[(m * 16 + gid) * SCS + TILE], s1 = sc[(m * 16 + gid + 8) * SCS + TILE];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[m][j][0] *= s0;
                o[m][j][1] *= s0;
                o[m][j][2] *= s1;
                o[m][j][3] *= s1;
            }
        }
        uint32_t acc[MT][4][2] = {};  // per-tile FP16 accumulation (2x tensor rate on GA10x), folded into FP32 below
#pragma unroll
        for (int kk = 0; kk < TILE / 16; ++kk) {
            // V fragments: lane needs keys kk*16 + {2tig, 2tig+1, 2tig+8, 2tig+9}, dims d0 + 8j + gid (j = 0..3)
            uint32_t bv[4][2];
            if constexpr (T4V) {
                // Turbo4 V: lane l = gid reads one 16-bit field per key (its 4 dims d0 + 8j + l), expands to int8 levels,
                // makes FP16 (1024 + level + 128) with PRMT, pairs keys per register, then (x - 1152) * scale exactly.
                uint32_t h01[4], h23[4], sh[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int key = kk * 16 + 2 * tig + (u & 1) + (u >> 1) * 8;
                    const uint8_t * rec = v3s + (buf * TILE + key) * VB;
                    const uint32_t f = *reinterpret_cast<const uint16_t *>(rec + 4 + (warp * 4 + gid / 2) * 4 + (gid % 2) * 2);
                    const uint32_t lv = t4k_expand(f) ^ 0x80808080u;
                    h01[u] = __byte_perm(lv, 0x64646464u, 0x5140);
                    h23[u] = __byte_perm(lv, 0x64646464u, 0x5342);
                    sh[u] = key < n ? *reinterpret_cast<const uint16_t *>(rec + 2 * vsub) : 0u;
                }
                const uint32_t s01 = sh[0] | (sh[1] << 16), s23 = sh[2] | (sh[3] << 16), NEG = 0xE480E480u;  // -1152
                auto fin = [&](uint32_t x, uint32_t sc2) { return hmul2_u(hadd2_u(x, NEG), sc2); };
                bv[0][0] = fin(__byte_perm(h01[0], h01[1], 0x5410), s01);
                bv[1][0] = fin(__byte_perm(h01[0], h01[1], 0x7632), s01);
                bv[2][0] = fin(__byte_perm(h23[0], h23[1], 0x5410), s01);
                bv[3][0] = fin(__byte_perm(h23[0], h23[1], 0x7632), s01);
                bv[0][1] = fin(__byte_perm(h01[2], h01[3], 0x5410), s23);
                bv[1][1] = fin(__byte_perm(h01[2], h01[3], 0x7632), s23);
                bv[2][1] = fin(__byte_perm(h23[2], h23[3], 0x5410), s23);
                bv[3][1] = fin(__byte_perm(h23[2], h23[3], 0x7632), s23);
            } else
            {
                // lane l = gid owns dims d0 + 8j + l: one 12-bit field per key (Turbo3V256 layout); |centroid| halves
                // come from a PRMT on an FP16 table, the sign by XOR, the norm by HMUL2 (two keys per register)
                uint32_t fld[4], nh[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int key = kk * 16 + 2 * tig + (u & 1) + (u >> 1) * 8;
                    const uint32_t * rec = reinterpret_cast<const uint32_t *>(v3s + (buf * TILE + key) * VB);
                    const int k0 = (12 * gid) / 32;
                    const uint32_t lo = rec[1 + 3 * warp + k0], hi = k0 < 2 ? rec[2 + 3 * warp + k0] : 0u;
                    fld[u] = uint32_t(((uint64_t(hi) << 32) | lo) >> (12 * gid - 32 * k0)) & 0xFFFu;
                    nh[u] = key < n ? (vsub ? rec[0] >> 16 : rec[0] & 0xFFFFu) : 0u;
                }
                const uint32_t n01 = nh[0] | (nh[1] << 16), n23 = nh[2] | (nh[3] << 16);
                constexpr uint32_t M0 = 0x2C47258Cu, M1 = 0x32162F9Au;  // fp16 pairs {0.021663, 0.066822}, {0.118786, 0.190207}
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    auto pair = [&](uint32_t fa, uint32_t fb, uint32_t nn) {
                        const uint32_t ca = (fa >> (3 * j)) & 7u, cb = (fb >> (3 * j)) & 7u;
                        const uint32_t sel = (0x10u + 0x22u * (ca & 3u)) | ((0x10u + 0x22u * (cb & 3u)) << 8);
                        const uint32_t sgn = ((~ca & 4u) << 13) | ((~cb & 4u) << 29);  // negative -> half sign bit
                        return hmul2_u(__byte_perm(M0, M1, sel) ^ sgn, nn);
                    };
                    bv[j][0] = pair(fld[0], fld[1], n01);
                    bv[j][1] = pair(fld[2], fld[3], n23);
                }
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= mt) break;
                const uint32_t * p0 = reinterpret_cast<const uint32_t *>(sc + (m * 16 + gid) * SCS) + kk * 8 + tig;
                const uint32_t * p1 = p0 + 8 * SCS;
                const uint32_t a0 = p0[0], a1 = p1[0], a2 = p0[4], a3 = p1[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) mma_f16acc(acc[m][j], a0, a1, a2, a3, bv[j][0], bv[j][1]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[m][j][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[m][j][1]));
                o[m][j][0] += lo.x;
                o[m][j][1] += lo.y;
                o[m][j][2] += hi.x;
                o[m][j][3] += hi.y;
            }
    };
    if (ntile > 0) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        qk_tile(0, beg, scb);
        __syncthreads();
        softmax_tile(scb);
        __syncthreads();
    }
    for (int u = 0; u < ntile; ++u) {
        const int cur = u & 1, nxt = cur ^ 1;
        if (u + 2 < ntile) stage_k(cur, beg + (u + 2) * stride);
        commit();
        if (u + 1 < ntile) qk_tile(nxt, beg + (u + 1) * stride, scb + nxt * MAXR * SCS);
        pv_tile(cur, beg + u * stride, scb + cur * MAXR * SCS);
        __syncthreads();
        if (u + 2 < ntile) stage_v(cur, beg + (u + 2) * stride);
        commit();
        if (u + 1 < ntile) softmax_tile(scb + nxt * MAXR * SCS);
        asm volatile("cp.async.wait_group 1;\n" ::);  // K(u + 2) landed; V(u + 2) may still be in flight
        __syncthreads();
    }
    auto out = [&](int r) { return part + ((int64_t(r / g) * heads + kh * g + r % g) * nsplit + split) * (DIM + 2); };
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = m * 16 + gid + (i >= 2 ? 8 : 0);
                if (r < rows) out(r)[d0 + j * 8 + tig * 2 + (i & 1)] = o[m][j][i];
            }
    if (t < rows) {
        out(t)[DIM] = mrun[t];
        out(t)[DIM + 1] = lrun[t];
    }
}

// Prefill attention over a Turbo4 K+V cache (FA2 style). Block = 8 warps x 16 rows = 128 (query, head) rows of one KV
// head; every 32-key tile is decoded once per block into shared memory (K int8 [key][256] + 2 scales, V FP16 [key][256]
// in the rotated basis), double-buffered, one barrier per tile. Each warp keeps S, P and O for its 16 rows in registers.
namespace fap {
constexpr int NW = 8, ROWS = NW * 16, TILE = 32, DIM = 256, QS = DIM + 16, KS = DIM + 16, VS = DIM + 8;
constexpr size_t smem() {
    return size_t(ROWS) * QS + 2 * size_t(TILE) * KS + 2 * size_t(TILE) * VS * 2 + 2 * 2 * TILE * 4 + ROWS * 2 * 4;
}
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(uint32_t(__cvta_generic_to_shared(p))));
}

__global__ void __launch_bounds__(fap::NW * 32, 1) attn_t4kv_prefill_k(const __nv_bfloat16 * q, const uint8_t * kc,
                                                                       const uint8_t * vc, __nv_bfloat16 * y,
                                                                       const int * dseq0, int nq, int heads, int kvheads,
                                                                       float scale) {
    using namespace fap;
    constexpr int RB = ctv::K4B;  // 132-byte Turbo4 record
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);            // [ROWS][QS]
    int8_t * k8 = q8 + ROWS * QS;                                   // [2][TILE][KS]
    __half * vh = reinterpret_cast<__half *>(k8 + 2 * TILE * KS);   // [2][TILE][VS]
    float * ksf = reinterpret_cast<float *>(vh + 2 * TILE * VS);    // [2][2 halves][TILE]
    float * qsc = ksf + 2 * 2 * TILE;                               // [ROWS][2]
    const int kh = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g;
    const int r0 = (gridDim.y - 1 - blockIdx.y) * ROWS;  // longest (latest) row blocks first
    const int seq0 = *dseq0;                             // query i sees keys < seq0 + i
    const int q_lo = r0 / g, q_hi = min(r0 + ROWS - 1, rows - 1) / g;
    const int key_end = seq0 + q_hi, ntile = (key_end + TILE - 1) / TILE, mask_from = seq0 + q_lo;

    // Q -> rotated (same transform as K) -> int8 per (row, 128); rows past the end repeat the last row
    for (int pr = warp; pr < ROWS * 2; pr += NW) {
        const int r = pr / 2, h = pr % 2, rr = min(r0 + r, rows - 1);
        const __nv_bfloat16 * qr = q + (int64_t(rr / g) * heads + kh * g + rr % g) * DIM + h * 128 + lane * 4;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = __bfloat162float(qr[j]);
            if ((ctv::S1BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
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
        float mx = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] *= 0.08838834764831845f;
            if ((ctv::S2BITS[lane / 8] >> ((lane * 4 + j) % 32)) & 1u) v[j] = -v[j];
            mx = fmaxf(mx, fabsf(v[j]));
        }
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float sq = mx / 127.f;
        uint32_t pk = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) pk |= (uint32_t(sq > 0.f ? __float2int_rn(v[j] / sq) : 0) & 0xFFu) << (8 * j);
        *reinterpret_cast<uint32_t *>(q8 + r * QS + h * 128 + lane * 4) = pk;
        if (lane == 0) qsc[r * 2 + h] = sq;
    }

    // tile loader: thread -> key t % 32, 32-dim chunk c = t / 32 (K words 4c..4c+3 = dims 32c..; V words 4c.. = chunk c);
    // key-major lanes keep the 16-byte shared stores conflict-free (row strides are 4 banks apart)
    const int lk = t % 32, lc = t / 32;
    uint32_t kr[4], vr[4], ksw = 0, vsw = 0;
    auto load = [&](int t0) {
        const int p = t0 + lk;
        if (p < key_end) {
            const uint32_t * rk = reinterpret_cast<const uint32_t *>(kc + (int64_t(p) * kvheads + kh) * RB);
            const uint32_t * rv = reinterpret_cast<const uint32_t *>(vc + (int64_t(p) * kvheads + kh) * RB);
#pragma unroll
            for (int i = 0; i < 4; ++i) kr[i] = __ldg(rk + 1 + 4 * lc + i), vr[i] = __ldg(rv + 1 + 4 * lc + i);
            ksw = __ldg(rk), vsw = __ldg(rv);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) kr[i] = vr[i] = 0u;
            ksw = vsw = 0u;
        }
    };
    auto store = [&](int buf) {
        uint4 lo, hi;
        lo.x = t4k_expand(kr[0] & 0xFFFFu), hi.x = t4k_expand(kr[0] >> 16);
        lo.y = t4k_expand(kr[1] & 0xFFFFu), hi.y = t4k_expand(kr[1] >> 16);
        lo.z = t4k_expand(kr[2] & 0xFFFFu), hi.z = t4k_expand(kr[2] >> 16);
        lo.w = t4k_expand(kr[3] & 0xFFFFu), hi.w = t4k_expand(kr[3] >> 16);
        int8_t * kd = k8 + (buf * TILE + lk) * KS + lc * 32;
        *reinterpret_cast<uint4 *>(kd) = lo;
        *reinterpret_cast<uint4 *>(kd + 16) = hi;
        if (lc < 2) {
            const __half2 s2 = *reinterpret_cast<const __half2 *>(&ksw);
            ksf[(buf * 2 + lc) * TILE + lk] = __half2float(lc ? __high2half(s2) : __low2half(s2));
        }
        // V: word i holds fields l = 2i, 2i + 1 (4 nibbles j each, dims 32c + 8j + l) -> half2 (l, l + 1) per j
        const uint32_t sh = (lc < 4 ? vsw : vsw >> 16) & 0xFFFFu, s2 = sh | (sh << 16), NEG = 0xE480E480u;  // -1152
        uint32_t out[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t la = t4k_expand(vr[i] & 0xFFFFu) ^ 0x80808080u, lb = t4k_expand(vr[i] >> 16) ^ 0x80808080u;
            const uint32_t a01 = __byte_perm(la, 0x64646464u, 0x5140), a23 = __byte_perm(la, 0x64646464u, 0x5342);
            const uint32_t b01 = __byte_perm(lb, 0x64646464u, 0x5140), b23 = __byte_perm(lb, 0x64646464u, 0x5342);
            out[0][i] = hmul2_u(hadd2_u(__byte_perm(a01, b01, 0x5410), NEG), s2);
            out[1][i] = hmul2_u(hadd2_u(__byte_perm(a01, b01, 0x7632), NEG), s2);
            out[2][i] = hmul2_u(hadd2_u(__byte_perm(a23, b23, 0x5410), NEG), s2);
            out[3][i] = hmul2_u(hadd2_u(__byte_perm(a23, b23, 0x7632), NEG), s2);
        }
        __half * vd = vh + (buf * TILE + lk) * VS + lc * 32;
#pragma unroll
        for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4 *>(vd + 8 * j) = make_uint4(out[j][0], out[j][1], out[j][2], out[j][3]);
    };

    if (ntile > 0) {
        load(0);
        store(0);
        if (ntile > 1) load(TILE);
    }
    __syncthreads();

    const int ra = warp * 16 + gid, rb = ra + 8;
    const float qa0 = qsc[ra * 2], qa1 = qsc[ra * 2 + 1], qb0 = qsc[rb * 2], qb1 = qsc[rb * 2 + 1];
    const int lim_a = seq0 + min(r0 + ra, rows - 1) / g, lim_b = seq0 + min(r0 + rb, rows - 1) / g;  // key < lim visible
    const float sl2 = scale * 1.4426950408889634f;
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
    const int8_t * qrow = q8 + (warp * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * QS + (lane / 16) * 16;

    for (int u = 0; u < ntile; ++u) {
        const int buf = u & 1, t0 = u * TILE;
        const int8_t * kt = k8 + buf * TILE * KS;
        const __half * vt = vh + buf * TILE * VS;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int ci[4][4] = {};
#pragma unroll
            for (int bp = 0; bp < 2; ++bp) {
                const int b0 = 4 * h + 2 * bp;
                uint32_t a0[4], a1[4];
                ldsm_x4(a0, qrow + b0 * 32);
                ldsm_x4(a1, qrow + (b0 + 1) * 32);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    uint32_t kb[4];
                    ldsm_x4(kb, kt + (nt * 8 + lane % 8) * KS + b0 * 32 + (lane / 8) * 16);
                    mma_s8(ci[nt], a0[0], a0[1], a0[2], a0[3], kb[0], kb[1]);
                    mma_s8(ci[nt], a1[0], a1[1], a1[2], a1[3], kb[2], kb[3]);
                }
            }
            const float qa = h ? qa1 : qa0, qb = h ? qb1 : qb0;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float2 kk = *reinterpret_cast<const float2 *>(ksf + (buf * 2 + h) * TILE + nt * 8 + 2 * tig);
                s[nt][0] += float(ci[nt][0]) * qa * kk.x;
                s[nt][1] += float(ci[nt][1]) * qa * kk.y;
                s[nt][2] += float(ci[nt][2]) * qb * kk.x;
                s[nt][3] += float(ci[nt][3]) * qb * kk.y;
            }
        }
        const bool msk = t0 + TILE > mask_from;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float x = s[nt][i] * sl2;
                if (msk && t0 + nt * 8 + 2 * tig + (i & 1) >= (i < 2 ? lim_a : lim_b)) x = -CUDART_INF_F;
                s[nt][i] = x;
                if (i < 2) mxa = fmaxf(mxa, x);
                else mxb = fmaxf(mxb, x);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);  // finite: every row sees key 0 in tile 0
        const float al = exp2f(m_a - na), bl = exp2f(m_b - nb);
        m_a = na, m_b = nb;
        uint32_t pf[2][4];
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = exp2f(s[nt][0] - na), p1 = exp2f(s[nt][1] - na);
            const float p2 = exp2f(s[nt][2] - nb), p3 = exp2f(s[nt][3] - nb);
            sa += p0 + p1, sb += p2 + p3;
            pf[nt / 2][(nt & 1) * 2] = pack_h2(p0, p1);
            pf[nt / 2][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        l_a = l_a * al + sa, l_b = l_b * bl + sb;
        if (__any_sync(0xffffffffu, al != 1.f || bl != 1.f))
#pragma unroll
            for (int j = 0; j < 32; ++j) o[j][0] *= al, o[j][1] *= al, o[j][2] *= bl, o[j][3] *= bl;
        const __half * vrow = vt + ((lane / 8) & 1) * 8 * VS + (lane % 8) * VS + (lane / 16) * 8;
#pragma unroll
        for (int np = 0; np < 16; ++np) {
            uint32_t acc[2][2] = {};
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                uint32_t vb[4];
                ldsm_x4_t(vb, vrow + kk * 16 * VS + np * 16);
                mma_f16acc(acc[0], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[0], vb[1]);
                mma_f16acc(acc[1], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[2], vb[3]);
            }
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[n][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[n][1]));
                o[2 * np + n][0] += lo.x, o[2 * np + n][1] += lo.y, o[2 * np + n][2] += hi.x, o[2 * np + n][3] += hi.y;
            }
        }
        if (u + 1 < ntile) store(buf ^ 1);
        if (u + 2 < ntile) load(t0 + 2 * TILE);
        __syncthreads();
    }
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const float ia = l_a > 0.f ? 1.f / l_a : 0.f, ib = l_b > 0.f ? 1.f / l_b : 0.f;
    const int ga = r0 + ra, gb = r0 + rb;
    __nv_bfloat16 * ya = y + (int64_t(ga / g) * heads + kh * g + ga % g) * DIM + 2 * tig;
    __nv_bfloat16 * yb = y + (int64_t(gb / g) * heads + kh * g + gb % g) * DIM + 2 * tig;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (ga < rows) *reinterpret_cast<__nv_bfloat162 *>(ya + 8 * j) = __floats2bfloat162_rn(o[j][0] * ia, o[j][1] * ia);
        if (gb < rows) *reinterpret_cast<__nv_bfloat162 *>(yb + 8 * j) = __floats2bfloat162_rn(o[j][2] * ib, o[j][3] * ib);
    }
}

// Prefill attention over a BF16 cache (FA2 style). Block = 4 warps x 16 rows = 64 (query, head) rows of one KV head;
// Q and double-buffered 32-key K / V tiles in shared memory (99 KB: one block per SM), S, P (BF16) and O in registers.
namespace fab {
constexpr int NW = 4, ROWS = NW * 16, TILE = 32, DIM = 256, LS = DIM + 8;
constexpr size_t smem() { return (size_t(ROWS) + 4 * TILE) * LS * 2; }
}

__global__ void __launch_bounds__(fab::NW * 32, 1) attn_bf16_prefill_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc,
                                                                       const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                                                                       const int * dseq0, int nq, int heads, int kvheads,
                                                                       float scale) {
    using namespace fab;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16 * qs = reinterpret_cast<__nv_bfloat16 *>(smem_raw);  // [ROWS][LS]
    __nv_bfloat16 * ks = qs + ROWS * LS;                                 // [2][TILE][LS]
    __nv_bfloat16 * vs = ks + 2 * TILE * LS;                             // [2][TILE][LS]
    const int kh = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g;
    const int r0 = (gridDim.y - 1 - blockIdx.y) * ROWS;  // longest (latest) row blocks first
    const int seq0 = *dseq0;                             // query i sees keys < seq0 + i
    const int q_lo = r0 / g, q_hi = min(r0 + ROWS - 1, rows - 1) / g;
    const int key_end = seq0 + q_hi, ntile = (key_end + TILE - 1) / TILE, mask_from = seq0 + q_lo;

    auto stage = [&](int buf, int t0) {
        for (int i = t; i < TILE * (DIM / 8); i += NW * 32) {
            const int p = i / (DIM / 8), c = i % (DIM / 8);
            __nv_bfloat16 * dk = ks + (buf * TILE + p) * LS + c * 8, * dv = vs + (buf * TILE + p) * LS + c * 8;
            if (t0 + p < key_end) {
                const int64_t src = (int64_t(t0 + p) * kvheads + kh) * DIM + c * 8;
                cp_async16(dk, kc + src);
                cp_async16(dv, vc + src);
            } else {  // zero tail: 0 * stale NaN in the MMA
                *reinterpret_cast<uint4 *>(dk) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4 *>(dv) = make_uint4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (ntile > 0) stage(0, 0);
    if (ntile > 1) stage(1, TILE);
    for (int i = t; i < ROWS * (DIM / 8); i += NW * 32) {  // rows past the end repeat the last row
        const int r = i / (DIM / 8), c = i % (DIM / 8), rr = min(r0 + r, rows - 1);
        *reinterpret_cast<uint4 *>(qs + r * LS + c * 8) =
            *reinterpret_cast<const uint4 *>(q + (int64_t(rr / g) * heads + kh * g + rr % g) * DIM + c * 8);
    }

    const int ra = warp * 16 + gid, rb = ra + 8;
    const int lim_a = seq0 + min(r0 + ra, rows - 1) / g, lim_b = seq0 + min(r0 + rb, rows - 1) / g;  // key < lim visible
    const float sl2 = scale * 1.4426950408889634f;
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
    const __nv_bfloat16 * qrow = qs + (warp * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * LS + (lane / 16) * 8;

    for (int u = 0; u < ntile; ++u) {
        const int buf = u & 1, t0 = u * TILE;
        if (u + 1 < ntile) asm volatile("cp.async.wait_group 1;\n" ::);
        else asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        const __nv_bfloat16 * kt = ks + buf * TILE * LS;
        const __nv_bfloat16 * vt = vs + buf * TILE * LS;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < DIM / 16; kk += 2) {
            uint32_t a0[4], a1[4];
            ldsm_x4(a0, qrow + kk * 16);
            ldsm_x4(a1, qrow + (kk + 1) * 16);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                uint32_t kb[4];
                ldsm_x4(kb, kt + (nt * 8 + lane % 8) * LS + kk * 16 + (lane / 8) * 8);
                mma_bf16(s[nt], a0[0], a0[1], a0[2], a0[3], kb[0], kb[1]);
                mma_bf16(s[nt], a1[0], a1[1], a1[2], a1[3], kb[2], kb[3]);
            }
        }
        const bool msk = t0 + TILE > mask_from;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float x = s[nt][i] * sl2;
                if (msk && t0 + nt * 8 + 2 * tig + (i & 1) >= (i < 2 ? lim_a : lim_b)) x = -CUDART_INF_F;
                s[nt][i] = x;
                if (i < 2) mxa = fmaxf(mxa, x);
                else mxb = fmaxf(mxb, x);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);  // finite: every row sees key 0 in tile 0
        const float al = exp2f(m_a - na), bl = exp2f(m_b - nb);
        m_a = na, m_b = nb;
        uint32_t pf[2][4];
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = exp2f(s[nt][0] - na), p1 = exp2f(s[nt][1] - na);
            const float p2 = exp2f(s[nt][2] - nb), p3 = exp2f(s[nt][3] - nb);
            sa += p0 + p1, sb += p2 + p3;
            pf[nt / 2][(nt & 1) * 2] = pack_bf16(p0, p1);
            pf[nt / 2][(nt & 1) * 2 + 1] = pack_bf16(p2, p3);
        }
        l_a = l_a * al + sa, l_b = l_b * bl + sb;
        if (__any_sync(0xffffffffu, al != 1.f || bl != 1.f))
#pragma unroll
            for (int j = 0; j < 32; ++j) o[j][0] *= al, o[j][1] *= al, o[j][2] *= bl, o[j][3] *= bl;
        const __nv_bfloat16 * vrow = vt + ((lane / 8) & 1) * 8 * LS + (lane % 8) * LS + (lane / 16) * 8;
#pragma unroll
        for (int np = 0; np < 16; ++np)
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                uint32_t vb[4];
                ldsm_x4_t(vb, vrow + kk * 16 * LS + np * 16);
                mma_bf16(o[2 * np], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[0], vb[1]);
                mma_bf16(o[2 * np + 1], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[2], vb[3]);
            }
        __syncthreads();  // buffer buf is refilled below
        if (u + 2 < ntile) stage(buf, t0 + 2 * TILE);
    }
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const float ia = l_a > 0.f ? 1.f / l_a : 0.f, ib = l_b > 0.f ? 1.f / l_b : 0.f;
    const int ga = r0 + ra, gb = r0 + rb;
    __nv_bfloat16 * ya = y + (int64_t(ga / g) * heads + kh * g + ga % g) * DIM + 2 * tig;
    __nv_bfloat16 * yb = y + (int64_t(gb / g) * heads + kh * g + gb % g) * DIM + 2 * tig;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (ga < rows) *reinterpret_cast<__nv_bfloat162 *>(ya + 8 * j) = __floats2bfloat162_rn(o[j][0] * ia, o[j][1] * ia);
        if (gb < rows) *reinterpret_cast<__nv_bfloat162 *>(yb + 8 * j) = __floats2bfloat162_rn(o[j][2] * ib, o[j][3] * ib);
    }
}

// In-block part of two-part verify attention: query z (a tree node, column z) over its ancestors root .. z in depth
// order (keys at cache slots pos + column), FP32 online softmax, merged with the nsplit cache partials (stride pstride;
// the attn_gen_merge_k arithmetic, block partial last) into y.
// par == chain gives a linear block; a node's result depends only on its path, never on the tree's other nodes.
template <int DIM>
__global__ void __launch_bounds__(DIM) attn_blk_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc,
                                                  const __nv_bfloat16 * vc, float * part, const int * dseq0,
                                                  const int * par, const int * dep, int heads, int kvheads, float scale,
                                                  int nsplit, int pstride, __nv_bfloat16 * y) {
    constexpr int MAXK = 64;  // qpos - split + 1 <= TREE_MAX_DEPTH + 32
    __shared__ float sc[MAXK];
    __shared__ int slot[MAXK];
    const int h = blockIdx.x, z = blockIdx.y, d = threadIdx.x, lane = d % 32, wid = d / 32;
    const int pos = *dseq0 - 1, kh = h / (heads / kvheads);
    int path[16], len = 0;
    for (int a = z; a >= 0; a = par[a]) path[len++] = a;  // z .. root; depth e is path[len - 1 - e]
    const int qpos = pos + len - 1, from = tq_split_point(qpos), nk = qpos - from + 1;
    if (d < nk) {  // canonical positions: committed cache, then this node's ancestors
        const int p = from + d;
        slot[d] = p < pos ? p : pos + path[len - 1 - (p - pos)];
    }
    __syncthreads();
    // scores: one warp per key, 8 dims per lane, fixed-order warp reduction
    const uint4 qv = *reinterpret_cast<const uint4 *>(q + (int64_t(z) * heads + h) * DIM + lane * 8);
    const auto * qh = reinterpret_cast<const __nv_bfloat162 *>(&qv);
    for (int i = wid; i < nk; i += DIM / 32) {
        const uint4 kv = *reinterpret_cast<const uint4 *>(kc + (int64_t(slot[i]) * kvheads + kh) * DIM + lane * 8);
        const auto * kh2 = reinterpret_cast<const __nv_bfloat162 *>(&kv);
        float v = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 a = __bfloat1622float2(qh[e]), b = __bfloat1622float2(kh2[e]);
            v += a.x * b.x;
            v += a.y * b.y;
        }
#pragma unroll
        for (int w = 16; w; w >>= 1) v += __shfl_xor_sync(0xffffffffu, v, w);
        if (lane == 0) sc[i] = v * scale;
    }
    __syncthreads();
    // softmax weights and the value sum in position order
    float m = -CUDART_INF_F;
    for (int i = 0; i < nk; ++i) m = fmaxf(m, sc[i]);
    float l = 0.f, o = 0.f;
    for (int i = 0; i < nk; ++i) {
        const float pr = __expf(sc[i] - m);
        l += pr;
        o += pr * __bfloat162float(vc[(int64_t(slot[i]) * kvheads + kh) * DIM + d]);
    }
    (void) dep;
    const float * base = part + (int64_t(z) * heads + h) * pstride * (DIM + 2);
    float mm = m;
    for (int s2 = 0; s2 < nsplit; ++s2) mm = fmaxf(mm, base[s2 * (DIM + 2) + DIM]);
    float zz = 0.f, acc = 0.f;
    for (int s2 = 0; s2 < nsplit; ++s2) {
        const float * pp = base + s2 * (DIM + 2);
        if (!(pp[DIM + 1] > 0.f)) continue;
        const float w = expf(pp[DIM] - mm);
        zz += pp[DIM + 1] * w;
        acc += pp[d] * w;
    }
    if (l > 0.f) {
        const float w = expf(m - mm);
        zz += l * w;
        acc += o * w;
    }
    y[(int64_t(z) * heads + h) * DIM + d] = __float2bfloat16_rn(acc / zz);
}

template <bool Q8, bool TURBO3V>
void mq_launch(const __nv_bfloat16 * q, const void * kc, const void * vc, const __half * ksc, const __half * vsc,
               __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit,
               cudaStream_t s, int query_offset, const int * par = nullptr, const int * dep = nullptr) {
    if (h % kh || ((h / kh) * nq > MQ_MAXROWS && !par)) throw std::invalid_argument("attention_mq256: nq * group must be <= 48");
    constexpr int DIM = 256, KS = DIM + 8;
    constexpr bool Q8V = Q8 && !TURBO3V;
    constexpr size_t V3B = (DIM / 128) * 50;
    const size_t smem = (size_t(MQ_MAXROWS) + (Q8 ? 2 : 4) * MQ_TILE) * KS * 2 +
                        (Q8 ? (Q8V ? 4 : 2) * size_t(MQ_TILE) * DIM : 0) +             // int8 K (and V) stages
                        (Q8 ? (Q8V ? 4 : 2) * size_t(MQ_TILE) * (DIM / 32) * 2 : 0) +  // their scales
                        (TURBO3V ? (2 * MQ_TILE * V3B + 15) / 16 * 16 + 8 * 4 : 0) + // Turbo3 V stages + LUT
                        size_t(MQ_MAXROWS) * (MQ_TILE + 4) * 4;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_mq_k<DIM, Q8, TURBO3V>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
        configured = true;
    }
    if (par) {  // two-part verify attention: cache keys here, the block's ancestors in attn_blk_k, merged in fixed order
        if (Q8 || TURBO3V || query_offset) throw std::invalid_argument("tree attention: BF16 K/V, one verify block");
        const int qmax = MQ_MAXROWS / (h / kh);  // queries per tiled launch; each query's rows are independent
        for (int z0 = 0; z0 < nq; z0 += qmax) {
            const int nz = std::min(qmax, nq - z0);
            if (dep == nullptr && z0) throw std::invalid_argument("tree attention: > 8 queries need node depths");
            attn_mq_k<DIM, Q8, TURBO3V><<<dim3(kh, nsplit), DIM, smem, s>>>(
                q + int64_t(z0) * h * DIM, kc, vc, ksc, vsc, part + int64_t(z0) * h * (nsplit + 1) * (DIM + 2), dseq0, nz, h,
                kh, scale, nsplit, 0, nsplit + 1, 1, dep ? dep + z0 : nullptr);
        }
        attn_blk_k<DIM><<<dim3(h, nq), DIM, 0, s>>>(q, static_cast<const __nv_bfloat16 *>(kc),
                                                    static_cast<const __nv_bfloat16 *>(vc), part, dseq0, par, dep, h, kh,
                                                    scale, nsplit, nsplit + 1, y);
        return;
    }
    attn_mq_k<DIM, Q8, TURBO3V><<<dim3(kh, nsplit), DIM, smem, s>>>(q, kc, vc, ksc, vsc, part, dseq0, nq, h, kh, scale, nsplit,
                                                                    query_offset, nsplit, 0);
    attn_gen_merge_k<DIM><<<dim3(h, nq), DIM, 0, s>>>(part, y, nsplit, h);
}

void attention_mq256(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, __nv_bfloat16 * y,
                     float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                     int query_offset, const int * par, const int * dep) {
    mq_launch<false, false>(q, kc, vc, nullptr, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset, par, dep);
}

void attention_mq256_prefill(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc,
                             __nv_bfloat16 * y, float * part, size_t part_floats, const int * dseq0, int nq, int h, int kh,
                             float scale, int nsplit, cudaStream_t s) {
    constexpr int DIM = 256;
    static bool configured = false;
    if (!configured) {
        if (cudaFuncSetAttribute(attn_bf16_prefill_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(fab::smem())) !=
            cudaSuccess)
            throw std::runtime_error("attention_mq256_prefill: 99 KB of shared memory per block unavailable");
        configured = true;
    }
    const int blocks = ((h / kh) * nq + fab::ROWS - 1) / fab::ROWS;
    attn_bf16_prefill_k<<<dim3(kh, blocks), fab::NW * 32, fab::smem(), s>>>(q, kc, vc, y, dseq0, nq, h, kh, scale);
}

void attention_mq256_q8(const __nv_bfloat16 * q, const int8_t * kc, const int8_t * vc, const __half * ksc,
                        const __half * vsc, __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh,
                        float scale, int nsplit, cudaStream_t s, int query_offset) {
    mq_launch<true, false>(q, kc, vc, ksc, vsc, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

template <bool T4K, bool T4V = false>
static void ctv_launch(const __nv_bfloat16 * q, const void * kc, const void * vc, const __half * ksc,
                       __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit,
                       cudaStream_t s, int query_offset) {
    if (h % kh || (h / kh) * nq > ctv::MAXR)
        throw std::invalid_argument("CTV attention: nq * group must be <= 48");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_ctv_k<T4K, T4V>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(ctv::smem(T4K, T4V)));
        configured = true;
    }
    attn_ctv_k<T4K, T4V><<<dim3(kh, nsplit), 256, ctv::smem(T4K, T4V), s>>>(q, kc, static_cast<const uint8_t *>(vc), ksc, part,
                                                                  dseq0, nq, h, kh, scale, nsplit, query_offset);
    attn_gen_merge_k<256><<<dim3(h, nq), 256, 0, s>>>(part, y, nsplit, h);
    turbo3_inverse_multi_k<<<nq * h * (256 / 128), 128, 0, s>>>(y, h, 256);  // V was accumulated in the rotated basis
}

void attention_mq256_q8_turbo3v(const __nv_bfloat16 * q, const int8_t * kc, const Turbo3V256 * vc,
                               const __half * ksc, __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h,
                               int kh, float scale, int nsplit, cudaStream_t s, int query_offset) {
    ctv_launch<false>(q, kc, vc, ksc, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

void attention_mq256_t4k_turbo3v(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo3V256 * vc,
                                 __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale,
                                 int nsplit, cudaStream_t s, int query_offset) {
    ctv_launch<true>(q, kc, vc, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

void attention_mq256_t4kv(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo4K256 * vc, __nv_bfloat16 * y,
                          float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                          int query_offset) {
    ctv_launch<true, true>(q, kc, vc, nullptr, y, part, dseq0, nq, h, kh, scale, nsplit, s, query_offset);
}

void attention_mq256_t4kv_prefill(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo4K256 * vc,
                                  __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale,
                                  int nsplit, cudaStream_t s) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_t4kv_prefill_k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(fap::smem()));
        configured = true;
    }
    const int blocks = ((h / kh) * nq + fap::ROWS - 1) / fap::ROWS;
    attn_t4kv_prefill_k<<<dim3(kh, blocks), fap::NW * 32, fap::smem(), s>>>(
        q, reinterpret_cast<const uint8_t *>(kc), reinterpret_cast<const uint8_t *>(vc), y, dseq0, nq, h, kh, scale);
    turbo3_inverse_multi_k<<<nq * h * (256 / 128), 128, 0, s>>>(y, h, 256);
}

} // namespace tqllm_cuda

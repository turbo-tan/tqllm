// Split-KV decode attention (flash-decoding) for one query token, GQA, head_dim 64 or 256.
// Pass 1: block (kv_head, split) stages one chunk of K/V in shared memory once and serves every query head of the
// GQA group, writing an unnormalised partial output plus its max and sum. Pass 2 merges the splits.
#include <cuda_fp16.h>
#include "common.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"
#include "attention_gen_kernels.cuh"

#include <math_constants.h>
#include <cstdlib>
#include <stdexcept>

namespace {

constexpr int ATT_MAXG = 8;
using tqllm_attn_gen::attn_gen_merge_k;

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

// qsa_seq0 (QSA rows of a chain block): only the rows whose split point is >= qsa_from
__global__ void turbo3_inverse_multi_k(__nv_bfloat16 * data, int heads, int dim, const int * qsa_seq0 = nullptr, int qsa_from = 0) {
    const int d = threadIdx.x, sub = blockIdx.x % (dim / 128), h = (blockIdx.x / (dim / 128)) % heads;
    const int q = blockIdx.x / (heads * (dim / 128));
    if (qsa_seq0 && tqllm_dev::tq_split_point(*qsa_seq0 - 1 + q) < qsa_from) return;
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
// SEL (QSA, qwen35_qsa.cuh): grid z = one row of the verify block; its keys are the 4 positions of each of its selected
// blocks sel[z][512], in that order, in place of the cache positions below its split point. A row whose split point is
// below `from` is left alone (a dense row). The row's partials depend on its own query and selection only.
template <int DIM, bool Q8, bool TURBO3V, bool SEL = false>
__global__ void __launch_bounds__(DIM) attn_mq_k(const __nv_bfloat16 * q, const void * kc_, const void * vc_,
                                                 const __half * ksc, const __half * vsc, float * part,
                                                 const int * dseq0, int nq, int heads, int kvheads, float scale,
                                                 int nsplit, int query_offset, int pstride, int blk,
                                                 const int * dep = nullptr, const int32_t * sel = nullptr, int from = 0) {
    constexpr int SEL_KEYS = 2048;  // 512 blocks of 4
    if constexpr (SEL) {
        const int z = blockIdx.z;
        if (tq_split_point(*dseq0 - 1 + (dep ? dep[z] : z)) < from) return;
        q += int64_t(z) * heads * DIM, part += int64_t(z) * heads * pstride * (DIM + 2), sel += int64_t(z) * (SEL_KEYS / 4);
        nq = 1, dep = nullptr;
    }
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
    const int stride = nsplit * MQ_TILE, beg = split * MQ_TILE, end = SEL ? SEL_KEYS : blk ? seq0 - 1 : seq_last;
    // blk (two-part verify attention): row r (query z = r / g at position qpos = seq0 - 1 + depth) takes keys below
    // tq_split_point(qpos) here; attn_blk_k does the rest. The split depends on qpos alone, never on where the block starts.
    __shared__ int lim[MQ_MAXROWS];
    if (t < MQ_MAXROWS) {
        const int z = t / g;
        lim[t] = SEL ? SEL_KEYS : blk ? tq_split_point(seq0 - 1 + (dep ? dep[min(z, nq - 1)] : z)) : seq0 + z;
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
        } else {
            for (int i = t; i < MQ_TILE * (DIM / 8); i += blockDim.x) {
                const int p = i / (DIM / 8), c = i % (DIM / 8);
                __nv_bfloat16 * dk = kbuf + (buf * MQ_TILE + p) * KS + c * 8, * dv = vbuf + (buf * MQ_TILE + p) * KS + c * 8;
                if (p < n) {
                    int key = t0 + p;
                    if constexpr (SEL) key = sel[key / 4] * 4 + key % 4;
                    const int64_t src = (int64_t(key) * kvheads + kh) * DIM + c * 8;
                    cp_async16(dk, kc + src);
                    cp_async16(dv, vc + src);
                } else {  // zero tail: 0 * stale NaN in the MMA
                    *reinterpret_cast<uint4 *>(dk) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4 *>(dv) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }
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

int attention_mq_partial_floats(int heads, int nq, int nsplit) { return nq * heads * nsplit * (256 + 2); }

#include "attention_ctv.inc"

#include "attention_fap.inc"

#include "attention_fab.inc"

#include "attention_qsa.inc"


} // namespace tqllm_cuda

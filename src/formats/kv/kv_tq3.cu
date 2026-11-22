// TQ3_0 KV-cache format kernels: writer + FA-style attention (see src/formats/kv/kv_tq3.cpp).
#include "common.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
using namespace tqllm_dev;
void attention_merge256(const float * part, __nv_bfloat16 * y, int nsplit, int heads, int nq, cudaStream_t);

// ---- TQ3_0 K + V attention (FA style, optional key split) ----
// Records: Tq3KV256 (8 fp16 RMS + 3-bit codes per 32-value block, rotated basis). Levels are the TQ3_0 centroids scaled
// by 127 / 1.996684 to int8 {-127, -82, -47, -16, 15, 46, 81, 127}; value = level * rms / F. K: int8 in shared memory,
// scores rescaled per 32-dim block (one IMMA k-step); V: FP16 in shared memory, output rotated back afterwards.
// Block = NW warps x 16 rows of (query, head) for one KV head and one key split. nsplit == 1: normalised BF16 output;
// else partials [q][head][split][258] (natural-log max) for attn_gen_merge_k.
namespace tq3a {
constexpr int TILE = 32, DIM = 256, QS = DIM + 16, KS = DIM + 16, VS = DIM + 8;
constexpr float F = 127.f / 1.996684f;
constexpr uint32_t LUT_LO = 0xF0D1AE81u, LUT_HI = 0x7F512E0Fu;  // bytes: -127 -82 -47 -16 | 15 46 81 127
// NB K/V tile buffers: 2 = double buffered (decode of the next tile overlaps nothing else in the block, 71 KB at NW 4);
// 1 = single buffered (45.5 KB at NW 4, so two blocks share an SM and hide each other's latency).
template <int NW, int NB = 2> constexpr size_t smem() {
    return size_t(NW * 16) * QS + NB * size_t(TILE) * KS + NB * size_t(TILE) * VS * 2 + NB * 8 * TILE * 4 + NW * 16 * 2 * 4;
}
__device__ __forceinline__ uint32_t levels4(const uint32_t (&w)[3], int g) {  // codes 4g .. 4g + 3 -> 4 int8 levels
    const int bit = 12 * g, k = bit / 32, sh = bit % 32;
    const uint32_t x = (sh <= 20 ? w[k] >> sh : __funnelshift_r(w[k], w[k + 1], sh)) & 0xFFFu;
    const uint32_t sel = (x & 7u) | ((x & 0x38u) << 1) | ((x & 0x1C0u) << 2) | ((x & 0xE00u) << 3);
    return __byte_perm(LUT_LO, LUT_HI, sel);
}
__device__ __forceinline__ uint32_t levels2(const uint32_t (&w)[3], int g) {
    const uint32_t x = (w[g / 4] >> (8 * (g % 4))) & 0xFFu;
    uint32_t y = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t code = (x >> (2 * i)) & 3u;
        const int value = code == 0 ? -96 : code == 1 ? -29 : code == 2 ? 29 : 96;
        y |= uint32_t(uint8_t(int8_t(value))) << (8 * i);
    }
    return y;
}
__device__ __constant__ float SIGNS[32] = {1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1,
                                           -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1};
__device__ __constant__ float BOUND[7] = {-1.644041f, -1.015870f, -0.493924f, -0.008701f, 0.477664f, 1.001362f, 1.633223f};

// int8 levels of one 32-value block -> FP16 values in shared memory (value = level * scale / F)
__device__ __forceinline__ void store_levels_h(const uint32_t (&vl)[8], __half sv, __half * vd) {
    const __half2 s2h = __halves2half2(sv, sv);
    const uint32_t s2 = *reinterpret_cast<const uint32_t *>(&s2h), NEG = 0xE480E480u;  // -1152
#pragma unroll
    for (int gq = 0; gq < 8; gq += 2) {
        const uint32_t la = vl[gq] ^ 0x80808080u, lb = vl[gq + 1] ^ 0x80808080u;
        const uint32_t w0 = hmul2u(hadd2u(__byte_perm(la, 0x64646464u, 0x5140), NEG), s2);
        const uint32_t w1 = hmul2u(hadd2u(__byte_perm(la, 0x64646464u, 0x5342), NEG), s2);
        const uint32_t w2 = hmul2u(hadd2u(__byte_perm(lb, 0x64646464u, 0x5140), NEG), s2);
        const uint32_t w3 = hmul2u(hadd2u(__byte_perm(lb, 0x64646464u, 0x5342), NEG), s2);
        *reinterpret_cast<uint4 *>(vd + 4 * gq) = make_uint4(w0, w1, w2, w3);
    }
}
}  // namespace tq3a

// K and V codecs of the split cache. A codec owns its record layout: RB bytes per (token, KV head, 256-dim row), load()
// fetches what one 32-value block needs into registers, then K codecs give int8 levels (score scale = s / F) and V codecs
// write the block's 32 FP16 values into shared memory. All work in the TQ3_0 rotated basis.
namespace kvc {
struct Tq3 {  // TQ3_0: fp16 RMS per 32 values + 3-bit codes (112 B)
    static constexpr int RB = sizeof(Tq3KV256), NW = 3;
    static constexpr bool ROT128 = false;  // K basis: TQ3_0 signs + 32-point WHT
    __device__ static float kscale(__half s) { return __half2float(s) / tq3a::F; }
    __device__ static void load(const uint8_t * r, int b, uint32_t (&w)[3], __half & s) {
#pragma unroll
        for (int j = 0; j < 3; ++j) w[j] = __ldg(reinterpret_cast<const uint32_t *>(r + 16 + 12 * b) + j);
        s = reinterpret_cast<const __half *>(r)[b];
    }
    __device__ static void levels(const uint32_t (&w)[3], uint32_t (&l)[8]) {
#pragma unroll
        for (int gq = 0; gq < 8; ++gq) l[gq] = tq3a::levels4(w, gq);
    }
    __device__ static void store_k(const uint32_t (&w)[3], __half s, int8_t * kd, float & ks, const float * cb) {
        uint32_t kl[8];
        levels(w, kl);
        *reinterpret_cast<uint4 *>(kd) = make_uint4(kl[0], kl[1], kl[2], kl[3]);
        *reinterpret_cast<uint4 *>(kd + 16) = make_uint4(kl[4], kl[5], kl[6], kl[7]);
        ks = kscale(s);
    }

    __device__ static void store_v(const uint32_t (&w)[3], __half s, __half * vd, const float * cb) {
        uint32_t l[8];
        levels(w, l);
        tq3a::store_levels_h(l, __float2half_rn(__half2float(s) / tq3a::F), vd);
    }
};
struct Tq2s {  // scalar 2-bit V: fp16 RMS per 32 values + 2-bit codes (80 B)
    static constexpr int RB = sizeof(Tq2V256), NW = 3;
    __device__ static void load(const uint8_t * r, int b, uint32_t (&w)[3], __half & s) {
#pragma unroll
        for (int j = 0; j < 3; ++j) w[j] = j < 2 ? __ldg(reinterpret_cast<const uint32_t *>(r + 16 + 8 * b) + j) : 0u;
        s = reinterpret_cast<const __half *>(r)[b];
    }
    __device__ static void store_v(const uint32_t (&w)[3], __half s, __half * vd, const float * cb) {
        uint32_t l[8];
#pragma unroll
        for (int gq = 0; gq < 8; ++gq) l[gq] = tq3a::levels2(w, gq);
        tq3a::store_levels_h(l, __float2half_rn(__half2float(s) / tq3a::F), vd);
    }
};
struct Q8 {  // rotated int8 per 32 values, fp16 scale (272 B)
    static constexpr int RB = sizeof(Q8KV256), NW = 8;
    static constexpr bool ROT128 = false;
    __device__ static float kscale(__half s) { return __half2float(s); }
    __device__ static void load(const uint8_t * r, int b, uint32_t (&w)[NW], __half & s) {
        const uint4 a = __ldg(reinterpret_cast<const uint4 *>(r + 16 + 32 * b));
        const uint4 c = __ldg(reinterpret_cast<const uint4 *>(r + 16 + 32 * b) + 1);
        w[0] = a.x, w[1] = a.y, w[2] = a.z, w[3] = a.w, w[4] = c.x, w[5] = c.y, w[6] = c.z, w[7] = c.w;
        s = reinterpret_cast<const __half *>(r)[b];
    }
    __device__ static void levels(const uint32_t (&w)[NW], uint32_t (&l)[8]) {
#pragma unroll
        for (int gq = 0; gq < 8; ++gq) l[gq] = w[gq];
    }
    __device__ static void store_k(const uint32_t (&w)[NW], __half s, int8_t * kd, float & ks, const float * cb) {
        uint32_t kl[8];
        levels(w, kl);
        *reinterpret_cast<uint4 *>(kd) = make_uint4(kl[0], kl[1], kl[2], kl[3]);
        *reinterpret_cast<uint4 *>(kd + 16) = make_uint4(kl[4], kl[5], kl[6], kl[7]);
        ks = kscale(s);
    }

    __device__ static void store_v(const uint32_t (&w)[NW], __half s, __half * vd, const float * cb) {
        uint32_t l[8];
        levels(w, l);
        tq3a::store_levels_h(l, s, vd);
    }
};
struct Q4 {  // rotated 4-bit per 32 values, fp16 scale, value (q - 8) * d (144 B)
    static constexpr int RB = sizeof(Q4KV256), NW = 4;
    static constexpr bool ROT128 = false;
    __device__ static float kscale(__half s) { return __half2float(s); }
    __device__ static void load(const uint8_t * r, int b, uint32_t (&w)[NW], __half & s) {
        const uint4 a = __ldg(reinterpret_cast<const uint4 *>(r + 16 + 16 * b));
        w[0] = a.x, w[1] = a.y, w[2] = a.z, w[3] = a.w;
        s = reinterpret_cast<const __half *>(r)[b];
    }
    __device__ static void levels(const uint32_t (&w)[NW], uint32_t (&l)[8]) {
#pragma unroll
        for (int gq = 0; gq < 8; ++gq) {
            const uint32_t x = (w[gq / 2] >> (16 * (gq % 2))) & 0xFFFFu;
            const uint32_t n = (x & 0xFu) | ((x & 0xF0u) << 4) | ((x & 0xF00u) << 8) | ((x & 0xF000u) << 12);
            l[gq] = __vsub4(n, 0x08080808u);
        }
    }
    __device__ static void store_k(const uint32_t (&w)[NW], __half s, int8_t * kd, float & ks, const float * cb) {
        uint32_t kl[8];
        levels(w, kl);
        *reinterpret_cast<uint4 *>(kd) = make_uint4(kl[0], kl[1], kl[2], kl[3]);
        *reinterpret_cast<uint4 *>(kd + 16) = make_uint4(kl[4], kl[5], kl[6], kl[7]);
        ks = kscale(s);
    }

    __device__ static void store_v(const uint32_t (&w)[NW], __half s, __half * vd, const float * cb) {
        uint32_t l[8];
        levels(w, l);
        tq3a::store_levels_h(l, s, vd);
    }
};
struct Tcq2;
struct Tcq3;
}  // namespace kvc

// ---- trellis-coded 2-bit V (TCQ). Idea: trellis-coded KV quantisation (spiritbuun/buun-llama-cpp); our own codebook and signs,
// trained / generated by tools/train_tcq_codebook.py ----
#include "tcq2_tables.inc"
#include "tcq3_tables.inc"
namespace tcq {
__device__ const float CB2V[256] = TQ_TCQ2_CODEBOOK_V;
__device__ const float CB3K[512] = TQ_TCQ3_CODEBOOK_K;
constexpr float cb_absmax(const float (&a)[512]) {
    float m = 0.f;
    for (int i = 0; i < 512; ++i) m = (a[i] < 0 ? -a[i] : a[i]) > m ? (a[i] < 0 ? -a[i] : a[i]) : m;
    return m;
}
constexpr float CB3K_HOST[512] = TQ_TCQ3_CODEBOOK_K;
constexpr float CB3K_MAX = cb_absmax(CB3K_HOST);
constexpr float CB3K_SCALE = 127.f / CB3K_MAX, CB3K_INV_SCALE = CB3K_MAX / 127.f;
__device__ __constant__ float S1[128] = TQ_TCQ_SIGNS1;
__device__ __constant__ float S2[128] = TQ_TCQ_SIGNS2;
constexpr float ALPHA_K = 1.0f, ALPHA_V = 1.04f, INV_SQRT_128 = 0.08838834764831845f;
}  // namespace tcq

namespace kvc {
struct Tcq2 {  // 2 x (128-value trellis group: 262 stream bits + fp16 norm), 72 B
    static constexpr int RB = sizeof(Tcq2V256), NW = 4;
    __device__ static void load(const uint8_t * r, int b, uint32_t (&w)[NW], __half & s) {
        const uint32_t * g = reinterpret_cast<const uint32_t *>(r) + 9 * (b / 4);
        const int sub = b % 4;
        w[0] = __ldg(g + 2 * sub), w[1] = __ldg(g + 2 * sub + 1), w[2] = __ldg(g + 2 * sub + 2), w[3] = __ldg(g + 8);
        const unsigned short hs = static_cast<unsigned short>(w[3] >> 16);
        s = *reinterpret_cast<const __half *>(&hs);
    }
    // cb: the codebook as FP16 pairs in shared memory; values = codebook * norm with one paired multiply per two values
    __device__ static void store_v(const uint32_t (&w)[NW], __half s, __half * vd, const float * cb) {
        const __half * hc = reinterpret_cast<const __half *>(cb);
        const __half2 n2 = __halves2half2(s, s);
#pragma unroll
        for (int j = 0; j < 32; j += 8) {
            uint32_t h[4];
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                const int b0 = 2 * (j + e), b1 = b0 + 2;
                const uint32_t s0 = __funnelshift_r(w[b0 / 32], w[b0 / 32 + 1], b0 % 32) & 0xFFu;
                const uint32_t s1 = __funnelshift_r(w[b1 / 32], w[b1 / 32 + 1], b1 % 32) & 0xFFu;
                const __half2 v = __hmul2(__halves2half2(hc[s0], hc[s1]), n2);
                h[e / 2] = *reinterpret_cast<const uint32_t *>(&v);
            }
            *reinterpret_cast<uint4 *>(vd + j) = make_uint4(h[0], h[1], h[2], h[3]);
        }
    }
    static void inverse(__nv_bfloat16 * y, int64_t rows, cudaStream_t s);
};
struct Tcq3 {  // K: 2 x (128-value 512-state trellis group: 390 stream bits + fp16 norm), 104 B; decoded to int8 per 32 for the score MMA
    static constexpr int RB = sizeof(Tcq3K256), NW = 5;
    static constexpr bool ROT128 = true;  // K basis: TCQ signs + 128-point WHT (Q rotated the same way)
    __device__ static void load(const uint8_t * r, int b, uint32_t (&w)[NW], __half & s) {
        const uint32_t * g = reinterpret_cast<const uint32_t *>(r) + 13 * (b / 4);
        const int sub = b % 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) w[j] = __ldg(g + 3 * sub + j);
        w[4] = __ldg(g + 12);
        const unsigned short hs = static_cast<unsigned short>(w[4] >> 16);
        s = *reinterpret_cast<const __half *>(&hs);
    }
    // cb: the codebook as int8 levels on one fixed scale (tcq::CB3K_SCALE), so a value's int8 level is a table lookup and the
    // group norm folds into the score scale (no per-block requantisation).
    __device__ static void store_k(const uint32_t (&w)[NW], __half s, int8_t * kd, float & ks, const float * cb) {
        const int8_t * lv = reinterpret_cast<const int8_t *>(cb);
#pragma unroll
        for (int j = 0; j < 32; j += 16) {
            uint32_t p[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                uint32_t x = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int b = 3 * (j + 4 * e + k);
                    const uint32_t st = __funnelshift_r(w[b / 32], w[b / 32 + 1], b % 32) & 0x1FFu;
                    x |= uint32_t(uint8_t(lv[st])) << (8 * k);
                }
                p[e] = x;
            }
            *reinterpret_cast<uint4 *>(kd + j) = make_uint4(p[0], p[1], p[2], p[3]);
        }
        ks = __half2float(s) * tcq::CB3K_INV_SCALE;
    }
};
}  // namespace kvc

// SEL (QSA, qwen35_qsa.cuh; two-part verify, pstride > 0): grid y = one row of the verify block; its keys are the 4
// positions of each of its selected blocks sel[z][512], in that order, in place of the cache positions below its split
// point. A row whose split point is below `from` is left alone (a dense row). Same decode and arithmetic as the dense
// cache part; the row's partials depend on its own query and selection only.
template <int NW, class KC, class VC, int NB, bool SEL = false>
__global__ void __launch_bounds__(NW * 32, NB == 1 ? 2 : 1) attn_tq3kv_k(const __nv_bfloat16 * q, const uint8_t * kc, const uint8_t * vc,
                                                           __nv_bfloat16 * y, float * part, const int * dseq0, int nq,
                                                           int heads, int kvheads, float scale, int nsplit,
                                                           const int * dep, int pstride, int zoff, const int * dstart,
                                                           int cstride, const float * kbias, int flat, int poff,
                                                           const int32_t * sel = nullptr, int from = 0) {
    constexpr int SEL_KEYS = 2048;  // 512 blocks of 4
    if constexpr (SEL) {
        const int z = blockIdx.y;
        if (tq_split_point(*dseq0 - 1 + (dep ? dep[z] : z)) < from) return;
        q += int64_t(z) * heads * tq3a::DIM, part += int64_t(z) * heads * pstride * (tq3a::DIM + 2), sel += int64_t(z) * (SEL_KEYS / 4);
        nq = 1, dep = nullptr, zoff = 0;
    }
    // kbias (TqCache stand-ins): natural-log weight per key and KV head added to the logit; flat: every query sees the
    // keys [0, *dseq0) (no causal step per column); poff: partial slot offset (with cstride)
    using namespace tq3a;
    constexpr int ROWS = NW * 16, NT = NW * 32, KRB = KC::RB, VRB = VC::RB, UPT = TILE * 8 / NT;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    // trellis codebooks staged in shared memory (the per-value lookups are data dependent gathers)
    constexpr bool KT = std::is_same_v<KC, kvc::Tcq3>, VT = std::is_same_v<VC, kvc::Tcq2>;
    __shared__ float s_cbk[KT ? 128 : 1], s_cbv[VT ? 128 : 1];  // s_cbk: 512 int8 levels; s_cbv: 256 FP16 values
    if constexpr (KT)
        for (int i = threadIdx.x; i < 512; i += NW * 32)
            reinterpret_cast<int8_t *>(s_cbk)[i] = int8_t(__float2int_rn(tcq::CB3K[i] * tcq::CB3K_SCALE));
    if constexpr (VT)
        for (int i = threadIdx.x; i < 256; i += NW * 32) reinterpret_cast<__half *>(s_cbv)[i] = __float2half_rn(tcq::CB2V[i]);
    if constexpr (KT || VT) __syncthreads();
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);
    int8_t * k8 = q8 + ROWS * QS;
    __half * vh = reinterpret_cast<__half *>(k8 + NB * TILE * KS);
    float * ksf = reinterpret_cast<float *>(vh + NB * TILE * VS);  // [NB][8 blocks][TILE]
    float * qsc = ksf + NB * 8 * TILE;                             // [ROWS][2]
    const int kh = blockIdx.x, split = blockIdx.z, t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g;
    const int r0 = SEL ? 0 : (gridDim.y - 1 - blockIdx.y) * ROWS;
    const int seq0 = *dseq0;
    const int q_lo = r0 / g, q_hi = min(r0 + ROWS - 1, rows - 1) / g;
    // pstride > 0 (two-part tree verify): query z at depth dep[z] (zoff + z for a chain) takes the keys below
    // tq_split_point(qpos) and writes a partial with stride pstride; attn_blk_codec_k adds the rest and merges
    auto qlim = [&](int z) { return SEL ? SEL_KEYS : pstride ? tq_split_point(seq0 - 1 + (dep ? dep[z] : zoff + z)) : flat ? seq0 : seq0 + z; };
    int key_end = qlim(q_hi), mask_from = qlim(q_lo);
    if (pstride)
        for (int z = q_lo; z <= q_hi; ++z) key_end = max(key_end, qlim(z)), mask_from = min(mask_from, qlim(z));
    // tiles interleaved over the splits (tile i -> split i % nsplit): a query's result does not depend on nq
    // dstart (TqCache): keys below it are served by a compact stand-in segment (attn_compact_k); multiple of TILE
    const int stride = nsplit * TILE, kbeg = (dstart ? *dstart : 0) + split * TILE, kend = key_end;
    const int ntile = kbeg < kend ? (kend - kbeg + stride - 1) / stride : 0;

    // Q -> TQ3_0 rotation per 32 (signs, WHT / sqrt(32)) -> int8 per (row, 128)
    for (int pr = warp; pr < ROWS * 2; pr += NW) {
        const int r = pr / 2, h = pr % 2, rr = min(r0 + r, rows - 1);
        const __nv_bfloat16 * qr = q + (int64_t(rr / g) * heads + kh * g + rr % g) * DIM + h * 128 + lane * 4;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] = __bfloat162float(qr[j]) * (KC::ROT128 ? tcq::S1[lane * 4 + j] : SIGNS[(lane * 4 + j) % 32]);
        float a = v[0], b = v[1];
        v[0] = a + b, v[1] = a - b;
        a = v[2], b = v[3];
        v[2] = a + b, v[3] = a - b;
        a = v[0], b = v[2];
        v[0] = a + b, v[2] = a - b;
        a = v[1], b = v[3];
        v[1] = a + b, v[3] = a - b;
#pragma unroll
        for (int m = 1; m < (KC::ROT128 ? 32 : 8); m <<= 1)  // 8 lanes = one 32-value block (32 lanes: the 128-point WHT)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
                v[j] = (lane & m) ? other - v[j] : v[j] + other;
            }
        float mx = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] *= KC::ROT128 ? tcq::INV_SQRT_128 * tcq::S2[lane * 4 + j] : 0.17677669529663687f;
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

    // loader: unit = (key u % 32, 32-dim block u / 32); key-major lanes keep the 16-byte stores conflict-free
    uint32_t kw[UPT][KC::NW], vw[UPT][VC::NW];
    __half kr[UPT], vr[UPT];
    auto load = [&](int t0) {
#pragma unroll
        for (int i = 0; i < UPT; ++i) {
            const int u = t + i * NT, lk = u % 32, b = u / 32;
            int p = t0 + lk;
            if (p < kend) {
                if constexpr (SEL) p = sel[p / 4] * 4 + p % 4;
                const uint8_t * rk = kc + (int64_t(p) * kvheads + kh) * KRB, * rv = vc + (int64_t(p) * kvheads + kh) * VRB;
                KC::load(rk, b, kw[i], kr[i]);
                VC::load(rv, b, vw[i], vr[i]);
            } else {
#pragma unroll
                for (int j = 0; j < KC::NW; ++j) kw[i][j] = 0u;
#pragma unroll
                for (int j = 0; j < VC::NW; ++j) vw[i][j] = 0u;
                kr[i] = vr[i] = __float2half(0.f);
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < UPT; ++i) {
            const int u = t + i * NT, lk = u % 32, b = u / 32;
            KC::store_k(kw[i], kr[i], k8 + (buf * TILE + lk) * KS + b * 32, ksf[(buf * 8 + b) * TILE + lk], s_cbk);
            VC::store_v(vw[i], vr[i], vh + (buf * TILE + lk) * VS + b * 32, s_cbv);
        }
    };

    if (ntile > 0) {
        load(kbeg);
        store(0);
        if (ntile > 1) load(kbeg + stride);
    }
    __syncthreads();

    const int ra = warp * 16 + gid, rb = ra + 8;
    const float qa0 = qsc[ra * 2], qa1 = qsc[ra * 2 + 1], qb0 = qsc[rb * 2], qb1 = qsc[rb * 2 + 1];
    const int lim_a = qlim(min(r0 + ra, rows - 1) / g), lim_b = qlim(min(r0 + rb, rows - 1) / g);
    const float sl2 = scale * 1.4426950408889634f;
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
    const int8_t * qrow = q8 + (warp * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * QS + (lane / 16) * 16;

    for (int u = 0; u < ntile; ++u) {
        const int buf = NB == 2 ? (u & 1) : 0, t0 = kbeg + u * stride;
        const int8_t * kt = k8 + buf * TILE * KS;
        const __half * vt = vh + buf * TILE * VS;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int bp = 0; bp < 4; ++bp) {  // blocks 2 bp, 2 bp + 1
            const int b0 = 2 * bp;
            uint32_t a0[4], a1[4];
            ldsm_x4(a0, qrow + b0 * 32);
            ldsm_x4(a1, qrow + (b0 + 1) * 32);
            const float qa = b0 < 4 ? qa0 : qa1, qb = b0 < 4 ? qb0 : qb1;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                uint32_t kb[4];
                ldsm_x4(kb, kt + (nt * 8 + lane % 8) * KS + b0 * 32 + (lane / 8) * 16);
                int c0[4] = {}, c1[4] = {};
                mma_s8(c0, a0[0], a0[1], a0[2], a0[3], kb[0], kb[1]);
                mma_s8(c1, a1[0], a1[1], a1[2], a1[3], kb[2], kb[3]);
                const float2 k0 = *reinterpret_cast<const float2 *>(ksf + (buf * 8 + b0) * TILE + nt * 8 + 2 * tig);
                const float2 k1 = *reinterpret_cast<const float2 *>(ksf + (buf * 8 + b0 + 1) * TILE + nt * 8 + 2 * tig);
                s[nt][0] += qa * (float(c0[0]) * k0.x + float(c1[0]) * k1.x);
                s[nt][1] += qa * (float(c0[1]) * k0.y + float(c1[1]) * k1.y);
                s[nt][2] += qb * (float(c0[2]) * k0.x + float(c1[2]) * k1.x);
                s[nt][3] += qb * (float(c0[3]) * k0.y + float(c1[3]) * k1.y);
            }
        }
        const bool msk = t0 + TILE > mask_from;
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float x = s[nt][i] * sl2;
                const int key = t0 + nt * 8 + 2 * tig + (i & 1);
                if (kbias && key < kend) x += kbias[int64_t(key) * kvheads + kh] * 1.4426950408889634f;
                if (key >= kend || (msk && key >= (i < 2 ? lim_a : lim_b))) x = -CUDART_INF_F;
                s[nt][i] = x;
                if (i < 2) mxa = fmaxf(mxa, x);
                else mxb = fmaxf(mxb, x);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);
        const float ra_ = na > -CUDART_INF_F ? na : 0.f, rb_ = nb > -CUDART_INF_F ? nb : 0.f;  // rows with no key yet
        const float al = exp2f(m_a - ra_), bl = exp2f(m_b - rb_);
        m_a = na, m_b = nb;
        uint32_t pf[2][4];
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = exp2f(s[nt][0] - ra_), p1 = exp2f(s[nt][1] - ra_);
            const float p2 = exp2f(s[nt][2] - rb_), p3 = exp2f(s[nt][3] - rb_);
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
        if constexpr (NB == 2) {
            if (u + 1 < ntile) store(buf ^ 1);
        } else if (u + 1 < ntile) {
            __syncthreads();  // every warp is done with the tile before it is overwritten
            store(0);
        }
        if (u + 2 < ntile) load(t0 + 2 * stride);
        __syncthreads();
    }
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const int ga = r0 + ra, gb = r0 + rb;
    const int64_t ha = int64_t(ga / g) * heads + kh * g + ga % g, hb = int64_t(gb / g) * heads + kh * g + gb % g;
    if (nsplit == 1 && !pstride && !cstride) {
        const float ia = l_a > 0.f ? 1.f / l_a : 0.f, ib = l_b > 0.f ? 1.f / l_b : 0.f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            if (ga < rows) *reinterpret_cast<__nv_bfloat162 *>(y + ha * DIM + 8 * j + 2 * tig) = __floats2bfloat162_rn(o[j][0] * ia, o[j][1] * ia);
            if (gb < rows) *reinterpret_cast<__nv_bfloat162 *>(y + hb * DIM + 8 * j + 2 * tig) = __floats2bfloat162_rn(o[j][2] * ib, o[j][3] * ib);
        }
        return;
    }
    const int ps = pstride ? pstride : cstride ? cstride : nsplit;
    float * pa = part + (ha * ps + split + poff) * (DIM + 2), * pb = part + (hb * ps + split + poff) * (DIM + 2);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        if (ga < rows) pa[8 * j + 2 * tig] = o[j][0], pa[8 * j + 2 * tig + 1] = o[j][1];
        if (gb < rows) pb[8 * j + 2 * tig] = o[j][2], pb[8 * j + 2 * tig + 1] = o[j][3];
    }
    if (tig == 0) {
        constexpr float LN2 = 0.6931471805599453f;
        if (ga < rows) pa[DIM] = m_a * LN2, pa[DIM + 1] = l_a;
        if (gb < rows) pb[DIM] = m_b * LN2, pb[DIM + 1] = l_b;
    }
}

// ---- Verify cache part for q8 K with q4 / q8 V (two-part tree verify, pstride > 0) ----
// Same per-row arithmetic as attn_tq3kv_k (same tiles per split in the same order, same score, softmax and PV
// expressions), so the partials are the same bit for bit; only the work is laid out differently:
// - the raw K / V records stream through a cp.async ring of S tiles into shared memory; the q8 K bytes are the int8 MMA
//   operand as stored, and V is turned into the FP16 MMA operand in registers by the warp that uses it;
// - warp (r, dq) of RG x DQ: row group r (16 (query, head) rows), value slice dq (256 / DQ values). The score tile is
//   split over the DQ warps of a row group by key octets and exchanged through shared memory, the PV product by values,
//   so a single query keeps 4 warps busy instead of 1;
// - the PV columns are permuted so a lane's values are one 32-value block of the record: n-tile j, column c = value
//   32 c + j;
// - one launch holds up to RG x 16 rows: a 16-column verify (96 rows at 6 heads per KV head) reads the cache once.
namespace vqa {
constexpr int TILE = 32, KRB = int(sizeof(Q8KV256)), QS = tq3a::QS, SST = TILE + 8;  // SST: score row stride (floats)
template <class VC> __host__ __device__ constexpr int stage_bytes() { return TILE * (KRB + VC::RB); }
constexpr int VS = tq3a::VS;  // decoded V tile row stride (FP16 values), as attn_tq3kv_k
template <class VC, int S, bool SV> size_t smem(int rg) {
    return size_t(rg) * 16 * QS + size_t(rg) * 16 * 2 * 4 + size_t(rg) * 16 * SST * 4 + size_t(S) * stage_bytes<VC>() +
           (SV ? size_t(TILE) * VS * 2 : 0);
}
// a record's 32-value block from shared memory, in the codec's load() register layout
template <class VC> __device__ __forceinline__ void load_smem(const uint8_t * r, int b, uint32_t (&w)[VC::NW], __half & sc) {
    const uint4 * p = reinterpret_cast<const uint4 *>(r + 16 + (VC::NW * 4) * b);
#pragma unroll
    for (int i = 0; i < VC::NW / 4; ++i) {
        const uint4 a = p[i];
        w[4 * i] = a.x, w[4 * i + 1] = a.y, w[4 * i + 2] = a.z, w[4 * i + 3] = a.w;
    }
    sc = reinterpret_cast<const __half *>(r)[b];
}
__device__ __forceinline__ void cp16(void * dst, const void * src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(valid ? 16 : 0));
}
// V operand of the PV MMA for keys (k, k + 1) and 8 consecutive values: b[i] = {V[k][v0 + i], V[k + 1][v0 + i]} as FP16,
// with the arithmetic of tq3a::store_levels_h (integer level exact in FP16, then times the key's FP16 scale).
template <class VC> struct VReg;
template <> struct VReg<kvc::Q4> {
    static constexpr int WPJ = 1;  // 32-bit words per key and 8 values
    __device__ static void words(const uint8_t * rec, int blk, int jc, uint32_t (&w)[WPJ]) {
        w[0] = *reinterpret_cast<const uint32_t *>(rec + 16 + 16 * blk + 4 * jc);
    }
    __device__ static void pair(const uint32_t (&x0)[WPJ], const uint32_t (&x1)[WPJ], uint32_t s2, uint32_t (&b)[8]) {
        const uint32_t NEG = 0xE408E408u;  // -1032: (1024 + q) - 1032 = q - 8
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const uint32_t t = __byte_perm(x0[0], x1[0], hh ? 0x7362 : 0x5140);  // bytes k, k+1 of values 4hh .. 4hh + 3
            const uint32_t e = t & 0x0F0F0F0Fu, o = (t >> 4) & 0x0F0F0F0Fu;
            b[4 * hh + 0] = hmul2u(hadd2u(__byte_perm(e, 0x64646464u, 0x5140), NEG), s2);
            b[4 * hh + 1] = hmul2u(hadd2u(__byte_perm(o, 0x64646464u, 0x5140), NEG), s2);
            b[4 * hh + 2] = hmul2u(hadd2u(__byte_perm(e, 0x64646464u, 0x5342), NEG), s2);
            b[4 * hh + 3] = hmul2u(hadd2u(__byte_perm(o, 0x64646464u, 0x5342), NEG), s2);
        }
    }
};
template <> struct VReg<kvc::Q8> {
    static constexpr int WPJ = 2;
    __device__ static void words(const uint8_t * rec, int blk, int jc, uint32_t (&w)[WPJ]) {
        const uint2 u = *reinterpret_cast<const uint2 *>(rec + 16 + 32 * blk + 8 * jc);
        w[0] = u.x, w[1] = u.y;
    }
    __device__ static void pair(const uint32_t (&x0)[WPJ], const uint32_t (&x1)[WPJ], uint32_t s2, uint32_t (&b)[8]) {
        const uint32_t NEG = 0xE480E480u;  // -1152
#pragma unroll
        for (int wi = 0; wi < 2; ++wi) {
            const uint32_t la = x0[wi] ^ 0x80808080u, lb = x1[wi] ^ 0x80808080u;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const uint32_t t = __byte_perm(la, lb, hh ? 0x7362 : 0x5140);
                b[4 * wi + 2 * hh + 0] = hmul2u(hadd2u(__byte_perm(t, 0x64646464u, 0x5140), NEG), s2);
                b[4 * wi + 2 * hh + 1] = hmul2u(hadd2u(__byte_perm(t, 0x64646464u, 0x5342), NEG), s2);
            }
        }
    }
};
}  // namespace vqa

template <int RG, int DQ, class VC, int S, bool SV>
__global__ void __launch_bounds__(RG * DQ * 32, RG * DQ <= 6 ? 2 : 1) attn_vq_k(const __nv_bfloat16 * q, const uint8_t * kc,
                                                                              const uint8_t * vc, float * part,
                                                                              const int * dseq0, int nq, int heads,
                                                                              int kvheads, float scale, int nsplit,
                                                                              const int * dep, int pstride, int zoff,
                                                                              const int * dstart) {
    using namespace tq3a;
    using vqa::KRB;
    using vqa::SST;
    constexpr int NW = RG * DQ, VRB = VC::RB, NT = NW * 32, KCH = KRB / 16, VCH = VRB / 16, CH = TILE * (KCH + VCH);
    constexpr int SB = vqa::stage_bytes<VC>(), NTW = 4 / DQ, JW = 4 / DQ;  // key octets and 8-value chunks per warp
    static_assert(DQ == 1 || DQ == 2 || DQ == 4, "DQ");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int kh = blockIdx.x, split = blockIdx.z, t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int wr = warp / DQ, dq = warp % DQ;
    const int g = heads / kvheads, rows = nq * g, rg = (rows + 15) / 16;
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);
    float * qsc = reinterpret_cast<float *>(q8 + rg * 16 * QS);
    float * ssc = qsc + rg * 16 * 2;  // [rows][SST] masked, scaled scores of the current tile
    uint8_t * ring = reinterpret_cast<uint8_t *>(ssc + rg * 16 * SST);
    __half * vh = reinterpret_cast<__half *>(ring + S * SB);  // SV: the tile's V decoded once for every row group
    const int seq0 = *dseq0;
    auto qlim = [&](int z) { return tq_split_point(seq0 - 1 + (dep ? dep[z] : zoff + z)); };
    int key_end = qlim(nq - 1), mask_from = qlim(0);
    for (int z = 0; z < nq; ++z) key_end = max(key_end, qlim(z)), mask_from = min(mask_from, qlim(z));
    const int stride = nsplit * TILE, kbeg = (dstart ? *dstart : 0) + split * TILE, kend = key_end;
    const int ntile = kbeg < kend ? (kend - kbeg + stride - 1) / stride : 0;

    // tile u -> stage st: K records [TILE][KRB], then V records [TILE][VRB], copied as stored (keys past kend zero-filled)
    auto issue = [&](int u, int st) {
        const int t0 = kbeg + u * stride;
        uint8_t * kd = ring + st * SB, * vd = kd + TILE * KRB;
        for (int c = t; c < CH; c += NT) {
            const bool isk = c < TILE * KCH;
            const int cc = isk ? c : c - TILE * KCH, key = isk ? cc / KCH : cc / VCH, off = 16 * (isk ? cc % KCH : cc % VCH);
            const int p = t0 + key;
            const bool valid = p < kend;
            const int64_t rec = int64_t(valid ? p : 0) * kvheads + kh;
            const uint8_t * src = isk ? kc + rec * KRB + off : vc + rec * VRB + off;
            vqa::cp16((isk ? kd + key * KRB : vd + key * VRB) + off, src, valid);
        }
    };
#pragma unroll
    for (int st = 0; st < S - 1; ++st) {
        if (st < ntile) issue(st, st);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // Q -> TQ3_0 rotation per 32 (signs, WHT / sqrt(32)) -> int8 per (row, 128); as attn_tq3kv_k (q8 K: ROT128 false)
    for (int pr = warp; pr < rg * 16 * 2; pr += NW) {
        const int r = pr / 2, h = pr % 2, rr = min(r, rows - 1);
        const __nv_bfloat16 * qr = q + (int64_t(rr / g) * heads + kh * g + rr % g) * DIM + h * 128 + lane * 4;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(qr[j]) * SIGNS[(lane * 4 + j) % 32];
        float a = v[0], b = v[1];
        v[0] = a + b, v[1] = a - b;
        a = v[2], b = v[3];
        v[2] = a + b, v[3] = a - b;
        a = v[0], b = v[2];
        v[0] = a + b, v[2] = a - b;
        a = v[1], b = v[3];
        v[1] = a + b, v[3] = a - b;
#pragma unroll
        for (int m = 1; m < 8; m <<= 1)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
                v[j] = (lane & m) ? other - v[j] : v[j] + other;
            }
        float mx = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] *= 0.17677669529663687f;
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
    __syncthreads();

    const bool active = wr < rg;
    const int wrc = min(wr, rg - 1);
    const int ra = wrc * 16 + gid, rb = ra + 8;
    const float qa0 = qsc[ra * 2], qa1 = qsc[ra * 2 + 1], qb0 = qsc[rb * 2], qb1 = qsc[rb * 2 + 1];
    const int lim_a = qlim(min(wr * 16 + gid, rows - 1) / g), lim_b = qlim(min(wr * 16 + gid + 8, rows - 1) / g);
    const float sl2 = scale * 1.4426950408889634f;
    constexpr int NJ = 32 / DQ;  // n-tiles (of 8 values) per warp: j = dq * NJ + jl
    float o[NJ][4];
#pragma unroll
    for (int j = 0; j < NJ; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
    const int8_t * qrow = q8 + (wrc * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * QS + (lane / 16) * 16;
    float * sra = ssc + ra * SST, * srb = ssc + rb * SST;

    for (int u = 0; u < ntile; ++u) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2));
        __syncthreads();  // tile u is in its stage for every thread; stage (u - 1) % S and the score tile are free
        if (u + S - 1 < ntile) issue(u + S - 1, (u + S - 1) % S);
        asm volatile("cp.async.commit_group;\n" ::);
        const int t0 = kbeg + u * stride;
        const uint8_t * kt = ring + (u % S) * SB, * vt = kt + TILE * KRB;
        const bool msk = t0 + TILE > mask_from;
        if constexpr (SV) {  // unit = (key un % 32, 32-value block un / 32), as attn_tq3kv_k's store
            for (int un = t; un < TILE * 8; un += NT) {
                const int lk = un % 32, b = un / 32;
                uint32_t w[VC::NW];
                __half sc;
                vqa::load_smem<VC>(vt + lk * VRB, b, w, sc);
                VC::store_v(w, sc, vh + lk * vqa::VS + b * 32, nullptr);
            }
        }
        if (active) {  // scores of key octets dq * NTW .. + NTW - 1
            float s[NTW][4];
#pragma unroll
            for (int n = 0; n < NTW; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int bp = 0; bp < 4; ++bp) {  // blocks 2 bp, 2 bp + 1
                const int b0 = 2 * bp;
                uint32_t a0[4], a1[4];
                ldsm_x4(a0, qrow + b0 * 32);
                ldsm_x4(a1, qrow + (b0 + 1) * 32);
                const float qa = b0 < 4 ? qa0 : qa1, qb = b0 < 4 ? qb0 : qb1;
#pragma unroll
                for (int n = 0; n < NTW; ++n) {
                    const int nt = dq * NTW + n;
                    uint32_t kb[4];
                    ldsm_x4(kb, kt + (nt * 8 + lane % 8) * KRB + 16 + b0 * 32 + (lane / 8) * 16);
                    int c0[4] = {}, c1[4] = {};
                    mma_s8(c0, a0[0], a0[1], a0[2], a0[3], kb[0], kb[1]);
                    mma_s8(c1, a1[0], a1[1], a1[2], a1[3], kb[2], kb[3]);
                    const __half * h0 = reinterpret_cast<const __half *>(kt + (nt * 8 + 2 * tig) * KRB);
                    const __half * h1 = reinterpret_cast<const __half *>(kt + (nt * 8 + 2 * tig + 1) * KRB);
                    const float2 k0 = make_float2(__half2float(h0[b0]), __half2float(h1[b0]));
                    const float2 k1 = make_float2(__half2float(h0[b0 + 1]), __half2float(h1[b0 + 1]));
                    s[n][0] += qa * (float(c0[0]) * k0.x + float(c1[0]) * k1.x);
                    s[n][1] += qa * (float(c0[1]) * k0.y + float(c1[1]) * k1.y);
                    s[n][2] += qb * (float(c0[2]) * k0.x + float(c1[2]) * k1.x);
                    s[n][3] += qb * (float(c0[3]) * k0.y + float(c1[3]) * k1.y);
                }
            }
#pragma unroll
            for (int n = 0; n < NTW; ++n) {
                const int nt = dq * NTW + n;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float x = s[n][i] * sl2;
                    const int key = t0 + nt * 8 + 2 * tig + (i & 1);
                    if (key >= kend || (msk && key >= (i < 2 ? lim_a : lim_b))) x = -CUDART_INF_F;
                    s[n][i] = x;
                }
                *reinterpret_cast<float2 *>(sra + nt * 8 + 2 * tig) = make_float2(s[n][0], s[n][1]);
                *reinterpret_cast<float2 *>(srb + nt * 8 + 2 * tig) = make_float2(s[n][2], s[n][3]);
            }
        }
        if constexpr (DQ > 1 || SV) __syncthreads();  // the row group's whole score tile (and the decoded V)
        else __syncwarp();
        if (!active) continue;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float2 xa = *reinterpret_cast<const float2 *>(sra + nt * 8 + 2 * tig);
            const float2 xb = *reinterpret_cast<const float2 *>(srb + nt * 8 + 2 * tig);
            s[nt][0] = xa.x, s[nt][1] = xa.y, s[nt][2] = xb.x, s[nt][3] = xb.y;
        }
        float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (i < 2) mxa = fmaxf(mxa, s[nt][i]);
                else mxb = fmaxf(mxb, s[nt][i]);
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);
        const float ra_ = na > -CUDART_INF_F ? na : 0.f, rb_ = nb > -CUDART_INF_F ? nb : 0.f;  // rows with no key yet
        const float al = exp2f(m_a - ra_), bl = exp2f(m_b - rb_);
        m_a = na, m_b = nb;
        uint32_t pf[2][4];
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p0 = exp2f(s[nt][0] - ra_), p1 = exp2f(s[nt][1] - ra_);
            const float p2 = exp2f(s[nt][2] - rb_), p3 = exp2f(s[nt][3] - rb_);
            sa += p0 + p1, sb += p2 + p3;
            pf[nt / 2][(nt & 1) * 2] = pack_h2(p0, p1);
            pf[nt / 2][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
        }
        l_a = l_a * al + sa, l_b = l_b * bl + sb;
        if (__any_sync(0xffffffffu, al != 1.f || bl != 1.f))
#pragma unroll
            for (int j = 0; j < NJ; ++j) o[j][0] *= al, o[j][1] *= al, o[j][2] *= bl, o[j][3] *= bl;
        if constexpr (SV) {  // PV as attn_tq3kv_k, n-tile pairs np of this warp's values
            const __half * vrow = vh + ((lane / 8) & 1) * 8 * vqa::VS + (lane % 8) * vqa::VS + (lane / 16) * 8;
#pragma unroll
            for (int npl = 0; npl < NJ / 2; ++npl) {
                const int np = dq * (NJ / 2) + npl;
                uint32_t acc[2][2] = {};
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    uint32_t vb[4];
                    ldsm_x4_t(vb, vrow + kk * 16 * vqa::VS + np * 16);
                    mma_f16acc(acc[0], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[0], vb[1]);
                    mma_f16acc(acc[1], pf[kk][0], pf[kk][1], pf[kk][2], pf[kk][3], vb[2], vb[3]);
                }
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[n][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[n][1]));
                    o[2 * npl + n][0] += lo.x, o[2 * npl + n][1] += lo.y, o[2 * npl + n][2] += hi.x, o[2 * npl + n][3] += hi.y;
                }
            }
            continue;
        }
        // keys of the B fragments: kk 0 -> (2 tig, +1) and (2 tig + 8, +9); kk 1 -> the same + 16
        uint32_t s2[4];
        const uint8_t * vk[4];
#pragma unroll
        for (int pi = 0; pi < 4; ++pi) {
            const int k = (pi >> 1) * 16 + (pi & 1) * 8 + 2 * tig;
            vk[pi] = vt + k * VRB;
            const __half2 sp = __halves2half2(reinterpret_cast<const __half *>(vk[pi])[gid], reinterpret_cast<const __half *>(vk[pi] + VRB)[gid]);
            s2[pi] = *reinterpret_cast<const uint32_t *>(&sp);
        }
#pragma unroll
        for (int jw = 0; jw < JW; ++jw) {
            const int jc = dq * JW + jw;
            uint32_t bv[4][8];
#pragma unroll
            for (int pi = 0; pi < 4; ++pi) {
                uint32_t w0[vqa::VReg<VC>::WPJ], w1[vqa::VReg<VC>::WPJ];
                vqa::VReg<VC>::words(vk[pi], gid, jc, w0);
                vqa::VReg<VC>::words(vk[pi] + VRB, gid, jc, w1);
                vqa::VReg<VC>::pair(w0, w1, s2[pi], bv[pi]);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int jl = jw * 8 + i;
                uint32_t acc[2] = {};
                mma_f16acc(acc, pf[0][0], pf[0][1], pf[0][2], pf[0][3], bv[0][i], bv[1][i]);
                mma_f16acc(acc, pf[1][0], pf[1][1], pf[1][2], pf[1][3], bv[2][i], bv[3][i]);
                const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[0]));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[1]));
                o[jl][0] += lo.x, o[jl][1] += lo.y, o[jl][2] += hi.x, o[jl][3] += hi.y;
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!active) return;
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const int ga = wr * 16 + gid, gb = ga + 8;
    const int64_t ha = int64_t(ga / g) * heads + kh * g + ga % g, hb = int64_t(gb / g) * heads + kh * g + gb % g;
    float * pa = part + (ha * pstride + split) * (DIM + 2), * pb = part + (hb * pstride + split) * (DIM + 2);
#pragma unroll
    for (int jl = 0; jl < NJ; ++jl) {  // n-tile j, columns 2 tig, 2 tig + 1 -> values 64 tig + j, 64 tig + 32 + j
        const int j = dq * NJ + jl, c0 = SV ? 8 * j + 2 * tig : 64 * tig + j, c1 = SV ? c0 + 1 : c0 + 32;  // SV: natural order
        if (ga < rows) pa[c0] = o[jl][0], pa[c1] = o[jl][1];
        if (gb < rows) pb[c0] = o[jl][2], pb[c1] = o[jl][3];
    }
    if (tig == 0 && dq == 0) {
        constexpr float LN2 = 0.6931471805599453f;
        if (ga < rows) pa[DIM] = m_a * LN2, pa[DIM + 1] = l_a;
        if (gb < rows) pb[DIM] = m_b * LN2, pb[DIM + 1] = l_b;
    }
}

// cache part of the two-part verify (partials only) for nq queries in one launch of <= RG x 16 rows
template <int RG, int DQ, class VC, int S, bool SV>
static void vq_launch(const __nv_bfloat16 * q, const uint8_t * kc, const uint8_t * vc, float * part, const int * dseq0,
                      int nq, int h, int kh, float scale, int nsplit, cudaStream_t s, const int * dep, int pstride, int zoff,
                      const int * dstart) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_vq_k<RG, DQ, VC, S, SV>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(vqa::smem<VC, S, SV>(RG)));
        configured = true;
    }
    const int rg = ((h / kh) * nq + 15) / 16;
    attn_vq_k<RG, DQ, VC, S, SV><<<dim3(kh, 1, nsplit), RG * DQ * 32, vqa::smem<VC, S, SV>(rg), s>>>(
        q, kc, vc, part, dseq0, nq, h, kh, scale, nsplit, dep, pstride, zoff, dstart);
}

// ---- Verify cache part for q8 K, 33-48 rows (7-8 queries at 6 heads per KV head: the DFlash2 tree verify), split by role ----
// Same per-row arithmetic as attn_tq3kv_k (same tiles per split in the same order, same score, softmax and PV
// expressions), so the partials are the same bit for bit. Roles:
// - three score warps (one per 16-row group, Q fragments in registers) stream the raw K records through their own
//   cp.async ring, compute the tile's scores, the online softmax and the FP16 P fragments, and hand P and the row
//   factors to the PV warps through a double-buffered shared tile (named barriers FULL / EMPTY per buffer);
// - four PV warps (64 values each, all three row groups, so a V fragment is decoded once per block and used three
//   times) stream the raw V records through a second ring, decode the tile's V fragments while the score warps work,
//   then rescale and accumulate P V. Column order as attn_vq_k's register path (n-tile j, column c = value 32 c + j).
// The score warps run up to two tiles ahead of the PV warps; no 64-row padding (a quarter of the old tile's work).
// One block per SM (84-85 KB of shared memory) takes two (KV head, split) items back to back through the same rings.
// 3090, 8 columns at 140K: 567 -> 394 us per layer with the old tile at 2 x 4 warps per SM (see the PR notes).
namespace v48 {
constexpr int RG = 3, QKW = RG, PVW = 4, NPB = 2, NI = 2;  // NI: work items per block
constexpr int BAR_K = 1, BAR_V = 2, BAR_FULL = 3, BAR_EMPTY = 3 + NPB;  // + buffer index
template <class VC, int S> constexpr size_t smem() {  // q4 V: 85 KB at S 4; q8 V: 84 KB at S 3 (99 KB per block)
    // Q [NI][48][QS] | qsc [NI][48][2] | K ring [S][TILE][KRB] | V ring [S][TILE][VRB] | P [NPB][RG][2][32] uint4 | factors [NPB][RG][16]
    return size_t(NI) * RG * 16 * tq3a::QS + size_t(NI) * RG * 16 * 2 * 4 + size_t(S) * vqa::TILE * (vqa::KRB + VC::RB) +
           size_t(NPB) * RG * 2 * 32 * 16 + size_t(NPB) * RG * 16 * 4;
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
}  // namespace v48

template <class VC, int S>
__global__ void __launch_bounds__((v48::QKW + v48::PVW) * 32, 1) attn_v48_k(const __nv_bfloat16 * q, const uint8_t * kc,
                                                                      const uint8_t * vc, float * part, const int * dseq0,
                                                                      int nq, int heads, int kvheads, float scale,
                                                                      int nsplit, const int * dep, int pstride, int zoff,
                                                                      const int * dstart) {
    using namespace tq3a;
    using namespace v48;
    using vqa::KRB;
    constexpr int VRB = VC::RB, KCH = KRB / 16, VCH = VRB / 16, NW = QKW + PVW, NTA = NW * 32, NTQ = QKW * 32,
                  NTV = PVW * 32;
    constexpr int NJ = 32 / PVW;  // n-tiles (of 8 values) per PV warp: j = pw * NJ + jl, one 8-value chunk of the block
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32, gid = lane / 4, tig = lane % 4;
    const int g = heads / kvheads, rows = nq * g;
    int8_t * q8 = reinterpret_cast<int8_t *>(smem_raw);                 // [NI][48][QS]
    float * qsc = reinterpret_cast<float *>(q8 + NI * RG * 16 * QS);   // [NI][48][2]
    uint8_t * kring = reinterpret_cast<uint8_t *>(qsc + NI * RG * 16 * 2);
    uint8_t * vring = kring + S * TILE * KRB;
    uint4 * pbuf = reinterpret_cast<uint4 *>(vring + S * TILE * VRB);  // [NPB][RG][2][32]
    float * fbuf = reinterpret_cast<float *>(pbuf + NPB * RG * 2 * 32);  // [NPB][RG][16]
    const int seq0 = *dseq0;
    auto qlim = [&](int z) { return tq_split_point(seq0 - 1 + (dep ? dep[z] : zoff + z)); };
    int key_end = qlim(nq - 1), mask_from = qlim(0);
    for (int z = 0; z < nq; ++z) key_end = max(key_end, qlim(z)), mask_from = min(mask_from, qlim(z));
    const int stride = nsplit * TILE, kend = key_end, d0 = dstart ? *dstart : 0;
    // work items (KV head, key split), item = split * kvheads + kh; block b takes items b and b + gridDim.x, one after the
    // other through the same rings, so the second item's loads overlap the end of the first
    const int nitem = kvheads * nsplit, it0 = blockIdx.x, it1 = blockIdx.x + gridDim.x;
    static_assert(NI == 2, "NI");
    const int kh0 = it0 % kvheads, kh1 = it1 % kvheads, sp0 = it0 / kvheads, sp1 = it1 / kvheads;
    const int kb0 = d0 + sp0 * TILE, kb1 = d0 + sp1 * TILE;
    const int nt0 = kb0 < kend ? (kend - kb0 + stride - 1) / stride : 0;
    const int ntot = nt0 + (it1 < nitem && kb1 < kend ? (kend - kb1 + stride - 1) / stride : 0), ni = it1 < nitem ? 2 : 1;
    auto item_kh = [&](int ii) { return ii ? kh1 : kh0; };
    auto item_split = [&](int ii) { return ii ? sp1 : sp0; };
    auto item_kb = [&](int ii) { return ii ? kb1 : kb0; };
    const bool qkw = warp < QKW;

    // virtual tile v (the block's tiles of item 0, then item 1) -> stage st of the role's ring, records copied as stored
    // (keys past kend zero-filled). A thread's chunks are fixed: shared and global offsets precomputed
    constexpr int KPT = (TILE * KCH + NTQ - 1) / NTQ, VPT = (TILE * VCH + NTV - 1) / NTV;
    constexpr int CPT = KPT > VPT ? KPT : VPT;
    int c_sm[CPT], c_gl[CPT];
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
        const int c = qkw ? t + i * NTQ : t - NTQ + i * NTV, nb = qkw ? KCH : VCH, rb = qkw ? KRB : VRB;
        const int key = c / nb, off = 16 * (c % nb);
        c_sm[i] = key * rb + off, c_gl[i] = key * kvheads * rb + off;
    }
    auto issue_k = [&](int v, int st) {
        const int ii = v < nt0 ? 0 : 1, kh = item_kh(ii), t0 = item_kb(ii) + (v - (ii ? nt0 : 0)) * stride;
        uint8_t * d = kring + st * TILE * KRB;
        const uint8_t * g0 = kc + (int64_t(t0) * kvheads + kh) * KRB;
        if (t0 + TILE <= kend) {
#pragma unroll
            for (int i = 0; i < KPT; ++i)
                if (TILE * KCH % NTQ == 0 || i < KPT - 1 || t + i * NTQ < TILE * KCH) vqa::cp16(d + c_sm[i], g0 + c_gl[i], true);
            return;
        }
        for (int c = t; c < TILE * KCH; c += NTQ) {  // the last tile: keys past kend zero-filled
            const int key = c / KCH, off = 16 * (c % KCH), p = t0 + key;
            const bool valid = p < kend;
            vqa::cp16(d + key * KRB + off, kc + (int64_t(valid ? p : 0) * kvheads + kh) * KRB + off, valid);
        }
    };
    auto issue_v = [&](int v, int st) {
        const int ii = v < nt0 ? 0 : 1, kh = item_kh(ii), t0 = item_kb(ii) + (v - (ii ? nt0 : 0)) * stride;
        uint8_t * d = vring + st * TILE * VRB;
        const uint8_t * g0 = vc + (int64_t(t0) * kvheads + kh) * VRB;
        if (t0 + TILE <= kend) {
#pragma unroll
            for (int i = 0; i < VPT; ++i)
                if (TILE * VCH % NTV == 0 || i < VPT - 1 || t - NTQ + i * NTV < TILE * VCH) vqa::cp16(d + c_sm[i], g0 + c_gl[i], true);
            return;
        }
        for (int c = t - NTQ; c < TILE * VCH; c += NTV) {  // the last tile: keys past kend zero-filled
            const int key = c / VCH, off = 16 * (c % VCH), p = t0 + key;
            const bool valid = p < kend;
            vqa::cp16(d + key * VRB + off, vc + (int64_t(valid ? p : 0) * kvheads + kh) * VRB + off, valid);
        }
    };
    auto issue = [&](int v, int st) { if (qkw) issue_k(v, st); else issue_v(v, st); };
#pragma unroll
    for (int st = 0; st < S - 1; ++st) {
        if (st < ntot) issue(st, st);
        asm volatile("cp.async.commit_group;\n" ::);
    }

    // Q -> TQ3_0 rotation per 32 (signs, WHT / sqrt(32)) -> int8 per (row, 128); as attn_tq3kv_k (q8 K: ROT128 false)
    for (int pr = warp; pr < ni * RG * 16 * 2; pr += NW) {
        const int ii = pr / (RG * 16 * 2), r = pr / 2 % (RG * 16), h = pr % 2, rr = min(r, rows - 1);
        const __nv_bfloat16 * qr = q + (int64_t(rr / g) * heads + item_kh(ii) * g + rr % g) * DIM + h * 128 + lane * 4;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(qr[j]) * SIGNS[(lane * 4 + j) % 32];
        float a = v[0], b = v[1];
        v[0] = a + b, v[1] = a - b;
        a = v[2], b = v[3];
        v[2] = a + b, v[3] = a - b;
        a = v[0], b = v[2];
        v[0] = a + b, v[2] = a - b;
        a = v[1], b = v[3];
        v[1] = a + b, v[3] = a - b;
#pragma unroll
        for (int m = 1; m < 8; m <<= 1)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float other = __shfl_xor_sync(0xffffffffu, v[j], m);
                v[j] = (lane & m) ? other - v[j] : v[j] + other;
            }
        float mx = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] *= 0.17677669529663687f;
            mx = fmaxf(mx, fabsf(v[j]));
        }
        for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float sq = mx / 127.f;
        uint32_t pk = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) pk |= (uint32_t(sq > 0.f ? __float2int_rn(v[j] / sq) : 0) & 0xFFu) << (8 * j);
        *reinterpret_cast<uint32_t *>(q8 + (ii * RG * 16 + r) * QS + h * 128 + lane * 4) = pk;
        if (lane == 0) qsc[(ii * RG * 16 + r) * 2 + h] = sq;
    }
    __syncthreads();
    constexpr float LN2 = 0.6931471805599453f;

    if (qkw) {  // ---- score warp: row group `warp`
        const int ra = warp * 16 + gid, rb = ra + 8;
        const int lim_a = qlim(min(ra, rows - 1) / g), lim_b = qlim(min(rb, rows - 1) / g);
        const float sl2 = scale * 1.4426950408889634f;
        int v = 0;
        for (int ii = 0; ii < ni; ++ii) {
            const float * qs = qsc + ii * RG * 16 * 2;
            const float qa0 = qs[ra * 2], qa1 = qs[ra * 2 + 1], qb0 = qs[rb * 2], qb1 = qs[rb * 2 + 1];
            uint32_t qf[8][4];
            const int8_t * qrow = q8 + (ii * RG * 16 + warp * 16 + (lane % 8) + ((lane / 8) & 1) * 8) * QS + (lane / 16) * 16;
#pragma unroll
            for (int b = 0; b < 8; ++b) ldsm_x4(qf[b], qrow + b * 32);
            float m_a = -CUDART_INF_F, m_b = -CUDART_INF_F, l_a = 0.f, l_b = 0.f;
            for (int u = 0; u < (ii ? ntot - nt0 : nt0); ++u, ++v) {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2));
                bar_sync(BAR_K, NTQ);  // tile v in its stage for every score warp; stage (v - 1) % S is free
                if (v + S - 1 < ntot) issue_k(v + S - 1, (v + S - 1) % S);
                asm volatile("cp.async.commit_group;\n" ::);
                const int t0 = item_kb(ii) + u * stride;
                const uint8_t * kt = kring + (v % S) * TILE * KRB;
                // K scales: the 8 block scales of keys nt * 8 + 2 tig, + 1 (record headers)
                uint4 kh0[4], kh1[4];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    kh0[nt] = *reinterpret_cast<const uint4 *>(kt + (nt * 8 + 2 * tig) * KRB);
                    kh1[nt] = *reinterpret_cast<const uint4 *>(kt + (nt * 8 + 2 * tig + 1) * KRB);
                }
                float s[4][4];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
                for (int bp = 0; bp < 4; ++bp) {  // blocks 2 bp, 2 bp + 1
                    const int b0 = 2 * bp;
                    const float qa = b0 < 4 ? qa0 : qa1, qb = b0 < 4 ? qb0 : qb1;
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        uint32_t kb[4];
                        ldsm_x4(kb, kt + (nt * 8 + lane % 8) * KRB + 16 + b0 * 32 + (lane / 8) * 16);
                        int c0[4] = {}, c1[4] = {};
                        mma_s8(c0, qf[b0][0], qf[b0][1], qf[b0][2], qf[b0][3], kb[0], kb[1]);
                        mma_s8(c1, qf[b0 + 1][0], qf[b0 + 1][1], qf[b0 + 1][2], qf[b0 + 1][3], kb[2], kb[3]);
                        const uint32_t w0 = bp == 0 ? kh0[nt].x : bp == 1 ? kh0[nt].y : bp == 2 ? kh0[nt].z : kh0[nt].w;
                        const uint32_t w1 = bp == 0 ? kh1[nt].x : bp == 1 ? kh1[nt].y : bp == 2 ? kh1[nt].z : kh1[nt].w;
                        const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&w0));  // key 2 tig: blocks b0, b0 + 1
                        const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&w1));  // key 2 tig + 1
                        const float2 k0 = make_float2(f0.x, f1.x), k1 = make_float2(f0.y, f1.y);
                        s[nt][0] += qa * (float(c0[0]) * k0.x + float(c1[0]) * k1.x);
                        s[nt][1] += qa * (float(c0[1]) * k0.y + float(c1[1]) * k1.y);
                        s[nt][2] += qb * (float(c0[2]) * k0.x + float(c1[2]) * k1.x);
                        s[nt][3] += qb * (float(c0[3]) * k0.y + float(c1[3]) * k1.y);
                    }
                }
                const bool msk = t0 + TILE > mask_from;
                float mxa = -CUDART_INF_F, mxb = -CUDART_INF_F;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float x = s[nt][i] * sl2;
                        const int key = t0 + nt * 8 + 2 * tig + (i & 1);
                        if (key >= kend || (msk && key >= (i < 2 ? lim_a : lim_b))) x = -CUDART_INF_F;
                        s[nt][i] = x;
                        if (i < 2) mxa = fmaxf(mxa, x);
                        else mxb = fmaxf(mxb, x);
                    }
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
                const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);
                const float ra_ = na > -CUDART_INF_F ? na : 0.f, rb_ = nb > -CUDART_INF_F ? nb : 0.f;  // rows with no key yet
                const float al = exp2f(m_a - ra_), bl = exp2f(m_b - rb_);
                m_a = na, m_b = nb;
                uint32_t pf[2][4];
                float sa = 0.f, sb = 0.f;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const float p0 = exp2f(s[nt][0] - ra_), p1 = exp2f(s[nt][1] - ra_);
                    const float p2 = exp2f(s[nt][2] - rb_), p3 = exp2f(s[nt][3] - rb_);
                    sa += p0 + p1, sb += p2 + p3;
                    pf[nt / 2][(nt & 1) * 2] = pack_h2(p0, p1);
                    pf[nt / 2][(nt & 1) * 2 + 1] = pack_h2(p2, p3);
                }
                l_a = l_a * al + sa, l_b = l_b * bl + sb;
                const int pb = v % NPB;
                if (v >= NPB) bar_sync(BAR_EMPTY + pb, NTA);  // the PV warps have taken tile v - NPB out of buffer pb
                uint4 * pd = pbuf + (pb * RG + warp) * 2 * 32;
                pd[lane] = make_uint4(pf[0][0], pf[0][1], pf[0][2], pf[0][3]);
                pd[32 + lane] = make_uint4(pf[1][0], pf[1][1], pf[1][2], pf[1][3]);
                if (tig == 0) fbuf[(pb * RG + warp) * 16 + gid] = al, fbuf[(pb * RG + warp) * 16 + 8 + gid] = bl;
                bar_arrive(BAR_FULL + pb, NTA);
            }
            l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
            l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
            l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
            l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
            if (tig == 0) {
                const int kh = item_kh(ii), split = item_split(ii);
                const int64_t ha = int64_t(ra / g) * heads + kh * g + ra % g, hb = int64_t(rb / g) * heads + kh * g + rb % g;
                float * pa = part + (ha * pstride + split) * (DIM + 2), * pbp = part + (hb * pstride + split) * (DIM + 2);
                if (ra < rows) pa[DIM] = m_a * LN2, pa[DIM + 1] = l_a;
                if (rb < rows) pbp[DIM] = m_b * LN2, pbp[DIM + 1] = l_b;
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        return;
    }

    // ---- PV warp pw: n-tiles pw * NJ .. + NJ - 1 (values 32 c + j) of all three row groups
    const int pw = warp - QKW, jc = pw;  // 8-value chunk jc of the lane's record block
    auto flush = [&](int ii, const float (&o)[RG][NJ][4]) {
        const int kh = item_kh(ii), split = item_split(ii);
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const int ga = r * 16 + gid, gb = ga + 8;
            const int64_t ha = int64_t(ga / g) * heads + kh * g + ga % g, hb = int64_t(gb / g) * heads + kh * g + gb % g;
            float * pa = part + (ha * pstride + split) * (DIM + 2), * pbp = part + (hb * pstride + split) * (DIM + 2);
#pragma unroll
            for (int jl = 0; jl < NJ; ++jl) {  // n-tile j, columns 2 tig, 2 tig + 1 -> values 64 tig + j, 64 tig + 32 + j
                const int j = pw * NJ + jl, c0 = 64 * tig + j, c1 = c0 + 32;
                if (ga < rows) pa[c0] = o[r][jl][0], pa[c1] = o[r][jl][1];
                if (gb < rows) pbp[c0] = o[r][jl][2], pbp[c1] = o[r][jl][3];
            }
        }
    };
    // one item: its tiles v0 .. v0 + n - 1 of the rings; called twice (two inlined copies keep o in registers)
    auto item = [&](int ii, int v0, int n) {
        float o[RG][NJ][4];
#pragma unroll
        for (int r = 0; r < RG; ++r)
#pragma unroll
            for (int j = 0; j < NJ; ++j) o[r][j][0] = o[r][j][1] = o[r][j][2] = o[r][j][3] = 0.f;
        for (int v = v0; v < v0 + n; ++v) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(S - 2));
            bar_sync(BAR_V, NTV);
            if (v + S - 1 < ntot) issue_v(v + S - 1, (v + S - 1) % S);
            asm volatile("cp.async.commit_group;\n" ::);
            const uint8_t * vt = vring + (v % S) * TILE * VRB;
            // V fragments of this tile: keys kk 0 -> (2 tig, +1) and (2 tig + 8, +9); kk 1 -> the same + 16
            uint32_t bv[4][NJ];
    #pragma unroll
            for (int pi = 0; pi < 4; ++pi) {
                const int k = (pi >> 1) * 16 + (pi & 1) * 8 + 2 * tig;
                const uint8_t * vk = vt + k * VRB;
                const __half2 sp = __halves2half2(reinterpret_cast<const __half *>(vk)[gid], reinterpret_cast<const __half *>(vk + VRB)[gid]);
                const uint32_t s2 = *reinterpret_cast<const uint32_t *>(&sp);
                uint32_t w0[vqa::VReg<VC>::WPJ], w1[vqa::VReg<VC>::WPJ];
                vqa::VReg<VC>::words(vk, gid, jc, w0);
                vqa::VReg<VC>::words(vk + VRB, gid, jc, w1);
                vqa::VReg<VC>::pair(w0, w1, s2, bv[pi]);
            }
            const int pb = v % NPB;
            bar_sync(BAR_FULL + pb, NTA);
    #pragma unroll
            for (int r = 0; r < RG; ++r) {
                const uint4 * ps = pbuf + (pb * RG + r) * 2 * 32;
                const uint4 x0 = ps[lane], x1 = ps[32 + lane];
                const float fa = fbuf[(pb * RG + r) * 16 + gid], fb = fbuf[(pb * RG + r) * 16 + 8 + gid];
                if (r == RG - 1 && v + NPB < ntot) bar_arrive(BAR_EMPTY + pb, NTA);
                if (__any_sync(0xffffffffu, fa != 1.f || fb != 1.f))
    #pragma unroll
                    for (int j = 0; j < NJ; ++j) o[r][j][0] *= fa, o[r][j][1] *= fa, o[r][j][2] *= fb, o[r][j][3] *= fb;
    #pragma unroll
                for (int jl = 0; jl < NJ; ++jl) {
                    uint32_t acc[2] = {};
                    mma_f16acc(acc, x0.x, x0.y, x0.z, x0.w, bv[0][jl], bv[1][jl]);
                    mma_f16acc(acc, x1.x, x1.y, x1.z, x1.w, bv[2][jl], bv[3][jl]);
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[0]));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[1]));
                    o[r][jl][0] += lo.x, o[r][jl][1] += lo.y, o[r][jl][2] += hi.x, o[r][jl][3] += hi.y;
                }
            }
        }
        flush(ii, o);
    };
    item(0, 0, nt0);
    if (ni > 1) item(1, nt0, ntot - nt0);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

template <class VC, int S>
static void v48_launch(const __nv_bfloat16 * q, const uint8_t * kc, const uint8_t * vc, float * part, const int * dseq0,
                       int nq, int h, int kh, float scale, int nsplit, cudaStream_t s, const int * dep, int pstride, int zoff,
                       const int * dstart) {
    static_assert(v48::smem<VC, S>() <= 99 * 1024, "attn_v48_k: more than 99 KB of shared memory");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_v48_k<VC, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(v48::smem<VC, S>()));
        configured = true;
    }
    const int items = kh * nsplit, blocks = (items + v48::NI - 1) / v48::NI;
    attn_v48_k<VC, S><<<blocks, (v48::QKW + v48::PVW) * 32, v48::smem<VC, S>(), s>>>(
        q, kc, vc, part, dseq0, nq, h, kh, scale, nsplit, dep, pstride, zoff, dstart);
}

// y [nq][heads][256]: undo the TQ3_0 rotation per 32 (WHT / sqrt(32), then signs); one warp per 32 values
__global__ void tq3_inverse_k(__nv_bfloat16 * y, int64_t nblocks) {
    const int64_t b = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (b >= nblocks) return;
    __nv_bfloat16 * p = y + b * 32 + lane;
    float v = __bfloat162float(*p);
    for (int m = 1; m < 32; m <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? other - v : v + other;
    }
    *p = __float2bfloat16_rn(v * 0.17677669529663687f * tq3a::SIGNS[lane]);
}

// In-block part of two-part tree verify attention on the split cache: query z (column z) over the keys
// [tq_split_point(qpos), qpos], committed rows then its ancestors (cache slots pos + column), decoded with the codecs'
// own loaders; FP32 dot products against Q in the K basis, FP32 softmax, V sum in the V basis. Merged with the pstride
// cache partials (attn_blk_k arithmetic, block partial last); y stays in the V basis for the inverse rotation.
// A node's result depends only on its path, so it equals that path verified as a chain.
template <class KC, class VC>
__global__ void __launch_bounds__(256) attn_blk_codec_k(const __nv_bfloat16 * q, const uint8_t * kc, const uint8_t * vc,
                                                        const float * part, const int * dseq0, const int * par, int heads,
                                                        int kvheads, float scale, int nsplit, int pstride,
                                                        __nv_bfloat16 * y, bool inv = false) {
    constexpr int DIM = 256, MAXK = TREE_MAX_DEPTH + 33, MAXS = 264;  // split count (<= 256) + the stand-in slot
    constexpr bool KT = std::is_same_v<KC, kvc::Tcq3>, VT = std::is_same_v<VC, kvc::Tcq2>;
    __shared__ float s_cbk[KT ? 128 : 1], s_cbv[VT ? 128 : 1];
    __shared__ __align__(16) int8_t k8[MAXK * DIM];
    __shared__ __align__(16) __half vh[MAXK * DIM];
    __shared__ float ksf[MAXK * 8], qr[DIM], sc[MAXK];
    __shared__ int slot[MAXK];
    const int h = blockIdx.x, z = blockIdx.y, d = threadIdx.x, lane = d % 32, wid = d / 32;
    if constexpr (KT)
        for (int i = d; i < 512; i += DIM) reinterpret_cast<int8_t *>(s_cbk)[i] = int8_t(__float2int_rn(tcq::CB3K[i] * tcq::CB3K_SCALE));
    if constexpr (VT)
        for (int i = d; i < 256; i += DIM) reinterpret_cast<__half *>(s_cbv)[i] = __float2half_rn(tcq::CB2V[i]);
    // independent global reads first, so their latencies overlap: the parent chain, every split's maximum and sum, the
    // first MPF partial values of this thread's value, the query
    constexpr int MPF = 16;
    __shared__ int s_par[TREE_MAX_DEPTH + 1];
    __shared__ float s_m[MAXS], s_l[MAXS], s_w[MAXS];
    __shared__ float s_mm;
    const float * base = part + (int64_t(z) * heads + h) * pstride * (DIM + 2);
    if (d <= z) s_par[d] = par[d];
    for (int i = d; i < nsplit; i += DIM) s_m[i] = base[i * (DIM + 2) + DIM], s_l[i] = base[i * (DIM + 2) + DIM + 1];
    float pv0[MPF];
#pragma unroll
    for (int i = 0; i < MPF; ++i) pv0[i] = i < nsplit ? base[i * (DIM + 2) + d] : 0.f;
    const float q0 = __bfloat162float(q[(int64_t(z) * heads + h) * DIM + d]);
    const int pos = *dseq0 - 1, kh = h / (heads / kvheads);
    __syncthreads();
    int path[TREE_MAX_DEPTH + 1], len = 0;
    for (int a = z; a >= 0; a = s_par[a]) path[len++] = a;  // z .. root; depth e is path[len - 1 - e]
    const int qpos = pos + len - 1, from = tq_split_point(qpos), nk = qpos - from + 1;
    if (d < nk) {
        const int p = from + d;
        slot[d] = p < pos ? p : pos + path[len - 1 - (p - pos)];
    }
    // Q into the K basis: signs, WHT over 32 (TQ3_0) or 128 (TCQ) values, then the codec's normalisation
    constexpr int WB = KC::ROT128 ? 128 : 32;
    qr[d] = q0 * (KC::ROT128 ? tcq::S1[d % 128] : tq3a::SIGNS[d % 32]);
    for (int m = 1; m < WB; m <<= 1) {
        __syncthreads();
        if (!(d & m)) {
            const float a = qr[d], b = qr[d + m];
            qr[d] = a + b, qr[d + m] = a - b;
        }
    }
    __syncthreads();
    qr[d] *= KC::ROT128 ? tcq::INV_SQRT_128 * tcq::S2[d % 128] : 0.17677669529663687f;
    for (int u = d; u < nk * 8; u += DIM) {  // decode (key, 32-value block) units
        const int i = u / 8, b = u % 8;
        uint32_t kw[KC::NW], vw[VC::NW];
        __half ks, vs;
        KC::load(kc + (int64_t(slot[i]) * kvheads + kh) * KC::RB, b, kw, ks);
        KC::store_k(kw, ks, k8 + i * DIM + b * 32, ksf[i * 8 + b], s_cbk);
        VC::load(vc + (int64_t(slot[i]) * kvheads + kh) * VC::RB, b, vw, vs);
        VC::store_v(vw, vs, vh + i * DIM + b * 32, s_cbv);
    }
    __syncthreads();
    for (int i = wid; i < nk; i += DIM / 32) {  // one warp per key, 8 dims per lane, fixed-order reduction
        float v = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) v += qr[lane * 8 + e] * float(k8[i * DIM + lane * 8 + e]);
        v *= ksf[i * 8 + lane / 4];
#pragma unroll
        for (int w = 16; w; w >>= 1) v += __shfl_xor_sync(0xffffffffu, v, w);
        if (lane == 0) sc[i] = v * scale;
    }
    __syncthreads();
    float m = -CUDART_INF_F;
    for (int i = 0; i < nk; ++i) m = fmaxf(m, sc[i]);
    float l = 0.f, o = 0.f;
    for (int i = 0; i < nk; ++i) {
        const float pr = __expf(sc[i] - m);
        l += pr;
        o += pr * __half2float(vh[i * DIM + d]);
    }
    // merge with the cache partials, in split order (the maxima, sums and weights staged in shared memory)
    if (d == 0) {
        float mm = m;
        for (int s2 = 0; s2 < nsplit; ++s2) mm = fmaxf(mm, s_m[s2]);
        s_mm = mm;
    }
    __syncthreads();
    const float mm = s_mm;
    for (int i = d; i < nsplit; i += DIM) s_w[i] = s_l[i] > 0.f ? expf(s_m[i] - mm) : 0.f;
    __syncthreads();
    float zz = 0.f, acc = 0.f;
    for (int s0 = 0; s0 < nsplit; s0 += MPF) {
        float pv[MPF];
#pragma unroll
        for (int i = 0; i < MPF; ++i) pv[i] = s0 == 0 ? pv0[i] : s0 + i < nsplit ? base[(s0 + i) * (DIM + 2) + d] : 0.f;
#pragma unroll
        for (int i = 0; i < MPF; ++i) {
            const int s2 = s0 + i;
            if (s2 >= nsplit || !(s_l[s2] > 0.f)) continue;
            const float w = s_w[s2];
            zz += s_l[s2] * w;
            acc += pv[i] * w;
        }
    }
    if (l > 0.f) {
        const float w = expf(m - mm);
        zz += l * w;
        acc += o * w;
    }
    __nv_bfloat16 out = __float2bfloat16_rn(acc / zz);
    if (inv) {  // tq3_inverse_k on the stored value: WHT over the warp's 32 values, / sqrt(32), signs
        float v = __bfloat162float(out);
        for (int m2 = 1; m2 < 32; m2 <<= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, v, m2);
            v = (lane & m2) ? other - v : v + other;
        }
        out = __float2bfloat16_rn(v * 0.17677669529663687f * tq3a::SIGNS[lane]);
    }
    y[(int64_t(z) * heads + h) * DIM + d] = out;
}

template <class VC> static void inverse_v(__nv_bfloat16 * y, int nq, int h, cudaStream_t s) {
    if constexpr (std::is_same_v<VC, kvc::Tcq2>) {
        kvc::Tcq2::inverse(y, int64_t(nq) * h, s);
    } else {
        const int64_t nb = int64_t(nq) * h * 8;
        tq3_inverse_k<<<unsigned((nb + 7) / 8), 256, 0, s>>>(y, nb);
    }
}

// in-block part + merge, with the TQ3_0 inverse rotation fused unless V is TCQ (own inverse kernel)
template <class KC, class VC>
static void blk_merge(const __nv_bfloat16 * q, const uint8_t * kc, const uint8_t * vc, const float * part, const int * dseq0,
                      const int * par, int nq, int h, int kh, float scale, int nsplit, int pstride, __nv_bfloat16 * y,
                      cudaStream_t s) {
    constexpr bool VT = std::is_same_v<VC, kvc::Tcq2>;
    attn_blk_codec_k<KC, VC><<<dim3(h, nq), 256, 0, s>>>(q, kc, vc, part, dseq0, par, h, kh, scale, nsplit, pstride, y, !VT);
    if (VT) inverse_v<VC>(y, nq, h, s);
}

// pstride > 0: tree verify cache part only (partials, no merge / inverse; see attn_blk_codec_k)
template <int NW, class KC, class VC, int NB = 2>
static void tq3_launch(const __nv_bfloat16 * q, const uint8_t * kc, const uint8_t * vc, __nv_bfloat16 * y,
                       float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                       const int * dep = nullptr, int pstride = 0, int zoff = 0, const int * dstart = nullptr,
                       int cstride = 0, const float * kbias = nullptr, int flat = 0, int poff = 0) {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn_tq3kv_k<NW, KC, VC, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tq3a::smem<NW, NB>()));
        configured = true;
    }
    const int blocks = ((h / kh) * nq + NW * 16 - 1) / (NW * 16);
    attn_tq3kv_k<NW, KC, VC, NB><<<dim3(kh, blocks, nsplit), NW * 32, tq3a::smem<NW, NB>(), s>>>(
        q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, dep, pstride, zoff, dstart, cstride, kbias, flat, poff);
    if (pstride || cstride) return;
    if (nsplit > 1) attention_merge256(part, y, nsplit, h, nq, s);
    inverse_v<VC>(y, nq, h, s);
}

// y [rows][256]: undo the TCQ rotation per 128-value group, x = S1 * H(S2 * z) / sqrt(128); one block of 128 threads
// per group.
__global__ void tcq_inverse_k(__nv_bfloat16 * y) {
    __shared__ float x[128];
    __nv_bfloat16 * p = y + int64_t(blockIdx.x) * 128;
    const int t = threadIdx.x, lane = t & 31;
    float v = __bfloat162float(p[t]) * tcq::S2[t];
    for (int m = 1; m < 32; m <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? other - v : v + other;
    }
    x[t] = v;
    __syncthreads();
    if (t < 64) {
        const int j = ((t >> 5) << 6) + (t & 31);
        const float a = x[j], b = x[j + 32];
        x[j] = a + b, x[j + 32] = a - b;
    }
    __syncthreads();
    if (t < 64) {
        const float a = x[t], b = x[t + 64];
        x[t] = a + b, x[t + 64] = a - b;
    }
    __syncthreads();
    p[t] = __float2bfloat16_rn(x[t] * tcq::INV_SQRT_128 * tcq::S1[t]);
}
void kvc::Tcq2::inverse(__nv_bfloat16 * y, int64_t rows, cudaStream_t s) {
    tcq_inverse_k<<<unsigned(rows * 2), 128, 0, s>>>(y);
}

// TCQ writer (KB = 2: 256-state V, KB = 3: 512-state K): one block, one thread per trellis state, per (token, kv head, 128-value
// group). Unit norm, 128-point WHT (S1, butterflies, 1/sqrt(128), S2), Viterbi over the right-shift trellis
// ns = (prev >> KB) | (out << 6) with a free initial state, backtrack, norm = original / reconstructed * alpha.
template <int KB>
__global__ void __launch_bounds__(64 << KB) cache_write_tcq(const __nv_bfloat16 * x0, int ts, uint32_t * c, const int * dpos0,
                                                         int kvheads, int dim) {
    constexpr int STATES = 64 << KB, NP = 1 << KB, NWARP = STATES / 32, WPG = (6 + 128 * KB + 31) / 32;
    const float * cbk = KB == 2 ? tcq::CB2V : tcq::CB3K;
    __shared__ float x[128], cost[STATES], cost_b[STATES], wmc[NWARP];
    __shared__ int wmi[NWARP], init_state;
    __shared__ uint8_t bt[128 * 64];
    const int grp = blockIdx.x % 2, kh = blockIdx.x / 2, tkn = blockIdx.y, sid = threadIdx.x;
    if (sid < 128) x[sid] = __bfloat162float(x0[int64_t(tkn) * ts + kh * dim + grp * 128 + sid]);
    __syncthreads();
    cost[sid] = sid < 128 ? x[sid] * x[sid] : 0.f;
    __syncthreads();
    for (int st = STATES / 2; st >= 32; st >>= 1) {
        if (sid < st) cost[sid] += cost[sid + st];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    const float gnorm = sqrtf(cost[0]), inv = gnorm > 1e-10f ? 1.f / gnorm : 0.f;
    __syncthreads();
    if (sid < 128) {
        float v = x[sid] * inv * tcq::S1[sid];
        const int lane = sid & 31;
        for (int m = 1; m < 32; m <<= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, v, m);
            v = (lane & m) ? other - v : v + other;
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        const float a = x[j], b = x[j + 32];
        x[j] = a + b, x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        const float a = x[sid], b = x[sid + 64];
        x[sid] = a + b, x[sid + 64] = a - b;
    }
    __syncthreads();
    if (sid < 128) x[sid] *= tcq::INV_SQRT_128 * tcq::S2[sid];
    cost[sid] = 0.f;
    const float cb = cbk[sid];
    __syncthreads();
    for (int t = 0; t < 128; ++t) {
        const float * rd = (t & 1) ? cost_b : cost;
        float * wr = (t & 1) ? cost : cost_b;
        const int base = (sid & 0x3F) << KB;
        float best = rd[base];
        int bp = 0;
#pragma unroll
        for (int p = 1; p < NP; ++p) {
            const float cc = rd[base | p];
            if (cc < best) best = cc, bp = p;
        }
        if (sid < 64) bt[t * 64 + sid] = uint8_t(bp);
        const float d = x[t] - cb;
        wr[sid] = best + d * d;
        __syncthreads();
    }
    {
        float mc = cost[sid];
        int mi = sid;
        for (int o = 16; o; o >>= 1) {
            const float oc = __shfl_xor_sync(0xffffffffu, mc, o);
            const int oi = __shfl_xor_sync(0xffffffffu, mi, o);
            if (oc < mc) mc = oc, mi = oi;
        }
        if (sid % 32 == 0) wmc[sid / 32] = mc, wmi[sid / 32] = mi;
    }
    __syncthreads();
    if (sid < 32) {
        float best = sid < NWARP ? wmc[sid] : 3.4028234663852886e38f;
        int bi = sid < NWARP ? wmi[sid] : 0;
        for (int o = 16; o; o >>= 1) {
            const float oc = __shfl_down_sync(0xffffffffu, best, o);
            const int oi = __shfl_down_sync(0xffffffffu, bi, o);
            if (oc < best) best = oc, bi = oi;
        }
        if (sid == 0) init_state = bi;
    }
    __syncthreads();
    uint8_t * out = reinterpret_cast<uint8_t *>(x);
    if (sid == 0) {
        int state = init_state;
        for (int t = 127; t >= 0; --t) {
            out[t] = uint8_t(state >> 6);
            state = ((state & 0x3F) << KB) | bt[t * 64 + (state & 0x3F)];
        }
        init_state = state;
    }
    __syncthreads();
    float rsq = 0.f;
    if (sid < 128) {
        int cs = init_state;
        for (int t = 0; t <= sid; ++t) cs = (cs >> KB) | (int(out[t]) << 6);
        const float cv = cbk[cs];
        rsq = cv * cv;
    }
    cost[sid] = rsq;
    __syncthreads();
    for (int st = STATES / 2; st >= 32; st >>= 1) {
        if (sid < st) cost[sid] += cost[sid + st];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    const float rn = sqrtf(cost[0]);
    const float norm = (rn > 1e-10f ? gnorm / rn : gnorm) * (KB == 2 ? tcq::ALPHA_V : tcq::ALPHA_K);
    if (sid < WPG) {
        const int ib = (init_state >> KB) & 0x3F, nbits = 6 + 128 * KB;
        uint32_t word = 0;
        for (int bit = 0; bit < 32; ++bit) {
            const int pos = sid * 32 + bit;
            int v = 0;
            if (pos < 6) v = (ib >> pos) & 1;
            else if (pos < nbits) v = (out[(pos - 6) / KB] >> ((pos - 6) % KB)) & 1;
            word |= uint32_t(v) << bit;
        }
        if (sid == WPG - 1) {
            const __half nh = __float2half(norm);
            word |= uint32_t(*reinterpret_cast<const unsigned short *>(&nh)) << 16;
        }
        c[(int64_t(*dpos0 + tkn) * kvheads + kh) * (2 * WPG) + WPG * grp + sid] = word;
    }
}

// TqCache stand-in segment: the stand-in rows are Q8KV256 records (rotated int8, fp16 scale per 32; keys in the TQ3_0
// basis), so the main kernel reads them as the Q8 / Q8 codec pair with a per-row bias and every query over the same n
// rows. Row splits write to split_buf and attn_compact_merge_k folds them into the slot (the partial layout of part is
// unchanged: one slot per stand-in segment).
constexpr int CMP_MAXS = 32;
__global__ void __launch_bounds__(256) attn_compact_merge_k(const float * sbuf, int splits, float * part, int ps, int slot) {
    constexpr int DIM = 256;
    const int r = blockIdx.x, t = threadIdx.x;  // r = z * heads + h
    const float * sp = sbuf + int64_t(r) * splits * (DIM + 2);
    float M = -1e30f;
    for (int s2 = 0; s2 < splits; ++s2)
        if (sp[s2 * (DIM + 2) + DIM + 1] > 0.f) M = fmaxf(M, sp[s2 * (DIM + 2) + DIM]);
    float L = 0.f, A = 0.f;
    for (int s2 = 0; s2 < splits; ++s2) {
        const float l = sp[s2 * (DIM + 2) + DIM + 1];
        if (!(l > 0.f)) continue;
        const float f = __expf(sp[s2 * (DIM + 2) + DIM] - M);
        L += l * f, A += sp[s2 * (DIM + 2) + t] * f;
    }
    float * pp = part + (int64_t(r) * ps + slot) * (DIM + 2);
    pp[t] = A;
    if (t == 0) pp[DIM] = M, pp[DIM + 1] = L;
}

size_t kv_compact_split_floats(int nver, int heads) { return size_t(nver) * heads * CMP_MAXS * (256 + 2); }

static void compact_partial(const __nv_bfloat16 * q, const KvCompactSeg & c, float * part, int nq, int h, int kh,
                            float scale, int ps, int slot, cudaStream_t s) {
    static int sms = 0;
    if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), sms = sms ? sms : 32;
    const bool mq = (h / kh) * nq <= 64;
    const int qtiles = ((h / kh) * nq + (mq ? 64 : 128) - 1) / (mq ? 64 : 128);
    int splits = std::clamp(2 * sms / (kh * qtiles), 1, CMP_MAXS);  // the row count is on the device: splits do not depend on it
    if (!c.split_buf || int64_t(nq) * h * splits > c.split_cap) splits = 1;
    float * dst = splits > 1 ? c.split_buf : part;
    const int cst = splits > 1 ? splits : ps, off = splits > 1 ? 0 : slot;
    const auto * k = reinterpret_cast<const uint8_t *>(c.k), * v = reinterpret_cast<const uint8_t *>(c.v);
    if (mq) tq3_launch<4, kvc::Q8, kvc::Q8, 1>(q, k, v, nullptr, dst, c.n, nq, h, kh, scale, splits, s, nullptr, 0, 0, nullptr, cst, c.bias, 1, off);
    else tq3_launch<8, kvc::Q8, kvc::Q8>(q, k, v, nullptr, dst, c.n, nq, h, kh, scale, splits, s, nullptr, 0, 0, nullptr, cst, c.bias, 1, off);
    if (splits > 1) attn_compact_merge_k<<<nq * h, 256, 0, s>>>(c.split_buf, splits, part, ps, slot);
}

// par (tree / chain verify block, <= TREE_MAX_DEPTH + 1 deep): two-part attention, partials need nsplit + 1 splits
template <class KC, class VC>
static void attention_kc_vc(const __nv_bfloat16 * q, const void * kc, const void * vc, __nv_bfloat16 * y, float * part,
                            const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s,
                            const int * par, const int * dep, const KvCompactSeg * cs, const int32_t * qsa_sel, int qsa_from,
                            bool qsa_mixed) {
    const int * d0 = cs ? cs->start : nullptr;
    if (h % kh) throw std::invalid_argument("attention_kv_split: heads % kvheads");
    const auto * k = static_cast<const uint8_t *>(kc);
    const auto * v = static_cast<const uint8_t *>(vc);
    if (qsa_sel) {  // QSA rows: their cache part over the selected keys; a block that crosses `from` runs the dense part first
        if (!par || cs || h / kh > 64 || (nq > 8 && !dep)) throw std::invalid_argument("QSA attention: a two-part verify block without stand-ins");
        const int ps = nsplit + 1, qmax = 64 / (h / kh);
        for (int z0 = 0; qsa_mixed && z0 < nq; z0 += qmax)
            tq3_launch<4, KC, VC, 1>(q + int64_t(z0) * h * 256, k, v, y, part + int64_t(z0) * h * ps * (256 + 2), dseq0,
                                     std::min(qmax, nq - z0), h, kh, scale, nsplit, s, dep ? dep + z0 : nullptr, ps, z0, nullptr);
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(attn_tq3kv_k<4, KC, VC, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tq3a::smem<4, 1>()));
            configured = true;
        }
        attn_tq3kv_k<4, KC, VC, 1, true><<<dim3(kh, nq, nsplit), 4 * 32, tq3a::smem<4, 1>(), s>>>(
            q, k, v, y, part, dseq0, nq, h, kh, scale, nsplit, dep, ps, 0, nullptr, 0, nullptr, 0, 0, qsa_sel, qsa_from);
        attn_blk_codec_k<KC, VC><<<dim3(h, nq), 256, 0, s>>>(q, k, v, part, dseq0, par, h, kh, scale, nsplit, ps, y);
        inverse_v<VC>(y, nq, h, s);
        return;
    }
    // 4-query launches single buffered (two blocks per SM; 5060 Ti 200K 86 -> 103 tok/s, 3090 +17-19% vs double)
    if (par) {  // cache part in launches of <= 64 rows with the same split count, so a query's result is width independent
        if (nsplit + 1 > 264) throw std::invalid_argument("attention_kv_split: more than 263 key splits");
        const int ps = nsplit + 1;
        constexpr bool VQ = std::is_same_v<KC, kvc::Q8> && (std::is_same_v<VC, kvc::Q4> || std::is_same_v<VC, kvc::Q8>);
        if constexpr (VQ) {
            // q8 K: up to 32 rows (1-5 queries at 6 heads per KV head) and 65-96 rows (11-16 queries: one pass instead of
            // two) take the raw-record ring kernel (attn_vq_k); 33-48 rows (7-8 queries: the DFlash2 verify) the
            // role-split kernel (attn_v48_k); 49-64 rows stay on attn_tq3kv_k
            if (h / kh <= 16) {
                constexpr int S = std::is_same_v<VC, kvc::Q4> ? 3 : 2;  // ring depth that fits in 99 KB with the decoded V tile
                const int qmax = 96 / (h / kh);
                for (int z0 = 0; z0 < nq; z0 += qmax) {
                    const int nz = std::min(qmax, nq - z0), rows = (h / kh) * nz;
                    const auto * qz = q + int64_t(z0) * h * 256;
                    float * pz = part + int64_t(z0) * h * ps * (256 + 2);
                    const int * dz = dep ? dep + z0 : nullptr;
                    if (rows <= 16) vq_launch<1, 4, VC, 2, false>(qz, k, v, pz, dseq0, nz, h, kh, scale, nsplit, s, dz, ps, z0, d0);
                    else if (rows > 64) vq_launch<6, 2, VC, S, true>(qz, k, v, pz, dseq0, nz, h, kh, scale, nsplit, s, dz, ps, z0, d0);
                    else if (rows <= 32) vq_launch<3, 4, VC, 2, true>(qz, k, v, pz, dseq0, nz, h, kh, scale, nsplit, s, dz, ps, z0, d0);
                    else if (rows <= 48) v48_launch<VC, std::is_same_v<VC, kvc::Q4> ? 4 : 3>(qz, k, v, pz, dseq0, nz, h, kh, scale, nsplit, s, dz, ps, z0, d0);
                    else tq3_launch<4, KC, VC, 1>(qz, k, v, y, pz, dseq0, nz, h, kh, scale, nsplit, s, dz, ps, z0, d0);
                }
                if (cs) compact_partial(q, *cs, part, nq, h, kh, scale, ps, nsplit, s);
                blk_merge<KC, VC>(q, k, v, part, dseq0, par, nq, h, kh, scale, nsplit + (cs ? 1 : 0), ps, y, s);
                return;
            }
        }
        const int qmax = 64 / (h / kh);
        for (int z0 = 0; z0 < nq; z0 += qmax) {
            const int nz = std::min(qmax, nq - z0);
            const auto * qz = q + int64_t(z0) * h * 256;
            float * pz = part + int64_t(z0) * h * ps * (256 + 2);
            const int * dz = dep ? dep + z0 : nullptr;
            tq3_launch<4, KC, VC, 1>(qz, k, v, y, pz, dseq0, nz, h, kh, scale, nsplit, s, dz, ps, z0, d0);
        }
        // the spare slot nsplit takes the stand-in segment; the block kernel then merges nsplit + 1 partials
        if (cs) compact_partial(q, *cs, part, nq, h, kh, scale, ps, nsplit, s);
        blk_merge<KC, VC>(q, k, v, part, dseq0, par, nq, h, kh, scale, nsplit + (cs ? 1 : 0), ps, y, s);
        return;
    }
    if (cs) {  // cache partials + one stand-in partial (slot ns), merged, then the inverse rotation
        const bool mq = (h / kh) * nq <= 64;
        const int ns = mq ? nsplit : 1;
        if (mq) tq3_launch<4, KC, VC, 1>(q, k, v, y, part, dseq0, nq, h, kh, scale, ns, s, nullptr, 0, 0, d0, ns + 1);
        else tq3_launch<8, KC, VC>(q, k, v, y, part, dseq0, nq, h, kh, scale, ns, s, nullptr, 0, 0, d0, ns + 1);
        compact_partial(q, *cs, part, nq, h, kh, scale, ns + 1, ns, s);
        attention_merge256(part, y, ns + 1, h, nq, s);
        inverse_v<VC>(y, nq, h, s);
        return;
    }
    if ((h / kh) * nq <= 64) {
        tq3_launch<4, KC, VC, 1>(q, k, v, y, part, dseq0, nq, h, kh, scale, nsplit, s);
    }
    else tq3_launch<8, KC, VC>(q, k, v, y, part, dseq0, nq, h, kh, scale, 1, s);
}

void attention_kv_split(KvCodec kcodec, KvCodec vcodec, const __nv_bfloat16 * q, const void * kc, const void * vc,
                        __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit,
                        cudaStream_t s, const int * par, const int * dep, const KvCompactSeg * cs, const int32_t * qsa_sel,
                        int qsa_from, bool qsa_mixed) {
    auto with_v = [&](auto kt) {
        using KC = decltype(kt);
        switch (vcodec) {
            case KvCodec::Tq3: attention_kc_vc<KC, kvc::Tq3>(q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s, par, dep, cs, qsa_sel, qsa_from, qsa_mixed); return;
            case KvCodec::Tq2s: attention_kc_vc<KC, kvc::Tq2s>(q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s, par, dep, cs, qsa_sel, qsa_from, qsa_mixed); return;
            case KvCodec::Q8: attention_kc_vc<KC, kvc::Q8>(q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s, par, dep, cs, qsa_sel, qsa_from, qsa_mixed); return;
            case KvCodec::Q4: attention_kc_vc<KC, kvc::Q4>(q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s, par, dep, cs, qsa_sel, qsa_from, qsa_mixed); return;
            case KvCodec::Tcq2: attention_kc_vc<KC, kvc::Tcq2>(q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s, par, dep, cs, qsa_sel, qsa_from, qsa_mixed); return;
        }
        throw std::invalid_argument("attention_kv_split: unsupported V codec");
    };
    switch (kcodec) {
        case KvCodec::Tq3: with_v(kvc::Tq3{}); return;
        case KvCodec::Q8: with_v(kvc::Q8{}); return;
        case KvCodec::Q4: with_v(kvc::Q4{}); return;
        case KvCodec::Tcq3: with_v(kvc::Tcq3{}); return;
        default: throw std::invalid_argument("attention_kv_split: unsupported K codec");
    }
}

void attention_tq3kv(const __nv_bfloat16 * q, const Tq3KV256 * kc, const Tq3KV256 * vc, __nv_bfloat16 * y, float * part,
                     const int * dseq0, int nq, int h, int kh, float scale, int nsplit, cudaStream_t s) {
    attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s);
}

void attention_tq3k_tq2v(const __nv_bfloat16 * q, const Tq3KV256 * kc, const Tq2V256 * vc, __nv_bfloat16 * y,
                         float * part, const int * dseq0, int nq, int h, int kh, float scale, int nsplit,
                         cudaStream_t s) {
    attention_kv_split(KvCodec::Tq3, KvCodec::Tq2s, q, kc, vc, y, part, dseq0, nq, h, kh, scale, nsplit, s);
}

// TQ3_0 writer: one warp per (token, kv head, 32-value block).
__global__ void cache_write_multi_tq3_k(const __nv_bfloat16 * x0, int ts, tqllm_cuda::Tq3KV256 * c, const int * dpos0,
                                        int kvheads, int dim) {
    const int lane = threadIdx.x, blk = blockIdx.x % (dim / 32), kh = blockIdx.x / (dim / 32), tkn = blockIdx.y;
    const float in = __bfloat162float(x0[int64_t(tkn) * ts + kh * dim + blk * 32 + lane]);
    float ss = in * in;
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float rms = sqrtf(ss / 32.f);
    if (rms < 1e-10f) rms = 1.f;
    const __half rh = __float2half_rn(rms);
    rms = __half2float(rh);
    float v = in / rms * tq3a::SIGNS[lane];
    for (int m = 1; m < 32; m <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? other - v : v + other;
    }
    const float r = v * 0.17677669529663687f;
    uint32_t idx = 0;
#pragma unroll
    for (int b = 0; b < 7; ++b) idx = r > tq3a::BOUND[b] ? b + 1 : idx;
    tqllm_cuda::Tq3KV256 * rec = c + int64_t(*dpos0 + tkn) * kvheads + kh;
    const int bit = 3 * lane;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int sh = bit - 32 * k;
        const uint32_t part = sh >= 32 || sh <= -3 ? 0u : sh >= 0 ? idx << sh : idx >> -sh;
        const uint32_t word = __reduce_or_sync(0xffffffffu, part);
        if (lane == k) rec->w[3 * blk + k] = word;
    }
    if (lane == 0) rec->rms[blk] = rh;
}

void cache_write_multi_tq3kv(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Tq3KV256 * kc,
                             Tq3KV256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t s) {
    if (dim != 256) throw std::invalid_argument("cache_write_multi_tq3kv: head_dim must be 256");
    cache_write_multi_tq3_k<<<dim3(kvheads * (dim / 32), n), 32, 0, s>>>(k0, ts, kc, dpos0, kvheads, dim);
    cache_write_multi_tq3_k<<<dim3(kvheads * (dim / 32), n), 32, 0, s>>>(v0, ts, vc, dpos0, kvheads, dim);
}

__global__ void cache_write_multi_tq2v_v(const __nv_bfloat16 * x0, int ts, Tq2V256 * c, const int * dpos0,
                                         int kvheads, int dim) {
    const int lane = threadIdx.x, blk = blockIdx.x % (dim / 32), kh = blockIdx.x / (dim / 32), tkn = blockIdx.y;
    const float in = __bfloat162float(x0[int64_t(tkn) * ts + kh * dim + blk * 32 + lane]);
    float ss = in * in;
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float rms = sqrtf(ss / 32.f);
    if (rms < 1e-10f) rms = 1.f;
    const __half rh = __float2half_rn(rms);
    rms = __half2float(rh);
    float v = in / rms * tq3a::SIGNS[lane];
    for (int m = 1; m < 32; m <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? other - v : v + other;
    }
    const float r = v * 0.17677669529663687f;
    const uint32_t code = r < -0.9816f ? 0u : r < 0.f ? 1u : r < 0.9816f ? 2u : 3u;
    Tq2V256 * rec = c + int64_t(*dpos0 + tkn) * kvheads + kh;
    const int word = lane / 16, shift = (lane % 16) * 2;
    const unsigned mask = lane < 16 ? 0x0000ffffu : 0xffff0000u;
    const uint32_t packed = __reduce_or_sync(mask, code << shift);
    if ((lane & 15) == 0) rec->w[2 * blk + word] = packed;
    if (lane == 0) rec->rms[blk] = rh;
}

// Rotated q8 / q4 writer: one warp per (token, kv head, 32-value block), TQ3_0 rotation, then absmax int8 or q4_0-style
// signed-max 4-bit (value (q - 8) * d).
template <bool Q4>
__global__ void cache_write_multi_qn(const __nv_bfloat16 * x0, int ts, uint8_t * c, const int * dpos0, int kvheads, int dim) {
    const int lane = threadIdx.x, blk = blockIdx.x % (dim / 32), kh = blockIdx.x / (dim / 32), tkn = blockIdx.y;
    float v = __bfloat162float(x0[int64_t(tkn) * ts + kh * dim + blk * 32 + lane]) * tq3a::SIGNS[lane];
    for (int m = 1; m < 32; m <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, m);
        v = (lane & m) ? other - v : v + other;
    }
    v *= 0.17677669529663687f;
    float amax = fabsf(v), mx = v;
    for (int o = 16; o; o >>= 1) {
        const float a2 = __shfl_xor_sync(0xffffffffu, amax, o), m2 = __shfl_xor_sync(0xffffffffu, mx, o);
        if (a2 > amax) amax = a2, mx = m2;
    }
    constexpr int RB = Q4 ? sizeof(Q4KV256) : sizeof(Q8KV256);
    uint8_t * rec = c + (int64_t(*dpos0 + tkn) * kvheads + kh) * RB;
    const __half dh = __float2half_rn(Q4 ? mx / -8.f : amax / 127.f);
    const float d = __half2float(dh), id = d != 0.f ? 1.f / d : 0.f;
    if constexpr (Q4) {
        const int qv = min(15, max(0, __float2int_rn(v * id) + 8));
        const int hi = __shfl_down_sync(0xffffffffu, qv, 1);
        if (!(lane & 1)) rec[16 + 16 * blk + lane / 2] = uint8_t(qv | (hi << 4));
    } else {
        rec[16 + 32 * blk + lane] = uint8_t(int8_t(max(-127, min(127, __float2int_rn(v * id)))));
    }
    if (lane == 0) reinterpret_cast<__half *>(rec)[blk] = dh;
}

void cache_write_kv_side(KvCodec codec, const __nv_bfloat16 * x0, int ts, int n, void * c, const int * dpos0, int kvheads,
                         int dim, cudaStream_t s) {
    if (dim != 256) throw std::invalid_argument("cache_write_kv_side: head_dim must be 256");
    const dim3 grid(kvheads * (dim / 32), n);
    switch (codec) {
        case KvCodec::Tq3: cache_write_multi_tq3_k<<<grid, 32, 0, s>>>(x0, ts, static_cast<Tq3KV256 *>(c), dpos0, kvheads, dim); return;
        case KvCodec::Tq2s: cache_write_multi_tq2v_v<<<grid, 32, 0, s>>>(x0, ts, static_cast<Tq2V256 *>(c), dpos0, kvheads, dim); return;
        case KvCodec::Q8: cache_write_multi_qn<false><<<grid, 32, 0, s>>>(x0, ts, static_cast<uint8_t *>(c), dpos0, kvheads, dim); return;
        case KvCodec::Q4: cache_write_multi_qn<true><<<grid, 32, 0, s>>>(x0, ts, static_cast<uint8_t *>(c), dpos0, kvheads, dim); return;
        case KvCodec::Tcq2: cache_write_tcq<2><<<dim3(kvheads * 2, n), 256, 0, s>>>(x0, ts, static_cast<uint32_t *>(c), dpos0, kvheads, dim); return;
        case KvCodec::Tcq3: cache_write_tcq<3><<<dim3(kvheads * 2, n), 512, 0, s>>>(x0, ts, static_cast<uint32_t *>(c), dpos0, kvheads, dim); return;
    }
    throw std::invalid_argument("cache_write_kv_side: unsupported codec");
}

void cache_write_multi_tq3k_tq2v(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Tq3KV256 * kc,
                                 Tq2V256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t s) {
    if (dim != 256) throw std::invalid_argument("cache_write_multi_tq3k_tq2v: head_dim must be 256");
    cache_write_multi_tq3_k<<<dim3(kvheads * (dim / 32), n), 32, 0, s>>>(k0, ts, kc, dpos0, kvheads, dim);
    cache_write_multi_tq2v_v<<<dim3(kvheads * (dim / 32), n), 32, 0, s>>>(v0, ts, vc, dpos0, kvheads, dim);
}

}  // namespace tqllm_cuda

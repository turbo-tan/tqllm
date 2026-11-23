// Inkling-2-Mini grouped routed experts for prompt chunks (see inkling2_grouped.cuh). A block computes 64 weight rows x
// 64 assignments of one expert: per 64-column step the weight tile is decoded to BF16 in shared memory (exact), the
// activations of the tile's assignments are gathered, and 8 warps run 16x16x16 BF16 WMMA with F32 accumulators.
#include "inkling2_grouped.cuh"

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mma.h>

#include "inkling2_fp4_mma.cuh"

#include <stdexcept>

namespace tqllm_cuda::inkling2 {
namespace {
constexpr int BM = 64, BN = kGroupTile, BK = 64, LDS = BK + 8, LDC = BN + 4;
constexpr int kMaxExperts = 1024;
constexpr int NF = BN / 32;          // accumulator fragments per warp (8 warps: 4 x 16 rows, 2 x BN / 2 columns)
constexpr int RPT = BM * BN / 256;  // epilogue rows per thread

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float2 e2m1x2(uint32_t byte) {
    return __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t(byte & 0xff), __NV_E2M1)));
}
__device__ __forceinline__ float e4m3f(uint8_t b) { __nv_fp8_e4m3 v; v.__x = b; return float(v); }
__device__ __forceinline__ uint8_t fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    const __nv_fp8_e4m3 v(x);
    return v.__x;
}
__device__ __forceinline__ float e8m0f(uint8_t s) { return s ? __int_as_float(int(s) << 23) : 5.877472e-39f; }
__device__ __forceinline__ void cp_async16(void * d, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned) __cvta_generic_to_shared(d)), "l"(g));
}
__device__ __forceinline__ void cp_async4(void * d, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned) __cvta_generic_to_shared(d)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// 16 values -> 8 NVFP4 code bytes + one UE4M3 scale (no row scale: the scale itself carries the magnitude)
__device__ __forceinline__ void quant16_ep(const float * v, uint8_t * d, uint8_t * sc) {
    __nv_bfloat16 b[16];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { b[j] = __float2bfloat16(v[j]); amax = fmaxf(amax, fabsf(__bfloat162float(b[j]))); }
    uint8_t s = fp32_to_ue4m3(amax / 6.0f);
    if (s > 0x7e) s = 0x7e;
    const float sd = e4m3f(s);
    const float is = sd > 0.0f ? 1.0f / sd : 0.0f;
    uint16_t * d16 = reinterpret_cast<uint16_t *>(d);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __nv_fp4x4_e2m1 q(make_float4(__bfloat162float(b[4 * j + 0]) * is, __bfloat162float(b[4 * j + 1]) * is,
                                            __bfloat162float(b[4 * j + 2]) * is, __bfloat162float(b[4 * j + 3]) * is));
        d16[j] = q.__x;
    }
    *sc = s;
}
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

// one block per expert: its count, its offset (assignments of lower experts) and its assignments in ascending order
__global__ void __launch_bounds__(256) group_k(const int * ids, int A, int * order, int * counts, int * offsets) {
    const int e = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ int wl[8], wc[8], base;
    int less = 0, cnt = 0;
    for (int a = threadIdx.x; a < A; a += blockDim.x) { const int id = ids[a]; less += id < e; cnt += id == e; }
    for (int o = 16; o; o >>= 1) { less += __shfl_xor_sync(0xffffffffu, less, o); cnt += __shfl_xor_sync(0xffffffffu, cnt, o); }
    if (lane == 0) { wl[warp] = less; wc[warp] = cnt; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int L = 0, C = 0;
        for (int w = 0; w < 8; ++w) { L += wl[w]; C += wc[w]; }
        offsets[e] = L; counts[e] = C; base = L;
    }
    __syncthreads();
    for (int a0 = 0; a0 < A; a0 += blockDim.x) {
        const int a = a0 + threadIdx.x;
        const bool f = a < A && ids[a] == e;
        const unsigned b = __ballot_sync(0xffffffffu, f);
        if (lane == 0) wc[warp] = __popc(b);
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < 8; ++w) { before += w < warp ? wc[w] : 0; total += wc[w]; }
        if (f) order[base + before + __popc(b & ((1u << lane) - 1))] = a;
        __syncthreads();
        if (threadIdx.x == 0) base += total;
        __syncthreads();
    }
}
__global__ void tile_off_k(const int * counts, int E, int * tile_off) {
    if (threadIdx.x != 0) return;
    int t = 0;
    for (int e = 0; e < E; ++e) { tile_off[e] = t; t += (counts[e] + BN - 1) / BN; }
    tile_off[E] = t;
}

// decode rows r0 .. r0 + 63, columns k0 .. k0 + 63 of expert e to BF16 in s[64][LDS]: thread = (row, 16 columns).
// FMT 1 MXFP8 rows, FMT 2 NVF4 planes (see ExpertMat)
template <int FMT>
__device__ __forceinline__ void decode_tile(const ExpertMat & m, int e, int cols, int r0, int k0, __nv_bfloat16 (*s)[LDS]) {
    const int r = threadIdx.x / 4, q = threadIdx.x % 4, c0 = k0 + q * 16;
    float v[16];
    if constexpr (FMT == 2) {
        const uint8_t * codes = (m.wp ? m.wp[e] : m.w + size_t(e) * m.stride) + size_t(r0 + r) * (cols / 2);
        const uint8_t * aux = (m.ap ? m.ap[e] : m.aux + size_t(e) * m.aux_stride) + size_t(r0 + r) * (cols / 16 + 4);
        const float sc = e4m3f(__ldg(aux + c0 / 16)), g = __ldg(reinterpret_cast<const float *>(aux + cols / 16));
        const uint2 cw = __ldg(reinterpret_cast<const uint2 *>(codes + c0 / 2));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = e2m1x2(cw.x >> (8 * j)), b = e2m1x2(cw.y >> (8 * j));
            v[2 * j] = (a.x * sc) * g; v[2 * j + 1] = (a.y * sc) * g;
            v[8 + 2 * j] = (b.x * sc) * g; v[8 + 2 * j + 1] = (b.y * sc) * g;
        }
    } else {
        const uint8_t * row = (m.wp ? m.wp[e] : m.w + size_t(e) * m.stride) + size_t(r0 + r) * (cols + cols / 32);
        const float sc = e8m0f(__ldg(row + cols + c0 / 32));
        const uint32_t * cp = reinterpret_cast<const uint32_t *>(row + c0);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = __ldg(cp + j);
#pragma unroll
            for (int b = 0; b < 4; ++b) v[j * 4 + b] = e4m3f(uint8_t(c >> (8 * b))) * sc;
        }
    }
    __nv_bfloat162 * d = reinterpret_cast<__nv_bfloat162 *>(&s[r][q * 16]);
#pragma unroll
    for (int j = 0; j < 8; ++j) d[j] = __floats2bfloat162_rn(v[2 * j], v[2 * j + 1]);
}

// GU: gate + up + SwiGLU into act[a][rows]; else down into y[a][rows]. x rows: GU token a / topk, else assignment a.
template <int FMT, bool GU>
__global__ void __launch_bounds__(256) grouped_k(ExpertMat ma, ExpertMat mb, int rows, int cols, const __nv_bfloat16 * x, int topk, const int * order,
                                                 const int * counts, const int * offsets, const int * tile_off, int E, __nv_bfloat16 * out, float gc,
                                                 float uc) {
    using namespace nvcuda;
    const int tile = blockIdx.y;
    if (tile >= tile_off[E]) return;
    int lo = 0, hi = E - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (tile_off[mid] <= tile) lo = mid; else hi = mid - 1; }
    const int e = lo, j0 = (tile - tile_off[e]) * BN, cnt = min(BN, counts[e] - j0), base = offsets[e] + j0, r0 = blockIdx.x * BM;
    constexpr int NT = GU ? 3 : 2;
    constexpr int SM_AB = ((NT - 1) * BM + BN) * LDS * 2;
    __shared__ __align__(128) unsigned char smem[SM_AB > BM * LDC * 4 ? SM_AB : BM * LDC * 4];
    __shared__ int arow[BN];
    auto sA = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem);
    auto sU = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem + BM * LDS * 2);
    auto sB = reinterpret_cast<__nv_bfloat16 (*)[LDS]>(smem + (NT - 1) * BM * LDS * 2);
    auto sC = reinterpret_cast<float (*)[LDC]>(smem);
    if (threadIdx.x < BN) arow[threadIdx.x] = threadIdx.x < cnt ? order[base + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x / 32, rb16 = (warp / 2) * 16, cb = (warp % 2) * (BN / 2);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cg[NF], cu[NF];
    for (int i = 0; i < NF; ++i) { wmma::fill_fragment(cg[i], 0.0f); if (GU) wmma::fill_fragment(cu[i], 0.0f); }
    for (int k0 = 0; k0 < cols; k0 += BK) {
        decode_tile<FMT>(ma, e, cols, r0, k0, sA);
        if (GU) decode_tile<FMT>(mb, e, cols, r0, k0, sU);
        for (int c = threadIdx.x; c < BN * (BK / 8); c += blockDim.x) {  // activations: 16 B per thread step
            const int j = c / (BK / 8), q = c % (BK / 8), a = arow[j];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (a >= 0) v = __ldg(reinterpret_cast<const uint4 *>(x + size_t(GU ? a / topk : a) * cols + k0) + q);
            *reinterpret_cast<uint4 *>(&sB[j][q * 8]) = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa, fu;
            wmma::load_matrix_sync(fa, &sA[rb16][kk], LDS);
            if (GU) wmma::load_matrix_sync(fu, &sU[rb16][kk], LDS);
#pragma unroll
            for (int i = 0; i < NF; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, &sB[cb + 16 * i][kk], LDS);
                wmma::mma_sync(cg[i], fa, fb, cg[i]);
                if (GU) wmma::mma_sync(cu[i], fu, fb, cu[i]);
            }
        }
        __syncthreads();
    }
    // epilogue: thread = assignment j, rows i0 .. i0 + 15 (two 16-byte stores)
    for (int i = 0; i < NF; ++i) wmma::store_matrix_sync(&sC[rb16][cb + 16 * i], cg[i], LDC, wmma::mem_row_major);
    __syncthreads();
    const int j = threadIdx.x % BN, i0 = (threadIdx.x / BN) * RPT;
    float g[RPT];
#pragma unroll
    for (int k = 0; k < RPT; ++k) {
        const int i = i0 + k;
        g[k] = bfr(sC[i][j]);
    }
    if constexpr (GU) {
        __syncthreads();
        for (int i = 0; i < NF; ++i) wmma::store_matrix_sync(&sC[rb16][cb + 16 * i], cu[i], LDC, wmma::mem_row_major);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < RPT; ++k) {
            const int i = i0 + k;
            const float u = bfr(sC[i][j]);
            g[k] = gc * tanhf(g[k] / gc) * (1.0f / (1.0f + expf(-g[k]))) * (uc * tanhf(u / uc));
        }
    }
    const int a = arow[j];
    if (a < 0) return;
    __align__(16) __nv_bfloat162 o[RPT / 2];
#pragma unroll
    for (int k = 0; k < RPT / 2; ++k) o[k] = __floats2bfloat162_rn(g[2 * k], g[2 * k + 1]);
    uint4 * dst = reinterpret_cast<uint4 *>(out + size_t(a) * rows + r0 + i0);
#pragma unroll
    for (int k = 0; k < RPT / 8; ++k) dst[k] = reinterpret_cast<const uint4 *>(o)[k];
}

// ---- native NVFP4 W4A4 prompt path ----
// The stored E2M1 codes stay in FP4. The only new quantization is on the activations: one UE4M3 scale per 16 values
// plus one FP32 row scale. The MMA consumes the original weight codes, so the weight side is not re-quantized.
constexpr int F4BK = 64;
constexpr int F4_LDC = BN + 4;
constexpr int F4_RPT = BM * BN / 256;
constexpr int F4_NF = 64 / 8;  // 8 n-tiles of 8 assignments per warp
constexpr int F4S = 3;         // K stages in flight

__global__ void __launch_bounds__(256) quantize_act_nvfp4_k(const __nv_bfloat16 * __restrict__ x, int cols,
                                                             uint8_t * __restrict__ codes, uint8_t * __restrict__ scales,
                                                             float * __restrict__ row_scale) {
    extern __shared__ __align__(16) unsigned char qsmem[];
    auto sx = reinterpret_cast<__nv_bfloat16 *>(qsmem);
    __shared__ float sh[8];
    const int row = blockIdx.x;
    const __nv_bfloat16 * src = x + size_t(row) * cols;

    float amax = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sx[c] = src[c];
        amax = fmaxf(amax, fabsf(__bfloat162float(src[c])));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = amax;
    __syncthreads();
    float m = 0.0f;
    for (int w = 0; w < 8; ++w) m = fmaxf(m, sh[w]);
    if (threadIdx.x == 0) row_scale[row] = m;

    const float inv = m > 0.0f ? 1.0f / m : 0.0f;
    for (int g = threadIdx.x; g < cols / 16; g += blockDim.x) {
        const __nv_bfloat16 * p = sx + g * 16;
        float sa = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) sa = fmaxf(sa, fabsf(__bfloat162float(p[j]) * inv));
        uint8_t sc = fp32_to_ue4m3(sa / 6.0f);
        if (sc > 0x7e) sc = 0x7e;
        const float sd = e4m3f(sc);
        const float s = sd > 0.0f ? inv / sd : 0.0f;
        uint8_t * d = codes + size_t(row) * (cols / 2) + g * 8;
        uint16_t * d16 = reinterpret_cast<uint16_t *>(d);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_fp4x4_e2m1 q(make_float4(__bfloat162float(p[4 * j + 0]) * s, __bfloat162float(p[4 * j + 1]) * s,
                                                __bfloat162float(p[4 * j + 2]) * s, __bfloat162float(p[4 * j + 3]) * s));
            d16[j] = q.__x;
        }
        scales[size_t(row) * (cols / 16) + g] = sc;
    }
}

// GU: gate + up + SwiGLU into act[a][rows]; else down into y[a][rows]. x rows: GU token a / topk, else assignment a.
template <bool GU, bool QACT = false>
__global__ void __launch_bounds__(256) grouped_fp4_k(ExpertMat ma, ExpertMat mb, int rows, int cols, const uint8_t * __restrict__ xcodes,
                                                     const uint8_t * __restrict__ xscales, const float * __restrict__ xrow, int topk,
                                                     const int * __restrict__ order, const int * __restrict__ counts,
                                                     const int * __restrict__ offsets, const int * __restrict__ tile_off, int E,
                                                     __nv_bfloat16 * __restrict__ out, float gc, float uc,
                                                     uint8_t * __restrict__ acodes, uint8_t * __restrict__ ascales) {
    using namespace inkling2_fp4_mma;
    const int lane = threadIdx.x, warp = threadIdx.y, t = warp * 32 + lane;
    const int tile_idx = blockIdx.y;
    if (tile_idx >= tile_off[E]) return;
    int lo = 0, hi = E - 1;
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (tile_off[mid] <= tile_idx) lo = mid; else hi = mid - 1; }
    const int e = lo, j0 = (tile_idx - tile_off[e]) * BN, cnt = min(BN, counts[e] - j0);
    const int base = offsets[e] + j0, r0 = blockIdx.x * BM;

    extern __shared__ __align__(128) unsigned char smem[];
    auto sC  = reinterpret_cast<float (*)[F4_LDC]>(smem);
    auto sAg = reinterpret_cast<uint32_t (*)[8]>(smem);
    auto sAu = sAg + F4S * BM;
    auto sB  = sAu + F4S * BM;
    __shared__ uint32_t sAscg[F4S][BM], sAscu[F4S][BM], sBsc[F4S][BN];
    __shared__ float sWg[BM], sWu[BM], sBg[BN];
    __shared__ int arow[BN];

    for (int j = t; j < BN; j += 256) {
        const int a = j < cnt ? order[base + j] : -1;
        arow[j] = a;
        sBg[j] = a >= 0 ? (xrow ? xrow[GU ? a / topk : a] : 1.0f) : 0.0f;
        if (a < 0)  // padding columns stay zero in every stage (their outputs are discarded)
            for (int st = 0; st < F4S; ++st) {
                reinterpret_cast<uint4 *>(&sB[st * BN + j][0])[0] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4 *>(&sB[st * BN + j][0])[1] = make_uint4(0, 0, 0, 0);
                sBsc[st][j] = 0;
            }
    }
    const uint8_t * wg = ma.wp ? ma.wp[e] : ma.w + size_t(e) * ma.stride;
    const uint8_t * ag = ma.ap ? ma.ap[e] : ma.aux + size_t(e) * ma.aux_stride;
    const uint8_t * wu = GU ? (mb.wp ? mb.wp[e] : mb.w + size_t(e) * mb.stride) : nullptr;
    const uint8_t * au = GU ? (mb.ap ? mb.ap[e] : mb.aux + size_t(e) * mb.aux_stride) : nullptr;
    const int ab = cols / 16 + 4;
    for (int r = t; r < BM; r += 256) {
        sWg[r] = __ldg(reinterpret_cast<const float *>(ag + size_t(r0 + r) * ab + cols / 16));
        if (GU) sWu[r] = __ldg(reinterpret_cast<const float *>(au + size_t(r0 + r) * ab + cols / 16));
    }
    __syncthreads();

    const int rb16 = (warp / 2) * 16, cb = (warp % 2) * 64;
    tile<16, 8, float> cg[F4_NF] = {}, cu[F4_NF] = {};

    // one cp.async group per K step: weight codes and scale quads, assignment codes and scales
    auto prefetch = [&](int st, int k0) {
        const int kq = k0 / 2, ks = k0 / 16;
        if (t < 2 * BM) {
            const int r = t / 2, q = t % 2;
            cp_async16(&sAg[st * BM + r][q * 4], wg + size_t(r0 + r) * (cols / 2) + kq + q * 16);
            if (GU) cp_async16(&sAu[st * BM + r][q * 4], wu + size_t(r0 + r) * (cols / 2) + kq + q * 16);
        }
        if (t < BM) {
            cp_async4(&sAscg[st][t], ag + size_t(r0 + t) * ab + ks);
            if (GU) cp_async4(&sAscu[st][t], au + size_t(r0 + t) * ab + ks);
        }
        {
            const int j = t / 2, q = t % 2, a = arow[j];
            if (a >= 0) {
                const int row = GU ? a / topk : a;
                cp_async16(&sB[st * BN + j][q * 4], xcodes + size_t(row) * (cols / 2) + kq + q * 16);
                if (q == 0) cp_async4(&sBsc[st][j], xscales + size_t(row) * (cols / 16) + ks);
            }
        }
        cp_commit();
    };

    prefetch(0, 0);
    if (cols > F4BK) prefetch(1, F4BK);
    int st = 0;
    for (int k0 = 0; k0 < cols; k0 += F4BK) {
        if (k0 + 2 * F4BK < cols) prefetch((st + 2) % F4S, k0 + 2 * F4BK);
        if (k0 + 2 * F4BK < cols) cp_wait<2>();
        else if (k0 + F4BK < cols) cp_wait<1>();
        else cp_wait<0>();
        __syncthreads();

        tile<16, 8, int> A, Au;
        load_ldmatrix(A, reinterpret_cast<const int *>(&sAg[st * BM + rb16][0]), 8);
        if (GU) load_ldmatrix(Au, reinterpret_cast<const int *>(&sAu[st * BM + rb16][0]), 8);
        const uint32_t sag = sAscg[st][rb16 + lane / 4 + (lane % 2) * 8];
        const uint32_t sau = GU ? sAscu[st][rb16 + lane / 4 + (lane % 2) * 8] : 0;
#pragma unroll
        for (int n = 0; n < F4_NF; ++n) {
            tile<8, 8, int> B;
            load_generic(B, reinterpret_cast<const int *>(&sB[st * BN + cb + n * 8][0]), 8);
            const uint32_t sb = sBsc[st][cb + n * 8 + lane / 4];
            mma_nvfp4(cg[n], A, B, sag, sb);
            if (GU) mma_nvfp4(cu[n], Au, B, sau, sb);
        }
        __syncthreads();
        st = (st + 1) % F4S;
    }

    using tile_C = tile<16, 8, float>;
#pragma unroll
    for (int n = 0; n < F4_NF; ++n)
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int i = rb16 + tile_C::get_i(l), j = cb + n * 8 + tile_C::get_j(l);
            sC[i][j] = cg[n].x[l] * sWg[i] * sBg[j];
        }
    __syncthreads();
    const int jj = t % BN, i0 = (t / BN) * F4_RPT;
    float g[F4_RPT];
#pragma unroll
    for (int k = 0; k < F4_RPT; ++k) g[k] = bfr(sC[i0 + k][jj]);
    if constexpr (GU) {
        __syncthreads();
#pragma unroll
        for (int n = 0; n < F4_NF; ++n)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = rb16 + tile_C::get_i(l), j = cb + n * 8 + tile_C::get_j(l);
                sC[i][j] = cu[n].x[l] * sWu[i] * sBg[j];
            }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < F4_RPT; ++k) {
            const float u = bfr(sC[i0 + k][jj]);
            g[k] = gc * tanhf(g[k] / gc) * (1.0f / (1.0f + expf(-g[k]))) * (uc * tanhf(u / uc));
        }
    }
    const int a = arow[jj];
    if (a < 0) return;
    if constexpr (GU && QACT) {  // the down input: codes / scales straight out of SwiGLU, no bf16 round trip
        uint8_t * cd = acodes + size_t(a) * (rows / 2) + (r0 + i0) / 2;
        uint8_t * sd = ascales + size_t(a) * (rows / 16) + (r0 + i0) / 16;
        quant16_ep(g, cd, sd);
        quant16_ep(g + 16, cd + 8, sd + 1);
    } else {
        __align__(16) __nv_bfloat162 o[F4_RPT / 2];
#pragma unroll
        for (int k = 0; k < F4_RPT / 2; ++k) o[k] = __floats2bfloat162_rn(g[2 * k], g[2 * k + 1]);
        uint4 * dst = reinterpret_cast<uint4 *>(out + size_t(a) * rows + r0 + i0);
#pragma unroll
        for (int k = 0; k < F4_RPT / 8; ++k) dst[k] = reinterpret_cast<const uint4 *>(o)[k];
    }
}

// whole expert stack to BF16 global memory (the same decode and rounding as decode_tile); thread = (row, 8 columns)
template <int FMT>
__global__ void __launch_bounds__(256) decode_bf16_k(ExpertMat m, int rows, int cols, int ldrows, __nv_bfloat16 * out) {
    __shared__ __align__(16) __nv_bfloat16 s[BM][LDS];
    const int e = blockIdx.z, r0 = blockIdx.y * BM, k0 = blockIdx.x * BK;
    decode_tile<FMT>(m, e, cols, r0, k0, s);
    __syncthreads();
    for (int i = threadIdx.x; i < BM * (BK / 8); i += blockDim.x) {
        const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
        *reinterpret_cast<uint4 *>(out + (size_t(e) * ldrows + r0 + r) * cols + k0 + c) = *reinterpret_cast<const uint4 *>(&s[r][c]);
    }
}

// xg[j][:] = x[order[j] / topk][:] (gate / up inputs in grouped order); thread = 8 columns
__global__ void gather_rows_k(const __nv_bfloat16 * x, const int * order, int topk, __nv_bfloat16 * xg, int cols) {
    const int j = blockIdx.x, q = int(threadIdx.x + blockIdx.y * blockDim.x);
    if (q * 8 >= cols) return;
    reinterpret_cast<uint4 *>(xg + size_t(j) * cols)[q] =
        reinterpret_cast<const uint4 *>(x + size_t(order[j] / topk) * cols)[q];
}
// act[j][:] = softcapped SwiGLU of gu[j][0 .. I) / gu[j][I .. 2I), the grouped_k epilogue's arithmetic; thread = 2 rows
__global__ void swiglu_k(const __nv_bfloat16 * gu, __nv_bfloat16 * act, int I, float gc, float uc, size_t total) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;  // over A * I / 2
    if (i * 2 >= total) return;
    const size_t a = (i * 2) / size_t(I), off = (i * 2) % size_t(I);
    const float2 g = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(gu + a * 2 * I + off));
    const float2 u = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(gu + a * 2 * I + I + off));
    const float v0 = gc * tanhf(g.x / gc) * (1.0f / (1.0f + expf(-g.x))) * (uc * tanhf(u.x / uc));
    const float v1 = gc * tanhf(g.y / gc) * (1.0f / (1.0f + expf(-g.y))) * (uc * tanhf(u.y / uc));
    *reinterpret_cast<__nv_bfloat162 *>(act + a * I + off) = __floats2bfloat162_rn(v0, v1);
}
// Y[n][rows] = X[n][cols] W^T, BF16 in / out, F32 accumulate (model.cpp's dense gemm, per expert)
void gemm_bf16(cublasHandle_t h, const __nv_bfloat16 * W, int rows, int cols, const __nv_bfloat16 * X, __nv_bfloat16 * Y, int n) {
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols, &one, W, CUDA_R_16BF, cols, X, CUDA_R_16BF, cols,
                                           &zero, Y, CUDA_R_16BF, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("inkling2 grouped bf16: cublas " + std::to_string(int(st)));
}

// one cublasGemmGroupedBatchedEx call for all non-empty experts of one matrix (W[e] = Wgu + e * rows * cols):
// Y[e][ho[e] .. +hc[e]] = Xg[ho[e] .. +hc[e]] W[e]^T; falls back to per-expert gemms if cuBLAS refuses
// the host descriptor arrays must stay intact while the enqueued grouped kernel reads them: one set per call site
// (gate/up and down in one visit run back to back; the per-visit stream sync separates visits)
bool gemm_grouped_bf16(cublasHandle_t h, const __nv_bfloat16 * W, int rows, int cols, const __nv_bfloat16 * Xg, __nv_bfloat16 * Y, int E,
                       const int * hc, const int * ho, int set) {
    if (E > kMaxExperts) return false;
    int ng = 0;
    static cublasOperation_t ta[2][kMaxExperts], tb[2][kMaxExperts];
    static int m[2][kMaxExperts], n[2][kMaxExperts], kk[2][kMaxExperts], lda[2][kMaxExperts], ldb[2][kMaxExperts], ldc[2][kMaxExperts], gs[2][kMaxExperts];
    static float al[2][kMaxExperts], be[2][kMaxExperts];
    static const void * A[2][kMaxExperts];
    static void * C[2][kMaxExperts];
    static const void * Bp[2][kMaxExperts];  // the grouped kernel reads all descriptor arrays at execution time (UVA): must outlive the call
    for (int e = 0; e < E; ++e) {
        if (hc[e] <= 0) continue;
        ta[set][ng] = CUBLAS_OP_T; tb[set][ng] = CUBLAS_OP_N;
        m[set][ng] = rows; n[set][ng] = hc[e]; kk[set][ng] = cols;
        al[set][ng] = 1.0f; be[set][ng] = 0.0f;
        A[set][ng] = W + size_t(e) * rows * cols; lda[set][ng] = cols;
        Bp[set][ng] = Xg + size_t(ho[e]) * cols; ldb[set][ng] = cols;
        C[set][ng] = Y + size_t(ho[e]) * rows; ldc[set][ng] = rows;
        gs[set][ng] = 1;
        ++ng;
    }
    if (ng == 0) return true;
    const cublasStatus_t st = cublasGemmGroupedBatchedEx(h, ta[set], tb[set], m[set], n[set], kk[set], al[set], A[set], CUDA_R_16BF, lda[set],
                                                         Bp[set], CUDA_R_16BF, ldb[set],
                                                         be[set], C[set], CUDA_R_16BF, ldc[set], ng, gs[set], CUBLAS_COMPUTE_32F);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    return true;
}

template <int FMT, bool GU>
void launch(const ExpertMat & a, const ExpertMat & b, int rows, int cols, const __nv_bfloat16 * x, int A, int E, int topk, const int * order,
            const int * counts, const int * offsets, const int * tile_off, __nv_bfloat16 * out, float gc, float uc, cudaStream_t s) {
    if (rows % BM || cols % BK) throw std::runtime_error("inkling2 grouped experts: rows % 64 or cols % 64");
    const dim3 grid(rows / BM, group_max_tiles(A, E));
    grouped_k<FMT, GU><<<grid, 256, 0, s>>>(a, b, rows, cols, x, topk, order, counts, offsets, tile_off, E, out, gc, uc);
}
}  // namespace

void group_assignments(const int * ids, int A, int E, int * order, int * counts, int * offsets, int * tile_off, cudaStream_t s) {
    if (A <= 0) return;
    group_k<<<E, 256, 0, s>>>(ids, A, order, counts, offsets);
    tile_off_k<<<1, 32, 0, s>>>(counts, E, tile_off);
}

void quantize_act_nvfp4(const __nv_bfloat16 * x, int rows, int cols, uint8_t * codes, uint8_t * scales, float * row_scale, cudaStream_t s) {
    if (rows <= 0) return;
    if (cols % 64) throw std::runtime_error("inkling2 fp4 activation quantize: cols % 64");
    quantize_act_nvfp4_k<<<rows, 256, size_t(cols) * 2, s>>>(x, cols, codes, scales, row_scale);
}

namespace {
__global__ void fp4_mma_probe_k(int * out) {
#if defined(INKLING2_FP4_MMA)
    *out = 1;
#else
    *out = 0;
#endif
}
}  // namespace
// whether this build's device code for the current GPU carries the block-scaled FP4 MMA (sm_120a / sm_121a targets)
bool fp4_mma_available() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    int * d = nullptr, h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) { (void)cudaGetLastError(); return (cached = 0) == 1; }
    fp4_mma_probe_k<<<1, 1>>>(d);
    const bool ok = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d);
    (void)cudaGetLastError();
    cached = ok && h == 1 ? 1 : 0;
    return cached == 1;
}

void grouped_gate_up_fp4(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const uint8_t * xcodes, const uint8_t * xscales,
                         const float * xrow, int A, int E, int topk, const int * order, const int * counts, const int * offsets,
                         const int * tile_off, __nv_bfloat16 * act, float gc, float uc, cudaStream_t s) {
    if (A <= 0) return;
    if (rows % BM || cols % F4BK) throw std::runtime_error("inkling2 grouped fp4: rows % 64 or cols % 64");
    const dim3 grid(rows / BM, group_max_tiles(A, E));
    grouped_fp4_k<true><<<grid, dim3(32, 8), size_t(BM) * F4_LDC * 4, s>>>(g, u, rows, cols, xcodes, xscales, xrow, topk, order, counts, offsets,
                                                          tile_off, E, act, gc, uc, nullptr, nullptr);
}

void grouped_gate_up_fp4_q(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const uint8_t * xcodes, const uint8_t * xscales,
                           const float * xrow, int A, int E, int topk, const int * order, const int * counts, const int * offsets,
                           const int * tile_off, uint8_t * acodes, uint8_t * ascales, float gc, float uc, cudaStream_t s) {
    if (A <= 0) return;
    if (rows % BM || cols % F4BK) throw std::runtime_error("inkling2 grouped fp4: rows % 64 or cols % 64");
    const dim3 grid(rows / BM, group_max_tiles(A, E));
    grouped_fp4_k<true, true><<<grid, dim3(32, 8), size_t(BM) * F4_LDC * 4, s>>>(g, u, rows, cols, xcodes, xscales, xrow, topk, order, counts, offsets,
                                                                 tile_off, E, nullptr, gc, uc, acodes, ascales);
}

void grouped_down_fp4(const ExpertMat & d, int rows, int cols, const uint8_t * xcodes, const uint8_t * xscales, const float * xrow, int A, int E,
                      const int * order, const int * counts, const int * offsets, const int * tile_off, __nv_bfloat16 * y, cudaStream_t s) {
    if (A <= 0) return;
    if (rows % BM || cols % F4BK) throw std::runtime_error("inkling2 grouped fp4: rows % 64 or cols % 64");
    const dim3 grid(rows / BM, group_max_tiles(A, E));
    grouped_fp4_k<false><<<grid, dim3(32, 8), size_t(BM) * F4_LDC * 4, s>>>(d, d, rows, cols, xcodes, xscales, xrow, 1, order, counts, offsets,
                                                            tile_off, E, y, 0.0f, 0.0f, nullptr, nullptr);
}

void grouped_gate_up(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const __nv_bfloat16 * x, int A, int E, int topk, const int * order,
                     const int * counts, const int * offsets, const int * tile_off, __nv_bfloat16 * act, float gc, float uc, cudaStream_t s) {
    if (A <= 0) return;
    if (g.fmt == 2) launch<2, true>(g, u, rows, cols, x, A, E, topk, order, counts, offsets, tile_off, act, gc, uc, s);
    else if (g.fmt == 1) launch<1, true>(g, u, rows, cols, x, A, E, topk, order, counts, offsets, tile_off, act, gc, uc, s);
    else throw std::runtime_error("inkling2 grouped experts: format");
}

void grouped_down(const ExpertMat & d, int rows, int cols, const __nv_bfloat16 * act, int A, int E, const int * order, const int * counts,
                  const int * offsets, const int * tile_off, __nv_bfloat16 * y, cudaStream_t s) {
    if (A <= 0) return;
    if (d.fmt == 2) launch<2, false>(d, d, rows, cols, act, A, E, 1, order, counts, offsets, tile_off, y, 0.0f, 0.0f, s);
    else if (d.fmt == 1) launch<1, false>(d, d, rows, cols, act, A, E, 1, order, counts, offsets, tile_off, y, 0.0f, 0.0f, s);
    else throw std::runtime_error("inkling2 grouped experts: format");
}

void experts_decode_bf16(const ExpertMat & m, int E, int rows, int cols, int ldrows, __nv_bfloat16 * out, cudaStream_t s) {
    if (rows % BM || cols % BK) throw std::runtime_error("inkling2 experts_decode_bf16: rows % 64 or cols % 64");
    const dim3 grid(cols / BK, rows / BM, E);
    if (m.fmt == 2) decode_bf16_k<2><<<grid, 256, 0, s>>>(m, rows, cols, ldrows, out);
    else if (m.fmt == 1) decode_bf16_k<1><<<grid, 256, 0, s>>>(m, rows, cols, ldrows, out);
    else throw std::runtime_error("inkling2 experts_decode_bf16: format");
}

void grouped_gate_up_bf16(const __nv_bfloat16 * Wgu, int I, int D, const __nv_bfloat16 * x, int A, int E, int topk, const int * order, const int * hc,
                          const int * ho, __nv_bfloat16 * xg, __nv_bfloat16 * gu, __nv_bfloat16 * act, float gc, float uc, cublasHandle_t blas,
                          cudaStream_t s) {
    if (A <= 0) return;
    gather_rows_k<<<dim3(A, unsigned((D + 2047) / 2048)), 256, 0, s>>>(x, order, topk, xg, D);
    if (!gemm_grouped_bf16(blas, Wgu, 2 * I, D, xg, gu, E, hc, ho, 0))
        for (int e = 0; e < E; ++e)
            if (hc[e] > 0) gemm_bf16(blas, Wgu + size_t(e) * 2 * I * D, 2 * I, D, xg + size_t(ho[e]) * D, gu + size_t(ho[e]) * 2 * I, hc[e]);
    const size_t total = size_t(A) * I;
    swiglu_k<<<unsigned((total / 2 + 255) / 256), 256, 0, s>>>(gu, act, I, gc, uc, total);
}

void grouped_down_bf16(const __nv_bfloat16 * Wd, int D, int I, const __nv_bfloat16 * act, int A, int E, const int * hc, const int * ho,
                       __nv_bfloat16 * yg, cublasHandle_t blas, cudaStream_t s) {
    if (A <= 0) return;
    if (!gemm_grouped_bf16(blas, Wd, D, I, act, yg, E, hc, ho, 1))
        for (int e = 0; e < E; ++e)
            if (hc[e] > 0) gemm_bf16(blas, Wd + size_t(e) * D * I, D, I, act + size_t(ho[e]) * I, yg + size_t(ho[e]) * D, hc[e]);
}

}  // namespace tqllm_cuda::inkling2

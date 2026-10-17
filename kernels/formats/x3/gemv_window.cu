// gemv_window.cu: X3 decode GEMV for n = 1..16 columns, built the TRUSS way (~/trellis-kernel
// src/kernels/moe/moe_window.cu gemv_item: decode in registers, one mma per tile with the weights as the A
// operand, no grid barrier) instead of the exllamav3 cooperative kernel (gemv_x3v.cu: two grid.sync, one
// launch per row part, f32 staging around it).
//
// One matvec = three launches over ALL row parts of the matrix (e.g. ffn_gate + ffn_up):
//   had_in  A_p[t]           = H128(x[t] * suh_p)                     one warp per (part, t, 128 block)
//   gemv    P_p[kp][t][cols] = A_p[t][k range kp] . W_p               persistent grid over the items
//   out     y[t][off_p + c]  = bf16(svh_p * H128(sum_kp P_p[kp][t]))  one warp per (part, t, 128 block)
// An item is (part, 64-column group, k part). The k split depends on the part's shape only, never on n, so
// column t is bit-identical for every n (the drafter's verify batch equals decode). n > 8 adds a second mma per
// decoded tile (batch half NB = 2): a 12-column tree verify decodes the weights once instead of twice. Deterministic: fixed
// reduction orders, no atomics. A single-launch queue version (H items first, epilogue by the last finisher,
// TRUSS v7a style) was measured slower here: 547 vs ~510 us on the 9 bench matrices (per-item signalling
// costs more than the two launches it removes).
//
// Rounding matches the f32 path of gemv_x3v.cu: x rounded to half, half * suh, fp32 butterfly, half
// A; fp16 mma accumulation folded to fp32 every FOLD k slices; fp32 epilogue Hadamard and svh.
#include "had128.cuh"
#include "ptx.cuh"
#include "window.hpp"
#include "x3_layout.hpp"

#include <cuda_bf16.h>

namespace tqllm {
namespace x3 {

// ---------------------------------------------------------------------------------------------------------
// had_in / out: one warp per (part, t, 128 block); the warp-index walk is the same for both.

struct XwWarpTask {
    int part, t, c0;     // c0 = this lane's first element (4 per lane)
};

// The calling warp's task: (part, t, 128-element block) over x (width = cols) or over y (width < 0: rows).
__device__ __forceinline__ bool xw_warp_task(const XwList& L, int width, XwWarpTask& w) {
    const int lane = threadIdx.x & 31;
    int g = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (int i = 0; i < L.count; ++i) {
        const int blocks = (width < 0 ? L.p[i].rows : width) / 128;
        const int tasks = L.n * blocks;
        if (g < tasks) {
            w.part = i;
            w.t = g / blocks;
            w.c0 = (g % blocks) * 128 + lane * 4;
            return true;
        }
        g -= tasks;
    }
    return false;
}

static __global__ void xw_had_in(XwList L, const half* __restrict__ x) {
    XwWarpTask w;
    if (!xw_warp_task(L, L.cols, w)) return;
    const XwPart& P = L.p[w.part];
    const half2* xv = reinterpret_cast<const half2*>(x + (size_t) w.t * L.cols + w.c0);
    const half2* sv = reinterpret_cast<const half2*>(P.suh + w.c0);
    const half2 a = __hmul2(xv[0], sv[0]), b = __hmul2(xv[1], sv[1]);
    const float v0 = __low2float(a), v1 = __high2float(a), v2 = __low2float(b), v3 = __high2float(b);
    const float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, threadIdx.x & 31);
    half2* o = reinterpret_cast<half2*>(P.A + (size_t) w.t * L.cols + w.c0);
    o[0] = __floats2half2_rn(h0 * kRScale, h1 * kRScale);
    o[1] = __floats2half2_rn(h2 * kRScale, h3 * kRScale);
}

static __global__ void xw_out(XwList L, __nv_bfloat16* __restrict__ y, int ldy) {
    XwWarpTask w;
    if (!xw_warp_task(L, -1, w)) return;
    const XwPart& P = L.p[w.part];
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int kp = 0; kp < P.kparts; ++kp) {
        const float4 q = *reinterpret_cast<const float4*>(P.P + ((size_t) kp * L.n + w.t) * P.rows + w.c0);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
    }
    const float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
    v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, threadIdx.x & 31);
    shuffle_had_f2x32(v.z, v.w, threadIdx.x & 31);
    const half2* sv = reinterpret_cast<const half2*>(P.svh + w.c0);
    const half2 sa = sv[0], sb = sv[1];
    __nv_bfloat16* o = y + (size_t) w.t * ldy + P.out_off + w.c0;
    o[0] = __float2bfloat16(v.x * kRScale * __low2float(sa));
    o[1] = __float2bfloat16(v.y * kRScale * __high2float(sa));
    o[2] = __float2bfloat16(v.z * kRScale * __low2float(sb));
    o[3] = __float2bfloat16(v.w * kRScale * __high2float(sb));
}

// ---------------------------------------------------------------------------------------------------------
// gemv: a persistent grid walks the items (XwGeom: WK k-split warps x WNT tiles, PF slices in flight).
// NB = batch halves of 8 columns (1: n <= 8, 2: n <= 16); every half gets its own mma on the same decoded tile.

// mma.m16n8k16 with the decoded tile as A (16 output columns x 16 k) and the batch as B (16 k x 8
// columns t), fp16 accumulate: one mma per tile, every batch slot used (TRUSS mma.cuh mma_w).
// c[0] = (column g, t = 2q, 2q+1), c[1] = (column g + 8, same t); g = lane / 4, q = lane % 4.
__device__ __forceinline__ void xw_mma(const FragB& f0, const FragB& f1, uint32_t act_lo, uint32_t act_hi, FragCh& c) {
    const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&f0);
    const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&f1);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(p0[0]), "r"(p1[0]), "r"(p0[1]), "r"(p1[1]), "r"(act_lo), "r"(act_hi));
}

// Shared memory of one gemv block: the A stage (n rows, padded stride) or the k-split reduction, whichever is larger.
__host__ __device__ constexpr int xw_a_stride(int k_slices) { return k_slices * 8 + 4; }
inline size_t xw_gemv_smem(int n, int k_slices) {
    const size_t a = (size_t) n * xw_a_stride(k_slices) * 4;
    const size_t red = (size_t) XwGeom::WK * XW_MAX_N * XwGeom::COLS * 4;
    return a > red ? a : red;
}

// One item (part, column group, k part): P_part[kp][t][group columns].
template <class Codec, int NB>
__device__ __forceinline__ void xw_item(const XwList& L, int item) {
    using G = XwGeom;
    constexpr int TWORDS = Codec::u32words;
    constexpr int LOADS = Codec::loads(G::WNT);
    constexpr int LSTRIDE = Codec::ld_stride;

    int pi = 0;
    while (pi + 1 < L.count && item >= L.p[pi + 1].item0) ++pi;
    const XwPart& P = L.p[pi];
    item -= P.item0;

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ntiles = P.rows / 16;
    const int groups = P.rows / G::COLS;
    const int group = item % groups, kp = item / groups;   // group-minor: neighbour blocks read neighbour tiles
    const int k_slices = L.cols / 16 / P.kparts;
    const int chunk = ceil_div(k_slices, G::WK);
    const int ks0 = kp * k_slices + warp * chunk;
    const int myn = max(0, min(chunk, k_slices - warp * chunk));

    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const uint32_t* bp = P.B + (size_t) x3_tile_offset(ks0, group * G::WNT, ntiles) * TWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
        if constexpr (Codec::ld_lanes < 32)
            return lane < Codec::ld_lanes ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[G::PF][LOADS];
    #pragma unroll
    for (int d = 0; d < G::PF; ++d)
        if (d < myn) {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);
        }

    // The item's activations (n rows x its k range) are staged in shared memory once, behind the first
    // weight loads: per-slice L2 reads of A were a stall of the loop. Row stride padded by 4 words so the
    // 8 batch rows of a warp read hit 32 distinct banks.
    extern __shared__ __align__(16) uint32_t xw_smem[];
    const int a_words = k_slices * 8, a_stride = xw_a_stride(k_slices);
    {
        const uint4* src = reinterpret_cast<const uint4*>(P.A) + (size_t) kp * a_words / 4;
        for (int idx = threadIdx.x; idx < L.n * a_words / 4; idx += G::THREADS) {
            const int r = idx / (a_words / 4), c = idx % (a_words / 4);
            reinterpret_cast<uint4*>(xw_smem + r * a_stride)[c] = src[(size_t) r * (L.cols / 8) + c];
        }
        __syncthreads();
    }
    const int r0 = lane >> 2;
    bool r_ok[NB];
    const uint32_t* As[NB];
    #pragma unroll
    for (int h = 0; h < NB; ++h) {
        r_ok[h] = r0 + 8 * h < L.n;
        As[h] = xw_smem + (r0 + 8 * h) * a_stride + warp * chunk * 8 + (lane & 3);
    }

    const typename Codec::Lane codec(lane);
    const half2 hzero = __half2half2(__ushort_as_half(0));
    FragCh ch[G::WNT][NB];
    float2 acc[G::WNT][NB][2];   // [t][h][f]: column t * 16 + f * 8 + g, batch rows 8h + 2q, 8h + 2q + 1
    #pragma unroll
    for (int t = 0; t < G::WNT; ++t)
        #pragma unroll
        for (int h = 0; h < NB; ++h) {
            ch[t][h][0] = ch[t][h][1] = hzero;
            acc[t][h][0] = acc[t][h][1] = make_float2(0.f, 0.f);
        }

    for (int ib = 0; ib < myn; ib += G::PF) {
        #pragma unroll
        for (int d = 0; d < G::PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;
            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
            uint32_t alo[NB], ahi[NB];
            #pragma unroll
            for (int h = 0; h < NB; ++h) {
                alo[h] = r_ok[h] ? As[h][i * 8] : 0u;
                ahi[h] = r_ok[h] ? As[h][i * 8 + 4] : 0u;
            }
            if (i + G::PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + G::PF, l);
            }
            #pragma unroll
            for (int t = 0; t < G::WNT; ++t) {
                FragB f0, f1;
                codec.tile(bw, t, f0, f1);
                #pragma unroll
                for (int h = 0; h < NB; ++h) xw_mma(f0, f1, alo[h], ahi[h], ch[t][h]);
            }
            if ((d + 1) % G::FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < G::WNT; ++t)
                    #pragma unroll
                    for (int h = 0; h < NB; ++h)
                        #pragma unroll
                        for (int f = 0; f < 2; ++f) {
                            acc[t][h][f].x += __low2float(ch[t][h][f]);
                            acc[t][h][f].y += __high2float(ch[t][h][f]);
                            ch[t][h][f] = hzero;
                        }
            }
        }
    }

    // k-split warps summed in fixed order through shared memory (reusing the A stage): red[warp][row][column]
    __syncthreads();
    auto red = reinterpret_cast<float (*)[XW_MAX_N][G::COLS]>(xw_smem);
    {
        const int g = lane >> 2, q2 = 2 * (lane & 3);
        #pragma unroll
        for (int t = 0; t < G::WNT; ++t)
            #pragma unroll
            for (int h = 0; h < NB; ++h)
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + g;
                    red[warp][8 * h + q2 + 0][col] = acc[t][h][f].x;
                    red[warp][8 * h + q2 + 1][col] = acc[t][h][f].y;
                }
    }
    __syncthreads();
    float* out = P.P + (size_t) kp * L.n * P.rows + group * G::COLS;
    for (int idx = threadIdx.x; idx < G::COLS * L.n; idx += G::THREADS) {
        const int r = idx / G::COLS, c = idx % G::COLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < G::WK; ++j) sum += red[j][r][c];
        out[(size_t) r * P.rows + c] = sum;
    }

    __syncthreads();   // the next item restages A over red
}

// Persistent: a resident grid walks the items, so block start-up and drain are paid once per launch.
template <class Codec, int NB>
__global__ __launch_bounds__(XwGeom::THREADS) void xw_gemv(XwList L) {
    for (int item = blockIdx.x; item < L.items; item += gridDim.x) xw_item<Codec, NB>(L, item);
}

// One gemv launch with NB batch halves; the resident grid is sized once per instantiation.
template <class Codec, int NB>
void xw_gemv_launch(const XwList& L, size_t smem, cudaStream_t s) {
    static int resident = [] {
        const size_t most = xw_gemv_smem(8 * NB, XW_MAX_ITEM_SLICES);   // NB = 2: above the 48 KB default
        cudaFuncSetAttribute(xw_gemv<Codec, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(most));
        int b = 1, sms = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, xw_gemv<Codec, NB>, XwGeom::THREADS, most);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        return max(1, b) * sms;
    }();
    xw_gemv<Codec, NB><<<min(L.items, resident), XwGeom::THREADS, smem, s>>>(L);
}

// ---------------------------------------------------------------------------------------------------------
// host side

inline int xw_warp_blocks(const XwList& L, bool in) {
    int warps = 0;
    for (int i = 0; i < L.count; ++i) warps += L.n * (in ? L.cols : L.p[i].rows) / 128;
    return ceil_div(warps, 8);
}

// The three launches. Every part of L must use Codec (the dispatcher splits mixed lists).
template <class Codec>
void xw_matvec(const XwList& L, const half* x, __nv_bfloat16* y, int ldy, cudaStream_t s) {
    xw_had_in<<<xw_warp_blocks(L, true), 256, 0, s>>>(L, x);
    int ks_max = 0;
    for (int i = 0; i < L.count; ++i) ks_max = max(ks_max, L.cols / 16 / L.p[i].kparts);
    const size_t smem = xw_gemv_smem(L.n, ks_max);
    if (L.n > 8) xw_gemv_launch<Codec, 2>(L, smem, s);
    else xw_gemv_launch<Codec, 1>(L, smem, s);
    xw_out<<<xw_warp_blocks(L, false), 256, 0, s>>>(L, y, ldy);
}

}  // namespace x3
}  // namespace tqllm

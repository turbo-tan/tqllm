#pragma once
// moe_window.cuh: routed-expert matvec on X3 banks with the window GEMV. One part per (token, slot): the build
// kernel reads the routing ids on the device and writes one XwMoeList per codec of the bank (an EXL3 pack mixes K
// per expert), then each codec runs the three window launches over its list read from device memory. The k split
// depends on the expert shape only, so a token's output is the same at every batch size.
#include "gemv_window.cuh"

namespace tqllm {
namespace x3 {

// Single thread: parts in (token, slot) order, so each list's item order is fixed for a given routing.
static __global__ void xw_moe_build(const int* __restrict__ ids, int count, int topk, bool per_slot_input,
                                    const XwExpert* __restrict__ bank, int rows, int cols, int kparts, int ncodec,
                                    const half* __restrict__ x, half* A, float* P, XwMoeList* lists) {
    if (threadIdx.x != 0) return;
    for (int c = 0; c < ncodec; ++c) {
        lists[c].count = 0; lists[c].cols = cols; lists[c].n = 1; lists[c].items = 0;
    }
    const int items = (rows / XwGeom::COLS) * kparts;
    for (int i = 0; i < count; ++i) {
        const XwExpert& e = bank[ids[i]];
        XwMoeList& L = lists[e.codec];
        XwPart& p = L.p[L.count++];
        p.B = e.B; p.suh = e.suh; p.svh = e.svh;
        p.rows = rows; p.kparts = kparts; p.item0 = L.items; p.out_off = i * rows;
        p.x = x + (size_t) (per_slot_input ? i : i / topk) * cols;
        p.A = A + (size_t) i * cols;
        p.P = P + (size_t) i * kparts * rows;
        L.items += items;
    }
}

// Verify blocks (n > 1 tokens): one part per distinct expert, all n columns (the window GEMV's column result does not
// depend on n), so an expert routed by several tokens is read once. slot[a] = the part of assignment a. Gate / up read
// x [n][cols] directly; the down projection's per-assignment inputs are staged per part first (xw_moe_stage).
static __global__ void xw_moe_build_dedup(const int* __restrict__ ids, int n, int topk, const XwExpert* __restrict__ bank,
                                          int rows, int cols, int kparts, int ncodec, const half* __restrict__ x,
                                          half* staged, half* A, float* P, int* slot, int* part_expert, int* nparts,
                                          XwMoeList* lists) {
    if (threadIdx.x != 0) return;
    for (int c = 0; c < ncodec; ++c) {
        lists[c].count = 0; lists[c].cols = cols; lists[c].n = n; lists[c].items = 0;
    }
    const int items = (rows / XwGeom::COLS) * kparts;
    int U = 0;
    for (int a = 0; a < n * topk; ++a) {
        const int e = ids[a];
        int u = 0;
        while (u < U && part_expert[u] != e) ++u;
        if (u == U) {
            part_expert[U++] = e;
            const XwExpert& ex = bank[e];
            XwMoeList& L = lists[ex.codec];
            XwPart& p = L.p[L.count++];
            p.B = ex.B; p.suh = ex.suh; p.svh = ex.svh;
            p.rows = rows; p.kparts = kparts; p.item0 = L.items; p.out_off = u * rows;
            p.x = staged ? staged + (size_t) u * n * cols : x;
            p.A = A + (size_t) u * n * cols;
            p.P = P + (size_t) u * kparts * n * rows;
            L.items += items;
        }
        slot[a] = u;
    }
    *nparts = U;
}

// staged[u][t] = x[a] for the assignment a of token t routed to part u, zero when t did not route to it
static __global__ void xw_moe_stage(const half* __restrict__ x, const int* __restrict__ slot, int n, int topk, int cols,
                                    const int* __restrict__ nparts, half* __restrict__ staged) {
    const int u = blockIdx.y, t = blockIdx.z;
    if (u >= *nparts) return;
    int a = -1;
    for (int s = 0; s < topk; ++s) if (slot[t * topk + s] == u) a = t * topk + s;
    half* o = staged + ((size_t) u * n + t) * cols;
    for (int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8; c < cols; c += gridDim.x * blockDim.x * 8)
        *reinterpret_cast<uint4*>(o + c) = a < 0 ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4*>(x + (size_t) a * cols + c);
}

// y[a][rows] = Y[t][slot[a]][rows] (Y = the parts' outputs, row stride ldy = parts * rows)
static __global__ void xw_moe_unslot(const __nv_bfloat16* __restrict__ Y, const int* __restrict__ slot, int topk, int rows,
                                     int ldy, __nv_bfloat16* __restrict__ y) {
    const int a = blockIdx.y, t = a / topk;
    const __nv_bfloat16* src = Y + (size_t) t * ldy + (size_t) slot[a] * rows;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < rows; c += gridDim.x * blockDim.x) y[(size_t) a * rows + c] = src[c];
}

// The list is read once per block into shared memory; the window bodies then run unchanged.
static __global__ void xw_moe_had_in(const XwMoeList* __restrict__ Lg) {
    __shared__ XwMoeList L;
    for (int i = threadIdx.x; i < int(sizeof(XwMoeList) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(&L)[i] = reinterpret_cast<const uint32_t*>(Lg)[i];
    __syncthreads();
    xw_had_in_body(L, nullptr);
}

static __global__ void xw_moe_out(const XwMoeList* __restrict__ Lg, __nv_bfloat16* __restrict__ y, int ldy) {
    __shared__ XwMoeList L;
    for (int i = threadIdx.x; i < int(sizeof(XwMoeList) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(&L)[i] = reinterpret_cast<const uint32_t*>(Lg)[i];
    __syncthreads();
    xw_out_body(L, y, ldy);
}

template <class Codec, int NB>
__global__ __launch_bounds__(XwGeom::THREADS) void xw_moe_gemv(const XwMoeList* __restrict__ Lg) {
    __shared__ XwMoeList L;
    for (int i = threadIdx.x; i < int(sizeof(XwMoeList) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(&L)[i] = reinterpret_cast<const uint32_t*>(Lg)[i];
    __syncthreads();
    for (int item = blockIdx.x; item < L.items; item += gridDim.x) xw_item<Codec, NB>(L, item);
}

template <class Codec, int NB>
void xw_moe_gemv_launch(const XwMoeList* L, int items, size_t smem, cudaStream_t s) {
    static int resident = [] {
        const size_t most = xw_gemv_smem(8 * NB, XW_MAX_ITEM_SLICES);
        cudaFuncSetAttribute(xw_moe_gemv<Codec, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(most));
        int b = 1, sms = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, xw_moe_gemv<Codec, NB>, XwGeom::THREADS, most);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        return max(1, b) * sms;
    }();
    xw_moe_gemv<Codec, NB><<<min(items, resident), XwGeom::THREADS, smem, s>>>(L);
}

// One codec's list (device memory) of at most `parts` parts of rows x cols with n columns each: the three launches
// sized for the bound; part u's column t lands at y[t * ldy + u * rows].
template <class Codec>
void xw_moe_matvec(const XwMoeList* L, int parts, int n, int rows, int cols, int kparts, __nv_bfloat16* y, int ldy,
                   cudaStream_t s) {
    const int in_blocks = ceil_div(parts * n * cols / 128, 8), out_blocks = ceil_div(parts * n * rows / 128, 8);
    xw_moe_had_in<<<in_blocks, 256, 0, s>>>(L);
    const int items = parts * (rows / XwGeom::COLS) * kparts;
    const size_t smem = xw_gemv_smem(n, cols / 16 / kparts);
    if (n > 8) xw_moe_gemv_launch<Codec, 2>(L, items, smem, s);
    else xw_moe_gemv_launch<Codec, 1>(L, items, smem, s);
    xw_moe_out<<<out_blocks, 256, 0, s>>>(L, y, ldy);
}

// Prompt chunks: one part per (expert, pass of up to XW_MAX_N of its assignments), every part at n = XW_MAX_N (a short
// pass stages zero columns); had_in reads x through the pass's assignment ids and out writes each valid column to its
// assignment's row of y, so the gemv runs one launch for many experts with no gather / scatter copies. The k split is
// the routed-expert one (shape only), so each column equals expert_matvec bit for bit.
static __global__ void xw_pmoe_had_in(const XwMoeList* __restrict__ Lg, const half* __restrict__ x) {
    __shared__ XwMoeList L;
    for (int i = threadIdx.x; i < int(sizeof(XwMoeList) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(&L)[i] = reinterpret_cast<const uint32_t*>(Lg)[i];
    __syncthreads();
    XwWarpTask w;
    if (!xw_warp_task(L, L.cols, w)) return;
    const XwPart& P = L.p[w.part];
    half* ar = P.A + (size_t) w.t * L.cols;
    if (w.t >= P.nv) {
        *reinterpret_cast<uint2*>(ar + w.c0) = make_uint2(0u, 0u);
        return;
    }
    xw_had_row(x + (size_t) (P.gsrc[w.t] / P.sdiv) * L.cols, P.suh, ar, w.c0);
}

static __global__ void xw_pmoe_out(const XwMoeList* __restrict__ Lg, __nv_bfloat16* __restrict__ y) {
    __shared__ XwMoeList L;
    for (int i = threadIdx.x; i < int(sizeof(XwMoeList) / 4); i += blockDim.x)
        reinterpret_cast<uint32_t*>(&L)[i] = reinterpret_cast<const uint32_t*>(Lg)[i];
    __syncthreads();
    XwWarpTask w;
    if (!xw_warp_task(L, -1, w)) return;
    const XwPart& P = L.p[w.part];
    if (w.t >= P.nv) return;
    xw_out_row(P, L.n, w.t, w.c0, y + (size_t) P.gsrc[w.t] * P.rows + w.c0);
}

// The gemv over a prompt list. An item = (part, 512-column group, k part): the 8 warps split the columns (64 each, the
// decode path's 4 tiles x 2 batch halves per warp) and every warp runs the item's whole k range, so one staged A slice
// feeds 512 columns (the decode geometry splits k across the warps and feeds 64; there the 49 KB A stage outweighed the
// 24 KB of K4 weights it served). No cross-warp reduction: each warp writes its columns' partial sums.
struct XwOnePart {
    XwPart p[1];
    int count, cols, n;
};
template <class Codec>
__device__ __forceinline__ void xw_pmoe_item(const XwPart& P, int cols, int n, int item) {
    using G = XwGeom;
    constexpr int NB = 2;
    constexpr int TWORDS = Codec::u32words;
    constexpr int LOADS = Codec::loads(G::WNT);
    constexpr int LSTRIDE = Codec::ld_stride;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ntiles = P.rows / 16;
    const int groups = P.rows / XW_PMOE_COLS;
    const int group = item % groups, kp = item / groups;
    const int k_slices = cols / 16 / P.kparts;
    const int ks0 = kp * k_slices;
    const int tile0 = (group * 8 + warp) * G::WNT;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const uint32_t* bp = P.B + (size_t) x3_tile_offset(ks0, tile0, ntiles) * TWORDS + lane;
    auto ld_b = [&](int i, int l) -> uint32_t {
        if constexpr (Codec::ld_lanes < 32)
            return lane < Codec::ld_lanes ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };
    uint32_t pf[G::PF][LOADS];
    #pragma unroll
    for (int d = 0; d < G::PF; ++d)
        if (d < k_slices) {
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(d, l);
        }
    extern __shared__ __align__(16) uint32_t xw_smem[];
    const int a_words = k_slices * 8, a_stride = xw_a_stride(k_slices);
    {
        const uint4* src = reinterpret_cast<const uint4*>(P.A) + (size_t) kp * a_words / 4;
        for (int idx = threadIdx.x; idx < n * a_words / 4; idx += G::THREADS) {
            const int r = idx / (a_words / 4), c = idx % (a_words / 4);
            reinterpret_cast<uint4*>(xw_smem + r * a_stride)[c] = src[(size_t) r * (cols / 8) + c];
        }
        __syncthreads();
    }
    const int r0 = lane >> 2;
    bool r_ok[NB];
    const uint32_t* As[NB];
    #pragma unroll
    for (int h = 0; h < NB; ++h) {
        r_ok[h] = r0 + 8 * h < n;
        As[h] = xw_smem + (r0 + 8 * h) * a_stride + (lane & 3);
    }
    const typename Codec::Lane codec(lane);
    const half2 hzero = __half2half2(__ushort_as_half(0));
    FragCh ch[G::WNT][NB];
    float2 acc[G::WNT][NB][2];
    #pragma unroll
    for (int t = 0; t < G::WNT; ++t)
        #pragma unroll
        for (int h = 0; h < NB; ++h) {
            ch[t][h][0] = ch[t][h][1] = hzero;
            acc[t][h][0] = acc[t][h][1] = make_float2(0.f, 0.f);
        }
    for (int ib = 0; ib < k_slices; ib += G::PF) {
        #pragma unroll
        for (int d = 0; d < G::PF; ++d) {
            const int i = ib + d;
            if (i >= k_slices) break;
            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];
            uint32_t alo[NB], ahi[NB];
            #pragma unroll
            for (int h = 0; h < NB; ++h) {
                alo[h] = r_ok[h] ? As[h][i * 8] : 0u;
                ahi[h] = r_ok[h] ? As[h][i * 8 + 4] : 0u;
            }
            if (i + G::PF < k_slices) {
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
            if ((d + 1) % G::FOLD == 0 || i + 1 == k_slices) {
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
    float* out = P.P + (size_t) kp * n * P.rows + tile0 * 16;
    {
        const int g = lane >> 2, q2 = 2 * (lane & 3);
        #pragma unroll
        for (int t = 0; t < G::WNT; ++t)
            #pragma unroll
            for (int h = 0; h < NB; ++h)
                #pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int col = t * 16 + f * 8 + g, r = 8 * h + q2;
                    if (r < n) out[(size_t) r * P.rows + col] = acc[t][h][f].x;
                    if (r + 1 < n) out[(size_t) (r + 1) * P.rows + col] = acc[t][h][f].y;
                }
    }
    __syncthreads();   // the next item restages A
}
template <class Codec>
__global__ __launch_bounds__(XwGeom::THREADS) void xw_pmoe_gemv(const XwMoeList* __restrict__ Lg) {
    __shared__ XwPart P;
    __shared__ int base;
    const int items = Lg->items;
    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        if (threadIdx.x == 0) {
            int lo = 0, hi = Lg->count - 1;
            while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (Lg->p[mid].item0 <= item) lo = mid; else hi = mid - 1; }
            P = Lg->p[lo];
            base = P.item0;
        }
        __syncthreads();
        xw_pmoe_item<Codec>(P, Lg->cols, Lg->n, item - base);   // ends with a barrier: P is free for the next item
    }
}
template <class Codec>
void xw_pmoe_gemv_launch(const XwMoeList* L, int items, size_t smem, cudaStream_t s) {
    // the resident grid for this launch's shared memory (49 KB at MiniMax's 96-slice items: two blocks per SM)
    static const int sms = [] {
        const size_t most = size_t(XW_MAX_N) * xw_a_stride(XW_MAX_ITEM_SLICES) * 4;
        cudaFuncSetAttribute(xw_pmoe_gemv<Codec>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(most));
        int v = 1;
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0);
        return v;
    }();
    static size_t last_smem = 0;
    static int resident = 0;
    if (smem != last_smem) {
        int b = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, xw_pmoe_gemv<Codec>, XwGeom::THREADS, smem);
        resident = max(1, b) * sms;
        last_smem = smem;
    }
    xw_pmoe_gemv<Codec><<<min(items, resident), XwGeom::THREADS, smem, s>>>(L);
}

// One list (device memory) of `parts` prompt parts: x [tokens or assignments][cols], y [assignments][rows].
template <class Codec>
void xw_pmoe_matvec(const XwMoeList* L, int parts, int rows, int cols, int kparts, const half* x, __nv_bfloat16* y,
                    cudaStream_t s) {
    constexpr int n = XW_MAX_N;
    xw_pmoe_had_in<<<ceil_div(parts * n * cols / 128, 8), 256, 0, s>>>(L, x);
    xw_pmoe_gemv_launch<Codec>(L, parts * (rows / XW_PMOE_COLS) * kparts, size_t(n) * xw_a_stride(cols / 16 / kparts) * 4, s);
    xw_pmoe_out<<<ceil_div(parts * n * rows / 128, 8), 256, 0, s>>>(L, y);
}

// Prompt path helpers: an expert's assignments (groups[0 .. count), assignment a = token * topk + slot) gathered into
// contiguous rows for one GEMM, and its output rows scattered back to y [assignment][rows].
static __global__ void xw_moe_gather(const half* __restrict__ x, const int* __restrict__ group, int count, int topk,
                                     bool per_slot_input, int cols, half* __restrict__ out) {
    const int r = blockIdx.y;
    const int a = group[r], src = per_slot_input ? a : a / topk;
    for (int c = (blockIdx.x * blockDim.x + threadIdx.x) * 8; c < cols; c += gridDim.x * blockDim.x * 8)
        *reinterpret_cast<uint4*>(out + (size_t) r * cols + c) = *reinterpret_cast<const uint4*>(x + (size_t) src * cols + c);
}
static __global__ void xw_moe_scatter_bf16(const __nv_bfloat16* __restrict__ in, const int* __restrict__ group, int rows,
                                           __nv_bfloat16* __restrict__ y) {
    const int r = blockIdx.y, a = group[r];
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < rows; c += gridDim.x * blockDim.x)
        y[(size_t) a * rows + c] = in[(size_t) r * rows + c];
}
static __global__ void xw_moe_scatter(const float* __restrict__ in, const int* __restrict__ group, int rows,
                                      __nv_bfloat16* __restrict__ y) {
    const int r = blockIdx.y, a = group[r];
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < rows; c += gridDim.x * blockDim.x)
        y[(size_t) a * rows + c] = __float2bfloat16(in[(size_t) r * rows + c]);
}

}  // namespace x3
}  // namespace tqllm

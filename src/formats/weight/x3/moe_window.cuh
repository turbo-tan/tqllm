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
        p.aidx = nullptr; p.xaidx = nullptr;
        L.items += items;
    }
}

// Verify blocks (n > 1 tokens): one part per distinct expert, all n columns (the window GEMV's column result does not
// depend on n), so an expert routed by several tokens is read once. Its inverse aindex[u * n + t] = the assignment a of
// token t that routed to part u (-1: token t did not route there), so the out launch writes each kept column straight at
// y[a][rows]. Gate / up read x [n][cols] directly; the down projection's per-assignment inputs are read through the same
// map (xaidx), so they are never staged into a per-part copy first.
// The dedup walk of one verify block, run by the whole block instead of by one thread. The serial version read the
// routing and its own `part_expert` list back through global memory: for each of the n * topk slots a dependent scan
// over the distinct experts found so far, one block of one active thread. The nsys map of a verify step (task note,
// head 27e1d7a) priced these builds at 2.2 ms of a 143 ms step, 120 calls per step. Same walk, same order:
// first[a] = no earlier slot routed this expert, u[a] = its part index in first-seen order, pos[a] = the slot it takes
// in its codec's list. aindex keeps the stride n of the parts that read it, and the slots are written in slot order, so
// a token routing the same expert twice still lands on its last assignment.
constexpr int XW_DEDUP_THREADS = 64;   // >= XW_MOE_MAX_PARTS slots

struct XwDedupPlan {
    int ids[XW_MOE_MAX_PARTS];
    int u[XW_MOE_MAX_PARTS];
    int first[XW_MOE_MAX_PARTS];
    int codec0[XW_MOE_MAX_PARTS];    // the slot's expert in bank0 / bank1 (bank1: the up half of a pair)
    int codec1[XW_MOE_MAX_PARTS];
    int pos0[XW_MOE_MAX_PARTS];
    int pos1[XW_MOE_MAX_PARTS];
    int aindex[XW_MOE_MAX_PARTS * XW_MAX_N];
    int groups;
};

// bank1 == nullptr: one bank (the down projection), one part per distinct expert. With a pair, gate and up append two
// parts per expert to the codec lists (the same codec for both whenever the banks are pairable), so a part's slot
// counts both halves of every earlier expert. Everything below is shared-memory ALU over at most 64 slots.
__device__ __forceinline__ void xw_dedup_plan(const int* __restrict__ ids, int n, int topk,
                                              const XwExpert* __restrict__ bank0, const XwExpert* __restrict__ bank1,
                                              XwDedupPlan& pl) {
    const int t = threadIdx.x, slots = n * topk;
    const bool pair = bank1 != nullptr;
    if (t < slots) pl.ids[t] = ids[t];
    __syncthreads();
    if (t < slots) {
        const int e = pl.ids[t];
        int f = 1;
        for (int s = 0; s < t; ++s) f &= (pl.ids[s] != e);
        pl.first[t] = f;
    }
    __syncthreads();
    if (t < slots) {
        const int e = pl.ids[t];
        int rank = 0, seen = 0;
        for (int s = 0; s < t; ++s)
            if (pl.first[s]) { if (pl.ids[s] == e) seen = rank; ++rank; }
        pl.u[t] = pl.first[t] ? rank : seen;
        pl.codec0[t] = bank0[e].codec;
        pl.codec1[t] = pair ? bank1[e].codec : pl.codec0[t];
    }
    __syncthreads();
    if (t < slots) {
        const int c0 = pl.codec0[t], c1 = pl.codec1[t];
        int p0 = 0, p1 = 0;
        for (int s = 0; s < t; ++s) {
            if (!pl.first[s]) continue;
            p0 += (pl.codec0[s] == c0) + (pair && pl.codec1[s] == c0);
            p1 += (pl.codec1[s] == c1) + (pair && pl.codec0[s] == c1);
        }
        // the up part of a pair follows the gate part of the same expert in the list when they share a codec
        pl.pos0[t] = p0; pl.pos1[t] = p1 + (pair && c0 == c1);
    }
    for (int i = t; i < XW_MOE_MAX_PARTS * n; i += XW_DEDUP_THREADS) pl.aindex[i] = -1;
    __syncthreads();
    if (t == 0) {
        int g = 0;
        for (int s = 0; s < slots; ++s) g += pl.first[s];
        pl.groups = g;
        for (int a = 0; a < slots; ++a) pl.aindex[pl.u[a] * n + a / topk] = a;
    }
    __syncthreads();
}

static __global__ void xw_moe_build_dedup(const int* __restrict__ ids, int n, int topk, const XwExpert* __restrict__ bank,
                                          int rows, int cols, int kparts, int ncodec, const half* __restrict__ x,
                                          bool per_slot_input, half* A, float* P, int* aindex, XwMoeList* lists) {
    __shared__ XwDedupPlan pl;
    xw_dedup_plan(ids, n, topk, bank, nullptr, pl);
    const int slots = n * topk, items = (rows / XwGeom::COLS) * kparts;
    for (int c = threadIdx.x; c < ncodec; c += XW_DEDUP_THREADS) {
        int cnt = 0;
        for (int s = 0; s < slots; ++s) cnt += (pl.first[s] && pl.codec0[s] == c);
        lists[c].count = cnt; lists[c].cols = cols; lists[c].n = n; lists[c].items = cnt * items;
    }
    if (threadIdx.x < slots && pl.first[threadIdx.x]) {
        const int u = pl.u[threadIdx.x], pos = pl.pos0[threadIdx.x];
        const XwExpert& ex = bank[pl.ids[threadIdx.x]];
        XwPart& p = lists[pl.codec0[threadIdx.x]].p[pos];
        p.B = ex.B; p.suh = ex.suh; p.svh = ex.svh;
        p.rows = rows; p.kparts = kparts; p.item0 = pos * items;
        p.out_off = 0;   // with aidx the row of y comes from the map; out_off only separates the halves of a pair
        p.x = x;
        p.A = A + (size_t) u * n * cols;
        p.P = P + (size_t) u * kparts * n * rows;
        p.aidx = aindex + (size_t) u * n;
        p.xaidx = per_slot_input ? aindex + (size_t) u * n : nullptr;
    }
    for (int i = threadIdx.x; i < pl.groups * n; i += XW_DEDUP_THREADS) aindex[i] = pl.aindex[i];
}

// gate and up of one layer in ONE pass over the routing (they share the ids, the shape and the codecs): two banks, two
// parts per routed slot, the up part's out_off addressing the up half of the caller's one output allocation
// (up_base = yb - ya in elements). The A / P slots of a pair are adjacent (2i + m). Bit-identical to two calls: every
// part keeps its own items, k split and partials, and the out launch sums each part's own P.
static __global__ void xw_moe_build2(const int* __restrict__ ids, int count, int topk,
                                     const XwExpert* __restrict__ bankA, const XwExpert* __restrict__ bankB,
                                     int rows, int cols, int kparts, int ncodec, const half* __restrict__ x, int up_base,
                                     half* A, float* P, XwMoeList* lists) {
    if (threadIdx.x != 0) return;
    for (int c = 0; c < ncodec; ++c) {
        lists[c].count = 0; lists[c].cols = cols; lists[c].n = 1; lists[c].items = 0;
    }
    const int items = (rows / XwGeom::COLS) * kparts;
    for (int i = 0; i < count; ++i) {
        for (int m = 0; m < 2; ++m) {
            const XwExpert& e = (m ? bankB : bankA)[ids[i]];
            XwMoeList& L = lists[e.codec];
            XwPart& p = L.p[L.count++];
            p.B = e.B; p.suh = e.suh; p.svh = e.svh;
            p.rows = rows; p.kparts = kparts; p.item0 = L.items; p.out_off = i * rows + (m ? up_base : 0);
            p.x = x + (size_t) (i / topk) * cols;
            p.A = A + (size_t) (2 * i + m) * cols;
            p.P = P + (size_t) (2 * i + m) * kparts * rows;
            p.aidx = nullptr; p.xaidx = nullptr;
            L.items += items;
        }
    }
}

// The same pair for a verify block: one part per (distinct expert, matrix) with all n columns. Gate and up read x
// [n][cols] directly (never a staged input), so the pair shares one aindex row and writes the two halves of y.
static __global__ void xw_moe_build_dedup2(const int* __restrict__ ids, int n, int topk,
                                           const XwExpert* __restrict__ bankA, const XwExpert* __restrict__ bankB,
                                           int rows, int cols, int kparts, int ncodec, const half* __restrict__ x,
                                           int up_base, half* A, float* P, int* aindex, XwMoeList* lists) {
    __shared__ XwDedupPlan pl;
    xw_dedup_plan(ids, n, topk, bankA, bankB, pl);
    const int slots = n * topk, items = (rows / XwGeom::COLS) * kparts;
    for (int c = threadIdx.x; c < ncodec; c += XW_DEDUP_THREADS) {
        int cnt = 0;
        for (int s = 0; s < slots; ++s)
            if (pl.first[s]) cnt += (pl.codec0[s] == c) + (pl.codec1[s] == c);   // gate and up share one codec space
        lists[c].count = cnt; lists[c].cols = cols; lists[c].n = n; lists[c].items = cnt * items;
    }
    if (threadIdx.x < slots && pl.first[threadIdx.x]) {
        const int e = pl.ids[threadIdx.x], u = pl.u[threadIdx.x];
        for (int m = 0; m < 2; ++m) {
            const int pos = (m ? pl.pos1[threadIdx.x] : pl.pos0[threadIdx.x]);
            const XwExpert& w = (m ? bankB : bankA)[e];
            XwPart& p = lists[(m ? pl.codec1[threadIdx.x] : pl.codec0[threadIdx.x])].p[pos];
            p.B = w.B; p.suh = w.suh; p.svh = w.svh;
            p.rows = rows; p.kparts = kparts; p.item0 = pos * items; p.out_off = (m ? up_base : 0);
            p.x = x;
            p.A = A + (size_t) (2 * u + m) * n * cols;
            p.P = P + (size_t) (2 * u + m) * kparts * n * rows;
            p.aidx = aindex + (size_t) u * n;
            p.xaidx = nullptr;   // gate and up read the token's own row of x
        }
    }
    for (int i = threadIdx.x; i < pl.groups * n; i += XW_DEDUP_THREADS) aindex[i] = pl.aindex[i];
}

// The list is read once per block into shared memory; the window bodies then run unchanged. Only the header and the
// `count` parts in use are copied: a whole XwMoeList is 5136 B (1284 words) whatever the count is, and these launches
// run one block per few thousand output elements, so the unused 64-part tail was the bulk of every block's start-up.
template <class List>
__device__ __forceinline__ void xw_load_list(const List* __restrict__ Lg, List& L) {
    if (threadIdx.x == 0) { L.count = Lg->count; L.cols = Lg->cols; L.n = Lg->n; L.items = Lg->items; }
    __syncthreads();
    for (int i = threadIdx.x; i < L.count; i += blockDim.x) L.p[i] = Lg->p[i];
    __syncthreads();
}

static __global__ void xw_moe_had_in(const XwMoeList* __restrict__ Lg) {
    __shared__ XwMoeList L;
    xw_load_list(Lg, L);
    xw_had_in_body(L, nullptr);
}

// A verify list writes each kept column straight at its assignment's row of y (p.aidx), so there is no part scratch
// between the gemv and the caller's output: the same bf16 values in the same addresses the unslot pass used to copy.
static __global__ void xw_moe_out(const XwMoeList* __restrict__ Lg, __nv_bfloat16* __restrict__ y, int ldy) {
    __shared__ XwMoeList L;
    xw_load_list(Lg, L);
    XwWarpTask w;
    if (!xw_warp_task(L, -1, w)) return;
    const XwPart& P = L.p[w.part];
    if (P.aidx) {
        const int a = P.aidx[w.t];
        if (a < 0) return;   // token t did not route to this expert: its column is computed for the mma shape only
        // out_off is 0, or the row offset of the up half when gate and up ran as one call
        xw_out_row(P, L.n, w.t, w.c0, y + P.out_off + (size_t) a * P.rows + w.c0);
        return;
    }
    xw_out_row(P, L.n, w.t, w.c0, y + (size_t) w.t * ldy + P.out_off + w.c0);
}

// The gemv reads six fields of a part. Its list copy is compact (40 B per part, not the 80 B XwPart) and lives at the
// front of the block's dynamic shared memory, sized for the parts the launch has: static shared memory would be there
// whatever n is, and 16 rows x 192 slices (96.3 KB) plus a 5.1 KB XwMoeList is over the 99 KB a block may opt into.
struct XwGemvPart {
    const uint32_t* B;
    half* A;
    float* P;
    int rows, kparts, item0;
};
struct XwGemvList {
    int count, cols, n, items;
    XwGemvPart p[XW_MOE_MAX_PARTS];
};
// shared memory of the copy of `parts` parts, in uint32 words, rounded so the A stage behind it stays 16 B aligned
__host__ __device__ constexpr int xw_gemv_list_words(int parts) {
    return ((16 + parts * int(sizeof(XwGemvPart)) + 15) / 16) * 4;
}

template <class Codec, int NB>
__global__ __launch_bounds__(XwGeom::THREADS) void xw_moe_gemv(const XwMoeList* __restrict__ Lg) {
    extern __shared__ __align__(16) uint32_t xw_smem[];
    XwGemvList& L = *reinterpret_cast<XwGemvList*>(xw_smem);
    if (threadIdx.x == 0) { L.count = Lg->count; L.cols = Lg->cols; L.n = Lg->n; L.items = Lg->items; }
    for (int i = threadIdx.x; i < Lg->count; i += blockDim.x) {
        const XwPart& q = Lg->p[i];
        L.p[i] = {q.B, q.A, q.P, q.rows, q.kparts, q.item0};
    }
    __syncthreads();
    uint32_t* stage = xw_smem + xw_gemv_list_words(L.count);
    for (int item = blockIdx.x; item < L.items; item += gridDim.x) xw_item<Codec, NB>(L, item, stage);
}

template <class Codec, int NB>
void xw_moe_gemv_launch(const XwMoeList* L, int items, size_t smem, cudaStream_t s) {
    static const int sms = [] {
        int v = 1;
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0);
        return max(1, v);
    }();
    static size_t attr_for = 0;
    if (smem > attr_for) {   // the launch's own stage: the 192-slice item is above the 48 KB default
        cudaFuncSetAttribute(xw_moe_gemv<Codec, NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
        attr_for = smem;
    }
    static size_t last_smem = 0;
    static int resident = 0;
    if (smem != last_smem) {   // the resident grid for this launch's shared memory, as the prompt lists do
        int b = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, xw_moe_gemv<Codec, NB>, XwGeom::THREADS, smem);
        resident = max(1, b) * sms;
        last_smem = smem;
    }
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
    const size_t smem = xw_gemv_smem(n, cols / 16 / kparts, n > 8 ? 2 : 1) + size_t(xw_gemv_list_words(parts)) * 4;
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
    constexpr bool PACKED = Codec::packed_wnt(G::WNT);
    constexpr int TWORDS = Codec::u32words;
    constexpr int LOADS = Codec::loads(G::WNT);
    constexpr int LSTRIDE = Codec::ld_stride_wnt(G::WNT);
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
        if constexpr (Codec::ld_lanes_wnt(G::WNT) < 32)
            return lane < Codec::ld_lanes_wnt(G::WNT) ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
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
                codec.template tile<PACKED>(bw, t, f0, f1);
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

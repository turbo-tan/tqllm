#include <algorithm>
#include <cstdint>
#include <cstring>
#include "debug.hpp"
#include <cstdio>
#include <utility>
#include <map>
#include <cstdlib>
#include "quant_kernels.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <climits>
#include <vector>
#include <math_constants.h>
#include <stdexcept>
#include <type_traits>

namespace {

using namespace tqllm_cuda;

// 8-level codebook for N(0,1) and the RHT sign pattern (llama.cpp-tq3 TQ3_0 definitions).
__constant__ float TQ3_SIGNS[32] = {
    +1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1,
};
__device__ __constant__ float TQ3_CODEBOOK[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                                                 0.230106f,  0.725222f,  1.277503f,  1.988943f};

// E3M5: 2^((b >> 5) - 9) * (1 + (b & 31) / 32), 0 for b == 0. Built directly as FP32 bits.
__device__ __forceinline__ float e3m5(uint32_t b) {
    return b ? __uint_as_float((((b >> 5) + 118u) << 23) | ((b & 31u) << 18)) : 0.f;
}

__device__ __forceinline__ float wht32(float v, int lane) {
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, s);
        v = (lane & s) ? o - v : v + o;
    }
    return v * 0.17677669529663687f;  // 1/sqrt(32)
}

#include "quant_tq3.inc"

#include "quant_q6k_i8h.inc"

#include "quant_u4h.inc"

#include "quant_misc.inc"

// one thread per 32-weight block of one expert: the block moves to its tile_expert_rows position
__global__ void tile_rows_k(const char * src, char * dst, int rows, int nb, int bpb, int count, size_t stride) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x, per = int64_t(rows) * nb;
    if (i >= per * count) return;
    const int64_t e = i / per, j = i % per;
    const int r = int(j / nb), s = int(j % nb);
    const int64_t to = (((int64_t(r >> 4) * (nb >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
    const char * a = src + size_t(e) * stride + j * bpb;
    char * b = dst + size_t(e) * stride + to;
    for (int q = 0; q < bpb; ++q) b[q] = a[q];
}
} // namespace

namespace tqllm_cuda {
void k_expert_matvec(int fmt, const uint8_t * w, const void * aux, const __half * x,
                     __nv_bfloat16 * y, int rows, int cols, const int * ids, int n, int topk,
                     bool per_expert_input, cudaStream_t stream) {
    if (!w || !aux || !x || !y || !ids || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 || fmt == 16 ? 128 : fmt == 15 || fmt == 0 ? 64 : 256) || n <= 0 || topk <= 0 || int64_t(n) * topk > 65535)
        throw std::invalid_argument("k_expert_matvec: invalid shape");
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(n * topk));
    const auto * scales = static_cast<const __half2 *>(aux);
    // Whole u4h family, not just K4/K5/K6: GROUPED is an indexing change in the same
    // template the dense kernels use (NCOL 1, MT 1, PF 0), so grouped decode stays
    // bit-identical to a row-sliced dense matvec (tests/moe_device.cu).
    switch (fmt) {
    case 0: u4h_matvec_k<1, __nv_bfloat16, 1, 0, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 1: u4h_matvec_k<1, __nv_bfloat16, 1, 1, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 2: u4h_matvec_k<1, __nv_bfloat16, 1, 2, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 4: u4h_matvec_k<1, __nv_bfloat16, 1, 4, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 5: u4h_matvec_k<1, __nv_bfloat16, 1, 5, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 6: u4h_matvec_k<1, __nv_bfloat16, 1, 6, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 7: u4h_matvec_k<1, __nv_bfloat16, 1, 7, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 8: u4h_matvec_k<1, __nv_bfloat16, 1, 8, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 9: u4h_matvec_k<1, __nv_bfloat16, 1, 9, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 10: u4h_matvec_k<1, __nv_bfloat16, 1, 10, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 11: u4h_matvec_k<1, __nv_bfloat16, 1, 11, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 12: u4h_matvec_k<1, __nv_bfloat16, 1, 12, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 13: u4h_matvec_k<1, __nv_bfloat16, 1, 13, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 14: u4h_matvec_k<1, __nv_bfloat16, 1, 14, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 15: u4h_matvec_k<1, __nv_bfloat16, 1, 15, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    case 16: u4h_matvec_k<1, __nv_bfloat16, 1, 16, 0, true><<<grid, 256, 0, stream>>>(
            w, scales, x, y, rows, cols, 1, nullptr, 0, ids, topk, per_expert_input); break;
    default: throw std::invalid_argument("k_expert_matvec: unsupported format");
    }
}


void k_expert_matvec_dedup(int fmt, const uint8_t * w, const void * aux, const __half * x, __nv_bfloat16 * y,
                           int rows, int cols, const int * dmap, int groups, cudaStream_t stream) {
    if (!w || !aux || !x || !y || !dmap || rows <= 0 || cols <= 0 || cols % (fmt == 14 || fmt == 16 ? 128 : fmt == 15 || fmt == 0 ? 64 : 256) || groups <= 0 || groups > 65535)
        throw std::invalid_argument("k_expert_matvec_dedup: invalid shape");
    // Two 16-row tiles per block (MT 2: per row the same k split and MMA order, bit-identical; GB10 down IQ4_NL 100
    // groups 371 -> 330 us, the paired gate / up launch below stays at MT 1: IQ3_XXS 572 -> 611 us with MT 2)
    // groups beyond the grid are walked by its blocks (a device-built map is padded with slot -1 after its groups)
    const dim3 grid(unsigned((rows + 31) / 32), unsigned(std::min(groups, kDedupGridGroups)));
    const auto * sc = static_cast<const __half2 *>(aux);
#define TQ_DEDUP(F) u4h_matvec_k<tqllm_cuda::kDedupCols, __nv_bfloat16, 2, F, 0, true><<<grid, 256, 0, stream>>>(w, sc, x, y, rows, cols, 0, nullptr, 0, dmap, 1, false, nullptr, nullptr, nullptr, groups)
    switch (fmt) {
        case 0: TQ_DEDUP(0); break;
        case 1: TQ_DEDUP(1); break;
        case 4: TQ_DEDUP(4); break;
        case 5: TQ_DEDUP(5); break;
        case 6: TQ_DEDUP(6); break;
        case 7: TQ_DEDUP(7); break;
        case 10: TQ_DEDUP(10); break;
        case 13: TQ_DEDUP(13); break;
        case 14: TQ_DEDUP(14); break;
        case 15: TQ_DEDUP(15); break;
        case 16: TQ_DEDUP(16); break;
        default: throw std::invalid_argument("k_expert_matvec_dedup: unsupported format");
    }
#undef TQ_DEDUP
}

void set_tiled_experts(bool on) { const int v = on ? 1 : 0; cudaMemcpyToSymbol(c_tiled_experts, &v, sizeof v); }
int tiled_block_bytes(int fmt) { return tiled_bpb(fmt); }
void tile_expert_rows(const char * src, char * dst, int rows, int k, int bpb) {
    const int nb = k / 32;
    for (int r = 0; r < rows; ++r)
        for (int s = 0; s < nb; ++s) {
            const int64_t to = (((int64_t(r >> 4) * (nb >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * bpb;
            std::memcpy(dst + to, src + (int64_t(r) * nb + s) * bpb, size_t(bpb));
        }
}
void tile_expert_rows_device(char * p, int count, int rows, int k, int bpb, cudaStream_t s) {
    if (!p || count <= 0 || rows % 16 || (k / 32) % 4 || bpb <= 0) throw std::invalid_argument("tile_expert_rows_device: invalid shape");
    const int nb = k / 32;
    const size_t stride = size_t(rows) * nb * size_t(bpb);
    void * tmp = nullptr;
    if (cudaMallocAsync(&tmp, stride * size_t(count), s) != cudaSuccess) throw std::runtime_error("tile_expert_rows_device: out of memory");
    cudaMemcpyAsync(tmp, p, stride * size_t(count), cudaMemcpyDeviceToDevice, s);
    const int64_t n = int64_t(rows) * nb * count;
    tile_rows_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(static_cast<const char *>(tmp), p, rows, nb, bpb, count, stride);
    cudaFreeAsync(tmp, s);
}
// gate and up in one launch (grid z = 2): same blocks, same per-row math as two k_expert_matvec_dedup calls
void k_expert_matvec2(int fmt, const uint8_t * wa, const void * auxa, const uint8_t * wb, const void * auxb, const __half * x,
                      __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows, int cols, const int * ids, int n, int topk, cudaStream_t stream) {
    if (!wa || !auxa || !wb || !auxb || !x || !ya || !yb || !ids || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 || fmt == 16 ? 128 : fmt == 15 || fmt == 0 ? 64 : 256) || n <= 0 || topk <= 0 || int64_t(n) * topk > 65535)
        throw std::invalid_argument("k_expert_matvec2: invalid shape");
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(n * topk), 2u);
    const auto * sa = static_cast<const __half2 *>(auxa); const auto * sb = static_cast<const __half2 *>(auxb);
#define TQ_PAIR(F) u4h_matvec_k<1, __nv_bfloat16, 1, F, 0, true><<<grid, 256, 0, stream>>>(wa, sa, x, ya, rows, cols, 1, nullptr, 0, ids, topk, false, wb, sb, yb)
    switch (fmt) {
        case 0: TQ_PAIR(0); break;
        case 1: TQ_PAIR(1); break;
        case 2: TQ_PAIR(2); break;
        case 4: TQ_PAIR(4); break;
        case 5: TQ_PAIR(5); break;
        case 6: TQ_PAIR(6); break;
        case 7: TQ_PAIR(7); break;
        case 8: TQ_PAIR(8); break;
        case 9: TQ_PAIR(9); break;
        case 10: TQ_PAIR(10); break;
        case 11: TQ_PAIR(11); break;
        case 12: TQ_PAIR(12); break;
        case 13: TQ_PAIR(13); break;
        case 14: TQ_PAIR(14); break;
        case 15: TQ_PAIR(15); break;
        case 16: TQ_PAIR(16); break;
        default: throw std::invalid_argument("k_expert_matvec2: unsupported format");
    }
#undef TQ_PAIR
}

// paired dedup launch with the prefetching variant on or off (both bit-identical, test-expert-dedup)
static void dedup2_launch(int fmt, const uint8_t * wa, const void * auxa, const uint8_t * wb, const void * auxb, const __half * x,
                          __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows, int cols, const int * dmap, int groups, bool pf, cudaStream_t stream) {
    const dim3 grid(unsigned((rows + 15) / 16), unsigned(std::min(groups, kDedupGridGroups)), 2u);
    const auto * sa = static_cast<const __half2 *>(auxa); const auto * sb = static_cast<const __half2 *>(auxb);
#define TQ_DEDUP2(F) (pf ? u4h_matvec_k<tqllm_cuda::kDedupCols, __nv_bfloat16, 1, F, 1, true><<<grid, 256, 0, stream>>>(wa, sa, x, ya, rows, cols, 0, nullptr, 0, dmap, 1, false, wb, sb, yb, groups) \
                         : u4h_matvec_k<tqllm_cuda::kDedupCols, __nv_bfloat16, 1, F, 0, true><<<grid, 256, 0, stream>>>(wa, sa, x, ya, rows, cols, 0, nullptr, 0, dmap, 1, false, wb, sb, yb, groups))
    switch (fmt) {
        case 0: TQ_DEDUP2(0); break;
        case 1: TQ_DEDUP2(1); break;
        case 4: TQ_DEDUP2(4); break;
        case 5: TQ_DEDUP2(5); break;
        case 6: TQ_DEDUP2(6); break;
        case 7: TQ_DEDUP2(7); break;
        case 10: TQ_DEDUP2(10); break;
        case 13: TQ_DEDUP2(13); break;
        case 14: TQ_DEDUP2(14); break;
        case 15: TQ_DEDUP2(15); break;
        case 16: TQ_DEDUP2(16); break;
        default: throw std::invalid_argument("k_expert_matvec_dedup2: unsupported format");
    }
#undef TQ_DEDUP2
}

// group count from which the paired launch takes the prefetching variant, per u4h format: measured at start-up on the
// real banks (calibrate_dedup_pf); 160 until then (GB10 IQ formats, untiled: 99 groups -2..-6 %, 30 groups +6..+9 %)
static int g_dedup_pf_min[32] = {};
static int dedup_pf_min(int fmt) { const int v = fmt >= 0 && fmt < 32 ? g_dedup_pf_min[fmt] : 0; return v > 0 ? v : 160; }
void set_dedup_pf_min(int fmt, int groups) { if (fmt >= 0 && fmt < 32) g_dedup_pf_min[fmt] = groups; }
int dedup_pf_threshold(int fmt) { return dedup_pf_min(fmt); }

void k_expert_matvec_dedup2(int fmt, const uint8_t * wa, const void * auxa, const uint8_t * wb, const void * auxb, const __half * x,
                            __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows, int cols, const int * dmap, int groups, cudaStream_t stream) {
    if (!wa || !auxa || !wb || !auxb || !x || !ya || !yb || !dmap || rows <= 0 || cols <= 0 || cols % (fmt == 14 || fmt == 16 ? 128 : fmt == 15 || fmt == 0 ? 64 : 256) || groups <= 0 || groups > 65535)
        throw std::invalid_argument("k_expert_matvec_dedup2: invalid shape");
    dedup2_launch(fmt, wa, auxa, wb, auxb, x, ya, yb, rows, cols, dmap, groups, groups >= dedup_pf_min(fmt), stream);
}

int calibrate_dedup_pf(int fmt, const uint8_t * wa, const void * auxa, const uint8_t * wb, const void * auxb, int rows, int cols,
                       int slots, cudaStream_t stream) {
    if (fmt < 0 || fmt >= 32 || !wa || !wb || rows <= 0 || cols <= 0 || slots < 16) return dedup_pf_min(fmt);
    constexpr int C = 2, ntok = 8;  // 2 columns per expert (a 4-8 stream verify block), 8 input rows
    std::vector<int> counts;
    for (int g : {16, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384}) if (g <= slots) counts.push_back(g);
    const int gmax = counts.back(), R = 4;  // R disjoint-ish expert sets per count, rotated so the weights come from DRAM
    __half * x = nullptr; __nv_bfloat16 * ya = nullptr, * yb = nullptr; int * dm = nullptr; cudaEvent_t e0 = nullptr, e1 = nullptr;
    const size_t yrows = size_t(gmax) * C;
    bool ok = cudaMalloc(&x, size_t(ntok) * cols * 2) == cudaSuccess && cudaMalloc(&ya, yrows * rows * 2) == cudaSuccess &&
              cudaMalloc(&yb, yrows * rows * 2) == cudaSuccess && cudaMalloc(&dm, size_t(R) * gmax * kDedupStride * 4) == cudaSuccess &&
              cudaEventCreate(&e0) == cudaSuccess && cudaEventCreate(&e1) == cudaSuccess;
    int found = 0;
    if (ok) {
        cudaMemsetAsync(x, 0, size_t(ntok) * cols * 2, stream);
        std::vector<float> t_pf(counts.size()), t_np(counts.size());
        for (size_t ci = 0; ci < counts.size(); ++ci) {
            const int G = counts[ci];
            std::vector<int> map(size_t(R) * G * kDedupStride, -1);
            for (int r = 0; r < R; ++r)
                for (int g = 0; g < G; ++g) {
                    int * e = map.data() + (size_t(r) * G + g) * kDedupStride;
                    e[0] = int((int64_t(r) * G + g) * 37 % slots); e[1] = C;
                    for (int c = 0; c < C; ++c) { e[2 + c] = (g + c) % ntok; e[2 + kDedupCols + c] = g * C + c; }
                }
            cudaMemcpyAsync(dm, map.data(), map.size() * 4, cudaMemcpyHostToDevice, stream);
            for (int v = 0; v < 2; ++v) {
                const bool pf = v == 1;
                auto run = [&](int i) { dedup2_launch(fmt, wa, auxa, wb, auxb, x, ya, yb, rows, cols, dm + size_t(i % R) * G * kDedupStride, G, pf, stream); };
                for (int i = 0; i < 4; ++i) run(i);
                cudaEventRecord(e0, stream);
                for (int i = 0; i < 12; ++i) run(i);
                cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                (pf ? t_pf : t_np)[ci] = ms;
            }
        }
        // the smallest count from which the prefetching variant is faster at every larger measured count
        found = INT32_MAX;
        for (size_t ci = counts.size(); ci-- > 0;) {
            if (t_pf[ci] < 0.98f * t_np[ci]) found = counts[ci]; else break;
        }
        ok = cudaGetLastError() == cudaSuccess;
    }
    cudaFree(x); cudaFree(ya); cudaFree(yb); cudaFree(dm);
    if (e0) cudaEventDestroy(e0); if (e1) cudaEventDestroy(e1);
    if (!ok) { cudaGetLastError(); return dedup_pf_min(fmt); }
    set_dedup_pf_min(fmt, found);
    return found;
}

void k_expert_gemm(int fmt, const uint8_t * w, const void * aux, const __half * x,
                   __nv_bfloat16 * y, int rows, int cols, const int * counts, const int * groups,
                   int capacity, int experts, int topk, bool per_expert_input, cudaStream_t stream) {
    if (!w || !aux || !x || !y || !counts || !groups || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 || fmt == 16 ? 128 : 256) || capacity <= 0 || capacity > 65535 * 64 ||
        experts <= 0 || experts > 65535 || topk <= 0)
        throw std::invalid_argument("k_expert_gemm: invalid shape");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts));
    const auto * scales = static_cast<const __half2 *>(aux);
    // Same family coverage as k_expert_matvec: u4h_gemm_k<FMT, GROUPED> is generic in FMT
    // (its GROUPED branch only reads counts/groups and offsets source_row), so prefill
    // stays bit-identical to a row-sliced dense gemm().
    switch (fmt) {
    case 0: u4h_gemm_k<0, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 1: u4h_gemm_k<1, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 2: u4h_gemm_k<2, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 4: u4h_gemm_k<4, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 5: u4h_gemm_k<5, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 6: u4h_gemm_k<6, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 7: u4h_gemm_k<7, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 8: u4h_gemm_k<8, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 9: u4h_gemm_k<9, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 10: u4h_gemm_k<10, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 11: u4h_gemm_k<11, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 12: u4h_gemm_k<12, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 13: u4h_gemm_k<13, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 14: u4h_gemm_k<14, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 15: u4h_gemm_k<15, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    case 16: u4h_gemm_k<16, true><<<grid, 256, 0, stream>>>(w, scales, x, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); break;
    default: throw std::invalid_argument("k_expert_gemm: unsupported format");
    }
}

// fmt: 0 u4h, 1 k4, 2 k3, 4 iq3, 5 iq3x, 6 k5, 7 k6, 8-12 UD types (u4h_matvec_k's FMT); gemm only: 13 iq4n, 14 iq4l, 15 i8h
void k_expert_gemm_i8(int fmt, const uint8_t * w, const void * aux, const int8_t * xq, const float * xs, const int16_t * xsum,
                      __nv_bfloat16 * y, int rows, int cols, const int * counts, const int * groups,
                      int capacity, int experts, int topk, bool per_expert_input, cudaStream_t stream, int mode) {
    if (!w || !aux || !xq || !xs || !xsum || !y || !counts || !groups || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 || fmt == 16 ? 128 : 256) || capacity <= 0 || capacity > 65535 * 64 ||
        experts <= 0 || experts > 65535 || topk <= 0)
        throw std::invalid_argument("k_expert_gemm_i8: invalid shape");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts));
    const auto * scales = static_cast<const __half2 *>(aux);
#define TQ_I8G(F) do { if (mode >= 3) { const int bn3 = i8_offset<F>() == 0 ? 64 : 128; const dim3 g3(unsigned((rows + 127) / 128), unsigned((capacity + bn3 - 1) / bn3), unsigned(experts)); \
                         if (mode == 4) u4h_i8v3_gemm_k<F, true><<<g3, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, counts, groups, capacity, topk, per_expert_input); \
                         else u4h_i8v3_gemm_k<F, false><<<g3, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, counts, groups, capacity, topk, per_expert_input); } \
                       else if (mode == 2) u4h_i8_gemm_k<F, true, true><<<grid, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); \
                       else u4h_i8_gemm_k<F, true, false><<<grid, 256, 0, stream>>>(w, scales, xq, xs, xsum, y, rows, cols, capacity, rows, counts, groups, capacity, topk, per_expert_input); } while (0)
    switch (fmt) {
        case 1: TQ_I8G(1); break;
        case 4: TQ_I8G(4); break;
        case 5: TQ_I8G(5); break;
        case 6: TQ_I8G(6); break;
        case 7: TQ_I8G(7); break;
        case 10: TQ_I8G(10); break;
        case 13: TQ_I8G(13); break;
        case 14: TQ_I8G(14); break;
        default: throw std::invalid_argument("k_expert_gemm_i8: unsupported format");
    }
#undef TQ_I8G
}
void k_quant_act_rows(const __half * x, const int * rows_of, const int * counts, int cap, int slots, int nrows, int k,
                      int8_t * q, float * s, int16_t * qsum, cudaStream_t stream, int mode) {
    if (k % 32 || !x || !q || !s || !qsum) throw std::invalid_argument("k_quant_act_rows: invalid shape");
    if (!rows_of) {
        if (nrows <= 0) return;
        if (mode == 2 || mode == 4) quant_act_rows_k<true><<<unsigned(nrows), 256, 0, stream>>>(x, nullptr, nullptr, 0, k, q, s, qsum);
        else quant_act_rows_k<false><<<unsigned(nrows), 256, 0, stream>>>(x, nullptr, nullptr, 0, k, q, s, qsum);
    } else if (cap > 0 && slots > 0) {
        if (mode == 2 || mode == 4) quant_act_rows_k<true><<<dim3(unsigned(cap), unsigned(slots)), 256, 0, stream>>>(x, rows_of, counts, cap, k, q, s, qsum);
        else quant_act_rows_k<false><<<dim3(unsigned(cap), unsigned(slots)), 256, 0, stream>>>(x, rows_of, counts, cap, k, q, s, qsum);
    }
}

// dense (non-grouped) int8-activation GEMM, f32 output: the layer-major dense projections (TQFN_DENSE_INT8)
bool u4h_family_gemm_i8_f32(int fmt, const uint8_t * w, const void * aux, const int8_t * xq, const float * xs, const int16_t * xsum,
                            float * y, int rows, int k, int ncols, cudaStream_t s, int ldy, int mode) {
    if (k % 64 || rows <= 0 || ncols <= 0) return false;
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((ncols + 63) / 64));
    const auto * scales = static_cast<const __half2 *>(aux);
#define TQ_I8D(F) do { if (mode == 2) u4h_i8_gemm_k<F, false, true, float><<<grid, 256, 0, s>>>(w, scales, xq, xs, xsum, y, rows, k, ncols, ldy ? ldy : rows, nullptr, nullptr, 0, 1, false); \
                       else u4h_i8_gemm_k<F, false, false, float><<<grid, 256, 0, s>>>(w, scales, xq, xs, xsum, y, rows, k, ncols, ldy ? ldy : rows, nullptr, nullptr, 0, 1, false); } while (0)
    switch (fmt) {
        case 1: TQ_I8D(1); break;
        case 4: TQ_I8D(4); break;
        case 5: TQ_I8D(5); break;
        case 6: TQ_I8D(6); break;
        case 7: TQ_I8D(7); break;
        case 10: TQ_I8D(10); break;
        case 13: TQ_I8D(13); break;
        case 14: TQ_I8D(14); break;
        default: return false;
    }
#undef TQ_I8D
    return true;
}

// gate + up in one grouped launch with the swiglu epilogue (library only): yh [assignment][rows] half
void k_expert_gemm_gateup(int fmt, const uint8_t * wg, const void * auxg, const uint8_t * wu, const void * auxu, const __half * x,
                          __half * yh, int rows, int cols, const int * counts, const int * groups,
                          int capacity, int experts, int topk, cudaStream_t stream) {
    if (!wg || !auxg || !wu || !auxu || !x || !yh || !counts || !groups || rows <= 0 || cols <= 0 ||
        cols % (fmt == 14 || fmt == 16 ? 128 : 256) || capacity <= 0 || capacity > 65535 * 64 || experts <= 0 || experts > 65535 || topk <= 0)
        throw std::invalid_argument("k_expert_gemm_gateup: invalid shape");
    const dim3 grid(unsigned((rows + 63) / 64), unsigned((capacity + 63) / 64), unsigned(experts));
    const auto * sg = static_cast<const __half2 *>(auxg); const auto * su = static_cast<const __half2 *>(auxu);
#define TQ_GU(F) u4h_gemm_k<F, true, __nv_bfloat16, true><<<grid, 256, 0, stream>>>(wg, sg, x, nullptr, rows, cols, capacity, rows, counts, groups, capacity, topk, false, wu, su, yh)
    switch (fmt) {
        case 1: TQ_GU(1); break;
        case 4: TQ_GU(4); break;
        case 5: TQ_GU(5); break;
        case 6: TQ_GU(6); break;
        case 7: TQ_GU(7); break;
        case 10: TQ_GU(10); break;
        case 13: TQ_GU(13); break;
        case 14: TQ_GU(14); break;
        default: throw std::invalid_argument("k_expert_gemm_gateup: unsupported format");
    }
#undef TQ_GU
}

void u4h_family_gemm(int fmt, const uint8_t * w, const void * aux, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    switch (fmt) {
        case 0: u4h_gemm_launch<0>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 1: u4h_gemm_launch<1>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 2: u4h_gemm_launch<2>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 4: u4h_gemm_launch<4>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 5: u4h_gemm_launch<5>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 6: u4h_gemm_launch<6>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 7: u4h_gemm_launch<7>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 8: u4h_gemm_launch<8>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 9: u4h_gemm_launch<9>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 10: u4h_gemm_launch<10>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 11: u4h_gemm_launch<11>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 12: u4h_gemm_launch<12>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 13: u4h_gemm_launch<13>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 14: u4h_gemm_launch<14>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 15: u4h_gemm_launch<15>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 16: u4h_gemm_launch<16>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        default: throw std::invalid_argument("u4h_family_gemm: unknown format");
    }
}
// Same, F32 output (prefill projections that feed F32 activations); false: format without a GEMM kernel.
bool u4h_family_gemm_f32(int fmt, const uint8_t * w, const void * aux, const __half * xh, float * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    switch (fmt) {
        case 0: u4h_gemm_launch<0, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 1: u4h_gemm_launch<1, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 2: u4h_gemm_launch<2, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 4: u4h_gemm_launch<4, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 5: u4h_gemm_launch<5, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 6: u4h_gemm_launch<6, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 7: u4h_gemm_launch<7, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 8: u4h_gemm_launch<8, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 9: u4h_gemm_launch<9, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 10: u4h_gemm_launch<10, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 11: u4h_gemm_launch<11, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 12: u4h_gemm_launch<12, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 13: u4h_gemm_launch<13, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 14: u4h_gemm_launch<14, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 15: u4h_gemm_launch<15, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        case 16: u4h_gemm_launch<16, float>(w, aux, xh, y, rows, k, ncols, s, ldy); break;
        default: return false;
    }
    return true;
}


void tq3_repack(void * w, int64_t nblocks, cudaStream_t s) {
    tq3_repack_k<<<int((nblocks + 255) / 256), 256, 0, s>>>(static_cast<uint4 *>(w), nblocks);
}

void tq3_rotate_i8(const __nv_bfloat16 * x, int8_t * xq, float * xs, int k, cudaStream_t s) {
    const int nblk = k / 32;
    tq3_rotate_i8_k<<<(nblk * 32 + 255) / 256, 256, 0, s>>>(x, xq, xs, nblk);
}

void tq3_matvec_i8(const void * w, const int8_t * xq, const float * xs, __nv_bfloat16 * y, int rows, int k,
                   cudaStream_t s) {
    static const Tq3Codebook8 cb = tq3_codebook_i8();
    constexpr int RPW = 2;
    tq3_matvec_i8_k<RPW><<<(rows + 8 * RPW - 1) / (8 * RPW), 256, 0, s>>>(
        static_cast<const uint4 *>(w), reinterpret_cast<const int4 *>(xq), xs, y, rows, k, cb.lo, cb.hi);
}

void tq3_rotate(const __nv_bfloat16 * x, float * xr, int k, int ncols, cudaStream_t s) {
    const int blocks = k / 32 * ncols;
    tq3_rotate_k<<<(blocks * 32 + 255) / 256, 256, 0, s>>>(x, xr, blocks);
}

void tq3_rotate_f32(const float * x, float * xr, int k, int ncols, cudaStream_t s) {
    if (!x || !xr || k <= 0 || k % 32 || ncols <= 0 || int64_t(k) * ncols > INT_MAX)
        throw std::invalid_argument("tq3_rotate_f32: invalid pointer or shape");
    const int blocks = k / 32 * ncols;
    tq3_rotate_f32_k<<<(int64_t(blocks) * 32 + 255) / 256, 256, 0, s>>>(x, xr, blocks);
}

void tq3_matvec_f32_raw(const void * w, const float * xr, float * y, int rows, int k, int ncols, cudaStream_t s) {
    if (!w || !xr || !y || rows <= 0 || k <= 0 || k % 32 || ncols <= 0 || ncols > 65535 ||
        int64_t(k) * ncols > INT_MAX || int64_t(rows) * ncols > INT_MAX)
        throw std::invalid_argument("tq3_matvec_f32_raw: invalid pointer or shape");
    tq3_matvec_f32_raw_k<false><<<dim3(unsigned(rows), unsigned(ncols)), 256, 0, s>>>(
        static_cast<const uint4 *>(w), xr, y, rows, k);
}

void tq3_matvec_f32_repacked(const void * w, const float * xr, float * y, int rows, int k, int ncols, cudaStream_t s) {
    if (!w || !xr || !y || rows <= 0 || k <= 0 || k % 32 || ncols <= 0 || ncols > 65535 ||
        int64_t(k) * ncols > INT_MAX || int64_t(rows) * ncols > INT_MAX)
        throw std::invalid_argument("tq3_matvec_f32_repacked: invalid pointer or shape");
    tq3_matvec_f32_raw_k<true><<<dim3(unsigned(rows), unsigned(ncols)), 256, 0, s>>>(
        static_cast<const uint4 *>(w), xr, y, rows, k);
}

void tq3_matvec(const void * w, const float * xr, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t s) {
    if (k % 32) throw std::invalid_argument("tq3_matvec: k must be a multiple of 32");
    if (ncols == 1) {
        constexpr int RPW = 4;
        tq3_matvec1_k<RPW><<<(rows + 8 * RPW - 1) / (8 * RPW), 256, 0, s>>>(static_cast<const uint4 *>(w), xr, y, rows, k);
        return;
    }
    switch (ncols) {
        case 2: return tq3_launch<2>(w, xr, y, rows, k, s);
        case 3: return tq3_launch<3>(w, xr, y, rows, k, s);
        case 4: return tq3_launch<4>(w, xr, y, rows, k, s);
        case 5: return tq3_launch<5>(w, xr, y, rows, k, s);
        case 6: return tq3_launch<6>(w, xr, y, rows, k, s);
        case 7: return tq3_launch<7>(w, xr, y, rows, k, s);
        case 8: return tq3_launch<8>(w, xr, y, rows, k, s);
        default: throw std::invalid_argument("tq3_matvec: 1..8 columns");
    }
}

void tq3_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    const int nblk = k / 32;
    tq3_row_k<<<(nblk + 7) / 8, 256, 0, s>>>(static_cast<const uint4 *>(w), row, y, nblk);
}

void q6k_matvec(const void * w, const float * x, float * y, int rows, int k, int ncols, cudaStream_t s) {
    if (k % Q6K_BLOCK) throw std::invalid_argument("q6k_matvec: k must be a multiple of 256");
    switch (ncols) {
        case 1: return q6k_launch<1>(w, x, y, rows, k, s);
        case 2: return q6k_launch<2>(w, x, y, rows, k, s);
        case 3: return q6k_launch<3>(w, x, y, rows, k, s);
        case 4: return q6k_launch<4>(w, x, y, rows, k, s);
        case 5: return q6k_launch<5>(w, x, y, rows, k, s);
        case 6: return q6k_launch<6>(w, x, y, rows, k, s);
        case 7: return q6k_launch<7>(w, x, y, rows, k, s);
        case 8: return q6k_launch<8>(w, x, y, rows, k, s);
        case 9: return q6k_launch<9>(w, x, y, rows, k, s);
        case 10: return q6k_launch<10>(w, x, y, rows, k, s);
        case 11: return q6k_launch<11>(w, x, y, rows, k, s);
        case 12: return q6k_launch<12>(w, x, y, rows, k, s);
        case 13: return q6k_launch<13>(w, x, y, rows, k, s);
        case 14: return q6k_launch<14>(w, x, y, rows, k, s);
        case 15: return q6k_launch<15>(w, x, y, rows, k, s);
        default: throw std::invalid_argument("q6k_matvec: 1..16 columns");
    }
}

void add_sumsq(__nv_bfloat16 * x, const __nv_bfloat16 * y, float * partial, int k, cudaStream_t s) {
    add_sumsq_k<<<(k + 255) / 256, 256, 0, s>>>(x, y, partial, k);
}

void sumsq(const __nv_bfloat16 * x, float * partial, int k, cudaStream_t s) {
    sumsq_k<<<(k + 255) / 256, 256, 0, s>>>(x, partial, k);
}

void norm_rotate_i8(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, int8_t * xq,
                    float * xs, int k, cudaStream_t s) {
    norm_rotate_i8_k<<<(k + 255) / 256, 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xq, xs, k);
}

void argmax_big(const float * x, int n, uint32_t * out, void * scratch, cudaStream_t s) {
    constexpr int NB = 512;
    auto * pv = static_cast<float *>(scratch);
    auto * pi = reinterpret_cast<uint32_t *>(pv + 1024);
    argmax1_k<<<NB, 256, 0, s>>>(x, n, pv, pi);
    argmax2_k<<<1, 32, 0, s>>>(pv, pi, NB, out);
}

void q6k_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    q6k_row_k<<<(k + 255) / 256, 256, 0, s>>>(static_cast<const uint8_t *>(w), row, y, k);
}

// One thread per element of one IQ3_S row (raw GGUF layout: 110 bytes per 256 weights).
// Block: d(FP16,0:2) | qs[64](2:66) | qh[8](66:74) | signs[32](74:106) | scales[4](106:110).
// Decode mirrors the proven iq3_dequant_h_k path: value = db * (iq3_code(grid, signs) - 16),
// db = d * (1 + 2 * scale_nibble).  Dual-mode: dtok==nullptr → single row; else multi-token.
__global__ void iq3s_row_k(const uint8_t * w, int row, __nv_bfloat16 * y, int k, const uint32_t * dtok = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (dtok) {  // multi-token: blockIdx.y selects the token, id read on the device
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * k;
    }
    const int nblk = k / 256;
    const int b = i / 256, e = i % 256;
    const int ib = e / 32, wi = (e % 32) / 4, j = e % 4;
    const uint8_t * blk = w + (int64_t(row) * nblk + b) * 110;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));  // offset 0: 2-byte aligned (110 % 2 == 0)
    const uint32_t qs_byte = blk[2 + ib * 8 + wi];
    const uint32_t qh_bit  = (blk[66 + ib] >> wi) & 1u;
    // signs word assembled from bytes: raw GGUF blocks are 110-byte-strided, so blk+74+4*ib is
    // NOT 4-byte aligned — a uint32_t load here would fault. Byte loads are always safe.
    const uint32_t sg = uint32_t(blk[74 + ib * 4]) | (uint32_t(blk[75 + ib * 4]) << 8) |
                        (uint32_t(blk[76 + ib * 4]) << 16) | (uint32_t(blk[77 + ib * 4]) << 24);
    const uint32_t sc_nib  = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xFu;
    const float db = d * float(1 + 2 * sc_nib);
    const uint32_t c = iq3_code(iq3s_grid_d[qs_byte | (qh_bit << 8)], (sg >> (4 * wi)) & 0xFu);
    const int code = int((c >> (8 * j)) & 0xFFu) - 16;
    y[i] = __float2bfloat16_rn(db * float(code));
}

void iq3s_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    iq3s_row_k<<<(k + 255) / 256, 256, 0, s>>>(static_cast<const uint8_t *>(w), row, y, k);
}

// One thread per element of one IQ4_XS row (raw GGUF layout: 136 bytes per 256 weights), for token-embedding
// lookups of llama.cpp IQ4_XS GGUFs. Block: d(FP16,0:2) | scales_h(u16,2:4) | scales_l[4](4:8) | qs[128](8:136).
// Mirrors dequantize_host iq4xs (formats/weight/dequant.cpp): element e of sub-block ib = e / 32 is
// d * (ls - 32) * kvalues_iq4nl[nibble], low nibble of qs[16 * ib + e % 16] for e % 32 < 16, high nibble otherwise.
// Dual-mode like iq3s_row_k: dtok == nullptr -> single row; else multi-token, blockIdx.y selects the token.
__constant__ int8_t iq4xs_embed_kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__global__ void iq4xs_row_k(const uint8_t * w, int row, __nv_bfloat16 * y, int k, const uint32_t * dtok = nullptr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    if (dtok) {
        row = int(dtok[blockIdx.y]);
        y += int64_t(blockIdx.y) * k;
    }
    const int nblk = k / 256;
    const int b = i / 256, e = i % 256, ib = e / 32, l = e % 32;
    const uint8_t * blk = w + (int64_t(row) * nblk + b) * 136;  // 136 % 2 == 0: the FP16 / u16 loads below are aligned
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    const uint32_t sh = uint32_t(blk[2]) | (uint32_t(blk[3]) << 8);
    const int ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | int(((sh >> (2 * ib)) & 3u) << 4);
    const uint8_t q = blk[8 + ib * 16 + (l % 16)];
    const int nib = l < 16 ? (q & 0xf) : (q >> 4);
    y[i] = __float2bfloat16_rn(d * float(ls - 32) * float(iq4xs_embed_kv[nib]));
}

void iq4xs_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t s) {
    iq4xs_row_k<<<(k + 255) / 256, 256, 0, s>>>(static_cast<const uint8_t *>(w), row, y, k);
}

void embed_multi_iq4xs(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    iq4xs_row_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), 0, y, k, dtok);
}

void tq3_rotate_h(const __nv_bfloat16 * x, __half * xh, int k, int ncols, cudaStream_t s) {
    const int blocks = k / 32 * ncols;
    tq3_rotate_h_k<<<(blocks * 32 + 255) / 256, 256, 0, s>>>(x, xh, blocks);
}

void norm_rotate_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k,
                   int ncols, cudaStream_t s) {
    norm_rotate_h_k<<<dim3((k + 255) / 256, ncols), 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xh, k);
}

// ---- W4A16 models: u4h weights take un-rotated FP16 activations ----
__global__ void norm_h_k(const __nv_bfloat16 * x, const float * w, const float * partial, int nparts, float eps,
                         __half * xh, int k) {
    __shared__ float inv_s;
    const int n = blockIdx.y;
    if (threadIdx.x < 32) {
        float t = 0.f;
        for (int i = threadIdx.x; i < nparts; i += 32) t += partial[n * nparts + i];
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) inv_s = rsqrtf(t / k + eps);
    }
    __syncthreads();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= k) return;
    const int64_t o = int64_t(n) * k + i;
    xh[o] = __float2half_rn(__bfloat162float(__float2bfloat16_rn(__bfloat162float(x[o]) * inv_s * w[i])));
}

// u4h group of 16 (8 bytes: element 4u + j = byte (u / 2) * 4 + j, nibble u % 2), (scale, min) per 32 -> FP16
__global__ void u4h_dequant_h_k(const uint8_t * w4, const __half2 * sm, __half * out, int64_t ngroups) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const uint2 raw = reinterpret_cast<const uint2 *>(w4)[g];
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&raw);
    const __half2 s = sm[g / 2];
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = (b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF;
        v[e] = __hfma(__int2half_rn(q), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

// i8h: int8 + FP16 scale per 16 -> FP16 (prefill dequant). One thread per 16-weight group.
__global__ void i8h_dequant_h_k(const int8_t * w8, const __half * sc, __half * out, int64_t ngroups) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int4 raw = reinterpret_cast<const int4 *>(w8)[g];
    const int8_t * q = reinterpret_cast<const int8_t *>(&raw);
    const __half s = sc[g];
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) v[e] = __hmul(__int2half_rn(q[e]), s);
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void iq3x_dequant_h_k(const uint8_t * w3, const uint8_t * rec, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32)), h = int(g % 2);
    const auto * wp = reinterpret_cast<const uint32_t *>(w3 + blk * 12);
    const uint32_t q = wp[h], a = wp[2];
    const __half2 s = iq3x_sm(rec + (row * (k / 256) + sblk / 8) * 4, a);
    __align__(16) __half v[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const uint32_t c = iq3x_code(iq3xxs_grid_d[(q >> (8 * u)) & 0xFF], iq3x_signs(a, 4 * h + u));
#pragma unroll
        for (int j = 0; j < 4; ++j) v[4 * u + j] = __hfma(__int2half_rn(int((c >> (8 * j)) & 0xFF)), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void iq3_dequant_h_k(const uint8_t * w3, const uint8_t * rec, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // one 16-weight group
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32)), h = int(g % 2);
    const auto * wp = reinterpret_cast<const uint32_t *>(w3 + blk * 12);
    const uint8_t * rc = rec + (row * (k / 256) + sblk / 8) * 16;
    const __half2 s = iq3_sm(rc, sblk % 8);
    const uint32_t q = wp[h], sg = wp[2], qh = rc[2 + sblk % 8];
    __align__(16) __half v[16];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int w = 4 * h + u;
        const uint32_t c = iq3_code(iq3s_grid_d[((q >> (8 * u)) & 0xFF) | (((qh >> w) & 1) << 8)], (sg >> (4 * w)) & 0xF);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[4 * u + j] = __hfma(__int2half_rn(int((c >> (8 * j)) & 0xFF)), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void k3_dequant_h_k(const uint8_t * w3, const uint8_t * kr, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;  // one 16-weight group
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32)), h = int(g % 2);
    const auto * wp = reinterpret_cast<const uint32_t *>(w3 + blk * 12);
    const uint32_t L = wp[h], H = wp[2];
    const uint8_t * rc = kr + (row * (k / 256) + sblk / 8) * 20;
    const float sc = __half2float(*reinterpret_cast<const __half *>(rc + 16)) * float(int8_t(rc[(sblk % 8) * 2 + h]));
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = int(((L >> (8 * j + 2 * u)) & 3) | (((H >> (8 * j + u + 4 * h)) & 1) << 2));
        v[e] = __float2half_rn(sc * float(q - 4));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void k4_dequant_h_k(const uint8_t * w4, const uint8_t * kh, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const uint2 raw = reinterpret_cast<const uint2 *>(w4)[g];
    const uint8_t * b = reinterpret_cast<const uint8_t *>(&raw);
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int sblk = int(blk % (k / 32));
    const __half2 s = k4_sm(kh + (row * (k / 256) + sblk / 8) * 16, sblk % 8);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = (b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF;
        v[e] = __hfma(__int2half_rn(q), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}

__global__ void embed_multi_bf16_k(const __nv_bfloat16 * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) y[int64_t(n) * k + i] = w[int64_t(dtok[n]) * k + i];
}
__global__ void embed_multi_q8_0_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 32) + i / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(reinterpret_cast<const int8_t *>(blk + 2)[i % 32]));
}

// Q4_K rows (144-byte blocks: d, dmin, scales[12], qs[128]) as in src/formats/weight/dequant.cpp q4k: element i of a block is the low
// (i % 64 < 32) or high nibble of qs[32 * (i / 64) + i % 32], sub-block j = i / 32 with its 6-bit scale and min
__global__ void embed_multi_q4k_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 256) + i / 256) * 144;
    const int e = i % 256, j = e / 32, l = e % 32;
    const uint8_t * sc = blk + 4;
    int s, m;
    if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const uint8_t q = blk[16 + 32 * (e / 64) + l];
    const float d = __half2float(reinterpret_cast<const __half *>(blk)[0]);
    const float dmin = __half2float(reinterpret_cast<const __half *>(blk)[1]);
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(s) * float((e % 64) < 32 ? (q & 0xF) : (q >> 4)) - dmin * float(m));
}

// Q3_K rows (110-byte blocks, layout as src/formats/weight/dequant.cpp q3k): 6-bit scale s of 16 = low nibble of byte s % 4 (+4 for
// s / 4 odd; high nibble for s / 4 >= 2) | 2 bits of byte 8 + s % 4 at 2 * (s / 4), minus 32
__global__ void embed_multi_q3k_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 256) + i / 256) * 110;
    const int e = i % 256, h = e / 128, j = (e % 128) / 32, l = e % 32, si = 8 * h + 2 * j + (l >= 16);
    const int g = si / 4, b = si % 4;
    const uint8_t * raw = blk + 96;
    const int lo = (g % 2 ? raw[4 + b] : raw[b]) >> (g >= 2 ? 4 : 0) & 0xF;
    const int sc = (lo | (((raw[8 + b] >> (2 * g)) & 3) << 4)) - 32;
    const int q = ((blk[32 + 32 * h + l] >> (2 * j)) & 3) - ((blk[l] >> (4 * h + j)) & 1 ? 0 : 4);
    const float d = __half2float(*reinterpret_cast<const __half *>(blk + 108));
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(sc) * float(q));
}
void embed_multi_q3k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q3k_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

void embed_multi_q4k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q4k_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

// Q5_K rows (176-byte blocks: d, dmin, scales[12], qh[32], qs[128]) as in q5k_to_i8_k / src/formats/weight/dequant.cpp q5k:
// 5-bit value = low/high nibble of qs[32 * (j / 2) + l] plus the qh[l] bit of sub-block j; affine with per-
// sub-block scale (s) and min (m), value = d * s * c - dmin * m
__global__ void embed_multi_q5k_k(const uint8_t * w, const uint32_t * dtok, __nv_bfloat16 * y, int k) {
    const int n = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const uint8_t * blk = w + (int64_t(dtok[n]) * (k / 256) + i / 256) * 176;
    const int e = i % 256, j = e / 32, l = e % 32;
    const uint8_t * sc = blk + 4;
    int s, m;
    if (j < 4) {
        s = sc[j] & 63;
        m = sc[j + 4] & 63;
    } else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
    const int q = blk[48 + 32 * (j / 2) + l];
    const int c = (j & 1 ? q >> 4 : q & 0xF) + ((blk[16 + l] >> j) & 1) * 16;
    const float d = __half2float(reinterpret_cast<const __half *>(blk)[0]);
    const float dmin = __half2float(reinterpret_cast<const __half *>(blk)[1]);
    y[int64_t(n) * k + i] = __float2bfloat16_rn(d * float(s) * float(c) - dmin * float(m));
}
void embed_multi_q5k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q5k_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

void norm_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k, int ncols,
            cudaStream_t s) {
    norm_h_k<<<dim3((k + 255) / 256, ncols), 256, 0, s>>>(x, w, partial, (k + 255) / 256, eps, xh, k);
}

void i8h_dequant_h(const int8_t * w8, const __half * sc, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    i8h_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w8, sc, out, ng);
}

void iq3x_dequant_h(const uint8_t * w3, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    iq3x_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w3, rec, out, ng, k);
}

void iq3_dequant_h(const uint8_t * w3, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    iq3_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w3, rec, out, ng, k);
}

void k3_dequant_h(const uint8_t * w3, const uint8_t * kr, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k3_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w3, kr, out, ng, k);
}

// UD types: one thread per 16-weight group, the matvec kernel's decoder with the grid read from global memory
template <int FMT>
__global__ void udq_dequant_h_k(const uint8_t * w, const uint8_t * kr, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2, row = blk / (k / 32);
    const int h = int(g % 2), sblk = int(blk % (k / 32));
    const auto * wp = reinterpret_cast<const uint32_t *>(w + blk * ud_bpb(FMT));
    const uint32_t w0 = wp[0], w1 = ud_bpb(FMT) == 8 ? wp[1] : 0u;
    const uint8_t * rec = kr + (row * (k / 256) + sblk / 8) * ud_rb(FMT);
    const uint32_t * grid = nullptr;
    if constexpr (FMT <= 11) grid = ud_grid_src<FMT>();
    uint32_t c[4];
    __half2 sm;
    ud_group<FMT>(w0, w1, rec, sblk, h, grid, c, sm);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) v[e] = __hfma(__int2half_rn(int((c[e / 4] >> (8 * (e % 4))) & 0xFF)), __low2half(sm), __high2half(sm));
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void udq_dequant_h(int fmt, const uint8_t * w, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    const unsigned nbk = unsigned((ng + 255) / 256);
    switch (fmt) {
        case 8: udq_dequant_h_k<8><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 9: udq_dequant_h_k<9><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 10: udq_dequant_h_k<10><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 11: udq_dequant_h_k<11><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        case 12: udq_dequant_h_k<12><<<nbk, 256, 0, s>>>(w, rec, out, ng, k); break;
        default: throw std::invalid_argument("udq_dequant_h: unknown format");
    }
}
// k6: per 32 weights 16 nibble bytes (k4 order) + 2 words of 2-bit highs; 20-byte record (16 int8 scales, d) per 256
__global__ void k6_dequant_h_k(const uint8_t * w6, const uint8_t * kr, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2;
    const int h = int(g % 2);
    const uint8_t * b = w6 + blk * 24 + h * 8;
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(w6 + blk * 24 + 16 + 4 * h);
    const int64_t row = blk / (k / 32);
    const int sblk = int(blk % (k / 32));
    const uint8_t * rc = kr + (row * (k / 256) + sblk / 8) * 20;
    const float f = __half2float(*reinterpret_cast<const __half *>(rc + 16)) * float(int8_t(rc[(sblk % 8) * 2 + h]));
    const __half sc = __float2half_rn(f), mn = __float2half_rn(-32.f * f);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = ((b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF) | int(((hi >> (2 * e)) & 3) << 4);
        v[e] = __hfma(__int2half_rn(q), sc, mn);
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void k6_dequant_h(const uint8_t * w6, const uint8_t * kr, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k6_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w6, kr, out, ng, k);
}
// k5: per 32 weights 16 nibble bytes (k4 order) + the 32 high bits; one thread per 16-weight group
__global__ void k5_dequant_h_k(const uint8_t * w5, const uint8_t * kh, __half * out, int64_t ngroups, int k) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= ngroups) return;
    const int64_t blk = g / 2;
    const int h = int(g % 2);
    const uint8_t * b = w5 + blk * 20 + h * 8;
    const uint32_t hi = *reinterpret_cast<const uint32_t *>(w5 + blk * 20 + 16) >> (16 * h);
    const int64_t row = blk / (k / 32);
    const int sblk = int(blk % (k / 32));
    const __half2 s = k4_sm(kh + (row * (k / 256) + sblk / 8) * 16, sblk % 8);
    __align__(16) __half v[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const int u = e / 4, j = e % 4;
        const int q = ((b[(u / 2) * 4 + j] >> ((u % 2) * 4)) & 0xF) | int(((hi >> e) & 1) << 4);
        v[e] = __hfma(__int2half_rn(q), __low2half(s), __high2half(s));
    }
    reinterpret_cast<uint4 *>(out)[g * 2] = reinterpret_cast<const uint4 *>(v)[0];
    reinterpret_cast<uint4 *>(out)[g * 2 + 1] = reinterpret_cast<const uint4 *>(v)[1];
}
void k5_dequant_h(const uint8_t * w5, const uint8_t * kh, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k5_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w5, kh, out, ng, k);
}

void k4_dequant_h(const uint8_t * w4, const uint8_t * kh, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    k4_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w4, kh, out, ng, k);
}

void u4h_dequant_h(const uint8_t * w4, const __half2 * sm, __half * out, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    u4h_dequant_h_k<<<unsigned((ng + 255) / 256), 256, 0, s>>>(w4, sm, out, ng);
}

void embed_multi_bf16(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_bf16_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(w), dtok, y, k);
}
void embed_multi_q8_0(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    embed_multi_q8_0_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), dtok, y, k);
}

void tq3_matvec_h_variant(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, int variant,
                          cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const auto * wp = static_cast<const uint4 *>(w);
    if (ncols > 8) throw std::invalid_argument("tq3_matvec_h_variant: 1..8 columns");
    switch (variant) {
        case 1: tq3_matvec_p_k<1><<<(rows + 15) / 16, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        case 2: tq3_matvec_p_k<2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        case 3: tq3_matvec_hw_k<1, 2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        case 4: tq3_matvec_hw_k<1, 4><<<(rows + 63) / 64, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi); break;
        default:
            if (ncols != 8) throw std::invalid_argument("tq3_matvec_h_variant: default variant at 8 columns");
            tq3_matvec_h_k<8><<<(rows + 15) / 16, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi);
    }
}

// Per-shape kernel variant for <= 8 columns, picked by tq3_tune at load (all variants are bit-identical per column):
// Tuning scratch: a full device must fail closed here, not time launches that never ran (or ran on null buffers).
static void tune_scratch(__half ** x, size_t xbytes, __nv_bfloat16 ** y, size_t ybytes, const char * who) {
    if (cudaMalloc(x, xbytes) != cudaSuccess || cudaMalloc(y, ybytes) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(*x);
        *x = nullptr;
        throw std::runtime_error(std::string(who) + ": tuning scratch allocation failed");
    }
}
static void tune_done(__half * x, __nv_bfloat16 * y, const char * who) {
    const cudaError_t e = cudaGetLastError();
    cudaFree(x);
    cudaFree(y);
    if (e != cudaSuccess) throw std::runtime_error(std::string(who) + ": " + cudaGetErrorString(e));
}

// Time of one launch with the weights out of L2, as decode reads them (each weight once per step, hundreds of MB apart):
// timing back-to-back launches of one matrix measured L2-resident weights, a GB10's 24 MB L2 holds most dense matrices.
// The average of `reps` launches, each after writing twice the L2 size of scratch.
template <class F>
static float timed_cold(F && launch, int reps, cudaStream_t s) {
    static size_t bytes = [] { int dev = 0, l2 = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
                               return std::max<size_t>(size_t(2) * size_t(l2), size_t(8) << 20); }();
    void * evict = nullptr;
    if (cudaMalloc(&evict, bytes) != cudaSuccess) { cudaGetLastError(); throw std::runtime_error("tuning: L2 eviction scratch allocation failed"); }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float total = 0.f;
    for (int i = 0; i < reps; ++i) {
        cudaMemsetAsync(evict, i & 0xff, bytes, s);
        cudaEventRecord(e0, s); launch(); cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); total += ms;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaFree(evict);
    return total / float(reps);
}

// 0 narrow, 1 / 2 prefetching with 1 / 2 row tiles, 3 / 4 row-tiled x2 / x4.
static std::map<std::pair<int, int>, int> & tq3_tuned() {
    static std::map<std::pair<int, int>, int> m;
    return m;
}

// Tuning launches the matvec on the weights it is given: only for device (or managed) memory. A host-side repack (repack_expert's
// Upload keeps the bytes on the host) must not run GPU kernels on host pointers: without HMM that is an illegal access at load.
static bool tune_pointer_ok(const void * p) {
    cudaPointerAttributes a{};
    if (!p || cudaPointerGetAttributes(&a, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged;
}
void tq3_tune(const void * w, int rows, int k, cudaStream_t s) {
    if (tq3_tuned().count({rows, k}) || !tune_pointer_ok(w)) return;
    __half * x = nullptr;
    __nv_bfloat16 * y = nullptr;
    tune_scratch(&x, size_t(8) * k * 2, &y, size_t(8) * rows * 2, "tq3_tune");
    cudaMemsetAsync(x, 0, size_t(8) * k * 2, s);
    int best = 0;
    float best_ms = 1e30f;
    for (int v : {0, 1, 2, 3, 4}) {
        for (int i = 0; i < 3; ++i) tq3_matvec_h_variant(w, x, y, rows, k, 8, v, s);  // warm-up (clocks)
        const float ms = timed_cold([&] { tq3_matvec_h_variant(w, x, y, rows, k, 8, v, s); }, 12, s);
        if (ms < best_ms * 0.98f) best_ms = ms, best = v;  // keep the lower variant unless clearly faster
    }
    tune_done(x, y, "tq3_tune");
    tq3_tuned()[{rows, k}] = best;
    if (tqllm::debug_on("tune")) std::fprintf(stderr, "tq3_tune %d x %d -> variant %d (%.1f us)\n", rows, k, best, best_ms * 1000.f);
}

void tq3_matvec_h(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const auto * wp = static_cast<const uint4 *>(w);
    const int grid = (rows + 15) / 16;
    if (ncols <= 8) {  // tuned shape
        const auto it = tq3_tuned().find({rows, k});
        if (it != tq3_tuned().end() && (it->second != 0 || ncols == 8)) {
            tq3_matvec_h_variant(w, xh, y, rows, k, ncols, it->second, s);
            return;
        }
    }
    // long-K matrices (FFN down) take the prefetching kernel with 2 row tiles per CTA
    // (ncu, base clocks, 27B verify: 5120 x 17408 at 8 columns 72.4 -> 62.3 us)
    if (k >= 12288 && ncols <= 8) {
        tq3_matvec_p_k<2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi);
        return;
    }
    switch (ncols) {
        case 1: tq3_matvec_h_k<1><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 2: tq3_matvec_h_k<2><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 3: tq3_matvec_h_k<3><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 4: tq3_matvec_h_k<4><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 5: tq3_matvec_h_k<5><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 6: tq3_matvec_h_k<6><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 7: tq3_matvec_h_k<7><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 8: tq3_matvec_h_k<8><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 9: tq3_matvec_h_k<9><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 10: tq3_matvec_h_k<10><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 11: tq3_matvec_h_k<11><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        case 12: tq3_matvec_h_k<12><<<grid, 256, 0, s>>>(wp, xh, y, rows, k, cb.lo, cb.hi); break;
        default:
            if (ncols > 12 && ncols <= 32) {
                auto go = [&](auto nt_c) {  // 2 row tiles per CTA
                    constexpr int NT = decltype(nt_c)::value;
                    tq3_matvec_pw_k<NT, 2><<<(rows + 31) / 32, 256, 0, s>>>(wp, xh, y, rows, k, ncols, cb.lo, cb.hi);
                };
                if (ncols <= 16) go(std::integral_constant<int, 2>{});
                else if (ncols <= 24) go(std::integral_constant<int, 3>{});
                else go(std::integral_constant<int, 4>{});
            }
            else throw std::invalid_argument("tq3_matvec_h: 1..32 columns");
    }
}

void argmax_cols(const float * x, int len, int ncols, uint32_t * out, void * scratch, cudaStream_t s) {
    constexpr int NB = 128;  // scratch: ncols * NB (value, index) pairs
    auto * pv = static_cast<float *>(scratch);
    auto * pi = reinterpret_cast<uint32_t *>(pv + ncols * NB);
    argmax_cols1_k<<<dim3(NB, ncols), 256, 0, s>>>(x, len, pv, pi, NB);
    argmax_cols2_k<<<ncols, 32, 0, s>>>(pv, pi, NB, out);
}

void embed_multi_tq3(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    const int nblk = k / 32;
    tq3_row_k<<<dim3((nblk + 7) / 8, n), 256, 0, s>>>(static_cast<const uint4 *>(w), 0, y, nblk, dtok);
}

void embed_multi_q6k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    q6k_row_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), 0, y, k, dtok);
}

void embed_multi_iq3s(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t s) {
    iq3s_row_k<<<dim3((k + 255) / 256, n), 256, 0, s>>>(static_cast<const uint8_t *>(w), 0, y, k, dtok);
}

// Q5_K -> int8 + FP16 scale per 16 (symmetric, max-abs / 127): the affine Q5_K values are not exactly representable,
// int8 adds about 1/8 of a Q5_K step of error. One thread per 16-group.
__global__ void q5k_to_i8_k(const uint8_t * w, int8_t * w8, __half * sc, int64_t ngroups) {
    const int64_t gi = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gi >= ngroups) return;
    const uint8_t * blk = w + (gi / 16) * 176;
    const int e0 = int(gi % 16) * 16, j = e0 / 32;
    const uint8_t *scr = blk + 4, *qh = blk + 16, *qs = blk + 48;
    int s, m;
    if (j < 4) {
        s = scr[j] & 63;
        m = scr[j + 4] & 63;
    } else {
        s = (scr[j + 4] & 0xF) | ((scr[j - 4] >> 6) << 4);
        m = (scr[j + 4] >> 4) | ((scr[j] >> 6) << 4);
    }
    const float d = __half2float(reinterpret_cast<const __half *>(blk)[0]) * float(s);
    const float mn = __half2float(reinterpret_cast<const __half *>(blk)[1]) * float(m);
    float v[16], amax = 0.f;
    for (int t = 0; t < 16; ++t) {
        const int l = (e0 + t) % 32;
        const uint8_t q = qs[32 * (j / 2) + l];
        const int c = (j % 2 ? q >> 4 : q & 0xF) + ((qh[l] >> j) & 1) * 16;
        v[t] = d * float(c) - mn;
        amax = fmaxf(amax, fabsf(v[t]));
    }
    const __half sh = __float2half_rn(amax / 127.f);
    const float st = __half2float(sh), inv = st > 0.f ? 1.f / st : 0.f;
    for (int t = 0; t < 16; ++t) w8[gi * 16 + t] = int8_t(max(-127, min(127, __float2int_rn(v[t] * inv))));
    sc[gi] = sh;
}
void q5k_to_i8(const void * q5k, int8_t * w8, __half * sc, int rows, int k, cudaStream_t s) {
    const int64_t ng = int64_t(rows) * k / 16;
    q5k_to_i8_k<<<int((ng + 255) / 256), 256, 0, s>>>(static_cast<const uint8_t *>(q5k), w8, sc, ng);
}
void q6k_to_i8(const void * q6k, int8_t * w8, __half * sc, int rows, int k, cudaStream_t s) {
    const int64_t n = int64_t(rows) * k;
    q6k_to_i8_k<<<int((n + 255) / 256), 256, 0, s>>>(static_cast<const uint8_t *>(q6k), w8, sc, rows, k);
}

// Exact int8 re-score of 16 candidate rows per column (candidates from the 4-bit head), argmax (ties -> lower id).
// Weight rounding as in i8h_matvec_k (q * s in FP16), FP32 accumulation. grid = ncols, 16 warps (one per candidate).
__global__ void __launch_bounds__(512) i8h_rescore_argmax_k(const int8_t * w8, const __half * sc, const __half * xh,
                                                            const uint32_t * cand, int k, uint32_t * out, float * scores) {
    __shared__ float sv[16];
    const int c = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t row = cand[c * 16 + warp];
    const int8_t * wr = w8 + int64_t(row) * k;
    const __half * sr = sc + int64_t(row) * (k / 16);
    const __half * xr = xh + int64_t(c) * k;
    float acc = 0.f;
    for (int i = lane * 8; i < k; i += 256) {
        const uint2 wq = __ldg(reinterpret_cast<const uint2 *>(wr + i));
        const uint4 xq = __ldg(reinterpret_cast<const uint4 *>(xr + i));
        const __half s = sr[i / 16];
        const int8_t * wb = reinterpret_cast<const int8_t *>(&wq);
        const __half * xv = reinterpret_cast<const __half *>(&xq);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc += __half2float(__hmul(__int2half_rn(wb[e]), s)) * __half2float(xv[e]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        sv[warp] = acc;
        if (scores) scores[c * 16 + warp] = acc;  // exact logits of the 16 candidates (sampling)
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = sv[0];
        uint32_t bi = cand[c * 16];
        for (int j = 1; j < 16; ++j) {
            const uint32_t id = cand[c * 16 + j];
            if (sv[j] > bv || (sv[j] == bv && id < bi)) bv = sv[j], bi = id;
        }
        out[c] = bi;
    }
}

void i8h_rescore_argmax(const int8_t * w8, const __half * sc, const __half * xh, const uint32_t * cand, int k,
                        int ncols, uint32_t * out, cudaStream_t s, float * scores) {
    if (k % 256) throw std::invalid_argument("i8h_rescore_argmax: k must be a multiple of 256");
    i8h_rescore_argmax_k<<<ncols, 512, 0, s>>>(w8, sc, xh, cand, k, out, scores);
}

void i8h_matvec(const int8_t * w8, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    const int grid = (rows + 15) / 16;
    switch (ncols) {
        case 1: i8h_matvec_k<1><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 2: i8h_matvec_k<2><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 3: i8h_matvec_k<3><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 4: i8h_matvec_k<4><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 5: i8h_matvec_k<5><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 6: i8h_matvec_k<6><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 7: i8h_matvec_k<7><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 8: i8h_matvec_k<8><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 9: i8h_matvec_k<9><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 10: i8h_matvec_k<10><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 11: i8h_matvec_k<11><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 12: i8h_matvec_k<12><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 13: i8h_matvec_k<13><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 14: i8h_matvec_k<14><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        case 15: i8h_matvec_k<15><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k); break;
        default: throw std::invalid_argument("i8h_matvec: 1..16 columns");
    }
}

void i8h_quantize(const float * x, int8_t * w8, __half * sc, int64_t n, cudaStream_t s) {
    const int64_t ng = n / 16;
    i8h_quantize_k<<<int((ng + 255) / 256), 256, 0, s>>>(x, w8, sc, ng);
}

static __global__ void bf16_matvec_cols_k(const __nv_bfloat16 * w, const __nv_bfloat16 * x, __nv_bfloat16 * y, int rows,
                                          int k, int ldy) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, n = blockIdx.y, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(r) * k);
    const uint4 * xr = reinterpret_cast<const uint4 *>(x + int64_t(n) * k);
    float acc = 0.f;
    for (int i = lane; i < k / 8; i += 32) {
        const uint4 a = __ldg(&wr[i]), b = __ldg(&xr[i]);
        const auto * ah = reinterpret_cast<const __nv_bfloat162 *>(&a);
        const auto * bh = reinterpret_cast<const __nv_bfloat162 *>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __bfloat1622float2(ah[j]), fb = __bfloat1622float2(bh[j]);
            acc = fmaf(fa.x, fb.x, acc);
            acc = fmaf(fa.y, fb.y, acc);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) y[int64_t(n) * ldy + r] = __float2bfloat16_rn(acc);
}

void bf16_matvec_cols(const __nv_bfloat16 * w, const __nv_bfloat16 * x, __nv_bfloat16 * y, int rows, int k, int ncols,
                      cudaStream_t s, int ldy) {
    if (k % 256) throw std::invalid_argument("bf16_matvec_cols: k must be a multiple of 256");
    bf16_matvec_cols_k<<<dim3(unsigned((rows + 7) / 8), unsigned(ncols)), 256, 0, s>>>(w, x, y, rows, k, ldy ? ldy : rows);
}

void i8h_matvec_bf16(const int8_t * w8, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    const int grid = (rows + 15) / 16;
    switch (ncols) {
        case 1: i8h_matvec_k<1, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 2: i8h_matvec_k<2, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 3: i8h_matvec_k<3, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 4: i8h_matvec_k<4, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 5: i8h_matvec_k<5, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 6: i8h_matvec_k<6, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 7: i8h_matvec_k<7, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 8: i8h_matvec_k<8, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 9: i8h_matvec_k<9, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 10: i8h_matvec_k<10, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 11: i8h_matvec_k<11, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 12: i8h_matvec_k<12, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 13: i8h_matvec_k<13, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 14: i8h_matvec_k<14, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        case 15: i8h_matvec_k<15, __nv_bfloat16><<<grid, 256, 0, s>>>(w8, sc, xh, y, rows, k, ldy); break;
        default: throw std::invalid_argument("i8h_matvec_bf16: 1..16 columns");
    }
}

// Per-shape row tiling for <= 8 columns (u4h_tune; bit-identical for every MT), one map per format, keyed by
// (rows, k * 2 + wide): one column (decode) and 2..8 columns (verify) are timed apart - the 8-column winner applied to
// one column ran MiniMax's 6144 x 6144 shared experts at ~150 GB/s instead of ~280 on a GB10.
template <int FMT>
static std::map<std::pair<int, int>, int> & tuned() {
    static std::map<std::pair<int, int>, int> m;
    return m;
}

template <class OUT, int FMT = 0>
static void u4h_launch(const uint8_t * w4, const __half2 * sm, const __half * xh, OUT * y, int rows, int k, int ncols,
                       cudaStream_t s, int mt = -1, const uint32_t * rid = nullptr, int ldy = 0) {
    if (k % 64) throw std::invalid_argument("u4h_matvec: k must be a multiple of 64");
    if (mt < 0 && ncols <= 8) {
        auto & tm = tuned<FMT>();
        const auto it = tm.find({rows, k * 2 + (ncols > 1 ? 1 : 0)});
        mt = it == tm.end() ? 1 : it->second;
    }
    if (ncols <= 8 && mt == 5) {  // MT 1 with prefetch
        u4h_matvec_k<8, OUT, 1, FMT, 1><<<(rows + 15) / 16, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
        return;
    }
    if (ncols <= 8 && mt > 1) {
        if (mt >= 4) u4h_matvec_k<8, OUT, 4, FMT><<<(rows + 63) / 64, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
        else u4h_matvec_k<8, OUT, 2, FMT><<<(rows + 31) / 32, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
        return;
    }
    const int grid = (rows + 15) / 16;
    switch (ncols) {
        case 1: u4h_matvec_k<1, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 1, rid, ldy); break;
        case 2: u4h_matvec_k<2, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 2, rid, ldy); break;
        case 3: u4h_matvec_k<3, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 3, rid, ldy); break;
        case 4: u4h_matvec_k<4, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 4, rid, ldy); break;
        case 5: u4h_matvec_k<5, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 5, rid, ldy); break;
        case 6: u4h_matvec_k<6, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 6, rid, ldy); break;
        case 7: u4h_matvec_k<7, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 7, rid, ldy); break;
        case 8: u4h_matvec_k<8, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, 8, rid, ldy); break;
        default:
            if (ncols > 8 && ncols <= 16) {
                // 9..16 columns: shared activation loads pay off (3090: head 248320 x 5120 MT 1 / 2 / 4 = 1478 / 1156 /
                // 982 us, 5120 x 17408 116 / 81 / 93 us)
                if (mt < 0) mt = rows >= 65536 ? 4 : 2;
                if (mt >= 4) u4h_matvec_k<16, OUT, 4, FMT><<<(rows + 63) / 64, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
                else if (mt == 2) u4h_matvec_k<16, OUT, 2, FMT><<<(rows + 31) / 32, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
                else u4h_matvec_k<16, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            }
            else if (ncols > 16 && ncols <= 24) u4h_matvec_k<24, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            // 25-32 columns (np 8 x 4-row blocks): two row tiles per block from 2048 rows halve the activation re-reads
            // (GB10 IQ4_XS 2560 -> 10240 133 -> 108 us, 12288 rows 164 -> 125); short matrices keep one (640 rows 12 -> 16)
            else if (ncols > 24 && ncols <= 32 && rows >= 2048) u4h_matvec_k<32, OUT, 2, FMT><<<(rows + 31) / 32, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            else if (ncols > 24 && ncols <= 32) u4h_matvec_k<32, OUT, 1, FMT><<<grid, 256, 0, s>>>(w4, sm, xh, y, rows, k, ncols, rid, ldy);
            else throw std::invalid_argument("u4h_matvec: 1..32 columns");
    }
}

void iq3x_matvec_bf16(const uint8_t * w3, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                      int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 5>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void iq3x_matvec(const uint8_t * w3, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                 cudaStream_t s) {
    u4h_launch<float, 5>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s);
}

void iq3_matvec_bf16(const uint8_t * w3, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 4>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void iq3_matvec(const uint8_t * w3, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    u4h_launch<float, 4>(w3, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s);
}


void k3_matvec_bf16(const uint8_t * w3, const uint8_t * kr, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 2>(w3, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k3_matvec(const uint8_t * w3, const uint8_t * kr, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 2>(w3, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s);
}

void k4_matvec_bf16(const uint8_t * w4, const uint8_t * kh, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 1>(w4, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k4_matvec(const uint8_t * w4, const uint8_t * kh, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 1>(w4, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s);
}

template <int FMT>
static void ud_dispatch_bf16(const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                             int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, FMT>(w, reinterpret_cast<const __half2 *>(rec), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void udq_matvec_bf16(int fmt, const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    switch (fmt) {
        case 8: ud_dispatch_bf16<8>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 9: ud_dispatch_bf16<9>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 10: ud_dispatch_bf16<10>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 11: ud_dispatch_bf16<11>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        case 12: ud_dispatch_bf16<12>(w, rec, xh, y, rows, k, ncols, s, ldy); break;
        default: throw std::invalid_argument("udq_matvec: unknown format");
    }
}
void udq_matvec(int fmt, const uint8_t * w, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    const auto * r = reinterpret_cast<const __half2 *>(rec);
    switch (fmt) {
        case 8: u4h_launch<float, 8>(w, r, xh, y, rows, k, ncols, s); break;
        case 9: u4h_launch<float, 9>(w, r, xh, y, rows, k, ncols, s); break;
        case 10: u4h_launch<float, 10>(w, r, xh, y, rows, k, ncols, s); break;
        case 11: u4h_launch<float, 11>(w, r, xh, y, rows, k, ncols, s); break;
        case 12: u4h_launch<float, 12>(w, r, xh, y, rows, k, ncols, s); break;
        default: throw std::invalid_argument("udq_matvec: unknown format");
    }
}
void k6_matvec_bf16(const uint8_t * w6, const uint8_t * kr, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 7>(w6, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k6_matvec(const uint8_t * w6, const uint8_t * kr, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 7>(w6, reinterpret_cast<const __half2 *>(kr), xh, y, rows, k, ncols, s);
}
void k5_matvec_bf16(const uint8_t * w5, const uint8_t * kh, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t s, int ldy) {
    u4h_launch<__nv_bfloat16, 6>(w5, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}
void k5_matvec(const uint8_t * w5, const uint8_t * kh, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t s) {
    u4h_launch<float, 6>(w5, reinterpret_cast<const __half2 *>(kh), xh, y, rows, k, ncols, s);
}

void u4h_matvec_bf16(const uint8_t * w4, const __half2 * sm, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t s, int ldy) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s, -1, nullptr, ldy);
}

void u4h_matvec(const uint8_t * w4, const __half2 * sm, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t s) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s);
}
void u4h_matvec_ids(const uint8_t * w4, const __half2 * sm, const uint32_t * rid, const __half * xh, float * y, int rows,
                    int k, int ncols, cudaStream_t s) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s, -1, rid);
}

void u4h_matvec_mt(const uint8_t * w4, const __half2 * sm, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                   int ncols, int mt, cudaStream_t s) {
    u4h_launch(w4, sm, xh, y, rows, k, ncols, s, mt);
}

template <int FMT>
static void tune_impl(const uint8_t * w4, const __half2 * sm, int rows, int k, cudaStream_t s) {
    if (tuned<FMT>().count({rows, k * 2}) || !tune_pointer_ok(w4)) return;
    __half * x = nullptr;
    __nv_bfloat16 * y = nullptr;
    tune_scratch(&x, size_t(8) * k * 2, &y, size_t(8) * rows * 2, "u4h_tune");
    cudaMemsetAsync(x, 0, size_t(8) * k * 2, s);
    for (int cols : {1, 8}) {
        int best = 1;
        float best_ms = 1e30f;
        for (int mt : {1, 2, 4, 5}) {  // 5: MT 1 with prefetch
            for (int i = 0; i < 3; ++i) u4h_launch<__nv_bfloat16, FMT>(w4, sm, x, y, rows, k, cols, s, mt);
            const float ms = timed_cold([&] { u4h_launch<__nv_bfloat16, FMT>(w4, sm, x, y, rows, k, cols, s, mt); }, 12, s);
            if (ms < best_ms * 0.98f) best_ms = ms, best = mt;
        }
        tuned<FMT>()[{rows, k * 2 + (cols > 1 ? 1 : 0)}] = best;
        if (tqllm::debug_on("tune"))
            std::fprintf(stderr, "u4h_tune fmt %d %d x %d, %d col -> MT %d (%.1f us)\n", FMT, rows, k, cols, best, best_ms * 1000.f);
    }
    tune_done(x, y, "u4h_tune");
}

void u4h_tune(const uint8_t * w4, const __half2 * sm, int rows, int k, cudaStream_t s) { tune_impl<0>(w4, sm, rows, k, s); }
void iq3x_tune(const uint8_t * w3, const uint8_t * rec, int rows, int k, cudaStream_t s) {
    tune_impl<5>(w3, reinterpret_cast<const __half2 *>(rec), rows, k, s);
}
void iq3_tune(const uint8_t * w3, const uint8_t * rec, int rows, int k, cudaStream_t s) {
    tune_impl<4>(w3, reinterpret_cast<const __half2 *>(rec), rows, k, s);
}
void k4_tune(const uint8_t * w4, const uint8_t * kh, int rows, int k, cudaStream_t s) {
    tune_impl<1>(w4, reinterpret_cast<const __half2 *>(kh), rows, k, s);
}
// matvec through the u4h kernel for any u4h-family fmt (incl. 13 iq4n, 14 iq4l, 15 i8h); mt as u4h_launch (-1: tuned)
void u4h_family_matvec_f32(int fmt, const uint8_t * w, const void * aux, const __half * xh, float * y, int rows, int k,
                           int ncols, cudaStream_t s) {
    const auto * a = reinterpret_cast<const __half2 *>(aux);
    switch (fmt) {
        case 13: u4h_launch<float, 13>(w, a, xh, y, rows, k, ncols, s); break;
        case 14: u4h_launch<float, 14>(w, a, xh, y, rows, k, ncols, s); break;
        case 15: u4h_launch<float, 15>(w, a, xh, y, rows, k, ncols, s); break;
        case 16: u4h_launch<float, 16>(w, a, xh, y, rows, k, ncols, s); break;
        default: throw std::invalid_argument("u4h_family_matvec_f32: fmt not wired");
    }
}
void u4h_family_tune(int fmt, const uint8_t * w, const void * aux, int rows, int k, cudaStream_t s) {
    const auto * a = reinterpret_cast<const __half2 *>(aux);
    switch (fmt) {
        case 13: tune_impl<13>(w, a, rows, k, s); break;
        case 14: tune_impl<14>(w, a, rows, k, s); break;
        case 15: tune_impl<15>(w, a, rows, k, s); break;
        case 16: tune_impl<16>(w, a, rows, k, s); break;
        default: throw std::invalid_argument("u4h_family_tune: fmt not wired");
    }
}
void u4h_family_matvec_bf16(int fmt, const uint8_t * w, const void * aux, const __half * xh, __nv_bfloat16 * y, int rows,
                            int k, int ncols, cudaStream_t s, int ldy, int mt) {
    const auto * a = reinterpret_cast<const __half2 *>(aux);
    switch (fmt) {
        case 13: u4h_launch<__nv_bfloat16, 13>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 14: u4h_launch<__nv_bfloat16, 14>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 15: u4h_launch<__nv_bfloat16, 15>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 16: u4h_launch<__nv_bfloat16, 16>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        case 6: u4h_launch<__nv_bfloat16, 6>(w, a, xh, y, rows, k, ncols, s, mt, nullptr, ldy); break;
        default: throw std::invalid_argument("u4h_family_matvec: fmt not wired");
    }
}

void udq_tune(int fmt, const uint8_t * w, const uint8_t * rec, int rows, int k, cudaStream_t s) {
    const auto * r = reinterpret_cast<const __half2 *>(rec);
    switch (fmt) {
        case 8: tune_impl<8>(w, r, rows, k, s); break;
        case 9: tune_impl<9>(w, r, rows, k, s); break;
        case 10: tune_impl<10>(w, r, rows, k, s); break;
        case 11: tune_impl<11>(w, r, rows, k, s); break;
        case 12: tune_impl<12>(w, r, rows, k, s); break;
        default: throw std::invalid_argument("udq_tune: unknown format");
    }
}
void k6_tune(const uint8_t * w6, const uint8_t * kr, int rows, int k, cudaStream_t s) {
    tune_impl<7>(w6, reinterpret_cast<const __half2 *>(kr), rows, k, s);
}
void k5_tune(const uint8_t * w5, const uint8_t * kh, int rows, int k, cudaStream_t s) {
    tune_impl<6>(w5, reinterpret_cast<const __half2 *>(kh), rows, k, s);
}
void k3_tune(const uint8_t * w3, const uint8_t * kr, int rows, int k, cudaStream_t s) {
    tune_impl<2>(w3, reinterpret_cast<const __half2 *>(kr), rows, k, s);
}

void u4h_from_i8h(const int8_t * w8, const __half * sc, uint8_t * w4, __half2 * sm, int64_t n, cudaStream_t s) {
    const int64_t nblk = n / 32;
    u4h_from_i8h_k<<<unsigned((nblk + 255) / 256), 256, 0, s>>>(w8, sc, w4, sm, nblk);
}

void tq3_gemm_h(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const dim3 grid((rows + 63) / 64, (ncols + 63) / 64);
    tq3_gemm_h_k<true><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), xh, y, rows, k, ncols, cb.lo, cb.hi);
}

void tq3_expert_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int rows, int k,
                       const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    if (!w || !x || !y || !ids || rows <= 0 || k <= 0 || k % 32 || n <= 0 || topk <= 0 ||
        int64_t(n) * topk > 65535)
        throw std::invalid_argument("tq3_expert_matvec: invalid shape or pointer");
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const dim3 grid((rows + 15) / 16, n * topk);
    tq3_matvec_h_k<1, true><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), x, y, rows, k,
                                               cb.lo, cb.hi, ids, topk, per_expert);
}

void tq3_expert_gemm(const void * w, const __half * x, __nv_bfloat16 * y, int rows, int k,
                     const int * counts, const int * groups, int capacity, int experts,
                     int topk, bool per_expert, cudaStream_t s) {
    if (!w || !x || !y || !counts || !groups || rows <= 0 || k <= 0 || k % 32 || capacity <= 0 ||
        experts <= 0 || experts > 65535 || topk <= 0 ||
        (int64_t(capacity) + 63) / 64 > 65535)
        throw std::invalid_argument("tq3_expert_gemm: invalid shape or pointer");
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const dim3 grid((rows + 63) / 64, (capacity + 63) / 64, experts);
    tq3_gemm_h_k<true, true><<<grid, 256, 0, s>>>(static_cast<const uint4 *>(w), x, y, rows, k, capacity,
                                               cb.lo, cb.hi, counts, groups, capacity, topk, per_expert);
}

void tq3_dequant_h(const void * w, __half * out, int rows, int k, cudaStream_t s) {
    static const Tq3CodebookU cb = tq3_codebook_u8();
    const int64_t nb = int64_t(rows) * (k / 32);
    tq3_dequant_h_k<<<unsigned((nb + 255) / 256), 256, 0, s>>>(static_cast<const uint4 *>(w), out, nb, cb.lo, cb.hi);
}

__global__ void h_to_bf16_k(const __half * x, __nv_bfloat16 * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(__half2float(x[i]));
}

void h_to_bf16(const __half * x, __nv_bfloat16 * y, int64_t n, cudaStream_t s) {
    h_to_bf16_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, y, n);
}

void f32_to_bf16(const float * x, __nv_bfloat16 * y, int64_t n, cudaStream_t s) {
    f32_to_bf16_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, y, n);
}

void topk16_big(const float * x, int len, int rows, float * vals, uint32_t * idx, void * scratch, cudaStream_t s) {
    constexpr int NCH = 64;
    const int chunk = (len + NCH - 1) / NCH;
    auto * cv = static_cast<float *>(scratch);
    auto * ci = reinterpret_cast<uint32_t *>(cv + size_t(rows) * NCH * 16);
    topk16_chunk_k<<<dim3(NCH, rows), 256, 0, s>>>(x, len, chunk, cv, ci);
    topk16_merge_k<<<rows, 32, 0, s>>>(cv, ci, NCH * 16, vals, idx);
}

void bf16_to_h(const __nv_bfloat16 * x, __half * y, int n, cudaStream_t s) {
    bf16_to_h_k<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
}

void bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t s) {
    bf16_to_f32_k<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
}

} // namespace tqllm_cuda

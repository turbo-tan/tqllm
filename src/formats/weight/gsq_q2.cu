#include <cstdint>
#include "gsq_q2.cuh"
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace tqllm_cuda {
namespace {
// GSQ Q2_0: F16 scale, 16 bytes of adjacent 2-bit codes per 64 weights; value = scale * (code - 1).
__device__ float value(const unsigned char * data, int64_t element) {
    const auto * block = data + (element / 64) * 18;
    const int j = int(element % 64);
    const float scale = __half2float(*reinterpret_cast<const __half *>(block));
    return scale * (int((block[2 + j / 4] >> (2 * (j % 4))) & 3u) - 1);
}
template<class T> __global__ void dot(const unsigned char * weights, const __half * x, T * y,
                                     int rows, int cols, int experts, const int * ids, int topk, bool per_expert) {
    const int row = blockIdx.x, assignment = blockIdx.y;
    const int expert = ids ? ids[assignment] : 0;
    const int input = per_expert ? assignment : assignment / topk;
    float sum = 0.f;
    if (expert >= 0 && expert < experts)
        for (int j = threadIdx.x; j < cols; j += blockDim.x)
            sum += value(weights, (int64_t(expert) * rows + row) * cols + j) * __half2float(x[int64_t(input) * cols + j]);
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    __shared__ float partial[4];
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 4 ? partial[threadIdx.x] : 0.f;
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (!threadIdx.x) {
            if (expert < 0 || expert >= experts) sum = nanf("");
            if constexpr (std::is_same_v<T, float>) y[int64_t(assignment) * rows + row] = sum;
            else y[int64_t(assignment) * rows + row] = __float2bfloat16_rn(sum);
        }
    }
}
// Warp per (row, assignment): each 64-weight block (18 bytes) is read by the whole warp, 2 weights per lane.
template<class T> __global__ void dot_warp(const unsigned char * weights, const __half * x, T * y,
                                          int rows, int cols, int experts, const int * ids, int topk, bool per_expert) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31, assignment = blockIdx.y;
    if (row >= rows) return;
    const int expert = ids ? ids[assignment] : 0;
    const int input = per_expert ? assignment : assignment / topk;
    float sum = 0.f;
    if (expert >= 0 && expert < experts) {
        const unsigned char * r = weights + (int64_t(expert) * rows + row) * (cols / 64) * 18;
        const __half2 * xv = reinterpret_cast<const __half2 *>(x + int64_t(input) * cols);
        for (int b = 0; b < cols / 64; ++b) {
            const unsigned char * block = r + b * 18;
            const float scale = __half2float(__ushort_as_half(uint16_t(block[0] | block[1] << 8)));
            const unsigned q = block[2 + lane / 2] >> (4 * (lane & 1));
            const float2 xx = __half22float2(xv[b * 32 + lane]);
            sum += scale * (int(q & 3u) - 1) * xx.x;
            sum += scale * (int((q >> 2) & 3u) - 1) * xx.y;
        }
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane) {
        if (expert < 0 || expert >= experts) sum = nanf("");
        if constexpr (std::is_same_v<T, float>) y[int64_t(assignment) * rows + row] = sum;
        else y[int64_t(assignment) * rows + row] = __float2bfloat16_rn(sum);
    }
}
// Grouped experts (moe_group lists): warp per (row, expert, up to C of its tokens); the row's blocks are read once for
// all C tokens. Per token the products and their order are those of dot_warp (bit-identical).
// Grouped experts (moe_group lists): block = 64 rows (8 warps x 8 rows) of one expert; the expert's tokens C at a time
// are staged in shared memory once per block (not once per row), each warp then streams its rows' 2-bit blocks.
// Per (row, token) the products and their order are those of dot_warp.
template<int C> __global__ void __launch_bounds__(256) dot_grouped(const unsigned char * weights, const __half * x, __nv_bfloat16 * y, int rows, int cols,
                                            const int * counts, const int * groups, int capacity, int topk, bool per_expert) {
    const int expert = blockIdx.y, count = counts[expert];
    if (count == 0) return;
    extern __shared__ __half2 xs[];  // [C][cols / 2]
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, half_cols = cols / 2;
    const int row0 = blockIdx.x * 64 + warp * 8;
    for (int c0 = 0; c0 < count; c0 += C) {
        const int nc = min(C, count - c0);
        const int * g = groups + int64_t(expert) * capacity + c0;
        __syncthreads();
        for (int i = threadIdx.x; i < nc * half_cols; i += blockDim.x) {
            const int c = i / half_cols, j = i % half_cols, a = g[c];
            xs[i] = reinterpret_cast<const __half2 *>(x + int64_t(per_expert ? a : a / topk) * cols)[j];
        }
        __syncthreads();
        for (int rr = 0; rr < 8; ++rr) {
            const int row = row0 + rr;
            if (row >= rows) break;
            const unsigned char * r = weights + (int64_t(expert) * rows + row) * (cols / 64) * 18;
            float sum[C];
#pragma unroll
            for (int c = 0; c < C; ++c) sum[c] = 0.f;
            for (int b = 0; b < cols / 64; ++b) {
                const unsigned char * block = r + b * 18;
                const float scale = __half2float(__ushort_as_half(uint16_t(block[0] | block[1] << 8)));
                const unsigned q = block[2 + lane / 2] >> (4 * (lane & 1));
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    if (c >= nc) break;
                    const float2 xx = __half22float2(xs[c * half_cols + b * 32 + lane]);
                    sum[c] += scale * (int(q & 3u) - 1) * xx.x;
                    sum[c] += scale * (int((q >> 2) & 3u) - 1) * xx.y;
                }
            }
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= nc) break;
                float v = sum[c];
                for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
                if (!lane) y[int64_t(g[c]) * rows + row] = __float2bfloat16_rn(v);
            }
        }
    }
}
__global__ void decode(const unsigned char * weights, __half * out, int64_t count) {
    const int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < count) out[j] = __float2half_rn(value(weights, j));
}
bool old_dot() { static const bool v = std::getenv("TQFN_OLD_Q2_DOT") != nullptr; return v; }  // A/B: one block per row
void shape(const void * w, const void * x, const void * y, int rows, int cols, int n, int topk = 1) {
    if (!w || !x || !y || rows <= 0 || cols <= 0 || cols % 64 || n <= 0 || topk <= 0 || int64_t(n) * topk > 65535)
        throw std::invalid_argument("GSQ Q2_0: invalid pointer, shape or grid");
}
}
void gsq_q2_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int r, int k, int n, cudaStream_t s) {
    shape(w, x, y, r, k, n);
    if (old_dot()) dot<<<dim3(unsigned(r), unsigned(n)), 128, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, 1, nullptr, 1, false);
    else dot_warp<<<dim3(unsigned((r + 7) / 8), unsigned(n)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, 1, nullptr, 1, false);
}
void gsq_q2_matvec_f32(const void * w, const __half * x, float * y, int r, int k, int n, cudaStream_t s) {
    shape(w, x, y, r, k, n);
    if (old_dot()) dot<<<dim3(unsigned(r), unsigned(n)), 128, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, 1, nullptr, 1, false);
    else dot_warp<<<dim3(unsigned((r + 7) / 8), unsigned(n)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, 1, nullptr, 1, false);
}
void gsq_q2_expert_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int r, int k,
                          int experts, const int * ids, int n, int topk, bool per_expert, cudaStream_t s) {
    shape(w, x, y, r, k, n, topk);
    if (!ids || experts <= 0 || int64_t(experts) * r > INT_MAX)
        throw std::invalid_argument("GSQ Q2_0: invalid expert dimensions");
    if (old_dot()) dot<<<dim3(unsigned(r), unsigned(n * topk)), 128, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, experts, ids, topk, per_expert);
    else dot_warp<<<dim3(unsigned((r + 7) / 8), unsigned(n * topk)), 256, 0, s>>>(static_cast<const unsigned char *>(w), x, y, r, k, experts, ids, topk, per_expert);
}
void gsq_q2_expert_grouped(const void * w, const __half * x, __nv_bfloat16 * y, int r, int k, const int * counts, const int * groups,
                           int capacity, int experts, int topk, bool per_expert, cudaStream_t s) {
    shape(w, x, y, r, k, 1, 1);
    if (!counts || !groups || experts <= 0 || experts > 65535 || capacity <= 0 || int64_t(experts) * r > INT_MAX)
        throw std::invalid_argument("GSQ Q2_0: invalid grouped expert dimensions");
    constexpr int C = 16;
    const size_t smem = size_t(C) * k * sizeof(__half);
    static int attr_for = 0;
    if (smem > 48 * 1024 && attr_for < int(smem)) { cudaFuncSetAttribute(dot_grouped<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)); attr_for = int(smem); }
    dot_grouped<C><<<dim3(unsigned((r + 63) / 64), unsigned(experts)), 256, smem, s>>>(
        static_cast<const unsigned char *>(w), x, y, r, k, counts, groups, capacity, topk, per_expert);
}
void gsq_q2_dequant(const void * w, __half * out, int r, int k, cudaStream_t s) {
    shape(w, out, out, r, k, 1);
    const int64_t count = int64_t(r) * k;
    if ((count + 255) / 256 > INT_MAX) throw std::invalid_argument("GSQ Q2_0: dequant grid overflow");
    decode<<<unsigned((count + 255) / 256), 256, 0, s>>>(static_cast<const unsigned char *>(w), out, count);
}
}

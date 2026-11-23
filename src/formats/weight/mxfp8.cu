// MXFP8 matrices (OCP microscaling FP8: E4M3 values, one E8M0 scale per 32 along a row), as the original MiniMax /
// DSpark checkpoints store their projections. Row-major rows of k E4M3 bytes followed by that row's k / 32 scale bytes,
// so stacked matrices and row ranges are plain byte ranges. One warp per row, FP32 accumulation, up to 16 columns: each
// column keeps its own accumulator, the same per-lane walk and the same warp tree, so a column's terms and their order do
// not depend on how many columns the launch holds.
#include "quant_kernels.cuh"

#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <stdexcept>

namespace {
__device__ __forceinline__ float e4m3(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t(b), __NV_E4M3)));
}
__device__ __forceinline__ float e8m0(uint8_t s) { return s ? __int_as_float(int(s) << 23) : 5.877472e-39f; }  // 2^(s-127)

template <int N, bool F32>
__global__ void mxfp8_matvec_k(const uint8_t * __restrict__ w, const __half * __restrict__ x, void * y, int rows, int k, int ldy) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const uint8_t * wr = w + size_t(row) * (k + k / 32);
    const uint8_t * sr = wr + k;
    float acc[N] = {};
    for (int i = lane * 16; i < k; i += 512) {  // 16 weights per lane: half of one 32-block
        const uint4 wv = __ldg(reinterpret_cast<const uint4 *>(wr + i));
        const uint8_t * we = reinterpret_cast<const uint8_t *>(&wv);
        const float s = e8m0(__ldg(sr + i / 32));
        float wf[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) wf[j] = e4m3(we[j]) * s;
#pragma unroll
        for (int c = 0; c < N; ++c) {
            const uint4 x0 = __ldg(reinterpret_cast<const uint4 *>(x + size_t(c) * k + i));
            const uint4 x1 = __ldg(reinterpret_cast<const uint4 *>(x + size_t(c) * k + i + 8));
            const __half * a = reinterpret_cast<const __half *>(&x0);
            const __half * b = reinterpret_cast<const __half *>(&x1);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[c] += wf[j] * __half2float(a[j]) + wf[j + 8] * __half2float(b[j]);
        }
    }
#pragma unroll
    for (int c = 0; c < N; ++c) {
        float v = acc[c];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) {
            if constexpr (F32) static_cast<float *>(y)[size_t(c) * ldy + row] = v;
            else static_cast<__nv_bfloat16 *>(y)[size_t(c) * ldy + row] = __float2bfloat16(v);
        }
    }
}

__global__ void mxfp8_dequant_h_k(const uint8_t * __restrict__ w, __half * out, size_t n, int k) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t r = i / size_t(k), c = i % size_t(k);
        const uint8_t * wr = w + r * size_t(k + k / 32);
        out[i] = __float2half(e4m3(wr[c]) * e8m0(wr[k + c / 32]));
    }
}

// any k % 32: a warp per row, 4 weights per lane per 128-column step (one column per launch slot as above)
template <int N, bool F32>
__global__ void mxfp8_matvec_any_k(const uint8_t * __restrict__ w, const __half * __restrict__ x, void * y, int rows, int k, int ldy) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const uint8_t * wr = w + size_t(row) * (k + k / 32);
    float acc[N] = {};
    for (int i = lane * 4; i < k; i += 128) {
        const uint32_t q = __ldg(reinterpret_cast<const uint32_t *>(wr + i));
        const float s = e8m0(__ldg(wr + k + i / 32));
        const float wf[4] = {e4m3(uint8_t(q)) * s, e4m3(uint8_t(q >> 8)) * s, e4m3(uint8_t(q >> 16)) * s, e4m3(uint8_t(q >> 24)) * s};
#pragma unroll
        for (int c = 0; c < N; ++c)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[c] += wf[j] * __half2float(x[size_t(c) * k + i + j]);
    }
#pragma unroll
    for (int c = 0; c < N; ++c) {
        float v = acc[c];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) {
            if constexpr (F32) static_cast<float *>(y)[size_t(c) * ldy + row] = v;
            else static_cast<__nv_bfloat16 *>(y)[size_t(c) * ldy + row] = __float2bfloat16(v);
        }
    }
}

// grid (rows / 4, n * topk): warp = (row, assignment)
__global__ void mxfp8_expert_matvec_k(const uint8_t * __restrict__ w, const __half * __restrict__ x, __nv_bfloat16 * y, int rows, int k,
                                      const int * ids, int topk, bool per_expert) {
    const int a = blockIdx.y, row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e = ids[a];
    if (row >= rows || e < 0) return;
    const uint8_t * wr = w + (size_t(e) * rows + row) * (k + k / 32);
    const __half * xr = x + size_t(per_expert ? a : a / topk) * k;
    float acc = 0.0f;
    for (int i = lane * 4; i < k; i += 128) {
        const uint32_t q = __ldg(reinterpret_cast<const uint32_t *>(wr + i));
        const float s = e8m0(__ldg(wr + k + i / 32));
        const __half2 x01 = *reinterpret_cast<const __half2 *>(xr + i), x23 = *reinterpret_cast<const __half2 *>(xr + i + 2);
        acc += (e4m3(uint8_t(q)) * __low2float(x01) + e4m3(uint8_t(q >> 8)) * __high2float(x01) + e4m3(uint8_t(q >> 16)) * __low2float(x23) +
                e4m3(uint8_t(q >> 24)) * __high2float(x23)) * s;
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[size_t(a) * rows + row] = __float2bfloat16(acc);
}

// grid (rows / 64, experts * tiles): block = 64 rows x 64 assignments of one expert; 32-column steps (one scale block)
constexpr int GM = 64, GN = 64, GK = 32, GL = GK + 8, GC = GN + 4;
__global__ void __launch_bounds__(128) mxfp8_expert_gemm_k(const uint8_t * __restrict__ w, const __half * __restrict__ x, __nv_bfloat16 * y, int rows,
                                                         int k, const int * counts, const int * groups, int capacity, int tiles, int topk,
                                                         bool per_expert) {
    using namespace nvcuda;
    const int e = blockIdx.y / tiles, j0 = (blockIdx.y % tiles) * GN, r0 = blockIdx.x * GM;
    const int cnt = counts[e] - j0;
    if (cnt <= 0) return;
    __shared__ __align__(128) unsigned char smem[2 * GM * GL * 2 > GM * GC * 4 ? 2 * GM * GL * 2 : GM * GC * 4];
    __shared__ int arow[GN];
    auto sA = reinterpret_cast<__half (*)[GL]>(smem);
    auto sB = reinterpret_cast<__half (*)[GL]>(smem + GM * GL * 2);
    auto sC = reinterpret_cast<float (*)[GC]>(smem);
    if (threadIdx.x < GN) arow[threadIdx.x] = int(threadIdx.x) < cnt ? groups[size_t(e) * capacity + j0 + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x / 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[4];
    for (int i = 0; i < 4; ++i) wmma::fill_fragment(c[i], 0.0f);
    const size_t rb = size_t(k) + k / 32;
    for (int k0 = 0; k0 < k; k0 += GK) {
        {   // weights: thread = (row, 16 columns)
            const int r = threadIdx.x / 2, q = threadIdx.x % 2;
            __half2 * d = reinterpret_cast<__half2 *>(&sA[r][q * 16]);
            if (r0 + r < rows) {
                const uint8_t * wr = w + (size_t(e) * rows + r0 + r) * rb;
                const float s = e8m0(__ldg(wr + k + k0 / 32));
                const uint32_t * cp = reinterpret_cast<const uint32_t *>(wr + k0 + q * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t v = __ldg(cp + j);
                    d[2 * j] = __floats2half2_rn(e4m3(uint8_t(v)) * s, e4m3(uint8_t(v >> 8)) * s);
                    d[2 * j + 1] = __floats2half2_rn(e4m3(uint8_t(v >> 16)) * s, e4m3(uint8_t(v >> 24)) * s);
                }
            } else {
                for (int j = 0; j < 8; ++j) d[j] = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        for (int c8 = threadIdx.x; c8 < GN * (GK / 8); c8 += blockDim.x) {  // activations, 8 halves per step
            const int jj = c8 / (GK / 8), q = c8 % (GK / 8), a = arow[jj];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (a >= 0) v = *reinterpret_cast<const uint4 *>(x + size_t(per_expert ? a : a / topk) * k + k0 + q * 8);
            *reinterpret_cast<uint4 *>(&sB[jj][q * 8]) = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, &sA[warp * 16][kk], GL);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, &sB[16 * i][kk], GL);
                wmma::mma_sync(c[i], fa, fb, c[i]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) wmma::store_matrix_sync(&sC[warp * 16][16 * i], c[i], GC, wmma::mem_row_major);
    __syncthreads();
    const int jj = threadIdx.x % GN, i0 = (threadIdx.x / GN) * (GM / 2), a = arow[jj];
    if (a < 0) return;
    for (int i = 0; i < GM / 2; ++i)
        if (r0 + i0 + i < rows) y[size_t(a) * rows + r0 + i0 + i] = __float2bfloat16(sC[i0 + i][jj]);
}

template <bool F32>
void launch_any(const uint8_t * w, const __half * x, void * y, int rows, int k, int n, cudaStream_t s, int ldy) {
    const dim3 grid((rows + 3) / 4), block(128);
    switch (n) {
        case 1: mxfp8_matvec_any_k<1, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 2: mxfp8_matvec_any_k<2, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 3: mxfp8_matvec_any_k<3, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 4: mxfp8_matvec_any_k<4, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 5: mxfp8_matvec_any_k<5, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 6: mxfp8_matvec_any_k<6, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 7: mxfp8_matvec_any_k<7, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 8: mxfp8_matvec_any_k<8, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        default:
            for (int c = 0; c < n; c += 8) {
                const int m = std::min(8, n - c);
                const __half * xc = x + size_t(c) * k;
                void * yc = F32 ? static_cast<void *>(static_cast<float *>(y) + size_t(c) * ldy) : static_cast<void *>(static_cast<__nv_bfloat16 *>(y) + size_t(c) * ldy);
                launch_any<F32>(w, xc, yc, rows, k, m, s, ldy);
            }
    }
}
template <bool F32>
void launch(const uint8_t * w, const __half * x, void * y, int rows, int k, int n, cudaStream_t s, int ldy) {
    if (!ldy) ldy = rows;
    const dim3 grid((rows + 3) / 4), block(128);
    switch (n) {
        case 1: mxfp8_matvec_k<1, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 2: mxfp8_matvec_k<2, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 3: mxfp8_matvec_k<3, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 4: mxfp8_matvec_k<4, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 5: mxfp8_matvec_k<5, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 6: mxfp8_matvec_k<6, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 7: mxfp8_matvec_k<7, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 8: mxfp8_matvec_k<8, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 9: mxfp8_matvec_k<9, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 10: mxfp8_matvec_k<10, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 11: mxfp8_matvec_k<11, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 12: mxfp8_matvec_k<12, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 13: mxfp8_matvec_k<13, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 14: mxfp8_matvec_k<14, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 15: mxfp8_matvec_k<15, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        case 16: mxfp8_matvec_k<16, F32><<<grid, block, 0, s>>>(w, x, y, rows, k, ldy); break;
        default: throw std::invalid_argument("mxfp8_matvec: 1..16 columns");
    }
}
}  // namespace

namespace tqllm_cuda {
void mxfp8_matvec(const void * w, const __half * x, void * y, bool f32_out, int rows, int k, int n, cudaStream_t s, int ldy) {
    if (k % 32) throw std::invalid_argument("mxfp8_matvec: columns must be a multiple of 32");
    if (n < 1 || n > 16) throw std::invalid_argument("mxfp8_matvec: 1..16 columns");
    const auto * q = static_cast<const uint8_t *>(w);
    if (k % 512) {  // 16 weights per lane need k % 512; other widths take 4 per lane
        f32_out ? launch_any<true>(q, x, y, rows, k, n, s, ldy ? ldy : rows) : launch_any<false>(q, x, y, rows, k, n, s, ldy ? ldy : rows);
        return;
    }
    f32_out ? launch<true>(q, x, y, rows, k, n, s, ldy) : launch<false>(q, x, y, rows, k, n, s, ldy);
}
void mxfp8_dequant_h(const void * w, __half * out, int rows, int k, cudaStream_t s) {
    const size_t n = size_t(rows) * k;
    const auto * q = static_cast<const uint8_t *>(w);
    mxfp8_dequant_h_k<<<int(std::min<size_t>((n + 255) / 256, 4096)), 256, 0, s>>>(q, out, n, k);
}
void mxfp8_expert_matvec(const void * w, const __half * x, __nv_bfloat16 * y, int rows, int k, const int * ids, int n, int topk,
                         bool per_expert_input, cudaStream_t s) {
    if (k % 32) throw std::invalid_argument("mxfp8_expert_matvec: columns must be a multiple of 32");
    if (n <= 0 || topk <= 0) return;
    mxfp8_expert_matvec_k<<<dim3((rows + 3) / 4, n * topk), 128, 0, s>>>(static_cast<const uint8_t *>(w), x, y, rows, k, ids, topk, per_expert_input);
}
void mxfp8_expert_gemm(const void * w, const __half * x, __nv_bfloat16 * y, int rows, int k, const int * counts, const int * groups, int capacity,
                       int experts, int topk, bool per_expert_input, cudaStream_t s) {
    if (k % 32) throw std::invalid_argument("mxfp8_expert_gemm: columns must be a multiple of 32");
    if (capacity <= 0 || experts <= 0) return;
    const int tiles = (capacity + GN - 1) / GN;  // blocks past an expert's count return at once
    mxfp8_expert_gemm_k<<<dim3((rows + GM - 1) / GM, experts * tiles), 128, 0, s>>>(static_cast<const uint8_t *>(w), x, y, rows, k, counts, groups,
                                                                                  capacity, tiles, topk, per_expert_input);
}
}  // namespace tqllm_cuda

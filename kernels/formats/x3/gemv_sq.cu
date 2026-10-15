// gemv_sq.cu: the n == 1 X3 GEMV ("sq" = plain int8, mode 2), ported verbatim (source line
// comments kept) from llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu (exl3_gemv_int8_kernel.cuh +
// exl3_gemv_int8.cu host side, sq path only).
//
// Activations are Hadamard'd (suh) and quantised to int8 per 128-span (q_s); the trellis windows
// are multiplied by the mul1 codebook magic (a Codec call) and dotted with the int8 activations via
// dp4a. The k splits write per-slice partials; the ksplit-th contributor runs the fixed-order
// epilogue. Templated on Codec: only codecs with supports_sq (mul1-family) instantiate this.
#include "had128.cuh"
#include "ptx.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace x3 {

#define X3_SQ_THREADS 256
#define X3_GEMV_STAGE_D 4
#define X3_SQ_KSPLIT_CAP 64
#define X3_SQ_MINROWS 16
#define X3_SQ_ROWS_MAX 512
#define X3_SQ_COUNTERS_CAP 4096
#define X3_SQ_WS_RESERVED (X3_SQ_COUNTERS_CAP + 4 * X3_SQ_KSPLIT_CAP * 8)

__host__ __device__ constexpr bool sq_stage_smem(int bits) {
    return bits == 3 || bits == 5 || bits == 7 || bits == 15 || bits == 25 || bits == 35;
}

__host__ __device__ constexpr int sq_rows_max(int M, bool residual) {
    int cap = (80 * 1024) / (32 + 64 * M * (residual ? 2 : 1));
    cap &= ~7;
    return cap < X3_SQ_ROWS_MAX ? cap : X3_SQ_ROWS_MAX;
}

// One k-row of an adjacent block pair: extract (Codec) + dp4a for both blocks.
template <class Codec, int M, bool residual>
__device__ __forceinline__ void gemv_int8_pair_row(const uint32_t* blockA, const uint32_t* blockB,
                                                   const uint32_t* as_kb, int slice_stride, int c2, int t0,
                                                   int* ia0, int* ia1, int* ib0, int* ib1,
                                                   int* ja0, int* ja1, int* jb0, int* jb1) {
    uint32_t w[8];
    Codec::sq_windows(blockA, t0, w);
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ia0[r] = dp4a_us(w[0], as01.x, ia0[r]);
        ia0[r] = dp4a_us(w[1], as01.y, ia0[r]);
        ia0[r] = dp4a_us(w[2], as89.x, ia0[r]);
        ia0[r] = dp4a_us(w[3], as89.y, ia0[r]);
        ia1[r] = dp4a_us(w[4], as01.x, ia1[r]);
        ia1[r] = dp4a_us(w[5], as01.y, ia1[r]);
        ia1[r] = dp4a_us(w[6], as89.x, ia1[r]);
        ia1[r] = dp4a_us(w[7], as89.y, ia1[r]);
    }
    Codec::sq_windows(blockB, t0, w);
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        const uint32_t* as = as_kb + r * slice_stride;
        uint2 as01 = *(const uint2*) (as + c2);
        uint2 as89 = *(const uint2*) (as + c2 + 8);
        ib0[r] = dp4a_us(w[0], as01.x, ib0[r]);
        ib0[r] = dp4a_us(w[1], as01.y, ib0[r]);
        ib0[r] = dp4a_us(w[2], as89.x, ib0[r]);
        ib0[r] = dp4a_us(w[3], as89.y, ib0[r]);
        ib1[r] = dp4a_us(w[4], as01.x, ib1[r]);
        ib1[r] = dp4a_us(w[5], as01.y, ib1[r]);
        ib1[r] = dp4a_us(w[6], as89.x, ib1[r]);
        ib1[r] = dp4a_us(w[7], as89.y, ib1[r]);
    }
}

// Shared reduction tail: four lanes share each n; exclusive plain stores (atomic = false).
template <int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_pair_tail(int* __restrict__ accs, size_t acc_stride, int nbp,
                                                    int lane, int size_n, int* ia0, int* ia1, int* ib0, int* ib1,
                                                    int* ja0, int* ja1, int* jb0, int* jb1) {
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        #pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            ia0[r] += __shfl_xor_sync(0xffffffff, ia0[r], o);
            ia1[r] += __shfl_xor_sync(0xffffffff, ia1[r], o);
            ib0[r] += __shfl_xor_sync(0xffffffff, ib0[r], o);
            ib1[r] += __shfl_xor_sync(0xffffffff, ib1[r], o);
        }
    }
    if ((lane & 3) == 0) {
        int nA = (nbp * 2) * 16 + (lane >> 2);
        int nB = nA + 16;
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            int* acc = accs + r * acc_stride;
            if constexpr (atomic) {
                atomicAdd(acc + nA, ia0[r]);
                atomicAdd(acc + nA + 8, ia1[r]);
                atomicAdd(acc + nB, ib0[r]);
                atomicAdd(acc + nB + 8, ib1[r]);
            } else {
                acc[nA] = ia0[r];
                acc[nA + 8] = ia1[r];
                acc[nB] = ib0[r];
                acc[nB + 8] = ib1[r];
            }
        }
    }
}

// Narrow generic unit (any K): warp per adjacent block pair, pointer-based extraction.
template <class Codec, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_narrow(const uint16_t* __restrict__ B, int* __restrict__ accs,
                                                      size_t acc_stride, const uint32_t* __restrict__ sh_as,
                                                      int slice_stride, int nb256, int kb0, int nrows, int size_n) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = Codec::row_u32(size_n);
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * Codec::words;
    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb) {
        const uint32_t* blockA = bp + (size_t) kb * row_stride;
        gemv_int8_pair_row<Codec, M, residual>(blockA, blockA + Codec::words / 2,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// Smem-staged generic unit: warp-private cp.async staging (K = 3, 3.5, 2.5, 1.5).
template <class Codec, int M, bool residual, bool atomic = true>
__device__ __forceinline__ void gemv_int8_unit_smem(const uint16_t* __restrict__ B, int* __restrict__ accs,
                                                    size_t acc_stride, const uint32_t* __restrict__ sh_as,
                                                    int slice_stride, uint32_t* __restrict__ sh_b,
                                                    int nb256, int kb0, int nrows, int size_n) {
    constexpr int D = X3_GEMV_STAGE_D;
    constexpr int pairwords = Codec::words;
    constexpr int chunks = pairwords / 4;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int nbp = nb256 * 8 + warp;
    const int row_stride = Codec::row_u32(size_n);
    const uint32_t* bp = ((const uint32_t*) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t* sb = sh_b + warp * (D * pairwords);

    auto stage_row = [&](int kb) {
        if (kb < nrows && lane < chunks)
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        cp_async_fence();
    };
    #pragma unroll
    for (int r = 0; r < D - 1; ++r) stage_row(r);

    int c2 = 2 * (lane & 3);
    int ia0[M] = {}, ia1[M] = {}, ib0[M] = {}, ib1[M] = {};
    int ja0[M] = {}, ja1[M] = {}, jb0[M] = {}, jb1[M] = {};

    for (int kb = 0; kb < nrows; ++kb) {
        cp_async_wait<D - 2>();
        __syncwarp();
        stage_row(kb + D - 1);
        const uint32_t* blockA = sb + (kb % D) * pairwords;
        gemv_int8_pair_row<Codec, M, residual>(blockA, blockA + Codec::words / 2,
            sh_as + (kb << 4), slice_stride, c2, lane << 3,
            ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
    }
    gemv_int8_pair_tail<M, residual, atomic>(accs, acc_stride, nbp, lane, size_n, ia0, ia1, ib0, ib1, ja0, ja1, jb0, jb1);
}

// Stage one slice for M activation rows: Hadamard from A into sh_ah, slice max -> q_s, int8 splats
// + exact sums per row. Bit-identical in every block.
template <int M, bool residual>
__device__ __forceinline__ void gemv_int8_stage_slice(const half* __restrict__ A, int size_m, int size_k,
                                                      const half* __restrict__ suh, float* __restrict__ qs,
                                                      half* __restrict__ sh_ah, uint32_t* __restrict__ sh_as,
                                                      int slice_stride, float* __restrict__ sh_red,
                                                      int kb0, int nrows, const int a_f32) {
    int t = threadIdx.x;
    int nel = nrows * 16;
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        uint32_t* as = sh_as + r * slice_stride;
        uint32_t* as2 = sh_as + (M + r) * slice_stride;
        float* qsr = qs + 4 * r;
        __syncthreads();
        if (r >= size_m) {
            for (int i = t; i < nel; i += X3_SQ_THREADS) {
                as[i] = 0;
                if constexpr (residual) as2[i] = 0;
            }
            if (t == 0) {
                qsr[0] = 1.0f;
                ((int*) qsr)[1] = 0;
                ((int*) qsr)[2] = 0;
            }
            continue;
        }
        const half* Ar = a_f32 ? (const half*) ((const float*) A + (size_t) r * size_k)
                               : A + (size_t) r * size_k;
        for (int sp = t >> 5; sp < (nel >> 7); sp += X3_SQ_THREADS >> 5) {
            if (a_f32) {
                had_hf_r_128_inner_f32<true, false>((const float*) Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7),
                                                    suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
            } else {
                had_hf_r_128_inner<true, false>(Ar + (kb0 << 4) + (sp << 7), sh_ah + (sp << 7),
                                                suh + (kb0 << 4) + (sp << 7), 0.088388347648f);
            }
        }
        __syncthreads();

        float mx = 0.0f;
        for (int i = t; i < nel; i += X3_SQ_THREADS) mx = fmaxf(mx, fabsf(__half2float(sh_ah[i])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        if ((t & 31) == 0) sh_red[t >> 5] = mx;
        __syncthreads();
        if (t < 32) {
            float v = t < (X3_SQ_THREADS >> 5) ? sh_red[t] : 0.0f;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
            if (t == 0) sh_red[32] = fmaxf(v, 1e-8f) / 127.0f;
        }
        __syncthreads();
        float q_s = sh_red[32];

        float rq = 1.0f / q_s;
        float rq2 = rq * 254.0f;
        int l1 = 0, l2 = 0;
        for (int i = t; i < nel; i += X3_SQ_THREADS) {
            float a = __half2float(sh_ah[i]);
            int v = __float2int_rn(a * rq);
            v = max(-127, min(127, v));
            as[i] = ((uint32_t)(uint8_t)(int8_t) v) * 0x01010101u;
            l1 += v;
            if constexpr (residual) {
                float rr = a - q_s * (float) v;
                int v2 = __float2int_rn(rr * rq2);
                v2 = max(-127, min(127, v2));
                as2[i] = ((uint32_t)(uint8_t)(int8_t) v2) * 0x01010101u;
                l2 += v2;
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            l1 += __shfl_xor_sync(0xffffffff, l1, o);
            l2 += __shfl_xor_sync(0xffffffff, l2, o);
        }
        if ((t & 31) == 0) { ((int*) sh_red)[t >> 5] = l1; sh_red[16 + (t >> 5)] = __int_as_float(l2); }
        __syncthreads();
        if (t < 32) {
            int v1 = t < (X3_SQ_THREADS >> 5) ? ((int*) sh_red)[t] : 0;
            int v2 = t < (X3_SQ_THREADS >> 5) ? __float_as_int(sh_red[16 + t]) : 0;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v1 += __shfl_xor_sync(0xffffffff, v1, o);
                v2 += __shfl_xor_sync(0xffffffff, v2, o);
            }
            if (t == 0) {
                qsr[0] = sh_red[32];
                ((int*) qsr)[1] = v1;
                ((int*) qsr)[2] = v2;
            }
        }
    }
    __syncthreads();
}

// Epilogue for one 256-column group: fixed-order combine over per-slice partials, warp per
// (row, 128-span). Volatile reads: contributions arrived from other blocks with no grid barrier.
template <int M, bool c_fp32, bool residual>
__device__ __forceinline__ void gemv_int8_epilogue_group_sq(const int* __restrict__ partials,
                                                            const float* __restrict__ qsums, int pstride,
                                                            int ksplit, int size_m, void* __restrict__ C,
                                                            const half* __restrict__ svh, float* __restrict__ sh_tmp,
                                                            int nb256, int size_n) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    float k_inv  = __half2float(__ushort_as_half(0x1eee));
    float k_bias = __half2float(__ushort_as_half(0xc931));
    float aff = 1024.0f * k_inv + k_bias;

    float* tmp = sh_tmp + warp * 128;
    volatile const int* vpar = (volatile const int*) partials;
    volatile const float* vqs = (volatile const float*) qsums;
    volatile const int* vqsi = (volatile const int*) qsums;
    for (int p = warp; p < 2 * size_m; p += (X3_SQ_THREADS >> 5)) {
        int row = p >> 1;
        int base = nb256 * 256 + (p & 1) * 128;
        float acc[4] = {};
        float corr = 0.0f;
        for (int sl = 0; sl < ksplit; ++sl) {
            int idx = sl * M + row;
            float q_s = vqs[4 * idx];
            float suma = q_s * (float) vqsi[4 * idx + 1];
            volatile const int* q = vpar + (size_t) idx * pstride + base;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[i] += q_s * (float) q[lane * 4 + i];
            if constexpr (residual) {
                float q2_s = q_s * (1.0f / 254.0f);
                suma += q2_s * (float) vqsi[4 * idx + 2];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    acc[i] += q2_s * (float) q[size_n + lane * 4 + i];
            }
            corr += aff * suma;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            tmp[lane * 4 + i] = k_inv * acc[i] + corr;
        __syncwarp();
        if constexpr (c_fp32)
            had_ff_r_128_inner<false, true>(tmp, ((float*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
        else
            had_fh_r_128_inner<false, true>(tmp, ((half*) C) + (size_t) row * size_n + base, svh + base, 0.088388347648f);
    }
}

// The sq kernel, m == 1, mode 2 (plain int8), regular launch.
template <class Codec, int M, bool c_fp32, bool residual>
__global__ __launch_bounds__(X3_SQ_THREADS)
void exl3_gemv_int8_sq_kernel(const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
                              const int size_m, const int size_k, const int size_n, int* __restrict__ locks,
                              const half* __restrict__ suh, half* __restrict__ A_had,
                              const half* __restrict__ svh, const int a_f32) {
    static_assert(Codec::supports_sq, "codec does not support the int8 sq path");
    (void) A_had;
    extern __shared__ uint32_t shmem[];

    int rows_total = size_k >> 4;
    int nb256_total = size_n / 256;
    int r = (rows_total * nb256_total + (int) gridDim.x - 1) / (int) gridDim.x;
    int rows_per = (r > 2 * r ? r : (2 * r < 32 ? 2 * r : 32));
    rows_per = (rows_per + 7) & ~7;
    rows_per = rows_per < X3_SQ_MINROWS ? X3_SQ_MINROWS : rows_per;
    rows_per = rows_per > sq_rows_max(M, residual) ? sq_rows_max(M, residual) : rows_per;
    rows_per = rows_per > ((rows_total + 7) & ~7) ? ((rows_total + 7) & ~7) : rows_per;
    if constexpr (M > 1) {
        constexpr int smem_budget = 49152;
        constexpr int stage = (Codec::bits == 3 || Codec::frac) ? 8 * X3_GEMV_STAGE_D * Codec::words * 4 : 0;
        int cap = (smem_budget - 1024 * M - stage) / (32 + 64 * M);
        cap &= ~7;
        if (cap < X3_SQ_MINROWS) cap = X3_SQ_MINROWS;
        rows_per = rows_per > cap ? cap : rows_per;
    }
    int ksplit = (rows_total + rows_per - 1) / rows_per;
    int units = nb256_total * ksplit;
    int slice_stride = rows_per * 16;
    int pstride = size_n * (residual ? 2 : 1);

    int* counters = locks;
    float* qsums = (float*) (locks + X3_SQ_COUNTERS_CAP);
    int* partials = locks + X3_SQ_WS_RESERVED;

    half* sh_ah = (half*) shmem;
    uint32_t* sh_as = shmem + rows_per * 8;
    uint32_t* sh_b = sh_as + slice_stride * M * (residual ? 2 : 1);
    float* sh_tmp = (float*) (sh_b + (sq_stage_smem(Codec::bits) ? 8 * X3_GEMV_STAGE_D * Codec::words : 0));
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
        int slice = unit / nb256_total;
        int nb256 = unit % nb256_total;
        int kb0 = slice * rows_per;
        int nrows = rows_per < (rows_total - kb0) ? rows_per : (rows_total - kb0);
        if (slice != prev_slice) {
            gemv_int8_stage_slice<M, residual>(A, size_m, size_k, suh, qsums + 4 * slice * M,
                                               sh_ah, sh_as, slice_stride, sh_red, kb0, nrows, a_f32);
            prev_slice = slice;
        }
        int* pacc = partials + (size_t) slice * M * pstride;
        if constexpr (sq_stage_smem(Codec::bits))
            gemv_int8_unit_smem<Codec, M, residual, false>(B, pacc, pstride, sh_as, slice_stride, sh_b, nb256, kb0, nrows, size_n);
        else
            gemv_int8_unit_narrow<Codec, M, residual, false>(B, pacc, pstride, sh_as, slice_stride, nb256, kb0, nrows, size_n);

        __threadfence();
        __syncthreads();
        if (t == 0) sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        __syncthreads();
        if (sh_last) {
            gemv_int8_epilogue_group_sq<M, c_fp32, residual>(partials, qsums, pstride, ksplit, size_m,
                                                             C, svh, sh_tmp, nb256, size_n);
            if (t == 0) counters[nb256] = 0;
        }
    }
}

// ---- host side ----

inline int sq_num_sms() {
    static const int n = [] { int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v > 0 ? v : 82; }();
    return n;
}

// Launch the sq kernel at M = 1 (the engine uses sq only at n == 1). A is f32, C is f32 (a_f32 = 1,
// c_fp32 = true), matching the llama-paw sq path so results bit-match. `locks` is the caller-owned
// workspace: SQ_WS_RESERVED + size_n * ksplit * M ints (the plan clamps ksplit to SQ_KSPLIT_CAP).
template <class Codec, bool c_fp32>
void launch_sq(const float* A, const uint16_t* B, float* C, int size_k, int size_n,
               const half* suh, const half* svh, int* locks, cudaStream_t stream) {
    constexpr int M = 1;
    constexpr bool residual = false;
    const size_t words = Codec::words;
    const int rows_total = size_k / 16;
    const int nb256 = size_n / 256;
    const int rows_max = sq_rows_max(M, residual);

    auto decomp = [&](int grid, int& ksplit, int& rows_per) {
        int r = (rows_total * nb256 + grid - 1) / grid;
        rows_per = (r > 2 * r ? r : (2 * r < 32 ? 2 * r : 32));
        rows_per = (rows_per + 7) & ~7;
        rows_per = rows_per < X3_SQ_MINROWS ? X3_SQ_MINROWS : rows_per;
        rows_per = rows_per > rows_max ? rows_max : rows_per;
        rows_per = rows_per > ((rows_total + 7) & ~7) ? ((rows_total + 7) & ~7) : rows_per;
        ksplit = (rows_total + rows_per - 1) / rows_per;
    };
    auto smem_for = [&](int rows_per) {
        size_t stage = sq_stage_smem(Codec::bits) ? (size_t) 8 * X3_GEMV_STAGE_D * words * 4 : 0;
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M + stage + (size_t) 2 * M * 128 * 4;
    };

    auto kfn = exl3_gemv_int8_sq_kernel<Codec, M, c_fp32, residual>;
    int grid, ksplit, rows_per;
    decomp(6 * sq_num_sms(), ksplit, rows_per);
    int maxb = 1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, kfn, X3_SQ_THREADS, smem_for(rows_per)) != cudaSuccess)
        maxb = 1;
    grid = std::min(maxb * sq_num_sms(), 1024);
    decomp(grid, ksplit, rows_per);
    if (const char* e = getenv("GGML_PAW_SQ_GRID_MAX")) {
        if (atoi(e) > 0 && grid > atoi(e)) { grid = atoi(e); decomp(grid, ksplit, rows_per); }
    }
    const size_t smem = smem_for(rows_per);
    if (ksplit > X3_SQ_KSPLIT_CAP || nb256 > X3_SQ_COUNTERS_CAP)
        throw std::runtime_error("x3 sq: shape exceeds workspace caps");

    int g_size_m = M;
    const int a_f32 = 1;
    half* g_null_a_had = nullptr;
    void* args[] = {(void*) &A, (void*) &B, (void*) &C, (void*) &g_size_m, (void*) &size_k, (void*) &size_n,
                    (void*) &locks, (void*) &suh, (void*) &g_null_a_had, (void*) &svh, (void*) &a_f32};
    cudaError_t err = cudaLaunchKernel((const void*) kfn, dim3(grid), dim3(X3_SQ_THREADS), args, smem, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("x3 sq launch: ") + cudaGetErrorString(err));
}

// Bytes a caller must provide for the sq `locks` workspace at n == 1: counters + qsums + partials.
// ksplit <= rows_total (rows_per >= X3_SQ_MINROWS), so sizing on rows_total is a safe upper bound.
inline size_t sq_workspace_bytes(int size_k, int size_n) {
    const int rows_total = size_k / 16;
    return (size_t) (X3_SQ_WS_RESERVED + size_n * rows_total) * sizeof(int);
}

}  // namespace x3
}  // namespace tqllm

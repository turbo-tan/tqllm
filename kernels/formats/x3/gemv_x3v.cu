// gemv_x3v.cu: small-m X3 GEMV (2 <= m <= 8, MMODE 1; MMODE 0 is the m==1 fast path) ported from
// llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu (x3v_gemv_kernel, exllamav3 exl3_gemv_kernel.cuh).
//
// Warps split k and never synchronize in the main loop: B streams to registers behind a prefetch
// ring and the codec windows are resolved by lane shuffles, so no dynamic shared memory is needed
// (only sh_red, ~16 KB). One m16n8k16 MMA pair per 16x16 tile with fp16 accumulation, folded to
// fp32 on a cadence. Templated on Codec: every K- and mul1-specific step is a Codec call.
//
// CFG 0 ("narrow", 512 threads, 2 n-tiles/warp, 16 k-splits) wins at attention sizes; CFG 1
// ("wide", 256 threads, 4 n-tiles/warp, 8 k-splits) wins at large-n FFN sizes.
#include "had128.cuh"
#include "ptx.cuh"
#include <cooperative_groups.h>

namespace tqllm {
namespace x3 {

#define X3V_MAX_M 8

// mma.m16n8k16 with the A operand supplied as two FragB halves, fp16 accumulate.
__device__ __forceinline__ void x3v_mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b, FragCh& c) {
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        : "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]), "r"(bb[0]), "r"(bb[1])
    );
}

// A: (size_m, size_k) half (or f32 when a_f32), B: trellis, C: (size_m, size_n) half or f32.
// suh folded into A once (grid-sync), svh applied to C at the end. locks unused here.
template <class Codec, bool c_fp32, int MMODE, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256, (Codec::frac && (MMODE == 1 || CFG == 1)) ? 2 : 0)
void x3v_gemv_kernel(const half* __restrict__ A, const uint16_t* __restrict__ B, void* __restrict__ C,
                     const int size_m, const int size_k, const int size_n,
                     int* __restrict__ locks, const half* __restrict__ suh, half* __restrict__ A_had,
                     const half* __restrict__ svh, const int a_f32) {
    static_assert(Codec::supports_gemv_x3v,
                  "codec does not support the x3v gemv register path (mul1: 2, 3, 4 bpw and frac)");
    namespace cg_ = cooperative_groups;

    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : X3V_MAX_M;
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = Codec::u32words;      // uint32 per 16x16 tile
    constexpr int LOADS  = Codec::loads(WNT);    // warp loads per k-slice
    constexpr int LSTRIDE = Codec::ld_stride;

    auto grid = cg_::this_grid();

    // suh (input Hadamard scale) folded into A once.
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (; this_warp < total_warps; this_warp += warps_grid) {
            if (a_f32) {
                had_hf_r_128_inner_f32<true, false>((const float*) A + this_warp * 128, A_had + this_warp * 128,
                                                    suh + (this_warp * 128) % size_k, 0.088388347648f);
            } else {
                had_hf_r_128_inner<true, false>(A + this_warp * 128, A_had + this_warp * 128,
                                                suh + (this_warp * 128) % size_k, 0.088388347648f);
            }
        }
        grid.sync();
        A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles     = size_n / 16;
    const int kslices    = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = ceil_div(kslices, WK);
    const int ks0   = warp * chunk;
    const int myn   = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    __shared__ float sh_red[WK][ROWS][COLS];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x) {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        auto ld_b = [&](int i, int l) -> uint32_t {
            if constexpr (Codec::ld_lanes < 32)
                return lane < Codec::ld_lanes ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0u;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        FragCh ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF) {
        #pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t) {
                FragB f0, f1;
                Codec::regs_decode(bw, t, lane, f0, f1);
                x3v_mma_ab_h(a01, a23, f0, ch[t][0]);
                x3v_mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
        }

        // Cross-warp reduction over the k splits.
        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc0[t][f].x;
                        sh_red[warp][sr0][col + 1] = acc0[t][f].y;
                    }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*) C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    // svh (output Hadamard scale), same semantics as the GEMM epilogue.
    {
        grid.sync();
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (; this_warp < total_warps; this_warp += warps_grid) {
            if constexpr (c_fp32) {
                had_ff_r_128_inner<false, true>(((const float*) C) + this_warp * 128,
                                                ((float*) C) + this_warp * 128,
                                                svh + (this_warp * 128) % size_n, 0.088388347648f);
            } else {
                had_hf_r_128_inner<false, true>(((const half*) C) + this_warp * 128,
                                                ((half*) C) + this_warp * 128,
                                                svh + (this_warp * 128) % size_n, 0.088388347648f);
            }
        }
    }
}

}  // namespace x3
}  // namespace tqllm

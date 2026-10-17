// reconstruct.cu: dequantise an X3 matrix to fp16 [rows][cols], templated on a Codec. Ported
// verbatim (source line comments kept) from llama-paw-k25 ggml/src/ggml-cuda/paw-x3.cu.
//
// reconstruct_had_kernel  : fused dequant + both-side Hadamard, emits ORIGINAL-basis weights
//                           W = diag(suh) . H128 . W_hat . H128 . diag(svh) per 128x128 tile.
// x3_reconstruct_plain_kernel : dequant + tensor-core-layout shuffle only (TRANSFORMED basis).
#include "had128.cuh"
#include "x3_layout.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace tqllm {
namespace x3 {

#define X3_RH_THREADS 256
#define X3_RP_THREADS 256

// T is the (rows/16, cols/16, Codec::words) trellis; suh[cols], svh[rows]; rows/cols multiples of 128.
// OutMajor (tqLLM addition, epilogue only): write the tile as [out][in] with row stride ld_out, the engine's
// [rows][cols] GEMM layout, instead of the fork's [in][out]. Same values, so bit-identical to reconstruct +
// transpose, without the scratch matrix and the transpose pass.
template <class Codec, bool OutMajor = false>
__global__ __launch_bounds__(X3_RH_THREADS)
void reconstruct_had_kernel(half* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
                            const half* __restrict__ suh, const half* __restrict__ svh,
                            int packed_blocks_n, int packed_n_offset, int ld_out = 0) {
    constexpr int packed_size = Codec::words;
    constexpr float r_scale = kRScale;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&](int R, int q, int p) { return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p; };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += X3_RH_THREADS) {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const size_t tix_lin = (size_t) x3_tile_offset(kb * 8 + j, n, packed_blocks_n) + packed_n_offset;
        const uint16_t* gp = g_packed + tix_lin * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (X3_RH_THREADS / 32); ++jj) {
        int j = (warp_id / 8) * (8 / (X3_RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        half2 frag[4];
        Codec::decode(reinterpret_cast<const uint16_t*>(s_packed[j][wn]), lane_id, frag);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[2], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[3], 4, 32);

        if (!(lane_id & 4)) {
            half2 m0 = __halves2half2(__low2half(frag[0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[2]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[2]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[3]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[3]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (X3_RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq) {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        half2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
        #pragma unroll
        for (int x = 0; x < 2; ++x) {
            half2* v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    constexpr int ROWS_PW = 128 / (X3_RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr) {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        if constexpr (OutMajor) {   // back into this lane's own slot of row R (only this warp touches row R)
            stile[base] = v.x;
            stile[base + 1] = v.y;
        } else {
            *((half4*) (g_unpacked + (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
        }
    }
    if constexpr (OutMajor) {
        // stile row R (in index) holds out columns o at half2 R*64 + ((o>>2) ^ ((R>>2)&31))*2 + ((o&3)>>1).
        // Each warp writes 16 out rows; a lane gathers in indices R = lane*4 .. +3 (so (R>>2)&31 == lane) and
        // stores them as one half4: 256 contiguous bytes per warp.
        __syncthreads();
        const half* sh = reinterpret_cast<const half*>(stile);
        #pragma unroll 4
        for (int oo = 0; oo < 128 / (X3_RH_THREADS / 32); ++oo) {
            const int o = warp_id * (128 / (X3_RH_THREADS / 32)) + oo;
            const int col = (((o >> 2) ^ lane_id) * 2 + ((o & 3) >> 1)) * 2 + (o & 1);   // half index in a row
            half4 v;
            v.x = __halves2half2(sh[(lane_id * 4 + 0) * 128 + col], sh[(lane_id * 4 + 1) * 128 + col]);
            v.y = __halves2half2(sh[(lane_id * 4 + 2) * 128 + col], sh[(lane_id * 4 + 3) * 128 + col]);
            *((half4*) (g_unpacked + (size_t) (nb * 128 + o) * ld_out + kb * 128 + lane_id * 4)) = v;
        }
    }
}

// Host launcher: W is (n, m) row-major fp16; T is (n/16, m/16, words); n and m multiples of 128.
template <class Codec>
void reconstruct_had(half* W, const uint16_t* T, const half* suh, const half* svh, int n, int m, cudaStream_t stream) {
    dim3 grid((m + 127) / 128, (n + 127) / 128);
    reconstruct_had_kernel<Codec><<<grid, X3_RH_THREADS, 0, stream>>>(W, T, suh, svh, m / 16, 0);
}

// Same matrix written out-major: W[o * ld + i] for out o < m, in i < n (the engine's [rows][cols]); ld >= n,
// ld a multiple of 4.
template <class Codec>
void reconstruct_had_rows(half* W, int ld, const uint16_t* T, const half* suh, const half* svh, int n, int m,
                          cudaStream_t stream) {
    dim3 grid((m + 127) / 128, (n + 127) / 128);
    reconstruct_had_kernel<Codec, true><<<grid, X3_RH_THREADS, 0, stream>>>(W, T, suh, svh, m / 16, 0, ld);
}

// W is (n, m) row-major fp16 in the TRANSFORMED basis; T is (n/16, m/16, words); m multiple of 128.
template <class Codec>
__global__ __launch_bounds__(X3_RP_THREADS)
void x3_reconstruct_plain_kernel(half* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
                                 int packed_blocks_n, int packed_n_offset) {
    constexpr int packed_size = Codec::words;  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (size_t) (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((const int4*) g_packed)[t];
    __syncthreads();

    half2 frag[4];
    Codec::decode(reinterpret_cast<const uint16_t*>(s_packed[warp_id]), lane_id, frag);

    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[2], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[3], 4, 32);

    if (!(lane_id & 4)) {
        half2 m0 = __halves2half2(__low2half(frag[0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[2]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[2]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[3]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[3]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (size_t) (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

template <class Codec>
void reconstruct_plain(half* W, const uint16_t* T, int n, int m, cudaStream_t stream) {
    dim3 grid(m / 128, n / 16);
    x3_reconstruct_plain_kernel<Codec><<<grid, X3_RP_THREADS, 0, stream>>>(W, T, m / 16, 0);
}

}  // namespace x3
}  // namespace tqllm

// tqkbench: TQ3_4S single-column matvec variants on real model shapes; reports effective weight GB/s.
// Variants are templated so the fastest can be ported to kernels/quant.cu once it also passes test-quant.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

__device__ __constant__ float CB[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                                       0.230106f,  0.725222f,  1.277503f,  1.988943f};

__device__ __forceinline__ float e3m5(uint32_t b) {
    return b ? __uint_as_float((((b >> 5) + 118u) << 23) | ((b & 31u) << 18)) : 0.f;
}

// CBMODE 0: shared-memory codebook; 1: register select tree.
template <int CBMODE>
__device__ __forceinline__ float code(uint32_t i, const float * cbs, const float (&cr)[8]) {
    if constexpr (CBMODE == 0) return cbs[i];
    const float lo = (i & 2) ? ((i & 1) ? cr[3] : cr[2]) : ((i & 1) ? cr[1] : cr[0]);
    const float hi = (i & 2) ? ((i & 1) ? cr[7] : cr[6]) : ((i & 1) ? cr[5] : cr[4]);
    return (i & 4) ? hi : lo;
}

// RPW rows per warp, UB consecutive blocks per lane per iteration (more loads in flight).
template <int RPW, int UB, int CBMODE>
__global__ void __launch_bounds__(256) mv(const uint4 * w, const float * xr, __nv_bfloat16 * y, int rows, int k) {
    __shared__ float cbs[8];
    if (threadIdx.x < 8) cbs[threadIdx.x] = CB[threadIdx.x];
    __syncthreads();
    float cr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) cr[i] = CB[i];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    for (int b0 = lane * UB; b0 < nblk; b0 += 32 * UB) {
        uint4 blk[UB][RPW];
#pragma unroll
        for (int u = 0; u < UB; ++u)
#pragma unroll
            for (int r = 0; r < RPW; ++r)
                blk[u][r] = (b0 + u < nblk && row0 + r < rows) ? __ldg(&w[int64_t(row0 + r) * nblk + b0 + u])
                                                               : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            if (b0 + u >= nblk) break;
            float xv[32];
            const float4 * xp = reinterpret_cast<const float4 *>(xr + int64_t(b0 + u) * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 t = __ldg(&xp[i]);
                xv[4 * i] = t.x;
                xv[4 * i + 1] = t.y;
                xv[4 * i + 2] = t.z;
                xv[4 * i + 3] = t.w;
            }
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const uint4 q = blk[u][r];
                const uint32_t gb[4] = {q.y, (q.y >> 24) | (q.z << 8), (q.z >> 16) | (q.w << 16), q.w >> 8};
                float s = 0.f;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    float part = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) part += code<CBMODE>((gb[g] >> (3 * j)) & 7, cbs, cr) * xv[g * 8 + j];
                    s += e3m5((q.x >> (8 * g)) & 255) * part;
                }
                acc[r] += s;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2bfloat16_rn(v);
    }
}


// ---- int8 path: repacked weights + int8 activations, PRMT codebook lookup + DP4A ----
// Repacked block (16 bytes): x = 4 E3M5 scales (unchanged); y,z,w = W0,W1,W2. Elements 0..23: low 3 bits of
// nibble (e % 8) of word W[e / 8]. Elements 24..31 (group 3): bit t of index (24 + m) is bit 3 of nibble m of W[t].
constexpr float CB_Q = 63.5f;  // int8 codebook scale: round(c * 63.5), |c| <= 1.997

__device__ __forceinline__ float e3m5_fast(uint32_t b) {  // b: one scale byte
    return b ? __uint_as_float((b << 18) + 0x3B000000u) : 0.f;
}

template <int RPW, int UB>
__global__ void __launch_bounds__(256) mv8(const uint4 * w, const int4 * xq, const float * xs, __nv_bfloat16 * y,
                                            int rows, int k, uint32_t cblo, uint32_t cbhi) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int row0 = (blockIdx.x * 8 + warp) * RPW;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
    for (int b0 = lane * UB; b0 < nblk; b0 += 32 * UB) {
        uint4 blk[UB][RPW];
#pragma unroll
        for (int u = 0; u < UB; ++u)
#pragma unroll
            for (int r = 0; r < RPW; ++r)
                blk[u][r] = (b0 + u < nblk && row0 + r < rows) ? __ldg(&w[int64_t(row0 + r) * nblk + b0 + u])
                                                               : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int b = b0 + u;
            if (b >= nblk) break;
            const int4 xa = __ldg(&xq[2 * b]), xb = __ldg(&xq[2 * b + 1]);
            const float sx = __ldg(&xs[b]);
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                const uint4 q = blk[u][r];
                const uint32_t W0 = q.y, W1 = q.z, W2 = q.w;
                int s0 = __dp4a(int(__byte_perm(cblo, cbhi, W0 & 0x7777)), xa.x, 0);
                s0 = __dp4a(int(__byte_perm(cblo, cbhi, (W0 >> 16) & 0x7777)), xa.y, s0);
                int s1 = __dp4a(int(__byte_perm(cblo, cbhi, W1 & 0x7777)), xa.z, 0);
                s1 = __dp4a(int(__byte_perm(cblo, cbhi, (W1 >> 16) & 0x7777)), xa.w, s1);
                int s2 = __dp4a(int(__byte_perm(cblo, cbhi, W2 & 0x7777)), xb.x, 0);
                s2 = __dp4a(int(__byte_perm(cblo, cbhi, (W2 >> 16) & 0x7777)), xb.y, s2);
                const uint32_t sa = ((W0 >> 3) & 0x1111) | ((W1 >> 2) & 0x2222) | ((W2 >> 1) & 0x4444);
                const uint32_t sb = ((W0 >> 19) & 0x1111) | ((W1 >> 18) & 0x2222) | ((W2 >> 17) & 0x4444);
                int s3 = __dp4a(int(__byte_perm(cblo, cbhi, sa)), xb.z, 0);
                s3 = __dp4a(int(__byte_perm(cblo, cbhi, sb)), xb.w, s3);
                const float bs = e3m5_fast(q.x & 255) * float(s0) + e3m5_fast((q.x >> 8) & 255) * float(s1) +
                                 e3m5_fast((q.x >> 16) & 255) * float(s2) + e3m5_fast(q.x >> 24) * float(s3);
                acc[r] += sx * bs;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2bfloat16_rn(v / CB_Q);
    }
}

// Host repack of one TQ3_4S block (GGUF layout -> PRMT layout).
void repack(const uint8_t * in, uint8_t * out) {
    uint8_t idx[32];
    for (int g = 0; g < 4; ++g) {
        const uint32_t v = in[4 + 3 * g] | (in[5 + 3 * g] << 8) | (in[6 + 3 * g] << 16);
        for (int j = 0; j < 8; ++j) idx[g * 8 + j] = (v >> (3 * j)) & 7;
    }
    uint32_t W[3] = {0, 0, 0};
    for (int e = 0; e < 24; ++e) W[e / 8] |= uint32_t(idx[e]) << (4 * (e % 8));
    for (int m = 0; m < 8; ++m)
        for (int t = 0; t < 3; ++t) W[t] |= uint32_t((idx[24 + m] >> t) & 1) << (4 * m + 3);
    for (int i = 0; i < 4; ++i) out[i] = in[i];
    for (int t = 0; t < 3; ++t) for (int i = 0; i < 4; ++i) out[4 + 4 * t + i] = uint8_t(W[t] >> (8 * i));
}

struct Shape {
    const char * name;
    int rows, k;
    int count = 1;  // matrices of this shape per decode token (weights the "mix" total)
};

template <int RPW, int UB, int CBMODE>
void run(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const float * xr,
         __nv_bfloat16 * y) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 8 * RPW - 1) / (8 * RPW);
        for (int i = 0; i < 5; ++i) mv<RPW, UB, CBMODE><<<grid, 256>>>(w, xr, y, s.rows, s.k);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mv<RPW, UB, CBMODE><<<grid, 256>>>(w, xr, y, s.rows, s.k);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}


template <int RPW, int UB>
void run8(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const int4 * xq, const float * xs,
          __nv_bfloat16 * y, uint32_t cblo, uint32_t cbhi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 8 * RPW - 1) / (8 * RPW);
        for (int i = 0; i < 5; ++i) mv8<RPW, UB><<<grid, 256>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mv8<RPW, UB><<<grid, 256>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// ---- INT8 tensor-core multi-column verify matvec (mma.sync m16n8k16 s8) ----
// Warp tile: 16 weight rows x 8 activation columns. Per 32-weight block, two k16 halves; each half spans two
// 8-weight scale groups, so it runs as two masked MMAs (threads holding the other group's k slice feed zeros).
// A fragment (row-major 16x16): lane holds rows g=lane/4 and g+8, k = 4*(lane%4)..+3. B (16x8): k = 4*(lane%4)..+3,
// column n = lane/4. C (16x8): rows g, g+8; columns 2*(lane%4), +1.
__device__ __forceinline__ void mma16816(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b));
}

// codes for elements 16h + 4t .. +3 of a repacked block, as 4 int8 (PRMT on the int8 codebook)
__device__ __forceinline__ uint32_t tq3_codes4(const uint4 & q, int h, int t, uint32_t cblo, uint32_t cbhi) {
    uint32_t sel;
    if (h == 0) sel = (t < 2 ? q.y : q.z) >> (16 * (t & 1));
    else if (t < 2) sel = q.w >> (16 * t);
    else if (t == 2) sel = ((q.y >> 3) & 0x1111) | ((q.z >> 2) & 0x2222) | ((q.w >> 1) & 0x4444);
    else sel = ((q.y >> 19) & 0x1111) | ((q.z >> 18) & 0x2222) | ((q.w >> 17) & 0x4444);
    return __byte_perm(cblo, cbhi, sel & 0x7777);
}

template <int NCOL>
__global__ void __launch_bounds__(256) mvt(const uint4 * w, const int8_t * xq, const float * xs, __nv_bfloat16 * y,
                                            int rows, int k, uint32_t cblo, uint32_t cbhi) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = (blockIdx.x * 8 + warp) * 16;
    if (row0 >= rows) return;
    const int nblk = k / 32;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const int n0 = 2 * t, n1 = 2 * t + 1;  // this lane's C columns
    float acc[4] = {0.f, 0.f, 0.f, 0.f};   // (ra,n0) (ra,n1) (rb,n0) (rb,n1)
    for (int b = 0; b < nblk; ++b) {
        const uint4 qa = __ldg(&w[int64_t(ra) * nblk + b]), qb = __ldg(&w[int64_t(rb) * nblk + b]);
        const float sx0 = n0 < NCOL ? __ldg(&xs[int64_t(n0) * nblk + b]) : 0.f;
        const float sx1 = n1 < NCOL ? __ldg(&xs[int64_t(n1) * nblk + b]) : 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t bx = g < NCOL ? __ldg(reinterpret_cast<const uint32_t *>(xq + int64_t(g) * k + b * 32 + 16 * h + 4 * t)) : 0u;
            const uint32_t a0 = tq3_codes4(qa, h, t, cblo, cbhi), a1 = tq3_codes4(qb, h, t, cblo, cbhi);
#pragma unroll
            for (int part = 0; part < 2; ++part) {  // group 2h + part: lanes t<2 own part 0, t>=2 part 1
                const bool mine = (t >= 2) == (part == 1);
                int c[4] = {0, 0, 0, 0};
                mma16816(c, mine ? a0 : 0u, mine ? a1 : 0u, bx);
                const int grp = 2 * h + part;
                const float da = e3m5_fast((qa.x >> (8 * grp)) & 255), db = e3m5_fast((qb.x >> (8 * grp)) & 255);
                acc[0] += da * sx0 * float(c[0]);
                acc[1] += da * sx1 * float(c[1]);
                acc[2] += db * sx0 * float(c[2]);
                acc[3] += db * sx1 * float(c[3]);
            }
        }
    }
    const float inv = 1.f / CB_Q;
    if (row0 + g < rows) {
        if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g] = __float2bfloat16_rn(acc[0] * inv);
        if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g] = __float2bfloat16_rn(acc[1] * inv);
    }
    if (row0 + g + 8 < rows) {
        if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g + 8] = __float2bfloat16_rn(acc[2] * inv);
        if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g + 8] = __float2bfloat16_rn(acc[3] * inv);
    }
}

template <int NCOL>
void runt(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const int8_t * xq, const float * xs,
          __nv_bfloat16 * y, uint32_t cblo, uint32_t cbhi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 127) / 128;
        for (int i = 0; i < 5; ++i) mvt<NCOL><<<grid, 256>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvt<NCOL><<<grid, 256>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// Split-K variant: a CTA of 8 warps owns one 16-row tile; warp w walks blocks w*UB, w*UB + 8*UB, ... (UB blocks per
// step, loads hoisted), partial C accumulators are reduced through shared memory at the end.
template <int NCOL, int UB, int ABL = 0>
__global__ void __launch_bounds__(256) mvt2(const uint4 * w, const int8_t * xq, const float * xs, __nv_bfloat16 * y,
                                             int rows, int k, uint32_t cblo, uint32_t cbhi) {
    __shared__ float red[8][32][4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16;
    const int nblk = k / 32;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    const int n0 = 2 * t, n1 = 2 * t + 1;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b0 = warp * UB; b0 < nblk; b0 += 8 * UB) {
        uint4 qa[UB], qb[UB];
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int b = min(b0 + u, nblk - 1);
            qa[u] = __ldg(&w[int64_t(ra) * nblk + b]);
            qb[u] = __ldg(&w[int64_t(rb) * nblk + b]);
        }
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int b = b0 + u;
            if (b >= nblk) break;
            const float sx0 = n0 < NCOL ? __ldg(&xs[int64_t(n0) * nblk + b]) : 0.f;
            const float sx1 = n1 < NCOL ? __ldg(&xs[int64_t(n1) * nblk + b]) : 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t bx = g < NCOL ? __ldg(reinterpret_cast<const uint32_t *>(xq + int64_t(g) * k + b * 32 + 16 * h + 4 * t)) : 0u;
                const uint32_t a0 = tq3_codes4(qa[u], h, t, cblo, cbhi), a1 = tq3_codes4(qb[u], h, t, cblo, cbhi);
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const bool mine = (t >= 2) == (part == 1);
                    int c[4] = {0, 0, 0, 0};
                    if constexpr (ABL == 2) {
                        c[0] = int(a0 ^ bx); c[1] = int(a1 + bx); c[2] = int(a0 - bx); c[3] = int(a1 | bx);
                    } else {
                        mma16816(c, mine ? a0 : 0u, mine ? a1 : 0u, bx);
                    }
                    if constexpr (ABL == 1) {
                        acc[0] += float(c[0]); acc[1] += float(c[1]); acc[2] += float(c[2]); acc[3] += float(c[3]);
                    } else {
                        const int grp = 2 * h + part;
                        const float da = e3m5_fast((qa[u].x >> (8 * grp)) & 255), db = e3m5_fast((qb[u].x >> (8 * grp)) & 255);
                        acc[0] += da * sx0 * float(c[0]);
                        acc[1] += da * sx1 * float(c[1]);
                        acc[2] += db * sx0 * float(c[2]);
                        acc[3] += db * sx1 * float(c[3]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        for (int w2 = 1; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] += red[w2][lane][i];
        const float inv = 1.f / CB_Q;
        if (row0 + g < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g] = __float2bfloat16_rn(acc[0] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g] = __float2bfloat16_rn(acc[1] * inv);
        }
        if (row0 + g + 8 < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g + 8] = __float2bfloat16_rn(acc[2] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g + 8] = __float2bfloat16_rn(acc[3] * inv);
        }
    }
}

template <int NCOL, int UB, int ABL = 0>
void runt2(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const int8_t * xq, const float * xs,
           __nv_bfloat16 * y, uint32_t cblo, uint32_t cbhi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 15) / 16;
        for (int i = 0; i < 5; ++i) mvt2<NCOL, UB, ABL><<<grid, 256>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvt2<NCOL, UB, ABL><<<grid, 256>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// GEMM-style staging: CTA = 4 warps, one 16-row tile. K is walked in stages of 32 blocks (1024 weights). Each stage the
// CTA copies 16 rows x 32 blocks (8 KB, every warp moves whole 512 B row segments) and the NCOL x 1024 int8
// activations + scales into shared memory with cp.async (double buffered); warp w then runs the tensor-core math on
// blocks 8w..8w+7 of the stage. Row stride padded by 16 B so the 8 row groups of a fragment hit different banks.
__device__ __forceinline__ void cp16(void * smem, const void * gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int NCOL>
__global__ void __launch_bounds__(128) mvt3(const uint4 * w, const int8_t * xq, const float * xs, __nv_bfloat16 * y,
                                             int rows, int k, uint32_t cblo, uint32_t cbhi) {
    constexpr int SB = 32, RS = SB + 1;  // blocks per stage, padded row stride (uint4 units)
    __shared__ uint4 ws[2][16 * RS];
    __shared__ __align__(16) int8_t xsm[2][NCOL][SB * 32];
    __shared__ float ssm[2][NCOL][SB];
    __shared__ float red[4][32][4];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nblk = k / 32, nstage = (nblk + SB - 1) / SB;
    const int n0 = 2 * t, n1 = 2 * t + 1;

    auto load = [&](int st, int buf) {
        const int b0 = st * SB, nb = min(SB, nblk - b0);
        for (int i = tid; i < 16 * SB; i += 128) {  // warp-contiguous: 32 consecutive blocks of one row
            const int r = i / SB, bl = i % SB;
            if (bl < nb) cp16(&ws[buf][r * RS + bl], &w[int64_t(min(row0 + r, rows - 1)) * nblk + b0 + bl]);
        }
        for (int i = tid; i < NCOL * nb * 2; i += 128) {  // 16-byte chunks of activations
            const int n = i / (nb * 2), c = i % (nb * 2);
            cp16(&xsm[buf][n][c * 16], xq + int64_t(n) * k + int64_t(b0) * 32 + c * 16);
        }
        for (int i = tid; i < NCOL * nb; i += 128) ssm[buf][i / nb][i % nb] = xs[int64_t(i / nb) * nblk + b0 + i % nb];
        cp_commit();
    };

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    load(0, 0);
    for (int st = 0; st < nstage; ++st) {
        const int buf = st & 1;
        if (st + 1 < nstage) {
            load(st + 1, buf ^ 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const int nb = min(SB, nblk - st * SB);
#pragma unroll 2
        for (int j = 0; j < SB / 4; ++j) {
            const int bl = warp * (SB / 4) + j;
            if (bl >= nb) break;
            const uint4 qa = ws[buf][g * RS + bl], qb = ws[buf][(g + 8) * RS + bl];
            const float sx0 = n0 < NCOL ? ssm[buf][n0][bl] : 0.f, sx1 = n1 < NCOL ? ssm[buf][n1][bl] : 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t bx = g < NCOL ? *reinterpret_cast<const uint32_t *>(&xsm[buf][g][bl * 32 + 16 * h + 4 * t]) : 0u;
                const uint32_t a0 = tq3_codes4(qa, h, t, cblo, cbhi), a1 = tq3_codes4(qb, h, t, cblo, cbhi);
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const bool mine = (t >= 2) == (part == 1);
                    int c[4] = {0, 0, 0, 0};
                    mma16816(c, mine ? a0 : 0u, mine ? a1 : 0u, bx);
                    const int grp = 2 * h + part;
                    const float da = e3m5_fast((qa.x >> (8 * grp)) & 255), db = e3m5_fast((qb.x >> (8 * grp)) & 255);
                    acc[0] += da * sx0 * float(c[0]);
                    acc[1] += da * sx1 * float(c[1]);
                    acc[2] += db * sx0 * float(c[2]);
                    acc[3] += db * sx1 * float(c[3]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        for (int w2 = 1; w2 < 4; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] += red[w2][lane][i];
        const float inv = 1.f / CB_Q;
        if (row0 + g < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g] = __float2bfloat16_rn(acc[0] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g] = __float2bfloat16_rn(acc[1] * inv);
        }
        if (row0 + g + 8 < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g + 8] = __float2bfloat16_rn(acc[2] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g + 8] = __float2bfloat16_rn(acc[3] * inv);
        }
    }
}

template <int NCOL>
void runt3(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const int8_t * xq, const float * xs,
           __nv_bfloat16 * y, uint32_t cblo, uint32_t cbhi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 15) / 16;
        for (int i = 0; i < 5; ++i) mvt3<NCOL><<<grid, 128>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvt3<NCOL><<<grid, 128>>>(w, xq, xs, y, s.rows, s.k, cblo, cbhi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// ---- FP16 tensor-core verify: dequantise straight into mma A fragments (each weight decoded once) ----
// mma.m16n8k16 f16 (fp32 accumulate). A fragment: lane (g = lane/4, t = lane%4) holds rows g, g+8 and k columns
// {2t, 2t+1} (group 2h of the 32-block for k16 half h) and {2t+8, 2t+9} (group 2h+1). In the repacked layout the
// pair (2t, 2t+1) of each 8-element word is byte t of that word; & 0x77 gives an index into a half2 pair table.
__device__ __forceinline__ void mma16816h(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// half2 (s, s) for scale byte `byte` (E3M5 -> fp16 bits: (b << 5) + 0x1800; b == 0 -> 0)
__device__ __forceinline__ uint32_t scale2(uint32_t byte) {
    const uint32_t h = byte ? ((byte << 5) + 0x1800u) : 0u;
    return h | (h << 16);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Scaled half2 pair for (h, which): which 0 -> elements 16h+2t,+1 (group 2h), 1 -> 16h+8+2t,+1 (group 2h+1).
__device__ __forceinline__ uint32_t pairh(const uint4 & q, int h, int which, int t, const uint32_t * lut) {
    uint32_t byte;
    if (h == 0) byte = ((which ? q.z : q.y) >> (8 * t)) & 0x77u;
    else if (which == 0) byte = (q.w >> (8 * t)) & 0x77u;
    else byte = (((q.y >> (8 * t + 3)) & 0x11u) | ((q.z >> (8 * t + 2)) & 0x22u) | ((q.w >> (8 * t + 1)) & 0x44u));
    return hmul2(lut[byte], scale2((q.x >> (8 * (2 * h + which))) & 255u));
}

template <int NCOL, int UB>
__global__ void __launch_bounds__(256) mvh(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows, int k) {
    __shared__ uint32_t lut[128];
    __shared__ float red[8][32][4];
    if (threadIdx.x < 128) {  // pair table: index = lo | hi << 4 (3-bit codes), value = half2(cb[lo], cb[hi])
        const int lo = threadIdx.x & 7, hi = (threadIdx.x >> 4) & 7;
        const __half2 v = __halves2half2(__float2half_rn(CB[lo]), __float2half_rn(CB[hi]));
        lut[threadIdx.x] = *reinterpret_cast<const uint32_t *>(&v);
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nblk = k / 32;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b0 = warp * UB; b0 < nblk; b0 += 8 * UB) {
        uint4 qa[UB], qb[UB];
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int b = min(b0 + u, nblk - 1);
            qa[u] = __ldg(&w[int64_t(ra) * nblk + b]);
            qb[u] = __ldg(&w[int64_t(rb) * nblk + b]);
        }
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int b = b0 + u;
            if (b >= nblk) break;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int kb = b * 32 + 16 * h;
                uint32_t b0r = 0, b1r = 0;
                if (g < NCOL) {
                    b0r = __ldg(reinterpret_cast<const uint32_t *>(xh + int64_t(g) * k + kb + 2 * t));
                    b1r = __ldg(reinterpret_cast<const uint32_t *>(xh + int64_t(g) * k + kb + 8 + 2 * t));
                }
                mma16816h(c, pairh(qa[u], h, 0, t, lut), pairh(qb[u], h, 0, t, lut), pairh(qa[u], h, 1, t, lut),
                          pairh(qb[u], h, 1, t, lut), b0r, b1r);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = c[i];
    __syncthreads();
    if (warp == 0) {
        for (int w2 = 1; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) c[i] += red[w2][lane][i];
        const int n0 = 2 * t, n1 = 2 * t + 1;
        if (row0 + g < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g] = __float2bfloat16_rn(c[0]);
            if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g] = __float2bfloat16_rn(c[1]);
        }
        if (row0 + g + 8 < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + row0 + g + 8] = __float2bfloat16_rn(c[2]);
            if (n1 < NCOL) y[int64_t(n1) * rows + row0 + g + 8] = __float2bfloat16_rn(c[3]);
        }
    }
}

template <int NCOL, int UB>
void runh(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const __half * xh, __nv_bfloat16 * y) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 15) / 16;
        for (int i = 0; i < 5; ++i) mvh<NCOL, UB><<<grid, 256>>>(w, xh, y, s.rows, s.k);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvh<NCOL, UB><<<grid, 256>>>(w, xh, y, s.rows, s.k);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// MT m-tiles (16 rows each) per warp share every B (activation) fragment; split-K over the 8 warps of the CTA.
template <int NCOL, int MT>
__global__ void __launch_bounds__(256) mvh2(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows, int k) {
    __shared__ uint32_t lut[128];
    __shared__ float red[8][32][4 * MT];
    if (threadIdx.x < 128) {
        const int lo = threadIdx.x & 7, hi = (threadIdx.x >> 4) & 7;
        const __half2 v = __halves2half2(__float2half_rn(CB[lo]), __float2half_rn(CB[hi]));
        lut[threadIdx.x] = *reinterpret_cast<const uint32_t *>(&v);
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[m][i] = 0.f;
    for (int b = warp; b < nblk; b += 8) {
        uint4 qa[MT], qb[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            qa[m] = __ldg(&w[int64_t(ra[m]) * nblk + b]);
            qb[m] = __ldg(&w[int64_t(rb[m]) * nblk + b]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int kb = b * 32 + 16 * h;
            uint32_t b0r = 0, b1r = 0;
            if (g < NCOL) {
                b0r = __ldg(reinterpret_cast<const uint32_t *>(xh + int64_t(g) * k + kb + 2 * t));
                b1r = __ldg(reinterpret_cast<const uint32_t *>(xh + int64_t(g) * k + kb + 8 + 2 * t));
            }
#pragma unroll
            for (int m = 0; m < MT; ++m)
                mma16816h(c[m], pairh(qa[m], h, 0, t, lut), pairh(qb[m], h, 0, t, lut), pairh(qa[m], h, 1, t, lut),
                          pairh(qb[m], h, 1, t, lut), b0r, b1r);
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][4 * m + i] = c[m][i];
    __syncthreads();
    if (warp == 0) {
        const int n0 = 2 * t, n1 = 2 * t + 1;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * m + i];
            const int r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < NCOL) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0]);
                if (n1 < NCOL) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1]);
            }
            if (r + 8 < rows) {
                if (n0 < NCOL) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2]);
                if (n1 < NCOL) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3]);
            }
        }
    }
}

template <int NCOL, int MT>
void runh2(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const __half * xh, __nv_bfloat16 * y) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 16 * MT - 1) / (16 * MT);
        for (int i = 0; i < 5; ++i) mvh2<NCOL, MT><<<grid, 256>>>(w, xh, y, s.rows, s.k);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvh2<NCOL, MT><<<grid, 256>>>(w, xh, y, s.rows, s.k);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// ---- register-only FP16 decode + k-permuted fragments ----
// k permutation (applied to A and B alike, so the MMA sum is unchanged): lane t of a k16 half owns the 4 consecutive
// weights 16h + 4t .. +3 (one scale group: 2h + (t >= 2)). Fragment cols {2t, 2t+1} <- weights 4t, 4t+1 and
// {2t+8, 2t+9} <- 4t+2, 4t+3; the activation B fragment is then one 8-byte load of x[16h + 4t .. +3].
// Decode: tq3_codes4 gives 4 int8 codebook bytes (codebook stored +128 as unsigned); PRMT with 0x64 bytes makes
// fp16 (1024 + u); HFMA2 with (s, -1152 s) gives (u - 128) * s = c_q * s (the 1/63.5 goes into s).
__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

template <int NCOL, int MT>
__global__ void __launch_bounds__(256) mvh3(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                                             uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * MT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[m][i] = 0.f;
    const float inv = 1.f / CB_Q;
    for (int b = warp; b < nblk; b += 8) {
        uint4 qa[MT], qb[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            qa[m] = __ldg(&w[int64_t(ra[m]) * nblk + b]);
            qb[m] = __ldg(&w[int64_t(rb[m]) * nblk + b]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint2 xb = make_uint2(0, 0);
            if (g < NCOL) xb = __ldg(reinterpret_cast<const uint2 *>(xh + int64_t(g) * k + b * 32 + 16 * h + 4 * t));
            const int grp = 2 * h + (t >= 2);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                uint32_t a[4];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const uint4 & q = rr ? qb[m] : qa[m];
                    const uint32_t u4 = tq3_codes4(q, h, t, cblo_u, cbhi_u);           // 4 bytes u = c_q + 128
                    const uint32_t lo = __byte_perm(u4, 0x64646464u, 0x5140);            // halves 1024+u0, 1024+u1
                    const uint32_t hi = __byte_perm(u4, 0x64646464u, 0x5342);            // halves 1024+u2, 1024+u3
                    const float sc = e3m5_fast((q.x >> (8 * grp)) & 255) * inv;
                    const __half2 s2 = __float2half2_rn(sc), o2 = __float2half2_rn(-1152.f * sc);
                    const uint32_t su = *reinterpret_cast<const uint32_t *>(&s2), ou = *reinterpret_cast<const uint32_t *>(&o2);
                    a[rr] = hfma2(lo, su, ou);      // rows g / g+8, fragment cols 2t, 2t+1
                    a[rr + 2] = hfma2(hi, su, ou);  // fragment cols 2t+8, 2t+9
                }
                mma16816h(c[m], a[0], a[1], a[2], a[3], xb.x, xb.y);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][4 * m + i] = c[m][i];
    __syncthreads();
    if (warp == 0) {
        const int n0 = 2 * t, n1 = 2 * t + 1;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * m + i];
            const int r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < NCOL) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0]);
                if (n1 < NCOL) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1]);
            }
            if (r + 8 < rows) {
                if (n0 < NCOL) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2]);
                if (n1 < NCOL) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3]);
            }
        }
    }
}

template <int NCOL, int MT>
void runh3(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const __half * xh, __nv_bfloat16 * y,
           uint32_t lo, uint32_t hi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 16 * MT - 1) / (16 * MT);
        for (int i = 0; i < 5; ++i) mvh3<NCOL, MT><<<grid, 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvh3<NCOL, MT><<<grid, 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// mvh3 + split-K across CTAs (grid.y = KS): each CTA covers nblk/KS blocks; results are atomically added to an FP32
// output (zeroed beforehand; the engine converts/uses FP32 directly).
template <int NCOL, int MT, int KS>
__global__ void __launch_bounds__(256) mvh4(const uint4 * w, const __half * xh, float * y, int rows, int k,
                                             uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * MT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    const int bper = (nblk + KS - 1) / KS, bbeg = blockIdx.y * bper, bend = min(nblk, bbeg + bper);
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[m][i] = 0.f;
    const float inv = 1.f / CB_Q;
    for (int b = bbeg + warp; b < bend; b += 8) {
        uint4 qa[MT], qb[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            qa[m] = __ldg(&w[int64_t(ra[m]) * nblk + b]);
            qb[m] = __ldg(&w[int64_t(rb[m]) * nblk + b]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint2 xb = make_uint2(0, 0);
            if (g < NCOL) xb = __ldg(reinterpret_cast<const uint2 *>(xh + int64_t(g) * k + b * 32 + 16 * h + 4 * t));
            const int grp = 2 * h + (t >= 2);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                uint32_t a[4];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const uint4 & q = rr ? qb[m] : qa[m];
                    const uint32_t u4 = tq3_codes4(q, h, t, cblo_u, cbhi_u);
                    const uint32_t lo = __byte_perm(u4, 0x64646464u, 0x5140), hi = __byte_perm(u4, 0x64646464u, 0x5342);
                    const float sc = e3m5_fast((q.x >> (8 * grp)) & 255) * inv;
                    const __half2 s2 = __float2half2_rn(sc), o2 = __float2half2_rn(-1152.f * sc);
                    const uint32_t su = *reinterpret_cast<const uint32_t *>(&s2), ou = *reinterpret_cast<const uint32_t *>(&o2);
                    a[rr] = hfma2(lo, su, ou);
                    a[rr + 2] = hfma2(hi, su, ou);
                }
                mma16816h(c[m], a[0], a[1], a[2], a[3], xb.x, xb.y);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][4 * m + i] = c[m][i];
    __syncthreads();
    if (warp == 0) {
        const int n0 = 2 * t, n1 = 2 * t + 1;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * m + i];
            const int r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < NCOL) atomicAdd(&y[int64_t(n0) * rows + r], v[0]);
                if (n1 < NCOL) atomicAdd(&y[int64_t(n1) * rows + r], v[1]);
            }
            if (r + 8 < rows) {
                if (n0 < NCOL) atomicAdd(&y[int64_t(n0) * rows + r + 8], v[2]);
                if (n1 < NCOL) atomicAdd(&y[int64_t(n1) * rows + r + 8], v[3]);
            }
        }
    }
}

// Picks split-K per shape: target >= ~2 CTAs per SM (82 SMs on the 3090).
template <int NCOL, int MT>
void runh4(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const __half * xh, float * y,
           uint32_t lo, uint32_t hi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int gx = (s.rows + 16 * MT - 1) / (16 * MT);
        const int ks = gx >= 320 ? 1 : gx >= 160 ? 2 : 4;
        auto launch = [&]() {
            cudaMemsetAsync(y, 0, size_t(NCOL) * s.rows * 4);
            if (ks == 1) mvh4<NCOL, MT, 1><<<dim3(gx, 1), 256>>>(w, xh, y, s.rows, s.k, lo, hi);
            else if (ks == 2) mvh4<NCOL, MT, 2><<<dim3(gx, 2), 256>>>(w, xh, y, s.rows, s.k, lo, hi);
            else mvh4<NCOL, MT, 4><<<dim3(gx, 4), 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        };
        for (int i = 0; i < 5; ++i) launch();
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) launch();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// mvh3 with integer-only scale decode and branch-free code selection (lane constants hoisted out of the loop).
template <int NCOL, int MT, int MINB = 1, int UB = 1>
__global__ void __launch_bounds__(256, MINB) mvh5(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                                             uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4 * MT];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16 * MT, nblk = k / 32;
    const bool hi2 = t >= 2;
    const int sh16 = 16 * (t & 1);               // which 16-bit half of a word holds this lane's 4 nibbles
    const int sg0 = 8 * (0 + hi2), sg1 = 8 * (2 + hi2);  // scale byte shifts for h = 0, 1
    const uint32_t NEG1152 = 0xE480E480u;        // half2(-1152, -1152)
    int ra[MT], rb[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        ra[m] = min(row0 + 16 * m + g, rows - 1);
        rb[m] = min(row0 + 16 * m + g + 8, rows - 1);
    }
    float c[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[m][i] = 0.f;
    for (int b0 = warp * UB; b0 < nblk; b0 += 8 * UB) {
      uint4 qaa[UB][MT], qbb[UB][MT];
#pragma unroll
      for (int u = 0; u < UB; ++u)
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int bb = min(b0 + u, nblk - 1);
            qaa[u][m] = __ldg(&w[int64_t(ra[m]) * nblk + bb]);
            qbb[u][m] = __ldg(&w[int64_t(rb[m]) * nblk + bb]);
        }
#pragma unroll
      for (int u = 0; u < UB; ++u) {
        const int b = b0 + u;
        if (b >= nblk) break;
        uint4 (&qa)[MT] = qaa[u];
        uint4 (&qb)[MT] = qbb[u];
        uint2 xb0 = make_uint2(0, 0), xb1 = make_uint2(0, 0);
        if (g < NCOL) {
            xb0 = __ldg(reinterpret_cast<const uint2 *>(xh + int64_t(g) * k + b * 32 + 4 * t));
            xb1 = __ldg(reinterpret_cast<const uint2 *>(xh + int64_t(g) * k + b * 32 + 16 + 4 * t));
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            uint32_t a0[4], a1[4];
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const uint4 q = rr ? qb[m] : qa[m];
                // h = 0: elements 4t..4t+3 -> nibbles of W0 (t < 2) or W1 (t >= 2)
                const uint32_t sel0 = ((hi2 ? q.z : q.y) >> sh16) & 0x7777u;
                // h = 1: elements 16 + 4t.. -> W2 nibbles (t < 2) or the bit-sliced elements 24.. (t >= 2)
                const uint32_t sl = ((q.y >> (3 + sh16)) & 0x1111u) | ((q.z >> (2 + sh16)) & 0x2222u) |
                                    ((q.w >> (1 + sh16)) & 0x4444u);
                const uint32_t sel1 = hi2 ? sl : ((q.w >> sh16) & 0x7777u);
                const uint32_t u40 = __byte_perm(cblo_u, cbhi_u, sel0), u41 = __byte_perm(cblo_u, cbhi_u, sel1);
                const uint32_t b0 = (q.x >> sg0) & 255u, b1 = (q.x >> sg1) & 255u;
                const uint32_t h0 = b0 ? (b0 << 5) + 0x1800u : 0u, h1 = b1 ? (b1 << 5) + 0x1800u : 0u;
                const uint32_t s0 = h0 | (h0 << 16), s1 = h1 | (h1 << 16);
                const uint32_t o0 = hmul2(s0, NEG1152), o1 = hmul2(s1, NEG1152);
                a0[rr] = hfma2(__byte_perm(u40, 0x64646464u, 0x5140), s0, o0);
                a0[rr + 2] = hfma2(__byte_perm(u40, 0x64646464u, 0x5342), s0, o0);
                a1[rr] = hfma2(__byte_perm(u41, 0x64646464u, 0x5140), s1, o1);
                a1[rr + 2] = hfma2(__byte_perm(u41, 0x64646464u, 0x5342), s1, o1);
            }
            mma16816h(c[m], a0[0], a0[1], a0[2], a0[3], xb0.x, xb0.y);
            mma16816h(c[m], a1[0], a1[1], a1[2], a1[3], xb1.x, xb1.y);
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][lane][4 * m + i] = c[m][i];
    __syncthreads();
    if (warp == 0) {
        const int n0 = 2 * t, n1 = 2 * t + 1;
        const float inv = 1.f / CB_Q;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
            for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][4 * m + i];
            const int r = row0 + 16 * m + g;
            if (r < rows) {
                if (n0 < NCOL) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
                if (n1 < NCOL) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
            }
            if (r + 8 < rows) {
                if (n0 < NCOL) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
                if (n1 < NCOL) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
            }
        }
    }
}

template <int NCOL, int MT, int MINB = 1, int UB = 1>
void runh5(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const __half * xh, __nv_bfloat16 * y,
           uint32_t lo, uint32_t hi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 16 * MT - 1) / (16 * MT);
        for (int i = 0; i < 5; ++i) mvh5<NCOL, MT, MINB, UB><<<grid, 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvh5<NCOL, MT, MINB, UB><<<grid, 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

// ---- lane-owns-a-block variant: quad lanes t = 0..3 take 4 consecutive blocks of the same rows ----
// k permutation: MMA u (u = 0..7) takes, from lane t, elements 4u..4u+3 of lane t's block (b0 + t); the activation
// fragment follows the same permutation. Each lane decodes each of its 2 blocks (rows g, g+8) exactly once.
template <int NCOL>
__global__ void __launch_bounds__(256) mvh6(const uint4 * w, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                                             uint32_t cblo_u, uint32_t cbhi_u) {
    __shared__ float red[8][32][4];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, g = lane / 4, t = lane % 4;
    const int row0 = blockIdx.x * 16, nblk = k / 32;
    const uint32_t NEG1152 = 0xE480E480u;
    const int ra = min(row0 + g, rows - 1), rb = min(row0 + g + 8, rows - 1);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b0 = warp * 4; b0 < nblk; b0 += 32) {
        const int b = b0 + t;
        const bool ok = b < nblk;
        const uint4 qa = ok ? __ldg(&w[int64_t(ra) * nblk + b]) : make_uint4(0, 0, 0, 0);
        const uint4 qb = ok ? __ldg(&w[int64_t(rb) * nblk + b]) : make_uint4(0, 0, 0, 0);
        uint4 x4[4] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (g < NCOL && ok) {
            const uint4 * xp = reinterpret_cast<const uint4 *>(xh + int64_t(g) * k + int64_t(b) * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) x4[i] = __ldg(&xp[i]);
        }
        const uint32_t * xw = reinterpret_cast<const uint32_t *>(x4);  // 16 words = 32 halves
        // per-row scale half2 (s, s) and offsets (-1152 s) for the 4 groups
        uint32_t sa[4], oa[4], sb[4], ob[4];
#pragma unroll
        for (int gi = 0; gi < 4; ++gi) {
            const uint32_t ba = (qa.x >> (8 * gi)) & 255u, bb = (qb.x >> (8 * gi)) & 255u;
            const uint32_t ha = ba ? (ba << 5) + 0x1800u : 0u, hb = bb ? (bb << 5) + 0x1800u : 0u;
            sa[gi] = ha | (ha << 16);
            sb[gi] = hb | (hb << 16);
            oa[gi] = hmul2(sa[gi], NEG1152);
            ob[gi] = hmul2(sb[gi], NEG1152);
        }
        const uint32_t sla = ((qa.y >> 3) & 0x1111u) | ((qa.z >> 2) & 0x2222u) | ((qa.w >> 1) & 0x4444u);
        const uint32_t slb = ((qb.y >> 3) & 0x1111u) | ((qb.z >> 2) & 0x2222u) | ((qb.w >> 1) & 0x4444u);
        const uint32_t sla2 = ((qa.y >> 19) & 0x1111u) | ((qa.z >> 18) & 0x2222u) | ((qa.w >> 17) & 0x4444u);
        const uint32_t slb2 = ((qb.y >> 19) & 0x1111u) | ((qb.z >> 18) & 0x2222u) | ((qb.w >> 17) & 0x4444u);
        const uint32_t sela[8] = {qa.y, qa.y >> 16, qa.z, qa.z >> 16, qa.w, qa.w >> 16, sla, sla2};
        const uint32_t selb[8] = {qb.y, qb.y >> 16, qb.z, qb.z >> 16, qb.w, qb.w >> 16, slb, slb2};
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t ua = __byte_perm(cblo_u, cbhi_u, sela[u] & 0x7777u);
            const uint32_t ub = __byte_perm(cblo_u, cbhi_u, selb[u] & 0x7777u);
            const int gi = u / 2;
            mma16816h(c, hfma2(__byte_perm(ua, 0x64646464u, 0x5140), sa[gi], oa[gi]),
                      hfma2(__byte_perm(ub, 0x64646464u, 0x5140), sb[gi], ob[gi]),
                      hfma2(__byte_perm(ua, 0x64646464u, 0x5342), sa[gi], oa[gi]),
                      hfma2(__byte_perm(ub, 0x64646464u, 0x5342), sb[gi], ob[gi]), xw[2 * u], xw[2 * u + 1]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) red[warp][lane][i] = c[i];
    __syncthreads();
    if (warp == 0) {
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        for (int w2 = 0; w2 < 8; ++w2)
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] += red[w2][lane][i];
        const int n0 = 2 * t, n1 = 2 * t + 1, r = row0 + g;
        const float inv = 1.f / CB_Q;
        if (r < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + r] = __float2bfloat16_rn(v[0] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + r] = __float2bfloat16_rn(v[1] * inv);
        }
        if (r + 8 < rows) {
            if (n0 < NCOL) y[int64_t(n0) * rows + r + 8] = __float2bfloat16_rn(v[2] * inv);
            if (n1 < NCOL) y[int64_t(n1) * rows + r + 8] = __float2bfloat16_rn(v[3] * inv);
        }
    }
}

template <int NCOL>
void runh6(const char * label, const std::vector<Shape> & shapes, const uint4 * w, const __half * xh, __nv_bfloat16 * y,
           uint32_t lo, uint32_t hi) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::printf("%-22s", label);
    double tot_bytes = 0, tot_ms = 0;
    for (const auto & s : shapes) {
        const int grid = (s.rows + 15) / 16;
        for (int i = 0; i < 5; ++i) mvh6<NCOL><<<grid, 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        const int iters = 200;
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) mvh6<NCOL><<<grid, 256>>>(w, xh, y, s.rows, s.k, lo, hi);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double bytes = double(s.rows) * s.k / 32 * 16;
        std::printf("  %s %6.1f", s.name, bytes * iters / (ms * 1e-3) / 1e9);
        tot_bytes += bytes * s.count;
        tot_ms += ms / iters * s.count;
    }
    std::printf("   | mix %6.1f GB/s  %.3f ms/token\n", tot_bytes / (tot_ms * 1e-3) / 1e9, tot_ms);
}

} // namespace

int main(int argc, char ** argv) {
    // 9B shapes (--27b: 5120 hidden, 17408 ffn): ffn gate/up, ffn down, qkv, out/z/o, small k/v, alpha/beta
    // Engine shapes after row concatenation, weighted by count per token. 9B: 32 layers (24 GDN + 8 attention).
    const bool big = argc > 1 && std::string(argv[1]) == "--27b";
    const std::vector<Shape> shapes = big
        ? std::vector<Shape>{{"gu", 34816, 5120, 64}, {"down", 5120, 17408, 64}, {"gdn_in", 16480, 5120, 48},
                             {"gdn_out", 5120, 6144, 48}, {"att_in", 14336, 5120, 16}, {"att_out", 5120, 6144, 16}}
        : std::vector<Shape>{{"gu", 24576, 4096, 32}, {"down", 4096, 12288, 32}, {"gdn_in", 12352, 4096, 24},
                             {"gdn_out", 4096, 4096, 24}, {"att_in", 10240, 4096, 8}, {"att_out", 4096, 4096, 8}};
    size_t maxb = 0;
    for (const auto & s : shapes) maxb = std::max(maxb, size_t(s.rows) * s.k / 32 * 16);
    std::vector<uint8_t> hw(maxb);
    std::mt19937 g(1);
    for (auto & v : hw) v = uint8_t(g());
    std::vector<float> hx(17408);
    std::normal_distribution<float> nd;
    for (auto & v : hx) v = nd(g);
    uint4 * w;
    float * xr;
    __nv_bfloat16 * y;
    cudaMalloc(&w, maxb);
    cudaMalloc(&xr, hx.size() * 4);
    cudaMalloc(&y, 34816 * 2 * 8);
    cudaMemcpy(w, hw.data(), maxb, cudaMemcpyHostToDevice);
    cudaMemcpy(xr, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    // int8 activation: per-32-block max-abs scale
    std::vector<int8_t> hq(hx.size());
    std::vector<float> hs(hx.size() / 32);
    for (size_t b = 0; b < hs.size(); ++b) {
        float m = 0;
        for (int j = 0; j < 32; ++j) m = std::max(m, std::fabs(hx[b * 32 + j]));
        hs[b] = m / 127.f;
        for (int j = 0; j < 32; ++j) hq[b * 32 + j] = int8_t(std::lrint(m ? hx[b * 32 + j] / hs[b] : 0));
    }
    std::vector<uint8_t> hw8(maxb);
    for (size_t b = 0; b < maxb / 16; ++b) repack(&hw[b * 16], &hw8[b * 16]);
    const float cbf[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f, 0.230106f, 0.725222f, 1.277503f, 1.988943f};
    uint32_t cblo = 0, cbhi = 0;
    for (int i = 0; i < 4; ++i) {
        cblo |= uint32_t(uint8_t(int8_t(std::lrint(cbf[i] * 63.5f)))) << (8 * i);
        cbhi |= uint32_t(uint8_t(int8_t(std::lrint(cbf[i + 4] * 63.5f)))) << (8 * i);
    }
    uint4 * w8;
    int4 * dq;
    float * ds;
    cudaMalloc(&w8, maxb);
    cudaMalloc(&dq, hq.size());
    cudaMalloc(&ds, hs.size() * 4);
    cudaMemcpy(w8, hw8.data(), maxb, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    { // numerics: int8 path vs float path on the "up" shape
        const int R = 12288, K = 4096;
        __nv_bfloat16 * y2;
        cudaMalloc(&y2, R * 2);
        mv<4, 1, 0><<<R / 32, 256>>>(w, xr, y, R, K);
        mv8<4, 1><<<R / 32, 256>>>(w8, dq, ds, y2, R, K, cblo, cbhi);
        std::vector<__nv_bfloat16> a(R), b(R);
        cudaMemcpy(a.data(), y, R * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), y2, R * 2, cudaMemcpyDeviceToHost);
        double se = 0, sr = 0;
        for (int i = 0; i < R; ++i) {
            const double fa = __bfloat162float(a[i]), fb = __bfloat162float(b[i]);
            se += (fa - fb) * (fa - fb);
            sr += fa * fa;
        }
        std::printf("int8 path vs float path: rel_rms %.3e\n", std::sqrt(se / sr));
    }
    std::printf("effective weight GB/s (3090 peak ~936)\n");
    run<4, 1, 0>("RPW4 UB1 cb-smem", shapes, w, xr, y);
    run8<1, 1>("i8 RPW1 UB1", shapes, w8, dq, ds, y, cblo, cbhi);
    run8<2, 1>("i8 RPW2 UB1", shapes, w8, dq, ds, y, cblo, cbhi);
    run8<4, 1>("i8 RPW4 UB1", shapes, w8, dq, ds, y, cblo, cbhi);
    run8<1, 2>("i8 RPW1 UB2", shapes, w8, dq, ds, y, cblo, cbhi);
    run8<2, 2>("i8 RPW2 UB2", shapes, w8, dq, ds, y, cblo, cbhi);
    run8<1, 4>("i8 RPW1 UB4", shapes, w8, dq, ds, y, cblo, cbhi);
    run8<4, 2>("i8 RPW4 UB2", shapes, w8, dq, ds, y, cblo, cbhi);
    { // tensor-core multi-column: 8 columns of int8 activations over the widest k
        const int KMAX = 17408, NC = 8;
        std::vector<int8_t> q8(size_t(NC) * KMAX);
        std::vector<float> s8(size_t(NC) * KMAX / 32);
        for (int n = 0; n < NC; ++n)
            for (int b = 0; b < KMAX / 32; ++b) {
                float m = 0;
                float v[32];
                for (int j = 0; j < 32; ++j) m = std::max(m, std::fabs(v[j] = nd(g)));
                s8[size_t(n) * KMAX / 32 + b] = m / 127.f;
                for (int j = 0; j < 32; ++j) q8[size_t(n) * KMAX + b * 32 + j] = int8_t(std::lrint(v[j] / (m / 127.f)));
            }
        int8_t * dq8;
        float * ds8;
        __nv_bfloat16 *yt, *yr;
        cudaMalloc(&dq8, q8.size());
        cudaMalloc(&ds8, s8.size() * 4);
        cudaMalloc(&yt, size_t(NC) * 34816 * 2);
        cudaMalloc(&yr, size_t(NC) * 34816 * 2);
        cudaMemcpy(dq8, q8.data(), q8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ds8, s8.data(), s8.size() * 4, cudaMemcpyHostToDevice);
        // correctness on "up" (12288 x 4096): xs/xq laid out per column with row length k = 4096 for this check
        const int R = 12288, K = 4096;
        std::vector<int8_t> qk(size_t(NC) * K);
        std::vector<float> sk(size_t(NC) * K / 32);
        for (int n = 0; n < NC; ++n) {
            std::copy(q8.begin() + size_t(n) * KMAX, q8.begin() + size_t(n) * KMAX + K, qk.begin() + size_t(n) * K);
            std::copy(s8.begin() + size_t(n) * KMAX / 32, s8.begin() + size_t(n) * KMAX / 32 + K / 32, sk.begin() + size_t(n) * K / 32);
        }
        int8_t * dqk;
        float * dsk;
        cudaMalloc(&dqk, qk.size());
        cudaMalloc(&dsk, sk.size() * 4);
        cudaMemcpy(dqk, qk.data(), qk.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dsk, sk.data(), sk.size() * 4, cudaMemcpyHostToDevice);
        mvt<NC><<<R / 128, 256>>>(w8, dqk, dsk, yt, R, K, cblo, cbhi);
        for (int n = 0; n < NC; ++n)
            mv8<2, 1><<<R / 16, 256>>>(w8, reinterpret_cast<const int4 *>(dqk + size_t(n) * K), dsk + size_t(n) * K / 32, yr + size_t(n) * R, R, K, cblo, cbhi);
        std::vector<__nv_bfloat16> a(size_t(NC) * R), bb(size_t(NC) * R);
        cudaMemcpy(a.data(), yt, a.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(bb.data(), yr, bb.size() * 2, cudaMemcpyDeviceToHost);
        double se = 0, sr = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double fa = __bfloat162float(a[i]), fb = __bfloat162float(bb[i]);
            se += (fa - fb) * (fa - fb);
            sr += fb * fb;
        }
        std::printf("tensor-core N=8 vs 8 x DP4A N=1: rel_rms %.3e\n", std::sqrt(se / sr));
        std::printf("-- tensor-core multi-column (GB/s of weights; one pass serves all N columns)\n");
        mvt2<NC, 4><<<R / 16, 256>>>(w8, dqk, dsk, yt, R, K, cblo, cbhi);
        cudaMemcpy(a.data(), yt, a.size() * 2, cudaMemcpyDeviceToHost);
        se = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double fa = __bfloat162float(a[i]), fb = __bfloat162float(bb[i]);
            se += (fa - fb) * (fa - fb);
        }
        std::printf("split-K N=8 vs 8 x DP4A N=1: rel_rms %.3e\n", std::sqrt(se / sr));
        mvt3<NC><<<R / 16, 128>>>(w8, dqk, dsk, yt, R, K, cblo, cbhi);
        cudaMemcpy(a.data(), yt, a.size() * 2, cudaMemcpyDeviceToHost);
        se = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double fa = __bfloat162float(a[i]), fb = __bfloat162float(bb[i]);
            se += (fa - fb) * (fa - fb);
        }
        std::printf("staged N=8 vs 8 x DP4A N=1: rel_rms %.3e\n", std::sqrt(se / sr));
        {
            std::vector<__half> xhh(size_t(NC) * KMAX);
            for (int n = 0; n < NC; ++n)
                for (int i = 0; i < KMAX; ++i) xhh[size_t(n) * KMAX + i] = __float2half_rn(n == 0 && i < 17408 ? hx[i] : nd(g));
            __half * dxh;
            cudaMalloc(&dxh, xhh.size() * 2);
            cudaMemcpy(dxh, xhh.data(), xhh.size() * 2, cudaMemcpyHostToDevice);
            // reference: float kernel (original layout) on column 0 with k = KMAX-strided data -> use K=4096 check
            std::vector<__half> x0(size_t(NC) * 4096);
            for (int n = 0; n < NC; ++n) for (int i = 0; i < 4096; ++i) x0[size_t(n) * 4096 + i] = xhh[size_t(n) * KMAX + i];
            __half * dx0;
            cudaMalloc(&dx0, x0.size() * 2);
            cudaMemcpy(dx0, x0.data(), x0.size() * 2, cudaMemcpyHostToDevice);
            mvh<NC, 2><<<12288 / 16, 256>>>(w8, dx0, yt, 12288, 4096);
            mv<4, 1, 0><<<12288 / 32, 256>>>(w, xr, yr, 12288, 4096);  // xr == hx (float) for column 0
            std::vector<__nv_bfloat16> ha(12288), hb(12288);
            cudaMemcpy(ha.data(), yt, 12288 * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), yr, 12288 * 2, cudaMemcpyDeviceToHost);
            double e2 = 0, r2 = 0;
            for (int i = 0; i < 12288; ++i) {
                const double fa = __bfloat162float(ha[i]), fb = __bfloat162float(hb[i]);
                e2 += (fa - fb) * (fa - fb);
                r2 += fb * fb;
            }
            std::printf("fp16 tensor-core col0 vs float kernel: rel_rms %.3e\n", std::sqrt(e2 / r2));
            mvh2<NC, 2><<<12288 / 32, 256>>>(w8, dx0, yt, 12288, 4096);
            cudaMemcpy(ha.data(), yt, 12288 * 2, cudaMemcpyDeviceToHost);
            e2 = 0;
            for (int i = 0; i < 12288; ++i) {
                const double fa = __bfloat162float(ha[i]), fb = __bfloat162float(hb[i]);
                e2 += (fa - fb) * (fa - fb);
            }
            std::printf("fp16 MT col0 vs float kernel: rel_rms %.3e\n", std::sqrt(e2 / r2));
            uint32_t ulo = 0, uhi = 0;  // unsigned (+128) int8 codebook
            for (int i = 0; i < 4; ++i) {
                ulo |= uint32_t(uint8_t(std::lrint(cbf[i] * 63.5f) + 128)) << (8 * i);
                uhi |= uint32_t(uint8_t(std::lrint(cbf[i + 4] * 63.5f) + 128)) << (8 * i);
            }
            mvh3<NC, 2><<<12288 / 32, 256>>>(w8, dx0, yt, 12288, 4096, ulo, uhi);
            cudaMemcpy(ha.data(), yt, 12288 * 2, cudaMemcpyDeviceToHost);
            e2 = 0;
            for (int i = 0; i < 12288; ++i) {
                const double fa = __bfloat162float(ha[i]), fb = __bfloat162float(hb[i]);
                e2 += (fa - fb) * (fa - fb);
            }
            std::printf("fp16 regdecode col0 vs float kernel: rel_rms %.3e\n", std::sqrt(e2 / r2));
            mvh5<NC, 2><<<12288 / 32, 256>>>(w8, dx0, yt, 12288, 4096, ulo, uhi);
            cudaMemcpy(ha.data(), yt, 12288 * 2, cudaMemcpyDeviceToHost);
            e2 = 0;
            for (int i = 0; i < 12288; ++i) {
                const double fa = __bfloat162float(ha[i]), fb = __bfloat162float(hb[i]);
                e2 += (fa - fb) * (fa - fb);
            }
            std::printf("fp16 intdecode col0 vs float kernel: rel_rms %.3e\n", std::sqrt(e2 / r2));
            mvh6<NC><<<12288 / 16, 256>>>(w8, dx0, yt, 12288, 4096, ulo, uhi);
            cudaMemcpy(ha.data(), yt, 12288 * 2, cudaMemcpyDeviceToHost);
            e2 = 0;
            for (int i = 0; i < 12288; ++i) {
                const double fa = __bfloat162float(ha[i]), fb = __bfloat162float(hb[i]);
                e2 += (fa - fb) * (fa - fb);
            }
            std::printf("lane-block col0 vs float kernel: rel_rms %.3e\n", std::sqrt(e2 / r2));
            runh6<1>("hmma6 N1", shapes, w8, dxh, yt, ulo, uhi);
            runh6<5>("hmma6 N5", shapes, w8, dxh, yt, ulo, uhi);
            runh6<8>("hmma6 N8", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 1, 1>("hmma5 N8 base", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 2, 1>("hmma5 N8 minb2", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 3, 1>("hmma5 N8 minb3", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 4, 1>("hmma5 N8 minb4", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 1, 2>("hmma5 N8 UB2", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 3, 2>("hmma5 N8 minb3 UB2", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1, 1, 4>("hmma5 N8 UB4", shapes, w8, dxh, yt, ulo, uhi);
            runh5<1, 2>("hmma5 N=1 MT2", shapes, w8, dxh, yt, ulo, uhi);
            runh5<5, 2>("hmma5 N=5 MT2", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 2>("hmma5 N=8 MT2", shapes, w8, dxh, yt, ulo, uhi);
            runh5<5, 1>("hmma5 N=5 MT1", shapes, w8, dxh, yt, ulo, uhi);
            runh5<8, 1>("hmma5 N=8 MT1", shapes, w8, dxh, yt, ulo, uhi);
            float * y32;
            cudaMalloc(&y32, size_t(NC) * 34816 * 4);
            runh4<5, 2>("splitKx N=5 MT2", shapes, w8, dxh, y32, ulo, uhi);
            runh4<5, 1>("splitKx N=5 MT1", shapes, w8, dxh, y32, ulo, uhi);
            runh4<8, 1>("splitKx N=8 MT1", shapes, w8, dxh, y32, ulo, uhi);
            runh3<1, 2>("hmma3 N=1 MT2", shapes, w8, dxh, yt, ulo, uhi);
            runh3<5, 2>("hmma3 N=5 MT2", shapes, w8, dxh, yt, ulo, uhi);
            runh3<8, 2>("hmma3 N=8 MT2", shapes, w8, dxh, yt, ulo, uhi);
            runh3<8, 1>("hmma3 N=8 MT1", shapes, w8, dxh, yt, ulo, uhi);
            runh2<1, 2>("hmma2 N=1 MT2", shapes, w8, dxh, yt);
            runh2<8, 2>("hmma2 N=8 MT2", shapes, w8, dxh, yt);
            runh2<8, 4>("hmma2 N=8 MT4", shapes, w8, dxh, yt);
            runh2<5, 4>("hmma2 N=5 MT4", shapes, w8, dxh, yt);
            runh<1, 2>("hmma N=1 UB2", shapes, w8, dxh, yt);
            runh<8, 2>("hmma N=8 UB2", shapes, w8, dxh, yt);
            runh<8, 4>("hmma N=8 UB4", shapes, w8, dxh, yt);
            runh<8, 1>("hmma N=8 UB1", shapes, w8, dxh, yt);
        }
        runt3<1>("staged N=1", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt3<4>("staged N=4", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt3<8>("staged N=8", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<8, 4, 1>("ABL no-scale N=8", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<8, 4, 2>("ABL no-mma N=8", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<1, 2>("splitK N=1 UB2", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<1, 4>("splitK N=1 UB4", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<4, 4>("splitK N=4 UB4", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<8, 2>("splitK N=8 UB2", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<8, 4>("splitK N=8 UB4", shapes, w8, dq8, ds8, yt, cblo, cbhi);
        runt2<8, 8>("splitK N=8 UB8", shapes, w8, dq8, ds8, yt, cblo, cbhi);
    }
    return 0;
}

// FNT encoder kernel (tools/convert/qwen4exp/fnt_quant.py): tail-biting Viterbi over the bitshift trellis of the FNT
// expert format (docs/reference/formats.md "FNT"; engine side src/formats/weight/fnt.cu): code c2 with an L-bit window
// (L = 16 in the format), KB stream bits per weight pair, blocks of T weights. Convention (forward): pair p reads the
// window w_p = bits [p*KB, p*KB + L) of the block stream (bit index mod T/2 * KB); state after pair p: w_p >> KB; from
// next state sn and chunk lo: w = (sn << KB) | lo, predecessor w & (S - 1). Output per pair: its KB-bit chunk.
// FP32 path metrics, branch metric sum of wt * (x - c)^2 (c the FP16 code pair). One CTA (256 threads) per block.
// Built on first use by fnt_quant.py with nvcc into ~/.cache/tqllm (loaded with ctypes; inputs are device pointers).
#include <cuda_fp16.h>
#include <cstdint>

#define HA 0x9E3779B1u
#define HB 0x7F4A7C15u
#define CC1 0x2C1B3C6Du
#define CD1 0x9E3779B9u
#define CC2 0x297A2D39u
#define CD2 0x7F4A7C15u

template <int L>
__device__ __forceinline__ float2 code_c2(uint32_t w) {
    const uint32_t x = (w * HA + HB) << (32 - L);
    const uint32_t y = __funnelshift_l(x, x, 16);
    const uint32_t z = y * CC1 + CD1, z2 = z * CC2 + CD2;
    const uint32_t h1 = (z & 0x8fff8fffu) ^ 0x3b603b60u, h2 = (z2 & 0x8fff8fffu) ^ 0x3b603b60u;
    const __half2 c = __hadd2(*reinterpret_cast<const __half2 *>(&h1), *reinterpret_cast<const __half2 *>(&h2));
    return __half22float2(c);
}

template <int L, int KB, int T>
__global__ void __launch_bounds__(256) vit_k(const float * __restrict__ X, const float * __restrict__ Wt, int nseq,
                                             uint8_t * out, float * cost_out, uint8_t * bp_all, int seq0, int A, int B1) {
    constexpr int S = 1 << (L - KB), NB = 1 << KB, P = T / 2, SPT = S / 256;
    static_assert(S >= 256, "need >= 256 states");
    __shared__ float pmA[S + S / 32], pmB[S + S / 32];  // padded: index i at i + i / 32 (bank spread)
    __shared__ float xs[T], ws[T];
    __shared__ int sstate;
    const int tid = threadIdx.x;
    const int seq = seq0 + blockIdx.x;
    const bool valid = seq < nseq;
    uint8_t * bp = bp_all + (size_t)blockIdx.x * P * S;
    for (int i = tid; i < T; i += 256) xs[i] = valid ? X[(size_t)seq * T + i] : 0.f;
    for (int i = tid; i < T; i += 256) ws[i] = (valid && Wt) ? Wt[(size_t)seq * T + i] : 1.f;
    __syncthreads();
    // pm holds metrics of the state before the step; returns with the final metrics in pmA
    auto run = [&](int roll, int start, int nsteps) -> void {
        float * pm = pmA;
        float * pn = pmB;
        for (int s = tid; s < S; s += 256) pm[s + (s >> 5)] = (start < 0 || s == start) ? 0.f : 1e30f;
        for (int tau = 0; tau < nsteps; ++tau) {
            const int j = (tau + roll) % P;
            const float x0 = xs[2 * j], x1 = xs[2 * j + 1], w0 = ws[2 * j], w1 = ws[2 * j + 1];
            __syncthreads();
#pragma unroll
            for (int q = 0; q < SPT; ++q) {
                const int sn = q * 256 + tid;
                const uint32_t wb = (uint32_t)sn << KB;
                float best = 3e38f;
                int bl = 0;
#pragma unroll 8
                for (int lo = 0; lo < NB; ++lo) {
                    const uint32_t w = wb | (uint32_t)lo;
                    const float2 c = code_c2<L>(w);
                    const float d0 = x0 - c.x, d1 = x1 - c.y;
                    const uint32_t pi = w & (S - 1);
                    const float m = pm[pi + (pi >> 5)] + w0 * d0 * d0 + w1 * d1 * d1;
                    if (m < best) { best = m; bl = lo; }
                }
                pn[sn + (sn >> 5)] = best;
                bp[(size_t)tau * S + sn] = (uint8_t)bl;
            }
            float * t = pm; pm = pn; pn = t;
        }
        __syncthreads();
        if (pm != pmA)
            for (int s = tid; s < S + S / 32; s += 256) pmA[s] = pm[s];
        __syncthreads();
    };
    auto prev_of = [&](int tau, int sn) -> int { return (int)((((uint32_t)sn << KB) | bp[(size_t)tau * S + sn]) & (S - 1)); };
    // pass 1: pairs P-A .. P-1, 0 .. B1-1, free start; the state after pair P-1 is the tail-biting state
    run(P - A, -1, A + B1);
    if (tid == 0) {
        int bs = 0;
        float bc = pmA[0];
        for (int s = 1; s < S; ++s) if (pmA[s + (s >> 5)] < bc) { bc = pmA[s + (s >> 5)]; bs = s; }
        for (int tau = A + B1 - 1; tau >= A; --tau) bs = prev_of(tau, bs);
        sstate = bs;
    }
    __syncthreads();
    const int o = sstate;
    __syncthreads();
    run(0, o, P);
    if (tid == 0 && valid) {
        int s = o;
        cost_out[seq] = pmA[o + (o >> 5)];
        for (int tau = P - 1; tau >= 0; --tau) {
            out[(size_t)seq * P + tau] = bp[(size_t)tau * S + s];
            s = prev_of(tau, s);
        }
        if (s != o || !(pmA[o + (o >> 5)] < 1e29f)) cost_out[seq] = -3e38f;
    }
}

template <int L, int KB, int T>
static int launch(const float * X, const float * Wt, int nseq, uint8_t * out, float * cost, int A, int B1) {
    constexpr int S = 1 << (L - KB), P = T / 2;
    if (A < 1 || A > P / 2 || B1 < 1 || B1 > P / 2) return 4;
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    const int wave = nsm * 4;
    uint8_t * bp = nullptr;
    if (cudaMalloc(&bp, (size_t)wave * P * S) != cudaSuccess) return 2;
    int err = 0;
    for (int s0 = 0; s0 < nseq; s0 += wave) {
        const int ctas = nseq - s0 < wave ? nseq - s0 : wave;
        vit_k<L, KB, T><<<ctas, 256>>>(X, Wt, nseq, out, cost, bp, s0, A, B1);
        if ((err = (int)cudaGetLastError())) break;
    }
    if (!err) err = (int)cudaDeviceSynchronize();
    cudaFree(bp);
    return err;
}

#define CASE(l, kb, t) if (L == l && KB == kb && T == t) return launch<l, kb, t>(X, Wt, nseq, out, cost, A, B1);
extern "C" int fnt_viterbi(const float * X, const float * Wt, int nseq, int L, int KB, int T, uint8_t * out, float * cost,
                           int A, int B1) {
    CASE(16, 4, 128) CASE(16, 5, 128) CASE(16, 6, 128) CASE(16, 7, 128) CASE(16, 4, 256) CASE(16, 5, 256) CASE(16, 6, 256)
    CASE(14, 4, 128) CASE(14, 5, 128) CASE(14, 6, 128) CASE(14, 5, 256)
    CASE(18, 6, 128) CASE(18, 6, 256)
    CASE(12, 4, 128) CASE(12, 4, 256)
    return 3;
}

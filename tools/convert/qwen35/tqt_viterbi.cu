// TQT encoder (tools/convert/qwen35/tqt.py): tail-biting Viterbi over the TQT trellis (docs/reference/formats.md "TQT",
// src/formats/weight/tqt.cu): 1024 states (the 10 stream bits two pairs share), 256 branches (the next byte), code c2.
// One CTA (256 threads) per 128-weight sequence. Branch metric |x - c|^2 in FP16 (the chosen path can differ from an
// exact search in near-ties), path metrics FP32, packed (metric | predecessor) integer argmin. Tail-biting: a short
// first pass over A steps before and B after the wrap point finds the shared start / end state, the second pass forces it.
// Built on first use by tqt.py with nvcc into ~/.cache/tqllm (loaded with ctypes; inputs are device pointers).
#include <cuda_fp16.h>
#include <cstdint>

#define HA 0x9E3779B1u
#define HB 0x7F4A7C15u
#define CC1 0x2C1B3C6Du
#define CD1 0x9E3779B9u
#define CC2 0x297A2D39u
#define CD2 0x7F4A7C15u

// Thread (m, bg) owns the successors m << 8 | B for its M bytes B; predecessors m | hi << (L - 16). Window of the branch
// from predecessor s with byte B: w = s << 8 | B. Pass taus [t0, t1); step tau processes pair j = (P - 1 - tau + roll) % P
// (the stream is decoded forwards, the trellis is searched backwards: the new byte enters at the low end).
template <int L>
__global__ void __launch_bounds__(256) viterbi2f_k(const float * __restrict__ X, int nseq, int T, float cmul, uint8_t * out,
                                                   float * cost_out, uint8_t * bp_all, int seq0, int A, int B1,
                                                   const float * __restrict__ Wt) {
    constexpr int S = 1 << (L - 8), M = 1 << (L - 16), G = 256 / M;
    const int P = T / 2;
    extern __shared__ float sm[];
    float * pmA = sm;
    float * pmB = sm + S;
    float * xs = sm + 2 * S;
    float * ws = sm + 2 * S + T;  // per-weight metric weights (all 1 without Wt)
    __shared__ int sstate;
    const int tid = threadIdx.x, m = tid / G, bg = tid % G;
    const int seq = seq0 + blockIdx.x;
    const bool valid = seq < nseq;
    uint8_t * bp = bp_all + (size_t)blockIdx.x * P * S;
    for (int i = tid; i < T; i += 256) xs[i] = valid ? X[(size_t)seq * T + i] * (1.f / cmul) : 0.f;  // raw code units
    for (int i = tid; i < T; i += 256) ws[i] = (valid && Wt) ? Wt[(size_t)seq * T + i] : 1.f;
    __syncthreads();
    // window w = hi << (L - 8 + 8 - ...) : w = (s << 8) | B with s = m | hi << (L - 16) -> w = hi << (L - 8) | m << 8 | B
    uint32_t xb[M];
#pragma unroll
    for (int q = 0; q < M; ++q) xb[q] = ((uint32_t)(m << 8) | (uint32_t)(bg * M + q)) * HA + HB;
    const uint32_t hstep = (1u << (L - 8)) * HA;
    // tau range [t0, t1) of a pass; bp index tau - t0
    auto run = [&](int roll, int start, int t0, int t1) -> void {
        float * pm = pmA;
        float * pn = pmB;
        for (int s = tid; s < S; s += 256) pm[s] = (start < 0 || s == start) ? 0.f : 1e30f;
        for (int tau = t0; tau < t1; ++tau) {
            const int j = (P - 1 - tau + roll) % P;
            const __half2 xv = __floats2half2_rn(xs[2 * j], xs[2 * j + 1]);
            const __half2 av = __floats2half2_rn(ws[2 * j], ws[2 * j + 1]);
            __syncthreads();
            uint32_t best[M];
#pragma unroll
            for (int q = 0; q < M; ++q) best[q] = 0xFFFFFFFFu;
            uint32_t hterm = 0;
            for (int hi = 0; hi < 256; ++hi, hterm += hstep) {
                const float p = pm[(uint32_t)m | ((uint32_t)hi << (L - 16))];
#pragma unroll
                for (int q = 0; q < M; ++q) {
                    const uint32_t x = (xb[q] + hterm) << (32 - L);
                    const uint32_t y = __funnelshift_l(x, x, 16);
                    const uint32_t z = y * CC1 + CD1, z2 = z * CC2 + CD2;
                    const uint32_t h1 = (z & 0x8fff8fffu) ^ 0x3b603b60u, h2 = (z2 & 0x8fff8fffu) ^ 0x3b603b60u;
                    const __half2 c = __hadd2(*reinterpret_cast<const __half2 *>(&h1), *reinterpret_cast<const __half2 *>(&h2));
                    const __half2 d = __hsub2(xv, c);
                    const __half2 dd = __hmul2(__hmul2(d, d), av);
                    const float cc = p + __half2float(__hadd(__low2half(dd), __high2half(dd)));
                    const uint32_t key = (__float_as_uint(cc) & 0xFFFFFF00u) | (uint32_t)hi;
                    best[q] = min(best[q], key);
                }
            }
#pragma unroll
            for (int q = 0; q < M; ++q) {
                const int sn = (m << 8) | (bg * M + q);
                pn[sn] = __uint_as_float(best[q] & 0xFFFFFF00u);
                bp[(size_t)(tau - t0) * S + sn] = (uint8_t)(best[q] & 255u);
            }
            float * tmp = pm; pm = pn; pn = tmp;
        }
        __syncthreads();
        if (pm != pmA)
            for (int s = tid; s < S; s += 256) pmA[s] = pm[s];
        __syncthreads();
    };
    auto prev_of = [&](int ti, int sn) -> int {
        const int hi = bp[(size_t)ti * S + sn];
        return (sn >> 8) | (hi << (L - 16));
    };
    // pass 1 (rolled by P/2: j = 0 at tau P/2 - 1): only taus [P/2 - A, P/2 + B1)
    const int t0 = P / 2 - A, t1 = P / 2 + B1;
    run(P / 2, -1, t0, t1);
    if (tid == 0) {
        int bs = 0;
        float bc = pmA[0];
        for (int s = 1; s < S; ++s) if (pmA[s] < bc) { bc = pmA[s]; bs = s; }
        for (int tau = t1 - 1; tau > P / 2 - 1; --tau) bs = prev_of(tau - t0, bs);
        sstate = bs;
    }
    __syncthreads();
    const int o = sstate;
    __syncthreads();
    run(0, o, 0, P);
    if (tid == 0 && valid) {
        int s = o;
        cost_out[seq] = pmA[o] * cmul * cmul;
        for (int tau = P - 1; tau >= 0; --tau) {
            out[(size_t)seq * P + (P - 1 - tau)] = (uint8_t)(s & 255);
            s = prev_of(tau, s);
        }
        if (s != o) cost_out[seq] = -3e38f;
    }
}


// Wt (optional, [nseq][128]): weight of each squared error in the metric (the block's diag(H), mean 1)
extern "C" int tqt_viterbi(const float * X, int nseq, float cmul, uint8_t * out, float * cost, int A, int B1, const float * Wt) {
#ifndef TQT_L
#define TQT_L 18
#endif
    constexpr int L = TQT_L, T = 128, S = 1 << (L - 8), P = T / 2;
    if (A < 1 || A > P / 2 || B1 < 1 || B1 > P / 2) return 4;
    const size_t smem = (size_t)(2 * S + 2 * T) * sizeof(float);
    int dev = 0, nsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    const int wave = nsm * 4;
    uint8_t * bp = nullptr;
    if (cudaMalloc(&bp, (size_t)wave * P * S) != cudaSuccess) return 2;
    int err = 0;
    for (int s0 = 0; s0 < nseq; s0 += wave) {
        const int ctas = nseq - s0 < wave ? nseq - s0 : wave;
        viterbi2f_k<L><<<ctas, 256, smem>>>(X, nseq, T, cmul, out, cost, bp, s0, A, B1, Wt);
        if ((err = (int)cudaGetLastError())) break;
    }
    if (!err) err = (int)cudaDeviceSynchronize();
    cudaFree(bp);
    return err;
}

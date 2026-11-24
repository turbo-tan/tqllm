// TQT3 encoder (tools/convert/inkling2/tqt3.py): tail-biting Viterbi over the 3-bit trellis code "tqt3-c1-v1".
// A block of 128 weights = 64 pairs; the stream is 64 six-bit symbols s_0..s_63 (cyclic, 384 bits), pair p reads the 18-bit
// window w_p = s_p | s_{p+1} << 6 | s_{p+2} << 12 and decodes it with the hash of tqt-c2-v1 (L = 18).
// State before step p = (s_p, s_{p+1}) (12 bits), step p picks s_{p+2}; a state has 64 predecessors (the dropped symbol z)
// and 64 successors. One CTA (256 threads) per 128-weight sequence, thread (r, g) owns the successors r | c << 6 for the 16 c of group g.
// Tail-biting as TQT: a first pass over A steps before and B after the wrap point (free start) picks the shared state at the
// wrap, the second pass starts and ends in it. Branch metric |x - c|^2 in FP32 on the FP16 pair values.
// Built on first use by tqt2.py with nvcc (loaded with ctypes; inputs are device pointers).
#include <cuda_fp16.h>
#include <cstdint>

#define HA 0x9E3779B1u
#define HB 0x7F4A7C15u
#define CC1 0x2C1B3C6Du
#define CD1 0x9E3779B9u
#define CC2 0x297A2D39u
#define CD2 0x7F4A7C15u

__device__ __forceinline__ float2 code_pair(uint32_t w) {
    const uint32_t x = (w * HA + HB) << 14, y = (x << 16) | (x >> 16);
    const uint32_t z = y * CC1 + CD1, z2 = z * CC2 + CD2;
    const uint32_t h1 = (z & 0x8FFF8FFFu) ^ 0x3B603B60u, h2 = (z2 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const __half2 a = *reinterpret_cast<const __half2 *>(&h1), b = *reinterpret_cast<const __half2 *>(&h2);
    return __half22float2(__hadd2(a, b));
}

constexpr int S = 4096, P = 64, MAXSTEPS = 128;

// steps k = 0 .. nsteps - 1 process pair (p0 + k) % 64; start < 0: every state starts at 0, else only `start`
__device__ void run(const float * xs, float * pm, float * pn, uint8_t * bp, int p0, int nsteps, int start) {
    const int r = threadIdx.x & 63, g = threadIdx.x >> 6, r256 = threadIdx.x;
    for (int s = r256; s < S; s += 256) pm[s] = (start < 0 || s == start) ? 0.f : 1e30f;
    __syncthreads();
    for (int k = 0; k < nsteps; ++k) {
        const int p = (p0 + k) % P;
        const float x0 = xs[2 * p], x1 = xs[2 * p + 1];
        uint8_t * bpk = bp + size_t(k) * S;
#pragma unroll 2
        for (int ci = 0; ci < 16; ++ci) {
            const int c = g * 16 + ci;
            float best = 3e38f;
            int bz = 0;
            for (int z = 0; z < 64; ++z) {
                const float2 v = code_pair(uint32_t(z) | uint32_t(r) << 6 | uint32_t(c) << 12);
                const float d0 = x0 - v.x, d1 = x1 - v.y;
                const float m = pm[(r << 6) | z] + d0 * d0 + d1 * d1;
                if (m < best) { best = m; bz = z; }
            }
            pn[(c << 6) | r] = best;
            bpk[(c << 6) | r] = uint8_t(bz);
        }
        __syncthreads();
        for (int s = r256; s < S; s += 256) pm[s] = pn[s];  // pm <- pn
        __syncthreads();
    }
}

// state after step k (k = 1 .. nsteps) of the best path ending in `end`, reading the pointers back to step `upto`
__device__ int state_at(const uint8_t * bp, int nsteps, int end, int upto) {
    int s = end;
    for (int k = nsteps - 1; k >= upto; --k) {  // state after step k = s; its predecessor before step k
        const int z = bp[size_t(k) * S + s];
        s = ((s & 63) << 6) | z;
    }
    return s;  // the state before step `upto`
}

__global__ void __launch_bounds__(256) viterbi_k(const float * __restrict__ X, int nseq, uint8_t * out, float * cost_out, uint8_t * bp_all, int seq0,
                                                 int A, int B) {
    __shared__ float pm[S], pn[S], xs[128];
    __shared__ int sh_state;
    const int tid = threadIdx.x, seq = seq0 + blockIdx.x;
    if (seq >= nseq) return;
    uint8_t * bp = bp_all + size_t(blockIdx.x) * MAXSTEPS * S;
    for (int i = tid; i < 128; i += 256) xs[i] = X[size_t(seq) * 128 + i];
    __syncthreads();
    // pass 1 around the wrap point: steps P - A .. P - 1, 0 .. B - 1
    run(xs, pm, pn, bp, P - A, A + B, -1);
    if (tid == 0) {
        int best = 0;
        for (int s = 1; s < S; ++s) if (pm[s] < pm[best]) best = s;
        sh_state = state_at(bp, A + B, best, A);  // the state before step A = the state before pair 0
    }
    __syncthreads();
    const int s0 = sh_state;
    run(xs, pm, pn, bp, 0, P, s0);
    if (tid == 0) {
        float c = pm[s0];
        uint8_t * o = out + size_t(seq) * 48;
        int s = s0;
        uint32_t nib[P];
        for (int k = P - 1; k >= 0; --k) {  // state after step k = s
            const int z = bp[size_t(k) * S + s];
            const int prev = ((s & 63) << 6) | z;
            nib[k] = z;  // s_k, the symbol dropped at step k
            s = prev;
        }
        for (int i = 0; i < 48; ++i) o[i] = 0;
        for (int i = 0; i < P; ++i)  // symbol i at bit 6 i, little-endian bit order
            for (int b = 0; b < 6; ++b) o[(6 * i + b) >> 3] |= uint8_t(((nib[i] >> b) & 1u) << ((6 * i + b) & 7));
        cost_out[seq] = c < 1e29f ? c : -1e39f;
    }
}

extern "C" int tqt3_viterbi(const float * X, int nseq, uint8_t * out, float * cost_out, int A, int B) {
    const int chunk = 512;
    uint8_t * bp = nullptr;
    if (cudaMalloc(&bp, size_t(chunk) * MAXSTEPS * S) != cudaSuccess) return 1;
    for (int s0 = 0; s0 < nseq; s0 += chunk) {
        const int n = nseq - s0 < chunk ? nseq - s0 : chunk;
        viterbi_k<<<n, 256>>>(X, nseq, out, cost_out, bp, s0, A, B);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    cudaFree(bp);
    return e == cudaSuccess ? 0 : int(e);
}

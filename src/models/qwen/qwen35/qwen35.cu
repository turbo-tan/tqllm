#include <cstdlib>
#include "qwen35_kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace {

__device__ float block_sum(float v, float * sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = (blockDim.x + 31) / 32;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}

// RoPE scaling (YaRN / linear): per rotary pair a factor on the frequency (stored as factor - 1, so zero-initialised
// constant memory means no scaling for every caller) and one scale on cos / sin. Off: 1 + 0 and 1, bit-identical.
__constant__ float c_rope_dmul[256];
__constant__ float c_rope_mscale = 1.f;

__global__ void rms_f_k(const __nv_bfloat16 * x, int xs, const float * w, __nv_bfloat16 * y, int ys, int n, float eps) {
    __shared__ float sh[32];
    const __nv_bfloat16 * xr = x + int64_t(blockIdx.x) * xs;
    __nv_bfloat16 * yr = y + int64_t(blockIdx.x) * ys;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = __float2bfloat16_rn(__bfloat162float(xr[i]) * inv * w[i]);
}

__global__ void rope_partial_k(__nv_bfloat16 * x, int heads, int dim, int rot, int pos, float theta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, half = rot / 2;
    if (i >= heads * half) return;
    const int h = i / half, d = i % half;
    const float inv_freq = 1.f / powf(theta, float(2 * d) / float(rot)) * (1.f + c_rope_dmul[d]);
    const float ang = float(pos) * inv_freq;
    const float c = __bfloat162float(__float2bfloat16_rn(cosf(ang) * c_rope_mscale));
    const float s = __bfloat162float(__float2bfloat16_rn(sinf(ang) * c_rope_mscale));
    __nv_bfloat16 * xh = x + int64_t(h) * dim;
    const float a = __bfloat162float(xh[d]), b = __bfloat162float(xh[d + half]);
    xh[d] = __float2bfloat16_rn(a * c - b * s);
    xh[d + half] = __float2bfloat16_rn(b * c + a * s);
}

__global__ void sigmoid_gate_k(__nv_bfloat16 * o, const __nv_bfloat16 * qg, int heads, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * dim) return;
    const int h = i / dim, d = i % dim;
    const float g = __bfloat162float(qg[int64_t(h) * 2 * dim + dim + d]);
    const float sg = __bfloat162float(__float2bfloat16_rn(1.f / (1.f + expf(-g))));
    o[i] = __float2bfloat16_rn(__bfloat162float(o[i]) * sg);
}

__global__ void conv_step_k(const __nv_bfloat16 * x, const __nv_bfloat16 * sin, __nv_bfloat16 * st, const float * w,
                            __nv_bfloat16 * y, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const __nv_bfloat16 s0 = sin[c], s1 = sin[C + c], s2 = sin[2 * C + c], xc = x[c];
    const float acc = w[c * 4 + 0] * __bfloat162float(s0) +
                      w[c * 4 + 1] * __bfloat162float(s1) +
                      w[c * 4 + 2] * __bfloat162float(s2) +
                      w[c * 4 + 3] * __bfloat162float(xc);
    const float conv = __bfloat162float(__float2bfloat16_rn(acc));
    y[c] = __float2bfloat16_rn(conv / (1.f + expf(-conv)));
    st[c] = s1;
    st[C + c] = s2;
    st[2 * C + c] = xc;
}

// One block per value head, one thread per value column j (dv threads). State layout [vh][dk][dv].
__global__ void gdn_step_k(const __nv_bfloat16 * qkv, const __nv_bfloat16 * a, const __nv_bfloat16 * b,
                           const float * neg_a, const float * dt_bias, const float * sin, float * state,
                           __nv_bfloat16 * out, int kheads, int vheads, int dk, int dv, bool tiled) {
    extern __shared__ float smem[];
    float * qs = smem;           // [dk]
    float * ks = smem + dk;      // [dk]
    float * red = smem + 2 * dk; // [32]
    const int h = blockIdx.x, j = threadIdx.x, kh = tiled ? h % kheads : h / (vheads / kheads);
    const __nv_bfloat16 * qp = qkv + kh * dk;
    const __nv_bfloat16 * kp = qkv + kheads * dk + kh * dk;
    const __nv_bfloat16 * vp = qkv + 2 * kheads * dk + h * dv;

    float sq = 0.f, sk = 0.f;
    for (int i = j; i < dk; i += blockDim.x) {
        const float qv = __bfloat162float(qp[i]), kv = __bfloat162float(kp[i]);
        qs[i] = qv;
        ks[i] = kv;
        sq += qv * qv;
        sk += kv * kv;
    }
    const float nq = rsqrtf(block_sum(sq, red) + 1e-6f) / sqrtf(float(dk));
    const float nk = rsqrtf(block_sum(sk, red) + 1e-6f);
    __syncthreads();
    for (int i = j; i < dk; i += blockDim.x) {
        qs[i] *= nq;
        ks[i] *= nk;
    }
    __syncthreads();

    const float beta = 1.f / (1.f + expf(-__bfloat162float(b[h])));
    const float x = __bfloat162float(a[h]) + dt_bias[h];
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    const float decay = expf(neg_a[h] * softplus);

    if (j < dv) {
        const float * Si = sin + int64_t(h) * dk * dv + j;
        float * S = state + int64_t(h) * dk * dv + j;
        float kv_mem = 0.f, sq_out = 0.f, kq = 0.f;
        for (int i = 0; i < dk; ++i) {
            const float s = Si[int64_t(i) * dv] * decay;
            kv_mem += s * ks[i];
            sq_out += s * qs[i];
            kq += ks[i] * qs[i];
        }
        const float delta = (__bfloat162float(vp[j]) - kv_mem) * beta;
        for (int i = 0; i < dk; ++i) S[int64_t(i) * dv] = Si[int64_t(i) * dv] * decay + ks[i] * delta;
        out[h * dv + j] = __float2bfloat16_rn(sq_out + delta * kq);
    }
}

__global__ void gated_rms_k(const __nv_bfloat16 * x, const __nv_bfloat16 * z, const float * w, __nv_bfloat16 * y, int dim,
                            float eps) {
    __shared__ float sh[32];
    const int64_t base = int64_t(blockIdx.x) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(x[base + i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float n = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[base + i]) * inv));
        const float zf = __bfloat162float(z[base + i]);
        y[base + i] = __float2bfloat16_rn(w[i] * n * (zf / (1.f + expf(-zf))));
    }
}


__global__ void conv_multi_k(const __nv_bfloat16 * xin, int xs_, int n, const __nv_bfloat16 * sin, __nv_bfloat16 * sout,
                             const float * w, __nv_bfloat16 * y, int C, const int * par = nullptr,
                             const tqllm_cuda::ConvReplay rp = {}) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (par) {  // tree: node i's window is its three nearest ancestors (then the conv state), as along a chain
        __shared__ int spar[16];
        if (threadIdx.x < n && threadIdx.x < 16) spar[threadIdx.x] = par[threadIdx.x];
        __syncthreads();
        if (c >= C) return;
        const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
        __nv_bfloat16 xs[16];
        for (int i = 0; i < n; ++i) xs[i] = xin[int64_t(i) * xs_ + c];
        __nv_bfloat16 hs[3] = {sin[c], sin[C + c], sin[2 * C + c]};  // s0, s1, s2 (s2 most recent)
        const int pend = rp.pend ? *rp.pend : 0;
        for (int r = 0; r < pend; ++r) {  // deferred commit: the previous step's accepted path, as a chain
            const __nv_bfloat16 xc = rp.x[int64_t(r) * xs_ + c];
            const float acc = w0 * __bfloat162float(hs[0]) + w1 * __bfloat162float(hs[1]) + w2 * __bfloat162float(hs[2]) +
                              w3 * __bfloat162float(xc);
            const float conv = __bfloat162float(__float2bfloat16_rn(acc));
            rp.y[int64_t(r) * C + c] = __float2bfloat16_rn(conv / (1.f + expf(-conv)));
            hs[0] = hs[1], hs[1] = hs[2], hs[2] = xc;
        }
        if (pend) rp.smid[c] = hs[0], rp.smid[C + c] = hs[1], rp.smid[2 * C + c] = hs[2];
        __nv_bfloat16 v[3];
        for (int i = 0; i < n; ++i) {
            const __nv_bfloat16 xc = xs[i];
            int a = spar[i], m = 0;  // v[0] = nearest ancestor input
            for (; m < 3 && a >= 0; ++m, a = spar[a]) v[m] = xs[a];
            for (int st = 2; m < 3; ++m, --st) v[m] = hs[st];  // then s2, s1, s0 of the state (after the replay)
            const float acc = w0 * __bfloat162float(v[2]) + w1 * __bfloat162float(v[1]) + w2 * __bfloat162float(v[0]) +
                              w3 * __bfloat162float(xc);
            const float conv = __bfloat162float(__float2bfloat16_rn(acc));
            y[int64_t(i) * C + c] = __float2bfloat16_rn(conv / (1.f + expf(-conv)));
            if (i == n - 1 && !rp.smid) {  // state after the last node (a chain's final state)
                sout[c] = v[1];
                sout[C + c] = v[0];
                sout[2 * C + c] = xc;
            }
        }
        return;
    }
    if (c >= C) return;
    __nv_bfloat16 s0 = sin[c], s1 = sin[C + c], s2 = sin[2 * C + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int i = 0; i < n; ++i) {
        const __nv_bfloat16 xc = xin[int64_t(i) * xs_ + c];
        const float acc = w0 * __bfloat162float(s0) + w1 * __bfloat162float(s1) + w2 * __bfloat162float(s2) +
                          w3 * __bfloat162float(xc);
        const float conv = __bfloat162float(__float2bfloat16_rn(acc));
        y[int64_t(i) * C + c] = __float2bfloat16_rn(conv / (1.f + expf(-conv)));
        s0 = s1;
        s1 = s2;
        s2 = xc;
    }
    sout[c] = s0;
    sout[C + c] = s1;
    sout[2 * C + c] = s2;
}

// Chain conv over a long block (prefill): one thread per (channel, run of CONV_RUN tokens) instead of one per channel;
// every output still reads its own 3 predecessors (the input or the incoming state) - same arithmetic as conv_multi_k.
constexpr int CONV_RUN = 32;
__global__ void conv_long_k(const __nv_bfloat16 * xin, int xs_, int n, const __nv_bfloat16 * sin, __nv_bfloat16 * sout,
                            const float * w, __nv_bfloat16 * y, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, i0 = blockIdx.y * CONV_RUN;
    if (c >= C) return;
    auto in = [&](int i) { return i >= 0 ? xin[int64_t(i) * xs_ + c] : sin[(3 + i) * C + c]; };  // i = -1: s2 ...
    __nv_bfloat16 s0 = in(i0 - 3), s1 = in(i0 - 2), s2 = in(i0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const int i1 = min(n, i0 + CONV_RUN);
    for (int i = i0; i < i1; ++i) {
        const __nv_bfloat16 xc = xin[int64_t(i) * xs_ + c];
        const float acc = w0 * __bfloat162float(s0) + w1 * __bfloat162float(s1) + w2 * __bfloat162float(s2) +
                          w3 * __bfloat162float(xc);
        const float conv = __bfloat162float(__float2bfloat16_rn(acc));
        y[int64_t(i) * C + c] = __float2bfloat16_rn(conv / (1.f + expf(-conv)));
        s0 = s1;
        s1 = s2;
        s2 = xc;
    }
    if (i1 == n) {
        sout[c] = s0;
        sout[C + c] = s1;
        sout[2 * C + c] = s2;
    }
}

// grid (vh, dv / GDN_CPB), 128 threads: value column j = blockIdx.y * GDN_CPB + warp * CPW + lane % CPW of head h is
// split over SL = GDN_SL threads (slice q = lane / CPW owns state rows R q .. R q + R - 1, in registers for all n tokens;
// 8 / 16 slices measured slower: every extra block repeats the per-token q / k loads and norms). All tokens'
// q / k / v and gates are fetched up front and the norms computed in one pass, so the recurrence itself only reads
// shared memory and registers (no block barriers between tokens).
constexpr int GDN_SL = 4, GDN_CPB = 4 * (32 / GDN_SL);
template <int DK>
__global__ void __launch_bounds__(128) gdn_multi_k(const __nv_bfloat16 * qkv, int n, const __nv_bfloat16 * a0,
                                                   const __nv_bfloat16 * b0, int abs_, const float * neg_a,
                                                   const float * dt_bias, const float * sin, float * sout,
                                                   __nv_bfloat16 * y, int kheads, int vheads, int dv, bool tiled,
                                                   const int * par = nullptr, const tqllm_cuda::GdnReplay rp = {}) {
    static_assert(DK == 128, "one element per thread for the q / k loads");
    constexpr int SL = GDN_SL, CPW = 32 / SL, R = DK / SL, NMAX = 16;
    // tree + deferred commit (rp.pend): the previous step's accepted path (rp.* rows, *rp.pend tokens) first, then its
    // state -> rp.smid, then the tree from there. Tokens 0 .. pend - 1 are the replay, pend .. pend + n - 1 the tree.
    const int pend = par && rp.pend ? *rp.pend : 0;
    __shared__ __align__(16) __nv_bfloat16 qs[NMAX][DK], ks[NMAX][DK];  // the inputs are BF16: exact, half the smem
    __shared__ float coef[NMAX][5];  // nq, nk, kq * nk * nq, beta, decay
    __shared__ float vsh[NMAX][GDN_CPB];  // v of this block's value columns per token
    const int h = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int sl = lane / CPW, j = blockIdx.y * GDN_CPB + warp * CPW + lane % CPW;
    const int kh = tiled ? h % kheads : h / (vheads / kheads);
    const int qkv_dim = 2 * kheads * DK + vheads * dv;
    float S[R];
    const float * Si = sin + (int64_t(h) * DK + sl * R) * dv + j;
#pragma unroll
    for (int i = 0; i < R; ++i) S[i] = Si[int64_t(i) * dv];
    // tree: parents and the initial state in shared memory (branch points rebuild from them without global loads)
    __shared__ int spar[NMAX];
    __shared__ float S0[R][128];
    if (par) {
        if (t < n) spar[t] = par[t];
#pragma unroll
        for (int i = 0; i < R; ++i) S0[i][t] = S[i];
    }
    const int ntot = n + pend;
    for (int g0 = 0; g0 < ntot; g0 += NMAX) {  // token groups; the state column stays in registers throughout
        const int ng = min(NMAX, ntot - g0);
    #pragma unroll
        for (int tkn = 0; tkn < NMAX; ++tkn) {
            if (tkn >= ng) break;
            const __nv_bfloat16 * row = g0 + tkn < pend ? rp.qkv + int64_t(g0 + tkn) * qkv_dim
                                                        : qkv + int64_t(g0 + tkn - pend) * qkv_dim;
            qs[tkn][t] = row[kh * DK + t];
            ks[tkn][t] = row[kheads * DK + kh * DK + t];
            if (sl == 0) vsh[tkn][warp * CPW + lane % CPW] = __bfloat162float(row[2 * kheads * DK + h * dv + j]);
        }
        if (t < ng) {
            const bool r = g0 + t < pend;
            const __nv_bfloat16 * bb = r ? rp.b0 + int64_t(g0 + t) * rp.abs_ : b0 + int64_t(g0 + t - pend) * abs_;
            const __nv_bfloat16 * aa = r ? rp.a0 + int64_t(g0 + t) * rp.abs_ : a0 + int64_t(g0 + t - pend) * abs_;
            coef[t][3] = 1.f / (1.f + expf(-__bfloat162float(bb[h])));
            const float x = __bfloat162float(aa[h]) + dt_bias[h];
            coef[t][4] = expf(neg_a[h] * (x > 20.f ? x : log1pf(expf(x))));
        }
        __syncthreads();
        for (int tkn = warp; tkn < ng; tkn += 4) {  // norms: one warp per token
            float sq = 0.f, sk = 0.f, kq = 0.f;
            for (int i = lane; i < DK; i += 32) {
                const float qv = __bfloat162float(qs[tkn][i]), kv = __bfloat162float(ks[tkn][i]);
                sq += qv * qv;
                sk += kv * kv;
                kq += kv * qv;
            }
            for (int o = 16; o; o >>= 1) {
                sq += __shfl_xor_sync(0xffffffffu, sq, o);
                sk += __shfl_xor_sync(0xffffffffu, sk, o);
                kq += __shfl_xor_sync(0xffffffffu, kq, o);
            }
            if (lane == 0) {
                const float nq = rsqrtf(sq + 1e-6f) / sqrtf(float(DK)), nk = rsqrtf(sk + 1e-6f);
                coef[tkn][0] = nq;
                coef[tkn][1] = nk;
                coef[tkn][2] = kq * nk * nq;
            }
        }
        __syncthreads();
        // one recurrence step for token tkn of this group (tree mode: every call goes through here, chains included)
        auto step = [&](int tkn, bool write) {
            const float nq = coef[tkn][0], nk = coef[tkn][1], kq = coef[tkn][2], beta = coef[tkn][3], decay = coef[tkn][4];
            float kv_mem = 0.f, sq_out = 0.f;  // q / k read from shared memory in both passes (no register copies)
#pragma unroll
            for (int i = 0; i < R; i += 4) {
                const uint2 k4 = *reinterpret_cast<const uint2 *>(&ks[tkn][sl * R + i]);
                const uint2 q4 = *reinterpret_cast<const uint2 *>(&qs[tkn][sl * R + i]);
                const float2 k01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&k4.x)),
                             k23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&k4.y)),
                             q01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&q4.x)),
                             q23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&q4.y));
                const float kr[4] = {k01.x, k01.y, k23.x, k23.y}, qr[4] = {q01.x, q01.y, q23.x, q23.y};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    S[i + e] *= decay;
                    kv_mem += S[i + e] * kr[e];
                    sq_out += S[i + e] * qr[e];
                }
            }
#pragma unroll
            for (int o = CPW; o < 32; o <<= 1) kv_mem += __shfl_xor_sync(0xffffffffu, kv_mem, o);
#pragma unroll
            for (int o = CPW; o < 32; o <<= 1) sq_out += __shfl_xor_sync(0xffffffffu, sq_out, o);
            const float delta = (vsh[tkn][warp * CPW + lane % CPW] - kv_mem * nk) * beta;
#pragma unroll
            for (int i = 0; i < R; i += 4) {
                const uint2 k4 = *reinterpret_cast<const uint2 *>(&ks[tkn][sl * R + i]);
                const float2 k01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&k4.x)),
                             k23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&k4.y));
                const float kr[4] = {k01.x, k01.y, k23.x, k23.y};
#pragma unroll
                for (int e = 0; e < 4; ++e) S[i + e] += kr[e] * nk * delta;
            }
            if (write && sl == 0) y[int64_t(g0 + tkn - pend) * vheads * dv + h * dv + j] = __float2bfloat16_rn(sq_out * nq + delta * kq);
        };
        if (par) {  // tree (pend + n <= NMAX): node i starts from its parent's state; off the chain, rebuilt from S0
            for (int tkn = 0; tkn < pend; ++tkn) step(tkn, false);  // deferred commit of the previous step
            if (pend) {
                float * Sm = rp.smid + (int64_t(h) * DK + sl * R) * dv + j;
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    Sm[int64_t(i) * dv] = S[i];
                    S0[i][t] = S[i];
                }
            }
            int prev = -1;
            for (int nd = 0; nd < n; ++nd) {
                const int p = spar[nd];
                if (p != prev) {
#pragma unroll
                    for (int i = 0; i < R; ++i) S[i] = S0[i][t];
                    int path[NMAX], len = 0;
                    for (int a = p; a >= 0; a = spar[a]) path[len++] = a;
                    for (int q = len - 1; q >= 0; --q) step(pend + path[q], false);
                }
                step(pend + nd, true);
                prev = nd;
            }
        } else {
            for (int tkn = 0; tkn < ng; ++tkn) step(tkn, true);
        }
        __syncthreads();  // qs / ks / coef are refilled by the next group
    }
    if (rp.smid) return;  // deferred-commit mode: the state after the tree is never used (the next step replays)
    float * So = sout + (int64_t(h) * DK + sl * R) * dv + j;
#pragma unroll
    for (int i = 0; i < R; ++i) So[int64_t(i) * dv] = S[i];
}

__global__ void gated_rms_multi_k(const __nv_bfloat16 * x, const __nv_bfloat16 * z0, int zs, int heads, const float * w,
                                  __nv_bfloat16 * y, int dim, float eps) {
    __shared__ float sh[32];
    const int tkn = blockIdx.x / heads, h = blockIdx.x % heads;
    const int64_t base = int64_t(blockIdx.x) * dim;
    const __nv_bfloat16 * z = z0 + int64_t(tkn) * zs + int64_t(h) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(x[base + i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float nrm = __bfloat162float(__float2bfloat16_rn(__bfloat162float(x[base + i]) * inv));
        const float zf = __bfloat162float(z[i]);
        y[base + i] = __float2bfloat16_rn(w[i] * nrm * (zf / (1.f + expf(-zf))));
    }
}


__global__ void rms_f_multi_k(const __nv_bfloat16 * x, int xts, int xrs, const float * w, __nv_bfloat16 * y, int yts,
                              int yrs, int len, int rows, float eps) {
    __shared__ float sh[32];
    const int i = blockIdx.x / rows, r = blockIdx.x % rows;
    const __nv_bfloat16 * xr = x + int64_t(i) * xts + int64_t(r) * xrs;
    __nv_bfloat16 * yr = y + int64_t(i) * yts + int64_t(r) * yrs;
    float ss = 0.f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float v = __bfloat162float(xr[j]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / len + eps);
    for (int j = threadIdx.x; j < len; j += blockDim.x) yr[j] = __float2bfloat16_rn(__bfloat162float(xr[j]) * inv * w[j]);
}

// rms_f_multi_k (one row of len == blockDim) followed by rope_partial_multi_k on that row, same arithmetic, one kernel
// Interleaved M-RoPE (ggml IMROPE): mpos [n][4] = (unused, t, h, w) per token; pair j takes h if j % 3 == 1 and
// j < 3 * sec.y, w if j % 3 == 2 and j < 3 * sec.z, else t. Text tokens (t = h = w) match the 1-D rotation exactly.
__device__ __forceinline__ int mrope_pos(const int * mpos, int i, int j, int3 sec) {
    const int * p = mpos + 4 * i;
    if (j % 3 == 1 && j < 3 * sec.y) return p[2];
    if (j % 3 == 2 && j < 3 * sec.z) return p[3];
    return p[1];
}

__global__ void rms_rope_multi_k(const __nv_bfloat16 * x, int xts, int xrs, const float * w, __nv_bfloat16 * y, int yts,
                                 int yrs, int len, int rows, float eps, int rot, const int * dpos0, float theta,
                                 const int * dep, const int * mpos, int3 sec) {
    __shared__ float sh[32];
    __shared__ __nv_bfloat16 sy[512];
    const int i = blockIdx.x / rows, r = blockIdx.x % rows, j = threadIdx.x, half = rot / 2;
    const __nv_bfloat16 * xr = x + int64_t(i) * xts + int64_t(r) * xrs;
    __nv_bfloat16 * yr = y + int64_t(i) * yts + int64_t(r) * yrs;
    const float v = __bfloat162float(xr[j]);
    const float inv = rsqrtf(block_sum(v * v, sh) / len + eps);
    sy[j] = __float2bfloat16_rn(v * inv * w[j]);
    __syncthreads();
    if (j < half) {
        const int p = mpos ? mrope_pos(mpos, i, j, sec) : *dpos0 + (dep ? dep[i] : i);
        const float ang = float(p) * (1.f / powf(theta, float(2 * j) / float(rot)) * (1.f + c_rope_dmul[j]));
        const float c = __bfloat162float(__float2bfloat16_rn(cosf(ang) * c_rope_mscale));
        const float s = __bfloat162float(__float2bfloat16_rn(sinf(ang) * c_rope_mscale));
        const float a = __bfloat162float(sy[j]), b = __bfloat162float(sy[j + half]);
        yr[j] = __float2bfloat16_rn(a * c - b * s);
        yr[j + half] = __float2bfloat16_rn(b * c + a * s);
    } else if (j >= rot) {
        yr[j] = sy[j];
    }
}

__global__ void rope_partial_multi_k(__nv_bfloat16 * x, int ts, int heads, int dim, int rot, const int * dpos0, float theta,
                                     const int * dep, const int * mpos, int3 sec) {
    const int tkn = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x, half = rot / 2;
    if (i >= heads * half) return;
    const int h = i / half, d = i % half;
    const int p = mpos ? mrope_pos(mpos, tkn, d, sec) : *dpos0 + (dep ? dep[tkn] : tkn);
    const float ang = float(p) * (1.f / powf(theta, float(2 * d) / float(rot)) * (1.f + c_rope_dmul[d]));
    const float c = __bfloat162float(__float2bfloat16_rn(cosf(ang) * c_rope_mscale));
    const float s = __bfloat162float(__float2bfloat16_rn(sinf(ang) * c_rope_mscale));
    __nv_bfloat16 * xh = x + int64_t(tkn) * ts + int64_t(h) * dim;
    const float a = __bfloat162float(xh[d]), b = __bfloat162float(xh[d + half]);
    xh[d] = __float2bfloat16_rn(a * c - b * s);
    xh[d + half] = __float2bfloat16_rn(b * c + a * s);
}

__global__ void cache_write_multi_k(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, __nv_bfloat16 * kc,
                                    __nv_bfloat16 * vc, const int * dpos0, int width, int ring) {
    const int pos0 = *dpos0;
    const int tkn = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int64_t dst = int64_t(ring ? (pos0 + tkn) % ring : pos0 + tkn) * width + i;
    kc[dst] = k0[int64_t(tkn) * ts + i];
    vc[dst] = v0[int64_t(tkn) * ts + i];
}

__global__ void sigmoid_gate_multi_k(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int heads, int dim) {
    const int tkn = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * dim) return;
    const int h = i / dim, d = i % dim;
    const float g = __bfloat162float(qg0[int64_t(tkn) * ts + int64_t(h) * 2 * dim + dim + d]);
    const float sg = __bfloat162float(__float2bfloat16_rn(1.f / (1.f + expf(-g))));
    __nv_bfloat16 * op = o + int64_t(tkn) * heads * dim + i;
    *op = __float2bfloat16_rn(__bfloat162float(*op) * sg);
}

// ---- fused "elementwise op -> TQ3 32-point rotation" (the rotation and rounding of tq3_rotate_h, same arithmetic) ----
__constant__ float TQ3_SIGNS_F[32] = {
    +1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1,
};
__device__ __forceinline__ __half tq3_rot32_h(__nv_bfloat16 x, int lane) {
    float v = __bfloat162float(x) * TQ3_SIGNS_F[lane];
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, s);
        v = (lane & s) ? o - v : v + o;
    }
    return __float2half_rn(v * 0.17677669529663687f);
}

// swiglu_multi + tq3_rotate_h: act never leaves registers (F % 32 == 0)
__global__ void swiglu_rot_k(const __nv_bfloat16 * gu, int F, __half * xh) {
    const int tkn = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;  // whole warps (F % 32 == 0)
    const float a = __bfloat162float(gu[int64_t(tkn) * 2 * F + f]), u = __bfloat162float(gu[int64_t(tkn) * 2 * F + F + f]);
    xh[int64_t(tkn) * F + f] = tq3_rot32_h(__float2bfloat16_rn(a / (1.f + expf(-a)) * u), threadIdx.x % 32);
}

// gated_rms_multi + tq3_rotate_h (dim == blockDim == 128)
__global__ void gated_rms_rot_k(const __nv_bfloat16 * x, const __nv_bfloat16 * z0, int zs, int heads, const float * w,
                                __half * xh, int dim, float eps) {
    __shared__ float sh[32];
    const int tkn = blockIdx.x / heads, h = blockIdx.x % heads, i = threadIdx.x;
    const int64_t base = int64_t(blockIdx.x) * dim;
    const __nv_bfloat16 * z = z0 + int64_t(tkn) * zs + int64_t(h) * dim;
    const float xv = __bfloat162float(x[base + i]);
    const float inv = rsqrtf(block_sum(xv * xv, sh) / dim + eps);
    const float nrm = __bfloat162float(__float2bfloat16_rn(xv * inv));
    const float zf = __bfloat162float(z[i]);
    xh[base + i] = tq3_rot32_h(__float2bfloat16_rn(w[i] * nrm * (zf / (1.f + expf(-zf)))), i % 32);
}

// sigmoid_gate_multi + tq3_rotate_h (heads * dim % 256 == 0); also writes the gated O
__global__ void sigmoid_gate_rot_k(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int heads, int dim, __half * xh) {
    const int tkn = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = i / dim, d = i % dim;
    const float g = __bfloat162float(qg0[int64_t(tkn) * ts + int64_t(h) * 2 * dim + dim + d]);
    const float sg = __bfloat162float(__float2bfloat16_rn(1.f / (1.f + expf(-g))));
    __nv_bfloat16 * op = o + int64_t(tkn) * heads * dim + i;
    const __nv_bfloat16 r = __float2bfloat16_rn(__bfloat162float(*op) * sg);
    *op = r;
    xh[int64_t(tkn) * heads * dim + i] = tq3_rot32_h(r, threadIdx.x % 32);
}

__global__ void swiglu_multi_k(const __nv_bfloat16 * gu, int F, __nv_bfloat16 * act) {
    const int tkn = blockIdx.y, f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F) return;
    const float a = __bfloat162float(gu[int64_t(tkn) * 2 * F + f]), u = __bfloat162float(gu[int64_t(tkn) * 2 * F + F + f]);
    act[int64_t(tkn) * F + f] = __float2bfloat16_rn(a / (1.f + expf(-a)) * u);
}



// grid (kvheads, n), block = dim threads: each warp quantises 32 consecutive values of K and V.
template <bool WRITE_V>
__global__ void cache_write_multi_q8_k(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int8_t * kc,
                                       int8_t * vc, __half * ksc, __half * vsc, const int * dpos0, int kvheads, int dim) {
    const int kh = blockIdx.x, tkn = blockIdx.y, d = threadIdx.x, lane = d % 32;
    const int pos = *dpos0 + tkn;
    const float kf = __bfloat162float(k0[int64_t(tkn) * ts + kh * dim + d]);
    float mk = fabsf(kf);
    float mv = 0.f;
    if constexpr (WRITE_V) mv = fabsf(__bfloat162float(v0[int64_t(tkn) * ts + kh * dim + d]));
    for (int o = 16; o; o >>= 1) {
        mk = fmaxf(mk, __shfl_xor_sync(0xffffffffu, mk, o));
        if constexpr (WRITE_V) mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, o));
    }
    const __half hk = __float2half_rn(mk / 127.f);
    const float sk = __half2float(hk);
    const int64_t o = (int64_t(pos) * kvheads + kh) * dim + d;
    kc[o] = int8_t(sk > 0.f ? max(-127, min(127, __float2int_rn(kf / sk))) : 0);
    if (lane == 0) {
        const int64_t si = (int64_t(pos) * kvheads + kh) * (dim / 32) + d / 32;
        ksc[si] = hk;
        if constexpr (WRITE_V) {
            const __half hv = __float2half_rn(mv / 127.f);
            vsc[si] = hv;
        }
    }
    if constexpr (WRITE_V) {
        const float vf = __bfloat162float(v0[int64_t(tkn) * ts + kh * dim + d]);
        const __half hv = __float2half_rn(mv / 127.f);
        const float sv = __half2float(hv);
        vc[o] = int8_t(sv > 0.f ? max(-127, min(127, __float2int_rn(vf / sv))) : 0);
    }
}

static __device__ __constant__ float turbo3_v_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static __device__ __constant__ float turbo3_v_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};
static __device__ __constant__ float turbo3_v_centroids[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f, 0.021663f, 0.066822f, 0.118786f, 0.190207f
};

static __device__ __constant__ int turbo4_k_mag[8] = {6, 18, 31, 44, 58, 75, 96, 127};
static __device__ __constant__ float tq3_0_signs[32] = {
    1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1};
static __device__ __constant__ float tq3_0_cent[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f,
                                                      0.230106f,  0.725222f,  1.277503f,  1.988943f};
static __device__ __constant__ float tq3_0_bound[7] = {-1.644041f, -1.015870f, -0.493924f, -0.008701f,
                                                       0.477664f,  1.001362f,  1.633223f};

// Quality experiments: quantise -> dequantise K or V rows in place (BF16) with a KV-cache format, so its accuracy can
// be measured through the unchanged BF16 attention. mode 1 = Turbo3 (128, corrected norm, 8 levels), 2 = Turbo4
// (128, corrected norm, int8 16 levels), 3 = llama.cpp-tq3 TQ3_0 (32, fp16 RMS scale, 32-point RHT, 8 levels).
// Grid (kvheads * (dim / B), n), B threads (B = 128 for modes 1-2, 32 for mode 3).
template <int B>
__global__ void kv_fakequant_k(__nv_bfloat16 * x0, int ts, int kvheads, int dim, int mode) {
    const int d = threadIdx.x, blk = blockIdx.x % (dim / B), kh = blockIdx.x / (dim / B), tkn = blockIdx.y;
    __nv_bfloat16 * x = x0 + int64_t(tkn) * ts + kh * dim + blk * B;
    __shared__ float v[B], red[B];
    const float in = __bfloat162float(x[d]);
    red[d] = in * in;
    __syncthreads();
    for (int o = B / 2; o; o >>= 1) {
        if (d < o) red[d] += red[d + o];
        __syncthreads();
    }
    const float ss = red[0];
    __syncthreads();
    auto wht = [&]() {
        for (int h = 1; h < B; h <<= 1) {
            __syncthreads();
            const int group = (d / (2 * h)) * (2 * h), l = d % (2 * h);
            if (l < h) {
                const float a = v[group + l], b = v[group + l + h];
                v[group + l] = a + b;
                v[group + l + h] = a - b;
            }
        }
        __syncthreads();
    };
    const float rs = rsqrtf(float(B));
    float out;
    if (mode == 3) {
        float rms = sqrtf(ss / B);
        if (rms < 1e-10f) rms = 1.f;
        rms = __half2float(__float2half_rn(rms));
        v[d] = in / rms * tq3_0_signs[d % 32];
        wht();
        const float r = v[d] * rs;
        int idx = 0;
        for (int b = 0; b < 7; ++b) idx = r > tq3_0_bound[b] ? b + 1 : idx;
        __syncthreads();
        v[d] = tq3_0_cent[idx];
        wht();
        out = v[d] * rs * tq3_0_signs[d % 32] * rms;
    } else {
        const float norm = sqrtf(ss);
        v[d] = (norm > 1e-10f ? in / norm : 0.f) * turbo3_v_s1[d];
        wht();
        const float r = v[d] * rs * turbo3_v_s2[d], a = fabsf(r);
        float q;
        if (mode == 1) {
            int idx = 0;
            idx = (r >= -0.154496f) + (r >= -0.092804f) + (r >= -0.044243f) + (r >= 0.f) + (r >= 0.044243f) +
                  (r >= 0.092804f) + (r >= 0.154496f);
            q = turbo3_v_centroids[idx];
        } else {
            const int rank = (a >= 0.022824f) + (a >= 0.046174f) + (a >= 0.070671f) + (a >= 0.097164f) +
                             (a >= 0.127026f) + (a >= 0.162947f) + (a >= 0.212203f);
            q = (r < 0.f ? -1.f : 1.f) * float(turbo4_k_mag[rank]) * (0.241529f / 127.f);
        }
        __syncthreads();
        red[d] = q * q;
        __syncthreads();
        for (int o = B / 2; o; o >>= 1) {
            if (d < o) red[d] += red[d + o];
            __syncthreads();
        }
        const float recon = sqrtf(red[0]);
        v[d] = q * turbo3_v_s2[d];
        wht();
        out = v[d] * rs * turbo3_v_s1[d] * (recon > 1e-10f ? norm / recon : 0.f);
    }
    x[d] = __float2bfloat16_rn(out);
}


// Grid (kvheads * (dim / 128), n), 128 threads per 128-value half of a K head vector.
// VL = false: K layout (int8 MMA operand words); VL = true: V layout (per 32-dim chunk c, 16-bit field l = 0..7 in
// word 4c + l / 2, half l % 2, holds the nibbles of dims 32c + 8j + l, j = 0..3: one PV fragment lane's values).
template <bool VL>
__global__ void cache_write_multi_turbo4_k_k(const __nv_bfloat16 * k0, int ts, tqllm_cuda::Turbo4K256 * kc,
                                             const int * dpos0, int kvheads, int dim) {
    const int d = threadIdx.x, sub = blockIdx.x % (dim / 128), kh = blockIdx.x / (dim / 128), tkn = blockIdx.y;
    const int pos = *dpos0 + tkn;
    __shared__ float x[128], scratch[128];
    __shared__ uint8_t codes[128];
    const float value = __bfloat162float(k0[int64_t(tkn) * ts + kh * dim + sub * 128 + d]);
    x[d] = value;
    scratch[d] = value * value;
    __syncthreads();
    for (int offset = 64; offset; offset >>= 1) {
        if (d < offset) scratch[d] += scratch[d + offset];
        __syncthreads();
    }
    const float norm = sqrtf(scratch[0]);
    x[d] = (norm > 1e-10f ? x[d] / norm : 0.f) * turbo3_v_s1[d];
    for (int h = 1; h < 128; h <<= 1) {
        __syncthreads();
        const int group = (d / (2 * h)) * (2 * h), lane = d % (2 * h);
        if (lane < h) {
            const float a = x[group + lane], b = x[group + lane + h];
            x[group + lane] = a + b;
            x[group + lane + h] = a - b;
        }
    }
    __syncthreads();
    const float r = x[d] * (0.08838834764831845f * turbo3_v_s2[d]), a = fabsf(r);
    const int rank = (a >= 0.022824f) + (a >= 0.046174f) + (a >= 0.070671f) + (a >= 0.097164f) + (a >= 0.127026f) +
                     (a >= 0.162947f) + (a >= 0.212203f);
    codes[d] = uint8_t(((r < 0.f) << 3) | rank);
    const float lv = float(turbo4_k_mag[rank]) * (0.241529f / 127.f);
    __syncthreads();
    scratch[d] = lv * lv;
    __syncthreads();
    for (int offset = 64; offset; offset >>= 1) {
        if (d < offset) scratch[d] += scratch[d + offset];
        __syncthreads();
    }
    tqllm_cuda::Turbo4K256 * rec = kc + int64_t(pos) * kvheads + kh;
    if (d < 16) {  // words 16 sub .. 16 sub + 15
        const int kbl = d / 4, tg = d % 4;
        uint32_t word = 0;
        for (int p = 0; p < 8; ++p) {
            const int o = VL ? 8 * (p % 4) + 2 * tg + p / 4 : (p < 4 ? 4 * tg + p : 16 + 4 * tg + (p - 4));
            word |= uint32_t(codes[kbl * 32 + o]) << (4 * p);
        }
        rec->w[16 * sub + d] = word;
    }
    if (d == 0) {
        const float recon = sqrtf(scratch[0]);
        rec->scale[sub] = __float2half_rn(recon > 1e-10f ? norm / recon * (0.241529f / 127.f) : 0.f);
    }
}

// Grid (kvheads * (dim / 128), n), 128 threads per vector.
__global__ void cache_write_multi_turbo3_v_k(const __nv_bfloat16 * v0, int ts, tqllm_cuda::Turbo3V256 * vc,
                                             const int * dpos0, int kvheads, int dim) {
    const int d = threadIdx.x, sub = blockIdx.x % (dim / 128), kh = blockIdx.x / (dim / 128), tkn = blockIdx.y;
    const int pos = *dpos0 + tkn;
    const int64_t input = int64_t(tkn) * ts + kh * dim + sub * 128;
    __shared__ float x[128];
    __shared__ float scratch[128];
    __shared__ uint8_t indices[128];
    const float value = __bfloat162float(v0[input + d]);
    x[d] = value;
    scratch[d] = value * value;
    __syncthreads();
    for (int offset = 64; offset; offset >>= 1) {
        if (d < offset) scratch[d] += scratch[d + offset];
        __syncthreads();
    }
    const float norm = sqrtf(scratch[0]);
    x[d] = (norm > 1e-10f ? x[d] / norm : 0.f) * turbo3_v_s1[d];
    for (int h = 1; h < 128; h <<= 1) {
        __syncthreads();
        const int group = (d / (2 * h)) * (2 * h), lane = d % (2 * h);
        if (lane < h) {
            const float a = x[group + lane], b = x[group + lane + h];
            x[group + lane] = a + b;
            x[group + lane + h] = a - b;
        }
    }
    __syncthreads();
    const float rotated = x[d] * (0.08838834764831845f * turbo3_v_s2[d]);
    uint8_t idx = 0;
    if (rotated >= -0.154496f) idx = 1;
    if (rotated >= -0.092804f) idx = 2;
    if (rotated >= -0.044243f) idx = 3;
    if (rotated >= 0.f) idx = 4;
    if (rotated >= 0.044243f) idx = 5;
    if (rotated >= 0.092804f) idx = 6;
    if (rotated >= 0.154496f) idx = 7;
    indices[d] = idx;
    scratch[d] = turbo3_v_centroids[idx] * turbo3_v_centroids[idx];
    __syncthreads();
    for (int offset = 64; offset; offset >>= 1) {
        if (d < offset) scratch[d] += scratch[d + offset];
        __syncthreads();
    }
    tqllm_cuda::Turbo3V256 * rec = vc + int64_t(pos) * kvheads + kh;
    __syncthreads();
    if (d < 12) {  // words 12 sub .. 12 sub + 11 hold chunks 4 sub .. 4 sub + 3 (this 128-value half)
        uint32_t word = 0;
        for (int b = 0; b < 32; ++b) {
            const int bit = d * 32 + b, cl = bit / 96, bb = bit % 96;
            const int l = bb / 12, j = (bb % 12) / 3, i = indices[cl * 32 + 8 * j + l];
            const uint32_t pos = i >= 4, code = (pos << 2) | (pos ? i - 4 : 3 - i);
            word |= ((code >> (bb % 3)) & 1u) << b;
        }
        rec->w[12 * sub + d] = word;
    }
    if (d == 0) {
        const float recon_norm = sqrtf(scratch[0]);
        rec->norm[sub] = __float2half_rn(recon_norm > 1e-10f ? norm / recon_norm : norm);
    }
}

} // namespace

namespace tqllm_cuda {

void rms_f(const __nv_bfloat16 * x, int xs, const float * w, __nv_bfloat16 * y, int ys, int n, int rows, float eps,
           cudaStream_t s) {
    rms_f_k<<<rows, 256, 0, s>>>(x, xs, w, y, ys, n, eps);
}

__global__ void kv_stage_copy_k(const uint4 * src, uint4 * dst, const int64_t * nbytes, const int64_t * off) {
    const int64_t o = off ? *off / 16 : 0, n = *nbytes / 16;
    for (int64_t i = o + blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) dst[i] = src[i];
}

void kv_stage_copy(const void * src, void * dst, const int64_t * nbytes, int blocks, cudaStream_t s, const int64_t * off) {
    kv_stage_copy_k<<<blocks, 256, 0, s>>>(static_cast<const uint4 *>(src), static_cast<uint4 *>(dst), nbytes, off);
}

void set_rope_scaling(const float * mul, int pairs, float mscale) {
    float m[256];
    for (int i = 0; i < 256; ++i) m[i] = i < pairs && mul ? mul[i] - 1.f : 0.f;
    cudaMemcpyToSymbol(c_rope_dmul, m, sizeof(m));
    cudaMemcpyToSymbol(c_rope_mscale, &mscale, sizeof(float));
}

void rope_partial(__nv_bfloat16 * x, int heads, int dim, int rot, int pos, float theta, cudaStream_t s) {
    const int n = heads * rot / 2;
    rope_partial_k<<<(n + 127) / 128, 128, 0, s>>>(x, heads, dim, rot, pos, theta);
}

void sigmoid_gate(__nv_bfloat16 * o, const __nv_bfloat16 * qg, int heads, int dim, cudaStream_t s) {
    sigmoid_gate_k<<<(heads * dim + 255) / 256, 256, 0, s>>>(o, qg, heads, dim);
}

void conv_step(const __nv_bfloat16 * x, __nv_bfloat16 * state, const float * w, __nv_bfloat16 * y, int channels,
               cudaStream_t s) {
    conv_step_k<<<(channels + 255) / 256, 256, 0, s>>>(x, state, state, w, y, channels);
}

void gdn_step(const __nv_bfloat16 * qkv, const __nv_bfloat16 * a, const __nv_bfloat16 * b, const float * neg_a,
              const float * dt_bias, float * state, __nv_bfloat16 * out, int kheads, int vheads, int dk, int dv,
              bool tiled, cudaStream_t s) {
    const int threads = dv < 32 ? 32 : dv;
    gdn_step_k<<<vheads, threads, (2 * dk + 32) * sizeof(float), s>>>(qkv, a, b, neg_a, dt_bias, state, state, out,
                                                                      kheads, vheads, dk, dv, tiled);
}

void conv_step2(const __nv_bfloat16 * x, const __nv_bfloat16 * in, __nv_bfloat16 * out, const float * w,
                __nv_bfloat16 * y, int channels, cudaStream_t s) {
    conv_step_k<<<(channels + 255) / 256, 256, 0, s>>>(x, in, out, w, y, channels);
}

void gdn_step2(const __nv_bfloat16 * qkv, const __nv_bfloat16 * a, const __nv_bfloat16 * b, const float * neg_a,
               const float * dt_bias, const float * in, float * out, __nv_bfloat16 * y, int kheads, int vheads, int dk,
               int dv, bool tiled, cudaStream_t s) {
    const int threads = dv < 32 ? 32 : dv;
    gdn_step_k<<<vheads, threads, (2 * dk + 32) * sizeof(float), s>>>(qkv, a, b, neg_a, dt_bias, in, out, y, kheads,
                                                                       vheads, dk, dv, tiled);
}

void gated_rms(const __nv_bfloat16 * x, const __nv_bfloat16 * z, const float * w, __nv_bfloat16 * y, int heads, int dim,
               float eps, cudaStream_t s) {
    gated_rms_k<<<heads, 128, 0, s>>>(x, z, w, y, dim, eps);
}

void conv_multi(const __nv_bfloat16 * xin, int x_stride, int n, const __nv_bfloat16 * in, __nv_bfloat16 * out,
                const float * w, __nv_bfloat16 * y, int channels, cudaStream_t s, const int * par, const ConvReplay & rp) {
    if (rp.smid && !par) throw std::invalid_argument("conv_multi: deferred replay needs tree mode");
    if (!par && !rp.x && n > 2 * CONV_RUN && in != out) {  // in != out: runs read the incoming state concurrently
        conv_long_k<<<dim3((channels + 127) / 128, (n + CONV_RUN - 1) / CONV_RUN), 128, 0, s>>>(xin, x_stride, n, in, out, w,
                                                                                                y, channels);
        return;
    }
    conv_multi_k<<<(channels + 255) / 256, 256, 0, s>>>(xin, x_stride, n, in, out, w, y, channels, par, rp);
}

void gdn_multi(const __nv_bfloat16 * qkv, int n, const __nv_bfloat16 * a0, const __nv_bfloat16 * b0, int ab_stride,
               const float * neg_a, const float * dt_bias, const float * in, float * out, __nv_bfloat16 * y, int kheads,
               int vheads, int dk, int dv, bool tiled, cudaStream_t s, const int * par, const GdnReplay & rp) {
    if (dk != 128 || dv % 32) throw std::invalid_argument("gdn_multi: dk 128, dv % 32 == 0");
    if (par && n > 16) throw std::invalid_argument("gdn_multi: tree mode needs n <= 16");
    if (rp.smid && (!par || n > 8)) throw std::invalid_argument("gdn_multi: deferred replay needs tree mode, n <= 8");
    if (dv % GDN_CPB) throw std::invalid_argument("gdn_multi: dv % columns per block");
    gdn_multi_k<128><<<dim3(vheads, dv / GDN_CPB), 128, 0, s>>>(qkv, n, a0, b0, ab_stride, neg_a, dt_bias, in, out, y, kheads,
                                                          vheads, dv, tiled, par, rp);
}

void gated_rms_multi(const __nv_bfloat16 * x, const __nv_bfloat16 * z0, int z_stride, int n, const float * w,
                     __nv_bfloat16 * y, int heads, int dim, float eps, cudaStream_t s) {
    gated_rms_multi_k<<<heads * n, 128, 0, s>>>(x, z0, z_stride, heads, w, y, dim, eps);
}

void rms_f_multi(const __nv_bfloat16 * x, int xts, int xrs, const float * w, __nv_bfloat16 * y, int yts, int yrs,
                 int len, int rows, int n, float eps, cudaStream_t s) {
    rms_f_multi_k<<<rows * n, 256, 0, s>>>(x, xts, xrs, w, y, yts, yrs, len, rows, eps);
}

void rms_rope_multi(const __nv_bfloat16 * x, int xts, int xrs, const float * w, __nv_bfloat16 * y, int yts, int yrs, int len,
                    int rows, int n, float eps, int rot, const int * dpos0, float theta, cudaStream_t s, const int * dep,
                    const int * mpos, int3 sec) {
    if (len > 512 || len % 32 || rot > len) throw std::invalid_argument("rms_rope_multi: len <= 512, len % 32, rot <= len");
    rms_rope_multi_k<<<rows * n, len, 0, s>>>(x, xts, xrs, w, y, yts, yrs, len, rows, eps, rot, dpos0, theta, dep, mpos, sec);
}

void rope_partial_multi(__nv_bfloat16 * x, int ts, int n, int heads, int dim, int rot, const int * dpos0, float theta,
                        cudaStream_t s, const int * dep, const int * mpos, int3 sec) {
    rope_partial_multi_k<<<dim3((heads * rot / 2 + 127) / 128, n), 128, 0, s>>>(x, ts, heads, dim, rot, dpos0, theta, dep,
                                                                                mpos, sec);
}

void cache_write_multi(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, __nv_bfloat16 * kc,
                       __nv_bfloat16 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t s, int ring) {
    const int width = kvheads * dim;
    cache_write_multi_k<<<dim3((width + 255) / 256, n), 256, 0, s>>>(k0, v0, ts, kc, vc, dpos0, width, ring);
}

void sigmoid_gate_multi(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int n, int heads, int dim, cudaStream_t s) {
    sigmoid_gate_multi_k<<<dim3((heads * dim + 255) / 256, n), 256, 0, s>>>(o, qg0, ts, heads, dim);
}

void swiglu_rot_h(const __nv_bfloat16 * gu, int n, int F, __half * xh, cudaStream_t s) {
    if (F % 32) throw std::invalid_argument("swiglu_rot_h: F % 32");
    swiglu_rot_k<<<dim3((F + 255) / 256, n), 256, 0, s>>>(gu, F, xh);
}
void gated_rms_rot_h(const __nv_bfloat16 * x, const __nv_bfloat16 * z0, int z_stride, int n, const float * w, __half * xh,
                     int heads, int dim, float eps, cudaStream_t s) {
    if (dim != 128) throw std::invalid_argument("gated_rms_rot_h: dim 128");
    gated_rms_rot_k<<<heads * n, 128, 0, s>>>(x, z0, z_stride, heads, w, xh, dim, eps);
}
void sigmoid_gate_rot_h(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int n, int heads, int dim, __half * xh,
                        cudaStream_t s) {
    if ((heads * dim) % 256) throw std::invalid_argument("sigmoid_gate_rot_h: heads * dim % 256");
    sigmoid_gate_rot_k<<<dim3(heads * dim / 256, n), 256, 0, s>>>(o, qg0, ts, heads, dim, xh);
}

void swiglu_multi(const __nv_bfloat16 * gu, int n, int F, __nv_bfloat16 * act, cudaStream_t s) {
    swiglu_multi_k<<<dim3((F + 255) / 256, n), 256, 0, s>>>(gu, F, act);
}

void cache_write_multi_q8(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, int8_t * kc, int8_t * vc,
                          __half * ksc, __half * vsc, const int * dpos0, int kvheads, int dim, cudaStream_t s) {
    cache_write_multi_q8_k<true><<<dim3(kvheads, n), dim, 0, s>>>(k0, v0, ts, kc, vc, ksc, vsc, dpos0, kvheads, dim);
}

void cache_write_multi_q8_turbo3v(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, int8_t * kc,
                                  __half * ksc, Turbo3V256 * vc, const int * dpos0, int kvheads, int dim,
                                  cudaStream_t s) {
    if (dim % 128) throw std::invalid_argument("cache_write_multi_q8_turbo3v: dim must be divisible by 128");
    cache_write_multi_q8_k<false><<<dim3(kvheads, n), dim, 0, s>>>(k0, nullptr, ts, kc, nullptr, ksc, nullptr, dpos0,
                                                                   kvheads, dim);
    cache_write_multi_turbo3_v_k<<<dim3(kvheads * (dim / 128), n), 128, 0, s>>>(v0, ts, vc, dpos0, kvheads, dim);
}

void cache_write_multi_t4k_turbo3v(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Turbo4K256 * kc,
                                   Turbo3V256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t s) {
    if (dim != 256) throw std::invalid_argument("cache_write_multi_t4k_turbo3v: head_dim must be 256");
    cache_write_multi_turbo4_k_k<false><<<dim3(kvheads * (dim / 128), n), 128, 0, s>>>(k0, ts, kc, dpos0, kvheads, dim);
    cache_write_multi_turbo3_v_k<<<dim3(kvheads * (dim / 128), n), 128, 0, s>>>(v0, ts, vc, dpos0, kvheads, dim);
}

void kv_fakequant(__nv_bfloat16 * x, int ts, int n, int kvheads, int dim, int mode, cudaStream_t s) {
    if (mode == 3) kv_fakequant_k<32><<<dim3(kvheads * (dim / 32), n), 32, 0, s>>>(x, ts, kvheads, dim, mode);
    else if (mode == 1 || mode == 2) kv_fakequant_k<128><<<dim3(kvheads * (dim / 128), n), 128, 0, s>>>(x, ts, kvheads, dim, mode);
}

void cache_write_multi_t4kv(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Turbo4K256 * kc,
                            Turbo4K256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t s) {
    if (dim != 256) throw std::invalid_argument("cache_write_multi_t4kv: head_dim must be 256");
    cache_write_multi_turbo4_k_k<false><<<dim3(kvheads * (dim / 128), n), 128, 0, s>>>(k0, ts, kc, dpos0, kvheads, dim);
    cache_write_multi_turbo4_k_k<true><<<dim3(kvheads * (dim / 128), n), 128, 0, s>>>(v0, ts, vc, dpos0, kvheads, dim);
}

} // namespace tqllm_cuda

// ---- DFlash drafter retrieval window (prototype, --draft-retrieve C) ----
// The drafter attends a 2048-position window; past it, a document's names and facts are out of view. Retrieval keeps the
// window size but fills C x 16 of it with the far-context chunks the anchor query scores highest, moved (re-rotated)
// to sit right before the recent part, so the drafter sees them at distances it was trained on.
namespace {
constexpr int RCHUNK = tqllm_cuda::kDflRetrChunk;
using tqllm_cuda::kDflRetrMaxChunks;

// score[h][c - first] = max over the chunk's 16 keys of <mean of the anchor query over head h's group, key>
__global__ void __launch_bounds__(128) dfl_chunk_score_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, int kd,
                                                         int first, int heads, int kvheads, float * score, int nsc, int ring) {
    const int c = first + blockIdx.x, h = blockIdx.y, d = threadIdx.x, grp = heads / kvheads;
    float qm = 0.f;
    for (int g = 0; g < grp; ++g) qm += __bfloat162float(q[(h * grp + g) * 128 + d]);
    qm /= float(grp);
    __shared__ float red[4];
    float best = -3.4e38f;
    for (int t = 0; t < RCHUNK; ++t) {
        const int p = ring ? (c * RCHUNK + t) % ring : c * RCHUNK + t;
        float v = qm * __bfloat162float(kc[int64_t(p) * kd + h * 128 + d]);
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (d % 32 == 0) red[d / 32] = v;
        __syncthreads();
        if (d == 0) best = fmaxf(best, red[0] + red[1] + red[2] + red[3]);
        __syncthreads();
    }
    if (d == 0) score[int64_t(h) * nsc + blockIdx.x] = best;
}

// per head: the top `take` of n scores (n <= 4096), chunk ids (first + index) ascending -> sel[h][take]
__global__ void __launch_bounds__(1024) dfl_chunk_top_k(const float * score, int nsc, int n, int take, int first, int * sel) {
    __shared__ float s[kDflRetrMaxChunks];
    __shared__ int ix[kDflRetrMaxChunks];
    const int h = blockIdx.x;
    int m = 1;
    while (m < n) m <<= 1;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        s[i] = i < n ? score[int64_t(h) * nsc + i] : -3.4e38f;
        ix[i] = i < n ? i : 0x7fffffff;
    }
    __syncthreads();
    for (int k = 2; k <= m; k <<= 1)  // bitonic: descending score, ties by ascending index
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < m; i += blockDim.x) {
                const int l = i ^ j;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const bool before = s[i] > s[l] || (s[i] == s[l] && ix[i] < ix[l]);
                    if (desc != before) {
                        const float ts = s[i]; s[i] = s[l]; s[l] = ts;
                        const int ti = ix[i]; ix[i] = ix[l]; ix[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    if (threadIdx.x == 0) {  // the chosen ids in ascending order (take <= 128)
        int v[128];
        for (int i = 0; i < take; ++i) v[i] = ix[i];
        for (int i = 1; i < take; ++i)
            for (int j = i; j > 0 && v[j - 1] > v[j]; --j) { const int t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
        for (int i = 0; i < take; ++i) sel[h * take + i] = first + v[i];
    }
}

// scratch row i (per head): i < R: chunk sel[i / 16], token i % 16, key rotated from its position to vbase + i;
// i >= R: position F + (i - R), copied. vbase = F - R, so the recent part keeps its own positions.
__global__ void __launch_bounds__(128) dfl_gather_k(const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, int kd,
                                                    const int * sel, int take, int F, float theta, int ring,
                                                    __nv_bfloat16 * sk, __nv_bfloat16 * sv) {
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x, R = take * RCHUNK;
    int src;
    float delta = 0.f;
    if (i < R) {
        src = sel[h * take + i / RCHUNK] * RCHUNK + i % RCHUNK;
        delta = float((F - R + i) - src);
    } else {
        src = F + (i - R);
    }
    const int row = ring ? src % ring : src;
    const __nv_bfloat16 * ks = kc + int64_t(row) * kd + h * 128;
    __nv_bfloat16 * kd_ = sk + int64_t(i) * kd + h * 128;
    sv[int64_t(i) * kd + h * 128 + d] = vc[int64_t(row) * kd + h * 128 + d];
    if (d < 64) {
        const float a = __bfloat162float(ks[d]), b = __bfloat162float(ks[d + 64]);
        if (delta != 0.f) {  // rotate by delta (RoPE angles add); no mscale (it is already in the stored key)
            const float inv_freq = 1.f / powf(theta, float(2 * d) / 128.f) * (1.f + c_rope_dmul[d]);
            const float ang = delta * inv_freq, c = cosf(ang), s = sinf(ang);
            kd_[d] = __float2bfloat16_rn(a * c - b * s);
            kd_[d + 64] = __float2bfloat16_rn(b * c + a * s);
        } else {
            kd_[d] = ks[d];
            kd_[d + 64] = ks[d + 64];
        }
    }
}
}  // namespace

namespace tqllm_cuda {
int dflash_retrieve_build(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, int kd, int heads,
                          int kvheads, int pos, int nq, int recent, int chunks, float theta, float * score, int * sel,
                          __nv_bfloat16 * sk, __nv_bfloat16 * sv, cudaStream_t s, int ring) {
    const int F = pos - recent;  // positions below F are far context
    const int nc = F > 0 ? F / RCHUNK : 0;
    if (nc <= 0 || chunks <= 0) return -1;
    const int n = std::min(nc, kDflRetrMaxChunks), first = nc - n, take = std::min(chunks, std::min(n, 128));
    dfl_chunk_score_k<<<dim3(n, kvheads), 128, 0, s>>>(q, kc, kd, first, heads, kvheads, score, kDflRetrMaxChunks, ring);
    dfl_chunk_top_k<<<kvheads, 1024, 0, s>>>(score, kDflRetrMaxChunks, n, take, first, sel);
    const int rows = take * RCHUNK + recent + nq;
    dfl_gather_k<<<dim3(rows, kvheads), 128, 0, s>>>(kc, vc, kd, sel, take, F, theta, ring, sk, sv);
    return take * RCHUNK;  // R: scratch rows before the recent part
}
}  // namespace tqllm_cuda

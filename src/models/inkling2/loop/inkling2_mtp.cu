// Inkling-2-Mini MTP drafter kernels (see inkling2_mtp.cuh).
#include "inkling2_mtp.cuh"

#include <cmath>
#include <stdexcept>

namespace tqllm_cuda::inkling2 {
namespace {
constexpr int HD = 128, PART = HD + 2;

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float block_sum(float v, float * sh) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < int(blockDim.x / 32); ++w) t += sh[w];
    return t;
}

__global__ void mix_norm_k(const __nv_bfloat16 * taps, int ntaps, const float * alpha, const float * w, __nv_bfloat16 * out, int D, float eps) {
    const int r = blockIdx.x;
    __shared__ float sh[8];
    float ss = 0.0f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float m = 0.0f;
        for (int j = 0; j < ntaps; ++j) m += alpha[j] * __bfloat162float(taps[(size_t(r) * ntaps + j) * D + c]);
        ss += m * m;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / float(D) + eps);
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float m = 0.0f;
        for (int j = 0; j < ntaps; ++j) m += alpha[j] * __bfloat162float(taps[(size_t(r) * ntaps + j) * D + c]);
        out[size_t(r) * D + c] = __float2bfloat16(m * inv * w[c]);
    }
}

// grid (n, heads), 128 threads = the head's dims
__global__ void __launch_bounds__(128) head_norm_rope_k(__nv_bfloat16 * x, int ld, const float * w, int pos0, float theta, float eps) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    __nv_bfloat16 * row = x + size_t(t) * ld + size_t(h) * HD;
    __shared__ float v[HD], red[4];
    float a = __bfloat162float(row[d]);
    float ss = warp_sum(a * a);
    if (d % 32 == 0) red[d / 32] = ss;
    __syncthreads();
    a = a * rsqrtf((red[0] + red[1] + red[2] + red[3]) / float(HD) + eps) * w[d];
    v[d] = a;
    __syncthreads();
    const int i = d % (HD / 2);
    const float ang = float(pos0 + t) * powf(theta, -2.0f * float(i) / float(HD));
    float sn, cs;
    sincosf(ang, &sn, &cs);
    const float y = d < HD / 2 ? v[d] * cs - v[d + HD / 2] * sn : v[d] * cs + v[d - HD / 2] * sn;
    row[d] = __float2bfloat16(y);
}

__global__ void store_kv_k(const __nv_bfloat16 * k, const __nv_bfloat16 * v, int ld, __nv_bfloat16 * kc, __nv_bfloat16 * vc, int C, int pos0, int ring) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int pos = pos0 + t;
    const size_t o = size_t(ring ? pos % ring : pos) * C + c;
    kc[o] = k[size_t(t) * ld + c];
    vc[o] = v[size_t(t) * ld + c];
}

// grid (splits, kvheads, nb): keys j in [0, nctx + nb): context key lo + j, then block key j - nctx
template <int QG>
__global__ void __launch_bounds__(128) attn_k(const __nv_bfloat16 * q, const __nv_bfloat16 * kb, const __nv_bfloat16 * vb, int ldq, int ldkv,
                                              const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, int ring, int lo, int nctx, int nb, int heads,
                                              int kvheads, int splits, float * part) {
    const int split = blockIdx.x, g = blockIdx.y, i = blockIdx.z, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int total = nctx + nb, per = (total + splits - 1) / splits, j0 = split * per, j1 = min(total, j0 + per);
    __shared__ float wm[4][QG], wl[4][QG], wacc[4][QG][HD];
    float qv[QG][4], m[QG], l[QG], acc[QG][4];
    for (int h = 0; h < QG; ++h) {
        const __nv_bfloat16 * qr = q + size_t(i) * ldq + size_t(g * QG + h) * HD + lane * 4;
        for (int d = 0; d < 4; ++d) { qv[h][d] = __bfloat162float(qr[d]); acc[h][d] = 0.0f; }
        m[h] = -INFINITY; l[h] = 0.0f;
    }
    const float scale = rsqrtf(float(HD));
    for (int j = j0 + warp; j < j1; j += 4) {
        const __nv_bfloat16 * kr, * vr;
        if (j < nctx) {
            const int key = lo + j;
            const size_t row = (size_t(ring ? key % ring : key) * kvheads + g) * HD + lane * 4;
            kr = kc + row; vr = vc + row;
        } else {
            const size_t row = size_t(j - nctx) * ldkv + size_t(g) * HD + lane * 4;
            kr = kb + row; vr = vb + row;
        }
        float kf[4], vf[4];
        for (int d = 0; d < 4; ++d) { kf[d] = __bfloat162float(kr[d]); vf[d] = __bfloat162float(vr[d]); }
        for (int h = 0; h < QG; ++h) {
            const float s = warp_sum(qv[h][0] * kf[0] + qv[h][1] * kf[1] + qv[h][2] * kf[2] + qv[h][3] * kf[3]) * scale;
            const float mn = fmaxf(m[h], s), c = __expf(m[h] - mn), p = __expf(s - mn);
            l[h] = l[h] * c + p;
            for (int d = 0; d < 4; ++d) acc[h][d] = acc[h][d] * c + p * vf[d];
            m[h] = mn;
        }
    }
    for (int h = 0; h < QG; ++h) {
        if (lane == 0) { wm[warp][h] = m[h]; wl[warp][h] = l[h]; }
        for (int d = 0; d < 4; ++d) wacc[warp][h][lane * 4 + d] = acc[h][d];
    }
    __syncthreads();
    for (int h = 0; h < QG; ++h) {
        const float M = fmaxf(fmaxf(wm[0][h], wm[1][h]), fmaxf(wm[2][h], wm[3][h]));
        float L = 0.0f, A = 0.0f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; ++w) {
                if (wm[w][h] == -INFINITY) continue;
                const float f = __expf(wm[w][h] - M);
                L += wl[w][h] * f;
                A += wacc[w][h][t] * f;
            }
        float * out = part + ((size_t(i) * heads + g * QG + h) * splits + split) * PART;
        if (t == 0) { out[0] = M; out[1] = L; }
        out[2 + t] = A;
    }
}
__global__ void __launch_bounds__(128) attn_merge_k(const float * part, int heads, int splits, __nv_bfloat16 * out) {
    const int h = blockIdx.x, i = blockIdx.y, t = threadIdx.x;
    const float * p = part + (size_t(i) * heads + h) * splits * PART;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, p[s * PART]);
    float L = 0.0f, A = 0.0f;
    for (int s = 0; s < splits; ++s) {
        if (p[s * PART] == -INFINITY) continue;
        const float f = __expf(p[s * PART] - M);
        L += p[s * PART + 1] * f;
        A += p[s * PART + 2 + t] * f;
    }
    out[(size_t(i) * heads + h) * HD + t] = __float2bfloat16(L > 0.0f ? A / L : 0.0f);
}

__global__ void silu_mul_k(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __nv_bfloat16 * a, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float x = __bfloat162float(g[i]);
        a[i] = __float2bfloat16(x / (1.0f + __expf(-x)) * __bfloat162float(u[i]));
    }
}
__global__ void add_k(float * h, const __nv_bfloat16 * x, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) h[i] += __bfloat162float(x[i]);
}
__global__ void rnn_inputs_k(const float * state, const int * token, const __nv_bfloat16 * w1, const __nv_bfloat16 * hidden, const __nv_bfloat16 * f_rnn,
                             __nv_bfloat16 * z, int R, int D) {
    const int tok = *token;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < 2 * R + 2 * D; c += gridDim.x * blockDim.x) {
        __nv_bfloat16 v;
        if (c < R) v = __float2bfloat16(state[c]);
        else if (c < 2 * R) v = w1[size_t(tok) * R + c - R];
        else if (c < 2 * R + D) v = hidden[c - 2 * R];
        else v = f_rnn[c - 2 * R - D];
        z[c] = v;
    }
}
// each op rounded to BF16 as the reference's eager ops (sglang _rnn_gates)
__global__ void rnn_gates_k(const float * proj, const __nv_bfloat16 * bias, float * state, __nv_bfloat16 * latent, int R) {
    for (int c = threadIdx.x; c < R; c += blockDim.x) {
        const float g = bfr(proj[c] + __bfloat162float(bias[c])), cand = bfr(proj[R + c] + __bfloat162float(bias[R + c]));
        const float o = bfr(proj[2 * R + c] + __bfloat162float(bias[2 * R + c]));
        const float st = bfr(state[c]);
        const float gate = bfr(1.0f / (1.0f + expf(-g)));
        const float kept = bfr(gate * st), blend = bfr(bfr(1.0f - gate) * bfr(tanhf(cand)));
        state[c] = bfr(kept + blend);
        latent[c] = __float2bfloat16(tanhf(o));
    }
}
// grid-stride over rows, warp = row (R = 256: one 16-byte load per lane); per block the best (value, lowest index)
constexpr int kStepBlocks = 512;
__global__ void __launch_bounds__(256) step_k(const float * base, const __nv_bfloat16 * fold, const float * fold_b, const __nv_bfloat16 * latent, int V,
                                              int R, float cap, float * partial) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ float bv[8]; __shared__ int bi[8];
    float lat[8];
    {
        const uint4 u = *reinterpret_cast<const uint4 *>(latent + lane * 8);
        const __nv_bfloat162 * b = reinterpret_cast<const __nv_bfloat162 *>(&u);
        for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(b[j]); lat[2 * j] = f.x; lat[2 * j + 1] = f.y; }
    }
    float best = -INFINITY; int idx = 0x7fffffff;
    for (int v = blockIdx.x * 8 + warp; v < V; v += gridDim.x * 8) {
        const uint4 u = __ldcs(reinterpret_cast<const uint4 *>(fold + size_t(v) * R) + lane);
        const __nv_bfloat162 * b = reinterpret_cast<const __nv_bfloat162 *>(&u);
        float d = 0.0f;
        for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(b[j]); d += f.x * lat[2 * j] + f.y * lat[2 * j + 1]; }
        d = warp_sum(d);
        const float val = cap * tanhf((base[v] + fold_b[v] + d) / cap);
        if (val > best || (val == best && v < idx)) { best = val; idx = v; }
    }
    if (lane == 0) { bv[warp] = best; bi[warp] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; ++w) if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
        partial[blockIdx.x * 2] = best;
        partial[blockIdx.x * 2 + 1] = __int_as_float(idx);
    }
}
__global__ void step_reduce_k(const float * partial, int n, int * token) {
    float best = -INFINITY; int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 32) {
        const float v = partial[2 * i]; const int x = __float_as_int(partial[2 * i + 1]);
        if (v > best || (v == best && x < idx)) { best = v; idx = x; }
    }
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o); const int ox = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > best || (ov == best && ox < idx)) { best = ov; idx = ox; }
    }
    if (threadIdx.x == 0) *token = idx;
}
__global__ void block_init_k(const float * src, const __nv_bfloat16 * mask, float * h, int nb, int D) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    h[size_t(r) * D + c] = r == 0 ? src[c] : __bfloat162float(mask[c]);
}
__global__ void tap_k(const float * x, int D, __nv_bfloat16 * taps, int j, int ntaps) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < D) taps[(size_t(t) * ntaps + j) * D + c] = __float2bfloat16(x[size_t(t) * D + c]);
}
int grid_of(size_t n) { return int(n < 256 * 4096 ? (n + 255) / 256 : 4096); }
}  // namespace

void mtp_mix_norm(const __nv_bfloat16 * taps, int ntaps, const float * alpha, const float * w, __nv_bfloat16 * out, int m, int D, float eps, cudaStream_t s) {
    if (m > 0) mix_norm_k<<<m, 256, 0, s>>>(taps, ntaps, alpha, w, out, D, eps);
}
void mtp_head_norm_rope(__nv_bfloat16 * x, int ld, int n, int heads, const float * w, int pos0, float theta, float eps, cudaStream_t s) {
    if (n > 0) head_norm_rope_k<<<dim3(n, heads), HD, 0, s>>>(x, ld, w, pos0, theta, eps);
}
void mtp_store_kv(const __nv_bfloat16 * k, const __nv_bfloat16 * v, int ldkv, __nv_bfloat16 * kc, __nv_bfloat16 * vc, int n, int kvheads, int pos0, int ring,
                  cudaStream_t s) {
    const int C = kvheads * HD;
    if (n > 0) store_kv_k<<<dim3((C + 255) / 256, n), 256, 0, s>>>(k, v, ldkv, kc, vc, C, pos0, ring);
}
int mtp_attention_floats(int heads, int nb, int splits) { return heads * nb * splits * PART; }
void mtp_attention(const __nv_bfloat16 * q, const __nv_bfloat16 * kb, const __nv_bfloat16 * vb, int ldq, int ldkv, const __nv_bfloat16 * kc,
                   const __nv_bfloat16 * vc, int ring, int pos0, int nb, int heads, int kvheads, int window, int splits, float * part,
                   __nv_bfloat16 * out, cudaStream_t s) {
    const int lo = window > 0 ? (pos0 > window ? pos0 - window : 0) : 0, nctx = pos0 - lo;
    if (heads == 3 * kvheads) attn_k<3><<<dim3(splits, kvheads, nb), 128, 0, s>>>(q, kb, vb, ldq, ldkv, kc, vc, ring, lo, nctx, nb, heads, kvheads, splits, part);
    else if (heads == 4 * kvheads) attn_k<4><<<dim3(splits, kvheads, nb), 128, 0, s>>>(q, kb, vb, ldq, ldkv, kc, vc, ring, lo, nctx, nb, heads, kvheads, splits, part);  // DFlash2
    else throw std::runtime_error("mtp attention: 3 or 4 query heads per kv head");
    attn_merge_k<<<dim3(heads, nb), HD, 0, s>>>(part, heads, splits, out);
}
void mtp_silu_mul(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __nv_bfloat16 * act, size_t n, cudaStream_t s) {
    if (n) silu_mul_k<<<grid_of(n), 256, 0, s>>>(g, u, act, n);
}
void mtp_add(float * h, const __nv_bfloat16 * x, size_t n, cudaStream_t s) { if (n) add_k<<<grid_of(n), 256, 0, s>>>(h, x, n); }
void mtp_rnn_inputs(const float * state, const int * token, const __nv_bfloat16 * w1, const __nv_bfloat16 * hidden, const __nv_bfloat16 * f_rnn,
                    __nv_bfloat16 * z, int R, int D, cudaStream_t s) {
    rnn_inputs_k<<<grid_of(size_t(2 * R + 2 * D)), 256, 0, s>>>(state, token, w1, hidden, f_rnn, z, R, D);
}
void mtp_rnn_gates(const float * proj, const __nv_bfloat16 * bias, float * state, __nv_bfloat16 * latent, int R, cudaStream_t s) {
    rnn_gates_k<<<1, 256, 0, s>>>(proj, bias, state, latent, R);
}
int mtp_step_floats(int) { return 2 * kStepBlocks; }
void mtp_step_argmax(const float * base, const __nv_bfloat16 * fold, const float * fold_b, const __nv_bfloat16 * latent, int V, int R, float cap,
                     float * partial, int * token, cudaStream_t s) {
    if (R != 256) throw std::runtime_error("mtp step: markov rank 256");
    step_k<<<kStepBlocks, 256, 0, s>>>(base, fold, fold_b, latent, V, R, cap, partial);
    step_reduce_k<<<1, 32, 0, s>>>(partial, kStepBlocks, token);
}
void mtp_tap(const float * x, int n, int D, __nv_bfloat16 * taps, int j, int ntaps, cudaStream_t s) {
    if (n > 0) tap_k<<<dim3((D + 255) / 256, n), 256, 0, s>>>(x, D, taps, j, ntaps);
}
void mtp_block_init(const float * src, const __nv_bfloat16 * mask, float * h, int nb, int D, cudaStream_t s) {
    block_init_k<<<dim3((D + 255) / 256, nb), 256, 0, s>>>(src, mask, h, nb, D);
}

}  // namespace tqllm_cuda::inkling2

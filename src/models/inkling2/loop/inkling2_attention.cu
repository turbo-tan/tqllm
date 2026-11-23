// Inkling-2-Mini attention kernels (see inkling2_attention.cuh). Split-K over the key range: one block per (query, head,
// split of 256 keys), 4 warps taking every 4th key, each lane 4 of the 128 dims, online softmax in F32; a merge kernel
// adds the sink column, normalises and applies the sigmoid output gate. The split count only depends on the span a query
// can see (the window, or the context), so a windowed layer costs the same at any context length.
#include "inkling2_attention.cuh"

#include <algorithm>
#include <cmath>

namespace tqllm_cuda {
namespace {
constexpr int HD = 128, DR = 16, CHUNK = 256, PART = HD + 2;

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float tau(float alpha, float floor_, int pos) {
    return 1.0f + alpha * logf(fmaxf(float(pos + 1) / floor_, 1.0f));
}
int span_of(int window, int context) { return window > 0 ? std::min(window, context) : context; }
int splits_of(int window, int context) { return std::max(1, (span_of(window, context) + CHUNK - 1) / CHUNK); }

// MODE 0: F32 scores, online softmax. MODE 1 / 2 (hf_bf16): the reference's eager BF16 arithmetic: q.k, the scaling and
// the relative bias each rounded to BF16; pass 1 the softmax statistics, pass 2 BF16-rounded probabilities times v.
template <int MODE>
__global__ void __launch_bounds__(128) attn_split_k(Inkling2AttnArgs a, int nsplit, const float * stats) {
    const int split = blockIdx.x, h = blockIdx.y, i = blockIdx.z, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int pos = a.pos0 + i, g = h / (a.heads / a.kvheads);
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int k0 = lo + split * CHUNK, k1 = min(pos + 1, k0 + CHUNK);
    __shared__ float qs[HD], rsv[DR], bias[CHUNK], red[4], wm[4], wl[4], wacc[4][HD];
    // q: scale, L2 clip, (log scale), each step rounded to BF16 as the reference stores it
    float x = __bfloat162float(a.q[(size_t(i) * a.heads + h) * HD + t]) * __bfloat162float(a.q_scale[h * HD + t]);
    float ss = warp_sum(x * x);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    const float qn = sqrtf(red[0] + red[1] + red[2] + red[3]);
    x = bf16r(x * fminf(1.0f, a.qk_clip / (qn + 1e-6f)));
    const float tp = tau(a.log_alpha, a.log_floor, pos);
    if (a.log_scale) x = bf16r(x * tp);
    qs[t] = x;
    if (warp == 0) {  // relative state: 16 lanes
        float r = lane < DR ? __bfloat162float(a.rs[(size_t(i) * a.heads + h) * DR + lane]) * __bfloat162float(a.rel_scale[h * DR + lane]) : 0.0f;
        const float rn = sqrtf(warp_sum(r * r));
        r = bf16r(r * fminf(1.0f, a.rel_clip / (rn + 1e-6f)));
        if (a.log_scale) r = bf16r(r * tp);
        if (lane < DR) rsv[lane] = r;
    }
    __syncthreads();
    for (int j = t; j < CHUNK; j += 128) {  // relative bias of this split's keys
        const int d = pos - (k0 + j);
        float b = 0.0f;
        if (k0 + j < k1 && d < a.rel_extent) {
            for (int r = 0; r < DR; ++r) b += rsv[r] * __bfloat162float(a.rel_proj[size_t(r) * a.rel_extent + d]);
            if (MODE) b = bf16r(b);
        }
        bias[j] = b;
    }
    __syncthreads();
    const float q0 = qs[lane * 4], q1 = qs[lane * 4 + 1], q2 = qs[lane * 4 + 2], q3 = qs[lane * 4 + 3];
    const float scale = rsqrtf(float(HD));
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float sM = MODE == 2 ? stats[(size_t(i) * a.heads + h) * 2] : 0.0f, sL = MODE == 2 ? stats[(size_t(i) * a.heads + h) * 2 + 1] : 1.0f;
    for (int key = k0 + warp; key < k1; key += 4) {
        const size_t row = (size_t(a.ring ? key % a.ring : key) * a.kvheads + g) * HD + lane * 4;
        const __nv_bfloat162 * kp = reinterpret_cast<const __nv_bfloat162 *>(a.kc + row);
        const float2 ka = __bfloat1622float2(kp[0]), kb = __bfloat1622float2(kp[1]);
        const float dot = warp_sum(q0 * ka.x + q1 * ka.y + q2 * kb.x + q3 * kb.y);
        const float s = MODE ? bf16r(bf16r(bf16r(dot) * scale) + bias[key - k0]) : dot * scale + bias[key - k0];
        const __nv_bfloat162 * vp = reinterpret_cast<const __nv_bfloat162 *>(a.vc + row);
        const float2 va = __bfloat1622float2(vp[0]), vb = __bfloat1622float2(vp[1]);
        if (MODE == 2) {
            const float p = bf16r(expf(s - sM) / sL);
            acc[0] += p * va.x; acc[1] += p * va.y; acc[2] += p * vb.x; acc[3] += p * vb.y;
            continue;
        }
        const float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
        acc[0] = acc[0] * c + p * va.x; acc[1] = acc[1] * c + p * va.y;
        acc[2] = acc[2] * c + p * vb.x; acc[3] = acc[3] * c + p * vb.y;
        m = mn;
    }
    if (MODE == 2) m = 0.0f;  // plain sums: every warp at the same reference point
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    for (int j = 0; j < 4; ++j) wacc[warp][lane * 4 + j] = acc[j];
    __syncthreads();
    const float M = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
    float L = 0.0f, A = 0.0f;
    if (M != -INFINITY)
        for (int w = 0; w < 4; ++w) {
            if (wm[w] == -INFINITY) continue;
            const float f = __expf(wm[w] - M);
            L += wl[w] * f;
            A += wacc[w][t] * f;
        }
    float * out = a.part + ((size_t(i) * a.heads + h) * nsplit + split) * PART;
    if (t == 0) { out[0] = M; out[1] = L; }
    out[2 + t] = A;
}

__global__ void __launch_bounds__(128) attn_merge_k(Inkling2AttnArgs a, int nsplit) {
    const int h = blockIdx.x, i = blockIdx.y, t = threadIdx.x;
    const float * p = a.part + (size_t(i) * a.heads + h) * nsplit * PART;
    const float sink = __bfloat162float(a.sink[size_t(i) * a.heads + h]);
    float M = sink;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, p[s * PART]);
    float L = __expf(sink - M), A = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float ms = p[s * PART];
        if (ms == -INFINITY) continue;
        const float f = __expf(ms - M);
        L += p[s * PART + 1] * f;
        A += p[s * PART + 2 + t] * f;
    }
    const size_t o = (size_t(i) * a.heads + h) * HD + t;
    const float gt = __bfloat162float(a.gate[o]);
    a.out[o] = __float2bfloat16(A / L * (1.0f / (1.0f + __expf(-gt))));
}

// hf_bf16 pass 1 -> per (query, head): M = max(sink, split maxima), L = exp(sink - M) + sum of the split sums
__global__ void attn_stats_k(Inkling2AttnArgs a, int nsplit, float * stats) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.nq * a.heads) return;
    const float * p = a.part + size_t(idx) * nsplit * PART;
    const float sink = __bfloat162float(a.sink[idx]);
    float M = sink;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, p[s * PART]);
    float L = expf(sink - M);
    for (int s = 0; s < nsplit; ++s) if (p[s * PART] != -INFINITY) L += p[s * PART + 1] * expf(p[s * PART] - M);
    stats[idx * 2] = M; stats[idx * 2 + 1] = L;
}
// hf_bf16 pass 2 -> out = bf16(bf16(sum of the splits' p.v) * sigmoid(gate))
__global__ void __launch_bounds__(128) attn_merge_hf_k(Inkling2AttnArgs a, int nsplit) {
    const int h = blockIdx.x, i = blockIdx.y, t = threadIdx.x;
    const float * p = a.part + (size_t(i) * a.heads + h) * nsplit * PART;
    float A = 0.0f;
    for (int s = 0; s < nsplit; ++s) A += p[s * PART + 2 + t];
    const size_t o = (size_t(i) * a.heads + h) * HD + t;
    const float gt = __bfloat162float(a.gate[o]);
    a.out[o] = __float2bfloat16(bf16r(A) * (1.0f / (1.0f + expf(-gt))));
}

__global__ void __launch_bounds__(128) kv_finish_k(const __nv_bfloat16 * k, const __nv_bfloat16 * v, __nv_bfloat16 * kc, __nv_bfloat16 * vc,
                                                   int kvheads, float clip, float eps, const int * dpos, int ring) {
    const int t = blockIdx.x / kvheads, g = blockIdx.x % kvheads;
    const size_t i = size_t(blockIdx.x) * HD + threadIdx.x;
    const int pos = (dpos ? *dpos : 0) + t;
    const size_t o = (size_t(ring ? pos % ring : pos) * kvheads + g) * HD + threadIdx.x;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __shared__ float rk[4], rv[4];
    const float kx = __bfloat162float(k[i]), vx = __bfloat162float(v[i]);
    const float sk = warp_sum(kx * kx), sv = warp_sum(vx * vx);
    if (lane == 0) { rk[warp] = sk; rv[warp] = sv; }
    __syncthreads();
    const float kn = sqrtf(rk[0] + rk[1] + rk[2] + rk[3]), vm = (rv[0] + rv[1] + rv[2] + rv[3]) / float(HD);
    kc[o] = __float2bfloat16(kx * fminf(1.0f, clip / (kn + 1e-6f)));
    vc[o] = __float2bfloat16(vx * rsqrtf(vm + eps));
}

// decode / verify: block (split, kv head g, query i), warp w < QG prepares query head g * QG + w; every warp then takes
// every 4th key of the split and scores it against all QG heads (one K / V read per group)
template <int QG>
__global__ void __launch_bounds__(128) attn_decode_k(Inkling2AttnArgs a) {
    const int split = blockIdx.x, g = blockIdx.y, i = blockIdx.z, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int pos = *a.dpos + i;
    const int span = a.window > 0 ? min(a.window, pos + 1) : pos + 1, lo = pos + 1 - span;
    const int per = (span + a.splits - 1) / a.splits, k0 = lo + split * per, k1 = min(pos + 1, k0 + per);
    __shared__ float qs[QG][HD], rsv[QG][DR], wm[4][QG], wl[4][QG], wacc[4][QG][HD];
    const float tp = tau(a.log_alpha, a.log_floor, pos);
    if (warp < QG) {
        const int h = g * QG + warp;
        float x[4], ss = 0.0f;
        for (int j = 0; j < 4; ++j) {
            x[j] = __bfloat162float(a.q[(size_t(i) * a.heads + h) * HD + lane * 4 + j]) * __bfloat162float(a.q_scale[h * HD + lane * 4 + j]);
            ss += x[j] * x[j];
        }
        const float f = fminf(1.0f, a.qk_clip / (sqrtf(warp_sum(ss)) + 1e-6f));
        for (int j = 0; j < 4; ++j) {
            float y = bf16r(x[j] * f);
            if (a.log_scale) y = bf16r(y * tp);
            qs[warp][lane * 4 + j] = y;
        }
        float r = lane < DR ? __bfloat162float(a.rs[(size_t(i) * a.heads + h) * DR + lane]) * __bfloat162float(a.rel_scale[h * DR + lane]) : 0.0f;
        const float rn = sqrtf(warp_sum(r * r));
        r = bf16r(r * fminf(1.0f, a.rel_clip / (rn + 1e-6f)));
        if (a.log_scale) r = bf16r(r * tp);
        if (lane < DR) rsv[warp][lane] = r;
    }
    __syncthreads();
    float q[QG][4], m[QG], l[QG], acc[QG][4];
    for (int j = 0; j < QG; ++j) {
        for (int d = 0; d < 4; ++d) { q[j][d] = qs[j][lane * 4 + d]; acc[j][d] = 0.0f; }
        m[j] = -INFINITY; l[j] = 0.0f;
    }
    const float scale = rsqrtf(float(HD));
    const int hh = lane >> 4, r = lane & 15;
    for (int key = k0 + warp; key < k1; key += 4) {
        const size_t row = (size_t(a.ring ? key % a.ring : key) * a.kvheads + g) * HD + lane * 4;
        const __nv_bfloat162 * kp = reinterpret_cast<const __nv_bfloat162 *>(a.kc + row);
        const __nv_bfloat162 * vp = reinterpret_cast<const __nv_bfloat162 *>(a.vc + row);
        const float2 ka = __bfloat1622float2(kp[0]), kb = __bfloat1622float2(kp[1]);
        const float2 va = __bfloat1622float2(vp[0]), vb = __bfloat1622float2(vp[1]);
        const int d = pos - key;
        float bias[QG];
        if (d < a.rel_extent) {  // lanes 0-15 / 16-31: one head each, 16 relative dims
            const float pr = __bfloat162float(a.rel_proj[size_t(r) * a.rel_extent + d]);
            for (int j0 = 0; j0 < QG; j0 += 2) {
                float b = j0 + hh < QG ? rsv[j0 + hh][r] * pr : 0.0f;
                for (int o = 8; o; o >>= 1) b += __shfl_xor_sync(0xffffffffu, b, o);
                bias[j0] = __shfl_sync(0xffffffffu, b, 0);
                if (j0 + 1 < QG) bias[j0 + 1] = __shfl_sync(0xffffffffu, b, 16);
            }
        } else {
            for (int j = 0; j < QG; ++j) bias[j] = 0.0f;
        }
        for (int j = 0; j < QG; ++j) {
            const float s = warp_sum(q[j][0] * ka.x + q[j][1] * ka.y + q[j][2] * kb.x + q[j][3] * kb.y) * scale + bias[j];
            const float mn = fmaxf(m[j], s), c = __expf(m[j] - mn), p = __expf(s - mn);
            l[j] = l[j] * c + p;
            acc[j][0] = acc[j][0] * c + p * va.x; acc[j][1] = acc[j][1] * c + p * va.y;
            acc[j][2] = acc[j][2] * c + p * vb.x; acc[j][3] = acc[j][3] * c + p * vb.y;
            m[j] = mn;
        }
    }
    for (int j = 0; j < QG; ++j) {
        if (lane == 0) { wm[warp][j] = m[j]; wl[warp][j] = l[j]; }
        for (int d = 0; d < 4; ++d) wacc[warp][j][lane * 4 + d] = acc[j][d];
    }
    __syncthreads();
    for (int j = 0; j < QG; ++j) {
        const float M = fmaxf(fmaxf(wm[0][j], wm[1][j]), fmaxf(wm[2][j], wm[3][j]));
        float L = 0.0f, A = 0.0f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; ++w) {
                if (wm[w][j] == -INFINITY) continue;
                const float f = __expf(wm[w][j] - M);
                L += wl[w][j] * f;
                A += wacc[w][j][t] * f;
            }
        float * out = a.part + ((size_t(i) * a.heads + g * QG + j) * a.splits + split) * PART;
        if (t == 0) { out[0] = M; out[1] = L; }
        out[2 + t] = A;
    }
}
}  // namespace

int inkling2_attention_partial_floats(int heads, int nq, int context) {
    // the largest span any of the nq queries sees: the window, or the whole context
    return heads * nq * splits_of(0, context) * PART;
}

void inkling2_attention(const Inkling2AttnArgs & a, cudaStream_t s) {
    if (a.nq <= 0) return;
    const int nsplit = splits_of(a.window, a.pos0 + a.nq);
    const dim3 grid(nsplit, a.heads, a.nq);
    if (!a.hf_bf16) {
        attn_split_k<0><<<grid, 128, 0, s>>>(a, nsplit, nullptr);
        attn_merge_k<<<dim3(a.heads, a.nq), 128, 0, s>>>(a, nsplit);
        return;
    }
    attn_split_k<1><<<grid, 128, 0, s>>>(a, nsplit, nullptr);
    attn_stats_k<<<(a.nq * a.heads + 127) / 128, 128, 0, s>>>(a, nsplit, a.stats);
    attn_split_k<2><<<grid, 128, 0, s>>>(a, nsplit, a.stats);
    attn_merge_hf_k<<<dim3(a.heads, a.nq), 128, 0, s>>>(a, nsplit);
}

int inkling2_attention_decode_floats(int heads, int nq, int splits) { return heads * nq * splits * PART; }

void inkling2_attention_decode(const Inkling2AttnArgs & a, cudaStream_t s) {
    if (a.nq <= 0) return;
    const dim3 grid(a.splits, a.kvheads, a.nq);
    switch (a.heads / a.kvheads) {
        case 1: attn_decode_k<1><<<grid, 128, 0, s>>>(a); break;
        case 2: attn_decode_k<2><<<grid, 128, 0, s>>>(a); break;
        case 3: attn_decode_k<3><<<grid, 128, 0, s>>>(a); break;
        case 4: attn_decode_k<4><<<grid, 128, 0, s>>>(a); break;
        default: return;
    }
    attn_merge_k<<<dim3(a.heads, a.nq), 128, 0, s>>>(a, a.splits);
}

void inkling2_kv_finish(const __nv_bfloat16 * k, const __nv_bfloat16 * v, __nv_bfloat16 * kc, __nv_bfloat16 * vc, int n, int kvheads,
                        float qk_clip, float eps, const int * dpos, int ring, cudaStream_t s) {
    if (n <= 0) return;
    kv_finish_k<<<n * kvheads, 128, 0, s>>>(k, v, kc, vc, kvheads, qk_clip, eps, dpos, ring);
}

}  // namespace tqllm_cuda

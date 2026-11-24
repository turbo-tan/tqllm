// Inkling-2-Mini attention kernels (see inkling2_attention.cuh). Split-K over the key range: one block per (query, head,
// split of 256 keys), 4 warps taking every 4th key, each lane 4 of the 128 dims, online softmax in F32; a merge kernel
// adds the sink column, normalises and applies the sigmoid output gate. The split count only depends on the span a query
// can see (the window, or the context), so a windowed layer costs the same at any context length.
#include "inkling2_attention.cuh"
#include "inkling2_attention_body.cuh"

#include <algorithm>
#include <cmath>

namespace tqllm_cuda {
namespace {
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

__global__ void __launch_bounds__(128) attn_merge_k(Inkling2AttnArgs a, int nsplit) { attn_merge_body(inkling2::blk_of_launch(), a, nsplit); }

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
    __shared__ float sm[8];
    kv_finish_body(inkling2::blk_of_launch(), k, v, kc, vc, kvheads, clip, eps, dpos, ring, sm);
}

template <int QG>
__global__ void __launch_bounds__(128) attn_decode_k(Inkling2AttnArgs a) {
    __shared__ AttnDecodeSmem<QG> S;
    attn_decode_body<QG>(inkling2::blk_of_launch(), a, S);
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

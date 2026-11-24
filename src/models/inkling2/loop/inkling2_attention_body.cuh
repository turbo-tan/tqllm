#pragma once
// Inkling-2-Mini decode attention bodies (the inkling2_attention.cu kernels; shared with fused kernels)
#include "inkling2_attention.cuh"
#include "inkling2_kernels.cuh"

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

// block of 128 (b.x = token * kvheads + kv head)
__device__ __forceinline__ void kv_finish_body(const inkling2::Blk & b, const __nv_bfloat16 * k, const __nv_bfloat16 * v, __nv_bfloat16 * kc,
                                               __nv_bfloat16 * vc, int kvheads, float clip, float eps, const int * dpos, int ring, float * sm) {
    const int t = b.x / kvheads, g = b.x % kvheads;
    const size_t i = size_t(b.x) * HD + b.t;
    const int pos = (dpos ? *dpos : 0) + t;
    const size_t o = (size_t(ring ? pos % ring : pos) * kvheads + g) * HD + b.t;
    const int lane = b.t % 32, warp = b.t / 32;
    float * rk = sm, * rv = sm + 4;
    const float kx = __bfloat162float(k[i]), vx = __bfloat162float(v[i]);
    const float sk = warp_sum(kx * kx), sv = warp_sum(vx * vx);
    if (lane == 0) { rk[warp] = sk; rv[warp] = sv; }
    b.sync();
    const float kn = sqrtf(rk[0] + rk[1] + rk[2] + rk[3]), vm = (rv[0] + rv[1] + rv[2] + rv[3]) / float(HD);
    kc[o] = __float2bfloat16(kx * fminf(1.0f, clip / (kn + 1e-6f)));
    vc[o] = __float2bfloat16(vx * rsqrtf(vm + eps));
}

template <int QG> struct AttnDecodeSmem { float qs[QG][HD], rsv[QG][DR], wm[4][QG], wl[4][QG], wacc[4][QG][HD]; };

// decode / verify: block (split, kv head g, query i) of 128, warp w < QG prepares query head g * QG + w; every warp then
// takes every 4th key of the split and scores it against all QG heads (one K / V read per group)
template <int QG>
__device__ __forceinline__ void attn_decode_body(const inkling2::Blk & b, const Inkling2AttnArgs & a, AttnDecodeSmem<QG> & S) {
    const int split = b.x, g = b.y, i = b.z, t = b.t, lane = t % 32, warp = t / 32;
    const int pos = *a.dpos + i;
    const int span = a.window > 0 ? min(a.window, pos + 1) : pos + 1, lo = pos + 1 - span;
    const int per = (span + a.splits - 1) / a.splits, k0 = lo + split * per, k1 = min(pos + 1, k0 + per);
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
            S.qs[warp][lane * 4 + j] = y;
        }
        float r = lane < DR ? __bfloat162float(a.rs[(size_t(i) * a.heads + h) * DR + lane]) * __bfloat162float(a.rel_scale[h * DR + lane]) : 0.0f;
        const float rn = sqrtf(warp_sum(r * r));
        r = bf16r(r * fminf(1.0f, a.rel_clip / (rn + 1e-6f)));
        if (a.log_scale) r = bf16r(r * tp);
        if (lane < DR) S.rsv[warp][lane] = r;
    }
    b.sync();
    float q[QG][4], m[QG], l[QG], acc[QG][4];
    for (int j = 0; j < QG; ++j) {
        for (int d = 0; d < 4; ++d) { q[j][d] = S.qs[j][lane * 4 + d]; acc[j][d] = 0.0f; }
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
                float bb = j0 + hh < QG ? S.rsv[j0 + hh][r] * pr : 0.0f;
                for (int o = 8; o; o >>= 1) bb += __shfl_xor_sync(0xffffffffu, bb, o);
                bias[j0] = __shfl_sync(0xffffffffu, bb, 0);
                if (j0 + 1 < QG) bias[j0 + 1] = __shfl_sync(0xffffffffu, bb, 16);
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
        if (lane == 0) { S.wm[warp][j] = m[j]; S.wl[warp][j] = l[j]; }
        for (int d = 0; d < 4; ++d) S.wacc[warp][j][lane * 4 + d] = acc[j][d];
    }
    b.sync();
    for (int j = 0; j < QG; ++j) {
        const float M = fmaxf(fmaxf(S.wm[0][j], S.wm[1][j]), fmaxf(S.wm[2][j], S.wm[3][j]));
        float L = 0.0f, A = 0.0f;
        if (M != -INFINITY)
            for (int w = 0; w < 4; ++w) {
                if (S.wm[w][j] == -INFINITY) continue;
                const float f = __expf(S.wm[w][j] - M);
                L += S.wl[w][j] * f;
                A += S.wacc[w][j][t] * f;
            }
        float * out = a.part + ((size_t(i) * a.heads + g * QG + j) * a.splits + split) * PART;
        if (t == 0) { out[0] = M; out[1] = L; }
        out[2 + t] = A;
    }
}

// block (head, query) of 128: the splits' partials, the sink column, the sigmoid gate
__device__ __forceinline__ void attn_merge_body(const inkling2::Blk & b, const Inkling2AttnArgs & a, int nsplit) {
    const int h = b.x, i = b.y, t = b.t;
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
}  // namespace
}  // namespace tqllm_cuda

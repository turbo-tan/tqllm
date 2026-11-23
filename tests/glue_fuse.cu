// Fused decode glue launches against the launch sequences they replace, at the Flash-Next decode / verify shapes (1-8 rows):
// - hc_down_inject_bf16x == fn_gemv_bf16x_tokens(down) + hc_low_silu_bf16 + fn_gemv_bf16x_tokens(inject)
// - hc_up_mix_bf16 == fn_gemv_bf16_tokens(up) + hc_mix + the FP16 / BF16 conversions of the mixed rows
// - fn_gemv_bf16_segments(alpha, beta) == two fn_gemv_bf16_tokens (the GDN alpha / beta projections)
// - fn_gdn_prep_gates == fn_gdn_prep_k + fn_gdn_gates_k; ple_conv == ple_conv_k + ple_history_k (verbatim copies below)
// - fn_gated_rms / fn_attention_gate / fn_swiglu_f32 with their FP16 image == the old kernels + a conversion
// - fn_moe_hc_combine_to == fn_moe_combine + hc_combine
// - fn_attention_prep == fn_norm_rope (q) + fn_norm_rope (k) + f32_to_bf16 of k and v
// Every output is compared byte for byte.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace ref {
__device__ __forceinline__ float ldf(float v) { return v; }
__device__ __forceinline__ float ldf(__half v) { return __half2float(v); }
template<class T> __global__ void ple_conv_k(const T * x, const float * history, const float * kernel, float * out,
                           int channels, int n, int taps, int dilation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels * n) return;
    const int token = i / channels, channel = i % channels, hist = (taps - 1) * dilation;
    float sum = 0.f;
    for (int tap = 0; tap < taps; ++tap) {
        const int position = token + tap * dilation;
        const float value = position < hist ? history[int64_t(position) * channels + channel]
                                            : ldf(x[int64_t(position - hist) * channels + channel]);
        sum = __fadd_rn(sum, __fmul_rn(value, kernel[channel * taps + tap]));
    }
    out[i] = sum / (1.f + expf(-sum));
}
template<class T> __global__ void ple_history_k(const T * x, const float * history, float * next,
                              int channels, int n, int hist) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hist * channels) return;
    const int position = n + i / channels, channel = i % channels;
    next[i] = position < hist ? history[int64_t(position) * channels + channel]
                              : ldf(x[int64_t(position - hist) * channels + channel]);
}
__global__ void fn_gdn_prep_k(const float * qkv, float * qk, int kh, int vh, int dim, float eps) {
    const int t = blockIdx.x, key_head = blockIdx.y, lane = threadIdx.x, channels = (2 * kh + vh) * dim, pieces = dim / 32;
    const float * row = qkv + int64_t(t) * channels;
    float q[8], k[8], qq = 0.f, kk = 0.f;
    for (int r = 0; r < pieces; ++r) {
        const int key = r * 32 + lane;
        q[r] = row[key_head * dim + key]; k[r] = row[kh * dim + key_head * dim + key];
        qq += q[r] * q[r]; kk += k[r] * k[r];
    }
    for (int delta = 16; delta; delta >>= 1) {
        qq += __shfl_xor_sync(0xffffffffu, qq, delta);
        kk += __shfl_xor_sync(0xffffffffu, kk, delta);
    }
    const float qscale = rsqrtf(fmaxf(qq, eps * eps)) / sqrtf(float(dim));
    const float kscale = rsqrtf(fmaxf(kk, eps * eps));
    float * qo = qk + (int64_t(t) * 2 * kh + key_head) * dim, * ko = qk + (int64_t(t) * 2 * kh + kh + key_head) * dim;
    for (int r = 0; r < pieces; ++r) { q[r] *= qscale; k[r] *= kscale; qo[r * 32 + lane] = q[r]; ko[r * 32 + lane] = k[r]; }
}
__global__ void fn_gdn_gates_k(const float * alpha, const float * beta, const float * neg_a, const float * bias, float * gates, int vh, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * vh) return;
    const int head = i % vh;
    const float a = alpha[i] + bias[head];
    const float softplus = fmaxf(a, 0.f) + log1pf(expf(-fabsf(a)));
    gates[2 * i] = expf(neg_a[head] * softplus);
    gates[2 * i + 1] = 1.f / (1.f + expf(-beta[i]));
}
__global__ void fn_gated_rms_k(const float * x, const float * gate, const float * weight, float * out,
                               int dim, float eps) {
    const int row = blockIdx.x, lane = threadIdx.x;
    float squared = 0.f;
    for (int i = lane; i < dim; i += 32) squared += x[int64_t(row) * dim + i] * x[int64_t(row) * dim + i];
    for (int delta = 16; delta; delta >>= 1) squared += __shfl_xor_sync(0xffffffffu, squared, delta);
    const float inv = rsqrtf(squared / dim + eps);
    for (int i = lane; i < dim; i += 32) {
        const int64_t at = int64_t(row) * dim + i;
        out[at] = __fmul_rn(__fmul_rn(__fmul_rn(x[at], inv), weight[i]), 1.f / (1.f + expf(-gate[at])));
    }
}
__global__ void fn_attention_gate_k(float * out, const float * qg, int dim, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] *= 1.f / (1.f + expf(-qg[(int64_t(i) / dim) * 2 * dim + dim + i % dim]));
}
__global__ void fn_swiglu_f32_k(const float * g, const float * u, float * o, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= n) return;
    const float a = g[i]; o[i] = a / (1.f + expf(-a)) * u[i];
}
__global__ void to_h_k(const float * x, __half * y, int64_t n) {  // vis_f32_to_h
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}
void to_h(const float * x, __half * y, int64_t n) { to_h_k<<<unsigned((n + 255) / 256), 256>>>(x, y, n); }
__global__ void to_bf16_k(const float * x, __nv_bfloat16 * y, int64_t n) {  // f32_to_bf16
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}
void to_bf16(const float * x, __nv_bfloat16 * y, int64_t n) { to_bf16_k<<<unsigned((n + 255) / 256), 256>>>(x, y, n); }
}  // namespace ref

static std::mt19937 rng(11);
template<class T> static T * dev(size_t n) { T * p = nullptr; cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)); cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)); return p; }
static float * upload(size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f); std::vector<float> h(n); for (auto & v : h) v = scale * nd(rng);
    float * d = dev<float>(n); cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice); return d;
}
static __nv_bfloat16 * upload_bf16(size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f); std::vector<__nv_bfloat16> h(n); for (auto & v : h) v = __float2bfloat16(scale * nd(rng));
    __nv_bfloat16 * d = dev<__nv_bfloat16>(n); cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice); return d;
}
static __half * upload_h(size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, 1.f); std::vector<__half> h(n); for (auto & v : h) v = __float2half(scale * nd(rng));
    __half * d = dev<__half>(n); cudaMemcpy(d, h.data(), n * 2, cudaMemcpyHostToDevice); return d;
}
static bool same(const void * a, const void * b, size_t bytes) {
    std::vector<char> x(bytes), y(bytes);
    cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost);
    return std::memcmp(x.data(), y.data(), bytes) == 0;
}
static int bad = 0, cases = 0;
static void check(bool ok, const char * what, int a, int b) { ++cases; if (!ok) { ++bad; std::printf("FAIL %s (%d, %d)\n", what, a, b); } }

int main() {
    const int h = 2560, c = 4, width = h * c;
    for (int lr : {320, 128}) for (int n = 1; n <= 8; ++n) {  // hyper-connection mixer: down + silu + inject, then up + mix
        __nv_bfloat16 * wd = upload_bf16(size_t(lr) * width, .02f), * wi = upload_bf16(size_t(c) * width, .02f), * wu = upload_bf16(size_t(width) * lr, .05f);
        __nv_bfloat16 * xb = upload_bf16(size_t(n) * width, 1.f);
        float * lowA = dev<float>(size_t(n) * lr), * lowB = dev<float>(size_t(n) * lr), * injA = dev<float>(size_t(n) * c), * injB = dev<float>(size_t(n) * c);
        __nv_bfloat16 * lbA = dev<__nv_bfloat16>(size_t(n) * lr), * lbB = dev<__nv_bfloat16>(size_t(n) * lr);
        if (!tqllm_cuda::fn_gemv_bf16x_tokens(wd, xb, lowA, lr, width, n, nullptr) || !tqllm_cuda::fn_gemv_bf16x_tokens(wi, xb, injA, c, width, n, nullptr)) {
            std::puts("reference bf16x path not taken"); return 1; }
        tqllm_cuda::hc_low_silu_bf16(lowA, lbA, n * lr, c, nullptr);
        const bool took = tqllm_cuda::hc_down_inject_bf16x(wd, wi, xb, lowB, lbB, injB, lr, c, width, n, c, nullptr);
        cudaDeviceSynchronize();
        check(took && same(lowA, lowB, size_t(n) * lr * 4) && same(lbA, lbB, size_t(n) * lr * 2) && same(injA, injB, size_t(n) * c * 4), "down + silu + inject", lr, n);
        float * norm = upload(size_t(n) * width, 1.f), * logits = dev<float>(size_t(n) * width), * mA = dev<float>(size_t(n) * h), * mB = dev<float>(size_t(n) * h);
        __half * hA = dev<__half>(size_t(n) * h), * hB = dev<__half>(size_t(n) * h); __nv_bfloat16 * bA = dev<__nv_bfloat16>(size_t(n) * h), * bB = dev<__nv_bfloat16>(size_t(n) * h);
        tqllm_cuda::fn_gemv_bf16_tokens(wu, lowA, logits, width, lr, n, nullptr);
        tqllm_cuda::hc_mix(norm, logits, mA, h, c, n, nullptr);
        ref::to_h(mA, hA, int64_t(n) * h); tqllm_cuda::fn_round_bf16(mA, bA, int64_t(n) * h, nullptr);
        const bool took2 = tqllm_cuda::hc_up_mix_bf16(wu, lowA, norm, mB, hB, bB, h, c, lr, n, nullptr);
        cudaDeviceSynchronize();
        check(took2 && same(mA, mB, size_t(n) * h * 4) && same(hA, hB, size_t(n) * h * 2) && same(bA, bB, size_t(n) * h * 2), "up + mix", lr, n);
        for (void * p : {(void *) wd, (void *) wi, (void *) wu, (void *) xb, (void *) lowA, (void *) lowB, (void *) injA, (void *) injB, (void *) lbA, (void *) lbB,
                         (void *) norm, (void *) logits, (void *) mA, (void *) mB, (void *) hA, (void *) hB, (void *) bA, (void *) bB}) cudaFree(p);
    }
    for (int vh : {16, 32, 48, 64}) for (int n = 1; n <= 8; ++n) {  // GDN alpha / beta projections
        const int cols = h;
        __nv_bfloat16 * wa = upload_bf16(size_t(vh) * cols, .02f), * wb = upload_bf16(size_t(vh) * cols, .02f); float * x = upload(size_t(n) * cols, 1.f);
        float * aA = dev<float>(size_t(n) * vh), * aB = dev<float>(size_t(n) * vh), * bA = dev<float>(size_t(n) * vh), * bB = dev<float>(size_t(n) * vh);
        tqllm_cuda::fn_gemv_bf16_tokens(wa, x, aA, vh, cols, n, nullptr); tqllm_cuda::fn_gemv_bf16_tokens(wb, x, bA, vh, cols, n, nullptr);
        tqllm_cuda::GemvSegs g; g.n = 2; g.w[0] = wa; g.y[0] = aB; g.ld[0] = vh; g.start[1] = vh; g.w[1] = wb; g.y[1] = bB; g.ld[1] = vh; g.start[2] = 2 * vh;
        tqllm_cuda::fn_gemv_bf16_segments(g, x, cols, n, nullptr);
        cudaDeviceSynchronize();
        check(same(aA, aB, size_t(n) * vh * 4) && same(bA, bB, size_t(n) * vh * 4), "alpha + beta", vh, n);
        for (void * p : {(void *) wa, (void *) wb, (void *) x, (void *) aA, (void *) aB, (void *) bA, (void *) bB}) cudaFree(p);
    }
    for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 64, 200}) {  // GDN prep + gates (Flash-Next: 16 key heads, 48 value heads, dim 128)
        const int kh = 16, vh = 48, dim = 128, channels = (2 * kh + vh) * dim; const float eps = 1e-6f;
        float * qkv = upload(size_t(n) * channels, 1.f), * al = upload(size_t(n) * vh, 2.f), * be = upload(size_t(n) * vh, 2.f), * na = upload(vh, .5f), * bi = upload(vh, .5f);
        float * qkA = dev<float>(size_t(n) * 2 * kh * dim), * qkB = dev<float>(size_t(n) * 2 * kh * dim), * gA = dev<float>(size_t(n) * vh * 2), * gB = dev<float>(size_t(n) * vh * 2);
        ref::fn_gdn_prep_k<<<dim3(n, kh), 32>>>(qkv, qkA, kh, vh, dim, eps);
        ref::fn_gdn_gates_k<<<(n * vh + 127) / 128, 128>>>(al, be, na, bi, gA, vh, n);
        tqllm_cuda::fn_gdn_prep_gates(qkv, al, be, na, bi, qkB, gB, kh, vh, dim, n, eps, nullptr);
        cudaDeviceSynchronize();
        check(same(qkA, qkB, size_t(n) * 2 * kh * dim * 4) && same(gA, gB, size_t(n) * vh * 2 * 4), "gdn prep + gates", n, 0);
        for (void * p : {(void *) qkv, (void *) al, (void *) be, (void *) na, (void *) bi, (void *) qkA, (void *) qkB, (void *) gA, (void *) gB}) cudaFree(p);
    }
    for (int half : {0, 1}) for (int n : {1, 2, 5, 8, 64, 300}) {  // causal conv + next history (GDN: 10240 channels, 4 taps)
        const int channels = 10240, taps = 4, hist = taps - 1;
        float * x = upload(size_t(n) * channels, 1.f); __half * xh = upload_h(size_t(n) * channels, 1.f);
        float * hi = upload(size_t(hist) * channels, 1.f), * k = upload(size_t(channels) * taps, .5f);
        float * oA = dev<float>(size_t(n) * channels), * oB = dev<float>(size_t(n) * channels), * nA = dev<float>(size_t(hist) * channels), * nB = dev<float>(size_t(hist) * channels);
        if (half) {
            ref::ple_conv_k<<<(channels * n + 255) / 256, 256>>>(xh, hi, k, oA, channels, n, taps, 1);
            ref::ple_history_k<<<(channels * hist + 255) / 256, 256>>>(xh, hi, nA, channels, n, hist);
            tqllm_cuda::ple_conv(xh, hi, k, oB, nB, channels, n, taps, 1, nullptr);
        } else {
            ref::ple_conv_k<<<(channels * n + 255) / 256, 256>>>(x, hi, k, oA, channels, n, taps, 1);
            ref::ple_history_k<<<(channels * hist + 255) / 256, 256>>>(x, hi, nA, channels, n, hist);
            tqllm_cuda::ple_conv(x, hi, k, oB, nB, channels, n, taps, 1, nullptr);
        }
        cudaDeviceSynchronize();
        check(same(oA, oB, size_t(n) * channels * 4) && same(nA, nB, size_t(hist) * channels * 4), "conv + history", half, n);
        for (void * p : {(void *) x, (void *) xh, (void *) hi, (void *) k, (void *) oA, (void *) oB, (void *) nA, (void *) nB}) cudaFree(p);
    }
    for (int n = 1; n <= 8; ++n) {  // gated RMS, attention gate and shared-expert SwiGLU with their FP16 images
        const int vh = 48, dim = 128, rows = vh * n, heads = 16, ad = 256, ff = 1024;
        float * x = upload(size_t(rows) * dim, 1.f), * gt = upload(size_t(rows) * dim, 2.f), * w = upload(dim, 1.f);
        float * oA = dev<float>(size_t(rows) * dim), * oB = dev<float>(size_t(rows) * dim); __half * hA = dev<__half>(size_t(rows) * dim), * hB = dev<__half>(size_t(rows) * dim);
        ref::fn_gated_rms_k<<<rows, 32>>>(x, gt, w, oA, dim, 1e-6f); ref::to_h(oA, hA, int64_t(rows) * dim);
        tqllm_cuda::fn_gated_rms(x, gt, w, oB, vh, dim, n, 1e-6f, nullptr, hB);
        cudaDeviceSynchronize();
        check(same(oA, oB, size_t(rows) * dim * 4) && same(hA, hB, size_t(rows) * dim * 2), "gated rms + fp16 image", n, 0);
        const size_t cnt = size_t(n) * heads * ad;
        float * att = upload(cnt, 1.f), * qg = upload(cnt * 2, 2.f), * gA = dev<float>(cnt), * gB = dev<float>(cnt); __half * ghA = dev<__half>(cnt), * ghB = dev<__half>(cnt);
        cudaMemcpy(gA, att, cnt * 4, cudaMemcpyDeviceToDevice); cudaMemcpy(gB, att, cnt * 4, cudaMemcpyDeviceToDevice);
        ref::fn_attention_gate_k<<<unsigned((cnt + 255) / 256), 256>>>(gA, qg, ad, int(cnt)); ref::to_h(gA, ghA, int64_t(cnt));
        tqllm_cuda::fn_attention_gate(gB, qg, heads, ad, n, nullptr, ghB);
        cudaDeviceSynchronize();
        check(same(gA, gB, cnt * 4) && same(ghA, ghB, cnt * 2), "attention gate + fp16 image", n, 0);
        const size_t sc = size_t(n) * ff;
        float * sg = upload(sc, 2.f), * su = upload(sc, 2.f), * sA = dev<float>(sc), * sB = dev<float>(sc); __half * shA = dev<__half>(sc), * shB = dev<__half>(sc);
        ref::fn_swiglu_f32_k<<<unsigned((sc + 255) / 256), 256>>>(sg, su, sA, int64_t(sc)); ref::to_h(sA, shA, int64_t(sc));
        tqllm_cuda::fn_swiglu_f32(sg, su, sB, int64_t(sc), nullptr, shB);
        cudaDeviceSynchronize();
        check(same(sA, sB, sc * 4) && same(shA, shB, sc * 2), "shared swiglu + fp16 image", n, 0);
        for (void * p : {(void *) x, (void *) gt, (void *) w, (void *) oA, (void *) oB, (void *) hA, (void *) hB, (void *) att, (void *) qg, (void *) gA, (void *) gB,
                         (void *) ghA, (void *) ghB, (void *) sg, (void *) su, (void *) sA, (void *) sB, (void *) shA, (void *) shB}) cudaFree(p);
    }
    for (int k : {8, 10}) for (int n = 1; n <= 8; ++n) {  // MoE combine + hyper-connection combine
        __nv_bfloat16 * r = upload_bf16(size_t(n) * k * h, 1.f); float * w = upload(size_t(n) * k, .3f), * sh = upload(size_t(n) * h, 1.f), * sw = upload(n, .5f);
        float * inj = upload(size_t(n) * c, 2.f), * res = upload(size_t(n) * width, 3.f), * fo = dev<float>(size_t(n) * h), * oA = dev<float>(size_t(n) * width), * oB = dev<float>(size_t(n) * width);
        tqllm_cuda::fn_moe_combine(r, w, sh, sw, fo, h, k, n, nullptr);
        tqllm_cuda::hc_combine(res, fo, inj, oA, h, c, n, nullptr);
        tqllm_cuda::fn_moe_hc_combine_to(r, w, sh, sw, inj, res, oB, h, k, c, n, nullptr);
        cudaDeviceSynchronize();
        check(same(oA, oB, size_t(n) * width * 4), "moe + hc combine", k, n);
        for (void * p : {(void *) r, (void *) w, (void *) sh, (void *) sw, (void *) inj, (void *) res, (void *) fo, (void *) oA, (void *) oB}) cudaFree(p);
    }
    for (int heads : {16, 32}) for (int kvh : {2, 4}) for (int pos : {0, 7, 31000}) for (int n = 1; n <= 8; ++n) {  // attention q / k / v prep
        const int dim = 256, rot = 64; const float theta = 1e7f, eps = 1e-6f; const size_t qn = size_t(n) * heads * dim, kn = size_t(n) * kvh * dim;
        float * qg = upload(qn * 2, 1.f), * k = upload(kn, 1.f), * v = upload(kn, 1.f), * qw = upload(dim, 1.f), * kw = upload(dim, 1.f);
        float * qA = dev<float>(qn), * qB = dev<float>(qn), * knA = dev<float>(kn);
        __nv_bfloat16 * kA = dev<__nv_bfloat16>(kn), * vA = dev<__nv_bfloat16>(kn), * kB = dev<__nv_bfloat16>(kn), * vB = dev<__nv_bfloat16>(kn);
        tqllm_cuda::fn_norm_rope(qg, qw, qA, heads, dim, n, pos, rot, theta, eps, true, nullptr);
        tqllm_cuda::fn_norm_rope(k, kw, knA, kvh, dim, n, pos, rot, theta, eps, false, nullptr);
        ref::to_bf16(knA, kA, int64_t(kn)); ref::to_bf16(v, vA, int64_t(kn));
        const bool took = tqllm_cuda::fn_attention_prep(qg, k, v, qw, kw, qB, kB, vB, heads, kvh, dim, n, pos, rot, theta, eps, nullptr);
        cudaDeviceSynchronize();
        check(took && same(qA, qB, qn * 4) && same(kA, kB, kn * 2) && same(vA, vB, kn * 2), "attention q / k / v prep", heads * 100 + kvh, n + 10 * pos);
        for (void * p : {(void *) qg, (void *) k, (void *) v, (void *) qw, (void *) kw, (void *) qA, (void *) qB, (void *) knA, (void *) kA, (void *) vA, (void *) kB, (void *) vB}) cudaFree(p);
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("glue_fuse: CUDA error"); return 1; }
    std::printf("glue_fuse: %d cases, %d differ\n%s\n", cases, bad, bad ? "GLUE_FUSE FAIL" : "GLUE_FUSE PASS");
    return bad ? 1 : 0;
}

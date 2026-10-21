#include <algorithm>
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "gdn_seq_fast.cuh"
// TQFN_* switches that won their A/Bs are on by default; set NAME=0 to turn one off
static inline bool tqfn_default_on(const char * n) { const char * v = std::getenv(n); return !v || (v[0] && v[0] != '0'); }

namespace tqllm_cuda {
namespace {
void shape(int hidden, int streams, int tokens) {
    if (hidden <= 0 || streams <= 0 || tokens <= 0 || int64_t(hidden) * streams > std::numeric_limits<int>::max() ||
        int64_t(hidden) * streams * tokens > std::numeric_limits<int>::max())
        throw std::invalid_argument("HC: invalid shape");
}
__global__ void init_k(const float * embedding, float * residual, int h, int c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) residual[i] = embedding[int64_t(i / (c * h)) * h + i % h];
}
__global__ void norm_k(const float * x, const float * gamma, float * out, int h, int c, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * src = x + int64_t(row) * h;
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) sum += src[j] * src[j];
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / h + eps);
    for (int j = tid; j < h; j += blockDim.x)
        out[int64_t(row) * h + j] = __fmul_rn(__fmul_rn(src[j], inv), gamma[(row % c) * h + j]);
}
// norm_k with the BF16 image of its output (what f32_to_bf16 of `out` gives) for the BF16 projections that read it
__global__ void norm_bf16_k(const float * x, const float * gamma, float * out, __nv_bfloat16 * outb, int h, int c, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * src = x + int64_t(row) * h;
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) sum += src[j] * src[j];
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / h + eps);
    for (int j = tid; j < h; j += blockDim.x) {
        const float v = __fmul_rn(__fmul_rn(src[j], inv), gamma[(row % c) * h + j]);
        out[int64_t(row) * h + j] = v; outb[int64_t(row) * h + j] = __float2bfloat16_rn(v);
    }
}
// norm_bf16_k without the fp32 output (row scale + BF16 image only); mix_rs_k recomputes the normalized value
// from the residual, the row scale and gamma with norm_k's two roundings, so its result equals mix_k on norm_k's output
__global__ void norm_rs_bf16_k(const float * x, const float * gamma, float * rs, __nv_bfloat16 * outb, int h, int c, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * src = x + int64_t(row) * h;
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) sum += src[j] * src[j];
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / h + eps);
    if (tid == 0) rs[row] = inv;
    for (int j = tid; j < h; j += blockDim.x)
        outb[int64_t(row) * h + j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(src[j], inv), gamma[(row % c) * h + j]));
}
__global__ void mix_rs_k(const float * x, const float * rs, const float * gamma, const float * logits, float * out, int h, int c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int token = i / h, j = i % h;
    float sum = 0.f;
    for (int stream = 0; stream < c; ++stream) {
        const int64_t at = (int64_t(token) * c + stream) * h + j;
        const float normed = __fmul_rn(__fmul_rn(x[at], rs[token * c + stream]), gamma[stream * h + j]);
        const float gate = 1.f / (1.f + expf(-logits[at]));
        sum = __fadd_rn(sum, __fmul_rn(normed, gate));
    }
    out[i] = sum / c;
}
__global__ void low_bf16_k(float * x, __nv_bfloat16 * xb, int count, int c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { const float v = x[i] / c; const float y = v / (1.f + expf(-v)); x[i] = y; xb[i] = __float2bfloat16_rn(y); }
}
__global__ void low_k(float * x, int count, int c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { const float v = x[i] / c; x[i] = v / (1.f + expf(-v)); }
}
__global__ void mix_k(const float * norm, const float * logits, float * out, int h, int c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int token = i / h, j = i % h;
    float sum = 0.f;
    for (int stream = 0; stream < c; ++stream) {
        const int64_t at = (int64_t(token) * c + stream) * h + j;
        const float gate = 1.f / (1.f + expf(-logits[at]));
        sum = __fadd_rn(sum, __fmul_rn(norm[at], gate));
    }
    out[i] = sum / c;
}
__global__ void combine_k(const float * residual, const float * block, const float * inject, float * out,
                          int h, int c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int token = i / (h * c), stream = (i / h) % c, j = i % h;
    const float scatter = 2.f / (1.f + expf(-inject[token * c + stream] / c));
    out[i] = __fadd_rn(residual[i], __fmul_rn(block[token * h + j], scatter));
}
// combine_k followed by norm_k in one launch (same arithmetic, same order): row = token * c + stream
__global__ void combine_norm_k(const float * residual, const float * block, const float * inject, const float * gamma,
                               float * combined, float * normed, int h, int c, float eps) {
    const int row = blockIdx.x, token = row / c, stream = row % c, tid = threadIdx.x;
    const float scatter = 2.f / (1.f + expf(-inject[token * c + stream] / c));
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) {
        const int64_t i = int64_t(row) * h + j;
        const float v = __fadd_rn(residual[i], __fmul_rn(block[int64_t(token) * h + j], scatter));
        combined[i] = v; sum += v * v;
    }
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / h + eps);
    for (int j = tid; j < h; j += blockDim.x) {
        const int64_t i = int64_t(row) * h + j;
        normed[i] = __fmul_rn(__fmul_rn(combined[i], inv), gamma[stream * h + j]);
    }
}
__global__ void ple_gate_k(const float * key, const float * query, float * gates, int h) {
    const int row = blockIdx.x, tid = threadIdx.x;
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) {
        const int64_t at = int64_t(row) * h + j;
        sum += key[at] * query[at];
    }
    __shared__ float reduce[256]; reduce[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float score = reduce[0] / sqrtf(float(h));
        const float magnitude = sqrtf(fmaxf(fabsf(score), 1e-6f));
        const float signed_root = score > 0.f ? magnitude : score < 0.f ? -magnitude : 0.f;
        gates[row] = 1.f / (1.f + expf(-signed_root));
    }
}
__global__ void ple_broadcast_k(const float * value, const float * gates, float * out, int h, int c, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int token = i / (h * c), stream = (i / h) % c;
        out[i] = __fmul_rn(value[token * h + i % h], gates[token * c + stream]);
    }
}
__global__ void ple_conv_k(const float * x, const float * history, const float * kernel, float * out,
                           int channels, int n, int taps, int dilation) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels * n) return;
    const int token = i / channels, channel = i % channels, hist = (taps - 1) * dilation;
    float sum = 0.f;
    for (int tap = 0; tap < taps; ++tap) {
        const int position = token + tap * dilation;
        const float value = position < hist ? history[int64_t(position) * channels + channel]
                                            : x[int64_t(position - hist) * channels + channel];
        sum = __fadd_rn(sum, __fmul_rn(value, kernel[channel * taps + tap]));
    }
    out[i] = sum / (1.f + expf(-sum));
}
__global__ void ple_history_k(const float * x, const float * history, float * next,
                              int channels, int n, int hist) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hist * channels) return;
    const int position = n + i / channels, channel = i % channels;
    next[i] = position < hist ? history[int64_t(position) * channels + channel]
                              : x[int64_t(position - hist) * channels + channel];
}
__global__ void ple_apply_k(const float * hidden, const float * gated, const float * conv, float * out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __fadd_rn(hidden[i], __fadd_rn(gated[i], conv[i]));
}
__global__ void fn_gdn_k(const float * qkv, const float * alpha, const float * beta,
                         const float * neg_a, const float * bias, const float * state, float * next,
                         float * out, int kh, int vh, int dim, int n, float eps, float * states = nullptr) {
    const int head = blockIdx.x, value = blockIdx.y * 4 + threadIdx.y, lane = threadIdx.x;
    const int key_head = head % kh, channels = (2 * kh + vh) * dim;
    float column[8];
    const int pieces = dim / 32;
    const int64_t base = (int64_t(head) * dim + value) * dim;
    for (int r = 0; r < pieces; ++r) column[r] = state[base + r * 32 + lane];
    for (int t = 0; t < n; ++t) {
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
        float memory = 0.f;
        for (int r = 0; r < pieces; ++r) { q[r] *= qscale; k[r] *= kscale; memory += column[r] * k[r]; }
        for (int delta = 16; delta; delta >>= 1) memory += __shfl_xor_sync(0xffffffffu, memory, delta);
        const float a = alpha[t * vh + head] + bias[head];
        const float softplus = fmaxf(a, 0.f) + log1pf(expf(-fabsf(a)));
        const float decay = expf(neg_a[head] * softplus);
        const float b = 1.f / (1.f + expf(-beta[t * vh + head]));
        const float delta_v = (row[2 * kh * dim + head * dim + value] - decay * memory) * b;
        float result = 0.f;
        for (int r = 0; r < pieces; ++r) { column[r] = decay * column[r] + k[r] * delta_v; result += column[r] * q[r]; }
        for (int delta = 16; delta; delta >>= 1) result += __shfl_xor_sync(0xffffffffu, result, delta);
        if (!lane) out[(int64_t(t) * vh + head) * dim + value] = result;
        if (states) for (int r = 0; r < pieces; ++r) states[int64_t(t) * vh * dim * dim + base + r * 32 + lane] = column[r];
    }
    for (int r = 0; r < pieces; ++r) next[base + r * 32 + lane] = column[r];
}
// TQFN_GDN_FAST: the per-(token, key head) q/k normalisation and per-(token, head) gates are computed once instead of by
// every value warp; the recurrence then runs the same float operations in the same order (bit-identical to fn_gdn_k).
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
template <int VPW>  // value columns per warp: independent chains for latency hiding, same per-column op order
__global__ void fn_gdn_main_k(const float * qkv, const float * qk, const float * gates, const float * state, float * next,
                              float * out, int kh, int vh, int dim, int n) {
    const int head = blockIdx.x, value0 = (blockIdx.y * 4 + threadIdx.y) * VPW, lane = threadIdx.x;
    const int key_head = head % kh, channels = (2 * kh + vh) * dim;
    float column[VPW][8];
    const int pieces = dim / 32;
    for (int v = 0; v < VPW; ++v) {
        const int64_t base = (int64_t(head) * dim + value0 + v) * dim;
        for (int r = 0; r < pieces; ++r) column[v][r] = state[base + r * 32 + lane];
    }
    for (int t = 0; t < n; ++t) {
        const float * qp = qk + (int64_t(t) * 2 * kh + key_head) * dim, * kp = qk + (int64_t(t) * 2 * kh + kh + key_head) * dim;
        float q[8], k[8], memory[VPW];
        for (int r = 0; r < pieces; ++r) { q[r] = qp[r * 32 + lane]; k[r] = kp[r * 32 + lane]; }
        for (int v = 0; v < VPW; ++v) { memory[v] = 0.f; for (int r = 0; r < pieces; ++r) memory[v] += column[v][r] * k[r]; }
        for (int delta = 16; delta; delta >>= 1) for (int v = 0; v < VPW; ++v) memory[v] += __shfl_xor_sync(0xffffffffu, memory[v], delta);
        const float decay = gates[2 * (t * vh + head)], b = gates[2 * (t * vh + head) + 1];
        float result[VPW];
        for (int v = 0; v < VPW; ++v) {
            const float delta_v = (qkv[int64_t(t) * channels + 2 * kh * dim + head * dim + value0 + v] - decay * memory[v]) * b;
            result[v] = 0.f;
            for (int r = 0; r < pieces; ++r) { column[v][r] = decay * column[v][r] + k[r] * delta_v; result[v] += column[v][r] * q[r]; }
        }
        for (int delta = 16; delta; delta >>= 1) for (int v = 0; v < VPW; ++v) result[v] += __shfl_xor_sync(0xffffffffu, result[v], delta);
        if (!lane) for (int v = 0; v < VPW; ++v) out[(int64_t(t) * vh + head) * dim + value0 + v] = result[v];
    }
    for (int v = 0; v < VPW; ++v) {
        const int64_t base = (int64_t(head) * dim + value0 + v) * dim;
        for (int r = 0; r < pieces; ++r) next[base + r * 32 + lane] = column[v][r];
    }
}
// Key-head GDN prefill: the gated delta rule token by token over the whole sub-chunk, one block per (key head, 32 value columns) and
// its VPK value heads (they share q and k); the state slice in registers (thread (c, rg): rows rg*32..+31 of column c),
// q / k / v / gates staged 8 tokens at a time with cp.async, double-buffered. Design after Strata's gdn_rec_kh_kernel
// (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors); own code on our layouts: state [head][value][key],
// q and k from fn_gdn_prep_k (q carries 1 / sqrt(dim)), gates [t][head][decay, beta] from fn_gdn_gates_k.
namespace gdnkh {
constexpr int S = 128, CB = 32, RG = 4, RPG = S / RG, TB = 8;
__device__ __forceinline__ void cp16(void * d, const void * g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned) __cvta_generic_to_shared(d)), "l"(g));
#else
    *reinterpret_cast<float4 *>(d) = *reinterpret_cast<const float4 *>(g);
#endif
}
__device__ __forceinline__ void cp4(void * d, const void * g) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned) __cvta_generic_to_shared(d)), "l"(g));
#else
    *reinterpret_cast<float *>(d) = *reinterpret_cast<const float *>(g);
#endif
}
__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void wait_prev() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
}  // namespace gdnkh
template <int VPK>
__global__ void __launch_bounds__(gdnkh::CB * gdnkh::RG) fn_gdn_kh_k(const float * __restrict__ qkv, const float * __restrict__ qk,
                                                                   const float * __restrict__ gates, const float * __restrict__ state,
                                                                   float * __restrict__ next, float * __restrict__ out, int kh, int vh, int n) {
    using namespace gdnkh;
    constexpr int NT = CB * RG, QP = S / 4, VP = CB / 4;
    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][CB];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK], rkv[VPK][RG][CB], ro[VPK][RG][CB];
    const int ncb = S / CB, qh = blockIdx.x / ncb, cb = blockIdx.x % ncb;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int64_t channels = int64_t(2 * kh + vh) * S;
    float st[VPK][RPG];
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float4 * src = reinterpret_cast<const float4 *>(state + (int64_t(qh + j * kh) * S + col) * S + rg * RPG);
#pragma unroll
        for (int r = 0; r < RPG / 4; ++r) { const float4 v = src[r]; st[j][4 * r] = v.x; st[j][4 * r + 1] = v.y; st[j][4 * r + 2] = v.z; st[j][4 * r + 3] = v.w; }
    }
    const int nblk = (n + TB - 1) / TB;
    auto stage = [&](int k) {
        const int bb = k & 1, t0 = k * TB;
        for (int p = tid; p < TB * 2 * QP; p += NT) {
            const int i = p / (2 * QP), w = p % (2 * QP), isk = w / QP, jj = (w % QP) * 4;
            if (t0 + i < n) cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], qk + (int64_t(t0 + i) * 2 * kh + (isk ? kh : 0) + qh) * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < n) cp16(&sv[bb][i][j][jj], qkv + int64_t(t0 + i) * channels + int64_t(2 * kh) * S + int64_t(qh + j * kh) * S + cb * CB + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < n) cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], gates + 2 * (int64_t(t0 + i) * vh + qh + j * kh) + isb);
        }
    };
    if (nblk > 0) stage(0);
    commit();
    for (int k = 0; k < nblk; ++k) {
        if (k + 1 < nblk) stage(k + 1);  // buffer (k + 1) & 1 was last read in block k - 1, before its last __syncthreads
        commit();
        wait_prev();
        __syncthreads();
        const int bb = k & 1, cnt = min(TB, n - k * TB);
        for (int i = 0; i < cnt; ++i) {
            const int64_t t = int64_t(k) * TB + i;
            float kc[RPG];
#pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float kv[VPK], o[VPK], delta[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) { kv[j] = 0.f; o[j] = 0.f; }
#pragma unroll
            for (int r = 0; r < RPG; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] = fmaf(st[j][r], kc[r], kv[j]);
#pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            __syncthreads();
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float mem = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - sg[bb][i][j] * mem) * sb[bb][i][j];
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
#pragma unroll
                for (int j = 0; j < VPK; ++j) { st[j][r] = fmaf(sg[bb][i][j], st[j][r], kc[r] * delta[j]); o[j] = fmaf(st[j][r], qr, o[j]); }
            }
#pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            __syncthreads();
            if (rg < VPK) out[(t * vh + qh + rg * kh) * S + col] = ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c];
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float4 * dst = reinterpret_cast<float4 *>(next + (int64_t(qh + j * kh) * S + col) * S + rg * RPG);
#pragma unroll
        for (int r = 0; r < RPG / 4; ++r) dst[r] = make_float4(st[j][4 * r], st[j][4 * r + 1], st[j][4 * r + 2], st[j][4 * r + 3]);
    }
}
// Chunked gated delta rule for long prefills (the chunkwise / WY form of flash-linear-attention, MIT; own code).
// S_t = a_t S_{t-1} + e_t k_t^T, e_t = b_t (v_t - a_t S_{t-1} k_t). With g_t = prod a within a chunk, (I + A) E = R where
// A_tj = b_t (g_t / g_j) k_t.k_j (j < t) and R = b V - b g (K S0^T); then O = g (Q S0^T) + M E with M_tj = (g_t / g_j) q_t.k_j
// (j <= t) and S_C = g_C S0 + sum_j (g_C / g_j) e_j k_j^T. Only the C-token chunks are sequential. Block = one value head x
// VS value columns (columns are independent), state slice in shared memory.
// K K^T and Q K^T of every C-token chunk and key head, all chunks at once: [kh][chunks][2][C][C] (lower triangle used)
template <int C>
__global__ void __launch_bounds__(256) fn_gdn_chunk_gram_k(const float * qk, float * gram, int kh, int dim, int n) {
    extern __shared__ float sm[];
    const int P = dim + 1, key_head = blockIdx.x, chunk = blockIdx.y, t0 = chunk * C, cn = min(C, n - t0), tid = threadIdx.x;
    float * K = sm; float * Q = K + C * P;
    for (int i = tid; i < C * dim; i += blockDim.x) {
        const int t = i / dim, d = i % dim;
        K[t * P + d] = t < cn ? qk[(int64_t(t0 + t) * 2 * kh + kh + key_head) * dim + d] : 0.f;
        Q[t * P + d] = t < cn ? qk[(int64_t(t0 + t) * 2 * kh + key_head) * dim + d] : 0.f;
    }
    __syncthreads();
    float * g = gram + (int64_t(key_head) * gridDim.y + chunk) * 2 * C * C;
    for (int i = tid; i < C * C; i += blockDim.x) {
        const int t = i / C, j = i % C;
        float kk = 0.f, qq = 0.f;
        if (j <= t) for (int d = 0; d < dim; ++d) { kk += K[t * P + d] * K[j * P + d]; qq += Q[t * P + d] * K[j * P + d]; }
        g[i] = kk; g[C * C + i] = qq;
    }
}
template <int C, int VS>
__global__ void __launch_bounds__(256) fn_gdn_chunk_k(const float * qkv, const float * qk, const float * gates, const float * gram,
                                                      const float * state, float * next, float * out, int kh, int vh, int dim, int n) {
    extern __shared__ float sm[];
    const int P = dim + 1;               // padded row stride: rows of a warp's accesses land in different banks
    float * S = sm;                      // [VS][P]
    float * K = S + VS * P;              // [C][P]
    float * Q = K + C * P;               // [C][P]
    float * A = Q + C * P;               // [C][C]
    float * M = A + C * C;               // [C][C]
    float * E = M + C * C;               // [C][VS] (R, then E)
    float * lg = E + C * VS;             // [C] log of the running decay
    float * bt = lg + C;                 // [C]
    float * eg = bt + C;                 // [C] g_t
    float * rg = eg + C;                 // [C] g_C / g_t
    const int head = blockIdx.x, v0 = blockIdx.y * VS, tid = threadIdx.x, key_head = head % kh;
    const int channels = (2 * kh + vh) * dim;
    for (int i = tid; i < VS * dim; i += blockDim.x) S[(i / dim) * P + i % dim] = state[(int64_t(head) * dim + v0 + i / dim) * dim + i % dim];
    for (int t0 = 0; t0 < n; t0 += C) {
        const int cn = min(C, n - t0);
        __syncthreads();
        for (int i = tid; i < C * dim; i += blockDim.x) {
            const int t = i / dim, d = i % dim;
            K[t * P + d] = t < cn ? qk[(int64_t(t0 + t) * 2 * kh + kh + key_head) * dim + d] : 0.f;
            Q[t * P + d] = t < cn ? qk[(int64_t(t0 + t) * 2 * kh + key_head) * dim + d] : 0.f;
        }
        if (tid == 0) {
            float acc = 0.f;
            for (int t = 0; t < C; ++t) {
                if (t < cn) { acc += logf(fmaxf(gates[2 * (int64_t(t0 + t) * vh + head)], 1e-30f)); bt[t] = gates[2 * (int64_t(t0 + t) * vh + head) + 1]; }
                else bt[t] = 0.f;
                lg[t] = acc;
            }
            for (int t = 0; t < C; ++t) { eg[t] = expf(lg[t]); rg[t] = t < cn ? expf(lg[cn - 1] - lg[t]) : 0.f; }
        }
        __syncthreads();
        const float * g = gram + (int64_t(key_head) * ((n + C - 1) / C) + t0 / C) * 2 * C * C;
        for (int i = tid; i < C * C; i += blockDim.x) {  // A (strict lower) and M (lower incl. diagonal) from the shared grams
            const int t = i / C, j = i % C;
            float a = 0.f, m = 0.f;
            if (j <= t && t < cn) {
                const float ratio = expf(lg[t] - lg[j]);
                if (j < t) a = bt[t] * ratio * g[i];
                m = ratio * g[C * C + i];
            }
            A[i] = a; M[i] = m;
        }
        for (int i = tid; i < C * VS; i += blockDim.x) {  // R = b v - b g (K S0^T)
            const int t = i / VS, v = i % VS;
            float r = 0.f;
            if (t < cn) {
                float ks = 0.f;
                for (int d = 0; d < dim; ++d) ks += K[t * P + d] * S[v * P + d];
                const float val = qkv[int64_t(t0 + t) * channels + 2 * kh * dim + head * dim + v0 + v];
                r = bt[t] * (val - eg[t] * ks);
            }
            E[i] = r;
        }
        __syncthreads();
        if (tid < VS)  // forward substitution, one value column per thread
            for (int t = 1; t < cn; ++t) {
                float e = E[t * VS + tid];
                for (int j = 0; j < t; ++j) e -= A[t * C + j] * E[j * VS + tid];
                E[t * VS + tid] = e;
            }
        __syncthreads();
        for (int i = tid; i < C * VS; i += blockDim.x) {  // O = g (Q S0^T) + M E
            const int t = i / VS, v = i % VS;
            if (t >= cn) continue;
            float o = 0.f;
            for (int d = 0; d < dim; ++d) o += Q[t * P + d] * S[v * P + d];
            o *= eg[t];
            for (int j = 0; j <= t; ++j) o += M[t * C + j] * E[j * VS + v];
            out[(int64_t(t0 + t) * vh + head) * dim + v0 + v] = o;
        }
        __syncthreads();
        const float gc = eg[cn - 1];
        for (int i = tid; i < VS * dim; i += blockDim.x) {  // S = g_C S0 + sum_j (g_C / g_j) e_j k_j^T
            const int v = i / dim, d = i % dim;
            float acc = gc * S[v * P + d];
            for (int j = 0; j < cn; ++j) acc += rg[j] * E[j * VS + v] * K[j * P + d];
            S[v * P + d] = acc;
        }
    }
    __syncthreads();
    for (int i = tid; i < VS * dim; i += blockDim.x) next[(int64_t(head) * dim + v0 + i / dim) * dim + i % dim] = S[(i / dim) * P + i % dim];
}
// WY form of the same chunked rule, split so that only state-dependent work stays sequential. Every chunk, in parallel:
// T = (I + A)^-1 (unit lower triangular), U = T (b V), W = T (b g K), M as above; then E = U - W S0^T. Layouts per
// value head and chunk: U, W [C][dim], M [C][C], lg [C] (running log decay inside the chunk).
template <int C>
__global__ void __launch_bounds__(256) fn_gdn_wy_k(const float * qkv, const float * qk, const float * gates, float * U, float * W,
                                                   float * Mg, float * lgo, int kh, int vh, int dim, int n) {
    extern __shared__ float sm[];
    const int P = dim + 1, head = blockIdx.x, chunk = blockIdx.y, t0 = chunk * C, cn = min(C, n - t0), tid = threadIdx.x;
    const int key_head = head % kh, channels = (2 * kh + vh) * dim;
    float * K = sm; float * Q = K + C * P; float * V = Q + C * P;
    float * A = V + C * P; float * T = A + C * (C + 1);
    float * lg = T + C * (C + 1); float * bt = lg + C; float * eg = bt + C;
    for (int i = tid; i < C * dim; i += blockDim.x) {
        const int t = i / dim, d = i % dim; const bool in = t < cn; const int64_t tt = int64_t(t0 + t);
        K[t * P + d] = in ? qk[(tt * 2 * kh + kh + key_head) * dim + d] : 0.f;
        Q[t * P + d] = in ? qk[(tt * 2 * kh + key_head) * dim + d] : 0.f;
        V[t * P + d] = in ? qkv[tt * channels + 2 * kh * dim + head * dim + d] : 0.f;
    }
    if (tid < 32) {  // C == 32: one lane per token, inclusive scan of the log decay
        const bool in = tid < cn;
        float x = in ? logf(fmaxf(gates[2 * ((int64_t(t0) + tid) * vh + head)], 1e-30f)) : 0.f;
        for (int o = 1; o < 32; o <<= 1) { const float y = __shfl_up_sync(0xffffffffu, x, o); if (tid >= o) x += y; }
        lg[tid] = x; eg[tid] = expf(x); bt[tid] = in ? gates[2 * ((int64_t(t0) + tid) * vh + head) + 1] : 0.f;
    }
    __syncthreads();
    float * Mo = Mg + (int64_t(head) * gridDim.y + chunk) * C * C;
    for (int i = tid; i < C * C; i += blockDim.x) {
        const int t = i / C, j = i % C;
        float a = 0.f, m = 0.f;
        if (j <= t && t < cn) {
            float kk = 0.f, qq = 0.f;
            for (int d = 0; d < dim; ++d) { const float kj = K[j * P + d]; kk += K[t * P + d] * kj; qq += Q[t * P + d] * kj; }
            const float ratio = expf(lg[t] - lg[j]);
            if (j < t) a = bt[t] * ratio * kk;
            m = ratio * qq;
        }
        A[t * (C + 1) + j] = a; Mo[i] = m;
    }
    __syncthreads();
    if (tid < C) {  // column tid of (I + A)^-1 by forward substitution; columns are independent
        const int c = tid;
        for (int t = 0; t < C; ++t) T[t * (C + 1) + c] = t == c ? 1.f : 0.f;
        for (int t = c + 1; t < C; ++t) {
            float acc = 0.f;
            for (int j = c; j < t; ++j) acc += A[t * (C + 1) + j] * T[j * (C + 1) + c];
            T[t * (C + 1) + c] = -acc;
        }
    }
    __syncthreads();
    float * Uo = U + (int64_t(head) * gridDim.y + chunk) * C * dim, * Wo = W + (int64_t(head) * gridDim.y + chunk) * C * dim;
    for (int i = tid; i < C * dim; i += blockDim.x) {
        const int t = i / dim, d = i % dim;
        float u = 0.f, w = 0.f;
        for (int j = 0; j <= t; ++j) { const float tb = T[t * (C + 1) + j] * bt[j]; u += tb * V[j * P + d]; w += tb * eg[j] * K[j * P + d]; }
        Uo[i] = u; Wo[i] = w;
    }
    if (tid < C) lgo[(int64_t(head) * gridDim.y + chunk) * C + tid] = lg[tid];
}
// The sequential part: per C-token chunk E = U - W S^T, O = g (Q S^T) + M E, S = g_C S + sum_j (g_C / g_j) e_j k_j^T.
// Block = one value head x VS value columns; a thread owns output rows t = tid / 8 and columns v = tid % 8 + 8 k.
template <int C, int VS>
__global__ void __launch_bounds__(256) fn_gdn_seq_k(const float * qk, const float * U, const float * W, const float * Mg,
                                                    const float * lgi, const float * state, float * next, float * out,
                                                    int kh, int vh, int dim, int n, int chunks) {
    extern __shared__ float sm[];
    const int P = dim + 1, head = blockIdx.x, v0 = blockIdx.y * VS, tid = threadIdx.x, key_head = head % kh;
    float * S = sm; float * Wc = S + VS * P; float * Qc = Wc + C * P; float * Kc = Qc + C * P;
    float * E = Kc + C * P; float * M = E + C * (VS + 1); float * eg = M + C * C; float * rg = eg + C;
    for (int i = tid; i < VS * dim; i += blockDim.x) S[(i / dim) * P + i % dim] = state[(int64_t(head) * dim + v0 + i / dim) * dim + i % dim];
    constexpr int CPT = VS / 8;  // value columns per thread in the E / O phase
    const int et = tid / 8, ev = tid % 8;
    for (int c = 0; c < chunks; ++c) {
        const int t0 = c * C, cn = min(C, n - t0);
        __syncthreads();
        const float * Ug = U + (int64_t(head) * chunks + c) * C * dim, * Wg = W + (int64_t(head) * chunks + c) * C * dim;
        for (int i = tid; i < C * dim; i += blockDim.x) {
            const int t = i / dim, d = i % dim; const bool in = t < cn; const int64_t tt = int64_t(t0 + t);
            Wc[t * P + d] = Wg[i];
            Qc[t * P + d] = in ? qk[(tt * 2 * kh + key_head) * dim + d] : 0.f;
            Kc[t * P + d] = in ? qk[(tt * 2 * kh + kh + key_head) * dim + d] : 0.f;
        }
        for (int i = tid; i < C * VS; i += blockDim.x) E[(i / VS) * (VS + 1) + i % VS] = Ug[(i / VS) * dim + v0 + i % VS];
        for (int i = tid; i < C * C; i += blockDim.x) M[i] = Mg[(int64_t(head) * chunks + c) * C * C + i];
        if (tid < C) {
            const float * l = lgi + (int64_t(head) * chunks + c) * C;
            eg[tid] = expf(l[tid]); rg[tid] = tid < cn ? expf(l[cn - 1] - l[tid]) : 0.f;
        }
        __syncthreads();
        float ws[CPT], qs[CPT];
        for (int k = 0; k < CPT; ++k) { ws[k] = 0.f; qs[k] = 0.f; }
        for (int d = 0; d < dim; ++d) {
            const float w = Wc[et * P + d], q = Qc[et * P + d];
            for (int k = 0; k < CPT; ++k) { const float sv = S[(ev + 8 * k) * P + d]; ws[k] += w * sv; qs[k] += q * sv; }
        }
        for (int k = 0; k < CPT; ++k) E[et * (VS + 1) + ev + 8 * k] -= ws[k];
        __syncthreads();
        if (et < cn)
            for (int k = 0; k < CPT; ++k) {
                const int v = ev + 8 * k;
                float o = eg[et] * qs[k];
                for (int j = 0; j <= et; ++j) o += M[et * C + j] * E[j * (VS + 1) + v];
                out[(int64_t(t0 + et) * vh + head) * dim + v0 + v] = o;
            }
        __syncthreads();
        const float gc = expf(lgi[(int64_t(head) * chunks + c) * C + cn - 1]);
        const int sv_ = tid / 8, sd0 = tid % 8;  // S update: thread owns value row sv_ (VS == 32) and key columns sd0 + 8 k
        for (int k = sd0; k < dim; k += 8) {
            float acc = gc * S[sv_ * P + k];
            for (int j = 0; j < cn; ++j) acc += rg[j] * E[j * (VS + 1) + sv_] * Kc[j * P + k];
            S[sv_ * P + k] = acc;
        }
    }
    __syncthreads();
    for (int i = tid; i < VS * dim; i += blockDim.x) next[(int64_t(head) * dim + v0 + i / dim) * dim + i % dim] = S[(i / dim) * P + i % dim];
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
__global__ void fn_norm_rope_k(const float * x, const float * weight, float * out,
                               int heads, int dim, int pos, int rotary, float theta, float eps, int input_heads) {
    const int row = blockIdx.x, head = row % heads, token = row / heads, lane = threadIdx.x;
    const float * src = x + (int64_t(token) * heads + head) * dim * input_heads;
    float ss = 0.f;
    for (int d = lane; d < dim; d += 32) ss += src[d] * src[d];
    for (int step = 16; step; step >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, step);
    const float inv = rsqrtf(ss / dim + eps);
    const int half = rotary / 2;
    for (int d = lane; d < dim; d += 32) {
        float v = __fmul_rn(__fmul_rn(src[d], inv), weight[d]);
        if (d < rotary) {
            const int pair = d % half;
            const double angle = double(pos + token) / pow(double(theta), double(2 * pair) / rotary);
            const float cosine = float(cos(angle)), sine = float(sin(angle));
            const float other = __fmul_rn(__fmul_rn(src[d < half ? d + half : d - half], inv), weight[d < half ? d + half : d - half]);
            v = d < half ? v * cosine - other * sine : v * cosine + other * sine;
        }
        out[int64_t(row) * dim + d] = v;
    }
}
__global__ void fn_attention_k(const float * q, const __nv_bfloat16 * k, const __nv_bfloat16 * v,
                               float * out, int heads, int kvheads, int dim, int pos) {
    const int head = blockIdx.x, token = blockIdx.y, tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int kh = head / (heads / kvheads), count = pos + token + 1;
    const float * query = q + (int64_t(token) * heads + head) * dim;
    extern __shared__ float scores[];
    __shared__ float reduction[128];
    for (int key = warp; key < count; key += 4) {
        float dot = 0.f;
        for (int d = lane; d < dim; d += 32) dot += query[d] * __bfloat162float(k[(int64_t(key) * kvheads + kh) * dim + d]);
        for (int step = 16; step; step >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, step);
        if (!lane) scores[key] = dot / sqrtf(float(dim));
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int key = tid; key < count; key += 128) maximum = fmaxf(maximum, scores[key]);
    reduction[tid] = maximum; __syncthreads();
    for (int step = 64; step; step >>= 1) { if (tid < step) reduction[tid] = fmaxf(reduction[tid], reduction[tid + step]); __syncthreads(); }
    maximum = reduction[0];
    // Every thread must finish reading the maximum before reduction is reused.
    __syncthreads();
    float sum = 0.f;
    for (int key = tid; key < count; key += 128) { scores[key] = expf(scores[key] - maximum); sum += scores[key]; }
    reduction[tid] = sum; __syncthreads();
    for (int step = 64; step; step >>= 1) { if (tid < step) reduction[tid] += reduction[tid + step]; __syncthreads(); }
    const float inv = 1.f / reduction[0];
    for (int d = tid; d < dim; d += 128) {
        float value = 0.f;
        for (int key = 0; key < count; ++key) value += scores[key] * __bfloat162float(v[(int64_t(key) * kvheads + kh) * dim + d]);
        out[(int64_t(token) * heads + head) * dim + d] = value * inv;
    }
}
__global__ void fn_attention_gate_k(float * out, const float * qg, int dim, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] *= 1.f / (1.f + expf(-qg[(int64_t(i) / dim) * 2 * dim + dim + i % dim]));
}
} // namespace

__global__ void fn_swiglu_bf16_h_k(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * o, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= n) return;
    const float a = __bfloat162float(g[i]); o[i] = __float2half_rn(a / (1.f + expf(-a)) * __bfloat162float(u[i]));
}
// same element arithmetic as fn_swiglu_bf16_h_k, only on the assignments of a grouped-expert pass (slot lists); blockIdx.x
// strides over the slot's rows (a cap x slots grid would launch mostly empty blocks)
__global__ void fn_swiglu_bf16_h_groups_k(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * o, const int * counts,
                                               const int * groups, int capacity, int ff) {
    const int slot = blockIdx.y, n = counts[slot];
    for (int col = blockIdx.x; col < n; col += gridDim.x) {
        const int64_t base = int64_t(groups[int64_t(slot) * capacity + col]) * ff;
        for (int j = threadIdx.x; j < ff; j += blockDim.x) {
            const int64_t i = base + j;
            const float a = __bfloat162float(g[i]); o[i] = __float2half_rn(__fmul_rn(__fdiv_rn(a, __fadd_rn(1.f, expf(-a))), __bfloat162float(u[i])));
        }
    }
}
__global__ void fn_swiglu_f32_k(const float * g, const float * u, float * o, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= n) return;
    const float a = g[i]; o[i] = a / (1.f + expf(-a)) * u[i];
}
__global__ void fn_moe_combine_k(const __nv_bfloat16 * r, const float * w, const float * sh, const float * sw, float * o, int h, int k, int n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= int64_t(n) * h) return;
    const int t = int(i / h), j = int(i % h); float v = 0.f;
    for (int e = 0; e < k; ++e) v += w[t * k + e] * __bfloat162float(r[(int64_t(t) * k + e) * h + j]);
    o[i] = v + sw[t] * sh[i];
}
// gridDim.y splits the columns: block (b, k) covers vectors [k*span, (k+1)*span) and writes part[k][row] (or y when
// gridDim.y == 1); fn_gemv_sum_k then adds the partials in a fixed order (deterministic).
__global__ void fn_gemv_bf16_k(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(row) * cols);
    const int nv = cols / 8, span = (nv + gridDim.y - 1) / gridDim.y, v0 = blockIdx.y * span, v1 = min(nv, v0 + span);
    float sum = 0.f;
    for (int v = v0 + lane; v < v1; v += 32) {
        const uint4 packed = wr[v];
        const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
        const float4 xa = reinterpret_cast<const float4 *>(x)[2 * v], xb = reinterpret_cast<const float4 *>(x)[2 * v + 1];
        const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        #pragma unroll
        for (int k = 0; k < 8; ++k) sum += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[k]));
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (!lane) y[int64_t(blockIdx.y) * rows + row] = sum;
}
__global__ void fn_gemv_sum_k(const float * part, float * y, int rows, int splits) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= rows) return;
    float t = 0.f; for (int k = 0; k < splits; ++k) t += part[int64_t(k) * rows + r]; y[r] = t;
}
// Split-K attention (flash decoding): block (head, token, split) scores keys [split*chunk, ...) with 8 warps, lanes over
// dims (dim <= 256, multiple of 32); partials (max, sum, acc[dim]) are merged by fn_attention_merge_k.
__global__ void fn_attention_split_k(const float * q, const __nv_bfloat16 * k, const __nv_bfloat16 * v, float * part,
                                     int heads, int kvheads, int dim, int pos, int chunk, int splits) {
    const int head = blockIdx.x, token = blockIdx.y, split = blockIdx.z, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kh = head / (heads / kvheads), count = pos + token + 1;
    const int begin = split * chunk, end = min(count, begin + chunk);
    float * out = part + ((int64_t(token) * heads + head) * splits + split) * (dim + 2);
    extern __shared__ float sm[];  // scores[chunk] then acc[8][dim]
    float * scores = sm; float * acc = sm + chunk;
    __shared__ float red[8];
    if (begin >= end) { if (!tid) { out[dim] = -INFINITY; out[dim + 1] = 0.f; } return; }
    const float * query = q + (int64_t(token) * heads + head) * dim;
    const float scale = 1.f / sqrtf(float(dim));
    for (int key = begin + warp; key < end; key += 8) {
        float dot = 0.f;
        const __nv_bfloat16 * kr = k + (int64_t(key) * kvheads + kh) * dim;
        for (int d = lane; d < dim; d += 32) dot += query[d] * __bfloat162float(kr[d]);
        for (int st = 16; st; st >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, st);
        if (!lane) scores[key - begin] = dot * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = tid; i < end - begin; i += blockDim.x) m = fmaxf(m, scores[i]);
    for (int st = 16; st; st >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, st));
    if (!lane) red[warp] = m;
    __syncthreads();
    m = red[0]; for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float l = 0.f;
    for (int i = tid; i < end - begin; i += blockDim.x) { const float e = expf(scores[i] - m); scores[i] = e; l += e; }
    for (int st = 16; st; st >>= 1) l += __shfl_xor_sync(0xffffffffu, l, st);
    if (!lane) red[warp] = l;
    __syncthreads();
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int per = dim / 32;
    for (int key = begin + warp; key < end; key += 8) {
        const float w = scores[key - begin]; const __nv_bfloat16 * vr = v + (int64_t(key) * kvheads + kh) * dim;
        for (int j = 0; j < per; ++j) a[j] += w * __bfloat162float(vr[lane + 32 * j]);
    }
    for (int j = 0; j < per; ++j) acc[warp * dim + lane + 32 * j] = a[j];
    __syncthreads();
    for (int d = tid; d < dim; d += blockDim.x) { float t = 0.f; for (int w = 0; w < 8; ++w) t += acc[w * dim + d]; out[d] = t; }
    if (!tid) { float t = 0.f; for (int w = 0; w < 8; ++w) t += red[w]; out[dim] = m; out[dim + 1] = t; }
}
__global__ void fn_attention_merge_k(const float * part, float * out, int heads, int dim, int splits) {
    const int head = blockIdx.x, token = blockIdx.y;
    const float * p = part + (int64_t(token) * heads + head) * splits * (dim + 2);
    float M = -INFINITY; for (int s = 0; s < splits; ++s) M = fmaxf(M, p[s * (dim + 2) + dim]);
    float L = 0.f; for (int s = 0; s < splits; ++s) { const float mm = p[s * (dim + 2) + dim]; if (mm > -INFINITY) L += expf(mm - M) * p[s * (dim + 2) + dim + 1]; }
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float t = 0.f;
        for (int s = 0; s < splits; ++s) { const float mm = p[s * (dim + 2) + dim]; if (mm > -INFINITY) t += expf(mm - M) * p[s * (dim + 2) + d]; }
        out[(int64_t(token) * heads + head) * dim + d] = t / L;
    }
}
template<int T> __global__ void fn_gemv_bf16_multi_k(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(row) * cols);
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.f;
    for (int v = lane; v < cols / 8; v += 32) {
        const uint4 packed = wr[v];
        const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xa = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v], xb = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v + 1];
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int k = 0; k < 8; ++k) sum[t] += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[k]));
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        for (int d = 16; d; d >>= 1) sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], d);
        if (!lane) y[int64_t(t) * rows + row] = sum[t];
    }
}
// One block per row (<= 256 threads): the columns spread over several warps instead of one.
template<int T> __global__ void fn_gemv_bf16_few_k(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, int64_t ldy) {
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(row) * cols);
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.f;
    for (int v = threadIdx.x; v < cols / 8; v += blockDim.x) {
        const uint4 packed = wr[v];
        const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xa = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v], xb = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v + 1];
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int k = 0; k < 8; ++k) sum[t] += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[k]));
        }
    }
    __shared__ float part[T][8];
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        for (int d = 16; d; d >>= 1) sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], d);
        if (!lane) part[t][warp] = sum[t];
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float r = 0.f;
        for (int k = 0; k < int(blockDim.x / 32); ++k) r += part[threadIdx.x][k];
        y[int64_t(threadIdx.x) * ldy + row] = r;
    }
}
// fn_gemv_bf16_few_k for a prefill chunk: R rows x T tokens per block, so x is read once per R rows instead of once
// per row. Each (row, token) sum keeps fn_gemv_bf16_few_k's order (same thread count, columns per thread, reductions).
template<int R, int T> __global__ void fn_gemv_bf16_tiles_k(const __nv_bfloat16 * w, const float * x, float * y, int rows,
                                                            int cols, int tokens, int64_t ldy) {
    const int row0 = blockIdx.x * R, t0 = blockIdx.y * T, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float sum[R][T];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < T; ++t) sum[r][t] = 0.f;
    for (int v = threadIdx.x; v < cols / 8; v += blockDim.x) {
        float xs[T][8];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const int tt = min(t0 + t, tokens - 1);
            const float4 xa = reinterpret_cast<const float4 *>(x + int64_t(tt) * cols)[2 * v], xb = reinterpret_cast<const float4 *>(x + int64_t(tt) * cols)[2 * v + 1];
            xs[t][0] = xa.x; xs[t][1] = xa.y; xs[t][2] = xa.z; xs[t][3] = xa.w; xs[t][4] = xb.x; xs[t][5] = xb.y; xs[t][6] = xb.z; xs[t][7] = xb.w;
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 packed = reinterpret_cast<const uint4 *>(w + int64_t(min(row0 + r, rows - 1)) * cols)[v];
            const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
            #pragma unroll
            for (int t = 0; t < T; ++t)
                #pragma unroll
                for (int k = 0; k < 8; ++k) sum[r][t] += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[t][k]));
        }
    }
    __shared__ float part[R * T][8];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            float s = sum[r][t];
            for (int d = 16; d; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
            if (!lane) part[r * T + t][warp] = s;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < R * T; i += blockDim.x) {
        const int r = i / T, t = i % T;
        if (row0 + r >= rows || t0 + t >= tokens) continue;
        float s = 0.f;
        for (int k = 0; k < int(blockDim.x / 32); ++k) s += part[i][k];
        y[int64_t(t0 + t) * ldy + row0 + r] = s;
    }
}
// fn_gemv_bf16_tiles_k on an input already rounded to bf16 (fn_round_bf16): the kernel rounds x to bf16 before every
// product anyway, so the sums are the same bits and each 8-column load is 16 bytes instead of 32.
template<int R, int T> __global__ void fn_gemv_bf16x_tiles_k(const __nv_bfloat16 * w, const __nv_bfloat16 * x, float * y, int rows,
                                                             int cols, int tokens, int64_t ldy) {
    const int row0 = blockIdx.x * R, t0 = blockIdx.y * T, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    float sum[R][T];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < T; ++t) sum[r][t] = 0.f;
    for (int v = threadIdx.x; v < cols / 8; v += blockDim.x) {
        float xs[T][8];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const uint4 p = reinterpret_cast<const uint4 *>(x + int64_t(min(t0 + t, tokens - 1)) * cols)[v];
            const __nv_bfloat16 * xv = reinterpret_cast<const __nv_bfloat16 *>(&p);
            #pragma unroll
            for (int k = 0; k < 8; ++k) xs[t][k] = __bfloat162float(xv[k]);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 packed = reinterpret_cast<const uint4 *>(w + int64_t(min(row0 + r, rows - 1)) * cols)[v];
            const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
            #pragma unroll
            for (int t = 0; t < T; ++t)
                #pragma unroll
                for (int k = 0; k < 8; ++k) sum[r][t] += __bfloat162float(wv[k]) * xs[t][k];
        }
    }
    __shared__ float part[R * T][8];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            float s = sum[r][t];
            for (int d = 16; d; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
            if (!lane) part[r * T + t][warp] = s;
        }
    __syncthreads();
    for (int i = threadIdx.x; i < R * T; i += blockDim.x) {
        const int r = i / T, t = i % T;
        if (row0 + r >= rows || t0 + t >= tokens) continue;
        float s = 0.f;
        for (int k = 0; k < int(blockDim.x / 32); ++k) s += part[i][k];
        y[int64_t(t0 + t) * ldy + row0 + r] = s;
    }
}
__global__ void fn_round_bf16_k(const float * x, __nv_bfloat16 * xb, int64_t n) {
    const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        const float4 v = *reinterpret_cast<const float4 *>(x + i);
        xb[i] = __float2bfloat16_rn(v.x); xb[i + 1] = __float2bfloat16_rn(v.y); xb[i + 2] = __float2bfloat16_rn(v.z); xb[i + 3] = __float2bfloat16_rn(v.w);
    } else for (int64_t j = i; j < n; ++j) xb[j] = __float2bfloat16_rn(x[j]);
}
// Short rows (<= 512 columns): 8 lanes per row, 4 rows per warp, so each lane keeps several loads in flight.
template<int T> __global__ void fn_gemv_bf16_short_k(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 8, sub = threadIdx.x & 7;
    const bool live = row < rows;
    const uint4 * wr = reinterpret_cast<const uint4 *>(w + int64_t(live ? row : 0) * cols);
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.f;
    for (int v = sub; live && v < cols / 8; v += 8) {
        const uint4 packed = wr[v];
        const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xa = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v], xb = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v + 1];
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int k = 0; k < 8; ++k) sum[t] += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[k]));
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        for (int d = 4; d; d >>= 1) sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], d);
        if (live && !sub) y[int64_t(t) * rows + row] = sum[t];
    }
}
void hc_init(const float * embedding, float * residual, int h, int c, int n, cudaStream_t s) {
    shape(h, c, n);
    if (!embedding || !residual) throw std::invalid_argument("HC init: null pointer");
    const int count = h * c * n;
    init_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(embedding, residual, h, c, count);
}
void hc_combine_norm(const float * residual, const float * block, const float * inject, const float * gamma, float * combined, float * normed,
                     int h, int c, int n, float eps, cudaStream_t s) {
    shape(h, c, n);
    if (!residual || !block || !inject || !gamma || !combined || !normed || !std::isfinite(eps) || eps <= 0.f) throw std::invalid_argument("HC combine+norm: invalid pointer or epsilon");
    combine_norm_k<<<c * n, 256, 0, s>>>(residual, block, inject, gamma, combined, normed, h, c, eps);
}
void hc_norm(const float * x, const float * gamma, float * out, int h, int c, int n, float eps, cudaStream_t s) {
    shape(h, c, n);
    if (!x || !gamma || !out || !std::isfinite(eps) || eps <= 0.f) throw std::invalid_argument("HC norm: invalid pointer or epsilon");
    norm_k<<<c * n, 256, 0, s>>>(x, gamma, out, h, c, eps);
}
void hc_norm_bf16(const float * x, const float * gamma, float * out, __nv_bfloat16 * outb, int h, int c, int n, float eps, cudaStream_t s) {
    shape(h, c, n);
    if (!x || !gamma || !out || !outb || !std::isfinite(eps) || eps <= 0.f) throw std::invalid_argument("HC norm: invalid pointer or epsilon");
    norm_bf16_k<<<c * n, 256, 0, s>>>(x, gamma, out, outb, h, c, eps);
}
// combine_k and norm_rs_bf16_k in one pass over the row (token, stream): out = residual + block * scatter is written
// once and its sum of squares taken in the same loop (the same per-thread strides and reduction as norm_rs_bf16_k, so rs
// and the bf16 image are bit-identical to the two kernels); the second pass reads the fresh row back from L2
__global__ void combine_norm_rs_bf16_k(const float * residual, const float * block, const float * inject, float * out,
                                       const float * gamma, float * rs, __nv_bfloat16 * outb, int h, int c, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x, token = row / c, stream = row % c;
    const float scatter = 2.f / (1.f + expf(-inject[token * c + stream] / c));
    const float * res = residual + int64_t(row) * h, * blk = block + int64_t(token) * h; float * dst = out + int64_t(row) * h;
    float sum = 0.f;
    for (int j = tid; j < h; j += blockDim.x) { const float v = __fadd_rn(res[j], __fmul_rn(blk[j], scatter)); dst[j] = v; sum += v * v; }
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / h + eps);
    if (tid == 0) rs[row] = inv;
    for (int j = tid; j < h; j += blockDim.x)
        outb[int64_t(row) * h + j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(dst[j], inv), gamma[stream * h + j]));
}
void hc_combine_norm_rs_bf16(const float * residual, const float * block, const float * inject, float * out, const float * gamma,
                             float * rs, __nv_bfloat16 * outb, int h, int c, int n, float eps, cudaStream_t s) {
    shape(h, c, n);
    if (!residual || !block || !inject || !out || !gamma || !rs || !outb || !std::isfinite(eps) || eps <= 0.f)
        throw std::invalid_argument("HC combine + norm (rs): invalid pointer or epsilon");
    combine_norm_rs_bf16_k<<<c * n, 256, 0, s>>>(residual, block, inject, out, gamma, rs, outb, h, c, eps);
}
void hc_norm_rs_bf16(const float * x, const float * gamma, float * rs, __nv_bfloat16 * outb, int h, int c, int n, float eps, cudaStream_t s) {
    shape(h, c, n);
    if (!x || !gamma || !rs || !outb || !std::isfinite(eps) || eps <= 0.f) throw std::invalid_argument("HC norm (rs): invalid pointer or epsilon");
    norm_rs_bf16_k<<<c * n, 256, 0, s>>>(x, gamma, rs, outb, h, c, eps);
}
void hc_mix_rs(const float * x, const float * rs, const float * gamma, const float * logits, float * out, int h, int c, int n, cudaStream_t s) {
    shape(h, c, n);
    if (!x || !rs || !gamma || !logits || !out) throw std::invalid_argument("HC mix (rs): null pointer");
    const int count = h * n;
    mix_rs_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(x, rs, gamma, logits, out, h, c, count);
}
void hc_low_silu_bf16(float * x, __nv_bfloat16 * xb, int count, int c, cudaStream_t s) {
    if (!x || !xb || count <= 0 || c <= 0) throw std::invalid_argument("HC low-rank: invalid shape or pointer");
    low_bf16_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(x, xb, count, c);
}
void hc_low_silu(float * x, int count, int c, cudaStream_t s) {
    if (!x || count <= 0 || c <= 0) throw std::invalid_argument("HC low-rank: invalid shape or pointer");
    low_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(x, count, c);
}
void hc_mix(const float * norm, const float * logits, float * out, int h, int c, int n, cudaStream_t s) {
    shape(h, c, n);
    if (!norm || !logits || !out) throw std::invalid_argument("HC mix: null pointer");
    const int count = h * n;
    mix_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(norm, logits, out, h, c, count);
}
void hc_combine(const float * residual, const float * block, const float * inject, float * out,
                 int h, int c, int n, cudaStream_t s) {
    shape(h, c, n);
    if (!residual || !block || !inject || !out) throw std::invalid_argument("HC combine: null pointer");
    const int count = h * c * n;
    combine_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(residual, block, inject, out, h, c, count);
}
void ple_gate(const float * key, const float * query, float * gates, int h, int c, int n, cudaStream_t s) {
    shape(h, c, n);
    if (!key || !query || !gates) throw std::invalid_argument("PLE gate: null pointer");
    ple_gate_k<<<c * n, 256, 0, s>>>(key, query, gates, h);
}
void ple_broadcast(const float * value, const float * gates, float * out, int h, int c, int n, cudaStream_t s) {
    shape(h, c, n);
    if (!value || !gates || !out) throw std::invalid_argument("PLE broadcast: null pointer");
    const int count = h * c * n;
    ple_broadcast_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(value, gates, out, h, c, count);
}
void ple_conv(const float * x, const float * history, const float * kernel, float * out, float * next,
               int channels, int n, int taps, int dilation, cudaStream_t s) {
    const int64_t hist64 = (int64_t(taps) - 1) * dilation;
    if (channels <= 0 || n <= 0 || taps < 2 || dilation <= 0 || hist64 > std::numeric_limits<int>::max() ||
        hist64 * channels > std::numeric_limits<int>::max() || int64_t(channels) * n > std::numeric_limits<int>::max() ||
        int64_t(channels) * taps > std::numeric_limits<int>::max() || hist64 + n > std::numeric_limits<int>::max() ||
        !x || !history || !kernel || !out || !next || next == history || next == x || next == kernel ||
        next == out || out == x || out == history || out == kernel)
        throw std::invalid_argument("PLE convolution: invalid shape, pointer or aliased history");
    const int hist = int(hist64);
    ple_conv_k<<<(int64_t(channels) * n + 255) / 256, 256, 0, s>>>(x, history, kernel, out, channels, n, taps, dilation);
    ple_history_k<<<(int64_t(channels) * hist + 255) / 256, 256, 0, s>>>(x, history, next, channels, n, hist);
}
void ple_apply(const float * hidden, const float * gated, const float * conv, float * out, int count, cudaStream_t s) {
    if (count <= 0 || !hidden || !gated || !conv || !out) throw std::invalid_argument("PLE apply: invalid shape or pointer");
    ple_apply_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(hidden, gated, conv, out, count);
}
static int g_gdn_chunk_min = -1;
static int g_gdn_kh = 1;  // key-head prefill kernel (fn_gdn_set_kh(0): the per-token kernel, tests)
void fn_gdn_set_chunk_min(int n) { g_gdn_chunk_min = n; }
// grow-only GDN prefill workspaces (sized by the largest chunk), visible to the server's memory ledger and released after
// a prompt (decode steps are small)
static struct { float * work = nullptr; size_t cap = 0; float * gram = nullptr; size_t gcap = 0; float * wbuf = nullptr; size_t wcap = 0; } g_gdn_ws;
// grows a GDN workspace; inside a CUDA graph capture (decode graphs: the first decode step after a prefill released the
// workspaces) the buffer is taken outside the graph and the old one freed after the capture
static bool gdn_ws_grow(float *& p, size_t & cap, size_t need, cudaStream_t s) {
    static std::vector<float *> later;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone; cudaStreamIsCapturing(s, &cs);
    const bool capturing = cs != cudaStreamCaptureStatusNone;
    if (!capturing) { for (float * q : later) cudaFree(q); later.clear(); }
    if (p) { if (capturing) later.push_back(p); else cudaFree(p); }
    p = nullptr; cap = 0;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    if (capturing) cudaThreadExchangeStreamCaptureMode(&mode);
    const bool ok = cudaMalloc(&p, need * sizeof(float)) == cudaSuccess;
    if (capturing) cudaThreadExchangeStreamCaptureMode(&mode);
    if (ok) cap = need; else { cudaGetLastError(); p = nullptr; }
    return ok;
}
size_t fn_gdn_workspace_bytes() { return (g_gdn_ws.cap + g_gdn_ws.gcap + g_gdn_ws.wcap) * sizeof(float); }
void fn_gdn_release_workspaces() {
    cudaDeviceSynchronize();
    cudaFree(g_gdn_ws.work); cudaFree(g_gdn_ws.gram); cudaFree(g_gdn_ws.wbuf);
    g_gdn_ws.work = g_gdn_ws.gram = g_gdn_ws.wbuf = nullptr; g_gdn_ws.cap = g_gdn_ws.gcap = g_gdn_ws.wcap = 0;
}
void fn_gdn(const float * qkv, const float * alpha, const float * beta, const float * neg_a,
            const float * bias, const float * state, float * next, float * out,
            int kh, int vh, int dim, int n, float eps, cudaStream_t s) {
    if (kh <= 0 || vh <= 0 || vh % kh || dim < 32 || dim > 256 || dim % 32 || n <= 0 ||
        vh > 65535 || (int64_t(2) * kh + vh) * dim * n > std::numeric_limits<int>::max() ||
        int64_t(vh) * dim * dim > std::numeric_limits<int>::max() || !std::isfinite(eps) || eps <= 0.f ||
        !qkv || !alpha || !beta || !neg_a || !bias || !state || !next || !out ||
        next == qkv || next == alpha || next == beta || next == neg_a || next == bias || next == out ||
        out == qkv || out == alpha || out == beta || out == neg_a || out == bias || out == state)
        throw std::invalid_argument("Flash-Next GDN: invalid shape, pointer, epsilon or buffer alias");
    {
        float *& work = g_gdn_ws.work; size_t & cap = g_gdn_ws.cap;
        const size_t need = size_t(n) * 2 * kh * dim + size_t(n) * vh * 2;
        if (need > cap && !gdn_ws_grow(work, cap, need, s)) throw std::runtime_error("Flash-Next GDN: workspace allocation failed");
        float * qk = work, * gates = work + size_t(n) * 2 * kh * dim;
        fn_gdn_prep_k<<<dim3(n, kh), 32, 0, s>>>(qkv, qk, kh, vh, dim, eps);
        fn_gdn_gates_k<<<(n * vh + 127) / 128, 128, 0, s>>>(alpha, beta, neg_a, bias, gates, vh, n);
        // the key-head kernel (fn_gdn_kh_k) for prefills of 64+ tokens at the Flash-Next shape (dim 128,
        // 3 value heads per key head): one launch walks the sub-chunk with the state in registers
        if (g_gdn_kh > 0 && n >= 64 && dim == 128 && vh == 3 * kh) {
            fn_gdn_kh_k<3><<<unsigned(kh * (128 / gdnkh::CB)), dim3(gdnkh::CB, gdnkh::RG), 0, s>>>(qkv, qk, gates, state, next, out, kh, vh, n);
            return;
        }
        // TQFN_GDN_CHUNK_MIN (default 64; 0 = off): prefills of that many tokens use the chunked kernel (C 32, 32 value columns)
        if (g_gdn_chunk_min < 0) g_gdn_chunk_min = std::getenv("TQFN_GDN_CHUNK_MIN") ? std::atoi(std::getenv("TQFN_GDN_CHUNK_MIN")) : 64;  // WY chunked path beats the per-token kernel from 64 tokens
        const int chunk_min = g_gdn_chunk_min;
        if (chunk_min > 0 && n >= chunk_min && dim <= 128 && dim % 32 == 0) {
            constexpr int C = 32, VS = 32;
            const size_t smem = (size_t(VS) * (dim + 1) + 2 * size_t(C) * (dim + 1) + 2 * C * C + size_t(C) * VS + 4 * C) * sizeof(float);
            static bool attr = false;
            if (!attr) { cudaFuncSetAttribute(fn_gdn_chunk_k<C, VS>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)); attr = true; }
            const int chunks = (n + C - 1) / C;
            float *& gram = g_gdn_ws.gram; size_t & gcap = g_gdn_ws.gcap;
            const size_t gneed = size_t(kh) * chunks * 2 * C * C;
            if (gneed > gcap && !gdn_ws_grow(gram, gcap, gneed, s)) throw std::runtime_error("GDN chunk: gram allocation failed");
            static const bool wy = !std::getenv("TQFN_GDN_WY") || std::atoi(std::getenv("TQFN_GDN_WY")) != 0;
            if (wy) {  // parallel per-chunk WY terms, then the sequential state pass
                float *& wbuf = g_gdn_ws.wbuf; size_t & wcap = g_gdn_ws.wcap;
                const size_t per = size_t(vh) * chunks;
                const size_t wneed = per * C * dim * 2 + per * C * C + per * C;
                if (wneed > wcap && !gdn_ws_grow(wbuf, wcap, wneed, s)) throw std::runtime_error("GDN WY: allocation failed");
                float * U = wbuf, * W = U + per * C * dim, * Mg = W + per * C * dim, * lg = Mg + per * C * C;
                const size_t wsm = (3 * size_t(C) * (dim + 1) + 2 * size_t(C) * (C + 1) + 3 * C) * sizeof(float);
                const size_t ssm = (size_t(VS) * (dim + 1) + 3 * size_t(C) * (dim + 1) + size_t(C) * (VS + 1) + size_t(C) * C + 2 * C) * sizeof(float);
                static bool wattr = false;
                if (!wattr) {
                    cudaFuncSetAttribute(fn_gdn_wy_k<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(wsm));
                    cudaFuncSetAttribute(fn_gdn_seq_k<C, VS>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(ssm));
                    wattr = true;
                }
                fn_gdn_wy_k<C><<<dim3(vh, chunks), 256, wsm, s>>>(qkv, qk, gates, U, W, Mg, lg, kh, vh, dim, n);
                // TQFN_GDN_SEQ_FAST (default on): Omega's sequential pass (gdn_seq_fast.cuh) at dim 128; 0 = fn_gdn_seq_k
                static const bool seq_fast = !(std::getenv("TQFN_GDN_SEQ_FAST") && std::atoi(std::getenv("TQFN_GDN_SEQ_FAST")) == 0);
                if (seq_fast && dim == 128) fn_gdn_seq_fast(qk, U, W, Mg, lg, state, next, out, kh, vh, dim, n, chunks, s);
                else fn_gdn_seq_k<C, VS><<<dim3(vh, dim / VS), 256, ssm, s>>>(qk, U, W, Mg, lg, state, next, out, kh, vh, dim, n, chunks);
                return;
            }
            const size_t gsmem = 2 * size_t(C) * (dim + 1) * sizeof(float);
            fn_gdn_chunk_gram_k<C><<<dim3(kh, chunks), 256, gsmem, s>>>(qk, gram, kh, dim, n);
            fn_gdn_chunk_k<C, VS><<<dim3(vh, dim / VS), 256, smem, s>>>(qkv, qk, gates, gram, state, next, out, kh, vh, dim, n);
            return;
        }
        if (dim % 32 == 0) fn_gdn_main_k<8><<<dim3(vh, dim / 32), dim3(32, 4), 0, s>>>(qkv, qk, gates, state, next, out, kh, vh, dim, n);  // 8 value columns per warp
        else fn_gdn_main_k<1><<<dim3(vh, dim / 4), dim3(32, 4), 0, s>>>(qkv, qk, gates, state, next, out, kh, vh, dim, n);
    }
}
void fn_gdn_set_kh(int on) { g_gdn_kh = on; }
__global__ void h_to_f32_k(const __half * x, float * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < n) y[i] = __half2float(x[i]);
}
void h_to_f32(const __half * x, float * y, int64_t n, cudaStream_t s) {
    if (n > 0) h_to_f32_k<<<unsigned((n + 255) / 256), 256, 0, s>>>(x, y, n);
}
void fn_gated_rms(const float * x, const float * gate, const float * weight, float * out,
                  int heads, int dim, int n, float eps, cudaStream_t s) {
    if (heads <= 0 || dim <= 0 || n <= 0 || int64_t(heads) * dim > std::numeric_limits<int>::max() / n ||
        !std::isfinite(eps) || eps <= 0.f || !x || !gate || !weight || !out || out == weight)
        throw std::invalid_argument("Flash-Next gated RMS: invalid shape, pointer or epsilon");
    fn_gated_rms_k<<<heads * n, 32, 0, s>>>(x, gate, weight, out, dim, eps);
}
void fn_norm_rope(const float * x, const float * weight, float * out, int heads, int dim,
                  int n, int pos, int rot, float theta, float eps, bool interleaved, cudaStream_t s) {
    if (heads <= 0 || dim <= 0 || n <= 0 || int64_t(heads) * dim > std::numeric_limits<int>::max() / n ||
        pos < 0 || int64_t(pos) + n > std::numeric_limits<int>::max() || rot < 2 || rot > dim || rot % 2 ||
        !std::isfinite(theta) || theta <= 0.f || !std::isfinite(eps) || eps <= 0.f || !x || !weight || !out || out == x || out == weight)
        throw std::invalid_argument("Flash-Next norm/RoPE: invalid shape, position, pointer or epsilon");
    fn_norm_rope_k<<<heads * n, 32, 0, s>>>(x, weight, out, heads, dim, pos, rot, theta, eps, interleaved ? 2 : 1);
}
size_t fn_dense_attention_workspace(int heads, int dim, int n, int pos) {
    if (heads <= 0 || dim <= 0 || n <= 0 || pos < 0 || dim % 32 || dim > 256) return 0;
    const int chunk = 128, max_splits = (pos + n + chunk - 1) / chunk;
    const size_t per_query = size_t(heads) * max_splits * (dim + 2);
    return std::max<size_t>(1, std::min<size_t>(size_t(n), (size_t(64) << 20) / (per_query * sizeof(float)))) * per_query;
}
void fn_dense_attention(const float * q, const __nv_bfloat16 * k, const __nv_bfloat16 * v, float * out,
                        int heads, int kvheads, int dim, int n, int pos, cudaStream_t s, float * workspace) {
    if (heads <= 0 || heads > 65535 || kvheads <= 0 || heads % kvheads || dim <= 0 || dim > 256 || n <= 0 ||
        pos < 0 || int64_t(pos) + n > (1 << 20) || !q || !k || !v || !out || out == q)
        throw std::invalid_argument("Flash-Next dense attention: invalid shape or context exceeds QSA budget");
    if (dim % 32 == 0 && dim <= 256) {
        const int chunk = 128, max_splits = (pos + n + chunk - 1) / chunk;
        // queries in blocks so the split workspace stays <= 64 MB (each block is the same causal call at pos + b)
        const size_t per_query = size_t(heads) * max_splits * (dim + 2) * sizeof(float);
        const int block = int(std::max<size_t>(1, std::min<size_t>(size_t(n), (size_t(64) << 20) / per_query)));
        float * part = workspace;
        if (!part && cudaMallocAsync(reinterpret_cast<void **>(&part), size_t(block) * per_query, s) != cudaSuccess)
            throw std::runtime_error("Flash-Next attention: workspace allocation failed");
        for (int b = 0; b < n; b += block) {
            const int nb = std::min(block, n - b), splits = (pos + b + nb + chunk - 1) / chunk;
            const float * qb = q + int64_t(b) * heads * dim; float * ob = out + int64_t(b) * heads * dim;
            fn_attention_split_k<<<dim3(heads, nb, splits), 256, (chunk + 8 * dim) * sizeof(float), s>>>(qb, k, v, part, heads, kvheads, dim, pos + b, chunk, splits);
            fn_attention_merge_k<<<dim3(heads, nb), 128, 0, s>>>(part, ob, heads, dim, splits);
        }
        if (!workspace) cudaFreeAsync(part, s);
        return;
    }
    fn_attention_k<<<dim3(heads, n), 128, (pos + n) * sizeof(float), s>>>(q, k, v, out, heads, kvheads, dim, pos);
}
void fn_attention_gate(float * out, const float * qg, int heads, int dim, int n, cudaStream_t s) {
    if (heads <= 0 || dim <= 0 || n <= 0 || int64_t(heads) * dim > std::numeric_limits<int>::max() / n || !out || !qg || out == qg)
        throw std::invalid_argument("Flash-Next attention gate: invalid shape or pointer");
    const int count = heads * dim * n;
    fn_attention_gate_k<<<(int64_t(count) + 255) / 256, 256, 0, s>>>(out, qg, dim, count);
}
void fn_swiglu_bf16_h(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * o, int64_t n, cudaStream_t s) {
    if (n <= 0 || !g || !u || !o) throw std::invalid_argument("Flash-Next swiglu: invalid shape or pointer");
    fn_swiglu_bf16_h_k<<<(n + 255) / 256, 256, 0, s>>>(g, u, o, n);
}
void fn_swiglu_bf16_h_groups(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * o, const int * counts, const int * groups,
                             int slots, int capacity, int ff, cudaStream_t s) {
    if (slots <= 0 || slots > 65535 || capacity <= 0 || ff <= 0 || !g || !u || !o || !counts || !groups)
        throw std::invalid_argument("Flash-Next grouped swiglu: invalid shape or pointer");
    fn_swiglu_bf16_h_groups_k<<<dim3(unsigned(std::min(capacity, 64)), unsigned(slots)), 256, 0, s>>>(g, u, o, counts, groups, capacity, ff);
}
void fn_swiglu_f32(const float * g, const float * u, float * o, int64_t n, cudaStream_t s) {
    if (n <= 0 || !g || !u || !o) throw std::invalid_argument("Flash-Next swiglu: invalid shape or pointer");
    fn_swiglu_f32_k<<<(n + 255) / 256, 256, 0, s>>>(g, u, o, n);
}
void fn_moe_combine(const __nv_bfloat16 * r, const float * w, const float * sh, const float * sw, float * o, int h, int k, int n, cudaStream_t s) {
    if (h <= 0 || k <= 0 || n <= 0 || !r || !w || !sh || !sw || !o) throw std::invalid_argument("Flash-Next moe combine: invalid shape or pointer");
    fn_moe_combine_k<<<(int64_t(n) * h + 255) / 256, 256, 0, s>>>(r, w, sh, sw, o, h, k, n);
}
void fn_gemv_bf16(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || cols % 8 || !w || !x || !y) throw std::invalid_argument("Flash-Next bf16 gemv: invalid shape or pointer");
    const int warps = 8, blocks = (rows + warps - 1) / warps;
    fn_gemv_bf16_k<<<blocks, warps * 32, 0, s>>>(w, x, y, rows, cols);  // column splits measured slower at n=1 (launch cost)
}
// Several row-blocks of bf16 weights against the same few-token input in one launch (router + shared gate of this and
// the next layer). Each row is computed exactly as fn_gemv_bf16_few_k (same threads, same order); outputs go to
// y[s] + t * ld[s] + row.
template<int T> __global__ void fn_gemv_bf16_seg_k(GemvSegs g, const float * x, int cols) {
    int seg = 0; while (seg + 1 < g.n && int(blockIdx.x) >= g.start[seg + 1]) ++seg;
    const int row = blockIdx.x - g.start[seg], lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const uint4 * wr = reinterpret_cast<const uint4 *>(g.w[seg] + int64_t(row) * cols);
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.f;
    for (int v = threadIdx.x; v < cols / 8; v += blockDim.x) {
        const uint4 packed = wr[v];
        const __nv_bfloat16 * wv = reinterpret_cast<const __nv_bfloat16 *>(&packed);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xa = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v], xb = reinterpret_cast<const float4 *>(x + int64_t(t) * cols)[2 * v + 1];
            const float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            #pragma unroll
            for (int k = 0; k < 8; ++k) sum[t] += __bfloat162float(wv[k]) * __bfloat162float(__float2bfloat16_rn(xs[k]));
        }
    }
    __shared__ float part[T][8];
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        for (int d = 16; d; d >>= 1) sum[t] += __shfl_xor_sync(0xffffffffu, sum[t], d);
        if (!lane) part[t][warp] = sum[t];
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float r = 0.f;
        for (int k = 0; k < int(blockDim.x / 32); ++k) r += part[threadIdx.x][k];
        g.y[seg][int64_t(threadIdx.x) * g.ld[seg] + row] = r;
    }
}
void fn_gemv_bf16_segments(const GemvSegs & g, const float * x, int cols, int tokens, cudaStream_t s) {
    if (g.n <= 0 || g.n > kGemvSegs || cols <= 0 || cols % 8 || tokens <= 0 || tokens > 4 || !x) throw std::invalid_argument("bf16 gemv segments: invalid shape");
    const int thr = std::min(256, std::max(32, (cols / 8 / 4 + 31) / 32 * 32));
    const int total = g.start[g.n];
    if (tokens == 1) fn_gemv_bf16_seg_k<1><<<total, thr, 0, s>>>(g, x, cols);
    else if (tokens == 2) fn_gemv_bf16_seg_k<2><<<total, thr, 0, s>>>(g, x, cols);
    else if (tokens == 3) fn_gemv_bf16_seg_k<3><<<total, thr, 0, s>>>(g, x, cols);
    else fn_gemv_bf16_seg_k<4><<<total, thr, 0, s>>>(g, x, cols);
}
void fn_gemv_bf16_chunk(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, int tokens, int64_t ldy, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || cols % 8 || tokens <= 0 || ldy < rows || !w || !x || !y) throw std::invalid_argument("Flash-Next bf16 chunk gemv: invalid shape or pointer");
    if (!(rows <= 16 || (rows <= 1024 && cols >= 1024))) throw std::invalid_argument("Flash-Next bf16 chunk gemv: rows take the warp-per-row path");
    const int thr = std::min(256, std::max(32, (cols / 8 / 4 + 31) / 32 * 32));
    // Each (row, token) sum depends only on the thread count and the columns per thread, not on the R x T tile, so the
    // tile is free to choose (same bits); TQFN_GEMV_TILE picks it: 8 (8 x 8), 16 (16 x 8, default: 0.47 vs 0.56 ms per 2K-token call, same output hash), 164 (16 x 4), 32 (32 x 2:
    // measured slower, the weight rows are re-read per 2-token tile).
    static const int tile = std::getenv("TQFN_GEMV_TILE") ? std::atoi(std::getenv("TQFN_GEMV_TILE")) : 16;
    if (rows >= 32 && tile == 32) {
        constexpr int R = 32, T = 2;
        fn_gemv_bf16_tiles_k<R, T><<<dim3(unsigned((rows + R - 1) / R), unsigned((tokens + T - 1) / T)), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
        return;
    }
    if (rows >= 16 && tile == 16) {
        constexpr int R = 16, T = 8;
        fn_gemv_bf16_tiles_k<R, T><<<dim3(unsigned((rows + R - 1) / R), unsigned((tokens + T - 1) / T)), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
        return;
    }
    if (rows >= 16 && tile == 164) {
        constexpr int R = 16, T = 4;
        fn_gemv_bf16_tiles_k<R, T><<<dim3(unsigned((rows + R - 1) / R), unsigned((tokens + T - 1) / T)), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
        return;
    }
    constexpr int T = 8;
    const unsigned tiles = unsigned((tokens + T - 1) / T);
    if (rows >= 8) fn_gemv_bf16_tiles_k<8, T><<<dim3(unsigned((rows + 7) / 8), tiles), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
    else fn_gemv_bf16_tiles_k<1, T><<<dim3(unsigned(rows), tiles), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
}
// Top softmax probability of one logit row (1 / sum exp(l - max)): the MTP draft confidence (one block).
__global__ void fn_softmax_peak_k(const float * x, int n, float * out) {
    __shared__ float red[32]; const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    float m = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
    for (int d = 16; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    if (!lane) red[warp] = m; __syncthreads();
    if (tid < 32) { float v = tid < nw ? red[tid] : -INFINITY; for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d)); if (!tid) red[0] = v; }
    __syncthreads(); const float mx = red[0]; __syncthreads();
    float s = 0.f;
    for (int i = tid; i < n; i += blockDim.x) s += __expf(x[i] - mx);
    for (int d = 16; d; d >>= 1) s += __shfl_xor_sync(0xffffffffu, s, d);
    if (!lane) red[warp] = s; __syncthreads();
    if (tid < 32) { float v = tid < nw ? red[tid] : 0.f; for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d); if (!tid) *out = 1.f / v; }
}
void fn_softmax_peak(const float * x, int n, float * out, cudaStream_t s) {
    if (n <= 0 || !x || !out) throw std::invalid_argument("Flash-Next softmax peak: invalid shape or pointer");
    fn_softmax_peak_k<<<1, 1024, 0, s>>>(x, n, out);
}
void fn_round_bf16(const float * x, __nv_bfloat16 * xb, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    fn_round_bf16_k<<<unsigned((n / 4 + 255) / 256 + 1), 256, 0, s>>>(x, xb, n);
}
// fn_gemv_bf16_chunk with x from fn_round_bf16: same bits; 8 x 8 tiles (0.66 vs 0.79 ms for the 16 x 8 fp32-input kernel
// on the router's 512 x 2560 x 2048 tokens, RTX 3090)
void fn_gemv_bf16_chunk_bx(const __nv_bfloat16 * w, const __nv_bfloat16 * x, float * y, int rows, int cols, int tokens, int64_t ldy, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || cols % 8 || tokens <= 0 || ldy < rows || !w || !x || !y) throw std::invalid_argument("Flash-Next bf16 chunk gemv: invalid shape or pointer");
    if (!(rows <= 16 || (rows <= 1024 && cols >= 1024))) throw std::invalid_argument("Flash-Next bf16 chunk gemv: rows take the warp-per-row path");
    const int thr = std::min(256, std::max(32, (cols / 8 / 4 + 31) / 32 * 32));
    constexpr int T = 8;
    const unsigned tiles = unsigned((tokens + T - 1) / T);
    if (rows >= 8) fn_gemv_bf16x_tiles_k<8, T><<<dim3(unsigned((rows + 7) / 8), tiles), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
    else fn_gemv_bf16x_tiles_k<1, T><<<dim3(unsigned(rows), tiles), thr, 0, s>>>(w, x, y, rows, cols, tokens, ldy);
}
void fn_gemv_bf16_tokens(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, int tokens, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || cols % 8 || tokens <= 0 || !w || !x || !y) throw std::invalid_argument("Flash-Next bf16 gemv: invalid shape or pointer");
    const int warps = 8, blocks = (rows + warps - 1) / warps;
    // Block per row when a warp per row leaves the GPU half empty and each row is long enough to share.
    const bool few = rows <= 16 || (rows <= 1024 && cols >= 1024);
    const bool shrt = !few && cols <= 512 && rows >= 2048;
    const int thr = std::min(256, std::max(32, (cols / 8 / 4 + 31) / 32 * 32));
    for (int t0 = 0; t0 < tokens; t0 += 4) {
        const int k = std::min(4, tokens - t0); const float * xt = x + int64_t(t0) * cols; float * yt = y + int64_t(t0) * rows;
        if (few) {
            if (k == 1) fn_gemv_bf16_few_k<1><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, rows);
            else if (k == 2) fn_gemv_bf16_few_k<2><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, rows);
            else if (k == 3) fn_gemv_bf16_few_k<3><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, rows);
            else fn_gemv_bf16_few_k<4><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, rows);
        } else if (shrt) {
            const int sb = (rows * 8 + 255) / 256;
            if (k == 1) fn_gemv_bf16_short_k<1><<<sb, 256, 0, s>>>(w, xt, yt, rows, cols);
            else if (k == 2) fn_gemv_bf16_short_k<2><<<sb, 256, 0, s>>>(w, xt, yt, rows, cols);
            else if (k == 3) fn_gemv_bf16_short_k<3><<<sb, 256, 0, s>>>(w, xt, yt, rows, cols);
            else fn_gemv_bf16_short_k<4><<<sb, 256, 0, s>>>(w, xt, yt, rows, cols);
        } else if (k == 1) fn_gemv_bf16_multi_k<1><<<blocks, warps * 32, 0, s>>>(w, xt, yt, rows, cols);
        else if (k == 2) fn_gemv_bf16_multi_k<2><<<blocks, warps * 32, 0, s>>>(w, xt, yt, rows, cols);
        else if (k == 3) fn_gemv_bf16_multi_k<3><<<blocks, warps * 32, 0, s>>>(w, xt, yt, rows, cols);
        else fn_gemv_bf16_multi_k<4><<<blocks, warps * 32, 0, s>>>(w, xt, yt, rows, cols);
    }
}
void fn_gemv_bf16_tokens_ld(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, int tokens, int64_t ldy, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || cols % 8 || tokens <= 0 || ldy < rows || !w || !x || !y) throw std::invalid_argument("Flash-Next bf16 strided gemv: invalid shape or pointer");
    if (!(rows <= 16 || (rows <= 1024 && cols >= 1024))) throw std::invalid_argument("Flash-Next bf16 strided gemv: rows take the warp-per-row path");
    const int thr = std::min(256, std::max(32, (cols / 8 / 4 + 31) / 32 * 32));
    for (int t0 = 0; t0 < tokens; t0 += 4) {
        const int k = std::min(4, tokens - t0); const float * xt = x + int64_t(t0) * cols; float * yt = y + int64_t(t0) * ldy;
        if (k == 1) fn_gemv_bf16_few_k<1><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, ldy);
        else if (k == 2) fn_gemv_bf16_few_k<2><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, ldy);
        else if (k == 3) fn_gemv_bf16_few_k<3><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, ldy);
        else fn_gemv_bf16_few_k<4><<<rows, thr, 0, s>>>(w, xt, yt, rows, cols, ldy);
    }
}
} // namespace tqllm_cuda

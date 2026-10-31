// GLM-5.3-Flash (glm5-next) decode kernels; see glm5next_kernels.cuh. Same math as tools/glm_ref.cpp.
#include "glm5next_kernels.cuh"

#include <cfloat>
#include <stdexcept>

namespace tqllm_cuda {
namespace {

void check_launch() {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm5next kernel: ") + cudaGetErrorString(e));
}

__device__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// block-wide sum, every thread gets the result; blockDim.x multiple of 32, <= 1024
__device__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    t = warp_sum(t);
    return t;
}
__device__ float block_max(float v, float * red) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : -FLT_MAX;
    for (int o = 16; o; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
    return t;
}
__device__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ float silu_f(float x) { return x / (1.f + expf(-x)); }

// one warp per row
__global__ void mv_f32_k(const float * w, const float * x, float * y, int rows, int cols) {
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (r >= rows) return;
    const float * row = w + size_t(r) * cols, * xt = x + size_t(t) * cols;
    float acc = 0.f;
    for (int c = lane; c < cols; c += 32) acc += row[c] * xt[c];
    acc = warp_sum(acc);
    if (lane == 0) y[size_t(t) * rows + r] = acc;
}

// grid (rows, splits, n): block sums its column slice of one row
__global__ void mv_splitk_k(const float * w, const float * x, float * partial, int rows, int cols, int splits) {
    __shared__ float red[32];
    const int r = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int per = (cols + splits - 1) / splits, c0 = sp * per, c1 = min(cols, c0 + per);
    const float * row = w + size_t(r) * cols, * xt = x + size_t(t) * cols;
    float acc = 0.f;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x) acc += row[c] * xt[c];
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) partial[(size_t(t) * rows + r) * splits + sp] = acc;
}
__global__ void splitk_sum_k(const float * partial, float * y, int count, int splits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float v = 0.f;
    for (int sp = 0; sp < splits; ++sp) v += partial[size_t(i) * splits + sp];
    y[i] = v;
}

__global__ void gather_cols_k(const float * in, int stride, int off, int width, int count, float * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = in[size_t(i / width) * stride + off + i % width];
}

__global__ void bf16_to_f32_k(const __nv_bfloat16 * x, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}
__global__ void f32_to_half_k(const float * x, __half * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

// one block per row
__global__ void rmsnorm_k(const float * x, const float * w, float * y, __half * yh, int n, float eps) {
    __shared__ float red[32];
    const float * xr = x + size_t(blockIdx.x) * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float s = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] * s * (w ? w[i] : 1.f);
        if (y) y[size_t(blockIdx.x) * n + i] = v;
        if (yh) yh[size_t(blockIdx.x) * n + i] = __float2half(v);
    }
}

// short rows (per-head q / k norms): one warp per row, 8 rows per block
__global__ void rmsnorm_warp_k(const float * x, const float * w, float * y, __half * yh, int n, float eps, int rows) {
    const int r = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    const float * xr = x + size_t(r) * n;
    float ss = 0.f;
    for (int i = lane; i < n; i += 32) ss += xr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float s = rsqrtf(ss / n + eps);
    for (int i = lane; i < n; i += 32) {
        const float v = xr[i] * s * (w ? w[i] : 1.f);
        if (y) y[size_t(r) * n + i] = v;
        if (yh) yh[size_t(r) * n + i] = __float2half(v);
    }
}

// one block: the 24 mixes -> pre / post / comb, then x = sum pre[i] S[i]
__global__ void hc_pre_k(const float * mix, const float * scale, const float * base, const float * S, float * x,
                         float * post, float * comb, int hidden, int iters, float eps, int mix_stride) {
    constexpr int hc = 4;
    __shared__ float pre[hc], C[hc][hc];  // C[dst][src]
    const int t = blockIdx.x;
    mix += size_t(t) * mix_stride; S += size_t(t) * hc * hidden; x += size_t(t) * hidden; post += t * hc; comb += t * hc * hc;
    if (threadIdx.x == 0) {
        for (int i = 0; i < hc; ++i) {
            pre[i] = sigm(mix[i] * scale[0] + base[i]) + eps;
            post[i] = 2.f * sigm(mix[hc + i] * scale[1] + base[hc + i]);
        }
        for (int s = 0; s < hc; ++s) {
            float m = -FLT_MAX;
            for (int d = 0; d < hc; ++d) {
                C[d][s] = mix[2 * hc + d + hc * s] * scale[2] + base[2 * hc + d + hc * s];
                m = fmaxf(m, C[d][s]);
            }
            float z = 0.f;
            for (int d = 0; d < hc; ++d) z += (C[d][s] = expf(C[d][s] - m));
            for (int d = 0; d < hc; ++d) C[d][s] = C[d][s] / z + eps;
        }
        auto cols = [&] {
            for (int d = 0; d < hc; ++d) {
                float z = 0.f;
                for (int s = 0; s < hc; ++s) z += C[d][s];
                for (int s = 0; s < hc; ++s) C[d][s] /= z + eps;
            }
        };
        auto rows = [&] {
            for (int s = 0; s < hc; ++s) {
                float z = 0.f;
                for (int d = 0; d < hc; ++d) z += C[d][s];
                for (int d = 0; d < hc; ++d) C[d][s] /= z + eps;
            }
        };
        cols();
        for (int i = 1; i < iters; ++i) { rows(); cols(); }
        for (int d = 0; d < hc; ++d) for (int s = 0; s < hc; ++s) comb[d * hc + s] = C[d][s];
    }
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = 0.f;
        for (int i = 0; i < hc; ++i) v += pre[i] * S[size_t(i) * hidden + j];
        x[j] = v;
    }
}

__global__ void hc_post_k(const float * f, const float * S, const float * post, const float * comb, float * out, int hidden) {
    constexpr int hc = 4;
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= hc * hidden) return;
    const int d = i / hidden, j = i % hidden;
    f += size_t(t) * hidden; S += size_t(t) * hc * hidden; post += t * hc; comb += t * hc * hc; out += size_t(t) * hc * hidden;
    float v = f[j] * post[d];
    for (int s = 0; s < hc; ++s) v += S[size_t(s) * hidden + j] * comb[d * hc + s];
    out[i] = v;
}

__global__ void hc_mean_k(const float * S, float * x, int hidden) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (j >= hidden) return;
    S += size_t(t) * 4 * hidden; x += size_t(t) * hidden;
    float v = 0.f;
    for (int i = 0; i < 4; ++i) v += S[size_t(i) * hidden + j] / 4.f;
    x[j] = v;
}

// one thread per (projection, channel)
__global__ void kda_conv_k(const float * proj, float * hist, const float * w, float * out, int d, int K, int n, int ts, int ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * d) return;
    const int p = i / d, ch = i % d;
    float * h = hist + size_t(p) * (K - 1) * d;
    const float * ww = w + size_t(p) * d * K + size_t(ch) * K;
    for (int t = 0; t < n; ++t) {
        const size_t at = size_t(t) * ts + size_t(p) * ps + ch;
        const float x = proj[at];
        float acc = 0.f;
        for (int j = 0; j < K - 1; ++j) acc += h[size_t(j) * d + ch] * ww[j];
        acc += x * ww[K - 1];
        out[at] = silu_f(acc);
        for (int j = 0; j + 1 < K - 1; ++j) h[size_t(j) * d + ch] = h[size_t(j + 1) * d + ch];
        h[size_t(K - 2) * d + ch] = x;
    }
}

__global__ void kda_gate_k(float * g, const float * dtb, const float * A, float lb, int heads, int dim, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * dim * n) return;
    const int c = i % (heads * dim);
    g[i] = lb * sigm(-((g[i] + dtb[c]) * A[c / dim]));
}

__global__ void sigmoid_k(float * x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sigm(x[i]);
}

// one block per head, dim threads (dim <= 1024, multiple of 32); thread b owns state column b
__global__ void kda_recur_k(const float * q, const float * k, const float * v, const float * g, const float * beta,
                            const float * S, float * S_out, float * o, int heads, int dim, int n, int qkv_stride) {
    extern __shared__ float sh[];  // qs[dim], ks[dim], e[dim]
    __shared__ float red[32];
    float * qs = sh, * ks = sh + dim, * e = sh + 2 * dim;
    const int h = blockIdx.x, b = threadIdx.x;
    const float * Sr = S + size_t(h) * dim * dim;
    float * St = S_out ? S_out + size_t(h) * dim * dim : const_cast<float *>(S) + size_t(h) * dim * dim;
    if (S_out)  // the tokens update the copy, so the buffer they read from stays the pre-block state
        for (int a = 0; a < dim; ++a) St[size_t(a) * dim + b] = Sr[size_t(a) * dim + b];
    for (int t = 0; t < n; ++t) {
        const size_t qb = size_t(t) * qkv_stride + size_t(h) * dim, gb = (size_t(t) * heads + h) * dim;
        const float qv = q[qb + b], kv = k[qb + b];
        const float qn = block_sum(qv * qv, red), kn = block_sum(kv * kv, red);
        __syncthreads();
        qs[b] = qv * rsqrtf(qn + 1e-6f) * rsqrtf(float(dim));
        ks[b] = kv * rsqrtf(kn + 1e-6f);
        e[b] = expf(g[gb + b]);
        __syncthreads();
        float sk = 0.f;
        for (int a = 0; a < dim; ++a) {
            const float s = St[size_t(a) * dim + b] * e[a];
            St[size_t(a) * dim + b] = s;
            sk += s * ks[a];
        }
        const float dl = (v[qb + b] - sk) * beta[t * heads + h];
        float acc = 0.f;
        for (int a = 0; a < dim; ++a) {
            const float s = St[size_t(a) * dim + b] + ks[a] * dl;
            St[size_t(a) * dim + b] = s;
            acc += s * qs[a];
        }
        o[gb + b] = acc;
        __syncthreads();
    }
}

// DIM == blockDim.x: thread b keeps state column b in registers for the whole chunk (one read, one write of the
// state per call). Same operations in the same order as kda_recur_k.
template <int DIM>
__global__ void __launch_bounds__(DIM) kda_recur_reg_k(const float * q, const float * k, const float * v, const float * g,
                                                       const float * beta, const float * S, float * S_out, float * o,
                                                       int heads, int n, int qkv_stride) {
    __shared__ float qs[DIM], ks[DIM], e[DIM];
    __shared__ float red[32];
    const int h = blockIdx.x, b = threadIdx.x;
    const float * Sr = S + size_t(h) * DIM * DIM;
    float * St = S_out ? S_out + size_t(h) * DIM * DIM : const_cast<float *>(S) + size_t(h) * DIM * DIM;
    float col[DIM];
#pragma unroll
    for (int a = 0; a < DIM; ++a) col[a] = Sr[size_t(a) * DIM + b];
    for (int t = 0; t < n; ++t) {
        const size_t qb = size_t(t) * qkv_stride + size_t(h) * DIM, gb = (size_t(t) * heads + h) * DIM;
        const float qv = q[qb + b], kv = k[qb + b];
        const float qn = block_sum(qv * qv, red), kn = block_sum(kv * kv, red);
        __syncthreads();
        qs[b] = qv * rsqrtf(qn + 1e-6f) * rsqrtf(float(DIM));
        ks[b] = kv * rsqrtf(kn + 1e-6f);
        e[b] = expf(g[gb + b]);
        __syncthreads();
        float sk = 0.f;
#pragma unroll
        for (int a = 0; a < DIM; ++a) {
            col[a] = col[a] * e[a];
            sk += col[a] * ks[a];
        }
        const float dl = (v[qb + b] - sk) * beta[t * heads + h];
        float acc = 0.f;
#pragma unroll
        for (int a = 0; a < DIM; ++a) {
            col[a] = col[a] + ks[a] * dl;
            acc += col[a] * qs[a];
        }
        o[gb + b] = acc;
        __syncthreads();
    }
#pragma unroll
    for (int a = 0; a < DIM; ++a) St[size_t(a) * DIM + b] = col[a];
}

// one block per head
__global__ void kda_out_k(const float * o, const float * g2, const float * w, float * y, __half * yh, int dim, float eps) {
    __shared__ float red[32];
    const size_t base = size_t(blockIdx.x) * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += o[base + i] * o[base + i];
    ss = block_sum(ss, red);
    const float s = rsqrtf(ss / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = o[base + i] * s * w[i] * sigm(g2[base + i]);
        if (y) y[base + i] = v;
        if (yh) yh[base + i] = __float2half(v);
    }
}

// block per head, thread per latent channel j
__global__ void mla_qabs_k(const float * q, const __half * kb, float * qabs, int R, int dk) {
    extern __shared__ float qh[];
    const int h = blockIdx.x, row_id = blockIdx.x + blockIdx.y * gridDim.x;  // head; (token, head) row
    for (int i = threadIdx.x; i < dk; i += blockDim.x) qh[i] = q[size_t(row_id) * dk + i];
    __syncthreads();
    for (int j = threadIdx.x; j < R; j += blockDim.x) {
        const __half * row = kb + (size_t(h) * R + j) * dk;
        float acc = 0.f;
        for (int i = 0; i < dk; ++i) acc += __half2float(row[i]) * qh[i];
        qabs[size_t(row_id) * R + j] = acc;
    }
}

// block per head: scores over n positions (dynamic shared memory n floats + R floats)
__device__ __forceinline__ float lat_f(float x) { return x; }
__device__ __forceinline__ float lat_f(__half x) { return __half2float(x); }
template <class LT>
__global__ void mla_attn_k(const float * qabs, const LT * lat, float * kv, int R, int n0, float scale, const int * d_pos) {
    extern __shared__ float sh[];
    __shared__ float red[32];
    float * qa = sh, * p = sh + R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int h = blockIdx.x + blockIdx.y * gridDim.x;  // (token, head) row of qabs / kv
    const int n = (d_pos ? *d_pos + 1 : n0) + int(blockIdx.y);
    for (int j = threadIdx.x; j < R; j += blockDim.x) qa[j] = qabs[size_t(h) * R + j];
    __syncthreads();
    for (int t = warp; t < n; t += nw) {
        float acc = 0.f;
        for (int j = lane; j < R; j += 32) acc += qa[j] * lat_f(lat[size_t(t) * R + j]);
        acc = warp_sum(acc);
        if (lane == 0) p[t] = acc * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int t = threadIdx.x; t < n; t += blockDim.x) m = fmaxf(m, p[t]);
    m = block_max(m, red);
    float z = 0.f;
    for (int t = threadIdx.x; t < n; t += blockDim.x) z += (p[t] = expf(p[t] - m));
    z = block_sum(z, red);
    __syncthreads();
    for (int j = threadIdx.x; j < R; j += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < n; ++t) acc += p[t] * lat_f(lat[size_t(t) * R + j]);
        kv[size_t(h) * R + j] = acc / z;
    }
}

__global__ void copy_rows_at_k(const float * src, float * dst, int R, int count, const int * d_pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[size_t(*d_pos) * R + i] = src[i];
}
__global__ void copy_rows_at_h_k(const float * src, __half * dst, int R, int count, const int * d_pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[size_t(*d_pos) * R + i] = __float2half(src[i]);
}

__global__ void mla_vb_k(const float * kv, const __half * vb, float * out, __half * outh, int R, int dv) {
    extern __shared__ float kh[];
    const int h = blockIdx.x, row_id = blockIdx.x + blockIdx.y * gridDim.x;
    for (int j = threadIdx.x; j < R; j += blockDim.x) kh[j] = kv[size_t(row_id) * R + j];
    __syncthreads();
    for (int i = threadIdx.x; i < dv; i += blockDim.x) {
        const __half * row = vb + (size_t(h) * dv + i) * R;
        float acc = 0.f;
        for (int j = 0; j < R; ++j) acc += __half2float(row[j]) * kh[j];
        out[size_t(row_id) * dv + i] = acc;
        if (outh) outh[size_t(row_id) * dv + i] = __float2half(acc);
    }
}

// one block per token; experts <= 1024 (288 for GLM-5.3-Flash). Stable top-k: larger biased prob first, then lower
// index; each round is a block-wide argmax over (value, -index).
__global__ void route_k(const float * logits, const float * bias, int * ids, float * w, int experts, int k, bool norm,
                        float scale) {
    __shared__ float p[1024], sel[1024];
    __shared__ int chosen[64];
    __shared__ float rv[32];
    __shared__ int ri[32];
    logits += size_t(blockIdx.x) * experts; ids += blockIdx.x * k; w += blockIdx.x * k;
    for (int i = threadIdx.x; i < experts; i += blockDim.x) {
        p[i] = sigm(logits[i]);
        sel[i] = p[i] + (bias ? bias[i] : 0.f);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -FLT_MAX;
        int bi = 0x7fffffff;
        for (int i = threadIdx.x; i < experts; i += blockDim.x)
            if (sel[i] > bv || (sel[i] == bv && i < bi)) { bv = sel[i]; bi = i; }
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = rv[0];
            int ix = ri[0];
            for (int q = 1; q < nw; ++q)
                if (rv[q] > v || (rv[q] == v && ri[q] < ix)) { v = rv[q]; ix = ri[q]; }
            chosen[r] = ix;
            sel[ix] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += p[chosen[r]];
        sum = fmaxf(sum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            float v = p[chosen[r]];
            if (norm) v /= sum;
            if (scale != 0.f && scale != 1.f) v *= scale;
            ids[r] = chosen[r];
            w[r] = v;
        }
    }
}

__global__ void swiglu_k(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gg = __bfloat162float(g[i]), uu = __bfloat162float(u[i]);
    if (limit > 1e-6f) { gg = fminf(gg, limit); uu = fminf(fmaxf(uu, -limit), limit); }
    out[i] = __float2half(silu_f(gg) * uu);
}

__global__ void swiglu_f32_k(const float * g, const float * u, __half * out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gg = g[i], uu = u[i];
    if (limit > 1e-6f) { gg = fminf(gg, limit); uu = fminf(fmaxf(uu, -limit), limit); }
    out[i] = __float2half(silu_f(gg) * uu);
}

__global__ void moe_sum_k(const __nv_bfloat16 * y, const float * w, const float * shared, float * out, int k, int hidden) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (j >= hidden) return;
    y += size_t(t) * k * hidden; w += t * k; out += size_t(t) * hidden;
    if (shared) shared += size_t(t) * hidden;
    float v = shared ? shared[j] : 0.f;
    for (int i = 0; i < k; ++i) v += w[i] * __bfloat162float(y[size_t(i) * hidden + j]);
    out[j] = v;
}

__global__ void add_k(float * a, const float * b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void argmax_k(const float * x, int n, int * out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    x += size_t(blockIdx.x) * n; out += blockIdx.x;
    float v = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
    bv[threadIdx.x] = v; bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) { bv[threadIdx.x] = o; bi[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = bi[0];
}

int blocks(int n, int t) { return (n + t - 1) / t; }

}  // namespace

void glm_mv_f32(const float * w, const float * x, float * y, int rows, int cols, int n, cudaStream_t s) {
    mv_f32_k<<<dim3(blocks(rows, 8), n), 256, 0, s>>>(w, x, y, rows, cols);
    check_launch();
}
void glm_mv_f32_splitk(const float * w, const float * x, float * y, float * partial, int rows, int cols, int n, int splits,
                       cudaStream_t s) {
    mv_splitk_k<<<dim3(rows, splits, n), 256, 0, s>>>(w, x, partial, rows, cols, splits);
    check_launch();
    splitk_sum_k<<<blocks(rows * n, 256), 256, 0, s>>>(partial, y, rows * n, splits);
    check_launch();
}
void glm_gather_cols(const float * in, int stride, int off, int width, int n, float * out, cudaStream_t s) {
    gather_cols_k<<<blocks(width * n, 256), 256, 0, s>>>(in, stride, off, width, width * n, out);
    check_launch();
}
void glm_bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t s) {
    bf16_to_f32_k<<<blocks(n, 256), 256, 0, s>>>(x, y, n);
    check_launch();
}
void glm_f32_to_half(const float * x, __half * y, int n, cudaStream_t s) {
    f32_to_half_k<<<blocks(n, 256), 256, 0, s>>>(x, y, n);
    check_launch();
}
void glm_rmsnorm(const float * x, const float * w, float * y, __half * yh, int n, float eps, cudaStream_t s, int rows) {
    if (n <= 256) rmsnorm_warp_k<<<(rows + 7) / 8, 256, 0, s>>>(x, w, y, yh, n, eps, rows);
    else rmsnorm_k<<<rows, 1024, 0, s>>>(x, w, y, yh, n, eps);
    check_launch();
}
void glm_rmsnorm_rows(const float * x, const float * w, float * y, int rows, int n, float eps, cudaStream_t s) {
    rmsnorm_k<<<rows, 128, 0, s>>>(x, w, y, nullptr, n, eps);
    check_launch();
}
void glm_hc_pre(const float * mix, const float * scale, const float * base, const float * S, float * x, float * post,
                float * comb, int hidden, int iters, float eps, cudaStream_t s, int n, int mix_stride) {
    hc_pre_k<<<n, 1024, 0, s>>>(mix, scale, base, S, x, post, comb, hidden, iters, eps, mix_stride);
    check_launch();
}
void glm_hc_post(const float * f, const float * S, const float * post, const float * comb, float * out, int hidden,
                 cudaStream_t s, int n) {
    hc_post_k<<<dim3(blocks(4 * hidden, 256), n), 256, 0, s>>>(f, S, post, comb, out, hidden);
    check_launch();
}
void glm_hc_mean(const float * S, float * x, int hidden, cudaStream_t s, int n) {
    hc_mean_k<<<dim3(blocks(hidden, 256), n), 256, 0, s>>>(S, x, hidden);
    check_launch();
}
void glm_kda_conv(const float * proj, float * hist, const float * w, float * out, int d, int K, cudaStream_t s, int n,
                  int ts, int ps) {
    if (K < 2) throw std::invalid_argument("glm_kda_conv: kernel size < 2");
    if (!ts) { ts = d; ps = n * d; }
    kda_conv_k<<<blocks(3 * d, 256), 256, 0, s>>>(proj, hist, w, out, d, K, n, ts, ps);
    check_launch();
}
void glm_kda_gate(float * g, const float * dt_bias, const float * A, float lb, int heads, int dim, cudaStream_t s, int n) {
    kda_gate_k<<<blocks(heads * dim * n, 256), 256, 0, s>>>(g, dt_bias, A, lb, heads, dim, n);
    check_launch();
}
void glm_sigmoid(float * x, int n, cudaStream_t s) {
    sigmoid_k<<<blocks(n, 256), 256, 0, s>>>(x, n);
    check_launch();
}
void glm_kda_recur(const float * q, const float * k, const float * v, const float * g, const float * beta, const float * S,
                   float * S_out, float * o, int heads, int dim, cudaStream_t s, int n, int qkv_stride) {
    if (dim % 32 || dim > 1024) throw std::invalid_argument("glm_kda_recur: head dim must be a multiple of 32, <= 1024");
    const int stride = qkv_stride ? qkv_stride : 3 * heads * dim;
    if (dim == 128) kda_recur_reg_k<128><<<heads, 128, 0, s>>>(q, k, v, g, beta, S, S_out, o, heads, n, stride);
    else kda_recur_k<<<heads, dim, size_t(3) * dim * sizeof(float), s>>>(q, k, v, g, beta, S, S_out, o, heads, dim, n, stride);
    check_launch();
}
// history after m accepted tokens: row j = row (m + j) of (the K-1 old rows ++ the block's inputs), oldest first
__global__ void kda_hist_rb_k(const float * old_hist, const float * proj, float * hist, int d, int K, int m, int ts, int ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * d) return;
    const int p = i / d, ch = i % d;
    const float * oh = old_hist + size_t(p) * (K - 1) * d;
    float * h = hist + size_t(p) * (K - 1) * d;
    float keep[8];
    for (int j = 0; j < K - 1; ++j) {
        const int row = m + j;
        keep[j] = row < K - 1 ? oh[size_t(row) * d + ch] : proj[size_t(row - (K - 1)) * ts + size_t(p) * ps + ch];
    }
    for (int j = 0; j < K - 1; ++j) h[size_t(j) * d + ch] = keep[j];
}
void glm_kda_hist_rollback(const float * old_hist, const float * proj, float * hist, int d, int K, int m, cudaStream_t s,
                           int ts, int ps) {
    if (K < 2 || K > 9) throw std::invalid_argument("glm_kda_hist_rollback: kernel size must be in [2, 9]");
    if (old_hist == hist) throw std::invalid_argument("glm_kda_hist_rollback: the old history must be a separate buffer");
    kda_hist_rb_k<<<blocks(3 * d, 256), 256, 0, s>>>(old_hist, proj, hist, d, K, m, ts, ps);
    check_launch();
}
void glm_kda_out(const float * o, const float * g2, const float * w, float * y, __half * yh, int heads, int dim, float eps,
                 cudaStream_t s, int n) {
    kda_out_k<<<heads * n, 128, 0, s>>>(o, g2, w, y, yh, dim, eps);
    check_launch();
}
void glm_mla_qabs(const float * q, const __half * kb, float * qabs, int heads, int R, int dk, cudaStream_t s, int n) {
    mla_qabs_k<<<dim3(heads, n), 256, size_t(dk) * sizeof(float), s>>>(q, kb, qabs, R, dk);
    check_launch();
}
void glm_mla_attn(const float * qabs, const float * lat, float * kv, int heads, int R, int n, float scale, cudaStream_t s,
                  int tokens, const int * d_pos) {
    // with d_pos, n is the largest position count the caller allows (shared memory is sized for it)
    const size_t sh = size_t(R + n + tokens - 1) * sizeof(float);
    if (sh > 48 * 1024) throw std::invalid_argument("glm_mla_attn: context beyond the dense MLA path");
    mla_attn_k<float><<<dim3(heads, tokens), 256, sh, s>>>(qabs, lat, kv, R, n, scale, d_pos);
    check_launch();
}
void glm_mla_attn(const float * qabs, const __half * lat, float * kv, int heads, int R, int n, float scale, cudaStream_t s,
                  int tokens, const int * d_pos) {
    const size_t sh = size_t(R + n + tokens - 1) * sizeof(float);
    if (sh > 48 * 1024) throw std::invalid_argument("glm_mla_attn: context beyond the dense MLA path");
    mla_attn_k<__half><<<dim3(heads, tokens), 256, sh, s>>>(qabs, lat, kv, R, n, scale, d_pos);
    check_launch();
}
void glm_copy_rows_at(const float * src, float * dst, int R, int n, const int * d_pos, cudaStream_t s) {
    copy_rows_at_k<<<blocks(R * n, 256), 256, 0, s>>>(src, dst, R, R * n, d_pos);
    check_launch();
}
void glm_copy_rows_at(const float * src, __half * dst, int R, int n, const int * d_pos, cudaStream_t s) {
    copy_rows_at_h_k<<<blocks(R * n, 256), 256, 0, s>>>(src, dst, R, R * n, d_pos);
    check_launch();
}
void glm_mla_vb(const float * kv, const __half * vb, float * out, __half * outh, int heads, int R, int dv, cudaStream_t s,
                int n) {
    mla_vb_k<<<dim3(heads, n), 256, size_t(R) * sizeof(float), s>>>(kv, vb, out, outh, R, dv);
    check_launch();
}
void glm_route(const float * logits, const float * bias, int * ids, float * w, int experts, int k, bool norm, float scale,
               cudaStream_t s, int n) {
    if (experts > 1024 || k > 64 || k > experts) throw std::invalid_argument("glm_route: experts <= 1024, k <= 64");
    route_k<<<n, 256, 0, s>>>(logits, bias, ids, w, experts, k, norm, scale);
    check_launch();
}
void glm_swiglu(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * out, int n, float limit, cudaStream_t s) {
    swiglu_k<<<blocks(n, 256), 256, 0, s>>>(g, u, out, n, limit);
    check_launch();
}
void glm_swiglu_f32(const float * g, const float * u, __half * out, int n, float limit, cudaStream_t s) {
    swiglu_f32_k<<<blocks(n, 256), 256, 0, s>>>(g, u, out, n, limit);
    check_launch();
}
void glm_moe_sum(const __nv_bfloat16 * y, const float * w, const float * shared, float * out, int k, int hidden,
                 cudaStream_t s, int n) {
    moe_sum_k<<<dim3(blocks(hidden, 256), n), 256, 0, s>>>(y, w, shared, out, k, hidden);
    check_launch();
}
void glm_add(float * a, const float * b, int n, cudaStream_t s) {
    add_k<<<blocks(n, 256), 256, 0, s>>>(a, b, n);
    check_launch();
}
void glm_argmax(const float * x, int n, int * out, cudaStream_t s) {
    argmax_k<<<1, 1024, 0, s>>>(x, n, out);
    check_launch();
}
void glm_argmax_rows(const float * x, int n, int rows, int * out, cudaStream_t s) {
    argmax_k<<<rows, 1024, 0, s>>>(x, n, out);
    check_launch();
}
}  // namespace tqllm_cuda

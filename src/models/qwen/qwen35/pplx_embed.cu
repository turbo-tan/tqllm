#include "pplx_embed.hpp"

#include "gguf.hpp"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"

#include <cuda_bf16.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tqllm {
namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("pplx embed: ") + what + ": " + cudaGetErrorString(e));
}
void ck(cublasStatus_t e, const char * what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("pplx embed: ") + what + ": cuBLAS status " + std::to_string(int(e)));
}

constexpr uint32_t kF32 = 0, kQ8 = 8, kBF16 = 30, kTQ4 = 300;

__device__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// every thread of the block gets the sum (blockDim.x a multiple of 32, <= 1024)
__device__ float block_sum(float v, float * sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < nw ? sh[l] : 0.f;
    return warp_sum(v);
}
__device__ float block_max(float v, float * sh) {
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < nw ? sh[l] : -INFINITY;
    return warp_max(v);
}
__device__ float silu(float x) { return x / (1.f + expf(-x)); }

__global__ void k_embed(const void * table, bool f32, const uint32_t * ids, float * out, int H) {
    const size_t row = size_t(ids[blockIdx.x]) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        out[size_t(blockIdx.x) * H + c] = f32 ? static_cast<const float *>(table)[row + c]
                                              : __uint_as_float(uint32_t(static_cast<const uint16_t *>(table)[row + c]) << 16);
}

// y = x * rsqrt(mean(x^2) + eps) * w, one row of d per block; in place when y == x
__global__ void k_rmsnorm(const float * x, const float * w, float * y, int d, float eps) {
    __shared__ float sh[32];
    const float * r = x + size_t(blockIdx.x) * d;
    float s = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) s += r[c] * r[c];
    const float inv = rsqrtf(block_sum(s, sh) / d + eps);
    for (int c = threadIdx.x; c < d; c += blockDim.x) y[size_t(blockIdx.x) * d + c] = r[c] * inv * w[c];
}

// q_proj output [n][heads][2*hd] -> q [n][heads][hd], gate [n][heads][hd]
__global__ void k_split_qgate(const float * qg, float * q, float * gate, int heads, int hd) {
    const size_t t = blockIdx.x;
    for (int i = threadIdx.x; i < heads * hd; i += blockDim.x) {
        const int h = i / hd, c = i % hd;
        q[t * heads * hd + i] = qg[(t * heads + h) * 2 * hd + c];
        gate[t * heads * hd + i] = qg[(t * heads + h) * 2 * hd + hd + c];
    }
}

// rotate_half rope on the first `rot` dims of every head (x [n][heads][hd]); position = token index
__global__ void k_rope(float * x, const float * inv_freq, int heads, int hd, int rot) {
    const int t = blockIdx.x, half = rot / 2;
    for (int i = threadIdx.x; i < heads * half; i += blockDim.x) {
        const int h = i / half, j = i % half;
        float * v = x + (size_t(t) * heads + h) * hd;
        const float a = float(t) * inv_freq[j], c = cosf(a), s = sinf(a);
        const float x1 = v[j], x2 = v[j + half];
        v[j] = x1 * c - x2 * s;
        v[j + half] = x2 * c + x1 * s;
    }
}

__global__ void k_softmax_rows(float * s, int n) {
    __shared__ float sh[32];
    float * r = s + size_t(blockIdx.x) * n;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < n; c += blockDim.x) m = fmaxf(m, r[c]);
    m = block_max(m, sh);
    float z = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) { const float e = expf(r[c] - m); r[c] = e; z += e; }
    const float inv = 1.f / block_sum(z, sh);
    for (int c = threadIdx.x; c < n; c += blockDim.x) r[c] *= inv;
}

__global__ void k_sigmoid_mul(float * x, const float * gate, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        x[i] *= 1.f / (1.f + expf(-gate[i]));
}
__global__ void k_silu_mul(float * g, const float * u, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        g[i] = silu(g[i]) * u[i];
}
__global__ void k_add(float * x, const float * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) x[i] += y[i];
}

// causal depthwise conv (kernel K, zero history) + SiLU: out[t][c] = silu(sum_j w[c][j] * x[t - K + 1 + j][c])
__global__ void k_conv_silu(const float * x, const float * w, float * out, int n, int C, int K) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float s = 0.f;
        for (int j = 0; j < K; ++j) {
            const int src = t - K + 1 + j;
            if (src >= 0) s += w[size_t(c) * K + j] * x[size_t(src) * C + c];
        }
        out[size_t(t) * C + c] = silu(s);
    }
}

// per (token, key head), blockDim = dk: q, k L2-normalised (q also * dk^-1/2). conv [n][2*kh*dk + vh*dv] = q | k | v.
__global__ void k_gdn_qk(const float * conv, float * q, float * k, int kh, int vh, int dk, int dv) {
    __shared__ float sh[32];
    const int t = blockIdx.x / kh, h = blockIdx.x % kh, c = threadIdx.x;
    const size_t row = size_t(t) * (2 * kh * dk + vh * dv);
    const float qv = conv[row + h * dk + c], kv = conv[row + kh * dk + h * dk + c];
    const float qn = rsqrtf(block_sum(qv * qv, sh) + 1e-6f), kn = rsqrtf(block_sum(kv * kv, sh) + 1e-6f);
    q[(size_t(t) * kh + h) * dk + c] = qv * qn * rsqrtf(float(dk));
    k[(size_t(t) * kh + h) * dk + c] = kv * kn;
}
// per (token, value head), blockDim = dv: v copied, g = A * softplus(a + dt_bias), beta = sigmoid(b)
__global__ void k_gdn_vgb(const float * conv, const float * a, const float * b, const float * A, const float * dt_bias,
                          float * v, float * g, float * beta, int kh, int vh, int dk, int dv) {
    const int t = blockIdx.x / vh, h = blockIdx.x % vh, c = threadIdx.x;
    const size_t row = size_t(t) * (2 * kh * dk + vh * dv);
    v[(size_t(t) * vh + h) * dv + c] = conv[row + 2 * kh * dk + h * dv + c];
    if (c == 0) {
        const float x = a[size_t(t) * vh + h] + dt_bias[h];
        const float sp = x > 20.f ? x : log1pf(expf(x));
        g[size_t(t) * vh + h] = A[h] * sp;
        beta[size_t(t) * vh + h] = 1.f / (1.f + expf(-b[size_t(t) * vh + h]));
    }
}

// gated delta rule, sequential over tokens: one warp per (head, value column j); lane holds S[i][j] for i = lane + 32 r.
// S *= exp(g); kv = S^T k; S += k (beta (v - kv))^T; o = S^T q. Value head h reads key head h / (vh / kh).
template <int DK>
__global__ void k_gdn_scan(const float * q, const float * k, const float * v, const float * g, const float * beta,
                           float * o, int n, int kh, int heads, int dv) {
    constexpr int R = DK / 32;
    const int h = blockIdx.x, j = blockIdx.y * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int hk = h / (heads / kh);
    if (j >= dv) return;
    float S[R];
    for (int r = 0; r < R; ++r) S[r] = 0.f;
    for (int t = 0; t < n; ++t) {
        const size_t th = size_t(t) * heads + h, tk = size_t(t) * kh + hk;
        const float decay = expf(g[th]);
        float kv = 0.f, kr[R];
        for (int r = 0; r < R; ++r) {
            kr[r] = k[tk * DK + lane + 32 * r];
            S[r] *= decay;
            kv += S[r] * kr[r];
        }
        kv = warp_sum(kv);
        const float delta = (v[th * dv + j] - kv) * beta[th];
        float out = 0.f;
        for (int r = 0; r < R; ++r) {
            S[r] += kr[r] * delta;
            out += S[r] * q[tk * DK + lane + 32 * r];
        }
        out = warp_sum(out);
        if (lane == 0) o[th * dv + j] = out;
    }
}

// per (token, head): o = rmsnorm(o) * w * silu(z), blockDim = dv
__global__ void k_gated_norm(float * o, const float * z, const float * w, float eps) {
    __shared__ float sh[32];
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const float x = o[i];
    const float inv = rsqrtf(block_sum(x * x, sh) / blockDim.x + eps);
    o[i] = w[threadIdx.x] * (x * inv) * silu(z[i]);
}

// out[s][c] = mean of x[t][c] over span s
__global__ void k_pool(const float * x, const uint32_t * spans, float * out, int H) {
    const int s = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= H) return;
    const uint32_t b = spans[2 * s], e = spans[2 * s + 1];
    float sum = 0.f;
    for (uint32_t t = b; t < e; ++t) sum += x[size_t(t) * H + c];
    out[size_t(s) * H + c] = sum / float(e > b ? e - b : 1);
}

__global__ void k_to_bf16(const float * x, uint16_t * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const uint32_t u = __float_as_uint(x[i]);
        y[i] = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  // round to nearest even
    }
}

__global__ void k_embed_bf16(const void * table, bool f32, const uint32_t * ids, __nv_bfloat16 * out, int H) {
    const size_t row = size_t(ids[blockIdx.x]) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        out[size_t(blockIdx.x) * H + c] = f32 ? __float2bfloat16_rn(static_cast<const float *>(table)[row + c])
                                              : static_cast<const __nv_bfloat16 *>(table)[row + c];
}
// x = bf16(x + y): the residual add of a BF16 run, after the projection was rounded to BF16 (as HF does). Folding it into
// the GEMM epilogue (beta 1, one rounding) was 2 % faster but moved the worst chunk away from HF (min cosine 0.9977 ->
// 0.9902 vs HF bf16, 0.9964 -> 0.9890 vs HF fp32), so it fails the gate's min clause.
__global__ void k_add_bf16(__nv_bfloat16 * x, const __nv_bfloat16 * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
}
__global__ void k_pool_bf16(const __nv_bfloat16 * x, const uint32_t * spans, float * out, int H) {
    const int s = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= H) return;
    const uint32_t b = spans[2 * s], e = spans[2 * s + 1];
    float sum = 0.f;
    for (uint32_t t = b; t < e; ++t) sum += __bfloat162float(x[size_t(t) * H + c]);
    out[size_t(s) * H + c] = sum / float(e > b ? e - b : 1);
}
__global__ void k_bf16_to_f32(const __nv_bfloat16 * x, float * y, size_t n) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        y[i] = __bfloat162float(x[i]);
}

// TQ4 (GGUF 300) rows -> BF16 in the original basis: per 128 weights an FP16 scale and 64 bytes of 4-bit codes (low nibble =
// even column); w_rot = L[code] * scale (int8 Turbo4 levels), w = w_rot R with R = H diag(s1) / sqrt(128) (H Sylvester
// Hadamard, s1 the Turbo4 signs): a fast Walsh-Hadamard transform per group. One warp per group, 4 weights per lane.
__constant__ float kTq4Levels[16] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
__constant__ uint32_t kTq4Signs[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
__global__ void k_tq4_unrot(const uint8_t * w, size_t groups, __nv_bfloat16 * out) {
    const size_t g = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (g >= groups) return;
    const uint8_t * b = w + g * 66;
    const float sc = __half2float(*reinterpret_cast<const __half *>(b));
    const uint8_t c0 = b[2 + 2 * lane], c1 = b[3 + 2 * lane];
    float v[4] = {kTq4Levels[c0 & 15] * sc, kTq4Levels[c0 >> 4] * sc, kTq4Levels[c1 & 15] * sc, kTq4Levels[c1 >> 4] * sc};
    float a = v[0], c = v[1]; v[0] = a + c; v[1] = a - c;   // h = 1
    a = v[2]; c = v[3]; v[2] = a + c; v[3] = a - c;
    a = v[0]; c = v[2]; v[0] = a + c; v[2] = a - c;         // h = 2
    a = v[1]; c = v[3]; v[1] = a + c; v[3] = a - c;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {                       // h = 4 m: partner lane ^ m
        const bool hi = lane & m;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = hi ? o - v[e] : v[e] + o;
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int j = 4 * lane + e;
        const float s = (kTq4Signs[j / 32] >> (j % 32) & 1) ? -1.f : 1.f;
        out[g * 128 + j] = __float2bfloat16_rn(v[e] * s * 0.08838834764831845f);
    }
}

int grid_for(size_t n) { return int(std::min<size_t>((n + 255) / 256, 4096)); }

struct Buf {
    float * p = nullptr;
    size_t cap = 0;
    float * get(size_t n) {
        if (n > cap) {
            if (p) cudaFree(p);
            p = nullptr;
            cap = 0;  // a failed allocation leaves no buffer, not a null one that a smaller request would write through
            ck(cudaMalloc(&p, n * sizeof(float)), "allocate activations");
            cap = n;
        }
        return p;
    }
    ~Buf() { if (p) cudaFree(p); }
};

// a fused projection of the fast mode: resident BF16, or resident GGUF Q8_0 parts dequantized to BF16 just before its GEMM
struct FMat {
    __nv_bfloat16 * bf = nullptr;
    struct Part { uint32_t type; const uint8_t * data; int rows; };  // Q8_0, TQ4 (300) or BF16
    std::vector<Part> q8;  // quantized (or mixed) parts, decoded into the scratch before the GEMM
    int rows = 0, cols = 0;
};
// fast mode: one BF16 matrix per fused projection. Full attention: in = [q+gate | k | v]; GDN: in = [qkv | z | a | b].
struct FastLayer {
    bool full = false;
    int rin = 0;  // rows of `in`
    FMat in, out, gu, down;
    float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float *ssm_a = nullptr, *dt_bias = nullptr, *conv = nullptr, *ssm_norm = nullptr;
};

struct Layer {
    bool full = false;
    float *attn_norm = nullptr, *post_norm = nullptr, *ffn_gate = nullptr, *ffn_up = nullptr, *ffn_down = nullptr;
    float *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float *wqkv = nullptr, *wz = nullptr, *walpha = nullptr, *wbeta = nullptr, *wout = nullptr;
    float *ssm_a = nullptr, *dt_bias = nullptr, *conv = nullptr, *ssm_norm = nullptr;
};

class Embedder final : public PplxEmbedder {
public:
    Embedder(const std::string & path, int device, Mode mode) : bf16_(mode == Mode::bf16_products), fast_(mode == Mode::fast) {
        ck(cudaSetDevice(device), "set device");
        const auto f = gguf_open(path);
        if (!f.has("pplx.embedding_length")) throw std::runtime_error("not a contextual embedding GGUF (no pplx.* metadata): " + path);
        if (f.has("pplx.attention_causal") && f.num("pplx.attention_causal") != 0)
            throw std::runtime_error("pplx embed: causal attention is not implemented (pplx.attention_causal)");
        const std::string A = "qwen35.";
        L_ = int(f.num(A + "block_count")); H_ = int(f.num(A + "embedding_length")); I_ = int(f.num(A + "feed_forward_length"));
        nh_ = int(f.num(A + "attention.head_count")); nkv_ = int(f.num(A + "attention.head_count_kv"));
        hd_ = int(f.num(A + "attention.key_length")); rot_ = int(f.num(A + "rope.dimension_count"));
        eps_ = float(f.num(A + "attention.layer_norm_rms_epsilon"));
        conv_k_ = int(f.num(A + "ssm.conv_kernel")); dv_ = int(f.num(A + "ssm.state_size"));
        gh_ = int(f.num(A + "ssm.group_count")); vh_ = int(f.num(A + "ssm.time_step_rank")); dk_ = dv_;
        if (vh_ % gh_) throw std::runtime_error("pplx embed: GDN value heads are not a multiple of the key heads");
        if (dk_ != 128) throw std::runtime_error("pplx embed: GDN head dim " + std::to_string(dk_) + " (128 implemented)");
        const int interval = int(f.num(A + "full_attention_interval"));
        ctx_ = size_t(f.num(A + "context_length"));
        dim_ = int(f.num("pplx.embedding_length"));
        prompt_.document_prefix = f.str("pplx.document_prefix");
        prompt_.boundary_marker = f.str("pplx.boundary_marker");
        prompt_.query_prefix_id = uint32_t(f.num("pplx.query_prefix_token_id"));
        const double theta = f.num(A + "rope.freq_base");
        std::vector<float> inv(rot_ / 2);
        for (int i = 0; i < rot_ / 2; ++i) inv[i] = float(1.0 / std::pow(theta, double(2 * i) / rot_));
        inv_freq_ = upload(inv);

        const auto & te = tensor(f, "token_embd.weight");
        if ((te.type != kBF16 && te.type != kF32 && te.type != kQ8) || te.ne[0] != uint64_t(H_))
            throw std::runtime_error("pplx embed: token_embd must be BF16, F32 or Q8_0 [H, vocab]");
        if (te.type == kQ8 && !fast_) throw std::runtime_error("pplx embed: a Q8_0 model runs in the fast mode only");
        embed_f32_ = te.type == kF32;
        embed_q8_ = te.type == kQ8;
        vocab_ = te.ne[1];
        const auto raw = f.read(te);
        ck(cudaMalloc(&embed_, raw.size()), "allocate the token embedding");
        ck(cudaMemcpy(embed_, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload the token embedding");
        output_norm_ = weight(f, "output_norm.weight", {uint64_t(H_)});
        proj_ = weight(f, "contextual.projection.weight", {uint64_t(H_), uint64_t(dim_)});
        ck(cublasCreate(&blas_), "cublasCreate");
        ck(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH), "cuBLAS FP32 math");
        // the kernels launch on the calling thread's default stream (built with per-thread default streams); cuBLAS on its
        // NULL stream would use the legacy stream, which synchronises with it and serialises embedders on other threads
        ck(cublasSetStream(blas_, cudaStreamPerThread), "cuBLAS stream");
        if (fast_) { load_fast(f, interval); return; }
        layers_.resize(L_);
        for (int l = 0; l < L_; ++l) {
            auto & y = layers_[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            y.full = (l + 1) % interval == 0;
            const uint64_t H = H_, I = I_;
            y.attn_norm = weight(f, p + "attn_norm.weight", {H});
            y.post_norm = weight(f, p + "post_attention_norm.weight", {H});
            y.ffn_gate = mat(f, p + "ffn_gate.weight", {H, I});
            y.ffn_up = mat(f, p + "ffn_up.weight", {H, I});
            y.ffn_down = mat(f, p + "ffn_down.weight", {I, H});
            if (y.full) {
                y.wq = mat(f, p + "attn_q.weight", {H, uint64_t(2 * nh_ * hd_)});
                y.wk = mat(f, p + "attn_k.weight", {H, uint64_t(nkv_ * hd_)});
                y.wv = mat(f, p + "attn_v.weight", {H, uint64_t(nkv_ * hd_)});
                y.wo = mat(f, p + "attn_output.weight", {uint64_t(nh_ * hd_), H});
                y.q_norm = weight(f, p + "attn_q_norm.weight", {uint64_t(hd_)});
                y.k_norm = weight(f, p + "attn_k_norm.weight", {uint64_t(hd_)});
            } else {
                const uint64_t C = 2 * gh_ * dk_ + vh_ * dv_, V = uint64_t(vh_) * dv_;
                y.wqkv = mat(f, p + "attn_qkv.weight", {H, C});
                y.wz = mat(f, p + "attn_gate.weight", {H, V});
                y.walpha = mat(f, p + "ssm_alpha.weight", {H, uint64_t(vh_)});
                y.wbeta = mat(f, p + "ssm_beta.weight", {H, uint64_t(vh_)});
                y.wout = mat(f, p + "ssm_out.weight", {V, H});
                y.ssm_a = weight(f, p + "ssm_a", {uint64_t(vh_)});
                y.dt_bias = weight(f, p + "ssm_dt.bias", {uint64_t(vh_)});
                y.conv = weight(f, p + "ssm_conv1d.weight", {uint64_t(conv_k_), C});
                y.ssm_norm = weight(f, p + "ssm_norm.weight", {uint64_t(dv_)});
            }
        }
    }
    ~Embedder() override {
        if (blas_) cublasDestroy(blas_);
        if (blas_fast_) cublasDestroy(blas_fast_);
        for (void * p : {static_cast<void *>(conv_zero_), static_cast<void *>(gdn_zero_), static_cast<void *>(dconst_)})
            if (p) cudaFree(p);
        for (int j = 0; j < kSide; ++j) {
            if (conv_scratch_[j]) cudaFree(conv_scratch_[j]);
            if (gdn_scratch_[j]) cudaFree(gdn_scratch_[j]);
            if (side_[j]) cudaStreamDestroy(side_[j]);
            if (joined_[j]) cudaEventDestroy(joined_[j]);
        }
        if (forked_) cudaEventDestroy(forked_);
        for (auto * p : owned_) cudaFree(p);
        if (embed_) cudaFree(embed_);
    }

    const EmbedPrompt & prompt() const override { return prompt_; }
    int dim() const override { return dim_; }
    size_t max_tokens() const override { return ctx_; }

    std::vector<float> embed(const EmbedInput & in, std::vector<std::vector<float>> * layers) override {
        if (fast_) return std::move(forward_fast({&in}, layers)[0]);
        const size_t n = in.ids.size(), m = in.spans.size();
        if (n == 0 || m == 0) throw std::runtime_error("pplx embed: empty input");
        if (n > ctx_) throw std::runtime_error("pplx embed: " + std::to_string(n) + " tokens, the model takes " + std::to_string(ctx_));
        for (auto id : in.ids) if (id >= vocab_) throw std::runtime_error("pplx embed: token id out of range");
        for (auto [b, e] : in.spans) if (b >= e || e > n) throw std::runtime_error("pplx embed: bad span");
        const int H = H_;
        uint32_t * ids = reinterpret_cast<uint32_t *>(ids_.get((n + 2 * m + 1)));
        ck(cudaMemcpy(ids, in.ids.data(), n * 4, cudaMemcpyHostToDevice), "upload ids");
        uint32_t * spans = ids + n;
        ck(cudaMemcpy(spans, in.spans.data(), m * 8, cudaMemcpyHostToDevice), "upload spans");
        float * x = x_.get(n * H), * h = h_.get(n * H), * y = y_.get(n * H);
        k_embed<<<unsigned(n), 256>>>(embed_, embed_f32_, ids, x, H);
        ck(cudaGetLastError(), "embed");
        if (layers) layers->clear();
        for (const auto & ly : layers_) {
            k_rmsnorm<<<unsigned(n), 256>>>(x, ly.attn_norm, h, H, eps_);
            if (ly.full) attention(ly, h, y, n); else gdn(ly, h, y, n);
            k_add<<<grid_for(n * H), 256>>>(x, y, n * H);
            k_rmsnorm<<<unsigned(n), 256>>>(x, ly.post_norm, h, H, eps_);
            float * g = a_.get(n * I_), * u = b_.get(n * I_);
            gemm(h, ly.ffn_gate, g, n, H, I_);
            gemm(h, ly.ffn_up, u, n, H, I_);
            k_silu_mul<<<grid_for(n * I_), 256>>>(g, u, n * I_);
            gemm(g, ly.ffn_down, y, n, I_, H);
            k_add<<<grid_for(n * H), 256>>>(x, y, n * H);
            ck(cudaGetLastError(), "layer");
            if (layers) layers->push_back(download(x, n * H));
        }
        k_rmsnorm<<<unsigned(n), 256>>>(x, output_norm_, h, H, eps_);
        if (layers) layers->push_back(download(h, n * H));
        float * pooled = pool_.get(m * H);
        k_pool<<<dim3(unsigned(m), unsigned((H + 255) / 256)), 256>>>(h, spans, pooled, H);
        float * out = out_.get(m * dim_);
        gemm(pooled, proj_, out, m, H, dim_);
        ck(cudaGetLastError(), "pool");
        return download(out, m * size_t(dim_));
    }

    std::vector<std::vector<float>> embed_many(const std::vector<EmbedInput> & in) override {
        if (!fast_) return PplxEmbedder::embed_many(in);
        // first-fit decreasing by length into passes of at most pass_tokens_ (one longer input runs alone): fewer, fuller
        // passes than packing in arrival order (50 gate docs at 12K: 21 passes, none single, vs 26 with 7 single)
        std::vector<size_t> order(in.size());
        for (size_t i = 0; i < in.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return in[a].ids.size() > in[b].ids.size(); });
        std::vector<std::pair<size_t, std::vector<size_t>>> bins;  // tokens, inputs
        for (size_t i : order) {
            auto it = std::find_if(bins.begin(), bins.end(), [&](const auto & b) { return b.first + in[i].ids.size() <= pass_tokens_; });
            if (it == bins.end()) bins.push_back({in[i].ids.size(), {i}});
            else { it->first += in[i].ids.size(); it->second.push_back(i); }
        }
        std::vector<std::vector<float>> out(in.size());
        for (const auto & b : bins) {
            std::vector<const EmbedInput *> p;
            for (size_t i : b.second) p.push_back(&in[i]);
            auto v = forward_fast(p, nullptr);
            for (size_t k = 0; k < b.second.size(); ++k) out[b.second[k]] = std::move(v[k]);
        }
        return out;
    }

private:
    void check_input(const EmbedInput & in) const {
        const size_t n = in.ids.size();
        if (n == 0 || in.spans.empty()) throw std::runtime_error("pplx embed: empty input");
        if (n > ctx_) throw std::runtime_error("pplx embed: " + std::to_string(n) + " tokens, the model takes " + std::to_string(ctx_));
        for (auto id : in.ids) if (id >= vocab_) throw std::runtime_error("pplx embed: token id out of range");
        for (auto [b, e] : in.spans) if (b >= e || e > n) throw std::runtime_error("pplx embed: bad span");
    }

    // ---- fast mode -------------------------------------------------------------------------------------------------
    void load_fast(const GgufFile & f, int interval) {
        ck(cublasCreate(&blas_fast_), "cublasCreate");
        ck(cublasSetStream(blas_fast_, cudaStreamPerThread), "cuBLAS stream");
        const uint64_t H = H_, I = I_;
        const int qd = nh_ * hd_, kd = nkv_ * hd_, C = 2 * gh_ * dk_ + vh_ * dv_, V = vh_ * dv_;
        auto cat = [&](std::initializer_list<std::pair<std::string, std::vector<uint64_t>>> parts) {
            FMat mtx;
            std::vector<float> all;
            for (const auto & [name, ne] : parts) {
                const auto & t = tensor(f, name);
                if (t.ne != ne) throw std::runtime_error("pplx embed: " + name + " has shape " + gguf_shape_string(t));
                mtx.cols = int(ne[0]);
                mtx.rows += int(ne[1]);
                if (t.type == kQ8 || t.type == kTQ4) {
                    const auto raw = f.read(t);
                    uint8_t * d = nullptr;
                    ck(cudaMalloc(&d, raw.size()), "allocate quantized weights");
                    ck(cudaMemcpy(d, raw.data(), raw.size(), cudaMemcpyHostToDevice), "upload quantized weights");
                    owned_.push_back(reinterpret_cast<float *>(d));
                    mtx.q8.push_back({t.type, d, int(ne[1])});
                } else {
                    auto v = host(f, name, ne);
                    mtx.q8.push_back({kBF16, nullptr, int(ne[1])});
                    all.insert(all.end(), v.begin(), v.end());
                }
            }
            const bool quantized = std::any_of(mtx.q8.begin(), mtx.q8.end(), [](const FMat::Part & p) { return p.type != kBF16; });
            if (quantized) {  // BF16 parts of a mixed projection: resident BF16, copied into the scratch with the rest
                size_t at = 0;
                for (auto & p : mtx.q8)
                    if (p.type == kBF16) {
                        const size_t n = size_t(p.rows) * mtx.cols;
                        p.data = reinterpret_cast<const uint8_t *>(upload_bf16(std::vector<float>(all.begin() + at, all.begin() + at + n)));
                        at += n;
                    }
                for (const auto & p : mtx.q8) q8_rows_ = std::max(q8_rows_, p.rows);
                q8_elems_ = std::max(q8_elems_, size_t(mtx.rows) * mtx.cols);
                return mtx;
            }
            mtx.q8.clear();
            if (mtx.q8.empty()) mtx.bf = upload_bf16(all);
            return mtx;
        };
        fast_layers_.resize(L_);
        for (int l = 0; l < L_; ++l) {
            auto & y = fast_layers_[l];
            const std::string p = "blk." + std::to_string(l) + ".";
            y.full = (l + 1) % interval == 0;
            y.attn_norm = weight(f, p + "attn_norm.weight", {H});
            y.post_norm = weight(f, p + "post_attention_norm.weight", {H});
            y.gu = cat({{p + "ffn_gate.weight", {H, I}}, {p + "ffn_up.weight", {H, I}}});
            y.down = cat({{p + "ffn_down.weight", {I, H}}});
            if (y.full) {
                y.rin = 2 * qd + 2 * kd;
                y.in = cat({{p + "attn_q.weight", {H, uint64_t(2 * qd)}}, {p + "attn_k.weight", {H, uint64_t(kd)}},
                            {p + "attn_v.weight", {H, uint64_t(kd)}}});
                y.out = cat({{p + "attn_output.weight", {uint64_t(qd), H}}});
                y.q_norm = weight(f, p + "attn_q_norm.weight", {uint64_t(hd_)});
                y.k_norm = weight(f, p + "attn_k_norm.weight", {uint64_t(hd_)});
            } else {
                y.rin = C + V + 2 * vh_;
                y.in = cat({{p + "attn_qkv.weight", {H, uint64_t(C)}}, {p + "attn_gate.weight", {H, uint64_t(V)}},
                            {p + "ssm_alpha.weight", {H, uint64_t(vh_)}}, {p + "ssm_beta.weight", {H, uint64_t(vh_)}}});
                y.out = cat({{p + "ssm_out.weight", {uint64_t(V), H}}});
                y.ssm_a = weight(f, p + "ssm_a", {uint64_t(vh_)});
                y.dt_bias = weight(f, p + "ssm_dt.bias", {uint64_t(vh_)});
                y.conv = weight(f, p + "ssm_conv1d.weight", {uint64_t(conv_k_), uint64_t(C)});
                y.ssm_norm = weight(f, p + "ssm_norm.weight", {uint64_t(dv_)});
            }
        }
        if (conv_k_ != 4) throw std::runtime_error("pplx embed: the fast path's conv kernel is 4 taps");
        if (q8_rows_) {  // Q8_0 dequantization: row i of a part is "token" i of the row-gather kernel
            std::vector<uint32_t> idx(static_cast<size_t>(q8_rows_));
            for (int i = 0; i < q8_rows_; ++i) idx[size_t(i)] = uint32_t(i);
            ck(cudaMalloc(&rowidx_, idx.size() * 4), "Q8_0 row index");
            ck(cudaMemcpy(rowidx_, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice), "Q8_0 row index");
            ck(cudaMalloc(&wq_, q8_elems_ * 2), "Q8_0 dequantization scratch");
        }
        // per-segment zero history: conv state [3][C] BF16, GDN state [vh][dk][dv] FP32; device ints 0 (rope) and 1 (seq0)
        ck(cudaMalloc(&conv_zero_, size_t(3) * C * 2), "conv state"); ck(cudaMemset(conv_zero_, 0, size_t(3) * C * 2), "conv state");
        for (auto & p : conv_scratch_) ck(cudaMalloc(&p, size_t(3) * C * 2), "conv state");
        const size_t sb = size_t(vh_) * dk_ * dv_ * 4;
        ck(cudaMalloc(&gdn_zero_, sb), "gdn state"); ck(cudaMemset(gdn_zero_, 0, sb), "gdn state");
        for (auto & p : gdn_scratch_) ck(cudaMalloc(&p, sb), "gdn state");
        for (int j = 0; j < kSide; ++j) {
            ck(cudaStreamCreateWithFlags(&side_[j], cudaStreamNonBlocking), "segment streams");
            ck(cudaEventCreateWithFlags(&joined_[j], cudaEventDisableTiming), "segment events");
        }
        ck(cudaEventCreateWithFlags(&forked_, cudaEventDisableTiming), "segment events");
        const int consts[2] = {0, 1};
        ck(cudaMalloc(&dconst_, sizeof consts), "constants");
        ck(cudaMemcpy(dconst_, consts, sizeof consts, cudaMemcpyHostToDevice), "constants");
        theta_ = float(f.num("qwen35.rope.freq_base"));
    }
    __nv_bfloat16 * upload_bf16(const std::vector<float> & v) {
        std::vector<uint16_t> b(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            uint32_t u; std::memcpy(&u, &v[i], 4);
            b[i] = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
        __nv_bfloat16 * d = nullptr;
        ck(cudaMalloc(&d, b.size() * 2), "allocate BF16 weights");
        ck(cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice), "upload BF16 weights");
        owned_.push_back(reinterpret_cast<float *>(d));
        return d;
    }
    // Y[rows][out] = X[rows][in] W[out][in]^T, BF16 in and out, FP32 accumulate (tensor cores)
    // the BF16 matrix of a fused projection (Q8_0 parts dequantized into the shared scratch, stream 0, just before use)
    const __nv_bfloat16 * wget(const FMat & m) {
        if (m.bf) return m.bf;
        int r0 = 0;
        for (const auto & p : m.q8) {
            __nv_bfloat16 * dst = wq_ + size_t(r0) * m.cols;
            const size_t n = size_t(p.rows) * m.cols;
            if (p.type == kQ8) tqllm_cuda::embed_multi_q8_0(p.data, rowidx_, p.rows, dst, m.cols, 0);
            else if (p.type == kTQ4) k_tq4_unrot<<<unsigned((n / 128 * 32 + 255) / 256), 256>>>(p.data, n / 128, dst);
            else ck(cudaMemcpyAsync(dst, p.data, n * 2, cudaMemcpyDeviceToDevice, 0), "BF16 part");
            r0 += p.rows;
        }
        return wq_;
    }
    // beta 1: Y += X W^T (the residual add in the GEMM's FP32 epilogue, one BF16 rounding)
    void gemm_bf16(const __nv_bfloat16 * X, const __nv_bfloat16 * W, __nv_bfloat16 * Y, size_t rows, int in, int out, float beta = 0.f) {
        const float one = 1.f;
        ck(cublasGemmEx(blas_fast_, CUBLAS_OP_T, CUBLAS_OP_N, out, int(rows), in, &one, W, CUDA_R_16BF, in, X, CUDA_R_16BF, in,
                        &beta, Y, CUDA_R_16BF, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm (fast)");
    }
    // per-segment work (rope, attention, conv, GDN) of a multi-segment pass runs on kSide streams at once: one segment's
    // grids are small (a query: a few blocks), so a batch only fills the GPU when its segments overlap
    int fork(size_t segs) {
        if (segs < 2) return 0;
        const int used = int(std::min<size_t>(segs, kSide));
        ck(cudaEventRecord(forked_, 0), "fork");
        for (int j = 0; j < used; ++j) ck(cudaStreamWaitEvent(side_[j], forked_, 0), "fork");
        return used;
    }
    void join(int used) {
        for (int j = 0; j < used; ++j) {
            ck(cudaEventRecord(joined_[j], side_[j]), "join");
            ck(cudaStreamWaitEvent(0, joined_[j], 0), "join");
        }
    }
    __nv_bfloat16 * bget(Buf & b, size_t elems) { return reinterpret_cast<__nv_bfloat16 *>(b.get((elems + 1) / 2)); }

    // Segments: the inputs concatenated; GEMMs, norms and the MLP run once over every token, while rope positions, the
    // conv history, the GDN state and attention restart per segment.
    std::vector<std::vector<float>> forward_fast(const std::vector<const EmbedInput *> & ins, std::vector<std::vector<float>> * layers) {
        std::vector<size_t> off{0};
        size_t m = 0;
        for (const auto * in : ins) { check_input(*in); off.push_back(off.back() + in->ids.size()); m += in->spans.size(); }
        const size_t T = off.back();
        const int H = H_, qd = nh_ * hd_, kd = nkv_ * hd_, C = 2 * gh_ * dk_ + vh_ * dv_, V = vh_ * dv_;
        // buffers sized for a full pass on first use (a pass of pass_tokens_; only a longer single input grows them):
        // a growing buffer is freed and reallocated, and on GB10 a cudaFree of a large buffer stalls (up to 0.2 s each)
        const size_t Tc = std::max(T, pass_tokens_), mc = std::max<size_t>(m, 4096);
        uint32_t * ids = reinterpret_cast<uint32_t *>(ids_.get(Tc + 2 * mc + 1));
        std::vector<uint32_t> hid(T + 2 * m);
        size_t sp = T;
        for (size_t i = 0; i < ins.size(); ++i) {
            std::copy(ins[i]->ids.begin(), ins[i]->ids.end(), hid.begin() + off[i]);
            for (auto [b, e] : ins[i]->spans) { hid[sp++] = uint32_t(off[i] + b); hid[sp++] = uint32_t(off[i] + e); }
        }
        ck(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "upload ids");
        const uint32_t * spans = ids + T;
        __nv_bfloat16 * x = bget(x_, Tc * H), * h = bget(h_, Tc * H), * y = bget(y_, Tc * H);
        size_t rmax = size_t(2) * I_;
        for (const auto & ly : fast_layers_) rmax = std::max<size_t>(rmax, ly.rin);
        __nv_bfloat16 * big = bget(a_, Tc * rmax);
        if (embed_q8_) tqllm_cuda::embed_multi_q8_0(embed_, ids, int(T), x, H, 0);
        else k_embed_bf16<<<unsigned(T), 256>>>(embed_, embed_f32_, ids, x, H);
        ck(cudaGetLastError(), "embed");
        const int * dzero = dconst_, * done = dconst_ + 1;
        const float scale = 1.f / std::sqrt(float(hd_));
        if (layers) layers->clear();
        for (const auto & ly : fast_layers_) {
            tqllm_cuda::rms_f_multi(x, H, 0, ly.attn_norm, h, H, 0, H, 1, int(T), eps_, 0);
            gemm_bf16(h, wget(ly.in), big, T, H, ly.rin);
            const int R = ly.rin;
            if (ly.full) {
                __nv_bfloat16 * q = bget(q_, Tc * std::max(qd, C)), * k = bget(k_, Tc * std::max(kd, V)), * v = bget(v_, Tc * kd), * ao = bget(o_, Tc * std::max(qd, V));
                ck(cudaMemcpy2DAsync(v, size_t(kd) * 2, big + 2 * qd + kd, size_t(R) * 2, size_t(kd) * 2, T, cudaMemcpyDeviceToDevice, 0), "v");
                const int used = fork(ins.size());
                for (size_t i = 0; i < ins.size(); ++i) {
                    const size_t o = off[i];
                    const int n = int(off[i + 1] - o);
                    cudaStream_t st = used ? side_[i % used] : 0;
                    tqllm_cuda::rms_rope_multi(big + o * R, R, 2 * hd_, ly.q_norm, q + o * qd, qd, hd_, hd_, nh_, n, eps_, rot_, dzero, theta_, st);
                    tqllm_cuda::rms_rope_multi(big + o * R + 2 * qd, R, hd_, ly.k_norm, k + o * kd, kd, hd_, hd_, nkv_, n, eps_, rot_, dzero, theta_, st);
                    tqllm_cuda::attention_mq256_prefill_bidir(q + o * qd, k + o * kd, v + o * kd, ao + o * qd, done, n, nh_, nkv_, scale, st);
                }
                join(used);
                for (size_t t0 = 0; t0 < T; t0 += kGridY)  // grid y = token
                    tqllm_cuda::sigmoid_gate_multi(ao + t0 * qd, big + t0 * R, R, int(std::min(kGridY, T - t0)), nh_, hd_, 0);
                gemm_bf16(ao, wget(ly.out), y, T, qd, H); k_add_bf16<<<grid_for(T * H), 256>>>(x, y, T * H);
            } else {
                __nv_bfloat16 * cv = bget(q_, Tc * std::max(qd, C)), * gy = bget(o_, Tc * std::max(qd, V)), * gn = bget(k_, Tc * std::max(kd, V));
                const int used = fork(ins.size());
                for (size_t i = 0; i < ins.size(); ++i) {
                    const size_t o = off[i];
                    const int n = int(off[i + 1] - o), j = used ? int(i % used) : 0;
                    cudaStream_t st = used ? side_[j] : 0;
                    tqllm_cuda::conv_multi(big + o * R, R, n, conv_zero_, conv_scratch_[j], ly.conv, cv + o * C, C, st);
                    tqllm_cuda::gdn_multi(cv + o * C, n, big + o * R + C + V, big + o * R + C + V + vh_, R, ly.ssm_a, ly.dt_bias,
                                          gdn_zero_, gdn_scratch_[j], gy + o * V, gh_, vh_, dk_, dv_, false, st);
                }
                join(used);
                tqllm_cuda::gated_rms_multi(gy, big + C, R, int(T), ly.ssm_norm, gn, vh_, dv_, eps_, 0);
                gemm_bf16(gn, wget(ly.out), y, T, V, H); k_add_bf16<<<grid_for(T * H), 256>>>(x, y, T * H);
            }
            tqllm_cuda::rms_f_multi(x, H, 0, ly.post_norm, h, H, 0, H, 1, int(T), eps_, 0);
            gemm_bf16(h, wget(ly.gu), big, T, H, 2 * I_);
            __nv_bfloat16 * act = bget(b_, Tc * I_);
            for (size_t t0 = 0; t0 < T; t0 += kGridY)
                tqllm_cuda::swiglu_multi(big + t0 * 2 * I_, int(std::min(kGridY, T - t0)), I_, act + t0 * I_, 0);
            gemm_bf16(act, wget(ly.down), y, T, I_, H); k_add_bf16<<<grid_for(T * H), 256>>>(x, y, T * H);
            ck(cudaGetLastError(), "fast layer");
            if (layers) layers->push_back(download_bf16(x, T * H));
        }
        tqllm_cuda::rms_f_multi(x, H, 0, output_norm_, h, H, 0, H, 1, int(T), eps_, 0);
        if (layers) layers->push_back(download_bf16(h, T * H));
        float * pooled = pool_.get(mc * H);
        k_pool_bf16<<<dim3(unsigned(m), unsigned((H + 255) / 256)), 256>>>(h, spans, pooled, H);
        float * out = out_.get(mc * dim_);
        gemm(pooled, proj_, out, m, H, dim_);
        ck(cudaGetLastError(), "pool");
        const auto all = download(out, m * size_t(dim_));
        std::vector<std::vector<float>> res;
        size_t r = 0;
        for (const auto * in : ins) {
            const size_t k = in->spans.size() * size_t(dim_);
            res.emplace_back(all.begin() + r, all.begin() + r + k);
            r += k;
        }
        return res;
    }
    std::vector<float> download_bf16(const __nv_bfloat16 * d, size_t count) {
        float * f = s_.get(count);
        k_bf16_to_f32<<<grid_for(count), 256>>>(d, f, count);
        return download(f, count);
    }

    static const GgufTensor & tensor(const GgufFile & f, const std::string & name) {
        const auto it = f.tensors.find(name);
        if (it == f.tensors.end()) throw std::runtime_error("pplx embed: GGUF has no " + name);
        return it->second;
    }
    float * upload(const std::vector<float> & v) {
        float * d = nullptr;
        ck(cudaMalloc(&d, v.size() * 4), "allocate weights");
        ck(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload weights");
        owned_.push_back(d);
        return d;
    }
    float * weight(const GgufFile & f, const std::string & name, const std::vector<uint64_t> & ne) { return upload(host(f, name, ne)); }
    // F32 host copy of a BF16 / F32 tensor of shape ne (ne[0] contiguous)
    static std::vector<float> host(const GgufFile & f, const std::string & name, const std::vector<uint64_t> & ne) {
        const auto & t = tensor(f, name);
        if (t.ne != ne) throw std::runtime_error("pplx embed: " + name + " has shape " + gguf_shape_string(t));
        const auto raw = f.read(t);
        size_t count = 1;
        for (auto d : ne) count *= d;
        std::vector<float> v(count);
        if (t.type == kF32) {
            std::memcpy(v.data(), raw.data(), count * 4);
        } else if (t.type == kBF16) {
            const auto * b = reinterpret_cast<const uint16_t *>(raw.data());
            for (size_t i = 0; i < count; ++i) { const uint32_t u = uint32_t(b[i]) << 16; std::memcpy(&v[i], &u, 4); }
        } else {
            throw std::runtime_error("pplx embed: " + name + " must be BF16 or F32");
        }
        return v;
    }
    // a matrix: F32 copy, plus a BF16 one in bf16 mode (looked up by the F32 pointer in gemm)
    float * mat(const GgufFile & f, const std::string & name, const std::vector<uint64_t> & ne) {
        float * w = weight(f, name, ne);
        if (bf16_) {
            const size_t count = ne[0] * ne[1];
            uint16_t * b = nullptr;
            ck(cudaMalloc(&b, count * 2), "allocate BF16 weights");
            k_to_bf16<<<grid_for(count), 256>>>(w, b, count);
            ck(cudaGetLastError(), "BF16 weights");
            owned_.push_back(reinterpret_cast<float *>(b));
            bf16_of_[w] = b;
        }
        return w;
    }
    std::vector<float> download(const float * d, size_t count) {
        std::vector<float> v(count);
        ck(cudaMemcpy(v.data(), d, count * 4, cudaMemcpyDeviceToHost), "download");
        return v;
    }
    // Y[rows][out] = X[rows][in] W[out][in]^T
    void gemm(const float * X, const float * W, float * Y, size_t rows, int in, int out) {
        const float one = 1.f, zero = 0.f;
        if (const auto it = bf16_of_.find(W); it != bf16_of_.end()) {
            uint16_t * xb = reinterpret_cast<uint16_t *>(xb_.get((rows * in + 1) / 2));
            k_to_bf16<<<grid_for(rows * in), 256>>>(X, xb, rows * in);
            ck(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, out, int(rows), in, &one, it->second, CUDA_R_16BF, in, xb, CUDA_R_16BF, in,
                            &zero, Y, CUDA_R_32F, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm (BF16)");
            return;
        }
        ck(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, out, int(rows), in, &one, W, in, X, in, &zero, Y, out), "gemm");
    }

    void attention(const Layer & ly, const float * h, float * y, size_t n) {
        const int qd = nh_ * hd_, kd = nkv_ * hd_;
        float * qg = a_.get(n * std::max<size_t>(2 * qd, I_));
        float * q = q_.get(n * qd), * gate = gate_.get(n * qd), * k = k_.get(n * kd), * v = v_.get(n * kd);
        float * ao = o_.get(n * qd);
        gemm(h, ly.wq, qg, n, H_, 2 * qd);
        gemm(h, ly.wk, k, n, H_, kd);
        gemm(h, ly.wv, v, n, H_, kd);
        k_split_qgate<<<unsigned(n), 256>>>(qg, q, gate, nh_, hd_);
        k_rmsnorm<<<unsigned(n * nh_), 256>>>(q, ly.q_norm, q, hd_, eps_);
        k_rmsnorm<<<unsigned(n * nkv_), 256>>>(k, ly.k_norm, k, hd_, eps_);
        k_rope<<<unsigned(n), 256>>>(q, inv_freq_, nh_, hd_, rot_);
        k_rope<<<unsigned(n), 256>>>(k, inv_freq_, nkv_, hd_, rot_);
        // bidirectional: every query row sees all n keys; scores in row chunks of at most ~256 MB
        const size_t rows = std::max<size_t>(1, std::min<size_t>(n, (size_t(64) << 20) / n));
        float * s = s_.get(rows * n);
        const float scale = 1.f / std::sqrt(float(hd_)), one = 1.f, zero = 0.f;
        const int group = nh_ / nkv_;
        for (int hh = 0; hh < nh_; ++hh) {
            const float * kh = k + (hh / group) * hd_, * vh = v + (hh / group) * hd_;
            for (size_t r0 = 0; r0 < n; r0 += rows) {
                const int R = int(std::min(rows, n - r0));
                ck(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, int(n), R, hd_, &scale, kh, kd, q + r0 * qd + hh * hd_, qd,
                               &zero, s, int(n)), "attention scores");
                k_softmax_rows<<<unsigned(R), 256>>>(s, int(n));
                ck(cublasSgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, hd_, R, int(n), &one, vh, kd, s, int(n), &zero,
                               ao + r0 * qd + hh * hd_, qd), "attention values");
            }
        }
        k_sigmoid_mul<<<grid_for(n * qd), 256>>>(ao, gate, n * qd);
        gemm(ao, ly.wo, y, n, qd, H_);
        ck(cudaGetLastError(), "attention");
    }

    void gdn(const Layer & ly, const float * h, float * y, size_t n) {
        const int C = 2 * gh_ * dk_ + vh_ * dv_, V = vh_ * dv_;
        float * qkv = a_.get(n * std::max<size_t>(C, I_)), * conv = b_.get(n * std::max<size_t>(C, I_));
        float * z = gate_.get(n * std::max(V, nh_ * hd_));
        float * al = al_.get(n * vh_), * be = be_.get(n * vh_), * g = gg_.get(n * vh_), * beta = gb_.get(n * vh_);
        float * q = q_.get(n * std::max(gh_ * dk_, nh_ * hd_)), * k = k_.get(n * std::max(gh_ * dk_, nkv_ * hd_));
        float * v = v_.get(n * std::max(V, nkv_ * hd_)), * o = o_.get(n * std::max(V, nh_ * hd_));
        gemm(h, ly.wqkv, qkv, n, H_, C);
        gemm(h, ly.wz, z, n, H_, V);
        gemm(h, ly.walpha, al, n, H_, vh_);
        gemm(h, ly.wbeta, be, n, H_, vh_);
        k_conv_silu<<<unsigned(n), 256>>>(qkv, ly.conv, conv, int(n), C, conv_k_);
        k_gdn_qk<<<unsigned(n * gh_), dk_>>>(conv, q, k, gh_, vh_, dk_, dv_);
        k_gdn_vgb<<<unsigned(n * vh_), dv_>>>(conv, al, be, ly.ssm_a, ly.dt_bias, v, g, beta, gh_, vh_, dk_, dv_);
        constexpr int warps = 4;
        k_gdn_scan<128><<<dim3(vh_, (dv_ + warps - 1) / warps), 32 * warps>>>(q, k, v, g, beta, o, int(n), gh_, vh_, dv_);
        k_gated_norm<<<unsigned(n * vh_), dv_>>>(o, z, ly.ssm_norm, eps_);
        gemm(o, ly.wout, y, n, V, H_);
        ck(cudaGetLastError(), "gated delta net");
    }

    int L_ = 0, H_ = 0, I_ = 0, nh_ = 0, nkv_ = 0, hd_ = 0, rot_ = 0, conv_k_ = 0, gh_ = 0, vh_ = 0, dk_ = 0, dv_ = 0, dim_ = 0;
    float eps_ = 1e-6f;
    bool bf16_ = false, fast_ = false;
    std::vector<FastLayer> fast_layers_;
    cublasHandle_t blas_fast_ = nullptr;
    static constexpr int kSide = 8;
    // 12K tokens per pass: GB10 sweep over the 50 gate docs, 6K / 9K / 12K / 16K / 32K = 41.9 / 42.5 / 44.2 / 43.7 / 35.6K tok/s
    // (larger passes fall out of L2 in the elementwise kernels)
    static constexpr size_t kGridY = 65535;
    // 12K tokens per pass: GB10 sweep over the 50 gate docs (first-fit-decreasing packing), 12K / 16K = 51.4K / 50.0K tok/s
    // median; 6K .. 32K before packing 41.9 / 42.5 / 44.2 / 43.7 / 35.6K (larger passes fall out of L2)
    static constexpr size_t pass_tokens_ = 12288;
    __nv_bfloat16 * conv_zero_ = nullptr, * conv_scratch_[kSide] = {};
    float * gdn_zero_ = nullptr, * gdn_scratch_[kSide] = {}, theta_ = 0.f;
    cudaStream_t side_[kSide] = {};
    cudaEvent_t joined_[kSide] = {}, forked_ = nullptr;
    int * dconst_ = nullptr;
    std::unordered_map<const float *, uint16_t *> bf16_of_;
    size_t ctx_ = 0;
    uint64_t vocab_ = 0;
    EmbedPrompt prompt_;
    void * embed_ = nullptr;
    bool embed_f32_ = false, embed_q8_ = false;
    int q8_rows_ = 0;
    size_t q8_elems_ = 0;
    uint32_t * rowidx_ = nullptr;
    __nv_bfloat16 * wq_ = nullptr;
    float * output_norm_ = nullptr, * proj_ = nullptr, * inv_freq_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<float *> owned_;
    cublasHandle_t blas_ = nullptr;
    Buf xb_, ids_, pool_, out_, x_, h_, y_, a_, b_, q_, gate_, k_, v_, o_, s_, al_, be_, gg_, gb_;
};

}  // namespace

std::unique_ptr<PplxEmbedder> PplxEmbedder::load(const std::string & gguf_path, int device, Mode mode) {
    return std::make_unique<Embedder>(gguf_path, device, mode);
}

void pplx_tq4_to_bf16(const void * w, size_t groups, void * out) {  // unit test entry (tests/pplx_tq4.cu)
    k_tq4_unrot<<<unsigned((groups * 32 + 255) / 256), 256>>>(static_cast<const uint8_t *>(w), groups, static_cast<__nv_bfloat16 *>(out));
}

bool PplxEmbedder::is_embedding_gguf(const std::string & gguf_path) {
    if (gguf_path.size() < 5 || gguf_path.compare(gguf_path.size() - 5, 5, ".gguf") != 0) return false;
    return gguf_open(gguf_path).has("pplx.embedding_length");
}

}  // namespace tqllm

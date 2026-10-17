#include "vision.hpp"

#include "gguf.hpp"
#include "vision_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "stb/stb_image.h"
#pragma GCC diagnostic pop

namespace tqllm {

bool decode_image(const void * data, size_t bytes, std::vector<uint8_t> & rgb, int & w, int & h) {
    int c = 0;
    unsigned char * p = stbi_load_from_memory(static_cast<const unsigned char *>(data), int(bytes), &w, &h, &c, 3);
    if (!p) return false;
    rgb.assign(p, p + size_t(w) * h * 3);
    stbi_image_free(p);
    return true;
}

namespace {

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("vision: ") + cudaGetErrorString(e));
}
void cb(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("vision: cuBLAS error " + std::to_string(int(s)));
}

struct Dev {
    void * p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t bytes) : n(bytes) { ck(cudaMalloc(&p, bytes)); }
    Dev(Dev && o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
    Dev & operator=(Dev && o) noexcept {
        std::swap(p, o.p);
        std::swap(n, o.n);
        return *this;
    }
    Dev(const Dev &) = delete;
    ~Dev() {
        if (p) cudaFree(p);
    }
    template <class T> T * as() const { return static_cast<T *>(p); }
};

uint16_t f32_to_f16_bits(float f) {
    const __half h = __float2half_rn(f);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
float f16_bits_to_f32(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}

// host FP32 copy of an F32 / F16 tensor
std::vector<float> host_f32(const GgufFile & f, const std::string & name, size_t expect) {
    const auto it = f.tensors.find(name);
    if (it == f.tensors.end()) throw std::runtime_error("mmproj: missing tensor " + name);
    const auto raw = f.read(it->second);
    std::vector<float> out(expect);
    if (it->second.type == GGML_F32 && raw.size() == expect * 4) std::memcpy(out.data(), raw.data(), raw.size());
    else if (it->second.type == GGML_F16 && raw.size() == expect * 2)
        for (size_t i = 0; i < expect; ++i) {
            uint16_t b;
            std::memcpy(&b, raw.data() + 2 * i, 2);
            out[i] = f16_bits_to_f32(b);
        }
    else throw std::runtime_error("mmproj: unexpected type or size for " + name);
    return out;
}

struct Lin {  // y[n][out] = x[n][in] W^T + b; W F16 [out][in]
    Dev w, b;
    int in = 0, out = 0;
};

struct Block {
    Dev ln1w, ln1b, ln2w, ln2b;
    Lin qkv, o, up, down;
};

class Qwen3VL final : public VisionEncoder {
public:
    Qwen3VL(const GgufFile & f, cudaStream_t s) : st(s) {
        if (f.kv.count("clip.projector_type") == 0 || f.kv.at("clip.projector_type").str != "qwen3vl_merger")
            throw std::runtime_error("mmproj: only projector qwen3vl_merger is supported");
        n_embd = int(f.num("clip.vision.embedding_length"));
        n_ff = int(f.num("clip.vision.feed_forward_length"));
        n_layer = int(f.num("clip.vision.block_count"));
        n_head = int(f.num("clip.vision.attention.head_count"));
        patch = int(f.num("clip.vision.patch_size"));
        proj = int(f.num("clip.vision.projection_dim"));
        merge = int(f.num("clip.vision.spatial_merge_size", 2));
        eps = float(f.num("clip.vision.attention.layer_norm_epsilon", 1e-6));
        if (const auto it = f.kv.find("clip.vision.is_deepstack_layers"); it != f.kv.end())
            for (double v : it->second.arr)
                if (v != 0) throw std::runtime_error("mmproj: deepstack layers are not supported");
        if (merge != 2 || n_embd % n_head || (n_embd / n_head) % 4)
            throw std::runtime_error("mmproj: unexpected Qwen3-VL shape");
        for (int c = 0; c < 3; ++c) {
            if (const auto it = f.kv.find("clip.vision.image_mean"); it != f.kv.end() && it->second.arr.size() == 3)
                mean[c] = float(it->second.arr[c]);
            if (const auto it = f.kv.find("clip.vision.image_std"); it != f.kv.end() && it->second.arr.size() == 3)
                stdv[c] = float(it->second.arr[c]);
        }
        set_token_limits(8, 4096);
        cb(cublasCreate(&blas));
        cb(cublasSetStream(blas, st));

        // patch conv: both temporal kernels see the same frame -> one kernel W0 + W1, [n_embd][3 * patch * patch]
        const size_t kk = size_t(3) * patch * patch;
        auto w0 = host_f32(f, "v.patch_embd.weight", kk * n_embd), w1 = host_f32(f, "v.patch_embd.weight.1", kk * n_embd);
        for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];
        patch_w = up(w0);
        patch_b = up(host_f32(f, "v.patch_embd.bias", n_embd));
        const auto & pe = f.tensors.at("v.position_embd.weight");
        pos_side = int(std::lround(std::sqrt(double(pe.ne[1]))));
        pos_tab = host_f32(f, "v.position_embd.weight", size_t(n_embd) * pe.ne[1]);
        blk.resize(size_t(n_layer));
        for (int l = 0; l < n_layer; ++l) {
            const std::string p = "v.blk." + std::to_string(l) + ".";
            auto & B = blk[size_t(l)];
            B.ln1w = up(host_f32(f, p + "ln1.weight", n_embd));
            B.ln1b = up(host_f32(f, p + "ln1.bias", n_embd));
            B.ln2w = up(host_f32(f, p + "ln2.weight", n_embd));
            B.ln2b = up(host_f32(f, p + "ln2.bias", n_embd));
            B.qkv = lin(f, p + "attn_qkv", n_embd, 3 * n_embd);
            B.o = lin(f, p + "attn_out", n_embd, n_embd);
            B.up = lin(f, p + "ffn_up", n_embd, n_ff);
            B.down = lin(f, p + "ffn_down", n_ff, n_embd);
        }
        post_w = up(host_f32(f, "v.post_ln.weight", n_embd));
        post_b = up(host_f32(f, "v.post_ln.bias", n_embd));
        mm0 = lin(f, "mm.0", 4 * n_embd, 4 * n_embd);
        mm2 = lin(f, "mm.2", 4 * n_embd, proj);
    }
    ~Qwen3VL() override {
        if (blas) cublasDestroy(blas);
    }

    int embd() const override { return proj; }
    int align() const override { return patch * merge; }
    void set_token_limits(int mn, int mx) override {
        const int area = patch * patch * merge * merge;
        min_px = mn * area;
        max_px = mx * area;
    }

    VisionImage preprocess(const uint8_t * rgb, int w, int h) const override {
        if (w <= 0 || h <= 0) throw std::runtime_error("vision: empty image");
        const int a = align();
        auto rnd = [a](float x) { return int(std::round(x / float(a))) * a; };
        auto cel = [a](float x) { return int(std::ceil(x / float(a))) * a; };
        auto flr = [a](float x) { return int(std::floor(x / float(a))) * a; };
        int tw = std::max(a, rnd(float(w))), th = std::max(a, rnd(float(h)));
        if (tw * th > max_px) {
            const float beta = std::sqrt(float(h) * float(w) / float(max_px));
            th = std::max(a, flr(float(h) / beta));
            tw = std::max(a, flr(float(w) / beta));
        } else if (tw * th < min_px) {
            const float beta = std::sqrt(float(min_px) / (float(h) * float(w)));
            th = cel(float(h) * beta);
            tw = cel(float(w) * beta);
        }
        std::vector<uint8_t> dst(size_t(tw) * th * 3, 0);
        if (tw == w && th == h) {
            std::memcpy(dst.data(), rgb, dst.size());
        } else {  // aspect-preserving bilinear resize, centred on black (llama.cpp PAD_CEIL)
            const float scale = std::min(float(tw) / float(w), float(th) / float(h));
            const int nw = std::min(int(std::ceil(float(w) * scale)), tw), nh = std::min(int(std::ceil(float(h) * scale)), th);
            const int ox = (tw - nw) / 2, oy = (th - nh) / 2;
            const float xr = nw > 1 ? float(w - 1) / float(nw - 1) : 0.f, yr = nh > 1 ? float(h - 1) / float(nh - 1) : 0.f;
            auto lerp = [](float s, float e, float t) { return s + (e - s) * t; };
            for (int y = 0; y < nh; ++y)
                for (int x = 0; x < nw; ++x) {
                    const float px = float(x) * xr, py = float(y) * yr;
                    const int x0 = std::min(int(px), w - 1), y0 = std::min(int(py), h - 1);
                    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
                    const float xf = px - float(x0), yf = py - float(y0);
                    for (int c = 0; c < 3; ++c) {
                        auto at = [&](int xx, int yy) { return float(rgb[(size_t(yy) * w + xx) * 3 + c]); };
                        const float top = lerp(at(x0, y0), at(x1, y0), xf), bot = lerp(at(x0, y1), at(x1, y1), xf);
                        dst[(size_t(y + oy) * tw + x + ox) * 3 + c] = uint8_t(lerp(top, bot, yf));
                    }
                }
        }
        VisionImage im;
        im.w = tw;
        im.h = th;
        im.chw.resize(size_t(3) * tw * th);
        for (int c = 0; c < 3; ++c)
            for (size_t i = 0; i < size_t(tw) * th; ++i)
                im.chw[size_t(c) * tw * th + i] = (float(dst[i * 3 + c]) / 255.f - mean[c]) / stdv[c];
        return im;
    }

    const float * encode(const VisionImage & im, int & nx, int & ny) override {
        const int pw = im.w / patch, ph = im.h / patch;
        if (pw % merge || ph % merge || pw <= 0 || ph <= 0) throw std::runtime_error("vision: image not aligned");
        const int N = pw * ph, kk = 3 * patch * patch, D = n_embd, hd = D / n_head;
        nx = pw / merge;
        ny = ph / merge;
        // tokens in 2x2-block-major order (the merger concatenates 4 consecutive tokens): t = ((by * pw/2 + bx) * 2 + dy) * 2 + dx
        std::vector<float> cols(size_t(N) * kk), pos_e(size_t(N) * D);
        std::vector<int> pos(size_t(2) * N);
        const float sx = pw > 1 && pos_side > 1 ? float(pw - 1) / float(pos_side - 1) : float(pw) / float(pos_side);
        const float sy = ph > 1 && pos_side > 1 ? float(ph - 1) / float(pos_side - 1) : float(ph) / float(pos_side);
        const bool same = pw == pos_side && ph == pos_side;
        for (int by = 0; by < ph / 2; ++by)
            for (int bx = 0; bx < pw / 2; ++bx)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int t = ((by * (pw / 2) + bx) * 2 + dy) * 2 + dx, py = 2 * by + dy, px = 2 * bx + dx;
                        float * col = cols.data() + size_t(t) * kk;
                        for (int c = 0; c < 3; ++c)
                            for (int ky = 0; ky < patch; ++ky)
                                for (int kx = 0; kx < patch; ++kx)
                                    col[(c * patch + ky) * patch + kx] =
                                        im.chw[(size_t(c) * im.h + py * patch + ky) * im.w + px * patch + kx];
                        pos[2 * t] = py;
                        pos[2 * t + 1] = px;
                        float * pe = pos_e.data() + size_t(t) * D;
                        if (same) {
                            std::memcpy(pe, pos_tab.data() + (size_t(py) * pos_side + px) * D, size_t(D) * 4);
                            continue;
                        }
                        // bilinear, align corners (ggml_interpolate BILINEAR | ALIGN_CORNERS)
                        const float fx = float(px) / sx, fy = float(py) / sy;
                        int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
                        int x1 = std::clamp(x0 + 1, 0, pos_side - 1), y1 = std::clamp(y0 + 1, 0, pos_side - 1);
                        x0 = std::clamp(x0, 0, pos_side - 1);
                        y0 = std::clamp(y0, 0, pos_side - 1);
                        const float ddx = std::clamp(fx - float(x0), 0.f, 1.f), ddy = std::clamp(fy - float(y0), 0.f, 1.f);
                        const float * a = pos_tab.data() + (size_t(y0) * pos_side + x0) * D;
                        const float * b = pos_tab.data() + (size_t(y0) * pos_side + x1) * D;
                        const float * c = pos_tab.data() + (size_t(y1) * pos_side + x0) * D;
                        const float * d = pos_tab.data() + (size_t(y1) * pos_side + x1) * D;
                        for (int i = 0; i < D; ++i)
                            pe[i] = a[i] * (1 - ddx) * (1 - ddy) + b[i] * ddx * (1 - ddy) + c[i] * (1 - ddx) * ddy + d[i] * ddx * ddy;
                    }
        reserve(N);
        ck(cudaMemcpyAsync(d_cols.p, cols.data(), cols.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemcpyAsync(d_pos.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemcpyAsync(d_t.p, pos_e.data(), pos_e.size() * 4, cudaMemcpyHostToDevice, st));
        auto * X = d_x.as<float>();
        {  // X = cols W^T + bias + position embedding (FP32 GEMM)
            const float one = 1.f, zero = 0.f;
            cb(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, D, N, kk, &one, patch_w.as<float>(), kk, d_cols.as<float>(), kk,
                           &zero, X, D));
            tqllm_cuda::vis_bias(X, patch_b.as<float>(), N, D, false, st);
            tqllm_cuda::vis_add(X, d_t.as<float>(), int64_t(N) * D, st);
        }
        auto * T = d_t.as<float>();
        auto * QKV = d_qkv.as<float>();
        auto * A = d_a.as<float>();
        auto * F = d_f.as<float>();
        const float scale = 1.f / std::sqrt(float(hd));
        for (const auto & B : blk) {
            tqllm_cuda::vis_layer_norm(X, B.ln1w.as<float>(), B.ln1b.as<float>(), T, N, D, eps, st);
            linear(B.qkv, T, QKV, N, false);
            tqllm_cuda::vis_rope(QKV, d_pos.as<int>(), N, n_head, hd, 3 * D, 10000.f, st);
            attention(QKV, A, N, hd, scale);
            linear(B.o, A, T, N, false);
            tqllm_cuda::vis_add(X, T, int64_t(N) * D, st);
            tqllm_cuda::vis_layer_norm(X, B.ln2w.as<float>(), B.ln2b.as<float>(), T, N, D, eps, st);
            linear(B.up, T, F, N, true);
            linear(B.down, F, T, N, false);
            tqllm_cuda::vis_add(X, T, int64_t(N) * D, st);
        }
        tqllm_cuda::vis_layer_norm(X, post_w.as<float>(), post_b.as<float>(), T, N, D, eps, st);
        // merger: 4 consecutive tokens -> [N / 4][4D] -> mm.0 + GELU -> mm.2
        linear(mm0, T, F, N / 4, true);
        linear(mm2, F, d_out.as<float>(), N / 4, false);
        return d_out.as<float>();
    }

private:
    cudaStream_t st;
    cublasHandle_t blas = nullptr;
    int n_embd = 0, n_ff = 0, n_layer = 0, n_head = 0, patch = 0, proj = 0, merge = 2, pos_side = 0;
    int min_px = 0, max_px = 0;
    float eps = 1e-6f, mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
    Dev patch_w, patch_b, post_w, post_b;
    std::vector<float> pos_tab;
    std::vector<Block> blk;
    Lin mm0, mm2;
    int cap = 0;  // tokens the activation buffers hold
    Dev d_cols, d_pos, d_x, d_t, d_qkv, d_a, d_f, d_h, d_s, d_out;
    static constexpr int QB = 1024;  // attention query block (scores QB x N FP32)

    Dev up(const std::vector<float> & v) {
        Dev d(v.size() * 4);
        ck(cudaMemcpy(d.p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return d;
    }
    Lin lin(const GgufFile & f, const std::string & p, int in, int out) {
        Lin L;
        L.in = in;
        L.out = out;
        const auto w = host_f32(f, p + ".weight", size_t(in) * out);
        std::vector<uint16_t> h(w.size());
        for (size_t i = 0; i < w.size(); ++i) h[i] = f32_to_f16_bits(w[i]);
        L.w = Dev(h.size() * 2);
        ck(cudaMemcpy(L.w.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        L.b = up(host_f32(f, p + ".bias", size_t(out)));
        return L;
    }
    void reserve(int N) {
        if (N <= cap) return;
        const size_t D = size_t(n_embd), kk = size_t(3) * patch * patch;
        d_cols = Dev(size_t(N) * kk * 4);
        d_pos = Dev(size_t(N) * 2 * 4);
        d_x = Dev(size_t(N) * D * 4);
        d_t = Dev(size_t(N) * D * 4);
        d_qkv = Dev(size_t(N) * 3 * D * 4);
        d_a = Dev(size_t(N) * D * 4);
        d_f = Dev(size_t(N) * std::max<size_t>(n_ff, D) * 4);
        d_h = Dev(size_t(N) * std::max<size_t>(n_ff, 4 * D) * 2);
        d_s = Dev(size_t(QB) * N * 4);
        d_out = Dev(size_t(N / 4) * proj * 4);
        cap = N;
    }
    // y[n][out] = x[n][in] W^T + b (FP16 operands, FP32 accumulate and output), optional GELU
    void linear(const Lin & L, const float * x, float * y, int n, bool gelu) {
        tqllm_cuda::vis_f32_to_h(x, d_h.as<__half>(), int64_t(n) * L.in, st);
        const float one = 1.f, zero = 0.f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, L.out, n, L.in, &one, L.w.p, CUDA_R_16F, L.in, d_h.p, CUDA_R_16F,
                        L.in, &zero, y, CUDA_R_32F, L.out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        tqllm_cuda::vis_bias(y, L.b.as<float>(), n, L.out, gelu, st);
    }
    // full (non-causal) attention per head, query blocks of QB: out[n][D]
    void attention(const float * qkv, float * out, int N, int hd, float scale) {
        const int D = n_embd, ld = 3 * D;
        const float one = 1.f, zero = 0.f;
        float * S = d_s.as<float>();
        for (int h = 0; h < n_head; ++h)
            for (int q0 = 0; q0 < N; q0 += QB) {
                const int qb = std::min(QB, N - q0);
                const float * Q = qkv + size_t(q0) * ld + h * hd;
                const float * K = qkv + D + h * hd;
                const float * V = qkv + 2 * D + h * hd;
                cb(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, qb, hd, &scale, K, ld, Q, ld, &zero, S, N));
                tqllm_cuda::vis_softmax_rows(S, qb, N, st);
                cb(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, qb, N, &one, V, ld, S, N, &zero,
                               out + size_t(q0) * D + h * hd, D));
            }
    }
};

}  // namespace

std::unique_ptr<VisionEncoder> VisionEncoder::load(const std::string & mmproj, cudaStream_t stream) {
    const GgufFile f = gguf_open(mmproj);
    return std::make_unique<Qwen3VL>(f, stream);
}

}  // namespace tqllm

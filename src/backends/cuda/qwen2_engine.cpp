// Converted Qwen2 (DIR/weights.tqbin): the first engine, kept for its tests and tools.
#include "dev_alloc.cuh"
#include "tqllm/model.hpp"
#include "debug.hpp"
#include "../../engine/fit_context.hpp"
#include "dev_alloc.cuh"
#include "qwen2_kernels.cuh"
#include "cuda_engines.hpp"
#include "nlohmann/json.hpp"
#include "gguf.hpp"
#include "hf_folder.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tqllm {
namespace {

using bf16 = __nv_bfloat16;

void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

void cb(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS call failed: " + std::to_string(int(e)));
}

template <class T> T rd(std::istream & in) {
    T x{};
    in.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (!in) throw std::runtime_error("truncated TQ weights index");
    return x;
}

std::string rstr(std::istream & in, uint16_t n) {
    std::string s(n, '\0');
    in.read(s.data(), n);
    if (!in) throw std::runtime_error("truncated tensor name");
    return s;
}

// Owning device allocation.
struct Buf {
    void * p = nullptr;
    size_t n = 0;

    Buf() = default;
    explicit Buf(size_t bytes) : n(bytes) {
        if (bytes) ck(tqllm_cuda::dev_malloc(&p, bytes));
    }
    ~Buf() {
        if (p) tqllm_cuda::dev_free(p);
    }
    Buf(Buf && o) noexcept : p(std::exchange(o.p, nullptr)), n(std::exchange(o.n, 0)) {}
    Buf & operator=(Buf && o) noexcept {
        if (this != &o) {
            if (p) tqllm_cuda::dev_free(p);
            p = std::exchange(o.p, nullptr);
            n = std::exchange(o.n, 0);
        }
        return *this;
    }
    Buf(const Buf &) = delete;
    Buf & operator=(const Buf &) = delete;

    template <class T> T * as() { return static_cast<T *>(p); }
    template <class T> const T * as() const { return static_cast<const T *>(p); }
};

struct Weight {
    WeightInfo info;
    Buf device;
};

// Qwen2 hyperparameters, read from the Hugging Face config.json next to the weights.
struct Config {
    uint32_t vocab = 0, layers = 0, hidden = 0, heads = 0, kvheads = 0, dim = 0, ffn_dim = 0;
    float eps = 0.f, theta = 0.f;
    bool tied = true;
};

// Minimal reader for flat top-level scalar keys of config.json (numbers and booleans).
std::string json_scalar(const std::string & text, const std::string & key, bool required) {
    const auto k = text.find("\"" + key + "\"");
    if (k == std::string::npos) {
        if (required) throw std::runtime_error("config.json missing key: " + key);
        return {};
    }
    auto i = text.find(':', k);
    if (i == std::string::npos) throw std::runtime_error("config.json malformed at key: " + key);
    i = text.find_first_not_of(" \t\r\n", i + 1);
    const auto j = text.find_first_of(",}\r\n", i);
    return text.substr(i, j - i);
}

Config read_config(const std::filesystem::path & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open model config: " + path.string());
    std::stringstream ss;
    ss << in.rdbuf();
    const auto text = ss.str();
    auto u32 = [&](const char * key) { return uint32_t(std::stoul(json_scalar(text, key, true))); };
    auto f32 = [&](const char * key) { return std::stof(json_scalar(text, key, true)); };

    Config c;
    c.vocab = u32("vocab_size");
    c.layers = u32("num_hidden_layers");
    c.hidden = u32("hidden_size");
    c.heads = u32("num_attention_heads");
    c.kvheads = u32("num_key_value_heads");
    c.ffn_dim = u32("intermediate_size");
    c.eps = f32("rms_norm_eps");
    c.theta = f32("rope_theta");
    const auto head_dim = json_scalar(text, "head_dim", false);
    c.dim = head_dim.empty() ? c.hidden / c.heads : uint32_t(std::stoul(head_dim));
    c.tied = json_scalar(text, "tie_word_embeddings", false) != "false";
    if (!c.vocab || !c.layers || !c.hidden || !c.heads || !c.kvheads || !c.dim || !c.ffn_dim || c.heads % c.kvheads)
        throw std::runtime_error("invalid model config: " + path.string());
    return c;
}

// Per-layer weight pointers, resolved once at load.
struct Layer {
    const bf16 *attn_norm, *ffn_norm;
    const bf16 *wq, *bq, *wk, *bk, *wv, *bv, *wo;
    const bf16 *w_gate, *w_up, *w_down;
};

} // namespace

struct Qwen2Engine final : Engine {
    Config cfg;
    uint32_t context_, eos_ = 0;
    cudaStream_t stream{};
    cublasHandle_t blas{};
    std::vector<Weight> weights;
    std::unordered_map<std::string, size_t> by_name;
    const bf16 *embed = nullptr, *out_norm = nullptr, *lm_head = nullptr;
    std::vector<Layer> layer;
    std::vector<Buf> key_cache, value_cache;
    Buf x, norm, q, k, v, attn, proj, resid, gate, up, mlp, logits_buf, part, token;
    Tokenizer tokenizer;

    Qwen2Engine(const std::string & path, const std::string & tokpath, uint32_t ctx) : context_(ctx) {
        if (!context_) throw std::invalid_argument("context must be positive");
        cfg = read_config(std::filesystem::path(path).parent_path() / "config.json");
        cudaDeviceProp prop{};
        int dev = 0;
        ck(cudaGetDevice(&dev));
        ck(cudaGetDeviceProperties(&prop, dev));
        if (prop.major < 8) throw std::runtime_error("Qwen BF16 execution requires compute capability 8.0 or newer");
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            tokenizer.load(tokpath);
            eos_ = tokenizer.eos_id();
            cb(cublasCreate(&blas));
            cb(cublasSetStream(blas, stream));
            cb(cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH));
            load_weights(path);
            resolve_weights();
            allocate();
        } catch (...) {
            if (blas) cublasDestroy(blas);
            if (stream) cudaStreamDestroy(stream);
            throw;
        }
    }

    ~Qwen2Engine() override {
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }

    void load_weights(const std::string & path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open converted weights: " + path);
        char magic[8];
        in.read(magic, 8);
        if (!in || std::string(magic, 8) != "TQWGT001") throw std::runtime_error("invalid TQ weights file");

        const auto count = rd<uint32_t>(in);
        std::vector<WeightInfo> infos;
        infos.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            WeightInfo t;
            t.name = rstr(in, rd<uint16_t>(in));
            t.dtype = rd<uint8_t>(in);
            const auto rank = rd<uint8_t>(in);
            if (rank == 0 || rank > 8) throw std::runtime_error("invalid tensor rank: " + t.name);
            for (uint8_t d = 0; d < rank; ++d) t.shape.push_back(rd<uint64_t>(in));
            t.offset = rd<uint64_t>(in);
            t.bytes = rd<uint64_t>(in);
            uint64_t elems = 1;
            for (auto d : t.shape) {
                if (!d || elems > UINT64_MAX / d) throw std::runtime_error("invalid tensor shape: " + t.name);
                elems *= d;
            }
            const uint64_t type_bytes = t.dtype == 1 || t.dtype == 2 ? 2 : t.dtype == 3 ? 4 : 0;
            if (!type_bytes || elems > UINT64_MAX / type_bytes || elems * type_bytes != t.bytes)
                throw std::runtime_error("invalid tensor size: " + t.name);
            infos.push_back(std::move(t));
        }

        in.seekg(0, std::ios::end);
        const auto file_size = uint64_t(in.tellg());
        for (auto & t : infos) {
            if (t.offset > file_size || t.bytes > file_size - t.offset)
                throw std::runtime_error("tensor out of file bounds: " + t.name);
            if (by_name.contains(t.name)) throw std::runtime_error("duplicate tensor: " + t.name);
            if (t.dtype != 1) throw std::runtime_error("runtime currently requires BF16 tensor storage: " + t.name);
            Buf b(t.bytes);
            std::vector<char> host(t.bytes);
            in.seekg(t.offset);
            in.read(host.data(), t.bytes);
            if (!in) throw std::runtime_error("failed reading tensor: " + t.name);
            ck(cudaMemcpy(b.p, host.data(), t.bytes, cudaMemcpyHostToDevice));
            by_name.emplace(t.name, weights.size());
            weights.push_back({std::move(t), std::move(b)});
        }
    }

    const bf16 * tensor(const std::string & name, std::vector<uint64_t> shape) const {
        const auto it = by_name.find(name);
        if (it == by_name.end()) throw std::runtime_error("missing required tensor: " + name);
        const auto & w = weights[it->second];
        if (w.info.shape != shape) throw std::runtime_error("tensor shape differs from config.json: " + name);
        return w.device.as<bf16>();
    }

    void resolve_weights() {
        const uint64_t H = cfg.hidden, QD = uint64_t(cfg.heads) * cfg.dim, KD = uint64_t(cfg.kvheads) * cfg.dim;
        const uint64_t F = cfg.ffn_dim;
        embed = tensor("model.embed_tokens.weight", {cfg.vocab, H});
        out_norm = tensor("model.norm.weight", {H});
        lm_head = cfg.tied ? embed : tensor("lm_head.weight", {cfg.vocab, H});
        layer.resize(cfg.layers);
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            const auto p = "model.layers." + std::to_string(l) + ".";
            auto & L = layer[l];
            L.attn_norm = tensor(p + "input_layernorm.weight", {H});
            L.ffn_norm = tensor(p + "post_attention_layernorm.weight", {H});
            L.wq = tensor(p + "self_attn.q_proj.weight", {QD, H});
            L.bq = tensor(p + "self_attn.q_proj.bias", {QD});
            L.wk = tensor(p + "self_attn.k_proj.weight", {KD, H});
            L.bk = tensor(p + "self_attn.k_proj.bias", {KD});
            L.wv = tensor(p + "self_attn.v_proj.weight", {KD, H});
            L.bv = tensor(p + "self_attn.v_proj.bias", {KD});
            L.wo = tensor(p + "self_attn.o_proj.weight", {H, QD});
            L.w_gate = tensor(p + "mlp.gate_proj.weight", {F, H});
            L.w_up = tensor(p + "mlp.up_proj.weight", {F, H});
            L.w_down = tensor(p + "mlp.down_proj.weight", {H, F});
        }
    }

    void allocate() {
        const size_t H = cfg.hidden, QD = size_t(cfg.heads) * cfg.dim, KD = size_t(cfg.kvheads) * cfg.dim;
        x = Buf(H * 2);
        norm = Buf(H * 2);
        q = Buf(QD * 2);
        k = Buf(KD * 2);
        v = Buf(KD * 2);
        attn = Buf(QD * 2);
        proj = Buf(H * 2);
        resid = Buf(H * 2);
        gate = Buf(size_t(cfg.ffn_dim) * 2);
        up = Buf(size_t(cfg.ffn_dim) * 2);
        mlp = Buf(size_t(cfg.ffn_dim) * 2);
        logits_buf = Buf(size_t(cfg.vocab) * 4);
        part = Buf(size_t(tqllm_cuda::attention_partial_floats(int(cfg.heads), int(context_), int(cfg.dim))) * 4);
        token = Buf(sizeof(uint32_t));
        key_cache.reserve(cfg.layers);
        value_cache.reserve(cfg.layers);
        for (uint32_t i = 0; i < cfg.layers; ++i) {
            key_cache.emplace_back(size_t(context_) * KD * 2);
            value_cache.emplace_back(size_t(context_) * KD * 2);
        }
    }

    // y[out] = W[out, in] * x[in], BF16 in/out, FP32 accumulate.
    void linear(const bf16 * w, int out, int in, const bf16 * input, bf16 * output) {
        const float alpha = 1.f, beta = 0.f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, out, 1, in, &alpha, w, CUDA_R_16BF, in, input, CUDA_R_16BF, in,
                        &beta, output, CUDA_R_16BF, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    void linear_logits(const bf16 * input) {
        const float alpha = 1.f, beta = 0.f;
        cb(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, cfg.vocab, 1, cfg.hidden, &alpha, lm_head, CUDA_R_16BF,
                        cfg.hidden, input, CUDA_R_16BF, cfg.hidden, &beta, logits_buf.as<float>(), CUDA_R_32F, cfg.vocab,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Runs one token through the model at `pos`, writing K/V. When `sample` is false (prompt tokens before the
    // last) the LM head, argmax and host sync are skipped and the return value is 0.
    // Copies a device BF16 vector into dst as FP32 (synchronous; validation only).
    void to_host(const bf16 * src, size_t n, float * dst) {
        std::vector<bf16> tmp(n);
        ck(cudaMemcpyAsync(tmp.data(), src, n * sizeof(bf16), cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < n; ++i) dst[i] = __bfloat162float(tmp[i]);
    }

    uint32_t step(uint32_t id, uint32_t pos, bool sample, float * capture) override {
        if (id >= cfg.vocab || pos >= context_) throw std::runtime_error("token or context outside configured bounds");
        const int H = int(cfg.hidden), heads = int(cfg.heads), kvh = int(cfg.kvheads), dim = int(cfg.dim);
        const int F = int(cfg.ffn_dim), seq = int(pos + 1);
        const float scale = 1.f / std::sqrt(float(dim));
        auto * xp = x.as<bf16>();
        auto * np = norm.as<bf16>();
        auto * qp = q.as<bf16>();
        auto * kp = k.as<bf16>();
        auto * vp = v.as<bf16>();
        auto * ap = attn.as<bf16>();
        auto * pp = proj.as<bf16>();
        auto * rp = resid.as<bf16>();
        auto * sp = part.as<float>();

        tqllm_cuda::gather(embed, xp, int(id), H, stream);
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            const auto & L = layer[l];
            auto * kc = key_cache[l].as<bf16>();
            auto * vc = value_cache[l].as<bf16>();

            ck(cudaMemcpyAsync(rp, xp, size_t(H) * 2, cudaMemcpyDeviceToDevice, stream));
            tqllm_cuda::rms(xp, L.attn_norm, np, H, cfg.eps, stream);
            linear(L.wq, heads * dim, H, np, qp);
            tqllm_cuda::bias(qp, L.bq, qp, heads * dim, stream);
            linear(L.wk, kvh * dim, H, np, kp);
            tqllm_cuda::bias(kp, L.bk, kp, kvh * dim, stream);
            linear(L.wv, kvh * dim, H, np, vp);
            tqllm_cuda::bias(vp, L.bv, vp, kvh * dim, stream);
            tqllm_cuda::rope(qp, heads, dim, int(pos), cfg.theta, stream);
            tqllm_cuda::rope(kp, kvh, dim, int(pos), cfg.theta, stream);
            tqllm_cuda::cache_write(kp, vp, kc, vc, int(pos), kvh, dim, stream);
            tqllm_cuda::attention_decode(qp, kc, vc, ap, sp, seq, heads, kvh, dim, scale, stream);
            linear(L.wo, H, heads * dim, ap, pp);
            tqllm_cuda::residual(pp, rp, H, stream);

            tqllm_cuda::rms(pp, L.ffn_norm, np, H, cfg.eps, stream);
            linear(L.w_gate, F, H, np, gate.as<bf16>());
            linear(L.w_up, F, H, np, up.as<bf16>());
            tqllm_cuda::swiglu(gate.as<bf16>(), up.as<bf16>(), mlp.as<bf16>(), F, stream);
            linear(L.w_down, H, F, mlp.as<bf16>(), xp);
            tqllm_cuda::residual(xp, pp, H, stream);
            if (capture && l + 1 < cfg.layers) to_host(xp, H, capture + size_t(l) * H);
        }
        if (!sample) return 0;

        tqllm_cuda::rms(xp, out_norm, np, H, cfg.eps, stream);
        if (capture) to_host(np, H, capture + size_t(cfg.layers - 1) * H);
        linear_logits(np);
        tqllm_cuda::argmax(logits_buf.as<float>(), int(cfg.vocab), token.as<uint32_t>(), stream);
        uint32_t next = 0;
        ck(cudaMemcpyAsync(&next, token.p, sizeof(next), cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream));
        return next;
    }
    void verify(const uint32_t * tokens, int n, uint32_t pos, uint32_t * next) override {
        for (int i = 0; i < n; ++i) next[i] = step(tokens[i], pos + i, true, nullptr);
    }
    void commit(int) override {}  // attention-only model: KV beyond the accepted prefix is simply overwritten
    const float * logits() const override { return logits_buf.as<float>(); }
    void synchronize() override { ck(cudaStreamSynchronize(stream)); }
    uint32_t layers() const override { return cfg.layers; }
    uint32_t hidden() const override { return cfg.hidden; }
    uint32_t vocab() const override { return cfg.vocab; }
    uint32_t eos() const override { return eos_; }
    uint32_t context() const override { return context_; }
};

std::unique_ptr<Engine> make_qwen2_engine(const std::string & weights, const std::string & tokenizer, uint32_t context) {
    return std::make_unique<Qwen2Engine>(weights, tokenizer, context);
}

}  // namespace tqllm

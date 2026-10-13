#include "dev_alloc.cuh"
#include "tqllm/model.hpp"
#include "debug.hpp"
#include "fit_context.hpp"
#include "dev_alloc.cuh"
#include "qwen2_kernels.cuh"
#include "engine.hpp"
#include "json.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
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

struct Model::Impl {
    std::unique_ptr<Engine> e;
};

namespace {
std::string model_type(const std::filesystem::path & dir) {
    std::ifstream in(dir / "config.json");
    if (!in) throw std::runtime_error("cannot open model config: " + (dir / "config.json").string());
    std::stringstream ss;
    ss << in.rdbuf();
    const Json root = parse_json(ss.str());
    const Json * t = root.get("model_type");
    return t ? t->str : std::string();
}
} // namespace

namespace {
// Builds the engine leaving o.fit_target_mb (-fitt) of device memory free for run-time allocations (library workspaces,
// graphs; ~0.4 GB measured on a 3090 at 32K). A shortfall or an out-of-memory load applies the next step and reloads:
// embeddings, then the int8 LM head to host memory, smaller prefill buffers, the turbo3 V cache, prefill weight blocks,
// then half the context (not below -fitc). Options the user set are left alone; -fit off loads once.
std::unique_ptr<Engine> load_fitted(const std::string & path, const std::string & tok, uint32_t ctx, int slots,
                                    EngineOptions opt) {
    // WSL / Windows (WDDM): cudaMemGetInfo reads ~0 near the budget, and 0.30 GB free at 262K ran at full speed with
    // no spill on a 5060 Ti (results/20261008_kv_host_tail), so the default margin and the floor are 256 MB there
    static const bool wsl = [] {
        std::ifstream v("/proc/version");
        std::string l;
        std::getline(v, l);
        return l.find("icrosoft") != std::string::npos;
    }();
    // parallel slots: each needs --slot-reserve MiB free after loading (its batched-verify / draft graphs); the fit makes
    // that room first (host placement, KV to host, context), then refuses an --np the memory cannot carry
    const size_t base_margin = fit_margin(opt.fit, opt.fit_target_mb, wsl);
    const size_t slot_b = slots > 1 ? size_t(std::max(0L, opt.slot_reserve_mb)) << 20 : 0;
    const size_t margin = base_margin ? base_margin + size_t(slots) * slot_b : 0;
    if (wsl) opt.graph_keep_mb = 0;  // the driver reads ~0 near the budget: the graph cache bounds itself by count only
    // in use before loading (CUDA context, desktop, other processes): read while far from the budget, so reliable on WSL too
    size_t free0 = 0, total0 = 0;
    if (cudaMemGetInfo(&free0, &total0) != cudaSuccess) free0 = total0 = 0;
    const size_t used_before = total0 - free0;
    if (wsl && opt.fit)
        std::fprintf(stderr, "tqllm: fit: WSL - judging free memory by the engine's own allocations (%.2f GB in use before load)\n",
                     double(used_before) / 1e9);
    auto measure = [&](size_t & fr, size_t & tot) {
        size_t df = 0;
        if (cudaMemGetInfo(&df, &tot) != cudaSuccess) return false;
        fr = fit_free(wsl, df, tot, used_before, tqllm_cuda::dev_bytes());
        if (tqllm::debug_on("memlog"))
            std::fprintf(stderr, "tqllm: fit measure: engine %.2f GB, driver free %.2f GB, total %.2f GB, before %.2f GB -> free %.2f GB\n",
                         double(tqllm_cuda::dev_bytes()) / 1e9, double(df) / 1e9, double(tot) / 1e9, double(used_before) / 1e9,
                         double(fr) / 1e9);
        return true;
    };
    const FitSetting asked{opt.kv_format(), ctx, long(opt.ubatch), long(opt.gemm_rows)};
    auto finish = [&](std::unique_ptr<Engine> en) {
        if (!en) return en;
        // the fit may move weights and KV between device and host (same results); a lossier KV codec or a shorter context
        // changes the answers, so it is refused unless asked for (-fitl on)
        if (const std::string lossy = fit_result_changes(asked, {opt.kv_format(), en->context(), long(opt.ubatch), long(opt.gemm_rows)});
            !lossy.empty() && !opt.fit_lossy)
            throw std::runtime_error("does not fit as asked (" + std::to_string(slots) + " slot(s), context " + std::to_string(asked.ctx) +
                                     " per slot, KV " + asked.kv + "): it would need " + lossy +
                                     ". Ask for that explicitly (--kv, -ub, --gemm-rows; e.g. --kv q4/tcq2 -ub 512), use fewer slots (--np) or a smaller --context, "
                                     "or allow it with -fitl on");
        if (slots <= 1) return en;
        size_t fr = 0, tot = 0;
        if (!measure(fr, tot)) return en;
        const uint32_t ok = fit_slots(uint32_t(slots), en->context(), opt.fit_min_ctx, fr, base_margin, slot_b);
        if (ok >= uint32_t(slots)) return en;
        char msg[512];
        std::snprintf(msg, sizeof msg,
                      "--np %d does not fit: %.2f GB free after loading, context %u per slot; each slot needs %u positions "
                      "(-fitc) and %ld MiB (--slot-reserve) above the %.2f GB margin. Largest that fits: --np %u "
                      "(or a smaller --context)", slots, double(fr) / 1e9, en->context(), opt.fit_min_ctx, opt.slot_reserve_mb,
                      double(base_margin) / 1e9, ok);
        throw std::runtime_error(msg);
    };
    const bool user_pf = opt.ubatch != 0, user_kv = opt.kv_format() != "bf16";
    const bool user_head = opt.head_host, user_embed = opt.embed_host;  // asked for: never given back
    const size_t floor_b = size_t(256) << 20;
    const bool host_auto = opt.kv_host < 0;
    int host_tries = 0;         // KV-to-host steps applied so far
    size_t host_prev_free = 0;  // free memory measured before the last of them
    bool measuring = false;  // the last load put all KV in host memory to measure what the device can hold
    bool measured = false;   // that measuring load was made (once per fit)
    bool returned = false;   // spare memory after a host-tail fit was tried once
    uint32_t ctx_short = 0;  // the last context that loaded but left less than the margin free
    size_t free_short = 0;
    for (int step = 0;;) {
        std::unique_ptr<Engine> e;
        try {
            e = make_qwen35_engine(path, tok, ctx, slots, opt);
        } catch (const std::exception & ex) {
            if (!margin || !std::strstr(ex.what(), "out of memory")) throw;
        }
        size_t fr = 0, tot = 0;
        if (e && (!margin || (measure(fr, tot) && fr >= margin))) {
            if (measuring) {  // bring back to the device as many positions as the free memory holds
                measuring = false;
                const size_t tb = e->kv_token_bytes(), round_b = e->kv_round_bytes();
                const uint64_t dev_tok = tb && fr > margin + round_b ? (fr - margin - round_b) / tb : 0;
                const uint32_t h = dev_tok >= ctx ? 0 : uint32_t(std::min<uint64_t>(ctx, (ctx - dev_tok + 1023) / 1024 * 1024));
                if (h < uint32_t(opt.kv_host)) {
                    e.reset();
                    opt.kv_host = h;
                    std::fprintf(stderr, "tqllm: fit: %.2f GB free with all KV in host memory: last %u positions stay there\n",
                                 double(fr) / 1e9, h);
                    continue;
                }
            }
            // KV tail in host memory with room left on the GPU: bring the spare back once, keep it if the margin holds
            if (host_auto && opt.kv_host > 0 && !returned) {
                returned = true;
                // measured device cost per position (pages round to granules), not the nominal row size
                const uint32_t dev_pos = ctx > uint32_t(opt.kv_host) ? ctx - uint32_t(opt.kv_host) : 0;
                const size_t per = std::max(e->kv_token_bytes(), dev_pos ? e->kv_device_bytes() / dev_pos : size_t(0));
                const long prev = opt.kv_host;
                uint32_t h = fit_host_return(uint32_t(prev), fr, margin, per);
                for (int k = 0; k < 2 && h < uint32_t(prev); ++k) {  // a short return, then half of it
                    e.reset();
                    opt.kv_host = h;
                    std::unique_ptr<Engine> e2;
                    try {
                        e2 = make_qwen35_engine(path, tok, ctx, slots, opt);
                    } catch (const std::exception & ex) {
                        if (!std::strstr(ex.what(), "out of memory")) throw;
                    }
                    size_t fr2 = 0;
                    if (e2 && measure(fr2, tot) && fr2 >= margin) {
                        std::fprintf(stderr, "tqllm: fit: %ld positions back on the GPU (%u in host memory), %.2f GB free\n",
                                     prev - long(h), h, double(fr2) / 1e9);
                        return finish(std::move(e2));
                    }
                    h = uint32_t(prev) - (uint32_t(prev) - h) / 2 / 1024 * 1024;
                }
                if (opt.kv_host != prev) {
                    opt.kv_host = prev;
                    e = make_qwen35_engine(path, tok, ctx, slots, opt);
                    if (!e || !measure(fr, tot)) throw std::runtime_error("fit: reload after a failed KV return");
                }
            }
            // a halved context that fits with room to spare: take the largest context the two loads predict
            // a halved context that fits with room to spare: take the largest context the loads predict; an attempt that
            // falls short becomes the new upper point and the prediction is refined (up to 3 loads)
            uint32_t ok_ctx = ctx;
            size_t ok_free = fr;
            bool reloaded = false;
            for (int k = 0; ctx_short && k < 3; ++k) {
                const uint32_t best = fit_context(ctx_short, free_short, ok_ctx, ok_free, margin);
                if (best <= ok_ctx) break;
                e.reset();
                reloaded = true;
                std::unique_ptr<Engine> e2;
                try {
                    e2 = make_qwen35_engine(path, tok, best, slots, opt);
                } catch (const std::exception & ex) {
                    if (!std::strstr(ex.what(), "out of memory")) throw;
                }
                size_t fr2 = 0;
                if (e2) measure(fr2, tot);
                if (e2 && fr2 >= margin) {
                    std::fprintf(stderr, "tqllm: fit: context %u (interpolated), %.2f GB free after %d step(s)\n", best,
                                 double(fr2) / 1e9, step + 1 + k);
                    return finish(std::move(e2));
                }
                std::fprintf(stderr, "tqllm: fit: interpolated context %u left %.2f GB; refining\n", best, double(fr2) / 1e9);
                if (!e2) break;  // out of memory: no measurement to refine with
                ctx_short = best, free_short = fr2;
            }
            if (reloaded) {
                e = make_qwen35_engine(path, tok, ok_ctx, slots, opt);
                if (!e || !measure(fr, tot)) throw std::runtime_error("fit: reload at the interpolated context");
            }
            // give back placement steps that cost speed while the load leaves room for them (LM head first: every token)
            struct Back { bool * on; bool user; const char * what; };
            for (const Back & b : {Back{&opt.head_host, user_head, "int8 LM head back on the GPU"},
                                   Back{&opt.embed_host, user_embed, "token embeddings back on the GPU"}}) {
                if (!*b.on || b.user || !fit_try_return(fr, margin)) continue;
                e.reset();
                *b.on = false;
                std::unique_ptr<Engine> e2;
                try {
                    e2 = make_qwen35_engine(path, tok, ok_ctx, slots, opt);
                } catch (const std::exception & ex) {
                    if (!std::strstr(ex.what(), "out of memory")) throw;
                }
                size_t fr2 = 0;
                if (e2 && measure(fr2, tot) && fr2 >= margin) {
                    std::fprintf(stderr, "tqllm: fit: %s, %.2f GB free\n", b.what, double(fr2) / 1e9);
                    e = std::move(e2);
                    fr = fr2;
                    continue;
                }
                e2.reset();
                *b.on = true;
                e = make_qwen35_engine(path, tok, ok_ctx, slots, opt);
                if (!e || !measure(fr, tot)) throw std::runtime_error("fit: reload after a failed give-back");
                break;
            }
            if (step) std::fprintf(stderr, "tqllm: fit: %.2f GB free after %d step(s)\n", double(fr) / 1e9, step);
            return finish(std::move(e));
        }
        // every memory step applied (or all KV already in host memory): keep the context if the floor is free
        if (e && fr >= floor_b && (step >= 8 || (host_auto && opt.kv_host >= long(ctx)))) {
            std::fprintf(stderr, "tqllm: fit: WARNING %.2f GB free (< %.2f GB margin) after %d step(s); keeping context %u\n",
                         double(fr) / 1e9, double(margin) / 1e9, step, ctx);
            return finish(std::move(e));
        }
        const bool loaded = bool(e);
        const size_t token_b = e ? e->kv_token_bytes() : 0;
        e.reset();
        std::string what;
        for (bool applied = false; !applied; ++step) {
            switch (step) {
            case 0:
                // drafter retrieval (default on) costs the drafter's K/V over the whole context; drafts never change the
                // output, so the fit may turn it off before anything else (an explicit --draft-retrieve N stays)
                if (opt.draft_retrieve < 0 && !opt.draft_model.empty()) {
                    opt.draft_retrieve = 0; what = "drafter retrieval off (--draft-retrieve N keeps it)"; applied = true; step = -1;
                } else if (!opt.embed_host) { opt.embed_host = true; what = "token embeddings to host memory"; applied = true; }
                break;
            case 1: if (!opt.head_host) { opt.head_host = true; what = "int8 LM head to host memory"; applied = true; } break;
            case 2: if (!user_pf) { opt.ubatch = 1024; what = "prefill buffers 1024"; applied = true; } break;
            case 3: if (!user_pf) { opt.ubatch = 512; what = "prefill buffers 512"; applied = true; } break;
            // q4 K + trellis 2-bit V: lowest KLD of the compact caches at 64K-200K (results/20261006_kv_longctx_kld)
            case 4: if (!user_kv) { opt.kv = "q4/tcq2"; what = "KV cache q4/tcq2"; applied = true; } break;
            case 5: if (!opt.gemm_rows) { opt.gemm_rows = 2048; what = "prefill FP16 weight blocks of 2048 rows"; applied = true; } break;
            case 6:  // KV tail to host memory (exact; decode reads it over PCIe) before giving up context
            case 7:  // refine while each step frees memory (the per-token estimate undershoots: buffers grow with the tail)
                if (!host_auto) break;
                if (host_tries > 0 && !fit_host_again(host_tries, fr, host_prev_free, uint32_t(std::max(0L, opt.kv_host)), ctx))
                    break;
                if (!loaded || !token_b) {  // out of memory: measure once with all KV on host
                    if (!measured) { opt.kv_host = ctx; measuring = measured = true; what = "all KV to host memory (measuring)"; applied = true; }
                    break;
                }
                if (const uint32_t h = fit_host_tokens(ctx, uint32_t(std::max(0L, opt.kv_host)), fr, margin, token_b, 1024, measured);
                    h > uint32_t(std::max(0L, opt.kv_host))) {
                    opt.kv_host = h;
                    measuring = h == ctx;
                    measured |= measuring;
                    what = measuring ? "all KV to host memory (measuring)" : "KV of the last " + std::to_string(h) + " positions to host memory";
                    applied = true;
                    if (host_tries++ > 0 && step == 7) step = 6;  // stay on the refine step (the loop increments to 7)
                    host_prev_free = fr;
                }
                break;
            default:
                if (ctx / 2 < opt.fit_min_ctx) {  // smallest setting: start with a warning if 256 MB are free, else give up
                    if (fr < (size_t(256) << 20))
                        throw std::runtime_error("model does not fit in device memory (even at context " + std::to_string(ctx) + ")");
                    std::fprintf(stderr, "tqllm: fit: WARNING only %.2f GB free at the smallest setting; long prompts may "
                                         "fail (-fitt lowers the margin)\n", double(fr) / 1e9);
                    return finish(make_qwen35_engine(path, tok, ctx, slots, opt));
                }
                if (loaded) ctx_short = ctx, free_short = fr;
                if (host_auto) opt.kv_host = -1, measuring = false, host_tries = 0;  // host memory did not make room: weights and buffers are too big
                ctx /= 2;
                what = "context " + std::to_string(ctx);
                applied = true;
            }
        }
        std::fprintf(stderr, "tqllm: fit: %.2f GB free (< %.2f GB margin): %s\n", fr ? double(fr) / 1e9 : 0.0,
                     double(margin) / 1e9, what.c_str());
    }
}
}  // namespace

// `weights` is either DIR/weights.tqbin (converted Qwen2) or a checkpoint directory / any file inside it.
Model::Model(const std::string & w, const std::string & t, uint32_t c, int slots, const EngineOptions & o)
    : impl_(std::make_unique<Impl>()) {
    const std::filesystem::path p(w);
    const auto dir = std::filesystem::is_directory(p) ? p : p.parent_path();
    if (slots != 1 && p.extension() != ".gguf") throw std::invalid_argument("multiple slots need a qwen35 GGUF model");
    if (p.extension() == ".gguf") impl_->e = load_fitted(p.string(), t, c, slots, o);  // qwen35 GGUF (llama.cpp-tq3)
    else if (model_type(dir) == "qwen3_5") impl_->e = make_qwen35_engine(dir.string(), t, c, 1, o);
    else impl_->e = std::make_unique<Qwen2Engine>(w, t, c);
}
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

uint32_t Model::forward(uint32_t token, uint32_t pos, bool sample) { return impl_->e->step(token, pos, sample, nullptr); }
void Model::synchronize() { impl_->e->synchronize(); }
bool Model::prefill_state(const std::vector<uint32_t> & tokens, uint32_t pos) { return impl_->e->prefill_state(tokens, pos); }
int Model::slots() const { return impl_->e->slots(); }
bool Model::incremental() const { return impl_->e->incremental(); }
uint32_t Model::context() const { return impl_->e->context(); }
bool Model::kv_idle_pending() const { return impl_->e->kv_idle_pending(); }
bool Model::kv_idle_step() { return impl_->e->kv_idle_step(); }
bool Model::has_drafter() const { return impl_->e->has_drafter(); }
bool Model::native_mtp() const { return impl_->e->native_mtp(); }
void Model::select_slot(int slot) { impl_->e->select_slot(slot); }
void Model::set_prompt_images(const std::vector<PromptImage> & images) { impl_->e->set_prompt_images(images); }
void Model::spec_begin(const std::vector<uint32_t> & prompt, uint32_t count, int draft) {
    impl_->e->spec_begin(prompt, count, draft);
}
bool Model::spec_advance() { return impl_->e->spec_advance(); }
void Model::spec_step(const std::vector<int> & slots, std::vector<char> & done) {
    std::unique_ptr<bool[]> d(new bool[slots.size()]());
    impl_->e->spec_step(slots.data(), int(slots.size()), d.get());
    done.assign(d.get(), d.get() + slots.size());
}
std::vector<uint32_t> Model::spec_finish(uint64_t * stats) { return impl_->e->spec_finish(stats); }
std::vector<uint32_t> Model::spec_tokens(int slot) const { return impl_->e->spec_tokens(slot); }
int Model::drafter_features(const std::vector<uint32_t> & tokens, std::vector<uint16_t> & feats) {
    return impl_->e->drafter_features(tokens, feats);
}
void Model::verify(const std::vector<uint32_t> & tokens, uint32_t pos, std::vector<uint32_t> & next) {
    next.resize(tokens.size());
    impl_->e->verify(tokens.data(), int(tokens.size()), pos, next.data());
}
void Model::commit(int accepted) { impl_->e->commit(accepted); }
const float * Model::verify_logits() const { return impl_->e->verify_logits(); }
void Model::set_full_logits(bool on) { impl_->e->set_full_logits(on); }
void Model::set_sampling(float temperature, float top_p, int top_k, uint64_t seed) {
    impl_->e->set_sampling(temperature, top_p, top_k, seed);
}

std::vector<uint32_t> Model::generate_spec(const std::vector<uint32_t> & prompt, uint32_t count, int draft,
                                           uint64_t * stats) {
    return impl_->e->generate_spec(prompt, count, draft, stats);
}
uint32_t Model::layers() const { return impl_->e->layers(); }
uint32_t Model::hidden_size() const { return impl_->e->hidden(); }
uint32_t Model::vocab_size() const { return impl_->e->vocab(); }

void Model::capture(const std::vector<uint32_t> & prompt, std::vector<float> & hidden, std::vector<float> & logits) {
    auto & e = *impl_->e;
    if (prompt.empty() || prompt.size() > e.context()) throw std::invalid_argument("capture: bad prompt length");
    const auto n = uint32_t(prompt.size());
    for (uint32_t i = 0; i + 1 < n; ++i) e.step(prompt[i], i, false, nullptr);
    hidden.assign(size_t(e.layers()) * e.hidden(), 0.f);
    e.step(prompt[n - 1], n - 1, true, hidden.data());
    logits.resize(e.vocab());
    ck(cudaMemcpy(logits.data(), e.logits(), logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

std::vector<uint32_t> Model::generate(const std::vector<uint32_t> & prompt, uint32_t count) {
    auto & e = *impl_->e;
    if (prompt.empty()) throw std::invalid_argument("empty prompt");
    if (prompt.size() > e.context() || uint64_t(prompt.size()) + count > e.context())
        throw std::runtime_error("request exceeds context capacity");
    const auto n = uint32_t(prompt.size());
    for (uint32_t i = 0; i + 1 < n; ++i) e.step(prompt[i], i, false, nullptr);
    uint32_t next = e.step(prompt[n - 1], n - 1, true, nullptr);
    std::vector<uint32_t> out;
    for (uint32_t i = 0; i < count; ++i) {
        out.push_back(next);
        if (next == e.eos()) break;
        if (i + 1 < count) next = e.step(next, n + i, true, nullptr);
    }
    return out;
}

} // namespace tqllm

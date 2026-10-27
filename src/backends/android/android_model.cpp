#include "android_model.hpp"
#include "gguf.hpp"
#include "host_matrix.hpp"
#include "host_attention_cache.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace tqllm::android_backend {
namespace {
using Vec = std::vector<float>;
float sigmoid(float x) { return x >= 0.f ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x)); }
void rms(float * x, const Vec & weight, float eps) {
    const int n = int(weight.size());
    float sum = host_dot(x, x, n), scale = 1.f / std::sqrt(sum / float(n) + eps);
    for (int i = 0; i < n; ++i) x[i] *= scale * weight[i];
}
void l2(float * x, int n, float scale) {
    scale /= std::sqrt(host_dot(x, x, n) + 1e-6f);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}
void rope(float * x, int rot, int pos, float theta) {
    for (int i = 0; i < rot / 2; ++i) {
        const float angle = float(pos) * std::pow(theta, -2.f * float(i) / float(rot));
        const float c = std::cos(angle), s = std::sin(angle), a = x[i], b = x[i + rot / 2];
        x[i] = a * c - b * s; x[i + rot / 2] = a * s + b * c;
    }
}
Options validate(Options o) {
    if (o.context <= 0 || o.context > 32768 || o.threads < 1 || o.threads > 8)
        throw std::invalid_argument("native Android requires context 1..32768 and threads 1..8");
    return o;
}
}
struct Model::Impl {
    Options options;
    GgufFile file;
    MappedFile mapping;
    HostExecutor executor;
    std::map<std::string, HostMatrix> matrices;
    std::map<std::string, Vec> scalars;
    struct Layer { Vec history, state; std::unique_ptr<HostAttentionCache> kv; };
    std::vector<Layer> cache;
    Vec logits;
    int layers, hidden, heads, kvheads, hd, rot, kh, vh, dk, dv, conv, interval, ffn, vocab, pos = 0;
    uint32_t eos;
    float eps, theta;
    explicit Impl(const std::string & path, Options opts)
        : options(validate(opts)), file(gguf_open(path)), mapping(path), executor(options.threads) {
        if (file.str("general.architecture") != "qwen35")
            throw std::invalid_argument("native Android currently supports dense qwen35 GGUF only");
        auto integer = [&](const std::string & key) {
            double v = file.num("qwen35." + key);
            if (!std::isfinite(v) || v <= 0 || v > 1048576 || v != std::floor(v))
                throw std::invalid_argument("invalid qwen35 metadata: " + key);
            return int(v);
        };
        layers = integer("block_count"); hidden = integer("embedding_length"); ffn = integer("feed_forward_length");
        heads = integer("attention.head_count"); kvheads = integer("attention.head_count_kv");
        hd = integer("attention.key_length"); rot = integer("rope.dimension_count");
        kh = integer("ssm.group_count"); vh = integer("ssm.time_step_rank"); dk = integer("ssm.state_size");
        const int inner = integer("ssm.inner_size"); dv = inner / vh;
        conv = integer("ssm.conv_kernel"); interval = integer("full_attention_interval");
        eps = float(file.num("qwen35.attention.layer_norm_rms_epsilon"));
        theta = float(file.num("qwen35.rope.freq_base"));
        if (layers > 128 || hidden > 8192 || ffn > 65536 || heads > 128 || kvheads > heads || heads % kvheads ||
            hd > 256 || rot > hd || rot % 2 || kh > vh || vh > 128 || vh % kh || dk > 256 || dv != dk ||
            inner % vh || conv < 2 || conv > 8 || !std::isfinite(eps) || eps <= 0.f ||
            !std::isfinite(theta) || theta <= 1.f || file.num("qwen35.rope.scaling.factor", 1) != 1 ||
            file.num("qwen35.nextn_predict_layers", 0) != 0)
            throw std::invalid_argument("unsupported native qwen35 dimensions, RoPE scaling or MTP layers");
        const auto & embedding = file.tensors.at("token_embd.weight");
        if (embedding.ne.size() != 2 || embedding.ne[1] == 0 || embedding.ne[1] > 300000)
            throw std::invalid_argument("invalid native embedding shape");
        vocab = int(embedding.ne[1]);
        double eos_value = file.num("tokenizer.ggml.eos_token_id");
        if (!std::isfinite(eos_value) || eos_value < 0 || eos_value >= vocab || std::floor(eos_value) != eos_value)
            throw std::invalid_argument("invalid native EOS token");
        eos = uint32_t(eos_value);
        const double train_context = file.num("qwen35.context_length");
        if (!std::isfinite(train_context) || train_context < 1 || options.context > train_context) throw std::invalid_argument("context exceeds model training context");
        auto matrix = [&](const std::string & name, int cols, int rows) {
            matrices.emplace(name, HostMatrix(file.tensors.at(name), mapping, cols, rows));
        };
        auto scalar = [&](const std::string & name, std::vector<uint64_t> shape) {
            const auto & t = file.tensors.at(name);
            if (t.ne != shape || (t.type != GGML_F32 && t.type != GGML_F16 && t.type != GGML_BF16))
                throw std::invalid_argument("native scalar shape/type: " + name);
            size_t n = 1;
            for (auto d : shape) n *= size_t(d);
            Vec v(n); dequantize_host(t.type, mapping.at(t.offset, t.bytes), v.data(), int64_t(n));
            for (float x : v) if (!std::isfinite(x)) throw std::invalid_argument("nonfinite native weight: " + name);
            scalars.emplace(name, std::move(v));
        };
        matrix("token_embd.weight", hidden, vocab);
        if (file.tensors.count("output.weight")) matrix("output.weight", hidden, vocab);
        scalar("output_norm.weight", {uint64_t(hidden)});
        const uint64_t attention_layers = uint64_t(layers / interval);
        const uint64_t recurrent_layers = uint64_t(layers) - attention_layers;
        const uint64_t kv_bytes = attention_layers * uint64_t(options.context) * uint64_t(kvheads) * hd * 8;
        const uint64_t state_bytes = recurrent_layers * (uint64_t(vh) * dv * dk +
            uint64_t(conv - 1) * (2 * kh * dk + vh * dv)) * 4;
        if (kv_bytes + state_bytes > (uint64_t(2) << 30))
            throw std::invalid_argument("native Android state exceeds 2 GiB; reduce context");
        cache.resize(layers);
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            scalar(p + "attn_norm.weight", {uint64_t(hidden)});
            scalar(p + "post_attention_norm.weight", {uint64_t(hidden)});
            matrix(p + "ffn_gate.weight", hidden, ffn); matrix(p + "ffn_up.weight", hidden, ffn);
            matrix(p + "ffn_down.weight", ffn, hidden);
            if ((l + 1) % interval == 0) {
                matrix(p + "attn_q.weight", hidden, 2 * heads * hd);
                matrix(p + "attn_k.weight", hidden, kvheads * hd); matrix(p + "attn_v.weight", hidden, kvheads * hd);
                matrix(p + "attn_output.weight", heads * hd, hidden);
                scalar(p + "attn_q_norm.weight", {uint64_t(hd)}); scalar(p + "attn_k_norm.weight", {uint64_t(hd)});
                cache[l].kv = std::make_unique<HostAttentionCache>(options.context, kvheads * hd);
            } else {
                const int width = 2 * kh * dk + vh * dv;
                matrix(p + "attn_qkv.weight", hidden, width); matrix(p + "attn_gate.weight", hidden, vh * dv);
                matrix(p + "ssm_alpha.weight", hidden, vh); matrix(p + "ssm_beta.weight", hidden, vh);
                matrix(p + "ssm_out.weight", vh * dv, hidden);
                scalar(p + "ssm_conv1d.weight", {uint64_t(conv), uint64_t(width)});
                scalar(p + "ssm_a", {uint64_t(vh)}); scalar(p + "ssm_dt.bias", {uint64_t(vh)});
                scalar(p + "ssm_norm.weight", {uint64_t(dv)});
                cache[l].history.resize(size_t(conv - 1) * width);
                cache[l].state.resize(size_t(vh) * dv * dk);
            }
        }
    }
    // One int8 copy of a projection input, shared by every matrix that reads it.
    struct Input {
        const Vec & x;
        HostQ8 q;
        bool quantized = false;
    };
    Vec mat(const std::string & name, Input & in) {
        const auto & m = matrices.at(name);
        if (options.exact || !m.int8_ready()) return m.multiply(in.x, executor);
        if (!in.quantized) { host_quantize_q8(in.x.data(), int(in.x.size()), in.q); in.quantized = true; }
        return m.multiply(in.q, executor);
    }
    Vec mat(const std::string & name, const Vec & x) { Input in{x}; return mat(name, in); }
    Vec attention(const std::string & p, const Vec & x, Layer & layer) {
        Input in{x};
        auto qg = mat(p + "attn_q.weight", in), k = mat(p + "attn_k.weight", in), v = mat(p + "attn_v.weight", in);
        for (int h = 0; h < heads; ++h) {
            rms(qg.data() + size_t(h) * 2 * hd, scalars.at(p + "attn_q_norm.weight"), eps);
            rope(qg.data() + size_t(h) * 2 * hd, rot, pos, theta);
        }
        for (int h = 0; h < kvheads; ++h) {
            rms(k.data() + h * hd, scalars.at(p + "attn_k_norm.weight"), eps);
            rope(k.data() + h * hd, rot, pos, theta);
        }
        layer.kv->write(pos, k, v);
        Vec out(heads * hd);
        executor.rows(heads, [&](int begin, int end) {
            Vec score(pos + 1);
            for (int h = begin; h < end; ++h) {
                const int offset = (h / (heads / kvheads)) * hd;
                const float * q = qg.data() + size_t(h) * 2 * hd;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int t = 0; t <= pos; ++t) {
                    score[t] = host_dot(q, layer.kv->key(t, offset), hd) / std::sqrt(float(hd));
                    maximum = std::max(maximum, score[t]);
                }
                float sum = 0.f;
                for (float & s : score) { s = std::exp(s - maximum); sum += s; }
                float * y = out.data() + h * hd;
                for (int t = 0; t <= pos; ++t) {
                    const float * value = layer.kv->value(t, offset);
                    const float a = score[t] / sum;
                    for (int j = 0; j < hd; ++j) y[j] += a * value[j];
                }
                for (int j = 0; j < hd; ++j) y[j] *= sigmoid(q[hd + j]);
            }
        });
        return mat(p + "attn_output.weight", out);
    }
    Vec gdn(const std::string & p, const Vec & x, Layer & layer) {
        Input in{x};
        auto qkv = mat(p + "attn_qkv.weight", in), z = mat(p + "attn_gate.weight", in);
        auto alpha = mat(p + "ssm_alpha.weight", in), beta = mat(p + "ssm_beta.weight", in);
        const int width = int(qkv.size());
        Vec filtered(width), out(vh * dv);
        const auto & cw = scalars.at(p + "ssm_conv1d.weight");
        for (int j = 0; j < width; ++j) {
            float s = qkv[j] * cw[size_t(j) * conv + conv - 1];
            for (int t = 0; t < conv - 1; ++t) s += layer.history[size_t(t) * width + j] * cw[size_t(j) * conv + t];
            filtered[j] = s * sigmoid(s);
        }
        std::move(layer.history.begin() + width, layer.history.end(), layer.history.begin());
        std::copy(qkv.begin(), qkv.end(), layer.history.end() - width);
        for (int h = 0; h < kh; ++h) {
            l2(filtered.data() + h * dk, dk, 1.f / std::sqrt(float(dk)));
            l2(filtered.data() + (kh + h) * dk, dk, 1.f);
        }
        executor.rows(vh, [&](int begin, int end) {
            for (int h = begin; h < end; ++h) {
                const float dt = alpha[h] + scalars.at(p + "ssm_dt.bias")[h];
                const float softplus = std::max(dt, 0.f) + std::log1p(std::exp(-std::abs(dt)));
                const float decay = std::exp(softplus * scalars.at(p + "ssm_a")[h]), b = sigmoid(beta[h]);
                const float * q = filtered.data() + (h % kh) * dk;
                const float * k = filtered.data() + (kh + h % kh) * dk;
                for (int r = 0; r < dv; ++r) {
                    float * state = layer.state.data() + (size_t(h) * dv + r) * dk;
                    for (int j = 0; j < dk; ++j) state[j] *= decay;
                    const float delta = (filtered[2 * kh * dk + h * dv + r] - host_dot(state, k, dk)) * b;
                    for (int j = 0; j < dk; ++j) state[j] += delta * k[j];
                    out[h * dv + r] = host_dot(state, q, dk);
                }
                rms(out.data() + h * dv, scalars.at(p + "ssm_norm.weight"), eps);
                for (int j = 0; j < dv; ++j) out[h * dv + j] *= z[h * dv + j] * sigmoid(z[h * dv + j]);
            }
        });
        return mat(p + "ssm_out.weight", out);
    }
    const Vec & forward(uint32_t token, bool head) {
        if (token >= uint32_t(vocab) || pos >= options.context) throw std::out_of_range("native token or context limit");
        auto x = matrices.at("token_embd.weight").row(int(token));
        for (int l = 0; l < layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            auto n = x; rms(n.data(), scalars.at(p + "attn_norm.weight"), eps);
            auto y = cache[l].kv ? attention(p, n, cache[l]) : gdn(p, n, cache[l]);
            for (int j = 0; j < hidden; ++j) x[j] += y[j];
            n = x; rms(n.data(), scalars.at(p + "post_attention_norm.weight"), eps);
            Input in{n};
            auto gate = mat(p + "ffn_gate.weight", in), up = mat(p + "ffn_up.weight", in);
            for (int j = 0; j < ffn; ++j) gate[j] *= sigmoid(gate[j]) * up[j];
            y = mat(p + "ffn_down.weight", gate);
            for (int j = 0; j < hidden; ++j) x[j] += y[j];
        }
        ++pos;
        logits.clear();
        if (head) {
            rms(x.data(), scalars.at("output_norm.weight"), eps);
            logits = mat(matrices.count("output.weight") ? "output.weight" : "token_embd.weight", x);
            for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("nonfinite native logits");
        }
        return logits;
    }
};
Model::Model(const std::string & path, Options options) : impl_(std::make_unique<Impl>(path, options)) {}
Model::~Model() = default;
void Model::reset() {
    impl_->pos = 0; impl_->logits.clear();
    for (auto & layer : impl_->cache) {
        std::fill(layer.history.begin(), layer.history.end(), 0.f);
        std::fill(layer.state.begin(), layer.state.end(), 0.f);
    }
}
const std::vector<float> & Model::forward(uint32_t token, bool head) { return impl_->forward(token, head); }
int Model::position() const { return impl_->pos; }
int Model::context() const { return impl_->options.context; }
int Model::vocabulary() const { return impl_->vocab; }
uint32_t Model::eos() const { return impl_->eos; }
}

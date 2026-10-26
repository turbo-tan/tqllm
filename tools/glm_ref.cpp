// glm_ref: FP32 CPU reference forward of GLM-5.3-Flash (GGUF arch `glm5-next` / `glm5next`), one token at a time.
// Exists to check the architecture against llama.cpp before any GPU kernel is written: the GPU engine is then tested
// against this, layer by layer. Weights stay quantised in the mmapped file and are dequantised row by row.
//
//   glm-ref MODEL.gguf --tokens 1,2,3 [--gen N] [--dump]
//
// Prints the greedy continuation (token ids) after teacher-forcing the prompt, and per step the top-1 id / logit.
// Context must stay <= the indexer's top_k (2048): there every cached position is selected, so dense MLA is exact.
#include "dequant.hpp"
#include "gguf.hpp"
#include "gguf_mapped.hpp"
#include "moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;

namespace {

struct Tensor {
    const char * p = nullptr;
    uint32_t type = 0;
    std::vector<uint64_t> ne;
    size_t row_bytes = 0;  // bytes of one ne0-row
};

class Weights {
    std::vector<std::unique_ptr<MappedGguf>> maps_;
    std::map<std::string, Tensor> t_;
public:
    GgufFile meta;
    explicit Weights(const std::string & path) : meta(gguf_open(path)) {
        std::map<std::string, const MappedGguf *> by_file;
        for (const auto & [name, t] : meta.tensors) {
            const std::string f = t.file.empty() ? meta.path : t.file;
            if (!by_file.count(f)) { maps_.push_back(std::make_unique<MappedGguf>(f)); by_file[f] = maps_.back().get(); }
            Tensor x;
            x.p = by_file[f]->tensor(name).data;
            x.type = t.type; x.ne = t.ne;
            uint64_t el = 0, by = 0;
            if (!gguf_block_layout(t.type, el, by)) throw std::runtime_error("unsized type for " + name);
            x.row_bytes = size_t(t.ne[0] / el * by);
            t_[name] = x;
        }
    }
    bool has(const std::string & n) const { return t_.count(n) != 0; }
    const Tensor & at(const std::string & n) const {
        const auto it = t_.find(n);
        if (it == t_.end()) throw std::runtime_error("missing tensor " + n);
        return it->second;
    }
};

// y[r] = sum_c W[r][c] x[c] over rows [r0, r0 + rows) of the 2-D slice starting at `base`
void matvec(const Tensor & w, const char * base, int64_t rows, const float * x, float * y) {
    const int64_t cols = int64_t(w.ne[0]);
#pragma omp parallel
    {
        std::vector<float> row(static_cast<size_t>(cols));
#pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const char * src = base + size_t(r) * w.row_bytes;
            const float * v = row.data();
            if (w.type == 0) v = reinterpret_cast<const float *>(src);
            else dequantize_host(w.type, src, row.data(), cols);
            double acc = 0;
            for (int64_t c = 0; c < cols; ++c) acc += double(v[c]) * x[c];
            y[r] = float(acc);
        }
    }
}
std::vector<float> mv(const Tensor & w, const float * x) {
    std::vector<float> y(static_cast<size_t>(w.ne[1]));
    matvec(w, w.p, int64_t(w.ne[1]), x, y.data());
    return y;
}
// slice k of a 3-D tensor [ne0, ne1, ne2]
std::vector<float> mv3(const Tensor & w, uint64_t k, const float * x) {
    std::vector<float> y(static_cast<size_t>(w.ne[1]));
    matvec(w, w.p + size_t(k) * w.ne[1] * w.row_bytes, int64_t(w.ne[1]), x, y.data());
    return y;
}
std::vector<float> vec(const Tensor & w) {
    uint64_t n = 1;
    for (auto d : w.ne) n *= d;
    std::vector<float> y(static_cast<size_t>(n));
    if (w.type == 0) std::memcpy(y.data(), w.p, size_t(n) * 4);
    else dequantize_host(w.type, w.p, y.data(), int64_t(n));
    return y;
}

void rmsnorm(float * x, int n, const float * w, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float s = float(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) x[i] = x[i] * s * (w ? w[i] : 1.f);
}
bool g_dump = false;
void dump(const char * name, int l, const float * v, size_t n) {
    if (!g_dump) return;
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += v[i];
    std::printf("dump %s-%d %.6f\n", name, l, sum);
}
void dump(const char * name, int l, const std::vector<float> & v) { dump(name, l, v.data(), v.size()); }
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
float silu(float x) { return x / (1.f + std::exp(-x)); }

struct Config {
    std::string a;  // key prefix
    int layers = 0, hidden = 0, heads = 0, dense_lead = 0, experts = 0, topk = 0, q_lora = 0, kv_lora = 0;
    int head_k_mla = 0, head_v_mla = 0, kda_dim = 0, conv = 0, hc = 0, sinkhorn = 0, vocab = 0, index_topk = 0;
    float rms_eps = 0, hc_eps = 0, gate_lb = 0, weight_scale = 0;
    bool weight_norm = false;
    std::vector<int> recurrent;
    std::vector<float> clamp_exp, clamp_shexp;
};

Config config(const GgufFile & f) {
    Config c;
    const std::string arch = f.str("general.architecture");
    if (arch != "glm5-next" && arch != "glm5next") throw std::runtime_error("not a GLM-5.3-Flash GGUF: " + arch);
    c.a = arch + ".";
    auto n = [&](const char * k) { return f.num(c.a + k); };
    c.layers = int(n("block_count")) - int(f.num(c.a + "nextn_predict_layers", 0));
    c.hidden = int(n("embedding_length")); c.heads = int(n("attention.head_count"));
    c.dense_lead = int(n("leading_dense_block_count")); c.experts = int(n("expert_count"));
    c.topk = int(n("expert_used_count")); c.q_lora = int(n("attention.q_lora_rank"));
    c.kv_lora = int(n("attention.kv_lora_rank")); c.head_k_mla = int(n("attention.key_length_mla"));
    c.head_v_mla = int(n("attention.value_length_mla")); c.kda_dim = int(n("kda.head_dim"));
    c.conv = int(n("ssm.conv_kernel")); c.hc = int(n("hyper_connection.count"));
    c.sinkhorn = int(n("hyper_connection.sinkhorn_iterations")); c.vocab = int(n("vocab_size"));
    c.index_topk = int(n("attention.indexer.top_k"));
    c.rms_eps = float(n("attention.layer_norm_rms_epsilon")); c.hc_eps = float(n("hyper_connection.epsilon"));
    c.gate_lb = float(n("kda.gate_lower_bound")); c.weight_scale = float(f.num(c.a + "expert_weights_scale", 0));
    c.weight_norm = f.num(c.a + "expert_weights_norm", 0) != 0;
    if (f.num(c.a + "expert_gating_func", 2) != 2) throw std::runtime_error("only sigmoid gating is implemented");
    if (f.num(c.a + "rope.dimension_count", 0) != 0) throw std::runtime_error("only the nope-only MLA is implemented");
    const auto & kv = f.kv.at(c.a + "attention.head_count_kv").arr;
    const auto & ce = f.kv.at(c.a + "swiglu_clamp_exp").arr;
    const auto & cs = f.kv.count(c.a + "swiglu_clamp_shexp") ? f.kv.at(c.a + "swiglu_clamp_shexp").arr : ce;
    if (int(kv.size()) < c.layers || int(ce.size()) < c.layers || int(cs.size()) < c.layers)
        throw std::runtime_error("per-layer arrays shorter than the block count");
    for (int l = 0; l < c.layers; ++l) {
        c.recurrent.push_back(kv[size_t(l)] == 0);
        c.clamp_exp.push_back(float(ce[size_t(l)])); c.clamp_shexp.push_back(float(cs[size_t(l)]));
    }
    if (c.hc != 4) throw std::runtime_error("mHC with hc != 4");
    return c;
}

struct Layer {
    // KDA state: conv history of the q / k / v projections (oldest first) and [heads][dim][dim] recurrent state
    std::vector<float> conv;   // [3][conv - 1][d_inner]
    std::vector<float> state;  // [heads][a: key][b: value]
    std::vector<std::vector<float>> latent;  // MLA: per position [kv_lora]
};

class Model {
    const Weights & w_;
public:
    Config c;
    std::vector<Layer> L;
    std::vector<float> logits;
    explicit Model(const Weights & w) : w_(w), c(config(w.meta)), L(size_t(c.layers)) {
        const int d = c.heads * c.kda_dim;
        for (int l = 0; l < c.layers; ++l) if (c.recurrent[size_t(l)]) {
            L[size_t(l)].conv.assign(size_t(3) * (c.conv - 1) * d, 0.f);
            L[size_t(l)].state.assign(size_t(c.heads) * c.kda_dim * c.kda_dim, 0.f);
        }
    }
    const Tensor & T(int l, const char * n) const { return w_.at("blk." + std::to_string(l) + "." + n); }

    // mHC: the 4 streams S [hc][H] -> sublayer input; post [hc], comb [dst][src]
    std::vector<float> hc_pre(const std::vector<float> & S, int l, const char * kind, std::vector<float> & post,
                              std::vector<float> & comb) const {
        const int H = c.hidden, hc = c.hc;
        std::vector<float> flat = S;
        rmsnorm(flat.data(), hc * H, nullptr, c.rms_eps);
        const std::string k = std::string("hc_") + kind;
        const auto mix = mv(T(l, (k + "_fn.weight").c_str()), flat.data());
        dump("hc_mixes", l, mix);
        const auto base = vec(T(l, (k + "_base.weight").c_str())), scale = vec(T(l, (k + "_scale.weight").c_str()));
        std::vector<float> pre(static_cast<size_t>(hc));
        post.assign(size_t(hc), 0.f);
        for (int i = 0; i < hc; ++i) {
            pre[size_t(i)] = sigmoid(mix[size_t(i)] * scale[0] + base[size_t(i)]) + c.hc_eps;
            post[size_t(i)] = 2.f * sigmoid(mix[size_t(hc + i)] * scale[1] + base[size_t(hc + i)]);
        }
        // comb(dst, src) = mix[2 hc + dst + hc src]: softmax over dst for each src, + eps, then Sinkhorn
        comb.assign(size_t(hc) * hc, 0.f);
        auto C = [&](int d, int s) -> float & { return comb[size_t(d) * hc + s]; };
        for (int s = 0; s < hc; ++s) {
            float m = -INFINITY;
            for (int d = 0; d < hc; ++d) {
                C(d, s) = mix[size_t(2 * hc + d + hc * s)] * scale[2] + base[size_t(2 * hc + d + hc * s)];
                m = std::max(m, C(d, s));
            }
            float z = 0;
            for (int d = 0; d < hc; ++d) z += (C(d, s) = std::exp(C(d, s) - m));
            for (int d = 0; d < hc; ++d) C(d, s) = C(d, s) / z + c.hc_eps;
        }
        auto norm_cols = [&] {  // for each dst: divide by sum over src (+ eps)
            for (int d = 0; d < hc; ++d) {
                float z = 0;
                for (int s = 0; s < hc; ++s) z += C(d, s);
                for (int s = 0; s < hc; ++s) C(d, s) /= z + c.hc_eps;
            }
        };
        auto norm_rows = [&] {  // for each src: divide by sum over dst (+ eps)
            for (int s = 0; s < hc; ++s) {
                float z = 0;
                for (int d = 0; d < hc; ++d) z += C(d, s);
                for (int d = 0; d < hc; ++d) C(d, s) /= z + c.hc_eps;
            }
        };
        norm_cols();
        for (int i = 1; i < c.sinkhorn; ++i) { norm_rows(); norm_cols(); }
        dump("hc_pre", l, pre); dump("hc_post", l, post); dump("hc_comb", l, comb);
        std::vector<float> x(static_cast<size_t>(H), 0.f);
        for (int i = 0; i < hc; ++i) for (int j = 0; j < H; ++j) x[size_t(j)] += pre[size_t(i)] * S[size_t(i) * H + j];
        return x;
    }
    void hc_post(const std::vector<float> & f, std::vector<float> & S, const std::vector<float> & post,
                 const std::vector<float> & comb) const {
        const int H = c.hidden, hc = c.hc;
        std::vector<float> out(S.size());
        for (int d = 0; d < hc; ++d) for (int j = 0; j < H; ++j) {
            float v = f[size_t(j)] * post[size_t(d)];
            for (int s = 0; s < hc; ++s) v += S[size_t(s) * H + j] * comb[size_t(d) * hc + s];
            out[size_t(d) * H + j] = v;
        }
        S = std::move(out);
    }

    std::vector<float> kda(int l, const std::vector<float> & x) {
        auto & st = L[size_t(l)];
        const int D = c.kda_dim, NH = c.heads, d = NH * D, K = c.conv;
        std::vector<float> qkv[3];
        const char * proj[3] = {"attn_q.weight", "attn_k.weight", "attn_v.weight"};
        const char * cw[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"};
        for (int i = 0; i < 3; ++i) {
            const auto p = mv(T(l, proj[i]), x.data());
            const auto w = vec(T(l, cw[i]));  // [d][K]
            float * hist = st.conv.data() + size_t(i) * (K - 1) * d;  // [K - 1][d], oldest first
            qkv[i].resize(size_t(d));
            for (int ch = 0; ch < d; ++ch) {
                float acc = 0;
                for (int j = 0; j < K - 1; ++j) acc += hist[size_t(j) * d + ch] * w[size_t(ch) * K + j];
                acc += p[size_t(ch)] * w[size_t(ch) * K + K - 1];
                qkv[i][size_t(ch)] = silu(acc);
            }
            dump(i == 0 ? "kda_q_conv" : i == 1 ? "kda_k_conv" : "kda_v_conv", l, qkv[i]);
            for (int j = 0; j + 1 < K - 1; ++j) std::memcpy(hist + size_t(j) * d, hist + size_t(j + 1) * d, size_t(d) * 4);
            std::memcpy(hist + size_t(K - 2) * d, p.data(), size_t(d) * 4);
        }
        auto g1 = mv(T(l, "ssm_f_b.weight"), mv(T(l, "ssm_f_a.weight"), x.data()).data());
        const auto dtb = vec(T(l, "ssm_dt.bias")), A = vec(T(l, "ssm_a"));
        for (int h = 0; h < NH; ++h) for (int i = 0; i < D; ++i) {
            float & g = g1[size_t(h) * D + i];
            g = c.gate_lb * sigmoid(-((g + dtb[size_t(h) * D + i]) * A[size_t(h)]));
        }
        dump("kda_g1", l, g1);
        auto beta = mv(T(l, "ssm_beta.weight"), x.data());
        for (auto & b : beta) b = sigmoid(b);
        dump("kda_beta", l, beta);
        std::vector<float> o(static_cast<size_t>(d));
        const float scale = 1.f / std::sqrt(float(D));
        for (int h = 0; h < NH; ++h) {
            float * q = qkv[0].data() + size_t(h) * D, * k = qkv[1].data() + size_t(h) * D;
            const float * v = qkv[2].data() + size_t(h) * D;
            for (float * z : {q, k}) {  // x / sqrt(sum x^2 + 1e-6)
                double ss = 0;
                for (int i = 0; i < D; ++i) ss += double(z[i]) * z[i];
                const float s = float(1.0 / std::sqrt(ss + 1e-6));
                for (int i = 0; i < D; ++i) z[i] *= s;
            }
            for (int i = 0; i < D; ++i) q[i] *= scale;
            float * S = st.state.data() + size_t(h) * D * D;  // S[a: key][b: value]
            const float * g = g1.data() + size_t(h) * D;
            for (int a = 0; a < D; ++a) {  // per-channel decay on the key index (FLA KDA)
                const float e = std::exp(g[a]);
                for (int b = 0; b < D; ++b) S[size_t(a) * D + b] *= e;
            }
            for (int b = 0; b < D; ++b) {
                float sk = 0;
                for (int a = 0; a < D; ++a) sk += S[size_t(a) * D + b] * k[a];
                const float dl = (v[b] - sk) * beta[size_t(h)];
                for (int a = 0; a < D; ++a) S[size_t(a) * D + b] += k[a] * dl;
            }
            for (int b = 0; b < D; ++b) {
                float acc = 0;
                for (int a = 0; a < D; ++a) acc += S[size_t(a) * D + b] * q[a];
                o[size_t(h) * D + b] = acc;
            }
        }
        dump("kda_scan_out", l, o);
        const auto g2 = mv(T(l, "ssm_g_b.weight"), mv(T(l, "ssm_g_a.weight"), x.data()).data());
        dump("kda_g2", l, g2);
        const auto on = vec(T(l, "ssm_norm.weight"));
        for (int h = 0; h < NH; ++h) {
            rmsnorm(o.data() + size_t(h) * D, D, on.data(), c.rms_eps);
            for (int i = 0; i < D; ++i) o[size_t(h) * D + i] *= sigmoid(g2[size_t(h) * D + i]);
        }
        dump("kda_gated", l, o);
        auto ko = mv(T(l, "attn_output.weight"), o.data());
        dump("kda_out", l, ko);
        return ko;
    }

    std::vector<float> mla(int l, const std::vector<float> & x) {
        auto & st = L[size_t(l)];
        auto qr = mv(T(l, "attn_q_a.weight"), x.data());
        rmsnorm(qr.data(), c.q_lora, vec(T(l, "attn_q_a_norm.weight")).data(), c.rms_eps);
        dump("q_resid", l, qr);
        const auto q = mv(T(l, "attn_q_b.weight"), qr.data());  // [heads][head_k_mla]
        auto lat = mv(T(l, "attn_kv_a_mqa.weight"), x.data());
        rmsnorm(lat.data(), c.kv_lora, vec(T(l, "attn_kv_a_norm.weight")).data(), c.rms_eps);
        dump("kv_cmpr", l, lat);
        st.latent.push_back(lat);
        if (int(st.latent.size()) > c.index_topk) throw std::runtime_error("context beyond the indexer top_k: sparse MLA not implemented");
        const auto & kb = T(l, "attn_k_b.weight"), & vb = T(l, "attn_v_b.weight");
        const int n = int(st.latent.size()), R = c.kv_lora;
        const float kq_scale = 1.f / std::sqrt(float(c.head_k_mla));
        std::vector<float> out(static_cast<size_t>(c.heads) * c.head_v_mla);
        for (int h = 0; h < c.heads; ++h) {
            const auto qa = mv3(kb, uint64_t(h), q.data() + size_t(h) * c.head_k_mla);  // [R]
            std::vector<float> p(static_cast<size_t>(n));
            float m = -INFINITY;
            for (int t = 0; t < n; ++t) {
                double s = 0;
                for (int j = 0; j < R; ++j) s += double(qa[size_t(j)]) * st.latent[size_t(t)][size_t(j)];
                p[size_t(t)] = float(s) * kq_scale; m = std::max(m, p[size_t(t)]);
            }
            double z = 0;
            for (auto & e : p) z += (e = std::exp(e - m));
            std::vector<float> kv(static_cast<size_t>(R), 0.f);
            for (int t = 0; t < n; ++t) for (int j = 0; j < R; ++j) kv[size_t(j)] += float(p[size_t(t)] / z) * st.latent[size_t(t)][size_t(j)];
            const auto o = mv3(vb, uint64_t(h), kv.data());
            std::copy(o.begin(), o.end(), out.begin() + size_t(h) * c.head_v_mla);
        }
        dump("kqv_out", l, out);
        auto ao = mv(T(l, "attn_output.weight"), out.data());
        dump("attn_out", l, ao);
        return ao;
    }

    std::vector<float> swiglu_ffn(int l, const char * g, const char * u, const char * dn, const std::vector<float> & x,
                                  float limit) const {
        auto a = mv(T(l, g), x.data());
        const auto b = mv(T(l, u), x.data());
        for (size_t i = 0; i < a.size(); ++i) {
            const float gg = limit > 1e-6f ? std::min(a[i], limit) : a[i];
            const float uu = limit > 1e-6f ? std::clamp(b[i], -limit, limit) : b[i];
            a[i] = silu(gg) * uu;
        }
        return mv(T(l, dn), a.data());
    }

    std::vector<float> moe(int l, const std::vector<float> & x) const {
        const auto logits = mv(T(l, "ffn_gate_inp.weight"), x.data());
        const auto bias = vec(T(l, "exp_probs_b.bias"));
        MoeConfig mc;
        mc.n_expert = c.experts; mc.n_expert_used = c.topk; mc.gating = MoeGating::Sigmoid;
        mc.norm_weights = c.weight_norm; mc.weight_scale = c.weight_scale;
        mc.swiglu_clamp = c.clamp_exp[size_t(l)];
        const auto sel = moe_route(mc, logits.data(), bias.data());
        std::vector<float> out = swiglu_ffn(l, "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", x,
                                            c.clamp_shexp[size_t(l)]);
        const float limit = c.clamp_exp[size_t(l)];
        for (size_t i = 0; i < sel.ids.size(); ++i) {
            const uint64_t e = uint64_t(sel.ids[i]);
            auto a = mv3(T(l, "ffn_gate_exps.weight"), e, x.data());
            const auto b = mv3(T(l, "ffn_up_exps.weight"), e, x.data());
            for (size_t j = 0; j < a.size(); ++j) a[j] = silu(std::min(a[j], limit)) * std::clamp(b[j], -limit, limit);
            const auto y = mv3(T(l, "ffn_down_exps.weight"), e, a.data());
            for (size_t j = 0; j < out.size(); ++j) out[j] += sel.weights[i] * y[j];
        }
        return out;
    }

    uint32_t step(uint32_t token) {
        const int H = c.hidden, hc = c.hc;
        const auto & emb = w_.at("token_embd.weight");
        std::vector<float> x(static_cast<size_t>(H));
        dequantize_host(emb.type, emb.p + size_t(token) * emb.row_bytes, x.data(), H);
        std::vector<float> S(static_cast<size_t>(hc) * H);
        for (int i = 0; i < hc; ++i) std::copy(x.begin(), x.end(), S.begin() + size_t(i) * H);
        std::vector<float> post, comb;
        for (int l = 0; l < c.layers; ++l) {
            auto h = hc_pre(S, l, "attn", post, comb);
            dump("hc_attn_pre", l, h);
            rmsnorm(h.data(), H, vec(T(l, "attn_norm.weight")).data(), c.rms_eps);
            dump("attn_norm", l, h);
            hc_post(c.recurrent[size_t(l)] ? kda(l, h) : mla(l, h), S, post, comb);
            dump("hc_attn_post", l, S);
            h = hc_pre(S, l, "ffn", post, comb);
            dump("hc_ffn_pre", l, h);
            rmsnorm(h.data(), H, vec(T(l, "ffn_norm.weight")).data(), c.rms_eps);
            dump("ffn_norm", l, h);
            const auto fo = l < c.dense_lead ? swiglu_ffn(l, "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", h,
                                                          c.clamp_shexp[size_t(l)])
                                             : moe(l, h);
            dump("ffn_out", l, fo);
            hc_post(fo, S, post, comb);
            dump("l_out", l, S);
        }
        std::vector<float> m(static_cast<size_t>(H), 0.f);
        for (int i = 0; i < hc; ++i) for (int j = 0; j < H; ++j) m[size_t(j)] += S[size_t(i) * H + j] / float(hc);
        dump("hc_head", -1, m);
        rmsnorm(m.data(), H, vec(w_.at("output_norm.weight")).data(), c.rms_eps);
        dump("result_norm", -1, m);
        logits = mv(w_.at("output.weight"), m.data());
        dump("result_output", -1, logits);
        return uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
};

std::vector<uint32_t> parse_ids(const std::string & s) {
    std::vector<uint32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(',', i);
        v.push_back(uint32_t(std::stoul(s.substr(i, j == std::string::npos ? std::string::npos : j - i))));
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: glm-ref MODEL.gguf --tokens 1,2,3 [--gen N]");
        std::vector<uint32_t> prompt;
        int gen = 8;
        std::vector<uint32_t> force;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) prompt = parse_ids(argv[++i]);
            else if (a == "--gen" && i + 1 < argc) gen = std::stoi(argv[++i]);
            else if (a == "--force" && i + 1 < argc) force = parse_ids(argv[++i]);
            else throw std::runtime_error("unknown option " + a);
        }
        if (prompt.empty()) throw std::runtime_error("--tokens is empty");
        g_dump = std::getenv("GLM_REF_DUMP") != nullptr;
        Weights w(argv[1]);
        Model m(w);
        std::fprintf(stderr, "glm-ref: %d layers (%d KDA), hidden %d, %d experts top-%d\n", m.c.layers,
                     int(std::count(m.c.recurrent.begin(), m.c.recurrent.end(), 1)), m.c.hidden, m.c.experts, m.c.topk);
        // top-4 log-softmax, the format llama-server's n_probs prints, for a like-for-like comparison
        auto top4 = [&](const char * tag, size_t i, uint32_t in) {
            const auto & lg = m.logits;
            const float mx = *std::max_element(lg.begin(), lg.end());
            double z = 0;
            for (float v : lg) z += std::exp(double(v) - mx);
            const double lz = mx + std::log(z);
            std::vector<uint32_t> idx(lg.size());
            for (size_t k = 0; k < idx.size(); ++k) idx[k] = uint32_t(k);
            std::partial_sort(idx.begin(), idx.begin() + 4, idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] > lg[b]; });
            std::printf("%s %zu in %u:", tag, i, in);
            for (int k = 0; k < 4; ++k) std::printf(" (%u, %.3f)", idx[size_t(k)], double(lg[idx[size_t(k)]]) - lz);
            std::printf("\n");
            std::fflush(stdout);
            return idx[0];
        };
        uint32_t next = 0;
        for (size_t i = 0; i < prompt.size(); ++i) { m.step(prompt[i]); next = top4("prompt", i, prompt[i]); }
        g_dump = false;
        if (!force.empty()) {  // teacher-forced continuation: the distribution after each forced token
            for (size_t i = 0; i + 1 < force.size(); ++i) { m.step(force[i]); top4("forced", i, force[i]); }
            return 0;
        }
        std::printf("gen:");
        for (int g = 0; g < gen; ++g) {
            std::printf(" %u", next);
            std::fflush(stdout);
            if (g + 1 < gen) next = m.step(next);
        }
        std::printf("\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "glm-ref: %s\n", e.what());
        return 1;
    }
}

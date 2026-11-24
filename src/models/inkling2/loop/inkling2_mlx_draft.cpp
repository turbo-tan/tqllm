#include "inkling2_mlx_draft.hpp"
#include "mlx_kv.hpp"
#include "safetensors.hpp"
#include "gguf.hpp"
#include "hf_folder.hpp"
#include "dflash_mlx.hpp"
#include <cstring>
#include <functional>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
mx::array f32(const mx::array& x) { return mx::astype(x, mx::float32); }
mx::array bf16(const mx::array& x) { return mx::astype(x, mx::bfloat16); }
mx::array linear(const mx::array& x, const mx::array& w) {
    return mx::matmul(mx::astype(x, w.dtype()), mx::swapaxes(w, -1, -2));
}
// The draft layers' projections in 8 bits: MLX affine weights, from mtp.safetensors quantised at load (group 64)
// or from Unsloth's Q8_0 drafter GGUF converted exactly (group 32, scale d, bias -128 d, FP16). The Markov head
// stays BF16.
struct Q8 {
    mx::array w, scales, biases;
    int group;
    Q8(mx::array weight, mx::array scale, mx::array bias, int g) : w(weight), scales(scale), biases(bias), group(g) {
        mx::eval({w, scales, biases});
    }
    static Q8 quantize(const mx::array& bf) { auto q = mx::quantize(bf, 64, 8); return Q8(q[0], q[1], q[2], 64); }
    // Q8_0 rows (blocks of a FP16 d and 32 int8) as affine 8-bit: w = d * q = d * (q + 128) - 128 d, exact in FP16
    static Q8 from_q8_0(const std::vector<char>& raw, int rows, int cols) {
        if (cols % 32 || raw.size() != size_t(rows) * cols / 32 * 34) throw std::runtime_error("inkling2 MLX draft: Q8_0 size");
        std::vector<uint8_t> codes(size_t(rows) * cols);
        std::vector<uint16_t> scale(size_t(rows) * cols / 32), bias(scale.size());
        for (size_t b = 0; b < scale.size(); ++b) {
            const char* block = raw.data() + b * 34;
            uint16_t d;
            std::memcpy(&d, block, 2);
            scale[b] = d;
            // -128 d: exponent + 7, sign flipped (d is a normal FP16 or 0 in these files)
            const uint16_t exp = (d >> 10) & 31;
            if (exp && exp + 7 >= 31) throw std::runtime_error("inkling2 MLX draft: Q8_0 scale out of FP16 range");
            bias[b] = d & 0x7fff ? uint16_t((d ^ 0x8000) + (exp ? (7u << 10) : 0)) : 0;
            if (!exp && (d & 0x7fff)) throw std::runtime_error("inkling2 MLX draft: subnormal Q8_0 scale");
            for (int i = 0; i < 32; ++i) codes[b * 32 + size_t(i)] = uint8_t(int(int8_t(block[2 + i])) + 128);
        }
        auto w = mx::view(mx::array(codes.data(), {rows, cols}, mx::uint8), mx::uint32);
        auto sc = mx::view(mx::array(scale.data(), {rows, cols / 32}, mx::uint16), mx::float16);
        auto bi = mx::view(mx::array(bias.data(), {rows, cols / 32}, mx::uint16), mx::float16);
        return Q8(mx::contiguous(w), mx::contiguous(sc), mx::contiguous(bi), 32);
    }
    static Q8 rows(const std::vector<Q8>& parts) {
        if (parts.size() == 1) return parts[0];
        std::vector<mx::array> w, s, b;
        for (const auto& p : parts) { w.push_back(p.w); s.push_back(p.scales); b.push_back(p.biases); }
        return Q8(mx::concatenate(w, 0), mx::concatenate(s, 0), mx::concatenate(b, 0), parts[0].group);
    }
};
mx::array linear(const mx::array& x, const Q8& q) {
    return mx::astype(mx::quantized_matmul(mx::astype(x, q.scales.dtype()), q.w, q.scales, q.biases, true, q.group, 8), mx::bfloat16);
}
// The drafter's tensors by their mtp.safetensors names: dense ones as stored (BF16 / F32) and projections (one or
// more stacked by rows) in 8 bits. From a checkpoint folder (mtp.safetensors) or a dflash-architecture GGUF.
struct Source {
    std::function<mx::array(const std::string&)> tensor;
    std::function<Q8(const std::vector<std::string>&)> projection;
};
Source safetensors_source(const std::string& folder) {
    auto index = std::make_shared<std::map<std::string, SafeTensor>>(
        safetensors_index_file(std::filesystem::path(folder) / "mtp.safetensors"));
    auto get = [index](const std::string& name) {
        const auto it = index->find(name);
        if (it == index->end() || it->second.dtype != "BF16") throw std::runtime_error("inkling2 MLX draft: missing BF16 " + name);
        mx::Shape shape;
        for (auto d : it->second.shape) shape.push_back(int(d));
        const auto bytes = safetensors_read(it->second);
        mx::array raw(reinterpret_cast<const uint8_t*>(bytes.data()), {int(bytes.size())}, mx::uint8);
        auto value = mx::reshape(mx::view(raw, mx::bfloat16), shape);
        mx::eval(value);
        return value;
    };
    return {get, [get](const std::vector<std::string>& names) {
        std::vector<mx::array> parts;
        for (const auto& n : names) parts.push_back(get(n));
        return Q8::quantize(parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0));
    }};
}
// GGUF (general.architecture dflash, Unsloth's dspark-Inkling-2-Mini-*.gguf) names of the mtp.safetensors tensors
std::string gguf_name(const std::string& hf) {
    static const std::map<std::string, std::string> top{{"fuse_weight", "dspark_fuse.weight"}, {"mask_embed", "mask_embd.weight"},
        {"norm.weight", "output_norm.weight"}, {"markov_head.dflare_norm.weight", "dspark_norm.weight"},
        {"markov_head.markov_w1.weight", "markov_w1.weight"}, {"markov_head.rnn_proj.weight", "joint_proj.weight"},
        {"markov_head.rnn_proj.bias", "joint_proj.bias"}, {"markov_head.hidden_out.weight", "dspark_hidden_out.weight"},
        {"markov_head.hidden_out.bias", "dspark_hidden_out.bias"}};
    static const std::map<std::string, std::string> layer{{"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"}, {"dflare_norm.weight", "dflare_norm.weight"},
        {"self_attn.q_norm.weight", "attn_q_norm.weight"}, {"self_attn.k_norm.weight", "attn_k_norm.weight"},
        {"self_attn.k_norm_ctx.weight", "attn_k_ctx_norm.weight"}, {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"}, {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"}, {"self_attn.k_proj_ctx.weight", "attn_k_ctx.weight"},
        {"self_attn.v_proj_ctx.weight", "attn_v_ctx.weight"}, {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"}, {"mlp.down_proj.weight", "ffn_down.weight"}};
    if (auto it = top.find(hf); it != top.end()) return it->second;
    if (hf.rfind("layers.", 0) == 0) {
        const auto dot = hf.find('.', 7);
        if (auto it = layer.find(hf.substr(dot + 1)); it != layer.end()) return "blk." + hf.substr(7, dot - 7) + "." + it->second;
    }
    throw std::runtime_error("inkling2 MLX draft: no GGUF name for " + hf);
}
Source gguf_source(const std::string& path) {
    auto file = std::make_shared<GgufFile>(gguf_open(path));
    if (file->str("general.architecture") != "dflash" || file->num("dflash.block_size") != 8 ||
        file->num("dflash.dspark.markov_rank", 0) != 256 || file->num("dflash.sample_from_anchor", 0) != 0)
        throw std::runtime_error("inkling2 MLX draft: " + path + " is not an Inkling-2 DSpark drafter GGUF");
    auto find = [file](const std::string& hf) -> const GgufTensor& {
        const auto it = file->tensors.find(gguf_name(hf));
        if (it == file->tensors.end()) throw std::runtime_error("inkling2 MLX draft: GGUF misses " + gguf_name(hf));
        return it->second;
    };
    auto tensor = [file, find](const std::string& hf) {
        const auto& t = find(hf);
        mx::Shape shape;
        for (auto it = t.ne.rbegin(); it != t.ne.rend(); ++it) shape.push_back(int(*it));
        const auto bytes = file->read(t);
        mx::array raw(reinterpret_cast<const uint8_t*>(bytes.data()), {int(bytes.size())}, mx::uint8);
        mx::Dtype type = t.type == GGML_F32 ? mx::float32 : t.type == GGML_BF16 ? mx::bfloat16 : t.type == GGML_F16 ? mx::float16 : mx::uint8;
        if (type == mx::uint8) throw std::runtime_error("inkling2 MLX draft: dense GGUF tensor type " + std::to_string(t.type));
        auto value = mx::reshape(mx::view(raw, type), shape);
        mx::eval(value);
        return value;
    };
    return {tensor, [file, find, tensor](const std::vector<std::string>& names) {
        std::vector<Q8> parts;
        for (const auto& n : names) {
            const auto& t = find(n);
            if (t.ne.size() != 2) throw std::runtime_error("inkling2 MLX draft: projection rank " + n);
            if (t.type == GGML_Q8_0) parts.push_back(Q8::from_q8_0(file->read(t), int(t.ne[1]), int(t.ne[0])));
            else parts.push_back(Q8::quantize(mx::astype(tensor(n), mx::bfloat16)));
        }
        return Q8::rows(parts);
    }};
}
mx::array cols(const mx::array& x, int from, int to) {
    mx::Shape start(x.ndim(), 0), stop = x.shape();
    start.back() = from; stop.back() = to;
    return mx::slice(x, start, stop);
}
constexpr float kEps = 1e-6f, kRopeBase = 1e6f;
constexpr int kLayers = 3, kHeads = 24, kKvHeads = 8, kHeadDim = 128, kRank = 256;
}  // namespace

struct Inkling2MlxDraft::Impl {
    struct Layer {
        mx::array in_norm, post_norm, dflare, q_norm, k_norm, k_norm_ctx;
        Q8 qkv, o, gate, up, down, kv_ctx;
        MlxKvCache cache;
    };
    std::vector<Layer> layers;
    std::optional<mx::array> mask_embed, norm, markov_norm, markov_w1, rnn_state, rnn_token, rnn_hidden, rnn_frnn, rnn_bias;
    std::optional<mx::array> alpha, fold, fold_bias, f_rnn;
    std::optional<Q8> fold8;  // the fold in 8 bits (fold itself is released after load)
    Inkling2MlxDraft::Head head;
    int D = 0, V = 0;

    Impl(const std::string& folder, int context, const mx::array& lm_head, int vocab, int window, Inkling2MlxDraft::Head run_head)
        : head(std::move(run_head)), V(vocab) {
        const bool gguf = folder.size() > 5 && folder.compare(folder.size() - 5, 5, ".gguf") == 0;
        const auto source = gguf ? gguf_source(folder) : safetensors_source(folder);
        auto get = [&](const std::string& name) { return bf16(source.tensor(name)); };
        auto vec = [&](const std::string& name) { auto v = f32(source.tensor(name)); mx::eval(v); return v; };
        D = lm_head.shape(1);
        for (int i = 0; i < kLayers; ++i) {
            const std::string d = "layers." + std::to_string(i) + ".";
            auto proj = [&](std::vector<std::string> names) {
                for (auto& n : names) n = d + n;
                return source.projection(names);
            };
            auto qkv = proj({"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"});
            if (qkv.w.shape(0) != (kHeads + 2 * kKvHeads) * kHeadDim) throw std::runtime_error("inkling2 MLX draft: qkv shape");
            layers.push_back(Layer{vec(d + "input_layernorm.weight"), vec(d + "post_attention_layernorm.weight"),
                vec(d + "dflare_norm.weight"), vec(d + "self_attn.q_norm.weight"), vec(d + "self_attn.k_norm.weight"),
                vec(d + "self_attn.k_norm_ctx.weight"), qkv, proj({"self_attn.o_proj.weight"}), proj({"mlp.gate_proj.weight"}),
                proj({"mlp.up_proj.weight"}), proj({"mlp.down_proj.weight"}),
                proj({"self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight"}),
                MlxKvCache(context, i != 1 ? window : 0)});  // local_layer_ids [0, 2]
        }
        {   // alpha = softmax(fuse_weight) per row: draft layers 0..2, then the Markov head
            auto a = mx::softmax(f32(source.tensor("fuse_weight")), -1, true);
            mx::eval(a);
            alpha = a;
        }
        mask_embed = vec("mask_embed"); norm = vec("norm.weight"); markov_norm = vec("markov_head.dflare_norm.weight");
        markov_w1 = get("markov_head.markov_w1.weight");
        // rnn_proj [3R, 2R + 2D] over z = [state | w1[prev] | hidden | f_rnn]: split by input so the hidden and f_rnn
        // parts run once per draft rather than once per Markov step.
        auto rnn = get("markov_head.rnn_proj.weight");
        if (rnn.shape() != mx::Shape{3 * kRank, 2 * kRank + 2 * D}) throw std::runtime_error("inkling2 MLX draft: rnn_proj shape");
        auto part = [&](int from, int to) { auto p = mx::contiguous(cols(rnn, from, to)); mx::eval(p); return p; };
        rnn_state = part(0, kRank); rnn_token = part(kRank, 2 * kRank);
        rnn_hidden = part(2 * kRank, 2 * kRank + D); rnn_frnn = part(2 * kRank + D, 2 * kRank + 2 * D);
        rnn_bias = vec("markov_head.rnn_proj.bias");
        // The latent goes through hidden_out and the target head: fold = unembed[:V] @ hidden_out ([V, R]) at load.
        const auto unembed = mx::slice(lm_head, {0, 0}, {V, D});
        auto hidden_out = get("markov_head.hidden_out.weight");  // [D, R]
        auto fb = f32(linear(mx::reshape(get("markov_head.hidden_out.bias"), {1, D}), unembed));
        fold = bf16(mx::matmul(unembed, hidden_out));
        fold_bias = mx::reshape(fb, {V});
        mx::eval({*fold, *fold_bias});
        // The fold is read once per drafted token (100 MB in BF16); drafts only propose (the verify decides), so it is
        // kept in 8 bits: about 2 % of a speculative step, acceptance unchanged on the golden prompts.
        fold8 = Q8::quantize(*fold); fold.reset();
    }
    static mx::array norm_rows(const mx::array& x, const mx::array& w) { return mx::fast::rms_norm(f32(x), w, kEps); }
    // [n, heads * HD] -> [1, heads, n, HD]: per-head RMSNorm, then NeoX RoPE at positions pos ..
    static mx::array heads(const mx::array& x, int count, const mx::array& w, int pos) {
        const int n = x.shape(0);
        auto y = bf16(mx::fast::rms_norm(f32(mx::reshape(x, {n, count, kHeadDim})), w, kEps));
        y = mx::expand_dims(mx::transpose(y, {1, 0, 2}), 0);
        return mx::fast::rope(y, kHeadDim, false, kRopeBase, 1.f, pos);
    }
    void commit(const mx::array& taps, int pos0) {
        const int m = taps.shape(0);
        if (taps.ndim() != 3 || taps.shape(1) != 4 || taps.shape(2) != D) throw std::invalid_argument("inkling2 MLX draft taps");
        auto mix = [&](int row) {  // sum_j alpha[row][j] * taps[:, j], F32 -> [m, D]
            auto a = mx::reshape(mx::slice(*alpha, {row, 0}, {row + 1, 4}), {1, 4, 1});
            return mx::sum(mx::multiply(f32(taps), a), 1);
        };
        for (int i = 0; i < kLayers; ++i) {
            auto& Y = layers[size_t(i)];
            auto cx = bf16(norm_rows(mix(i), Y.dflare));
            auto kv = linear(cx, Y.kv_ctx);
            auto k = heads(cols(kv, 0, kKvHeads * kHeadDim), kKvHeads, Y.k_norm_ctx, pos0);
            auto v = mx::expand_dims(mx::transpose(mx::reshape(cols(kv, kKvHeads * kHeadDim, 2 * kKvHeads * kHeadDim),
                                                               {m, kKvHeads, kHeadDim}), {1, 0, 2}), 0);
            Y.cache.append(k, v);
            Y.cache.retain_window();
        }
        f_rnn = bf16(norm_rows(mx::slice(mix(3), {m - 1, 0}, {m, D}), *markov_norm));
    }
    mx::array propose(const mx::array& x0, const mx::array& anchor, int P, int steps) {
        if (!f_rnn) throw std::runtime_error("inkling2 MLX draft: propose before any committed token");
        if (steps < 1 || steps >= kBlock) throw std::invalid_argument("inkling2 MLX draft steps");
        auto h = mx::concatenate({f32(x0), mx::broadcast_to(mx::reshape(*mask_embed, {1, D}), {kBlock - 1, D})}, 0);
        for (auto& Y : layers) {
            auto qkv = linear(bf16(norm_rows(h, Y.in_norm)), Y.qkv);
            const int q_end = kHeads * kHeadDim, k_end = q_end + kKvHeads * kHeadDim;
            auto q = heads(cols(qkv, 0, q_end), kHeads, Y.q_norm, P);
            auto k = heads(cols(qkv, q_end, k_end), kKvHeads, Y.k_norm, P);
            auto v = mx::expand_dims(mx::transpose(mx::reshape(cols(qkv, k_end, k_end + kKvHeads * kHeadDim),
                                                               {kBlock, kKvHeads, kHeadDim}), {1, 0, 2}), 0);
            // Context keys (window for the local layers) then all 8 block keys; no causal mask inside the block.
            auto [keys, values] = Y.cache.peek(k, v);
            auto att = mx::fast::scaled_dot_product_attention(q, keys, values, 1.f / std::sqrt(float(kHeadDim)), "");
            att = mx::reshape(mx::transpose(mx::squeeze(att, 0), {1, 0, 2}), {kBlock, kHeads * kHeadDim});
            h = mx::add(h, f32(linear(att, Y.o)));
            auto xb = bf16(norm_rows(h, Y.post_norm));
            auto g = f32(linear(xb, Y.gate)), u = f32(linear(xb, Y.up));
            auto act = bf16(mx::multiply(mx::multiply(g, mx::sigmoid(g)), u));
            h = mx::add(h, f32(linear(act, Y.down)));
        }
        auto hid = bf16(norm_rows(mx::slice(h, {1, 0}, {1 + steps, D}), *norm));  // row i + 1 predicts draft i
        auto base = mx::add(f32(cols(head(hid), 0, V)), *fold_bias);           // [steps, V]
        auto fixed = mx::add(mx::add(f32(linear(hid, *rnn_hidden)), f32(linear(*f_rnn, *rnn_frnn))), *rnn_bias);
        auto state = mx::zeros({1, kRank}, mx::float32);
        auto prev = mx::reshape(mx::astype(anchor, mx::uint32), {1});
        std::vector<mx::array> tokens;
        for (int i = 0; i < steps; ++i) {
            auto proj = mx::add(mx::add(f32(linear(bf16(state), *rnn_state)), f32(linear(mx::take(*markov_w1, prev, 0), *rnn_token))),
                                mx::slice(fixed, {i, 0}, {i + 1, 3 * kRank}));
            auto gate = mx::sigmoid(cols(proj, 0, kRank));
            auto cand = mx::tanh(cols(proj, kRank, 2 * kRank));
            state = mx::add(mx::multiply(gate, state), mx::multiply(mx::subtract(mx::array(1.f), gate), cand));
            auto latent = bf16(mx::tanh(cols(proj, 2 * kRank, 3 * kRank)));
            // The softcap is monotonic, so the argmax of the uncapped logit is the draft token.
            auto logit = mx::add(mx::slice(base, {i, 0}, {i + 1, V}), f32(linear(latent, *fold8)));
            prev = mx::reshape(mx::astype(mx::argmax(logit, -1), mx::uint32), {1});
            tokens.push_back(prev);
        }
        return mx::concatenate(tokens, 0);
    }
};

Inkling2MlxDraft::Inkling2MlxDraft(const std::string& folder, int context, const mx::array& unembed, int vocab, int window, Head head)
    : p_(std::make_unique<Impl>(folder, context, unembed, vocab, window, std::move(head))) {}
Inkling2MlxDraft::~Inkling2MlxDraft() = default;
void Inkling2MlxDraft::commit(const mx::array& taps, int pos0) { p_->commit(taps, pos0); }
mx::array Inkling2MlxDraft::propose(const mx::array& x0, const mx::array& anchor, int position, int steps) {
    return p_->propose(x0, anchor, position, steps);
}
std::vector<mx::array> Inkling2MlxDraft::state() const {
    std::vector<mx::array> out;
    for (const auto& Y : p_->layers) Y.cache.collect(out);
    if (p_->f_rnn) out.push_back(*p_->f_rnn);
    return out;
}
namespace {
struct DSparkKept { std::vector<MlxKvCache> caches; std::optional<mx::array> f_rnn; };
struct DFlash2Kept { std::vector<MlxKvCache> caches; };
}  // namespace
std::any Inkling2MlxDraft::keep() const {
    DSparkKept k;
    for (const auto& Y : p_->layers) k.caches.push_back(Y.cache);
    k.f_rnn = p_->f_rnn;
    return k;
}
void Inkling2MlxDraft::restore(const std::any& kept) {
    const auto& k = std::any_cast<const DSparkKept&>(kept);
    for (size_t i = 0; i < p_->layers.size(); ++i) p_->layers[i].cache = k.caches.at(i);
    p_->f_rnn = k.f_rnn;
}
void Inkling2MlxDraft::reset() {
    for (auto& Y : p_->layers) Y.cache.reset();
    p_->f_rnn.reset();
}

// ---- DFlash2 (incoai/Inkling-2-Mini-DFlash2) ----
// Reference: sglang 0.5.21 srt/models/dflash.py (DFlash2DraftModel, DFlashGroupedConv, CandidateSelector) with the
// drafter's Inkling patch (sglang_inkling_dflash2.py: capture ids 5 / 23 / 41 / 48, selector softcap after the edge
// correction, candidates over the real vocabulary).
namespace {
// A projection: 8-bit affine (served) or BF16 as stored (tests)
struct Proj {
    std::optional<Q8> q;
    std::optional<mx::array> w;
    mx::array apply(const mx::array& x) const { return q ? linear(x, *q) : bf16(linear(x, *w)); }
    // rows [a, b) of the output (a sub-projection of a stacked one)
    Proj rows(int a, int b) const {
        if (!q) { auto r = mx::contiguous(mx::slice(*w, {a, 0}, {b, w->shape(1)})); mx::eval(r); return Proj{std::nullopt, r}; }
        auto cut = [&](const mx::array& t) { return mx::contiguous(mx::slice(t, {a, 0}, {b, t.shape(1)})); };  // Q8 evaluates
        return Proj{Q8(cut(q->w), cut(q->scales), cut(q->biases), q->group), std::nullopt};
    }
};
}  // namespace

struct Inkling2MlxDFlash2::Impl {
    struct Layer {
        mx::array in_norm, post_norm, q_norm, k_norm, attn_base, mlp_base;
        Proj qkv, kv, o, attn_kernel, mlp_kernel, gate_up, down;
        MlxKvCache cache;
    };
    std::vector<Layer> layers;
    std::optional<mx::array> fc_norm, norm, predecessor, successor, mask_row;
    std::optional<Proj> fc, selector;
    Inkling2MlxDraft::Head head;
    int D = 0, V = 0, H = 0, G = 0, HD = 0, I = 0, group = 0, rank = 0;
    float eps = 1e-6f, theta = 1e4f;
    std::optional<Trace> trace;

    Impl(const std::string& folder, int context, int vocab, Inkling2MlxDraft::Head run_head, const Embed& embed, bool bf16_proj)
        : head(std::move(run_head)), V(vocab) {
        HfFolder f(folder);
        if (f.architecture() != "DFlash2DraftModel") throw std::runtime_error("inkling2 MLX DFlash2: not a DFlash2DraftModel folder: " + folder);
        const auto* dc = f.section("dflash_config");
        if (!dc) throw std::runtime_error("inkling2 MLX DFlash2: no dflash_config");
        auto dn = [&](const char* key) { const auto* v = dc->get(key); if (!v) throw std::runtime_error(std::string("dflash_config.") + key); return v->num; };
        std::vector<int> taps;
        if (const auto* t = dc->get("target_layer_ids")) for (const auto& v : t->arr) taps.push_back(int(v.num));
        const auto* softcap_mode = dc->get("selector_softcap_mode");
        D = int(f.num("hidden_size")); H = int(f.num("num_attention_heads")); G = int(f.num("num_key_value_heads"));
        HD = int(f.num("head_dim")); I = int(f.num("intermediate_size")); eps = float(f.num("rms_norm_eps"));
        group = int(dn("conv_group_size")); rank = int(dn("selector_rank"));
        if (const auto* rp = f.section("rope_parameters")) if (const auto* th = rp->get("rope_theta")) theta = float(th->num);
        const int L = int(f.num("num_hidden_layers")), window = int(f.num("sliding_window"));
        // the taps are the DSpark ones (Inkling2Mlx::build): entering applications 5, 23, 41 and the final norm (48); the
        // selector's softcap after its edge correction is monotone, so the greedy walk ignores it (dflash_select)
        if (taps != std::vector<int>{5, 23, 41, 48} || dn("block_size") != kBlock || dn("conv_kernel_size") != 2 ||
            dn("selector_top_k") != 16 || rank != 256 || group != 16 || HD != 128 || D % group ||
            !softcap_mode || softcap_mode->str != "after_correction" || window <= 0 || L <= 0)
            throw std::runtime_error("inkling2 MLX DFlash2: unsupported drafter configuration");
        auto get = [&](const std::string& name) {
            const auto& t = f.at(name);
            if (t.dtype != "BF16") throw std::runtime_error("inkling2 MLX DFlash2: " + name + " is not BF16");
            mx::Shape shape;
            for (auto d : t.shape) shape.push_back(int(d));
            const auto bytes = safetensors_read(t);
            mx::array raw(reinterpret_cast<const uint8_t*>(bytes.data()), {int(bytes.size())}, mx::uint8);
            auto value = mx::reshape(mx::view(raw, mx::bfloat16), shape);
            mx::eval(value);
            return value;
        };
        // every array kept is evaluated here: a lazy one would carry this thread's stream into the server's decode
        // thread ("There is no Stream(gpu, 0) in current thread")
        auto vec = [&](const std::string& name, mx::Shape shape = {}) {
            auto v = f32(get(name));
            if (!shape.empty()) v = mx::reshape(v, shape);
            v = mx::contiguous(v);
            mx::eval(v);
            return v;
        };
        auto proj = [&](const std::vector<std::string>& names) {
            std::vector<mx::array> parts;
            for (const auto& n : names) parts.push_back(get(n));
            auto w = parts.size() == 1 ? parts[0] : mx::concatenate(parts, 0);
            if (bf16_proj) { w = mx::contiguous(w); mx::eval(w); return Proj{std::nullopt, w}; }
            return Proj{Q8::quantize(w), std::nullopt};
        };
        const int qn = H * HD, kn = G * HD;
        for (int l = 0; l < L; ++l) {
            const std::string p = "layers." + std::to_string(l) + ".";
            auto qkv = proj({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"});
            layers.push_back(Layer{vec(p + "input_layernorm.weight"), vec(p + "post_attention_layernorm.weight"),
                vec(p + "self_attn.q_norm.weight"), vec(p + "self_attn.k_norm.weight"),
                vec(p + "attention_conv.base_kernel", {4 * D}), vec(p + "mlp_conv.base_kernel", {4 * D}),
                qkv, qkv.rows(qn, qn + 2 * kn), proj({p + "self_attn.o_proj.weight"}),
                proj({p + "attention_conv.kernel_projection.weight"}), proj({p + "mlp_conv.kernel_projection.weight"}),
                proj({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}), proj({p + "mlp.down_proj.weight"}),
                MlxKvCache(context, window)});
        }
        fc = proj({"fc.weight"});
        if (fc->apply(mx::zeros({1, 4 * D}, mx::bfloat16)).shape(1) != D) throw std::runtime_error("inkling2 MLX DFlash2: fc shape");
        fc_norm = vec("hidden_norm.weight"); norm = vec("norm.weight");
        selector = Proj{std::nullopt, get("candidate_selector.hidden_projection.weight")};
        predecessor = get("candidate_selector.predecessor_codebook"); successor = get("candidate_selector.successor_codebook");
        if (predecessor->shape(1) != rank || predecessor->shape(0) < V || successor->shape() != predecessor->shape())
            throw std::runtime_error("inkling2 MLX DFlash2: selector codebook shape");
        const uint32_t mask = uint32_t(dn("mask_token_id"));
        mask_row = f32(embed(mx::array(&mask, {1}, mx::uint32)));
        mx::eval(*mask_row);
        if (mask_row->shape() != mx::Shape{1, D}) throw std::runtime_error("inkling2 MLX DFlash2: target width");
    }
    mx::array rms(const mx::array& x, const mx::array& w) const { return mx::fast::rms_norm(f32(x), w, eps); }
    // [n, count * HD] -> [1, count, n, HD]: per-head RMSNorm, then NeoX RoPE at positions pos ..
    mx::array heads(const mx::array& x, int count, const mx::array& w, int pos) const {
        const int n = x.shape(0);
        auto y = bf16(mx::fast::rms_norm(f32(mx::reshape(x, {n, count, HD})), w, eps));
        y = mx::expand_dims(mx::transpose(y, {1, 0, 2}), 0);
        return mx::fast::rope(y, HD, false, theta, 1.f, pos);
    }
    mx::array values(const mx::array& x, int count) const {
        return mx::expand_dims(mx::transpose(mx::reshape(x, {x.shape(0), count, HD}), {1, 0, 2}), 0);
    }
    void commit(const mx::array& taps, int pos0) {
        const int m = taps.shape(0);
        if (taps.ndim() != 3 || taps.shape(1) != 4 || taps.shape(2) != D) throw std::invalid_argument("inkling2 MLX DFlash2 taps");
        // concat(taps 5, 23, 41, 48) -> fc -> hidden_norm: the context rows every layer projects to keys / values
        auto ctx = bf16(rms(fc->apply(bf16(mx::reshape(taps, {m, 4 * D}))), *fc_norm));
        const int kn = G * HD;
        for (auto& Y : layers) {
            auto kv = Y.kv.apply(ctx);
            Y.cache.append(heads(cols(kv, 0, kn), G, Y.k_norm, pos0), bf16(values(cols(kv, kn, 2 * kn), G)));
            Y.cache.retain_window();
        }
    }
    mx::array propose(const mx::array& x0, const mx::array& anchor, int P, int steps) {
        if (steps < 1 || steps >= kBlock) throw std::invalid_argument("inkling2 MLX DFlash2 steps");
        const int n = kBlock, qn = H * HD, kn = G * HD;
        auto x = mx::concatenate({f32(x0), mx::broadcast_to(*mask_row, {n - 1, D})}, 0);
        for (auto& Y : layers) {
            // attention sublayer: conv(side 0) of the normed rows in, conv(side 1) of its output + the residual out, both
            // from one kernel projection of the normed rows
            auto normed = bf16(rms(x, Y.in_norm));
            auto kernel = Y.attn_kernel.apply(normed);
            auto input = dflash_conv(f32(normed), f32(kernel), Y.attn_base, group, 0);
            auto qkv = Y.qkv.apply(bf16(input));
            auto q = heads(cols(qkv, 0, qn), H, Y.q_norm, P);
            auto k = heads(cols(qkv, qn, qn + kn), G, Y.k_norm, P);
            auto v = bf16(values(cols(qkv, qn + kn, qn + 2 * kn), G));
            auto [keys, vals] = Y.cache.peek(k, v);  // the context window, then the block's own rows (not causal)
            auto att = mx::fast::scaled_dot_product_attention(q, keys, vals, 1.f / std::sqrt(float(HD)), "");
            att = mx::reshape(mx::transpose(mx::squeeze(att, 0), {1, 0, 2}), {n, qn});
            x = dflash_conv(f32(Y.o.apply(bf16(att))), f32(kernel), Y.attn_base, group, 1, x);
            // MLP sublayer, the same wrapping
            normed = bf16(rms(x, Y.post_norm));
            kernel = Y.mlp_kernel.apply(normed);
            input = dflash_conv(f32(normed), f32(kernel), Y.mlp_base, group, 0);
            auto gu = f32(Y.gate_up.apply(bf16(input)));
            auto g = cols(gu, 0, I), u = cols(gu, I, 2 * I);
            auto act = bf16(mx::multiply(bf16(mx::multiply(g, mx::sigmoid(g))), u));
            x = dflash_conv(f32(Y.down.apply(act)), f32(kernel), Y.mlp_base, group, 1, x);
        }
        auto hid = bf16(rms(x, *norm));
        // candidates: the target head over the real tokens (row 0, the anchor's, is not read by the walk)
        auto logits = mx::contiguous(f32(cols(head(hid), 0, V)));
        auto [ids, unary] = dflash_top16(logits);
        auto gate = f32(selector->apply(hid));
        auto succ = f32(mx::take(*successor, ids, 0));
        auto anchor_id = mx::reshape(mx::astype(anchor, mx::uint32), {});
        trace = Trace{f32(hid), ids, unary};
        return dflash_select(ids, unary, gate, succ, *predecessor, anchor_id, steps);
    }
};

Inkling2MlxDFlash2::Inkling2MlxDFlash2(const std::string& folder, int context, int vocab, Head head, Embed embed, bool bf16_projections)
    : p_(std::make_unique<Impl>(folder, context, vocab, std::move(head), embed, bf16_projections)) {}
Inkling2MlxDFlash2::~Inkling2MlxDFlash2() = default;
bool Inkling2MlxDFlash2::is_drafter_folder(const std::string& path) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(path) || !fs::exists(fs::path(path) / "config.json") || !fs::exists(fs::path(path) / "model.safetensors"))
        return false;
    try { return HfFolder(path).architecture() == "DFlash2DraftModel"; } catch (const std::exception&) { return false; }
}
void Inkling2MlxDFlash2::commit(const mx::array& taps, int pos0) { p_->commit(taps, pos0); }
mx::array Inkling2MlxDFlash2::propose(const mx::array& x0, const mx::array& anchor, int position, int steps) {
    return p_->propose(x0, anchor, position, steps);
}
std::vector<mx::array> Inkling2MlxDFlash2::state() const {
    std::vector<mx::array> out;
    for (const auto& Y : p_->layers) Y.cache.collect(out);
    return out;
}
void Inkling2MlxDFlash2::reset() { for (auto& Y : p_->layers) Y.cache.reset(); p_->trace.reset(); }
std::any Inkling2MlxDFlash2::keep() const {
    DFlash2Kept k;
    for (const auto& Y : p_->layers) k.caches.push_back(Y.cache);
    return k;
}
void Inkling2MlxDFlash2::restore(const std::any& kept) {
    const auto& k = std::any_cast<const DFlash2Kept&>(kept);
    for (size_t i = 0; i < p_->layers.size(); ++i) p_->layers[i].cache = k.caches.at(i);
}
Inkling2MlxDFlash2::Trace Inkling2MlxDFlash2::last_trace() const {
    if (!p_->trace) throw std::logic_error("inkling2 MLX DFlash2: no propose yet");
    return *p_->trace;
}
}  // namespace tqllm::mlx_backend

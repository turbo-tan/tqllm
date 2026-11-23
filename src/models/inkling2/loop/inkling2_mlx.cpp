#include "inkling2_mlx.hpp"
#include "inkling2_mlx_ops.hpp"
#include "hf_folder.hpp"
#include "safetensors.hpp"
#include "mlx_floatx.hpp"
#include "mlx_kv.hpp"
#include "inkling2_mlx_draft.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <sys/sysctl.h>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
mx::array f32(const mx::array& x) { return mx::astype(x, mx::float32); }
mx::array bf16(const mx::array& x) { return mx::astype(x, mx::bfloat16); }
// RMSNorm (eps 1e-6, F32 math) as one kernel from any input dtype: F32 out, or BF16 with norm_bf16
mx::array norm(const mx::array& x, const std::optional<mx::array>& w = std::nullopt) { return inkling2_norm(x, w, mx::float32); }
mx::array norm_bf16(const mx::array& x, const std::optional<mx::array>& w = std::nullopt) { return inkling2_norm(x, w, mx::bfloat16); }
mx::array linear(const mx::array& x, const mx::array& w) {
    return mx::matmul(mx::astype(x, w.dtype()), mx::swapaxes(w, -1, -2));
}
mx::array row(const mx::array& x, int at, int axis = 0) {
    // A strided view (slice + squeeze), not a gather kernel.
    if (axis < 0) axis += x.ndim();
    mx::Shape start(x.ndim(), 0), stop = x.shape();
    start[axis] = at; stop[axis] = at + 1;
    return mx::squeeze(mx::slice(x, start, stop), axis);
}
// Depth attention (block attention residuals): the finished blocks and the normalised partial, mixed per query.
mx::array depth(const std::vector<mx::array>& blocks, const mx::array& partial, const mx::array& query) {
    return inkling2_depth(blocks, partial, query).first;  // one fused kernel (inkling2_mlx_ops.hpp)
}
mx::array clip(const mx::array& in, float limit, const std::optional<mx::array>& scale = std::nullopt,
               const std::optional<mx::array>& row_scale = std::nullopt) {
    return inkling2_clip(in, limit, scale, row_scale);  // one kernel: scale, L2 size, min(limit / size, 1), BF16 (x row scale)
}
mx::array read_tensor(const SafeTensor& t) {
    mx::Shape shape;
    for (auto d : t.shape) {
        if (!d || d > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error("inkling2 MLX tensor shape");
        shape.push_back(int(d));
    }
    if (t.bytes > uint64_t(std::numeric_limits<int>::max())) throw std::runtime_error("inkling2 MLX tensor too large");
    auto bytes = safetensors_read(t);
    mx::array raw(reinterpret_cast<const uint8_t*>(bytes.data()), {int(bytes.size())}, mx::uint8);
    if (t.dtype == "BF16") return mx::reshape(mx::view(raw, mx::bfloat16), shape);
    if (t.dtype == "F32") return mx::reshape(mx::view(raw, mx::float32), shape);
    if (t.dtype == "U8" || t.dtype == "F8_E4M3" || t.dtype == "F8_E4M3FN") return mx::reshape(raw, shape);
    throw std::runtime_error("inkling2 MLX tensor dtype " + t.dtype);
}
struct Visit {
    MlxKvCache kv;
    std::array<std::optional<mx::array>, 4> conv;  // k, v, attention out, mlp out: [3, C] F32 history
    // During a verify block: each convolution's history before the block and its input rows, for commit()'s rollback.
    std::array<std::optional<mx::array>, 4> before, input;
    Visit(int context, int window) : kv(context, window) {}
};
}  // namespace

struct Inkling2Mlx::Impl {
    int D, L, H, G, HD, DR, I, E, topk, window, extent, core0, core1, passes, V, context, length = 0;
    float qclip, rclip, logfloor, logalpha, route_scale, gate_cap, up_cap, logit_cap;
    std::map<std::string, mx::array> weights;
    std::vector<std::array<FloatxExperts, 2>> experts;
    std::vector<Visit> visits;
    std::vector<bool> full;
    std::vector<mx::array> message_queries;
    std::array<double,5>* phase_ms=nullptr;
    bool lazy=false, verifying=false;

    // --exact off: the big projections (q, k, v, output, output gate) and the LM head as 8-bit affine weights (group
    // 64; greedy tokens still come from the BF16 head, see head_approx); the sinks, relative bias, conv projections,
    // controllers and router stay BF16. Not exact against the
    // reference: gated by KLD / PPL against BF16 (0.00267 / 5.671 vs 0.00241 / 5.667; 4-bit fails at 0.018).
    static constexpr int kQ8Group = 64;
    std::map<std::string, std::vector<mx::array>> q8;
    // Exact greedy decode: a 6-bit copy of the BF16 LM head (group 64, F32 scales, so the scores are F32) with each
    // row's error bound; greedy steps and verify blocks score all rows with it and recompute in BF16 only the rows that
    // can still win (inkling2_head_select): 0.35 GB read a token in place of 0.92 GB, about 2 rows recomputed. The full
    // head stays for forward()'s logits. (8-bit: ~1 row recomputed but 0.46 GB; 4-bit: some tokens recompute most rows.)
    static constexpr int kHeadBits = 6, kHeadGroup = 64;
    std::vector<mx::array> head_approx;
    std::optional<mx::array> head_bound, head_input;
    Impl(const std::string& path, int ctx, bool exact) : context(ctx) {
        mx::set_default_device(mx::Device::gpu);
        mx::set_cache_limit(size_t(512)*1024*1024);
        HfFolder folder(path);
        if (folder.architecture() != "Inkling2LoopForConditionalGeneration") throw std::runtime_error("not an Inkling2 loop checkpoint");
        auto n = [&](const char* key) { return int(folder.num(key)); };
        D = n("hidden_size"); L = n("num_hidden_layers"); H = n("num_attention_heads"); G = n("num_key_value_heads");
        HD = n("head_dim"); DR = n("d_rel"); I = n("intermediate_size"); E = n("n_routed_experts");
        topk = n("num_experts_per_tok"); window = n("sliding_window_size"); extent = n("rel_extent"); V = n("unpadded_vocab_size");
        const auto* loop = folder.section("loop_config");
        if (!loop) throw std::runtime_error("inkling2 MLX missing loop config");
        auto ln = [&](const char* key) { auto* v = loop->get(key); if (!v) throw std::runtime_error(key); return int(v->num); };
        core0 = ln("core_start"); core1 = ln("core_end"); passes = ln("num_passes");
        qclip = float(folder.num("qk_l2_clip_threshold")); rclip = float(folder.num("rel_l2_clip_threshold"));
        logfloor = float(folder.num("log_scaling_n_floor")); logalpha = float(folder.num("log_scaling_alpha"));
        route_scale = float(folder.num("route_scale")); gate_cap = float(folder.num("mlp_gate_softcap"));
        up_cap = float(folder.num("mlp_up_softcap")); logit_cap = float(folder.num("final_logit_softcapping"));
        if (D != 2304 || L != 24 || H != 24 || G != 8 || HD != 128 || DR != 16 || I != 1152 || E != 128 || topk != 8 ||
            core0 != 6 || core1 != 18 || passes != 3 || n("sconv_kernel_size") != 4 ||
            n("n_layers_per_attn_res_block") != 6 || folder.num("rms_norm_eps") != 1e-6 || context <= 0)
            throw std::runtime_error("unsupported Inkling2 MLX shape/configuration");
        const auto index = safetensors_index(path);
        size_t bytes = 0;
        for (const auto& [name, tensor] : index) {
            if (name.rfind("model.llm.", 0) != 0) continue;
            auto array = read_tensor(tensor);
            mx::eval(array);
            bytes += tensor.bytes;
            weights.emplace(name.substr(10), std::move(array));
        }
        // Norm weights, clip scales and static convolution taps are only ever used in F32: convert once.
        for (auto& [name, value] : weights) {
            auto ends = [&](const std::string& tail) {
                return name.size() >= tail.size() && name.compare(name.size()-tail.size(), tail.size(), tail) == 0;
            };
            if (ends("norm.weight") || ends("_clip.scale") || ends("sconv.weight") || ends("sconv.static_w"))
                value = f32(value);
        }
        for (int l = 0; l < L; ++l) {
            const auto p = prefix(l);
            full.push_back(weights.count(p + "attn_norm.weight") == 0);
            experts.push_back({expert(p + "mlp.experts.w13_weight"), expert(p + "mlp.experts.w2_weight")});
            // The router and message-gate outputs accumulate in F32 in the reference.
            weights.at(p + "mlp.gate.weight") = f32(w(p + "mlp.gate.weight"));
            if (l >= core0 && l < core1) {
                const int j = l - core0;
                message_queries.push_back(mx::stack({row(w("middle_cycle_cross_pass_queries"), 2*j),
                    row(w("middle_cycle_cross_pass_kv_queries"), 2*j), row(w("middle_cycle_cross_pass_kv_queries"), 2*j+1),
                    row(w("middle_cycle_cross_pass_queries"), 2*j+1)},0));
                for (const char* c : {"attn", "k", "v", "mlp"})
                    for (const char* k : {"forget_weight", "gain_weight"}) {
                        const auto name = p + "controllers." + c + "." + k;
                        weights.at(name) = f32(w(name));
                    }
            }
        }
        if (!exact) {
            std::vector<std::string> names{"unembed.weight"};
            for (int l = 0; l < L; ++l)
                for (const char* m : {"attn.wq_du.weight", "attn.wk_dv.weight", "attn.wv_dv.weight", "attn.wo_ud.weight", "attn.gate_attn.weight"})
                    names.push_back(prefix(l) + m);
            for (const auto& name : names) {
                auto q = mx::quantize(w(name), kQ8Group, 8);
                mx::eval(q);
                q8.emplace(name, std::move(q));
                // The BF16 LM head stays for the drafter's fold (built from it at enable_draft); the rest is replaced.
                if (name != "unembed.weight") weights.erase(name);
            }
        }
        build_head_bound();  // both modes: --exact off's greedy tokens also come from the BF16 head
        for (int l : schedule()) visits.emplace_back(context, full[l] ? 0 : window);
        for(const auto& [name,value]:weights) mx::eval(value);
        for(const auto& q:message_queries) mx::eval(q);
        for(const auto& layer:experts) for(const auto& e:layer) {
            mx::eval({e.codes,e.scales});if(e.global) mx::eval(*e.global);
        }
        std::fprintf(stderr, "inkling2 MLX: original NVFP4/MXFP8, %zu text bytes, %zu independent visits\n", bytes, visits.size());
    }
    // bound_r = |w_r - q_r| (1 + 1e-4) + 3e-4 |w_r|: the 6-bit row error norm, then a worst-case allowance for F32
    // accumulation over D terms in both dot products (D x 2^-24 x 2 < 3e-4), so |x . w_r - x . q_r| <= |x| bound_r.
    void build_head_bound() {
        const auto& head = w("unembed.weight");
        if (head.dtype() != mx::bfloat16 || head.shape(1) % 256) return;
        const int rows = head.shape(0), chunk = 16384;
        std::array<std::vector<mx::array>, 3> parts;
        std::vector<mx::array> bounds;
        for (int r0 = 0; r0 < rows; r0 += chunk) {
            auto wf = f32(mx::slice(head, {r0, 0}, {std::min(rows, r0 + chunk), D}));
            auto q = mx::quantize(wf, kHeadGroup, kHeadBits);
            auto error = mx::subtract(wf, mx::dequantize(q[0], q[1], q[2], kHeadGroup, kHeadBits));
            auto b = mx::add(mx::multiply(mx::sqrt(mx::sum(mx::square(error), -1)), mx::array(1.0f + 1e-4f)),
                             mx::multiply(mx::sqrt(mx::sum(mx::square(wf), -1)), mx::array(3e-4f)));
            mx::eval({q[0], q[1], q[2], b});
            for (int i = 0; i < 3; ++i) parts[i].push_back(q[i]);
            bounds.push_back(b);
        }
        for (auto& p : parts) { head_approx.push_back(mx::concatenate(p, 0)); mx::eval(head_approx.back()); }
        head_bound = mx::concatenate(bounds, 0);
        mx::eval(*head_bound);
    }
    // The BF16 head's greedy token for each row of x [n, D]: [n] uint32 (see head_approx).
    mx::array head_argmax(const mx::array& x) const {
        auto xf = f32(x);
        auto approx = mx::quantized_matmul(xf, head_approx[0], head_approx[1], head_approx[2], true, kHeadGroup, kHeadBits);
        auto best = mx::argmax(mx::slice(approx, {0, 0}, {approx.shape(0), V}), -1);
        auto norm = mx::sqrt(mx::sum(mx::square(xf), -1));
        return mx::argmax(inkling2_head_select(x, w("unembed.weight"), approx, *head_bound, norm, best, V, logit_cap), -1);
    }
    static std::string prefix(int layer) { return "layers." + std::to_string(layer) + "."; }
    const mx::array& w(const std::string& name) const { return weights.at(name); }
    std::vector<int> schedule() const {
        std::vector<int> layers;
        for (int l=0; l<core0; ++l) layers.push_back(l);
        for (int pass=0; pass<passes; ++pass) for (int l=core0; l<core1; ++l) layers.push_back(l);
        for (int l=core1; l<L; ++l) layers.push_back(l);
        return layers;
    }
    FloatxExperts expert(const std::string& name) const {
        const bool nv = weights.count(name + "_amax") != 0;
        std::optional<mx::array> global;
        if (nv) global = mx::divide(mx::multiply(w(name + "_amax"), mx::array(2688.f)), w(name + "_outer_scale_denominator"));
        return FloatxExperts(w(name), w(name + "_scale"), global, nv ? 4 : 8);
    }
    mx::array project(const mx::array& x, const std::string& name) const {
        const auto it = q8.find(name);
        if (it == q8.end()) return linear(x, w(name));
        const auto& q = it->second;
        return mx::quantized_matmul(bf16(x), q[0], q[1], q[2], true, kQ8Group, 8);
    }
    mx::array gate(const mx::array& h, const mx::array& message, const std::string& p) const {
        auto both = mx::concatenate({norm(h), norm(message)}, -1);
        auto low = f32(project(both, p + "down_weight"));
        auto forget = mx::tanh(project(low, p + "forget_weight"));
        auto gain = mx::tanh(project(low, p + "gain_weight"));
        return mx::add(mx::multiply(mx::subtract(mx::array(1.f), forget), f32(h)), mx::multiply(gain, message));
    }
    // The gate followed by its RMSNorm, BF16 out. Prompt chunks: the inputs normalised straight to BF16 (the down
    // product reads them in BF16 anyway), forget and gain as one product, merge and norm in one kernel
    // (inkling2_gate_merge); a few rows keep gate() and norm_bf16.
    mx::array gate_normed(const mx::array& h, const mx::array& message, const std::string& p, const mx::array& weight) const {
        if (h.shape(0) <= 8 || w(p + "down_weight").dtype() != mx::bfloat16) return norm_bf16(gate(h, message, p), weight);
        auto low = f32(project(mx::concatenate({norm_bf16(h), norm_bf16(message)}, -1), p + "down_weight"));
        auto fg = linear(low, mx::concatenate({w(p + "forget_weight"), w(p + "gain_weight")}, 0));
        return inkling2_gate_merge(fg, h, message, weight);
    }
    // Pure causal convolution over [3 history rows; input]; returns (output, next history).
    // dynamic: the input's projection through <p>.weight_proj ([n, 4*G]) for the dynamic-tap (full attention) form.
    std::pair<mx::array, mx::array> convolution(const mx::array& input, const std::string& p, const mx::array& old,
                          const std::optional<mx::array>& dynamic = std::nullopt) const {
        const int n = input.shape(0), c = input.shape(1);
        if (!dynamic) {
            auto taps=mx::reshape(f32(w(p+".weight")),{c,4});
            auto r = inkling2_conv(input, std::nullopt, std::nullopt, old, taps);  // one kernel
            return {r[0], r[1]};
        }
        auto x = f32(input);
        auto values = mx::concatenate({old, x},0);
        auto dyn = mx::reshape(f32(*dynamic), {n, G, 1, 4});
        const auto stat = mx::reshape(mx::transpose(f32(w(p + ".static_w"))), {G, HD, 4});
        auto taps = mx::add(stat, dyn);
        std::optional<mx::array> sum;
        for (int lag=0; lag<4; ++lag) {
            auto v = mx::reshape(mx::slice(values, {3-lag, 0}, {3-lag+n, c}), {n, G, HD});
            auto term = mx::reshape(mx::multiply(v, row(taps, lag, 3)), {n, c});
            sum = sum ? mx::add(*sum, term) : term;
        }
        return {bf16(mx::divide(mx::add(*sum, x), mx::array(3.f))), mx::slice(values, {n, 0}, {n+3, c})};
    }
    mx::array state_or_zeros(const std::optional<mx::array>& state, int c) const {
        return state.value_or(mx::zeros({3, c}, mx::float32));
    }
    // A visit is three pure pieces around the KV-cache attention. Decode compiles each piece per layer;
    // prefill calls them directly. Inputs and outputs are flat array lists (mx::compile's contract).
    //   pre:  [h, position, conv_k, conv_v, (bank_block, bank_partial), blocks...]
    //      -> [incoming, query, keys, values, bias, gate, conv_k, conv_v, k_in, v_in, sinks, (message)]
    //   post: [attention, gate, h, conv_attn] -> [residual, conv_attn, attn_in]
    //   mlp:  [residual, conv_mlp, (message), blocks...] -> [result, conv_mlp, mlp_in]
    // (*_in: the convolution inputs, kept for a verify block's rollback.)
    // bias [n, H, extent]: the relative bias by distance; sinks [n, H]: the sink logits (see inkling2_mask).
    std::vector<mx::array> pre(int l, bool messaged, const std::vector<mx::array>& in) const {
        const auto p = prefix(l);
        const auto& h = in[0];
        const int first_block = messaged ? 6 : 4;
        std::vector<mx::array> blocks(in.begin()+first_block, in.end());
        std::optional<mx::array> message;
        if (messaged) message = depth({in[4]}, in[5], message_queries[l-core0]);
        // A sliding layer without a message feeds q, k and v the same normalised row.
        const bool shared = !full[l] && !message;
        const char* names[3] = {"q", "k", "v"};
        const char* controls[3] = {"attn", "k", "v"};
        auto input_norm = [&](int j) { return w(p+(full[l] ? std::string(names[j])+"_input_norm.weight" : "attn_norm.weight")); };
        std::optional<mx::array> incoming, a, ready;
        if (l) {
            const auto& query = w("attn_res.attn_queries." + std::to_string(l-1));
            std::vector<mx::array> weights_;  // without a message, the depth kernel also writes the normalised inputs
            if (!message) for (int j = 0; j < int(mx::reshape(query, {-1, D}).shape(0)); ++j) weights_.push_back(input_norm(full[l] ? j : 0));
            auto r = inkling2_depth_normed(blocks, h, query, weights_);
            a = r[0]; incoming = r[1];
            if (!message) ready = r[2];
        } else {
            incoming = norm(h); a = mx::expand_dims(f32(h), 1);
        }
        std::vector<mx::array> inputs;
        for (int j=0; j<(shared ? 1 : 3); ++j) {
            const int at = full[l] ? j : 0;
            if (ready) { inputs.push_back(row(*ready, at, 1)); continue; }
            inputs.push_back(row(*a, at, 1));
            inputs[j] = message ? gate_normed(inputs[j], row(*message,j,1), p+"controllers."+controls[j]+".", input_norm(j))
                                : norm_bf16(inputs[j], input_norm(j));
        }
        if (shared) { inputs.push_back(inputs[0]); inputs.push_back(inputs[0]); }
        const auto a_ = p + "attn.";
        const int n = h.shape(0);
        std::vector<mx::array> qs;  // q, rel, sink, gate
        for (const char* name : {"wq_du.weight", "wr_du.weight", "attn_sink_proj.weight", "gate_attn.weight"})
            qs.push_back(project(inputs[0], a_ + name));
        auto kr = project(inputs[1], a_ + "wk_dv.weight"), vr = project(inputs[2], a_ + "wv_dv.weight");
        std::optional<mx::array> kdyn, vdyn;
        if (full[l]) { kdyn = project(inputs[1], a_ + "k_sconv.weight_proj"); vdyn = project(inputs[2], a_ + "v_sconv.weight_proj"); }
        // full layers: the log-position scale tau of each row, applied to the clipped q and rel by the clip kernel
        std::optional<mx::array> tau;
        if (full[l]) {
            auto positions = mx::add(mx::arange(0, n, mx::float32), in[1]);
            tau = mx::add(mx::array(1.f), mx::multiply(mx::array(logalpha),
                mx::log(mx::maximum(mx::divide(mx::add(positions, mx::array(1.f)), mx::array(logfloor)), mx::array(1.f)))));
        }
        auto q = clip(mx::reshape(qs[0], {n, H, HD}), qclip, w(a_ + "q_l2_clip.scale"), tau);
        auto rel = clip(mx::reshape(qs[1], {n, H, DR}), rclip, w(a_ + "rel_l2_clip.scale"), tau);
        mx::array k = kr, v = vr, conv_k = in[2], conv_v = in[3];
        if (n > 8) {
            // prompt chunks: convolution + clip (keys) or norm (values) in one kernel a (row, head)
            auto taps = [&](const char* conv) {
                return full[l] ? mx::transpose(w(a_ + conv + ".static_w")) : mx::reshape(w(a_ + conv + ".weight"), {G * HD, 4});
            };
            std::tie(k, conv_k) = inkling2_kv_prep(kr, in[2], taps("k_sconv"), kdyn, true, qclip);
            std::tie(v, conv_v) = inkling2_kv_prep(vr, in[3], taps("v_sconv"), vdyn, false, 0.f);
            k = mx::reshape(k, {n, G, HD}); v = mx::reshape(v, {n, G, HD});
        } else {
            std::tie(k, conv_k) = convolution(kr, a_ + "k_sconv", in[2], kdyn);
            std::tie(v, conv_v) = convolution(vr, a_ + "v_sconv", in[3], vdyn);
            k = clip(mx::reshape(k, {n, G, HD}), qclip);
            v = norm_bf16(mx::reshape(v, {n, G, HD}));
        }
        auto bias = linear(rel, mx::transpose(w(a_ + "rel_logits_proj.proj")));
        auto keys = mx::expand_dims(mx::transpose(k, {1, 0, 2}), 0);
        auto vals = mx::expand_dims(mx::transpose(v, {1, 0, 2}), 0);
        auto query = mx::expand_dims(mx::transpose(q,{1,0,2}),0);
        std::vector<mx::array> out{*incoming, query, keys, vals, bias, qs[3], conv_k, conv_v, kr, vr, qs[2]};
        if (message) out.push_back(*message);
        return out;
    }
    // Uncompiled: the KV cache grows every step, so these shapes change.
    mx::array attend(int l, Visit& visit, const mx::array& query, const mx::array& keys, const mx::array& vals,
                     const mx::array& bias, const mx::array& sinks) {
        const int n = query.shape(2), base = visit.kv.position();
        auto [kc, vc] = visit.kv.append(keys, vals);
        const int s = kc.shape(2) - 1, start = visit.kv.start();  // last row: zero sink key
        const float scale = 1.f / std::sqrt(float(HD));
        // Prompt chunks: the matrix-unit attention with the bias and mask in place, each 64-row block over only the keys
        // it can see (its sliding window or causal prefix), the cache read in place (inkling2_prompt_attention).
        if (n > 8) {
            auto out = inkling2_prompt_attention(query, kc, vc, bias, sinks, base, start, s, full[l] ? 0 : window, scale);
            if (!verifying) visit.kv.retain_window();
            return out;
        }
        auto mask = inkling2_mask(bias, sinks, base, start, s, full[l] ? 0 : window);  // [1, H, n, s + 1]
        // A verify block's few rows as heads of one single-query call: row r of head h becomes head h * n + r, which
        // keeps the GQA grouping (query heads h of a key head stay together), so MLX runs its split-key vector kernel
        // instead of the multi-row tiles (64K verify of 6 rows: ~17 ms per full-attention visit).
        const bool fold = n > 1 && n <= 8;
        auto out = fold ? mx::reshape(inkling2_sink_attention(mx::reshape(query, {1, H * n, 1, HD}), kc, vc,
                                                              mx::reshape(mask, {1, H * n, 1, s + 1}), scale),
                                      {1, H, n, HD})
                        : inkling2_sink_attention(query,kc,vc,mask,scale);
        if (!verifying) visit.kv.retain_window();  // a verify block trims in commit(), after a rollback
        return out;
    }
    std::vector<mx::array> post(int l, const std::vector<mx::array>& in) const {
        const auto p = prefix(l);
        const auto& h = in[2];
        const int n = h.shape(0);
        auto out = mx::reshape(mx::transpose(mx::reshape(in[0], {H,n,HD}), {1,0,2}), {n,H*HD});
        out = bf16(mx::multiply(f32(out), mx::sigmoid(f32(in[1]))));
        auto projected = project(out, p + "attn.wo_ud.weight");
        // the convolution with the residual add (none at a block's first layer), F32 out
        auto r = inkling2_conv(projected, std::nullopt, std::nullopt, in[3], mx::reshape(w(p+"attn_sconv.weight"), {D, 4}),
                               l % 6 == 0 ? std::nullopt : std::optional<mx::array>(h), mx::float32);
        return {r[0], r[1], projected};
    }
    std::vector<mx::array> mlp(int l, bool messaged, const std::vector<mx::array>& in) const {
        const auto p = prefix(l);
        const int first_block = messaged ? 3 : 2;
        std::vector<mx::array> blocks(in.begin()+first_block, in.end());
        const auto& residual = in[0];
        const auto& query = w("attn_res.mlp_queries."+std::to_string(l));
        mx::array m = [&] {
            if (messaged) return gate_normed(row(depth(blocks,residual,query),0,1),row(in[2],3,1),p+"controllers.mlp.",w(p+"mlp_norm.weight"));
            const int Q = mx::reshape(query, {-1, D}).shape(0);  // the depth kernel writes the normalised row
            return row(inkling2_depth_normed(blocks,residual,query,std::vector<mx::array>(size_t(Q), w(p+"mlp_norm.weight")))[2],0,1);
        }();
        auto [experts, routes, restore] = moe(m,l);
        const int c = experts.shape(-1);
        auto r = inkling2_conv(std::nullopt, experts, routes, in[1], mx::reshape(w(p+"mlp_sconv.weight"), {c, 4}), residual,
                               mx::bfloat16, restore);  // the combine, the convolution and the residual add
        return {r[0], r[1], r[2]};
    }
    // The routed experts: {each token's K expert outputs [n, K, D], their route weights [n, K]}; the weighted sum is
    // done by the MLP convolution's kernel (inkling2_conv with the experts).
    // {expert outputs, route weights, restore}: outputs [n, K, D] in route order, or (restore given) the grouped rows
    // [n K, 1, D] with pair (t, k) at restore[t K + k], which the combine reads in place.
    std::tuple<mx::array, mx::array, std::optional<mx::array>> moe(const mx::array& x, int l) const {
        const auto p = prefix(l);
        // the router: logits as a matvec, then sigmoid, top-k and route weights in one kernel (inkling2_route)
        auto [ids, routes] = inkling2_route(project(f32(x), p + "mlp.gate.weight"), w(p + "mlp.gate.bias"), topk, route_scale);
        const int n = x.shape(0);
        // A chunk of tokens runs grouped by expert; one token keeps the per-row matvec path.
        if (verifying && n > 1 && n <= 8 && experts[l][0].nax_codes) {
            // Verify block on the matrix unit: rows grouped by expert, each expert's weights read once for its rows.
            // pairs sorted by expert in one kernel (inkling2_pair_sort): sorted ids, token rows, restore order
            auto sorting = inkling2_pair_sort(ids);
            auto sorted = mx::reshape(mx::slice(sorting, {0, 0}, {1, n * topk}), {n * topk});
            auto token = mx::reshape(mx::slice(sorting, {1, 0}, {2, n * topk}), {n * topk});
            auto restore = mx::reshape(mx::slice(sorting, {2, 0}, {3, n * topk}), {n * topk});
            auto gu = experts[l][0].rows(x, token, sorted);  // [n*topk, 2I]
            auto g = f32(mx::slice(gu, {0, 0}, {n * topk, 2 * I}, {1, 2}));
            auto u = f32(mx::slice(gu, {0, 1}, {n * topk, 2 * I}, {1, 2}));
            auto act = mx::multiply(mx::multiply(mx::tanh(mx::divide(g, mx::array(gate_cap))), mx::array(gate_cap)), mx::sigmoid(g));
            act = bf16(mx::multiply(act, mx::multiply(mx::tanh(mx::divide(u, mx::array(up_cap))), mx::array(up_cap))));
            // the down product in sorted order; the combine reads pair (t, k) at restore[t K + k]
            auto y = experts[l][1].rows(act, mx::arange(n * topk, mx::uint32), sorted);  // [n*topk, D]
            return {mx::reshape(y, {n * topk, 1, D}), routes, restore};
        }
        if (verifying && n > 1 && n <= 8) {
            // Verify block on gather_qmm (no matrix-unit copy): the same one-kernel pair sort, the products grouped by
            // expert, the combine reading the sorted rows through the restore order.
            auto sorting = inkling2_pair_sort(ids);
            const int P = n * topk;
            auto sorted = mx::reshape(mx::slice(sorting, {0, 0}, {1, P}), {P});
            auto token = mx::reshape(mx::slice(sorting, {1, 0}, {2, P}), {P});
            auto restore = mx::reshape(mx::slice(sorting, {2, 0}, {3, P}), {P});
            auto gu = experts[l][0](mx::expand_dims(mx::take(x, token, 0), 1), sorted, true);  // [P, 1, 2I]
            const mx::Shape stop{P, 1, 2 * I};
            auto g = f32(mx::slice(gu, {0, 0, 0}, stop, {1, 1, 2}));
            auto u = f32(mx::slice(gu, {0, 0, 1}, stop, {1, 1, 2}));
            auto act = mx::multiply(mx::multiply(mx::tanh(mx::divide(g, mx::array(gate_cap))), mx::array(gate_cap)), mx::sigmoid(g));
            act = mx::multiply(act, mx::multiply(mx::tanh(mx::divide(u, mx::array(up_cap))), mx::array(up_cap)));
            return {experts[l][1](bf16(act), sorted, true), routes, restore};
        }
        std::optional<ExpertGroups> groups;
        if (n * topk >= 64) groups.emplace(ids);  // a prompt chunk reads each expert once for its rows
        // gu is [n*topk, 1, 2I] grouped, [n, topk, 2I] per row; gate and up interleave on the last axis.
        auto gu = groups ? experts[l][0](groups->gather(x, topk), groups->ids, true)
                         : mx::squeeze(experts[l][0](mx::reshape(x, {n,1,1,D}), ids), 2);
        const mx::Shape stop{gu.shape(0), gu.shape(1), 2*I};
        auto g = f32(mx::slice(gu, {0,0,0}, stop, {1,1,2}));
        auto u = f32(mx::slice(gu, {0,0,1}, stop, {1,1,2}));
        auto act = mx::multiply(mx::multiply(mx::tanh(mx::divide(g, mx::array(gate_cap))), mx::array(gate_cap)), mx::sigmoid(g));
        act = mx::multiply(act, mx::multiply(mx::tanh(mx::divide(u, mx::array(up_cap))), mx::array(up_cap)));
        if (groups) return {experts[l][1](bf16(act), groups->ids, true), routes, groups->restore};
        return {mx::squeeze(experts[l][1](mx::expand_dims(bf16(act), 2), ids), 2), routes, std::nullopt};
    }
    using Piece = std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
    // Decode runs each piece through mx::compile (fuses the elementwise chains); one entry per layer and message use.
    std::map<int, std::array<Piece,3>> compiled;
    const std::array<Piece,3>& pieces(int l, bool messaged) {
        const int key = 2*l + messaged;
        auto it = compiled.find(key);
        if (it != compiled.end()) return it->second;
        std::array<Piece,3> made{
            mx::compile(Piece([this,l,messaged](const std::vector<mx::array>& in) { return pre(l,messaged,in); })),
            mx::compile(Piece([this,l](const std::vector<mx::array>& in) { return post(l,in); })),
            mx::compile(Piece([this,l,messaged](const std::vector<mx::array>& in) { return mlp(l,messaged,in); }))};
        return compiled.emplace(key, std::move(made)).first->second;
    }
    mx::array visit_layer(const mx::array& h, std::vector<mx::array>& blocks, int l, int visit_index,
                           const std::optional<std::pair<mx::array, mx::array>>& bank) {
        const char* phase="input";
        try {
        using Clock=std::chrono::steady_clock;
        auto start=phase_ms?Clock::now():Clock::time_point{};
        auto checkpoint=[&](const mx::array& value,int index) {
            if(!phase_ms) return;
            mx::eval(value);mx::synchronize();auto now=Clock::now();
            (*phase_ms)[index]+=std::chrono::duration<double,std::milli>(now-start).count();start=now;
        };
        auto& state = visits[visit_index];
        const bool messaged = bank.has_value();
        const Piece direct_pre = [&](const std::vector<mx::array>& in) { return pre(l,messaged,in); };
        const Piece direct_post = [&](const std::vector<mx::array>& in) { return post(l,in); };
        const Piece direct_mlp = [&](const std::vector<mx::array>& in) { return mlp(l,messaged,in); };
        const auto* fused = &pieces(l, messaged);  // compiled for every row count (fuses the elementwise chains)
        const int kv_width = G*HD;
        std::vector<mx::array> in{h, mx::array(float(state.kv.position())),
            state_or_zeros(state.conv[0], kv_width), state_or_zeros(state.conv[1], kv_width)};
        if (bank) { in.push_back(bank->first); in.push_back(bank->second); }
        in.insert(in.end(), blocks.begin(), blocks.end());
        auto a = fused ? (*fused)[0](in) : direct_pre(in);
        if (verifying) { state.before[0] = in[2]; state.before[1] = in[3]; state.input[0] = a[8]; state.input[1] = a[9]; }
        state.conv[0] = a[6]; state.conv[1] = a[7];
        if (phase_ms) mx::eval(a);  // every pre output in the pre phase (else keys, values and bias land in the next)
        checkpoint(a[1],0);
        phase="attention";
        auto attention = attend(l, state, a[1], a[2], a[3], a[4], a[10]);
        std::vector<mx::array> post_in{attention, a[5], h, state_or_zeros(state.conv[2], D)};
        auto b = fused ? (*fused)[1](post_in) : direct_post(post_in);
        if (verifying) { state.before[2] = post_in[3]; state.input[2] = b[2]; }
        state.conv[2] = b[1];
        checkpoint(b[0],1);
        if (l % 6 == 0) blocks.push_back(a[0]);
        phase="experts";
        std::vector<mx::array> mlp_in{b[0], state_or_zeros(state.conv[3], D)};
        if (messaged) mlp_in.push_back(a[11]);
        mlp_in.insert(mlp_in.end(), blocks.begin(), blocks.end());
        auto c = fused ? (*fused)[2](mlp_in) : direct_mlp(mlp_in);
        if (verifying) { state.before[3] = mlp_in[1]; state.input[3] = c[2]; }
        state.conv[3] = c[1];
        checkpoint(c[0],3);
        auto result = c[0];
        // Decode (one token): keep the whole step as one graph and evaluate it once in forward().
        if (!lazy) {
            mx::eval(result);
            state.kv.settle();
            for (const auto& s : state.conv) if (s) mx::eval(*s);
        }
        checkpoint(result,4);
        return result;
        } catch (const std::exception& e) {
            throw std::runtime_error("inkling2 layer "+std::to_string(l)+" visit "+std::to_string(visit_index)+" "+phase+": "+e.what());
        }
    }
    // Builds one forward over ids [n] and returns its (lazy) logits. With `lazy` set nothing is evaluated here:
    // the caller evaluates the logits together with state_arrays().
    mx::array build(const mx::array& ids, bool all_logits, std::vector<mx::array>* trace) {
        auto x0 = norm_bf16(mx::take(w("embed.weight"),ids,0),w("embed_norm.weight"));
        auto h = x0;
        std::vector<mx::array> blocks;
        int visit = 0;
        // The drafter reads the state entering visits 5, 23 and 41 and the final normalised state of every row.
        std::vector<mx::array> tapped;
        auto tap = [&] { if (draft && (visit == 5 || visit == 23 || visit == 41)) tapped.push_back(h); };
        if(trace) { trace->clear();trace->push_back(f32(h)); }
        for (int l=0;l<core0;++l) {
            tap();
            h=visit_layer(h,blocks,l,visit++,std::nullopt); stream_ahead(h, visit);
            if(trace) trace->push_back(f32(h));
        }
        const auto prelude=h;
        const auto prelude_blocks=blocks;
        std::optional<mx::array> bank_block,bank_partial;
        for (int pass=0;pass<passes;++pass) {
            h=prelude; blocks=prelude_blocks;
            // The message bank (block 2 and the final state of the previous pass) is shared by the pass's visits.
            std::optional<std::pair<mx::array, mx::array>> bank;
            if (pass) bank.emplace(*bank_block, *bank_partial);
            for (int l=core0;l<core1;++l) {
                tap();
                h=visit_layer(h,blocks,l,visit++,bank); stream_ahead(h, visit);
                if(trace) trace->push_back(f32(h));
            }
            bank_block=blocks.at(2); bank_partial=h;
        }
        for (int l=core1;l<L;++l) {
            h=visit_layer(h,blocks,l,visit++,std::nullopt); stream_ahead(h, visit);
            if(trace) trace->push_back(f32(h));
        }
        const auto& oq = w("attn_res.output_query");
        auto final=row(inkling2_depth_normed(blocks,h,oq,std::vector<mx::array>(size_t(mx::reshape(oq,{-1,D}).shape(0)),w("norm.weight")))[2],0,1);
        if(trace) trace->back()=f32(final);
        if (draft) { tapped.push_back(final); taps = mx::stack(tapped, 1); }
        if (!all_logits) final=mx::slice(final,{final.shape(0)-1,0},{final.shape(0),D});
        head_input=final;
        auto logits=f32(project(final,"unembed.weight"));
        logits=mx::slice(logits,{0,0},{logits.shape(0),V});
        logits=mx::multiply(mx::tanh(mx::divide(logits,mx::array(logit_cap))),mx::array(logit_cap));
        return logits;
    }
    // A verify block is evaluated synchronously (its accepted count shapes the next round), so the GPU would wait for
    // the whole graph to be built: every 12 visits it gets the visits built so far, and runs them while the rest is
    // built (~1 % of a speculative step). Decode steps are pipelined instead (greedy_step), prompt chunks are long.
    void stream_ahead(const mx::array& h, int visit) const {
        if (verifying && lazy && visit % 12 == 0 && visit < int(visits.size())) mx::async_eval(h);
    }
    // Every visit's KV and convolution state, so a lazily built step is evaluated together with its logits.
    std::vector<mx::array> state_arrays() const {
        std::vector<mx::array> out;
        for (const auto& visit : visits) {
            visit.kv.collect(out);
            for (const auto& c : visit.conv) if (c) out.push_back(*c);
        }
        return out;
    }
    mx::array forward(std::span<const uint32_t> tokens, bool all_logits, std::vector<mx::array>* trace) {
        if (tokens.empty() || tokens.size() > size_t(context-length)) throw std::invalid_argument("inkling2 MLX token count/context");
        for (auto t : tokens) if (t >= uint32_t(w("embed.weight").shape(0))) throw std::invalid_argument("inkling2 MLX token outside vocabulary");
        // A decode step or a prompt chunk is one graph, evaluated once with all state (a chunk evaluated visit by visit
        // leaves the GPU idle while each next visit is built: ~4 % of prefill). Traced or timed runs evaluate each visit.
        lazy = !phase_ms && !trace;
        auto logits = build(mx::array(tokens.data(), {int(tokens.size())}, mx::uint32), all_logits, trace);
        if (draft) draft->commit(*taps, length);  // every token the target consumes becomes draft context
        auto pending = lazy ? state_arrays() : std::vector<mx::array>{};
        if (draft) for (auto& a : draft->state()) pending.push_back(a);
        pending.push_back(logits);
        mx::eval(pending);
        length+=int(tokens.size());
        return logits;
    }
    mx::array greedy_step(const mx::array& token) {
        if (token.size() != 1 || length >= context) throw std::invalid_argument("inkling2 MLX greedy step token/context");
        lazy = true;
        auto logits = build(mx::reshape(mx::astype(token, mx::uint32), {1}), false, nullptr);
        // exact mode: the bounded 8-bit head picks the same token without reading the whole BF16 head (logits unused)
        auto next = head_bound ? head_argmax(*head_input) : mx::argmax(logits, -1);
        if (draft) draft->commit(*taps, length);
        auto pending = state_arrays();
        if (draft) for (auto& a : draft->state()) pending.push_back(a);
        pending.push_back(next);
        mx::async_eval(pending);
        ++length;
        return next;
    }
    // DSpark speculation (enable_draft): draft `depth` tokens after `anchor`, verify [anchor, drafts] in one target
    // forward, keep the longest prefix the target agrees with and roll the rest back. One host sync per step.
    std::unique_ptr<Inkling2MlxDraft> draft;
    std::optional<mx::array> taps;  // [n, 4, D] of the last build() when drafting
    void enable_draft(const std::string& folder) {
        if (length) throw std::runtime_error("inkling2 MLX: enable the drafter before the first forward");
        draft = std::make_unique<Inkling2MlxDraft>(folder, context, w("unembed.weight"), V, window,
            [this](const mx::array& x) {  // drafts only propose (the verify decides): the 6-bit head copy when built
                if (head_approx.empty()) return project(x, "unembed.weight");
                return mx::quantized_matmul(f32(x), head_approx[0], head_approx[1], head_approx[2], true, kHeadGroup, kHeadBits);
            });
        // Verify blocks run the NVFP4 experts on the matrix unit from a second, fragment-ordered copy (about 0.57 GB
        // per layer); MXFP8 layers keep gather_qmm. Only when it fits: resident weights + the copy + this context's
        // full-attention and drafter K / V within 70 % of unified memory (else verify keeps gather_qmm; a long context
        // that swaps loses far more than the copy gains: 256K 1.3 tok/s with it).
        size_t copy = 0;
        for (const auto& layer : experts)
            if (layer[0].bits == 4 && layer[1].bits == 4)
                for (const auto& e : layer) copy += e.codes.nbytes() + e.scales.nbytes();
        int full_visits = 0;
        for (int l : schedule()) full_visits += full[l];
        const size_t kv = size_t(context) * size_t(G) * HD * 2 * 2 * size_t(full_visits + 1);  // + the drafter's global layer
        uint64_t memory = 0;
        size_t size = sizeof(memory);
        sysctlbyname("hw.memsize", &memory, &size, nullptr, 0);
        const double need = double(mx::get_active_memory()) + double(copy) + double(kv);
        if (memory && need > 0.7 * double(memory)) {
            std::fprintf(stderr, "inkling2 MLX: verify experts stay on gather_qmm (matrix-unit copy %.1f GB + K/V %.1f GB would pass "
                         "70 %% of %.0f GB)\n", copy / 1e9, kv / 1e9, memory / 1e9);
            return;
        }
        for (auto& layer : experts)
            if (layer[0].bits == 4 && layer[1].bits == 4) for (auto& e : layer) e.pack_nax();
    }
    // Depth 0: the drafts per step that maximise expected tokens per millisecond, from this sequence's own record:
    // the acceptance of each draft position (given the ones before it were accepted) and the step time per depth.
    // Both decay (half-life ~35 steps) so the choice follows the text; neighbours are tried every 16th step.
    struct DepthPolicy {
        static constexpr int kMax = Inkling2MlxDraft::kBlock - 1;
        std::array<double, kMax> reached{}, kept{};    // position i: steps that drafted and reached it, and kept it
        std::array<double, kMax + 1> ms{}, steps{};     // per depth: decayed step time sum and step count
        int last = 3, count = 0;
        double expected(int d) const {  // 1 (the target's own token) + the chance each draft prefix survives
            double e = 1, survive = 1;
            for (int i = 0; i < d; ++i) { survive *= (kept[size_t(i)] + 1) / (reached[size_t(i)] + 2); e += survive; }
            return e;
        }
        double cost(int d) const {  // measured, else a line through the measured depths, else +6 % per draft
            if (steps[size_t(d)] >= 2) return ms[size_t(d)] / steps[size_t(d)];
            double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
            for (int k = 1; k <= kMax; ++k)
                if (steps[size_t(k)] >= 2) { const double y = ms[size_t(k)] / steps[size_t(k)]; sx += k; sy += y; sxx += k * k; sxy += k * y; ++n; }
            if (n >= 2 && sxx * n != sx * sx) {
                const double b = (sxy * n - sx * sy) / (sxx * n - sx * sx);
                return std::max(1e-3, (sy - b * sx) / n + b * d);
            }
            if (n == 1) { const double k = sx, y = sy; return y * (1 + 0.06 * (d - k)); }
            return 1 + 0.06 * d;
        }
        int choose(int limit) {
            limit = std::max(1, std::min(limit, kMax));
            int best = 1;
            for (int d = 1; d <= limit; ++d) if (expected(d) / cost(d) > expected(best) / cost(best)) best = d;
            if (++count % 16 == 0) best = std::max(1, std::min(limit, best + ((count / 16) % 2 ? 1 : -1)));
            return last = best;
        }
        void record(int d, int accepted, double step_ms) {
            constexpr double decay = 0.98;
            for (auto& v : reached) v *= decay;
            for (auto& v : kept) v *= decay;
            for (auto& v : ms) v *= decay;
            for (auto& v : steps) v *= decay;
            for (int i = 0; i < d && i <= accepted; ++i) { reached[size_t(i)] += 1; kept[size_t(i)] += i < accepted; }
            ms[size_t(d)] += step_ms; steps[size_t(d)] += 1;
        }
    } policy;
    Inkling2Mlx::Speculation speculate(uint32_t anchor, int depth) {
        if (!draft) throw std::runtime_error("inkling2 MLX: speculate needs enable_draft()");
        const bool adaptive = depth == 0;
        const auto started = std::chrono::steady_clock::now();
        if (adaptive) depth = policy.choose(context - length - 1);
        depth = std::min({depth, Inkling2MlxDraft::kBlock - 1, context - length - 1});
        if (depth < 1 || anchor >= uint32_t(V)) throw std::invalid_argument("inkling2 MLX speculate depth/anchor/context");
        const int base = length, n = depth + 1;
        auto first = mx::array(&anchor, {1}, mx::uint32);
        auto x0 = norm_bf16(mx::take(w("embed.weight"), first, 0), w("embed_norm.weight"));
        auto drafts = draft->propose(x0, first, base, depth);
        mx::async_eval(drafts);  // the GPU drafts while the verify graph is built (~6 % of a speculative step)
        lazy = true; verifying = true;
        mx::array best = mx::array(0);
        try {
            auto logits = build(mx::concatenate({first, drafts}, 0), true, nullptr);
            best = mx::astype(head_bound ? head_argmax(*head_input) : mx::argmax(logits, -1), mx::uint32);
            auto pending = state_arrays();
            for (auto& a : draft->state()) pending.push_back(a);
            pending.push_back(drafts); pending.push_back(best); pending.push_back(*taps);
            mx::eval(pending);
        } catch (...) { verifying = false; throw; }
        verifying = false;
        const auto* d = drafts.data<uint32_t>();
        const auto* t = best.data<uint32_t>();
        int accepted = 0;
        while (accepted < depth && t[accepted] == d[accepted]) ++accepted;
        const int keep = accepted + 1;
        // Commit: rows past the kept prefix leave the KV caches and the convolution histories restart from the
        // history before the block plus the kept input rows. All lazy: the next step evaluates it.
        for (auto& visit : visits) {
            if (keep < n) {
                visit.kv.truncate(base + keep);
                for (int c = 0; c < 4; ++c)  // one kernel each (inkling2_history_rollback)
                    visit.conv[c] = inkling2_history_rollback(*visit.before[c], *visit.input[c], keep);
            }
            visit.kv.retain_window();
            visit.before = {}; visit.input = {};
        }
        length = base + keep;
        draft->commit(mx::slice(*taps, {0, 0, 0}, {keep, 4, D}), base);
        Inkling2Mlx::Speculation out{std::vector<uint32_t>(d, d + accepted), accepted, depth};
        out.tokens.push_back(t[accepted]);
        if (adaptive) policy.record(depth, accepted, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
        return out;
    }
};

Inkling2Mlx::Inkling2Mlx(const std::string& folder,int context,bool exact) : p_(std::make_unique<Impl>(folder,context,exact)) {}
Inkling2Mlx::~Inkling2Mlx()=default;
mx::array Inkling2Mlx::forward(std::span<const uint32_t> tokens,bool all,std::vector<mx::array>* trace,std::array<double,5>* phase_ms) {
    p_->phase_ms=phase_ms;
    try {auto result=p_->forward(tokens,all,trace);p_->phase_ms=nullptr;return result;}
    catch(...) {p_->phase_ms=nullptr;throw;}
}
void Inkling2Mlx::reset() {
    for (auto& v:p_->visits) { v.kv.reset(); v.conv={}; v.before={}; v.input={}; }
    if (p_->draft) p_->draft->reset();
    p_->policy = {};
    p_->length=0;
}
mx::array Inkling2Mlx::greedy_step(const mx::array& token) { return p_->greedy_step(token); }
void Inkling2Mlx::enable_draft(const std::string& folder) { p_->enable_draft(folder); }
bool Inkling2Mlx::drafting() const { return p_->draft != nullptr; }
Inkling2Mlx::Speculation Inkling2Mlx::speculate(uint32_t anchor, int depth) { return p_->speculate(anchor, depth); }
int Inkling2Mlx::position() const { return p_->length; }
int Inkling2Mlx::vocabulary() const { return p_->V; }
int Inkling2Mlx::hidden_size() const { return p_->D; }
int Inkling2Mlx::layer_count() const { return p_->L; }
int Inkling2Mlx::context() const { return p_->context; }
}  // namespace tqllm::mlx_backend

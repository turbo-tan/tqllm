#include "moe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace tqllm {

float moe_sigmoid(float x) {
    // Match ggml_sigmoid_f32: 1/(1+exp(-x)). For very negative x, exp(-x) overflows to inf and
    // 1/inf = 0, which is the correct limit, so no special casing is needed.
    return 1.0f / (1.0f + std::exp(-x));
}

float moe_softplus(float x) {
    // Numerically stable form of log(1+exp(x)) (ggml computes log1p(exp(x)) with an |x|<=15
    // guard; this form is equivalent to within FP32 and cannot overflow).
    return x > 0.f ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

MoeSelection moe_route(const MoeConfig & cfg, const float * logits, const float * probs_bias) {
    if (cfg.n_expert <= 0) throw std::runtime_error("moe_route: n_expert must be positive");
    if (cfg.n_expert_used <= 0 || cfg.n_expert_used > cfg.n_expert)
        throw std::runtime_error("moe_route: n_expert_used out of range");

    const int n = cfg.n_expert;

    // Step 1-2: gating function -> probs. Mirrors the switch in build_moe_ffn (llama-graph.cpp
    // :2100). SOFTMAX_WEIGHT leaves probs as the raw logits because its softmax is applied
    // later, to the selected weights only.
    std::vector<float> probs(n);
    switch (cfg.gating) {
        case MoeGating::Softmax: {
            float mx = -INFINITY;
            for (int i = 0; i < n; ++i) mx = std::max(mx, logits[i]);
            float sum = 0.f;
            for (int i = 0; i < n; ++i) {
                probs[i] = std::exp(logits[i] - mx);
                sum += probs[i];
            }
            for (int i = 0; i < n; ++i) probs[i] /= sum;
            break;
        }
        case MoeGating::Sigmoid:
            for (int i = 0; i < n; ++i) probs[i] = moe_sigmoid(logits[i]);
            break;
        case MoeGating::SoftmaxWeight:
            for (int i = 0; i < n; ++i) probs[i] = logits[i];
            break;
        case MoeGating::SqrtSoftplus:
            for (int i = 0; i < n; ++i) probs[i] = std::sqrt(moe_softplus(logits[i]));
            break;
        case MoeGating::None:
        default:
            throw std::runtime_error("moe_route: unsupported gating function");
    }

    // Step 3: the selection bias (exp_probs_b, DeepSeek V3) is added to build SELECTION probs
    // only. The reference keeps `probs` unbiased for the weights - "leave probs unbiased as
    // it's later used to get expert weights" (llama-graph.cpp:2123).
    std::vector<float> sel_probs(probs);
    if (probs_bias)
        for (int i = 0; i < n; ++i) sel_probs[i] += probs_bias[i];

    // Step 4: top-k by descending selection probability. ggml_argsort_top_k sorts with
    // cmp_argsort<DESC> = `data[a] > data[b]` over indices initialised 0..n-1, using
    // std::sort - which is NOT stable, so equal probabilities resolve arbitrarily in the
    // reference. We use a deterministic stable sort (descending value, ascending index) so our
    // selection is reproducible. Consequence, stated rather than hidden: on an exact tie the
    // chosen expert may differ from the reference's. Ties are a near-tie class, handled by the
    // same rule the project already applies elsewhere (characterise, don't pretend exactness).
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (sel_probs[a] != sel_probs[b]) return sel_probs[a] > sel_probs[b];
        return a < b;
    });

    MoeSelection out;
    out.ids.assign(order.begin(), order.begin() + cfg.n_expert_used);

    // Step 5: weights come from the UNBIASED probs gathered at the selected ids
    // (llama-graph.cpp:2183 `ggml_get_rows(probs, selected_experts)`).
    out.weights.resize(cfg.n_expert_used);
    for (int k = 0; k < cfg.n_expert_used; ++k) out.weights[k] = probs[out.ids[k]];

    // Step 6: SOFTMAX_WEIGHT applies softmax across the selected weights.
    if (cfg.gating == MoeGating::SoftmaxWeight) {
        float mx = -INFINITY;
        for (float w : out.weights) mx = std::max(mx, w);
        float sum = 0.f;
        for (float & w : out.weights) {
            w = std::exp(w - mx);
            sum += w;
        }
        for (float & w : out.weights) w /= sum;
    }

    // Step 7: normalise, then scale. The clamp is the reference's guard against dividing by
    // zero (llama-graph.cpp:2201), which also matters for sigmoid routing: a token whose every
    // sigmoid prob underflows would otherwise produce 0/0 = NaN and poison the residual stream.
    if (cfg.norm_weights) {
        float sum = 0.f;
        for (float w : out.weights) sum += w;
        sum = std::max(sum, MoeConfig::WEIGHT_SUM_EPS);
        for (float & w : out.weights) w /= sum;
    }
    if (cfg.weight_scale != 0.f && cfg.weight_scale != 1.f)
        for (float & w : out.weights) w *= cfg.weight_scale;

    return out;
}

void moe_expert_ffn(const float * x, int hidden, const float * w_gate, const float * w_up,
                    const float * w_down, int ff, float swiglu_clamp, bool clamp_gate,
                    float * out) {
    std::vector<float> gate(ff), up(ff);
    for (int o = 0; o < ff; ++o) {
        float sg = 0.f, su = 0.f;
        for (int i = 0; i < hidden; ++i) {
            sg += w_gate[size_t(o) * hidden + i] * x[i];
            su += w_up[size_t(o) * hidden + i] * x[i];
        }
        gate[o] = sg;
        up[o] = su;
    }

    const float limit = swiglu_clamp;
    constexpr float eps = 1e-6f;
    std::vector<float> act(ff);
    if (limit > eps) {
        // The reference clamps `up` to [-limit, limit] in both branches, then either
        // clamps gate to [-inf, limit] and calls swiglu_split (the GLM/DeepSeek-V4 branch),
        // or clamps silu(gate) (the generic branch). swiglu_split(g, u) = silu(g) * u.
        for (int o = 0; o < ff; ++o) up[o] = std::clamp(up[o], -limit, limit);
        for (int o = 0; o < ff; ++o) {
            const float g = clamp_gate ? std::min(gate[o], limit) : gate[o];
            float s = g / (1.f + std::exp(-g));  // silu
            if (!clamp_gate) s = std::min(s, limit);
            act[o] = s * up[o];
        }
    } else {
        for (int o = 0; o < ff; ++o) {
            const float g = gate[o];
            act[o] = (g / (1.f + std::exp(-g))) * up[o];
        }
    }

    for (int i = 0; i < hidden; ++i) {
        float s = 0.f;
        for (int o = 0; o < ff; ++o) s += w_down[size_t(i) * ff + o] * act[o];
        out[i] = s;
    }
}

void moe_combine(const MoeSelection & sel, const float * expert_outs, int hidden, float * out) {
    for (size_t k = 0; k < sel.ids.size(); ++k) {
        const float w = sel.weights[k];
        const float * src = expert_outs + k * size_t(hidden);
        for (int i = 0; i < hidden; ++i) out[i] += w * src[i];
    }
}

} // namespace tqllm

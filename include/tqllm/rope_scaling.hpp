#pragma once
// RoPE context extension (llama.cpp names: --rope-scaling none|linear|yarn, --rope-scale, --yarn-orig-ctx,
// --yarn-ext-factor, --yarn-attn-factor, --yarn-beta-fast, --yarn-beta-slow). The rotation of pair d becomes
// pos * inv_freq(d) * mul[d] with cos / sin scaled by mscale, the same as ggml's rope_yarn:
//   theta = theta_interp * (1 - ramp) + theta_extrap * ramp, theta_interp = theta_extrap / factor
//   ramp  = (1 - clamp((d - low) / (high - low), 0, 1)) * ext_factor
//   mscale = attn_factor * (1 + 0.1 ln(factor))  (YaRN only)
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace tqllm {

struct RopeScaling {
    std::string type = "none";  // none | linear | yarn
    float factor = 1.f;         // context multiplier (1 / freq_scale)
    long orig_ctx = 0;          // training context (0: the model's context_length)
    float ext_factor = -1.f;    // YaRN extrapolation mix (-1: 1 for yarn)
    float attn_factor = 1.f;
    float beta_fast = 32.f, beta_slow = 1.f;
    bool on() const { return type != "none" && factor != 1.f; }
};

struct RopeTable {
    std::vector<float> mul;  // per rotary pair
    float mscale = 1.f;
};

// rot: rotary dims (pairs = rot / 2), base: rope theta, orig_ctx: training context
inline RopeTable rope_scaling_table(const RopeScaling & s, int rot, float base, long orig_ctx) {
    RopeTable t;
    const int pairs = rot / 2;
    t.mul.assign(size_t(pairs), 1.f);
    if (!s.on()) return t;
    const float fs = 1.f / s.factor;
    if (s.type == "linear") {
        std::fill(t.mul.begin(), t.mul.end(), fs);
        return t;
    }
    const float ext = s.ext_factor < 0 ? 1.f : s.ext_factor;
    auto corr = [&](float n_rot) {
        return float(rot) * std::log(float(orig_ctx) / (n_rot * 2.f * float(M_PI))) / (2.f * std::log(base));
    };
    const float low = std::max(0.f, std::floor(corr(s.beta_fast))), high = std::min(float(rot - 1), std::ceil(corr(s.beta_slow)));
    for (int d = 0; d < pairs; ++d) {
        const float y = (float(d) - low) / std::max(0.001f, high - low);
        const float ramp = (1.f - std::min(1.f, std::max(0.f, y))) * ext;
        t.mul[size_t(d)] = fs * (1.f - ramp) + ramp;
    }
    t.mscale = s.attn_factor * (ext != 0.f ? 1.f + 0.1f * std::log(1.f / fs) : 1.f);
    return t;
}

}  // namespace tqllm

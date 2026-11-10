#pragma once
#include "gguf.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tqllm::spec {
struct DSpineConfig {
    int hidden = 0, code = 0, rank = 0, layers = 0, block = 0, topk = 0;
    float epsilon = 0;
};

inline bool is_dspine(const GgufFile& f) {
    return f.has("dflash.dspine_version") || f.tensors.count("dspine_codebook.weight") != 0;
}

inline DSpineConfig dspine_config(const GgufFile& f, int hidden, int vocab, int target_layers, const std::string& target_arch = "") {
    auto integer = [&](const std::string& key) {
        const double n = f.num("dflash." + key);
        if (!std::isfinite(n) || n < 1 || n > std::numeric_limits<int>::max() || std::floor(n) != n)
            throw std::invalid_argument("DSpine: invalid " + key);
        return int(n);
    };
    if (integer("dspine_version") != 1 || f.num("dflash.draft_block_causal") != 1)
        throw std::invalid_argument("DSpine: unsupported format or noncausal draft block");
    if (f.tensors.count("selector_hidden.weight") || f.tensors.count("selector_predecessor.weight") ||
        f.tensors.count("selector_successor.weight") || f.tensors.count("markov_w1.weight"))
        throw std::invalid_argument("DSpine cannot be combined with a DFlash2 or DSpark selector");
    if (!target_arch.empty() && f.str("dflash.target_architecture") != target_arch)
        throw std::invalid_argument("DSpine: target architecture does not match the trained drafter");
    DSpineConfig c;
    c.hidden = integer("embedding_length"); c.layers = integer("block_count");
    c.block = integer("block_size"); c.topk = integer("selector_top_k");
    c.epsilon = float(f.num("dflash.dspine_epsilon"));
    if (c.hidden != hidden || integer("target_block_count") != target_layers || c.block < 2 ||
        c.block > 16 || c.topk != 16 || vocab < c.topk || !std::isfinite(c.epsilon) || c.epsilon <= 0)
        throw std::invalid_argument("DSpine: incompatible target or unsupported block dimensions");
    auto shape = [&](const std::string& name, std::vector<uint64_t> dims) {
        const auto it = f.tensors.find(name);
        if (it == f.tensors.end() || it->second.ne != dims ||
            (it->second.type != GGML_BF16 && it->second.type != GGML_F32))
            throw std::invalid_argument("DSpine: missing, quantized, or malformed tensor " + name);
    };
    const auto cb = f.tensors.find("dspine_codebook.weight"), gate = f.tensors.find("dspine_gate.weight");
    if (cb == f.tensors.end() || gate == f.tensors.end() || cb->second.ne.size() != 2 || gate->second.ne.size() != 2 ||
        cb->second.ne[0] == 0 || cb->second.ne[0] > 4096 || gate->second.ne[1] == 0 || gate->second.ne[1] > 4096)
        throw std::invalid_argument("DSpine: invalid bridge dimensions");
    c.code = int(cb->second.ne[0]); c.rank = int(gate->second.ne[1]);
    const uint64_t C = uint64_t(c.code), R = uint64_t(c.rank), H = uint64_t(hidden);
    shape("dspine_codebook.weight", {C, uint64_t(vocab)});
    shape("dspine_gate.weight", {C, R}); shape("dspine_pred_gate.weight", {C, R});
    shape("dspine_prev.weight", {C, R}); shape("dspine_gate.bias", {R});
    for (int l = 0; l < c.layers; ++l) {
        const auto p = "blk." + std::to_string(l) + ".";
        shape(p + "dspine_read.weight", {H, C}); shape(p + "dspine_write.weight", {R, H});
    }
    const auto& ids = f.kv.at("dflash.target_layers").arr;
    if (ids.empty()) throw std::invalid_argument("DSpine: target feature layers are empty");
    std::vector<bool> seen(size_t(target_layers), false);
    for (const auto id : ids) {
        if (!std::isfinite(id) || id < 0 || id >= target_layers || std::floor(id) != id || seen[size_t(id)])
            throw std::invalid_argument("DSpine: invalid or duplicate target feature layer");
        seen[size_t(id)] = true;
    }
    return c;
}

// CPU reference follows Tencent's BF16 rounding boundaries, not fused RMSNorm.
inline float dspine_bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, sizeof(u));
    if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u; std::memcpy(&x, &u, sizeof(x)); return x;
}

struct DSpineWeights {
    const float *read, *write, *codebook, *gate, *pred_gate, *previous, *bias;
};

inline std::vector<float> dspine_write_reference(const DSpineConfig& c, const DSpineWeights& w,
                                                const std::vector<float>& x, const std::vector<uint32_t>& anchors) {
    const int H = c.hidden, C = c.code, R = c.rank, B = c.block;
    if (H < 1 || C < 1 || R < 1 || B < 2 || anchors.empty() || x.size() != anchors.size() * B * H)
        throw std::invalid_argument("DSpine: invalid reference block");
    const int N = int(anchors.size()) * B;
    std::vector<float> out(x), codes(size_t(N) * C), rms(N), packed(size_t(N) * 3 * R);
    for (int row = 0; row < N; ++row) {
        float sum = 0;
        for (int h = 0; h < H; ++h) { const float v = dspine_bf16(x[size_t(row) * H + h]); sum += v * v; }
        rms[row] = std::sqrt(std::max(sum / H, std::numeric_limits<float>::min()));
        float norm = 0;
        for (int k = 0; k < C; ++k) {
            float dot = 0;
            for (int h = 0; h < H; ++h)
                dot += w.read[size_t(k) * H + h] * dspine_bf16(dspine_bf16(x[size_t(row) * H + h]) / (rms[row] + c.epsilon));
            codes[size_t(row) * C + k] = dspine_bf16(dot);
            norm += codes[size_t(row) * C + k] * codes[size_t(row) * C + k];
        }
        for (int k = 0; k < C; ++k)
            codes[size_t(row) * C + k] = dspine_bf16(std::sqrt(float(C)) *
                (row % B ? codes[size_t(row) * C + k] / (std::sqrt(norm) + c.epsilon)
                         : w.codebook[size_t(anchors[size_t(row / B)]) * C + k]));
        for (int r = 0; r < R; ++r) {
            float q = 0, s = 0, v = 0;
            for (int k = 0; k < C; ++k) {
                const float code = codes[size_t(row) * C + k];
                q += w.gate[size_t(r) * C + k] * code;
                s += w.pred_gate[size_t(r) * C + k] * code;
                v += w.previous[size_t(r) * C + k] * code;
            }
            packed[size_t(row) * 3 * R + r] = dspine_bf16(q);
            packed[size_t(row) * 3 * R + R + r] = dspine_bf16(s);
            packed[size_t(row) * 3 * R + 2 * R + r] = dspine_bf16(v);
        }
    }
    for (int row = 0; row < N; ++row) {
        if (row % B == 0) continue;
        for (int h = 0; h < H; ++h) {
            float delta = 0;
            for (int r = 0; r < R; ++r) {
                const float z = packed[size_t(row) * 3 * R + r] + packed[size_t(row - 1) * 3 * R + R + r] + w.bias[r];
                const float msg = dspine_bf16(packed[size_t(row - 1) * 3 * R + 2 * R + r] / (1.f + std::exp(-z)));
                delta += w.write[size_t(h) * R + r] * msg;
            }
            out[size_t(row) * H + h] = dspine_bf16(dspine_bf16(x[size_t(row) * H + h]) + rms[row] * dspine_bf16(delta));
        }
    }
    return out;
}
}  // namespace tqllm::spec

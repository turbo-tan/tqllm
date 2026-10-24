#pragma once
// Qwen4 dense (27B): the Flash-Next block (hyper-connection mixer, gated attention with QSA, GDN) with a dense SwiGLU FFN
// in place of the routed experts. One sequence; each step is a chunk of up to 2048 tokens (attention_core's QSA budget).
#include "fn_layers.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

class Qwen4Dense {
public:
    explicit Qwen4Dense(const Checkpoint & cp)
        : cp_(cp), c_(cp.config), h_(c_.hidden), width_(c_.hidden * c_.hc), rec_(size_t(c_.layers)), att_(size_t(c_.layers)),
          ple_past_(size_t(c_.layers)) {
        require(c_.experts == 0 && c_.dense_ff > 0, "qwen4 dense: the checkpoint has routed experts");
        if (!c_.ple_layers.empty()) history_.emplace(c_.ple, size_t(g_kv_capacity));
    }
    int position() const { return pos_; }

    // The head input [n][hidden] after the next n tokens; attention and recurrent state carry over to the next call.
    // trace: when set, the last token's residual after every layer ([layers][hidden * streams])
    std::unique_ptr<Device<float>> step(const std::vector<int> & tokens, std::vector<float> * trace = nullptr) {
        const int n = int(tokens.size());
        require(n > 0 && n <= 2048, "qwen4 dense: a step takes 1..2048 tokens");
        require(pos_ + n <= g_kv_capacity, "qwen4 dense: context exceeds the KV capacity");
        const std::vector<uint64_t> rows(tokens.begin(), tokens.end());
        const auto embedding = cp_.file("token_embd.weight").gather_rows("token_embd.weight", rows);
        Device<float> de(embedding.size()); de.put(embedding);
        auto residual = std::make_unique<Device<float>>(size_t(width_) * n);
        tqllm_cuda::hc_init(de.p, residual->p, h_, c_.hc, n, nullptr);
        std::unique_ptr<Device<float>> ple;  // [n][hidden]: each token's hashed n-gram rows of per_layer_token_embd
        if (history_) {
            const auto ids = history_->append_batch(std::vector<int32_t>(tokens.begin(), tokens.end()));
            const auto rows_h = cp_.file("per_layer_token_embd.weight").gather_rows("per_layer_token_embd.weight",
                                                                                   std::vector<uint64_t>(ids.begin(), ids.end()));
            ple = std::make_unique<Device<float>>(rows_h.size()); ple->put(rows_h);
        }
        if (trace) trace->clear();
        for (int l = 0; l < c_.layers; ++l) {
            layer(l, residual, n, ple ? ple->p : nullptr);
            if (trace) {
                std::vector<float> last(static_cast<size_t>(width_));
                ck(cudaMemcpy(last.data(), residual->p + size_t(n - 1) * width_, last.size() * sizeof(float), cudaMemcpyDeviceToHost));
                trace->insert(trace->end(), last.begin(), last.end());
            }
        }
        pos_ += n;
        Device<float> norm(size_t(width_) * n);
        tqllm_cuda::hc_norm(residual->p, cp_.dense_device("output_hc_norm.weight"), norm.p, h_, c_.hc, n, float(cp_.epsilon), nullptr);
        auto out = std::make_unique<Device<float>>(size_t(h_) * n);
        mix_device(cp_, "output_hc_", norm.p, n, out->p, nullptr);
        return out;
    }
    std::vector<float> logits(const Device<float> & head, int n) const { return cp_.head_logits_device(head.p, n); }

private:
    void layer(int l, std::unique_ptr<Device<float>> & residual, int n, const float * ple_rows) {
        const std::string prefix = "blk." + std::to_string(l) + ".";
        const float eps = float(cp_.epsilon);
        if (ple_rows && std::find(c_.ple_layers.begin(), c_.ple_layers.end(), l) != c_.ple_layers.end())
            ple_block(cp_, l, residual, n, ple_rows, [&](const float * pn, float * conv) {
                auto & past = ple_past_[size_t(l)];
                if (!past) {
                    past = std::make_unique<Device<float>>(size_t((c_.ple_kernel - 1) * c_.ple.ngram) * width_);
                    ck(cudaMemsetAsync(past->p, 0, past->count * sizeof(float), nullptr));
                }
                Device<float> next(past->count);
                tqllm_cuda::ple_conv(pn, past->p, cp_.dense_device(prefix + "ple_conv1d.weight"), conv, next.p, width_, n,
                                     c_.ple_kernel, c_.ple.ngram, nullptr);
                ck(cudaMemcpyAsync(past->p, next.p, next.count * sizeof(float), cudaMemcpyDeviceToDevice, nullptr));
            });
        Device<float> norm(size_t(width_) * n), mixed(size_t(h_) * n), inject(size_t(c_.hc) * n), block(size_t(h_) * n),
            combined(size_t(width_) * n);
        tqllm_cuda::hc_norm(residual->p, cp_.dense_device(prefix + "hc_attn_norm.weight"), norm.p, h_, c_.hc, n, eps, nullptr);
        mix_device(cp_, prefix + "hc_attn_", norm.p, n, mixed.p, inject.p);
        if (c_.compress_ratios[size_t(l)]) attention_core(cp_, mixed.p, block.p, n, l, nullptr, &att_[size_t(l)]);
        else recurrent_core(cp_, mixed.p, block.p, n, l, nullptr, &rec_[size_t(l)]);
        tqllm_cuda::hc_combine(residual->p, block.p, inject.p, combined.p, h_, c_.hc, n, nullptr);
        tqllm_cuda::hc_norm(combined.p, cp_.dense_device(prefix + "hc_ffn_norm.weight"), norm.p, h_, c_.hc, n, eps, nullptr);
        mix_device(cp_, prefix + "hc_ffn_", norm.p, n, mixed.p, inject.p);
        const int ff = c_.dense_ff;
        Device<float> gate(size_t(n) * ff), up(gate.count), act(gate.count), ffn(size_t(n) * h_);
        {
            const Checkpoint::SharedInput shared_in(cp_, mixed.p, size_t(n) * h_);
            cp_.project(prefix + "ffn_gate.weight", mixed.p, gate.p, h_, ff, n);
            cp_.project(prefix + "ffn_up.weight", mixed.p, up.p, h_, ff, n);
        }
        tqllm_cuda::fn_swiglu_f32(gate.p, up.p, act.p, int64_t(act.count), nullptr);
        cp_.project(prefix + "ffn_down.weight", act.p, ffn.p, ff, h_, n);
        auto updated = std::make_unique<Device<float>>(size_t(width_) * n);
        tqllm_cuda::hc_combine(combined.p, ffn.p, inject.p, updated->p, h_, c_.hc, n, nullptr);
        residual = std::move(updated);
    }

    const Checkpoint & cp_;
    const Qwen4ExpConfig & c_;
    int h_, width_, pos_ = 0;
    std::vector<RecurrentState> rec_;
    std::vector<AttentionState> att_;
    std::optional<PleTokenHistory> history_;  // n-gram hash history (PLE checkpoints only)
    std::vector<std::unique_ptr<Device<float>>> ple_past_;  // per layer: the PLE conv history
};

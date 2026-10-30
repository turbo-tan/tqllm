#pragma once
#include "weights.hpp"
#include "kv_tier.hpp"
#include <memory>
#include <map>

namespace tqllm::mlx_backend {
// Experimental dense text forward. One independent cache per instance.
class Qwen35 {
public:
    class Snapshot {
        struct State;
        std::shared_ptr<const State> state_;
        friend class Qwen35;
    };
    Snapshot checkpoint() const;
    void restore(const Snapshot& snapshot);
    class Verification {
        struct State;
        std::shared_ptr<const State> state_;
        friend class Qwen35;
    };
    mlx::core::array verify(std::span<const uint32_t> tokens, Verification& batch);
    // verify with the batch ids already on the GPU (e.g. the drafter's unevaluated output): no host readback between
    // draft and verify
    mlx::core::array verify_device(const mlx::core::array& ids, Verification& batch);
    void accept_prefix(const Verification& batch, size_t count);
    mlx::core::array hidden_states() const;
    void enable_mtp();
    void draft_shortlist(const mlx::core::array& target_logits, int count);
    class DraftSnapshot {
        struct State;
        std::shared_ptr<const State> state_;
        friend class Qwen35;
    };
    DraftSnapshot draft_checkpoint() const;
    void draft_restore(const DraftSnapshot& snapshot);
    mlx::core::array draft_forward(std::span<const uint32_t> tokens,
                                  const mlx::core::array& previous_hidden, int position, bool logits=true);
    mlx::core::array draft_ids() const;
    mlx::core::array draft_hidden_states() const;
    void draft_inject(std::span<const uint32_t> tokens,
                      const mlx::core::array& previous_hidden, int position);
    void feature_layers(const std::vector<int>& layers);
    mlx::core::array feature_states() const;
    mlx::core::array embed_tokens(std::span<const uint32_t> tokens) const;
    // the same embedding of ids already on the GPU (uint32 [n]), no readback
    mlx::core::array embed_device(const mlx::core::array& ids) const;
    // forward / verify submit their graph and return without waiting (draft-ahead chains the next draft behind it)
    void defer_eval(bool enabled);
    mlx::core::array shared_draft_head(const mlx::core::array& normalized) const;
    void enable_draft_head4();
    Qwen35(const std::string& path, int context);
    ~Qwen35();
    mlx::core::array forward(std::span<const uint32_t> tokens, bool all_logits = false);
    void reset();
    // the next request's length (prompt + new tokens + a verify block): its KV cache is allocated once at that size
    void reserve_kv(int rows);
    // this request's KV format (kv_tier.hpp), chosen before its prefill from the memory budget
    void kv_format(KvTier tier);
    KvTier kv_format() const;
    // the densest-needed format for a request of `rows` positions on this device (memory in use + working set)
    KvTier choose_kv_format(int rows,int prefill_chunk,size_t other_row_bytes=0) const;
    // the format plus whether the drafter keeps its far-context caches (far_row_bytes per row), kv_tier.hpp plan_kv
    KvPlan plan_kv_format(int prompt_rows,int answer_rows,int prefill_chunk,size_t far_row_bytes) const;
    // memory for a new request's plan: in use now without this engine's KV buffers, the working set, the prefill transient
    KvBudget kv_budget(int prefill_chunk) const;
    int attention_layers() const;
    int kv_heads() const;
    int q_heads() const;
    int head_dim() const;
    int position() const;
    int vocabulary() const;
    void profile(bool enabled);
    void profile_stages(bool enabled);
    const std::map<std::string,double>& stage_timings() const;
    std::pair<double, double> timings() const; // body, head milliseconds of the last forward
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

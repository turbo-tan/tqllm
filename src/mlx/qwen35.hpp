#pragma once
#include "weights.hpp"
#include <memory>
#include <map>

namespace tqllm::mlx_backend {
// Experimental dense text forward. One independent cache per instance.
class Qwen35 {
public:
    enum class GdnMode { baseline,native,scan };
    void gdn_mode(GdnMode mode);
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
    Qwen35(const std::string& path, int context);
    ~Qwen35();
    mlx::core::array forward(std::span<const uint32_t> tokens, bool all_logits = false);
    void reset();
    int position() const;
    int vocabulary() const;
    void profile(bool enabled);
    void prefill_gemm(bool enabled);
    void verify_tiles(bool enabled);
    void tq3_vector_load(bool enabled);
    void profile_stages(bool enabled);
    const std::map<std::string,double>& stage_timings() const;
    std::pair<double, double> timings() const; // body, head milliseconds of the last forward
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

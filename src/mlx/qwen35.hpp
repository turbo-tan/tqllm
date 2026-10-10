#pragma once
#include "weights.hpp"
#include <memory>
#include <map>

namespace tqllm::mlx_backend {
// Experimental dense text forward. One independent cache per instance, no speculation.
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

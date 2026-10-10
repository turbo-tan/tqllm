#pragma once
#include "weights.hpp"
#include <memory>

namespace tqllm::mlx_backend {
// Experimental dense text forward. One independent cache per instance, no speculation.
class Qwen35 {
public:
    Qwen35(const std::string& path, int context);
    ~Qwen35();
    mlx::core::array forward(std::span<const uint32_t> tokens, bool all_logits = false);
    void reset();
    int position() const;
    int vocabulary() const;
    void profile(bool enabled);
    std::pair<double, double> timings() const; // body, head milliseconds of the last forward
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

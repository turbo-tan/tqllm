#pragma once
#include <mlx/mlx.h>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace tqllm::mlx_backend {
class Inkling2Mlx {
public:
    // exact = false (--exact off): 8-bit projections and LM head, gated by KLD against BF16 (see Impl::q8)
    Inkling2Mlx(const std::string& folder, int context, bool exact = true);
    ~Inkling2Mlx();
    mlx::core::array forward(std::span<const uint32_t> tokens, bool all_logits = false,
                             std::vector<mlx::core::array>* visit_trace = nullptr,
                             std::array<double,5>* synchronized_phase_ms = nullptr);
    // One greedy decode step queued on the GPU without waiting: feeds `token` (a [1] array, host value or the
    // result of the previous greedy_step) and returns its argmax token, still being computed. Queue the next
    // step before reading this one (item()) so graph building overlaps the GPU. The step consumes context
    // whether or not its token is read.
    mlx::core::array greedy_step(const mlx::core::array& token);
    // DSpark speculative decoding (before the first forward): `source` is the checkpoint folder (its mtp.safetensors)
    // or a DSpark drafter GGUF (Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf).
    void enable_draft(const std::string& folder);
    bool drafting() const;
    struct Speculation {
        std::vector<uint32_t> tokens;  // the accepted drafts, then the target's own next token (the next anchor)
        int accepted = 0, drafted = 0;
    };
    // One greedy speculative step after `anchor` (the last emitted token, not yet consumed) with `depth` (1..7)
    // drafts, or 0: the depth this sequence's acceptance and step times favour. Its tokens are the verify path's
    // greedy decode.
    Speculation speculate(uint32_t anchor, int depth);
    void reset();
    int position() const;
    int vocabulary() const;
    int hidden_size() const;
    int layer_count() const;
    int context() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace tqllm::mlx_backend

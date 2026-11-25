#pragma once
#include <mlx/mlx.h>
#include <array>
#include <cstdint>
#include <any>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace tqllm::mlx_backend {
// the drafter's arithmetic (A/B of acceptance against cost): its projections as stored (BF16) instead of 8-bit, and the
// BF16 LM head instead of the 6-bit copy for its candidates
struct Inkling2DraftPrecision { bool bf16_projections = false, bf16_head = false; };
class Inkling2Mlx {
public:
    // exact = false (--exact off): dense_bits (8 or 4) projections and LM head, gated by KLD against BF16 (see Impl::q8)
    Inkling2Mlx(const std::string& folder, int context, bool exact = true, int dense_bits = 8);
    ~Inkling2Mlx();
    // verify_block: run the tokens as a speculative verify block does (its kernels), committing them as forward does
    // (row-invariance checks: row j must equal the j-th of single-token forwards)
    mlx::core::array forward(std::span<const uint32_t> tokens, bool all_logits = false,
                             std::vector<mlx::core::array>* visit_trace = nullptr,
                             std::array<double,5>* synchronized_phase_ms = nullptr, bool verify_block = false);
    // One greedy decode step queued on the GPU without waiting: feeds `token` (a [1] array, host value or the
    // result of the previous greedy_step) and returns its argmax token, still being computed. Queue the next
    // step before reading this one (item()) so graph building overlaps the GPU. The step consumes context
    // whether or not its token is read.
    mlx::core::array greedy_step(const mlx::core::array& token);
    // Speculative decoding (before the first forward): `source` is the checkpoint folder (DSpark: its mtp.safetensors),
    // a DSpark drafter GGUF (Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf) or a DFlash2 drafter folder
    // (incoai/Inkling-2-Mini-DFlash2: config.json + model.safetensors).
    using DraftPrecision = Inkling2DraftPrecision;
    void enable_draft(const std::string& folder, DraftPrecision precision = {});
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
    // Run each row count's kernels once (a short prompt, a decode step and, with a drafter, a verify block of every
    // depth), then reset: Metal compiles a kernel per shape at first use, which otherwise lands in a request's first
    // steps (a 2-token verify cost more than a 3-token one in --blocks).
    void warm_up();
    // Turn reuse: the whole state after a prefill (every visit's KV rows and convolution histories, the drafter's
    // caches, the position). restore() continues from it as if that prefill had just run; the arrays are shared, so a
    // kept state costs a copy only of the buffers the next request writes to.
    struct Kept {
        std::shared_ptr<const std::any> state;
        int position = 0;
    };
    Kept keep() const;
    void restore(const Kept& kept);
    int position() const;
    int vocabulary() const;
    int hidden_size() const;
    int layer_count() const;
    int context() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
// a role token (<|message_user|>, <|message_model|>, <|message_system|>, <|message_tool|>): where a message starts
inline bool inkling2_message_start(uint32_t token) { return token >= 200000u && token <= 200003u; }
}  // namespace tqllm::mlx_backend

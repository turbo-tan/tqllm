#pragma once
#include <mlx/mlx.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tqllm::mlx_backend {
// Inkling-2's DSpark drafter (mtp.safetensors) on MLX: three dense layers over a block [anchor, mask x 7] whose keys
// come from the committed tokens' target taps, then the rnn_hidden Markov head folded into the target's LM head.
// It only proposes tokens; the target's verify decides, so its arithmetic follows the CUDA drafter
// (src/models/inkling2/loop/inkling2_mtp.cu) without its bf16 rounding of every step.
class Inkling2MlxDraft {
public:
    // source: a checkpoint folder (its mtp.safetensors) or a dflash-architecture GGUF of the same drafter (Unsloth's
    // dspark-Inkling-2-Mini-Q8_0.gguf); unembed: the target's LM head [>= vocab, D] (BF16, for the Markov fold);
    // vocab: rows that are real tokens;
    // window: sliding window; head: the target's LM head as it runs (x [n, D] -> logits [n, >= vocab]).
    using Head = std::function<mlx::core::array(const mlx::core::array&)>;
    Inkling2MlxDraft(const std::string& source, int context, const mlx::core::array& unembed, int vocab, int window, Head head);
    ~Inkling2MlxDraft();
    // Draft context of m committed tokens: their taps [m, 4, D] (BF16), at positions pos0 .. pos0 + m - 1.
    void commit(const mlx::core::array& taps, int pos0);
    // `steps` (1..7) draft tokens after the anchor at `position`: anchor_x0 is the anchor's normalised embedding
    // [1, D] and anchor the token itself ([1] uint32, may be lazy). Returns [steps] uint32, not evaluated.
    mlx::core::array propose(const mlx::core::array& anchor_x0, const mlx::core::array& anchor, int position, int steps);
    // Arrays a lazily built step must evaluate with its outputs (the draft context caches).
    std::vector<mlx::core::array> state() const;
    void reset();
    static constexpr int kBlock = 8;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace tqllm::mlx_backend

#pragma once
#include <mlx/mlx.h>
#include <any>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tqllm::mlx_backend {
// A drafter of the Inkling-2 MLX engine: committed tokens' target taps in, a block of greedy proposals out (the target's
// verify decides, so a drafter may round differently from its reference).
class Inkling2MlxDrafter {
public:
    virtual ~Inkling2MlxDrafter() = default;
    // Draft context of m committed tokens: their taps [m, 4, D] (BF16): the states entering applications 5, 23, 41 and
    // the final-norm output, at positions pos0 .. pos0 + m - 1.
    virtual void commit(const mlx::core::array& taps, int pos0) = 0;
    // `steps` (1..7) draft tokens after the anchor at `position`: anchor_x0 is the anchor's normalised embedding
    // [1, D] and anchor the token itself ([1] uint32, may be lazy). Returns [steps] uint32, not evaluated.
    virtual mlx::core::array propose(const mlx::core::array& anchor_x0, const mlx::core::array& anchor, int position, int steps) = 0;
    // Arrays a lazily built step must evaluate with its outputs (the draft context caches).
    virtual std::vector<mlx::core::array> state() const = 0;
    virtual void reset() = 0;
    // the context caches as they stand (turn reuse); restore() puts them back
    virtual std::any keep() const = 0;
    virtual void restore(const std::any& kept) = 0;
    static constexpr int kBlock = 8;
};
// Inkling-2's DSpark drafter (mtp.safetensors) on MLX: three dense layers over a block [anchor, mask x 7] whose keys
// come from the committed tokens' target taps, then the rnn_hidden Markov head folded into the target's LM head.
// It only proposes tokens; the target's verify decides, so its arithmetic follows the CUDA drafter
// (src/models/inkling2/loop/inkling2_mtp.cu) without its bf16 rounding of every step.
class Inkling2MlxDraft final : public Inkling2MlxDrafter {
public:
    // source: a checkpoint folder (its mtp.safetensors) or a dflash-architecture GGUF of the same drafter (Unsloth's
    // dspark-Inkling-2-Mini-Q8_0.gguf); unembed: the target's LM head [>= vocab, D] (BF16, for the Markov fold);
    // vocab: rows that are real tokens;
    // window: sliding window; head: the target's LM head as it runs (x [n, D] -> logits [n, >= vocab]).
    using Head = std::function<mlx::core::array(const mlx::core::array&)>;
    Inkling2MlxDraft(const std::string& source, int context, const mlx::core::array& unembed, int vocab, int window, Head head);
    ~Inkling2MlxDraft() override;
    void commit(const mlx::core::array& taps, int pos0) override;
    mlx::core::array propose(const mlx::core::array& anchor_x0, const mlx::core::array& anchor, int position, int steps) override;
    std::vector<mlx::core::array> state() const override;
    void reset() override;
    std::any keep() const override;
    void restore(const std::any& kept) override;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
// incoai/Inkling-2-Mini-DFlash2 (Apache-2.0): a five-layer block-diffusion drafter over [anchor, mask x 7] whose keys
// come from the committed tokens' taps (fc + hidden_norm), with two-tap dynamic convolutions around attention and MLP,
// the target's LM head for the top-16 candidates per position and a rank-256 selector walking one path through them.
class Inkling2MlxDFlash2 final : public Inkling2MlxDrafter {
public:
    using Head = Inkling2MlxDraft::Head;
    // ids [n] uint32 -> the target's normalised input rows [n, D] (embedding + embed_norm, as the target's first visit)
    using Embed = std::function<mlx::core::array(const mlx::core::array&)>;
    // folder: config.json (DFlash2DraftModel) + model.safetensors; vocab: rows of the head that are real tokens.
    // bf16_projections: the projections as stored (tests against the reference); else 8-bit (served).
    Inkling2MlxDFlash2(const std::string& folder, int context, int vocab, Head head, Embed embed, bool bf16_projections = false);
    ~Inkling2MlxDFlash2() override;
    static bool is_drafter_folder(const std::string& path);
    void commit(const mlx::core::array& taps, int pos0) override;
    mlx::core::array propose(const mlx::core::array& anchor_x0, const mlx::core::array& anchor, int position, int steps) override;
    std::vector<mlx::core::array> state() const override;
    void reset() override;
    std::any keep() const override;
    void restore(const std::any& kept) override;
    // test hook: the last propose()'s final normalised block rows [8, D] (F32) and top-16 ids / logits [8, 16]
    struct Trace { mlx::core::array hidden, ids, logits; };
    Trace last_trace() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}  // namespace tqllm::mlx_backend

#pragma once
// MoE routing and expert combination, arch-independent and host-only (FP32).
//
// This mirrors llama.cpp's llm_graph_context::build_moe_ffn (src/llama-graph.cpp:2053) step
// for step, because a wrong router produces plausible-looking but wrong text - the failure is
// silent. The single subtlety that must not be lost is step 3 vs step 5 below: the expert
// selection bias (exp_probs_b, introduced in DeepSeek V3) changes WHICH experts are selected
// but the weights must still be taken from the UNBIASED probs. Reading `probs` for both is the
// classic bug.
//
// Nothing here knows about a model family, a weight format, or CUDA. The kernels that consume
// a MoeSelection are arch-specific and live elsewhere.
#include <cstdint>
#include <vector>

namespace tqllm {

// llama_expert_gating_func_type (src/llama-hparams.h:16). GLM-5.3-Flash ships 2 = SIGMOID
// (read from its GGUF: glm5next.expert_gating_func = 2).
enum class MoeGating : int {
    None = 0,
    Softmax = 1,
    Sigmoid = 2,
    SoftmaxWeight = 3,  // softmax applied to the selected weights, not the logits
    SqrtSoftplus = 4,
};

struct MoeConfig {
    int n_expert = 0;
    int n_expert_used = 0;   // top-k
    MoeGating gating = MoeGating::Sigmoid;
    bool norm_weights = false;  // expert_weights_norm
    float weight_scale = 0.f;   // expert_weights_scale (0 and 1 both mean "no scaling")
    // GLM/DeepSeek-V4 clamp the SwiGLU inputs before the activation (swiglu_clamp_exp).
    // <= 0 disables the clamp, matching the reference's `limit > eps` test with eps 1e-6.
    float swiglu_clamp = 0.f;
    // ggml clamps the weight sum to the smallest F16 normal before dividing.
    static constexpr float WEIGHT_SUM_EPS = 6.103515625e-5f;
};

// Result of routing one token. `ids` are expert indices in descending biased-probability
// order; `weights` are the (normalised, scaled) coefficients from the UNBIASED probs.
struct MoeSelection {
    std::vector<int> ids;
    std::vector<float> weights;
};

// Router logits for one token -> a selection. `logits` has cfg.n_expert entries and already
// includes any router bias (gate_inp_b); `probs_bias` is exp_probs_b (nullptr when absent).
MoeSelection moe_route(const MoeConfig & cfg, const float * logits, const float * probs_bias);

// Activations, exposed so they can be unit-tested independently of routing.
float moe_sigmoid(float x);
float moe_softplus(float x);

// One expert's FFN contribution for a single token, in FP32:
//   out = W_down * swiglu(W_gate * x, W_up * x)
// with the reference's optional clamping applied to `up` and (for the GLM family) to `gate`.
// Matrices are row-major [rows][cols]; gate/up are [ff][hidden], down is [hidden][ff].
// `clamp_gate` reproduces the GGML_ARCH_GLM5NEXT branch, which clamps gate to [-inf, limit]
// while the generic branch clamps silu(gate) instead.
void moe_expert_ffn(const float * x, int hidden, const float * w_gate, const float * w_up,
                    const float * w_down, int ff, float swiglu_clamp, bool clamp_gate,
                    float * out);

// Weighted sum of per-expert outputs into `out` (which must already be zeroed by the caller):
//   out += sum_i weights[i] * expert_out[i]
void moe_combine(const MoeSelection & sel, const float * expert_outs, int hidden, float * out);

} // namespace tqllm

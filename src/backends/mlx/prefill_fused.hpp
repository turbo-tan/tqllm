#pragma once
#include <mlx/mlx.h>
#include <tuple>
#include <utility>

namespace tqllm::mlx_backend {
// Prefill elementwise passes in one launch each, written to give the bits of the MLX op chains they replace (the
// engine checks that once on device and falls back otherwise, prefill_fused_exact).
// rms_norm(x, w, eps) rounded to BF16 values: {FP32, the same values as FP16}, x FP32 [rows, d] with d > 4096 (MLX's
// looped reduction: 1024 threads, 4 reads per thread per step)
std::pair<mlx::core::array, mlx::core::array> rms_norm_bf16_pair(const mlx::core::array& x, const mlx::core::array& w, float eps);
// residual add first: s = bf(x + r) (FP32), then rms_norm(s, w, eps) as above -> {s, FP32 norm, FP16 norm}; the decode
// / verify layer's add_bf + norm + FP16 matmul copy in one launch (j1038: ~4 dispatches per residual)
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> rms_norm_bf16_pair_residual(const mlx::core::array& x, const mlx::core::array& r,
                                                                                            const mlx::core::array& w, float eps);
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> rms_norm_bf16_pair_residual_reference(const mlx::core::array& x, const mlx::core::array& r,
                                                                                                      const mlx::core::array& w, float eps);
// SwiGLU of a gate | up product [rows, 2F] FP16: f16(bf(bf(g * sigmoid(g)) * u)) with g, u the BF16 roundings of the
// halves, [rows, F] FP16
mlx::core::array swiglu_pack16(const mlx::core::array& gate_up);
// the MLX op chains above, for the check and the fallback
std::pair<mlx::core::array, mlx::core::array> rms_norm_bf16_pair_reference(const mlx::core::array& x, const mlx::core::array& w, float eps);
mlx::core::array swiglu_pack16_reference(const mlx::core::array& gate_up);

// Prefill attention q/k RMS norm + RoPE, reading the packed q|gate, k and v FP16 products in place and writing the
// layouts full_attention consumes, bit for bit equal to the MLX chain (round_product, bf(rms_norm), transpose, bf(rope)).
// q_prod FP16 [m, heads*2*hd] (q | gate interleaved per head), k_prod / v_prod FP16 [m, kvheads*hd], weights FP32 [hd].
// q: FP32 [1, heads, m, hd]; k, v: FP32 [1, kvheads, m, hd]; gate: FP32 [m, heads, hd]. All BF16-rounded values.
struct AttnQkNormRope {
    mlx::core::array q, k, v, gate;
};
AttnQkNormRope attn_qk_norm_rope(const mlx::core::array& q_prod, const mlx::core::array& k_prod, const mlx::core::array& v_prod,
                                 const mlx::core::array& q_norm_w, const mlx::core::array& k_norm_w, int heads, int kvheads,
                                 int hd, int rot, float theta, float eps, int offset);
AttnQkNormRope attn_qk_norm_rope_reference(const mlx::core::array& q_prod, const mlx::core::array& k_prod,
                                           const mlx::core::array& v_prod, const mlx::core::array& q_norm_w,
                                           const mlx::core::array& k_norm_w, int heads, int kvheads, int hd, int rot,
                                           float theta, float eps, int offset);
}  // namespace tqllm::mlx_backend

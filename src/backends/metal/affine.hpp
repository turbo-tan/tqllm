#pragma once
// MLX's own affine layout (codes [rows, columns * bits / 32] uint32, FP16 scales / biases [rows, columns / group]) on
// MLX's quantized_matmul: what GPUs without the M5 matrix unit run for weights stored in the NAX layouts.
#include "nax_affine.hpp"
#include <mlx/mlx.h>

namespace tqllm::mlx_backend {
struct AffineWeights {mlx::core::array blocks, scales, biases; int group, bits;};
// 4-bit: the stored codes repacked (nax_affine_to_mlx, no requantization). 6-bit: MLX quantizes the stored values again
// (they are affine at this group size, so the result differs from them by at most MLX's choice of scale and bias).
inline AffineWeights affine_from_nax(const NaxAffine& w) {
    if (w.tq4 || w.q6) throw std::invalid_argument("affine_from_nax: TQ4 and Q6 keep their own formats");
    if (w.bits == 4) {
        auto q = nax_affine_to_mlx(w);
        return {q.codes, q.scales, q.biases, q.group, 4};
    }
    auto q = mlx::core::quantize(nax_affine_dequant(w), w.group, w.bits, "affine");
    return {q[0], q[1], q[2], w.group, w.bits};
}
}  // namespace tqllm::mlx_backend

#pragma once
// MLX's own affine layout (codes [rows, columns * bits / 32] uint32, FP16 scales / biases [rows, columns / group]) on
// MLX's quantized_matmul: what GPUs without the M5 matrix unit run for weights stored in the NAX layouts.
#include "nax_affine.hpp"
#include "tq4.hpp"
#include <mlx/mlx.h>

namespace tqllm::mlx_backend {
struct AffineWeights {
    mlx::core::array blocks, scales, biases; int group, bits;
    bool rotated = false;  // repacked TQ4 (affine8_from_tq4): the matrix reads the TQ4-rotated input
};
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
// TQ4 with integer levels (value = FP16 scale * level, the 16 levels +-6 .. +-127 of tq4.cpp: not evenly spaced, so no
// 4-bit affine form) as 8-bit affine codes: code = level + 128, bias = -128 * scale (exact in FP16), one scale per 128
// columns as stored. The same values at twice the bytes, on MLX's quantized_matmul: for GPUs without the M5 matrix unit.
inline AffineWeights affine8_from_tq4(const Tq4Weights& t) {
    namespace mx = mlx::core;
    if (!t.integer_levels || t.codes.ndim() != 2 || t.codes.dtype() != mx::uint8 || t.scales.dtype() != mx::float16 ||
        t.codes.shape(1) % 64 != 0 || t.scales.shape(1) != t.codes.shape(1) / 64)
        throw std::invalid_argument("affine8_from_tq4: needs integer-level TQ4 weights, one scale per 128 columns");
    const int rows = t.codes.shape(0), columns = t.codes.shape(1) * 2;
    static const uint32_t table[16] = {134, 146, 159, 172, 186, 203, 224, 255, 122, 110, 97, 84, 70, 53, 32, 1};
    const auto lut = mx::array(table, {16}, mx::uint32);
    // column 2j is byte j's low nibble, column 2j + 1 its high nibble; four columns per uint32, lowest column in the low byte
    const auto bytes = mx::reshape(mx::astype(t.codes, mx::uint32), {rows, columns / 4, 2});
    const auto lo = mx::take(lut, mx::bitwise_and(bytes, mx::array(15u)), 0), hi = mx::take(lut, mx::right_shift(bytes, mx::array(4u)), 0);
    const uint32_t shifts[4] = {0, 8, 16, 24};
    const auto cols = mx::reshape(mx::stack({lo, hi}, 3), {rows, columns / 4, 4});  // [.., byte, nibble] -> columns in order
    const auto words = mx::sum(mx::left_shift(cols, mx::array(shifts, {4}, mx::uint32)), 2);
    return {mx::astype(words, mx::uint32), t.scales, mx::multiply(t.scales, mx::array(-128.0f, mx::float16)), 128, 8, true};
}
}  // namespace tqllm::mlx_backend

#pragma once
// Per-row top k of [rows, vocab] logits on the device, returned row-contiguous so the host can read row r at r * k.
// (An astype to the same dtype returns its input: the sliced argpartition stayed a strided view with the vocab's row
// stride, and reading it flat gave row 0 right and every later row wrong.)
#include <mlx/ops.h>
#include <mlx/random.h>
#include <cmath>
#include <cstdint>
#include <utility>

namespace tqllm::mlx_backend {
inline std::pair<mlx::core::array, mlx::core::array> row_topk(const mlx::core::array& scaled, int k) {
    namespace mx = mlx::core;
    const int rows = scaled.shape(0);
    auto idx = mx::slice(mx::argpartition(mx::negative(scaled), k - 1, -1), {0, 0}, {rows, k});
    auto val = mx::take_along_axis(scaled, idx, -1);
    return {mx::contiguous(mx::astype(idx, mx::uint32)), mx::contiguous(val)};
}
// row_topk over a wide vocab without sorting whole rows: one max per 128-wide block, the k best blocks hold the row's top
// k (each top-k entry's block max is at least the k-th value, and at most k blocks reach it), then row_topk over those
// k x 128 candidates. Same ids / values as row_topk up to ties.
inline std::pair<mlx::core::array, mlx::core::array> row_topk_blocked(const mlx::core::array& logits, int k) {
    namespace mx = mlx::core;
    constexpr int B = 128;
    const int rows = logits.shape(0), vocab = logits.shape(1), pad = (B - vocab % B) % B, nb = (vocab + pad) / B;
    if (k * B * 4 > vocab) return row_topk(logits, k);
    auto x = pad ? mx::pad(logits, {{0, 0}, {0, pad}}, mx::array(-INFINITY, logits.dtype())) : logits;
    auto blocks = mx::reshape(x, {rows, nb, B});
    auto best = mx::slice(mx::argpartition(mx::negative(mx::max(blocks, -1)), k - 1, -1), {0, 0}, {rows, k});  // [rows, k]
    auto cand = mx::reshape(mx::take_along_axis(blocks, mx::reshape(best, {rows, k, 1}), 1), {rows, k * B});
    auto ids = mx::reshape(mx::add(mx::multiply(mx::reshape(mx::astype(best, mx::uint32), {rows, k, 1}), mx::array(uint32_t(B))),
                                   mx::reshape(mx::arange(B, mx::uint32), {1, 1, B})), {rows, k * B});
    auto [local, val] = row_topk(cand, k);
    return {mx::contiguous(mx::take_along_axis(ids, mx::astype(local, mx::int32), -1)), val};
}
// One token per row of [rows, vocab] logits drawn on the device from the host sampler's distribution
// (filtered_distribution: logits / temperature, the top k (64 when k <= 0), then the smallest best-first prefix holding
// top_p of the mass). A tree walk that keeps the child equal to its parent row's draw outputs exact samples: the draw is
// a sample of the target, and with fixed (point-mass) candidates it keeps a child exactly as often as rejection
// sampling over the siblings would (the children's total probability).
inline mlx::core::array sample_rows(const mlx::core::array& logits, float temperature, int top_k, float top_p, uint64_t seed) {
    namespace mx = mlx::core;
    const int rows = logits.shape(0), k = top_k > 0 ? top_k : 64;
    // the top k of the raw logits (1 / temperature > 0 keeps the order): only the k survivors are scaled in FP32
    auto [idx, raw] = row_topk_blocked(logits, k);
    auto val = mx::multiply(mx::astype(raw, mx::float32), mx::array(1.0f / temperature));
    auto order = mx::argsort(mx::negative(val), -1);
    val = mx::take_along_axis(val, order, -1);
    idx = mx::take_along_axis(idx, order, -1);
    auto p = mx::softmax(val, -1, true);
    auto keep = mx::less(mx::subtract(mx::cumsum(p, -1), p), mx::array(top_p));
    auto masked = mx::where(keep, val, mx::array(-INFINITY));
    auto pick = mx::random::categorical(masked, -1, mx::random::key(seed));
    return mx::reshape(mx::take_along_axis(idx, mx::reshape(mx::astype(pick, mx::int32), {rows, 1}), -1), {rows});
}
}  // namespace tqllm::mlx_backend

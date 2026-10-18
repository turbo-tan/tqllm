#pragma once
// Per-row top k of [rows, vocab] logits on the device, returned row-contiguous so the host can read row r at r * k.
// (An astype to the same dtype returns its input: the sliced argpartition stayed a strided view with the vocab's row
// stride, and reading it flat gave row 0 right and every later row wrong.)
#include <mlx/ops.h>
#include <utility>

namespace tqllm::mlx_backend {
inline std::pair<mlx::core::array, mlx::core::array> row_topk(const mlx::core::array& scaled, int k) {
    namespace mx = mlx::core;
    const int rows = scaled.shape(0);
    auto idx = mx::slice(mx::argpartition(mx::negative(scaled), k - 1, -1), {0, 0}, {rows, k});
    auto val = mx::take_along_axis(scaled, idx, -1);
    return {mx::contiguous(mx::astype(idx, mx::uint32)), mx::contiguous(val)};
}
}  // namespace tqllm::mlx_backend

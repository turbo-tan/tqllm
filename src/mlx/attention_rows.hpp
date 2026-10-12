#pragma once
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
inline mlx::core::array causal_attention_rows(const mlx::core::array& q,
        const mlx::core::array& k, const mlx::core::array& v, float scale,
        bool decode_order) {
    namespace mx = mlx::core;
    const int n = q.shape(2), past = k.shape(2) - n;
    if (past < 0 || v.shape(2) != k.shape(2))
        throw std::invalid_argument("attention cache shorter than query block");
    if (!decode_order || n == 1)
        return mx::fast::scaled_dot_product_attention(q, k, v, scale, n > 1 ? "causal" : "");
    std::vector<mx::array> output;
    output.reserve(n);
    for (int t = 0; t < n; ++t) {
        auto query = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {q.shape(0), q.shape(1), t + 1, q.shape(3)}));
        auto keys = mx::contiguous(mx::slice(k, {0, 0, 0, 0}, {k.shape(0), k.shape(1), past + t + 1, k.shape(3)}));
        auto values = mx::contiguous(mx::slice(v, {0, 0, 0, 0}, {v.shape(0), v.shape(1), past + t + 1, v.shape(3)}));
        output.push_back(mx::fast::scaled_dot_product_attention(query, keys, values, scale, ""));
    }
    return mx::concatenate(output, 2);
}
}

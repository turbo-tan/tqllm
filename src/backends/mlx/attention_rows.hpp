#pragma once
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "attention_decode.hpp"

namespace tqllm::mlx_backend {
// Causal attention of q's rows (the last n of `length` positions) over a BF16 KV buffer whose first `kv_valid`
// positions are filled (kv_valid <= 0: all of it). Up to 16 rows (decode, verify) run the batch-invariant decode-rows
// kernel; prefill chunks run MLX's fused SDPA in FP16 (K / V hold BF16 values, exact in FP16 within its range; j735:
// prefill +0.6-4.1 %, KL below the chunk-64 floor on 8 of 8 windows).
// Prefill attention of q's rows (the last of the keys' positions), FP16 operands. MLX takes its fused kernel at head dim
// 256 only from 1024 query rows (NAX); a shorter prefill slice (the last slice of a prompt, an ANE plan's remainder)
// takes the unfused path, which materialises heads x rows x keys scores: 951 rows x 250,551 keys = 11 GB, the 256K
// request's 44 GB peak (k1179). Such calls run in row chunks under kPrefillScoreBudget bytes of scores, each with the
// full key length and an explicit causal mask, so every row sees the same operations as in the single call (the test
// checks the result bit for bit).
constexpr size_t kPrefillScoreBudget = size_t(512) << 20;
inline mlx::core::array prefill_attention_rows(const mlx::core::array& q16, const mlx::core::array& keys16,
                                               const mlx::core::array& values16, float scale) {
    namespace mx = mlx::core;
    const int heads = q16.shape(1), n = q16.shape(2), length = keys16.shape(2);
    const size_t row_bytes = size_t(heads) * size_t(length) * 2;
    if (n >= 1024 || size_t(n) * row_bytes <= kPrefillScoreBudget)
        return mx::fast::scaled_dot_product_attention(q16, keys16, values16, scale, "causal");
    const int chunk = std::max<int>(1, int(kPrefillScoreBudget / row_bytes));
    std::vector<mx::array> parts;
    auto key_pos = mx::reshape(mx::arange(length, mx::int32), {1, length});
    for (int r0 = 0; r0 < n; r0 += chunk) {
        const int r1 = std::min(n, r0 + chunk);
        auto qc = mx::slice(q16, {0, 0, r0, 0}, {q16.shape(0), heads, r1, q16.shape(3)});
        auto row_pos = mx::reshape(mx::arange(length - n + r0, length - n + r1, mx::int32), {r1 - r0, 1});
        auto mask = mx::greater_equal(row_pos, key_pos);   // the same mask the causal fallback builds, for these rows
        auto out = mx::fast::scaled_dot_product_attention(qc, keys16, values16, scale, "", mask);
        mx::eval(out);   // one chunk's scores alive at a time
        parts.push_back(out);
    }
    return mx::concatenate(parts, 2);
}

inline mlx::core::array causal_attention_rows(const mlx::core::array& q, const mlx::core::array& k,
                                              const mlx::core::array& v, float scale, int kv_valid = -1) {
    namespace mx = mlx::core;
    const int n = q.shape(2), length = kv_valid > 0 ? kv_valid : k.shape(2);
    if (length < n || v.shape(2) != k.shape(2) || length > k.shape(2) || k.dtype() != mx::bfloat16 || v.dtype() != mx::bfloat16)
        throw std::invalid_argument("attention needs a BF16 KV buffer at least as long as the query block");
    if (n <= 16) return decode_rows_attention(q, k, v, scale, length);
    auto keys = mx::astype(mx::slice(k, {0, 0, 0, 0}, {k.shape(0), k.shape(1), length, k.shape(3)}), mx::float16);
    auto values = mx::astype(mx::slice(v, {0, 0, 0, 0}, {v.shape(0), v.shape(1), length, v.shape(3)}), mx::float16);
    auto out = prefill_attention_rows(mx::astype(q, mx::float16), keys, values, scale);
    return mx::astype(out, q.dtype());
}
}  // namespace tqllm::mlx_backend

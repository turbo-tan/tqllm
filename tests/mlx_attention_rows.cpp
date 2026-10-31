// causal_attention_rows: a batch of n query rows over a BF16 KV buffer gives each row the same result, after the BF16
// rounding the engine applies, as that row alone over its own prefix (the spec == AR property of verify attention).
#include "attention_rows.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
mx::array fixture(int heads, int rows, int dim, int seed) {
    std::vector<float> data(size_t(heads) * rows * dim);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = std::sin(float((i * 17 + size_t(seed)) % 1009) * .031f);
    return mx::astype(mx::array(data.data(), {1, heads, rows, dim}), mx::bfloat16);
}
mx::array bf(const mx::array& x) {
    return mx::astype(mx::astype(x, mx::bfloat16), mx::float32);
}
int main() {
    try {
        size_t changed = 0;
        for (int past : {0, 17, 257, 1024, 4097}) for (int n : {1, 2, 4, 8, 16}) {
            const int dim = 256;
            auto q = mx::astype(fixture(8, n, dim, 23), mx::float32);
            auto k = fixture(2, past + n, dim, 41), v = fixture(2, past + n, dim, 79);
            const float scale = 1.f / std::sqrt(float(dim));
            std::vector<mx::array> rows;
            for (int t = 0; t < n; ++t) {
                auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, 8, t + 1, dim}));
                rows.push_back(tqllm::mlx_backend::causal_attention_rows(qt, k, v, scale, past + t + 1));
            }
            auto ref = bf(mx::concatenate(rows, 2));
            auto out = bf(tqllm::mlx_backend::causal_attention_rows(q, k, v, scale, past + n));
            mx::eval(ref, out);
            size_t different = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                if (!std::isfinite(ref.data<float>()[i]) || !std::isfinite(out.data<float>()[i]))
                    throw std::runtime_error("nonfinite attention result");
                different += ref.data<float>()[i] != out.data<float>()[i];
            }
            changed += different;
            std::cout << "ATTENTION_ROW_CHECK past=" << past << " width=" << n << " changed=" << different << '\n';
        }
        if (changed) throw std::runtime_error("batched attention differs from each row alone");
        std::cout << "ATTENTION_ROWS_PASS\n";
    } catch (const std::exception& e) { std::cerr << "ATTENTION_ROWS_ERROR " << e.what() << '\n'; return 1; }
}

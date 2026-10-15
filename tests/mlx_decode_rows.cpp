#include "attention_decode.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
mx::array random_array(std::mt19937& rng, mx::Shape shape, float scale) {
    size_t n = 1;
    for (int d : shape) n *= size_t(d);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return mx::array(v.data(), shape, mx::float32);
}
bool run(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    auto q = random_array(rng, {1, heads, n, dim}, 1.0f);
    auto k = random_array(rng, {1, kvheads, length, dim}, 1.0f);
    auto v = random_array(rng, {1, kvheads, length, dim}, 1.0f);
    const float scale = 1.0f / std::sqrt(float(dim));
    auto y = mx::contiguous(decode_rows_attention(q, k, v, scale));
    auto ref = mx::fast::scaled_dot_product_attention(q, k, v, scale, n > 1 ? "causal" : "");
    ref = mx::contiguous(ref);
    mx::eval(y, ref);
    const float* a = y.data<float>();
    const float* b = ref.data<float>();
    double worst = 0, peak = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        worst = std::max(worst, double(std::fabs(a[i] - b[i])));
        peak = std::max(peak, double(std::fabs(b[i])));
    }
    bool ok = worst / peak < 2e-5;
    // batch invariance: row t of the n-row call equals the 1-row call over its visible keys, bit for bit
    bool exact = true;
    for (int t = 0; t < n && exact; ++t) {
        const int vis = length - n + t + 1;
        auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        auto kt = mx::contiguous(mx::slice(k, {0, 0, 0, 0}, {1, kvheads, vis, dim}));
        auto vt = mx::contiguous(mx::slice(v, {0, 0, 0, 0}, {1, kvheads, vis, dim}));
        auto one = mx::contiguous(decode_rows_attention(qt, kt, vt, scale));
        auto full = mx::contiguous(mx::slice(y, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        mx::eval(one, full);
        exact = std::memcmp(one.data<float>(), full.data<float>(), one.size() * sizeof(float)) == 0;
    }
    // a KV buffer longer than the filled length (in-place cache) gives the same bits as the exact-length arrays
    {
        const int cap = length + 77;
        auto kb = mx::concatenate({k, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto vb = mx::concatenate({v, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto yb = mx::contiguous(decode_rows_attention(q, kb, vb, scale, length));
        mx::eval(yb);
        exact = exact && std::memcmp(yb.data<float>(), y.data<float>(), y.size() * sizeof(float)) == 0;
    }
    std::printf("DECODE_ROWS heads=%d kv=%d dim=%d length=%d n=%d rel=%.2e batch_invariant=%d %s\n", heads, kvheads, dim,
                length, n, worst / peak, int(exact), ok && exact ? "PASS" : "FAIL");
    return ok && exact;
}
}  // namespace

int main() {
    std::mt19937 rng(7);
    bool ok = true;
    for (int length : {8, 63, 64, 65, 300, 1900})
        for (int n : {1, 3, 8})
            if (length >= n) ok &= run(24, 4, 256, length, n, rng);
    ok &= run(8, 2, 128, 200, 8, rng);
    std::printf(ok ? "DECODE_ROWS_PASS\n" : "DECODE_ROWS_FAIL\n");
    return ok ? 0 : 1;
}

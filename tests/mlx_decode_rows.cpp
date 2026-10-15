#include "attention_decode.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
int version = 1;
mx::array attend(const mx::array& q, const mx::array& k, const mx::array& v, float scale, int valid = -1) {
    return version == 3 ? decode_rows_attention_v3(q, k, v, scale, valid) : version == 2 ? decode_rows_attention_v2(q, k, v, scale, valid)
                        : decode_rows_attention_v1(q, k, v, scale, valid);
}
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
    auto y = mx::contiguous(attend(q, k, v, scale));
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
        auto one = mx::contiguous(attend(qt, kt, vt, scale));
        auto full = mx::contiguous(mx::slice(y, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        mx::eval(one, full);
        exact = std::memcmp(one.data<float>(), full.data<float>(), one.size() * sizeof(float)) == 0;
    }
    // a KV buffer longer than the filled length (in-place cache) gives the same bits as the exact-length arrays
    {
        const int cap = length + 77;
        auto kb = mx::concatenate({k, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto vb = mx::concatenate({v, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto yb = mx::contiguous(attend(q, kb, vb, scale, length));
        mx::eval(yb);
        exact = exact && std::memcmp(yb.data<float>(), y.data<float>(), y.size() * sizeof(float)) == 0;
        // BF16 storage of BF16-rounded keys / values gives the same bits as their FP32 copies
        auto kr = mx::astype(mx::astype(kb, mx::bfloat16), mx::float32), vr = mx::astype(mx::astype(vb, mx::bfloat16), mx::float32);
        auto y32 = mx::contiguous(attend(q, kr, vr, scale, length));
        auto y16 = mx::contiguous(attend(q, mx::astype(kb, mx::bfloat16), mx::astype(vb, mx::bfloat16), scale, length));
        mx::eval(y32, y16);
        exact = exact && std::memcmp(y32.data<float>(), y16.data<float>(), y16.size() * sizeof(float)) == 0;
    }
    std::printf("DECODE_ROWS v%d heads=%d kv=%d dim=%d length=%d n=%d rel=%.2e batch_invariant=%d %s\n", version, heads, kvheads, dim,
                length, n, worst / peak, int(exact), ok && exact ? "PASS" : "FAIL");
    return ok && exact;
}

// microseconds per call of the v1 and v2 kernels on a BF16 KV buffer of `length` keys
void bench(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    auto q = random_array(rng, {1, heads, n, dim}, 1.0f);
    auto k = mx::astype(random_array(rng, {1, kvheads, length, dim}, 1.0f), mx::bfloat16);
    auto v = mx::astype(random_array(rng, {1, kvheads, length, dim}, 1.0f), mx::bfloat16);
    mx::eval(q, k, v);
    const float scale = 1.0f / std::sqrt(float(dim));
    for (int ver : {1, 3}) {
        auto call = [&] { return ver == 1 ? decode_rows_attention_v1(q, k, v, scale) : decode_rows_attention_v3(q, k, v, scale); };
        mx::eval(call());
        const int reps = 20;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) mx::eval(call());
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / reps;
        std::printf("DECODE_ROWS_BENCH v%d length=%d n=%d %.1f us\n", ver, length, n, us);
    }
}
}  // namespace

int main() {
    std::mt19937 rng(7);
    bool ok = true;
    for (int ver : {1, 3}) {
        version = ver;
        for (int length : {8, 63, 64, 65, 255, 256, 257, 300, 511, 512, 513, 1900, 4100})
            for (int n : {1, 3, 8})
                if (length >= n) ok &= run(24, 4, 256, length, n, rng);
        ok &= run(8, 2, 128, 200, 8, rng);
        ok &= run(8, 2, 128, 3000, 1, rng);
        ok &= run(24, 4, 256, 16384, 8, rng);
    }
    for (int length : {300, 2048, 16384})
        for (int n : {1, 8}) bench(24, 4, 256, length, n, rng);
    std::printf(ok ? "DECODE_ROWS_PASS\n" : "DECODE_ROWS_FAIL\n");
    return ok ? 0 : 1;
}

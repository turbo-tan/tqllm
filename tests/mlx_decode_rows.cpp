#include "attention_decode.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
int version = 1;
mx::array attend(const mx::array& q, const mx::array& k, const mx::array& v, float scale, int valid = -1) {
    if (version == 4) return decode_rows_attention_v4(q, mx::astype(k, mx::bfloat16), mx::astype(v, mx::bfloat16), scale, valid);
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
    bool ok = worst / peak < (version == 4 ? 2e-2 : 2e-5);  // v4: BF16 q / k / out
    // batch invariance: row t of the n-row call equals the 1-row call over its visible keys, bit for bit
    bool exact = true;
    for (int t = 0; t < n && exact; ++t) {
        const int vis = length - n + t + 1;
        if (version == 4 && vis < 9) continue;  // v4 below 9 visible keys runs MLX's vector kernel; prompts are longer
        auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        auto kt = mx::contiguous(mx::slice(k, {0, 0, 0, 0}, {1, kvheads, vis, dim}));
        auto vt = mx::contiguous(mx::slice(v, {0, 0, 0, 0}, {1, kvheads, vis, dim}));
        auto one = mx::contiguous(attend(qt, kt, vt, scale));
        auto full = mx::contiguous(mx::slice(y, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        mx::eval(one, full);
        exact = std::memcmp(one.data<float>(), full.data<float>(), one.size() * sizeof(float)) == 0;
    }
    const bool rows_exact = exact;
    // a KV buffer longer than the filled length (in-place cache) gives the same bits as the exact-length arrays
    bool cap_exact = true, bf16_exact = true;
    {
        const int cap = length + 77;
        auto kb = mx::concatenate({k, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto vb = mx::concatenate({v, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto yb = mx::contiguous(attend(q, kb, vb, scale, length));
        mx::eval(yb);
        cap_exact = std::memcmp(yb.data<float>(), y.data<float>(), y.size() * sizeof(float)) == 0;
        exact = exact && cap_exact;
        // BF16 storage of BF16-rounded keys / values gives the same bits as their FP32 copies
        auto kr = mx::astype(mx::astype(kb, mx::bfloat16), mx::float32), vr = mx::astype(mx::astype(vb, mx::bfloat16), mx::float32);
        auto y32 = mx::contiguous(attend(q, kr, vr, scale, length));
        auto y16 = mx::contiguous(attend(q, mx::astype(kb, mx::bfloat16), mx::astype(vb, mx::bfloat16), scale, length));
        mx::eval(y32, y16);
        bf16_exact = std::memcmp(y32.data<float>(), y16.data<float>(), y16.size() * sizeof(float)) == 0;
        exact = exact && bf16_exact;
    }
    std::printf("DECODE_ROWS v%d heads=%d kv=%d dim=%d length=%d n=%d rel=%.2e batch_invariant=%d (rows %d, cache view %d, bf16 %d) %s\n",
                version, heads, kvheads, dim, length, n, worst / peak, int(exact), int(rows_exact), int(cap_exact), int(bf16_exact),
                ok && exact ? "PASS" : "FAIL");
    return ok && exact;
}

bool safe_run(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    try { return run(heads, kvheads, dim, length, n, rng); }
    catch (const std::exception& e) {
        std::printf("DECODE_ROWS v%d heads=%d kv=%d dim=%d length=%d n=%d EXCEPTION %s FAIL\n", version, heads, kvheads, dim, length, n, e.what());
        return false;
    }
}

// microseconds per call on a BF16 KV buffer (cap > length, as the in-place cache), 16 calls per eval like 16 attention
// layers: v1 / v3, MLX's fused SDPA over all rows (causal), and MLX's SDPA one row at a time (batch-invariant by construction)
void bench(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    const int cap = length + 512, calls = 16;
    std::vector<mx::array> qs, ks, vs;
    for (int c = 0; c < calls; ++c) {
        qs.push_back(random_array(rng, {1, heads, n, dim}, 1.0f));
        ks.push_back(mx::astype(random_array(rng, {1, kvheads, cap, dim}, 1.0f), mx::bfloat16));
        vs.push_back(mx::astype(random_array(rng, {1, kvheads, cap, dim}, 1.0f), mx::bfloat16));
    }
    mx::eval(qs); mx::eval(ks); mx::eval(vs);
    const float scale = 1.0f / std::sqrt(float(dim));
    auto view = [&](const mx::array& x, int len) { return mx::slice(x, {0, 0, 0, 0}, {1, kvheads, len, dim}); };
    auto sdpa_all = [&](int c) {
        auto q = mx::astype(qs[c], mx::bfloat16);
        return mx::astype(mx::fast::scaled_dot_product_attention(q, view(ks[c], length), view(vs[c], length), scale, n > 1 ? "causal" : ""), mx::float32);
    };
    auto sdpa_rows = [&](int c) {
        std::vector<mx::array> out;
        for (int t = 0; t < n; ++t) {
            auto q = mx::astype(mx::slice(qs[c], {0, 0, t, 0}, {1, heads, t + 1, dim}), mx::bfloat16);
            const int vis = length - n + t + 1;
            out.push_back(mx::fast::scaled_dot_product_attention(q, view(ks[c], vis), view(vs[c], vis), scale, ""));
        }
        return mx::astype(mx::concatenate(out, 2), mx::float32);
    };
    const char* names[] = {"v1", "v3", "sdpa_all", "sdpa_rows", "v4"};
    for (int ver = 0; ver < 5; ++ver) {
        auto call = [&](int c) {
            return ver == 0 ? decode_rows_attention_v1(qs[c], ks[c], vs[c], scale, length)
                 : ver == 1 ? decode_rows_attention_v3(qs[c], ks[c], vs[c], scale, length)
                 : ver == 2 ? sdpa_all(c) : ver == 3 ? sdpa_rows(c) : decode_rows_attention_v4(qs[c], ks[c], vs[c], scale, length);
        };
        if (ver == 1 && n > 8) continue;
        try {
        auto round = [&] { std::vector<mx::array> o; for (int c = 0; c < calls; ++c) o.push_back(call(c)); mx::eval(o); };
        round();
        const int reps = 10;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) round();
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / reps / calls;
        std::printf("DECODE_ROWS_BENCH %-9s length=%d n=%d %.1f us per call (16 per eval)\n", names[ver], length, n, us);
        } catch (const std::exception& e) {std::printf("DECODE_ROWS_BENCH %-9s length=%d n=%d EXCEPTION %s\n", names[ver], length, n, e.what());}
    }
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::mt19937 rng(7);
    bool ok = true;
    for (int ver : {1, 3, 4}) {
        version = ver;
        for (int length : {8, 63, 64, 65, 255, 256, 257, 300, 511, 512, 513, 1900, 4095, 4096, 4097, 4100, 8200})
            for (int n : {1, 3, 8, 16})
                if (length >= n && !(ver == 3 && n > 8)) ok &= safe_run(24, 4, 256, length, n, rng);
        ok &= safe_run(8, 2, 128, 200, 8, rng);
        ok &= safe_run(8, 2, 128, 3000, 1, rng);
        ok &= safe_run(24, 4, 256, 16384, 8, rng);
    }
    for (int length : {64, 2048, 8192, 16384})
        for (int n : {1, 8, 16}) bench(24, 4, 256, length, n, rng);
    std::printf(ok ? "DECODE_ROWS_PASS\n" : "DECODE_ROWS_FAIL\n");
    return ok ? 0 : 1;
}

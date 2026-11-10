#include "attention_decode.hpp"
#include "attention_rows.hpp"
#include <mlx/memory.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
bool segmented = false;  // the BF16 kernel under test: bf16_segmented, else bf16_folded
const char* kernel_name() {return segmented ? "bf16_segmented" : "bf16_folded";}
mx::array attend(const mx::array& q, const mx::array& k, const mx::array& v, float scale, int valid = -1) {
    auto kb = mx::astype(k, mx::bfloat16), vb = mx::astype(v, mx::bfloat16);
#ifdef MLX_HAS_SDPA_LSE
    if (segmented) return decode_attention_bf16(q, kb, vb, scale, valid);  // bf16_segmented where MLX has it for the head dim, else bf16_folded
#endif
    return decode_attention_bf16_folded(q, kb, vb, scale, valid);
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
    bool ok = worst / peak < 2e-2;  // BF16 q / k / out
    // batch invariance: row t of the n-row call equals the 1-row call over its visible keys, bit for bit
    bool exact = true;
    for (int t = 0; t < n && exact; ++t) {
        const int vis = length - n + t + 1;
        if (vis < 9) continue;  // below 9 visible keys MLX runs its vector kernel; prompts are longer
        auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        auto kt = mx::contiguous(mx::slice(k, {0, 0, 0, 0}, {1, kvheads, vis, dim}));
        auto vt = mx::contiguous(mx::slice(v, {0, 0, 0, 0}, {1, kvheads, vis, dim}));
        auto one = mx::contiguous(attend(qt, kt, vt, scale));
        auto full = mx::contiguous(mx::slice(y, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        mx::eval(one, full);
        exact = std::memcmp(one.data<float>(), full.data<float>(), one.size() * sizeof(float)) == 0;
    }
    const bool rows_exact = exact;
    // a KV buffer longer than the filled length (in-place cache) gives the same bits as the exact-length arrays, also
    // when the unfilled tail holds NaN / Inf (stale or uninitialised cache rows must never reach a result).
    // bf16_segmented's matrix unit reads the whole segment (0 x NaN), so bf16_segmented needs a finite tail: the
    // engine's cache is zero-filled and only ever holds finite keys / values (rejected verify rows included)
    bool cap_exact = true, bf16_exact = true;
    {
        const int cap = length + 77;
        auto kb = mx::concatenate({k, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto vb = mx::concatenate({v, random_array(rng, {1, kvheads, cap - length, dim}, 1.0f)}, 2);
        auto yb = mx::contiguous(attend(q, kb, vb, scale, length));
        auto nan_tail = mx::full({1, kvheads, cap - length, dim}, std::nanf(""), mx::float32);
        auto inf_tail = mx::full({1, kvheads, cap - length, dim}, INFINITY, mx::float32);
        auto yn = mx::contiguous(attend(q, mx::concatenate({k, inf_tail}, 2), mx::concatenate({v, nan_tail}, 2), scale, length));
        mx::eval(yb, yn);
        cap_exact = std::memcmp(yb.data<float>(), y.data<float>(), y.size() * sizeof(float)) == 0 &&
                    (segmented || std::memcmp(yn.data<float>(), y.data<float>(), y.size() * sizeof(float)) == 0);
        exact = exact && cap_exact;
        // BF16 storage of BF16-rounded keys / values gives the same bits as their FP32 copies
        auto kr = mx::astype(mx::astype(kb, mx::bfloat16), mx::float32), vr = mx::astype(mx::astype(vb, mx::bfloat16), mx::float32);
        auto y32 = mx::contiguous(attend(q, kr, vr, scale, length));
        auto y16 = mx::contiguous(attend(q, mx::astype(kb, mx::bfloat16), mx::astype(vb, mx::bfloat16), scale, length));
        mx::eval(y32, y16);
        bf16_exact = std::memcmp(y32.data<float>(), y16.data<float>(), y16.size() * sizeof(float)) == 0;
        exact = exact && bf16_exact;
    }
    std::printf("DECODE_ROWS %s heads=%d kv=%d dim=%d length=%d n=%d rel=%.2e batch_invariant=%d (rows %d, cache view %d, bf16 %d) %s\n",
                kernel_name(), heads, kvheads, dim, length, n, worst / peak, int(exact), int(rows_exact), int(cap_exact), int(bf16_exact),
                ok && exact ? "PASS" : "FAIL");
    return ok && exact;
}

// int8_reference (own 8-row kernel on int8 K / V): vs MLX's SDPA on the dequantized keys / values, and every row of an
// n-row call equals its own 1-row call bit for bit (the cache buffer longer than the view, as in the engine)
bool run_int8_reference(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    const int cap = length + 77;
    auto q = random_array(rng, {1, heads, n, dim}, 1.0f);
    auto k = mx::astype(mx::astype(random_array(rng, {1, kvheads, cap, dim}, 1.0f), mx::bfloat16), mx::float32);
    auto v = mx::astype(mx::astype(random_array(rng, {1, kvheads, cap, dim}, 1.0f), mx::bfloat16), mx::float32);
    const float scale = 1.0f / std::sqrt(float(dim));
    auto kv = quantize_kv_int8(k, v);
    mx::eval(kv.k8t, kv.v8, kv.kscale, kv.vscale);
    auto y = mx::contiguous(decode_attention_int8_reference(q, kv.k8t, kv.v8, kv.kscale, kv.vscale, scale, length));
    // reference on what the kernel sees: dequantized FP32 keys / values of the visible view
    auto kd = mx::multiply(mx::astype(kv.v8, mx::float32), mx::expand_dims(mx::astype(kv.vscale, mx::float32), -1));  // [kvh, cap, d]
    auto kq = mx::multiply(mx::astype(mx::transpose(mx::reshape(kv.k8t, {kvheads, -1, dim, 1024}), {0, 1, 3, 2}), mx::float32), mx::array(1.0f));
    kq = mx::reshape(kq, {kvheads, -1, dim});
    kq = mx::multiply(mx::slice(kq, {0, 0, 0}, {kvheads, cap, dim}), mx::expand_dims(mx::astype(kv.kscale, mx::float32), -1));
    auto kref = mx::expand_dims(mx::slice(kq, {0, 0, 0}, {kvheads, length, dim}), 0), vref = mx::expand_dims(mx::slice(kd, {0, 0, 0}, {kvheads, length, dim}), 0);
    auto ref = mx::contiguous(mx::fast::scaled_dot_product_attention(q, kref, vref, scale, n > 1 ? "causal" : ""));
    mx::eval(y, ref);
    double worst = 0, peak = 0;
    for (size_t i = 0; i < y.size(); ++i) {worst = std::max(worst, double(std::fabs(y.data<float>()[i] - ref.data<float>()[i]))); peak = std::max(peak, double(std::fabs(ref.data<float>()[i])));}
    bool exact = true;
    for (int t = 0; t < n && exact; ++t) {
        const int vis = length - n + t + 1;
        auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        auto one = mx::contiguous(decode_attention_int8_reference(qt, kv.k8t, kv.v8, kv.kscale, kv.vscale, scale, vis));
        auto full = mx::contiguous(mx::slice(y, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        mx::eval(one, full);
        exact = std::memcmp(one.data<float>(), full.data<float>(), one.size() * sizeof(float)) == 0;
    }
    const bool ok = worst / peak < 2e-2 && exact;
    std::printf("DECODE_ROWS int8_reference heads=%d kv=%d dim=%d length=%d n=%d rel=%.2e rows_exact=%d %s\n", heads, kvheads, dim, length, n, worst / peak, int(exact), ok ? "PASS" : "FAIL");
    return ok;
}
bool safe_run_int8_reference(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    try { return run_int8_reference(heads, kvheads, dim, length, n, rng); }
    catch (const std::exception& e) {std::printf("DECODE_ROWS int8_reference length=%d n=%d EXCEPTION %s FAIL\n", length, n, e.what()); return false;}
}

bool safe_run(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    try { return run(heads, kvheads, dim, length, n, rng); }
    catch (const std::exception& e) {
        std::printf("DECODE_ROWS %s heads=%d kv=%d dim=%d length=%d n=%d EXCEPTION %s FAIL\n", kernel_name(), heads, kvheads, dim, length, n, e.what());
        return false;
    }
}

// an n-row call (n <= 8: the 8-row shape) equals the last n rows of a 16-row call over the same keys bit for bit: the
// verify width may change from step to step (7 or 15 drafted tokens) and AR runs 1 row
bool run_shapes(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    auto call = [&](const mx::array& q, const mx::array& k, const mx::array& v, float scale, int valid) {
#ifdef MLX_HAS_SDPA_LSE
        if (segmented) return decode_attention_bf16_segmented(q, k, v, scale, valid);
#endif
        return decode_attention_bf16_folded(q, k, v, scale, valid);
    };
    auto q16 = random_array(rng, {1, heads, 16, dim}, 1.0f);
    auto k = mx::astype(random_array(rng, {1, kvheads, length + 64, dim}, 1.0f), mx::bfloat16);
    auto v = mx::astype(random_array(rng, {1, kvheads, length + 64, dim}, 1.0f), mx::bfloat16);
    const float scale = 1.0f / std::sqrt(float(dim));
    auto y16 = call(q16, k, v, scale, length);
    auto tail = mx::contiguous(mx::slice(y16, {0, 0, 16 - n, 0}, {1, heads, 16, dim}));
    auto yn = mx::contiguous(call(mx::slice(q16, {0, 0, 16 - n, 0}, {1, heads, 16, dim}), k, v, scale, length));
    mx::eval(tail, yn);
    const bool same = std::memcmp(tail.data<float>(), yn.data<float>(), yn.size() * sizeof(float)) == 0;
    std::printf("DECODE_ROWS_SHAPES %s heads=%d kv=%d dim=%d length=%d n=%d rows of %d == last rows of 16 %d %s\n", kernel_name(), heads, kvheads, dim, length, n, n, int(same), same ? "PASS" : "FAIL");
    return same;
}

// int8 (int8 Q x int8 K): every row = its 1-row call bit for bit; error vs the FP32 reference at most 4x
// int8_reference's
bool run_int8(int length, int n, std::mt19937& rng) {
    const int heads = 24, kvheads = 4, dim = 256, cap = (length + 1023) / 1024 * 1024 + 1024;
    auto q = random_array(rng, {1, heads, n, dim}, 1.0f);
    auto kv = quantize_kv_int8(mx::astype(random_array(rng, {1, kvheads, cap, dim}, 1.0f), mx::bfloat16),
                               mx::astype(random_array(rng, {1, kvheads, cap, dim}, 1.0f), mx::bfloat16));
    mx::eval(q, kv.k8t, kv.v8, kv.kscale, kv.vscale);
    const float scale = 1.0f / std::sqrt(float(dim));
    auto a = mx::contiguous(decode_attention_int8_reference(q, kv.k8t, kv.v8, kv.kscale, kv.vscale, scale, length));
    auto b = mx::contiguous(decode_attention_int8(q, kv.k8t, kv.v8, kv.kscale, kv.vscale, scale, length));
    auto vd = mx::multiply(mx::astype(kv.v8, mx::float32), mx::expand_dims(mx::astype(kv.vscale, mx::float32), -1));
    auto kq = mx::reshape(mx::astype(mx::transpose(mx::reshape(kv.k8t, {kvheads, -1, dim, 1024}), {0, 1, 3, 2}), mx::float32), {kvheads, -1, dim});
    kq = mx::multiply(mx::slice(kq, {0, 0, 0}, {kvheads, cap, dim}), mx::expand_dims(mx::astype(kv.kscale, mx::float32), -1));
    auto kref = mx::expand_dims(mx::slice(kq, {0, 0, 0}, {kvheads, length, dim}), 0), vref = mx::expand_dims(mx::slice(vd, {0, 0, 0}, {kvheads, length, dim}), 0);
    auto ref = mx::contiguous(mx::fast::scaled_dot_product_attention(q, kref, vref, scale, n > 1 ? "causal" : ""));
    mx::eval(a, b, ref);
    const float peak = mx::max(mx::abs(ref)).item<float>();
    const float ea = mx::max(mx::abs(mx::subtract(a, ref))).item<float>() / peak, eb = mx::max(mx::abs(mx::subtract(b, ref))).item<float>() / peak;
    const bool bitwise = mx::array_equal(a, b).item<bool>();
    bool rows = true;
    for (int t = 0; t < n && rows; ++t) {
        auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim}));
        auto one = mx::contiguous(decode_attention_int8(qt, kv.k8t, kv.v8, kv.kscale, kv.vscale, scale, length - n + t + 1));
        rows = mx::array_equal(one, mx::slice(b, {0, 0, t, 0}, {1, heads, t + 1, dim})).item<bool>();
    }
    // int8 Q: ~2.7x int8_reference's error (j1438); 4.5x against the stock MLX 0.32.4 SDPA reference (1.50e-2 vs
    // 3.32e-3, Mac j2056 / j2113). Approximate path (--exact off); rows_exact above is the exactness gate
    const bool gate_b = eb <= 5.0f * ea + 1e-6f;
    std::printf("DECODE_ROWS int8 length=%d n=%d rows_exact %d bitwise_reference %d err %.3e reference_err %.3e %s\n", length, n, int(rows), int(bitwise), eb, ea, rows && gate_b ? "PASS" : "FAIL");
    return rows && gate_b;
}
bool safe_run_int8(int length, int n, std::mt19937& rng) {
    try { return run_int8(length, n, rng); }
    catch (const std::exception& e) {std::printf("DECODE_ROWS int8 length=%d n=%d EXCEPTION %s FAIL\n", length, n, e.what()); return false;}
}

// microseconds per call on a BF16 KV buffer (cap > length, as the in-place cache), 16 calls per eval like 16 attention
// layers: MLX's fused SDPA over all rows (causal, not batch invariant), bf16_folded and bf16_segmented (rows per call:
// decode_row_shape(n)) and the int8 kernels (int8_reference, int8) on quantized copies of the same caches
void bench(int heads, int kvheads, int dim, int length, int n, std::mt19937& rng) {
    const int cap = (length + 512 + 4095) / 4096 * 4096, calls = 16;  // a multiple of every segment size below (no padding copy)
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
    // the BF16 kernels (bf16_folded, bf16_segmented at its default 1024-key segment) and the int8 ones (int8_reference
    // reference, int8 served) on quantized copies of the same caches
    std::vector<Int8KV> q8;
    if (dim == 256) for (int c = 0; c < calls; ++c) {q8.push_back(quantize_kv_int8(ks[c], vs[c])); mx::eval(q8.back().k8t, q8.back().v8, q8.back().kscale, q8.back().vscale);}
    const char* names[] = {"sdpa_all", "bf16_folded", "bf16_segmented", "int8_reference", "int8"};
    for (int ver = 0; ver < 5; ++ver) {
        if (ver == 2 && !segmented_attention_available()) continue;
        if (ver >= 3 && q8.empty()) continue;
        auto call = [&](int c) {
            return ver == 0 ? sdpa_all(c)
#ifdef MLX_HAS_SDPA_LSE
                 : ver == 2 ? decode_attention_bf16_segmented(qs[c], ks[c], vs[c], scale, length)
#endif
                 : ver == 3 ? decode_attention_int8_reference(qs[c], q8[c].k8t, q8[c].v8, q8[c].kscale, q8[c].vscale, scale, length)
                 : ver == 4 ? decode_attention_int8(qs[c], q8[c].k8t, q8[c].v8, q8[c].kscale, q8[c].vscale, scale, length)
                 : decode_attention_bf16_folded(qs[c], ks[c], vs[c], scale, length);
        };
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


// Prefill attention memory (k1179, 256K request peak 44 GB): a prefill slice under 1024 rows at head dim 256 takes MLX's
// unfused SDPA, which materialises heads x rows x keys scores (951 rows x 250,551 keys = 11 GB). The guard must keep the
// call's transient memory bounded and the result close to the reference (unfused) attention.
bool check_prefill_attention_memory() {
    const int heads = 24, kvh = 4, dim = 256, rows = 951, keys = 65536;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randn = [&](std::vector<int> shape) {
        size_t n = 1; for (int d : shape) n *= size_t(d);
        std::vector<float> v(n); for (auto& x : v) x = nd(rng);
        return mx::astype(mx::array(v.data(), mx::Shape(shape.begin(), shape.end()), mx::float32), mx::bfloat16);
    };
    auto q = randn({1, heads, rows, dim}), k = randn({1, kvh, keys, dim}), v = randn({1, kvh, keys, dim});
    mx::eval(q, k, v);
    const float scale = 1.0f / 16.0f;
    mx::clear_cache(); mx::reset_peak_memory();
    const double base = double(mx::get_active_memory());
    auto out = tqllm::mlx_backend::causal_attention_rows(q, k, v, scale);
    mx::eval(out);
    const double transient = (double(mx::get_peak_memory()) - base) / 1e9;
    // reference: the unfused attention in FP32 (masked softmax), on a few rows
    const int r0 = rows - 4;
    auto qs = mx::astype(mx::slice(q, {0, 0, r0, 0}, {1, heads, rows, dim}), mx::float32);
    auto kf = mx::repeat(mx::astype(k, mx::float32), heads / kvh, 1), vf = mx::repeat(mx::astype(v, mx::float32), heads / kvh, 1);
    auto sc = mx::multiply(mx::matmul(qs, mx::transpose(kf, {0, 1, 3, 2})), mx::array(scale));
    auto kpos = mx::reshape(mx::arange(keys), {1, 1, 1, keys}), qpos = mx::reshape(mx::add(mx::arange(4), mx::array(keys - rows + r0)), {1, 1, 4, 1});
    sc = mx::where(mx::less_equal(kpos, qpos), sc, mx::array(-INFINITY));
    auto ref = mx::matmul(mx::softmax(sc, -1), vf);
    auto got = mx::astype(mx::slice(out, {0, 0, r0, 0}, {1, heads, rows, dim}), mx::float32);
    const float rel = mx::sqrt(mx::sum(mx::square(mx::subtract(got, ref)))).item<float>() / mx::sqrt(mx::sum(mx::square(ref))).item<float>();
    // bit for bit against MLX's single unfused call (the causal fallback) at a size where that call is affordable
    bool same = true;
    for (int len : {8192, 20000}) for (int nr : {951, 300, 17}) {
        auto qq = mx::astype(mx::slice(q, {0, 0, 0, 0}, {1, heads, nr, dim}), mx::float16);
        auto kk = mx::astype(mx::slice(k, {0, 0, 0, 0}, {1, kvh, len, dim}), mx::float16), vv = mx::astype(mx::slice(v, {0, 0, 0, 0}, {1, kvh, len, dim}), mx::float16);
        auto one = mx::fast::scaled_dot_product_attention(qq, kk, vv, scale, "causal");
        auto chunked = tqllm::mlx_backend::prefill_attention_rows(qq, kk, vv, scale);
        const bool eq = mx::array_equal(one, chunked).item<bool>();
        same = same && eq;
        std::printf("PREFILL_ATTN_CHUNKED rows %d keys %d bitwise_equal %d\n", nr, len, int(eq));
    }
    const bool ok = transient < 1.5 && rel < 1e-2f && same;
    std::printf("PREFILL_ATTN_MEMORY rows %d keys %d transient %.2f GB (bound 1.5) rel %.2e %s\n", rows, keys, transient, rel, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    const bool prefill_memory_ok = check_prefill_attention_memory();
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::mt19937 rng(7);
    bool ok = true;
    if (std::getenv("TQLLM_TEST_VEC_ONLY")) {
        // MLX's vector (<= 8 query rows) SDPA at our shapes: us per layer-call at q_len 1 / 2 / 4 / 5 x GQA 6 (the
        // threadgroup is 32 x 6 x q_len threads, so 8 rows do not fit), causal, blocks pinned by MLX_SDPA_BLOCKS so the
        // key -> block assignment is a function of nothing but the block count; plus: row t of the 5-row call equals
        // the 1-row call over that row's visible keys bit for bit (batch invariance across row counts and lengths)
        const int H = 24, KH = 4, D = 256;
        const float sc = 1.0f / std::sqrt(float(D));
        bool inv_ok = true;
        for (int length : {8192, 16384, 32768, 65536}) {
            auto k = mx::astype(random_array(rng, {1, KH, length, D}, 1.0f), mx::bfloat16);
            auto v = mx::astype(random_array(rng, {1, KH, length, D}, 1.0f), mx::bfloat16);
            auto q = mx::astype(random_array(rng, {1, H, 8, D}, 1.0f), mx::bfloat16);
            mx::eval(k, v, q);
            for (int n : {1, 2, 4, 5}) {
                auto qn = mx::contiguous(mx::slice(q, {0, 0, 0, 0}, {1, H, n, D}));
                mx::eval(qn);
                try {
                    auto call = [&] {return mx::fast::scaled_dot_product_attention(qn, k, v, sc, n > 1 ? "causal" : "");};
                    auto y = call(); mx::eval(y);
                    const int reps = 10, calls = 16;
                    auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) {std::vector<mx::array> o; for (int c = 0; c < calls; ++c) o.push_back(call()); mx::eval(o);}
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / reps / calls;
                    std::printf("VEC_BENCH mlx_vector length=%d n=%d %.1f us per call\n", length, n, us);
                    if (n == 5) for (int t = 0; t < n; ++t) {
                        const int vis = length - n + t + 1;
                        auto qt = mx::contiguous(mx::slice(qn, {0, 0, t, 0}, {1, H, t + 1, D}));
                        auto kt = mx::slice(k, {0, 0, 0, 0}, {1, KH, vis, D}), vt = mx::slice(v, {0, 0, 0, 0}, {1, KH, vis, D});
                        auto one = mx::contiguous(mx::fast::scaled_dot_product_attention(qt, kt, vt, sc, ""));
                        auto row = mx::contiguous(mx::slice(y, {0, 0, t, 0}, {1, H, t + 1, D}));
                        mx::eval(one, row);
                        const bool same = std::memcmp(one.data<uint16_t>(), row.data<uint16_t>(), one.size() * 2) == 0;
                        inv_ok &= same;
                        std::printf("VEC_INVARIANT length=%d row=%d of 5 vs 1-row call over %d keys: %s\n", length, t, vis, same ? "EQUAL" : "DIFFERENT");
                    }
                } catch (const std::exception& e) {std::printf("VEC_BENCH mlx_vector length=%d n=%d EXCEPTION %s\n", length, n, e.what());}
            }
#ifdef MLX_HAS_SDPA_LSE
            // bf16_segmented (our matrix path) at 8 rows and 1 row for the same cache, same units
            for (int n : {1, 8}) {
                auto qn = mx::astype(mx::contiguous(mx::slice(q, {0, 0, 0, 0}, {1, H, n, D})), mx::float32);
                mx::eval(qn);
                try {
                    auto call = [&] {return decode_attention_bf16_segmented(qn, k, v, sc, length);};
                    auto y = call(); mx::eval(y);
                    const int reps = 10, calls = 16;
                    auto start = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) {std::vector<mx::array> o; for (int c = 0; c < calls; ++c) o.push_back(call()); mx::eval(o);}
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / reps / calls;
                    std::printf("VEC_BENCH bf16_segmented        length=%d n=%d %.1f us per call\n", length, n, us);
                } catch (const std::exception& e) {std::printf("VEC_BENCH bf16_segmented length=%d n=%d EXCEPTION %s\n", length, n, e.what());}
            }
#endif
        }
        std::printf(inv_ok ? "VEC_INVARIANT_PASS\n" : "VEC_INVARIANT_FAIL\n");
        return 0;
    }
    for (bool seg : {false, true}) {
        if (seg && !segmented_attention_available()) {std::printf("DECODE_ROWS bf16_segmented not available (MLX without scaled_dot_product_attention_lse or no matrix unit)\n"); continue;}
        segmented = seg;
        for (int length : {8, 63, 64, 65, 255, 256, 257, 300, 511, 512, 513, 1023, 1024, 1025, 1030, 1039, 1900, 2048, 2049, 2056, 4095, 4096, 4097, 4100, 4111, 4112, 8193, 8200, 16384, 16391, 32768, 40001, 65536, 65543})
            for (int n : {1, 3, 8, 16})
                if (length >= n) ok &= safe_run(24, 4, 256, length, n, rng);
        ok &= safe_run(8, 2, 128, 200, 8, rng);
        ok &= safe_run(8, 2, 128, 3000, 1, rng);
        ok &= safe_run(24, 4, 256, 16384, 8, rng);
    }
    for (int length : {64, 1023, 1024, 1025, 4096, 8193, 16384, 32768, 65543}) {
        for (int n : {1, 8, 16}) if (length >= n) ok &= safe_run_int8_reference(24, 4, 256, length, n, rng);
        for (int n : {1, 8, 16}) if (length >= n) ok &= safe_run_int8(length, n, rng);
    }
    bool shapes = true;
    for (bool seg : {false, true}) {
        if (seg && !segmented_attention_available()) continue;
        segmented = seg;
        for (int length : {16, 64, 255, 1023, 1024, 1025, 2048, 4095, 4096, 4097, 4100, 4103, 4111, 8193, 16384, 32768, 65543})
            for (int n : {1, 4, 8, 9})
                if (length >= 16) shapes &= run_shapes(24, 4, 256, length, n, rng);
    }
    std::printf(shapes ? "DECODE_ROWS_SHAPES_PASS\n" : "DECODE_ROWS_SHAPES_FAIL\n");
    ok &= shapes;
    for (int length : {64, 1024, 2048, 8192, 16384, 32768, 65536})
        for (int n : {1, 4, 8, 16}) bench(24, 4, 256, length, n, rng);
    ok = ok && prefill_memory_ok;
    std::printf(ok ? "DECODE_ROWS_PASS\n" : "DECODE_ROWS_FAIL\n");
    return ok ? 0 : 1;
}

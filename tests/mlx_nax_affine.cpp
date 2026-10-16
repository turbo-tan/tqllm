#include "nax_affine.hpp"
#include <mlx/random.h>
#include "tq4.hpp"
#include "q6.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
uint16_t bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
float bf16_value(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}
std::vector<float> rows_of(const mx::array& y) {
    auto c = mx::contiguous(y);
    mx::eval(c);
    return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}
bool run(int rows, int k, int bits, int group, std::mt19937& rng) {
    const int levels = 1 << bits, gpr = k / group;
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() % levels);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f), xd(-2.0f, 2.0f);
    std::vector<uint16_t> s(size_t(rows) * gpr), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(i % 7 ? sd(rng) : sd(rng) * 1e-3f); b[i] = bf16_bits(bd(rng));}
    auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, bits, group), rows, k, bits, group);
    const int m = 40;
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    auto full = rows_of(nax_affine_matmul(w, mx::array(x.data(), {m, k}, mx::float32)));
    double worst = 0, peak = 0;
    for (int r = 0; r < m; ++r)
        for (int n = 0; n < rows; ++n) {
            double acc = 0;
            for (int c = 0; c < k; ++c) {
                const size_t g = size_t(n) * gpr + c / group;
                const float v = float(mx::float16_t(std::fma(float(codes[size_t(n) * k + c]), bf16_value(s[g]), bf16_value(b[g]))));
                acc += double(v) * x[size_t(r) * k + c];
            }
            peak = std::max(peak, std::abs(acc));
            worst = std::max(worst, std::abs(acc - full[size_t(r) * rows + n]));
        }
    // every output row depends only on its own input row: compare one-row, 8-row and 16-row calls bitwise
    size_t mismatches = 0;
    for (int mm : {1, 8, 16}) {
        for (int start : {0, 33 - std::min(33, mm - 1)}) {
            if (start + mm > m) continue;
            auto part = rows_of(nax_affine_matmul(w, mx::array(x.data() + size_t(start) * k, {mm, k}, mx::float32)));
            for (int r = 0; r < mm; ++r)
                mismatches += std::memcmp(&part[size_t(r) * rows], &full[size_t(start + r) * rows], size_t(rows) * 4) != 0;
        }
    }
    std::printf("NAX affine rows=%d k=%d bits=%d group=%d rel_err=%.3e row_mismatches=%zu\n", rows, k, bits, group,
                worst / peak, mismatches);
    return worst / peak < 1e-3 && mismatches == 0;
}
// real FFN-down shape (split-K active): rows of a 1-row, 8-row and 40-row call must agree bitwise
bool run_invariance(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() & 15u);
    std::vector<uint16_t> s(size_t(rows) * (k / 128)), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(0.01f + float(rng() % 100) * 1e-4f); b[i] = bf16_bits(float(int(rng() % 200) - 100) * 1e-3f);}
    auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, 4, 128), rows, k, 4, 128);
    const int m = 40;
    std::vector<float> x(size_t(m) * k);
    std::uniform_real_distribution<float> xd(-2.0f, 2.0f);
    for (auto& v : x) v = xd(rng);
    auto full = rows_of(nax_affine_matmul(w, mx::array(x.data(), {m, k}, mx::float32)));
    size_t mismatches = 0;
    for (int mm : {1, 8, 16})
        for (int start : {0, 33 - std::min(33, mm - 1)}) {
            auto part = rows_of(nax_affine_matmul(w, mx::array(x.data() + size_t(start) * k, {mm, k}, mx::float32)));
            for (int r = 0; r < mm; ++r)
                mismatches += std::memcmp(&part[size_t(r) * rows], &full[size_t(start + r) * rows], size_t(rows) * 4) != 0;
        }
    std::printf("NAX affine invariance rows=%d k=%d splits=%d row_mismatches=%zu\n", rows, k, nax_affine_splits(w), mismatches);
    return mismatches == 0;
}
// rows picked from a stored payload give the same products as those rows of the full matrix
bool run_select(int bits, int group, std::mt19937& rng) {
    const int rows = 512, k = 512, gpr = k / group;
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() % (1u << bits));
    std::vector<uint16_t> s(size_t(rows) * gpr), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(0.01f + float(rng() % 100) * 1e-4f); b[i] = bf16_bits(float(int(rng() % 200) - 100) * 1e-3f);}
    auto payload = nax_affine_pack(codes, s, b, rows, k, bits, group);
    auto full = nax_affine_load(payload, rows, k, bits, group);
    std::vector<uint32_t> keep(128);
    for (auto& r : keep) r = uint32_t(rng() % rows);
    auto sub = nax_affine_select_rows(payload, rows, k, bits, group, keep);
    std::vector<float> x(size_t(3) * k);
    for (auto& v : x) v = float(int(rng() % 2001) - 1000) * 1e-3f;
    auto yf = rows_of(nax_affine_matmul(full, mx::array(x.data(), {3, k}, mx::float32)));
    auto ys = rows_of(nax_affine_matmul(sub, mx::array(x.data(), {3, k}, mx::float32)));
    double worst = 0, peak = 0;
    for (int m = 0; m < 3; ++m)
        for (int j = 0; j < 128; ++j) {
            const double a = yf[size_t(m) * rows + keep[j]], c = ys[size_t(m) * 128 + j];
            peak = std::max(peak, std::abs(a));
            worst = std::max(worst, std::abs(a - c));
        }
    std::printf("NAX affine select rows bits=%d group=%d rel_err=%.3e\n", bits, group, worst / peak);
    return worst / peak < 1e-5;
}
// TQ4 codebook weights on the matrix unit equal the existing TQ4 kernel, and stay row-independent
bool run_tq4(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * (k / 2));
    for (auto& c : codes) c = uint8_t(rng());
    std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
    for (auto& v : scales) v = mx::float16_t(0.01f + float(rng() % 100) * 4e-4f);
    Tq4Weights t{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(scales.data(), {rows, k / 128}, mx::float16), true};
    auto w = nax_from_tq4(t);
    const int m = 40;  // covers the 16-row and the 32-row (prefill) kernels
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(float(int(rng() % 2001) - 1000) * 1e-3f));
    auto X = mx::array(x.data(), {m, k}, mx::float32);
    auto got = rows_of(nax_affine_matmul(w, X));
    auto ref = rows_of(tq4_matmul(t, X));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < got.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - got[i])));}
    size_t mismatches = 0;
    for (int r = 0; r < m; ++r) {
        auto one = rows_of(nax_affine_matmul(w, mx::array(x.data() + size_t(r) * k, {1, k}, mx::float32)));
        mismatches += std::memcmp(one.data(), &got[size_t(r) * rows], size_t(rows) * 4) != 0;
    }
    std::printf("NAX TQ4 rows=%d k=%d rel_err_vs_tq4_matmul=%.3e row_mismatches=%zu\n", rows, k, worst / peak, mismatches);
    return worst / peak < 2e-3 && mismatches == 0;
}

// dequantized FP16 values equal the host decode bitwise; the GEMM path matches the row kernel on a prefill chunk
bool run_gemm(int rows, int k, int bits, int group, std::mt19937& rng) {
    const int levels = 1 << bits, gpr = k / group;
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() % levels);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f), xd(-2.0f, 2.0f);
    std::vector<uint16_t> s(size_t(rows) * gpr), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng));}
    auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, bits, group), rows, k, bits, group);
    auto deq = mx::contiguous(nax_affine_dequant(w));
    mx::eval(deq);
    size_t bad = 0;
    const auto* dq = deq.data<mx::float16_t>();
    for (int n = 0; n < rows; ++n)
        for (int c = 0; c < k; ++c) {
            const size_t g = size_t(n) * gpr + c / group;
            const auto expect = mx::float16_t(std::fma(float(codes[size_t(n) * k + c]), bf16_value(s[g]), bf16_value(b[g])));
            bad += std::memcmp(&expect, &dq[size_t(n) * k + c], 2) != 0;
        }
    const int m = 64;
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    auto X = mx::array(x.data(), {m, k}, mx::float32);
    auto ref = rows_of(nax_affine_matmul(w, X, true)), got = rows_of(nax_affine_gemm(w, X, true));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < ref.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - got[i])));}
    std::printf("NAX gemm rows=%d k=%d bits=%d group=%d dequant_mismatches=%zu rel_err_vs_row_kernel=%.3e\n", rows, k, bits, group, bad, worst / peak);
    return bad == 0 && worst / peak < 1e-2;  // FP16 GEMM output vs the row kernel's FP32 accumulate
}
// microseconds per 1024-row chunk at the FFN-down shape: row kernel vs dequantize + GEMM
void bench_gemm(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() & 15u);
    std::vector<uint16_t> s(size_t(rows) * (k / 128)), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(0.01f); b[i] = bf16_bits(-0.05f);}
    auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, 4, 128), rows, k, 4, 128);
    auto X = mx::random::normal({1024, k});
    mx::eval(X);
    for (int path : {0, 1}) {
        auto call = [&] {return path ? nax_affine_gemm(w, X, true) : nax_affine_matmul(w, X, true);};
        mx::eval(call());
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) mx::eval(call());
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 5;
        std::printf("NAX_PREFILL_BENCH %s rows=%d k=%d m=1024 %.0f us (%.1f GB/s weight-equivalent at 4 bit)\n",
                    path ? "gemm" : "row_kernel", rows, k, us, double(rows) * k * 0.5 / us * 1e-3);
    }
}

// fused FP16 rotation equals tq4_rotate rounded to FP16 (to an FP16 ulp: summation order differs), rows independent
bool run_rotate(std::mt19937& rng) {
    bool ok = true;
    for (int m : {1, 8, 40}) for (int k : {128, 5120, 17408}) {
        std::vector<float> x(size_t(m) * k);
        std::uniform_real_distribution<float> xd(-3.0f, 3.0f);
        for (auto& v : x) v = xd(rng);
        auto X = mx::array(x.data(), {m, k}, mx::float32);
        auto ref = rows_of(mx::astype(tq4_rotate(X), mx::float32));
        auto got = rows_of(mx::astype(tq4_rotate_half(X), mx::float32));
        double worst = 0;
        for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::abs(ref[i] - got[i]) / (std::abs(ref[i]) * 9.8e-4 + 6e-5));
        size_t mismatches = 0;
        if (m > 1) {
            auto one = rows_of(mx::astype(tq4_rotate_half(mx::array(x.data() + k, {1, k}, mx::float32)), mx::float32));
            mismatches = std::memcmp(one.data(), &got[size_t(k)], size_t(k) * 4) != 0;
        }
        std::printf("TQ4 rotate_half m=%d k=%d worst_fp16_ulps=%.2f row_mismatch=%zu\n", m, k, worst, mismatches);
        ok = ok && worst <= 1.01 && mismatches == 0;
    }
    return ok;
}

// Q6_K head repacked for the matrix unit matches q6_matmul (FP32 reference) up to FP16 weight/input rounding; rows independent
bool run_q6(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> blocks(size_t(rows) * (k / 256) * 210);
    for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = uint8_t(rng());
    for (size_t b = 0; b < blocks.size() / 210; ++b) {
        const mx::float16_t d(0.0005f + float(rng() % 100) * 1e-5f);
        std::memcpy(&blocks[b * 210 + 208], &d, 2);
    }
    auto w = load_q6_blocks(blocks, rows, k);
    auto n = nax_from_q6(w);
    const int m = 9;
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(float(int(rng() % 2001) - 1000) * 1e-3f));
    auto X = mx::array(x.data(), {m, k}, mx::float32);
    auto got = rows_of(nax_affine_matmul(n, X)), ref = rows_of(q6_matmul(w, X));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < got.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - got[i])));}
    size_t mismatches = 0;
    for (int r = 0; r < m; ++r) {
        auto one = rows_of(nax_affine_matmul(n, mx::array(x.data() + size_t(r) * k, {1, k}, mx::float32)));
        mismatches += std::memcmp(one.data(), &got[size_t(r) * rows], size_t(rows) * 4) != 0;
    }
    std::printf("NAX Q6 head rows=%d k=%d rel_err_vs_q6_matmul=%.3e row_mismatches=%zu\n", rows, k, worst / peak, mismatches);
    return worst / peak < 2e-3 && mismatches == 0;
}
bool run_embedding(std::mt19937& rng) {
    const int rows = 300, k = 256;
    std::vector<uint8_t> payload(affine8_embedding_bytes(rows, k));
    for (int i = 0; i < rows * k; ++i) payload[i] = uint8_t(rng());
    std::vector<uint16_t> sb(size_t(rows) * (k / 64) * 2);
    for (auto& v : sb) v = bf16_bits(std::uniform_real_distribution<float>(-0.05f, 0.05f)(rng) * (rng() % 5 ? 1.0f : 1e-4f));
    std::memcpy(payload.data() + rows * k, sb.data(), sb.size() * 2);
    auto table = affine8_embedding_load(payload, rows, k);
    std::vector<uint32_t> tokens = {0, 299, 17, 17};
    auto got = rows_of(affine8_embedding(table, tokens));
    double worst = 0;
    for (size_t t = 0; t < tokens.size(); ++t)
        for (int c = 0; c < k; ++c) {
            const size_t g = size_t(tokens[t]) * (k / 64) + c / 64;
            const double ref = double(payload[size_t(tokens[t]) * k + c]) * bf16_value(sb[g]) + bf16_value(sb[size_t(rows) * (k / 64) + g]);
            worst = std::max(worst, std::abs(ref - got[t * k + c]));
        }
    std::printf("affine8 embedding max_abs_err=%.3e\n", worst);
    return worst < 1e-5;
}
}

int main() {
    std::mt19937 rng(7);
    bool ok = run(256, 512, 4, 128, rng) && run(128, 256, 4, 64, rng) && run(256, 512, 6, 64, rng) && run_invariance(5120, 17408, rng) && run_invariance(1024, 5120, rng) && run_select(6, 64, rng) && run_select(4, 128, rng) && run_tq4(256, 512, rng) && run_tq4(5120, 2048, rng) && run_rotate(rng) && run_q6(256, 512, rng) && run_q6(2048, 5120, rng) && run_gemm(256, 512, 4, 128, rng) && run_gemm(256, 512, 6, 64, rng) && run_gemm(1024, 5120, 4, 128, rng) && run_embedding(rng);
    bench_gemm(17408, 5120, rng);
    bench_gemm(5120, 17408, rng);
    std::puts(ok ? "MLX_NAX_AFFINE_PASS" : "MLX_NAX_AFFINE_FAIL");
    return ok ? 0 : 1;
}

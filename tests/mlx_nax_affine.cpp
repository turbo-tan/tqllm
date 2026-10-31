#include "nax_affine.hpp"
#include <mlx/random.h>
#include "tq4.hpp"
#include "q6.hpp"
#include <algorithm>
#include <array>
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
    auto c = mx::contiguous(mx::astype(y, mx::float32));
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
// MLX quantize output repacked on the GPU (nax_affine_from_quantized) equals the host pack of the same codes word for
// word, and its 16-row product matches quantized_matmul
bool run_from_quantized(int rows, int k, std::mt19937& rng) {
    std::uniform_real_distribution<float> wd(-0.1f, 0.1f);
    std::vector<float> w(size_t(rows) * k);
    for (auto& v : w) v = wd(rng);
    auto q = mx::quantize(mx::astype(mx::array(w.data(), {rows, k}, mx::float32), mx::bfloat16), 64, 4, "affine");
    mx::eval(q);
    auto got = nax_affine_from_quantized(q[0], q[1], q[2], 64);
    auto cq = mx::contiguous(q[0]), sq = mx::contiguous(mx::view(q[1], mx::uint16)), bq = mx::contiguous(mx::view(q[2], mx::uint16));
    mx::eval(cq, sq, bq);
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t((cq.data<uint32_t>()[i / 8] >> (4 * (i % 8))) & 15u);
    std::vector<uint16_t> sc(sq.data<uint16_t>(), sq.data<uint16_t>() + sq.size()), bi(bq.data<uint16_t>(), bq.data<uint16_t>() + bq.size());
    auto want = nax_affine_load(nax_affine_pack(codes, sc, bi, rows, k, 4, 64), rows, k, 4, 64);
    auto bytes_equal = [](const mx::array& a, const mx::array& b) {
        auto ca = mx::contiguous(a), cb = mx::contiguous(b);
        mx::eval(ca, cb);
        return ca.nbytes() == cb.nbytes() && std::memcmp(ca.data<void>(), cb.data<void>(), ca.nbytes()) == 0;
    };
    const bool same = bytes_equal(got.words, want.words) && bytes_equal(got.scales, want.scales) && bytes_equal(got.biases, want.biases);
    std::vector<float> x(size_t(16) * k);
    for (auto& v : x) v = float(mx::float16_t(wd(rng) * 10.0f));
    auto X = mx::array(x.data(), {16, k}, mx::float32);
    auto y = rows_of(nax_affine_matmul(got, X, false));
    auto ref = rows_of(mx::quantized_matmul(X, q[0], q[1], q[2], true, 64, 4, "affine"));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < y.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - y[i])));}
    std::printf("NAX from quantized rows=%d k=%d words_equal_host_pack=%d rel_err_vs_quantized_matmul=%.3e\n", rows, k, int(same), worst / peak);
    return same && worst / peak < 2e-3;
}
// stored codes repacked into MLX's layout (nax_affine_to_mlx) equal the host MLX packing of the same codes word for word,
// FP16 scales / biases are the BF16 values, and the prefill product on quantized_matmul matches the dequant + GEMM path
bool run_to_mlx(int rows, int k, int group, std::mt19937& rng) {
    const int gpr = k / group;
    std::vector<uint8_t> codes(size_t(rows) * k);
    for (auto& c : codes) c = uint8_t(rng() & 15u);
    std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f), xd(-2.0f, 2.0f);
    std::vector<uint16_t> s(size_t(rows) * gpr), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng));}
    auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, 4, group), rows, k, 4, group);
    auto q = nax_affine_to_mlx(w);
    auto cq = mx::contiguous(q.codes), sq = mx::contiguous(q.scales), bq = mx::contiguous(q.biases);
    mx::eval(cq, sq, bq);
    size_t bad_words = 0, bad_scales = 0;
    for (size_t i = 0; i < size_t(rows) * k / 8; ++i) {
        uint32_t want = 0;
        for (int e = 0; e < 8; ++e) want |= uint32_t(codes[i * 8 + e]) << (4 * e);
        bad_words += cq.data<uint32_t>()[i] != want;
    }
    // exact in FP16's normal range; below it (|v| < 6.1e-5) FP16 subnormals round by at most half their step, 3e-8
    for (size_t i = 0; i < s.size(); ++i)
        bad_scales += std::abs(float(sq.data<mx::float16_t>()[i]) - bf16_value(s[i])) > 3e-8f ||
                      std::abs(float(bq.data<mx::float16_t>()[i]) - bf16_value(b[i])) > 3e-8f;
    const int m = 64;
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    auto X = mx::array(x.data(), {m, k}, mx::float32);
    auto ref = rows_of(nax_affine_gemm(w, X, false));
    auto y = rows_of(mx::astype(nax_affine_qmm(q, X), mx::float32));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < y.size(); ++i) {peak = std::max(peak, double(std::abs(ref[i]))); worst = std::max(worst, double(std::abs(ref[i] - y[i])));}
    std::printf("NAX to MLX rows=%d k=%d group=%d bad_words=%zu bad_scales=%zu qmm_rel_err_vs_gemm=%.3e\n", rows, k, group,
                bad_words, bad_scales, worst / peak);
    return bad_words == 0 && bad_scales == 0 && worst / peak < 2e-3;
}
// the prefill kernel on the stored codes matches the dequant + GEMM path (affine and TQ4), every tile shape, an uneven
// row count (partial row tiles) and partial column threadgroups
bool run_prefill(int rows, int k, int group, bool tq4, int m, std::mt19937& rng) {
    NaxAffine w = [&] {
        if (tq4) {
            std::vector<uint8_t> codes(size_t(rows) * (k / 2));
            for (auto& c : codes) c = uint8_t(rng());
            std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
            for (auto& v : scales) v = mx::float16_t(0.01f + float(rng() % 100) * 4e-4f);
            return nax_from_tq4(Tq4Weights{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(scales.data(), {rows, k / 128}, mx::float16), true});
        }
        std::vector<uint8_t> codes(size_t(rows) * k);
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f);
        std::vector<uint16_t> s(size_t(rows) * (k / group)), b(s.size());
        for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng));}
        return nax_affine_load(nax_affine_pack(codes, s, b, rows, k, 4, group), rows, k, 4, group);
    }();
    std::uniform_real_distribution<float> xd(-2.0f, 2.0f);
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    auto X = mx::array(x.data(), {m, k}, mx::float32);
    auto ref = rows_of(nax_affine_gemm(w, X, false));
    std::vector<float> g(size_t(m) * rows);
    for (auto& v : g) v = float(mx::float16_t(xd(rng)));
    auto G = mx::array(g.data(), {m, rows}, mx::float32);
    double worst_all = 0;
    for (auto [rs, tn, sgs, tb, pf] : {std::array{32, 64, 4, 1, 1}, std::array{16, 64, 8, 1, 0}, std::array{16, 32, 2, 1, 1},
                                       std::array{32, 64, 4, 0, 1}, std::array{16, 32, 2, 0, 0}, std::array{32, 64, 8, 0, 0}})
        for (int ep = 0; ep < 3; ++ep) {
            auto y = rows_of(mx::astype(nax_affine_prefill(w, X, ep, ep ? std::optional<mx::array>(G) : std::nullopt, rs, tn, sgs, tb, pf), mx::float32));
            double worst = 0, peak = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                const float want = ep == 1 ? ref[i] * g[i] / (1.0f + std::exp(-g[i])) : ep == 2 ? ref[i] + g[i] : ref[i];
                peak = std::max(peak, double(std::abs(want)));
                worst = std::max(worst, double(std::abs(want - y[i])));
            }
            std::printf("NAX prefill rows=%d k=%d group=%d tq4=%d m=%d tile %dx%d sg %d tb %d pf %d epilogue %d rel_err_vs_gemm=%.3e\n", rows, k,
                        group, int(tq4), m, rs, tn, sgs, tb, pf, ep, worst / peak);
            worst_all = std::max(worst_all, worst / peak);
        }
    // rows are independent (the wrapper pads a partial row tile): the m rows equal the first m rows of a longer input
    auto more = mx::concatenate({X, mx::random::uniform(-2.0f, 2.0f, {131, k})}, 0);
    const bool same = mx::array_equal(nax_affine_prefill(w, X), mx::slice(nax_affine_prefill(w, more), {0, 0}, {m, rows})).item<bool>();
    std::printf("NAX prefill rows=%d k=%d m=%d rows independent of the row tile: %s\n", rows, k, m, same ? "yes" : "NO");
    return worst_all < 2e-3 && same;
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

// gate + up + SwiGLU in one launch (nax_affine_swiglu) equals the two row-kernel matmuls followed by
// bf16(bf16(silu(bf16 gate)) * bf16 up) within one BF16 rounding of the activation, and every row is bitwise equal to its
// own 1-row call (affine 4-bit and TQ4; the 27B FFN shape among them)
bool run_swiglu(int rows, int k, bool tq4, std::mt19937& rng) {
    auto make = [&]() -> NaxAffine {
        if (tq4) {
            std::vector<uint8_t> codes(size_t(rows) * (k / 2));
            for (auto& c : codes) c = uint8_t(rng());
            std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
            for (auto& v : scales) v = mx::float16_t(0.01f + float(rng() % 100) * 4e-4f);
            return nax_from_tq4(Tq4Weights{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(scales.data(), {rows, k / 128}, mx::float16), true});
        }
        std::vector<uint8_t> codes(size_t(rows) * k);
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f);
        std::vector<uint16_t> s(size_t(rows) * (k / 64)), b(s.size());
        for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng));}
        return nax_affine_load(nax_affine_pack(codes, s, b, rows, k, 4, 64), rows, k, 4, 64);
    };
    auto g = make(), u = make();
    std::uniform_real_distribution<float> xd(-2.0f, 2.0f);
    const int m = 16;
    std::vector<float> x(size_t(m) * k);
    for (auto& v : x) v = float(mx::float16_t(xd(rng)));
    auto X = mx::array(x.data(), {m, k}, mx::float32);
    auto got = rows_of(nax_affine_swiglu(g, u, X));
    auto gv = rows_of(nax_affine_matmul(g, X, true)), uv = rows_of(nax_affine_matmul(u, X, true));
    double worst = 0, peak = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float act = bf16_value(bf16_bits(gv[i] / (1.0f + std::exp(-gv[i]))));
        // the kernel hands the BF16 value on as FP16 (ffn_down's matmul input): the reference takes the same rounding,
        // so values past the FP16 range compare as the same inf the old FP32 -> FP16 cast produced
        const float ref = float(mx::float16_t(bf16_value(bf16_bits(act * uv[i]))));
        if (std::isinf(ref) && std::isinf(got[i]) && (ref > 0) == (got[i] > 0)) continue;
        worst = std::max(worst, double(std::fabs(got[i] - ref))); peak = std::max(peak, double(std::fabs(ref)));
    }
    int mismatches = 0;
    for (int r = 0; r < m; ++r) {
        auto one = rows_of(nax_affine_swiglu(g, u, mx::array(x.data() + size_t(r) * k, {1, k}, mx::float32)));
        if (std::memcmp(one.data(), got.data() + size_t(r) * rows, size_t(rows) * sizeof(float))) ++mismatches;
    }
    std::printf("NAX_SWIGLU rows=%d k=%d tq4=%d rel=%.2e row_mismatches=%d\n", rows, k, int(tq4), worst / std::max(peak, 1e-30), mismatches);
    return mismatches == 0 && worst / std::max(peak, 1e-30) < 1e-2;
}
int main() {
    std::mt19937 rng(7);
    bool ok = run(256, 512, 4, 128, rng) && run(128, 256, 4, 64, rng) && run(256, 512, 6, 64, rng) && run_invariance(5120, 17408, rng) && run_invariance(1024, 5120, rng) && run_select(6, 64, rng) && run_select(4, 128, rng) && run_tq4(256, 512, rng) && run_tq4(5120, 2048, rng) && run_rotate(rng) && run_q6(256, 512, rng) && run_q6(2048, 5120, rng) && run_gemm(256, 512, 4, 128, rng) && run_gemm(256, 512, 6, 64, rng) && run_gemm(1024, 5120, 4, 128, rng) && run_embedding(rng) && run_from_quantized(256, 512, rng) && run_from_quantized(16384 + 512, 1024, rng) && run_to_mlx(256, 512, 64, rng) && run_to_mlx(256, 512, 128, rng) && run_to_mlx(5120, 2048, 64, rng) && run_prefill(256, 512, 64, false, 1910, rng) && run_prefill(5120, 2048, 64, false, 300, rng) && run_prefill(256, 512, 128, true, 77, rng) && run_prefill(256, 512, 64, false, 2048, rng) && run_prefill(256, 512, 128, true, 256, rng) && run_swiglu(256, 512, false, rng) && run_swiglu(256, 512, true, rng) && run_swiglu(17408, 5120, false, rng) && run_swiglu(17408, 5120, true, rng);
    bench_gemm(17408, 5120, rng);
    bench_gemm(5120, 17408, rng);
    std::puts(ok ? "MLX_NAX_AFFINE_PASS" : "MLX_NAX_AFFINE_FAIL");
    return ok ? 0 : 1;
}

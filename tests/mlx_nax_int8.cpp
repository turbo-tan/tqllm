// int8 prefill path for TQ4 matrices: exact weights (integer levels, per-128-column FP16 scales), per-row int8 input.
// 1. inputs on the int8 grid (n * 2^-7, every block holding +-127): the result equals a float64 host reference (only summation order differs)
// 2. realistic inputs: relative error vs the host reference (the activation quantization alone)
// 3. throughput vs the FP16 GEMM path at prefill shapes
#include "nax_affine.hpp"
#include "tq4.hpp"
#include <mlx/random.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
const float kLevel[16] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
struct Fixture { std::vector<uint8_t> codes; std::vector<mx::float16_t> scales; NaxAffine w; int rows, k; };
Fixture make(int rows, int k, std::mt19937& rng) {
    std::vector<uint8_t> codes(size_t(rows) * (k / 2));
    for (auto& c : codes) c = uint8_t(rng());
    std::vector<mx::float16_t> scales(size_t(rows) * (k / 128));
    for (auto& v : scales) v = mx::float16_t(0.01f + float(rng() % 100) * 4e-4f);
    Tq4Weights t{mx::array(codes.data(), {rows, k / 2}, mx::uint8), mx::array(scales.data(), {rows, k / 128}, mx::float16), true};
    return {codes, scales, nax_from_tq4(t), rows, k};
}
// host float64 reference y[m][n] = sum_k x[m][k] * level[n][k] * scale[n][k / 128]; codes pack two columns per byte,
// low nibble first (as tq4_matmul reads them)
std::vector<double> reference(const Fixture& f, const std::vector<float>& x, int m) {
    std::vector<double> y(size_t(m) * f.rows);
    std::vector<float> wrow(f.k);
    for (int n = 0; n < f.rows; ++n) {
        for (int c = 0; c < f.k; ++c) {
            const uint8_t byte = f.codes[size_t(n) * (f.k / 2) + c / 2];
            wrow[c] = kLevel[(c & 1) ? (byte >> 4) : (byte & 15)] * float(f.scales[size_t(n) * (f.k / 128) + c / 128]);
        }
        for (int r = 0; r < m; ++r) {
            double acc = 0;
            const float* xr = x.data() + size_t(r) * f.k;
            for (int c = 0; c < f.k; ++c) acc += double(xr[c]) * double(wrow[c]);
            y[size_t(r) * f.rows + n] = acc;
        }
    }
    return y;
}
std::vector<float> host(const mx::array& a) {
    auto c = mx::contiguous(mx::astype(a, mx::float32));
    mx::eval(c);
    return {c.data<float>(), c.data<float>() + c.size()};
}
double rel_l2(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < got.size(); ++i) {num += (got[i] - ref[i]) * (got[i] - ref[i]); den += ref[i] * ref[i];}
    return std::sqrt(num / den);
}
template <class F> double best_ms(F f) {
    mx::eval(f());
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        auto t = std::chrono::steady_clock::now();
        mx::eval(f());
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
    }
    return best;
}
}  // namespace

int main() {
  try {
    std::mt19937 rng(7);
    bool ok = true;
    // m = 200 exercises the M bound (row tiles of 64); rows must be a multiple of 128
    for (int rows : {1024, 1152}) {
        const int k = 5120, m = 200;
        auto f = make(rows, k, rng);
        // 1. grid inputs: every |x| <= 127 * 2^-7, so the row scale is exactly 2^-7 and the int8 codes are exact
        std::vector<float> xg(size_t(m) * k);
        for (auto& v : xg) v = float(int(rng() % 255) - 127) / 128.0f;
        for (int r = 0; r < m; ++r) for (int g = 0; g < k / 128; ++g) xg[size_t(r) * k + g * 128] = (g & 1) ? 127.0f / 128.0f : -127.0f / 128.0f;  // block scale exactly 2^-7
        auto Xg = mx::array(xg.data(), {m, k}, mx::float32);
        auto got = host(nax_int8_gemm(f.w, Xg, false, true));
        auto ref = reference(f, xg, m);
        const double e1 = rel_l2(got, ref);
        // 2. realistic inputs: uniform in [-1, 1] (rotated activations are outlier-free)
        std::vector<float> xr(size_t(m) * k);
        for (auto& v : xr) v = float(mx::float16_t(float(int(rng() % 2001) - 1000) * 1e-3f));
        auto Xr = mx::array(xr.data(), {m, k}, mx::float32);
        auto got2 = host(nax_int8_gemm(f.w, Xr, false, true));
        auto ref2 = reference(f, xr, m);
        const double e2 = rel_l2(got2, ref2);
        const double e4 = rel_l2(host(nax_int8_gemm(f.w, Xr, false, false)), ref2);
        auto fp16 = host(nax_affine_gemm(f.w, Xr, false));
        const double e3 = rel_l2(fp16, ref2);
        std::printf("NAX_INT8 rows=%d m=%d exact: grid_rel_l2=%.2e realistic_rel_l2=%.2e; fast (per-row) realistic_rel_l2=%.2e (fp16 path %.2e)\n", rows, m, e1, e2, e4, e3);
        ok = ok && e1 < 1e-5 && e2 < 1e-2;
        // 2b. outlier channels (8 columns x 40, as unrotated LLM activations have): per-token input quantization loses the
        // small channels; smoothing folds the outliers into the per-row weight requant
        std::vector<float> xo = xr;
        for (int r = 0; r < m; ++r) for (int c = 0; c < 8; ++c) { float& v = xo[size_t(r) * k + c * 640 + 7]; v = float(mx::float16_t(v * 40.0f)); }
        auto Xo = mx::array(xo.data(), {m, k}, mx::float32);
        auto refo = reference(f, xo, m);
        const double e5 = rel_l2(host(nax_int8_gemm(f.w, Xo, false, false, 0.0f)), refo);
        const double e6 = rel_l2(host(nax_int8_gemm(f.w, Xo, false, false, 0.5f)), refo);
        const double e7 = rel_l2(host(nax_int8_gemm(f.w, Xr, false, false, 0.5f)), ref2);
        std::printf("NAX_INT8_SMOOTH rows=%d outliers x40: fast rel_l2=%.2e smooth(0.5) rel_l2=%.2e; outlier-free smooth(0.5) rel_l2=%.2e (fast %.2e)\n", rows, e5, e6, e7, e4);
        ok = ok && e6 < e5 && e7 < 2 * e4;
    }
    // 3. throughput at the FFN shapes (gate/up: 17408 x 5120; down: 5120 x 17408), m = 2048
    for (auto [rows, k] : std::vector<std::pair<int, int>>{{17408, 5120}, {5120, 17408}}) {
        auto f = make(rows, k, rng);
        auto X = mx::astype(mx::subtract(mx::multiply(mx::random::uniform({2048, k}), mx::array(2.0f)), mx::array(1.0f)), mx::float16);
        mx::eval(X);
        const double flop = 2.0 * 2048 * rows * k;
        const double t16 = best_ms([&] { return nax_affine_gemm(f.w, X, true); });
        const double t8 = best_ms([&] { return nax_int8_gemm(f.w, X, true, false); });
        const double t8s = best_ms([&] { return nax_int8_gemm(f.w, X, true, false, 0.5f); });
        const double t8e = best_ms([&] { return nax_int8_gemm(f.w, X, true, true); });
        std::printf("NAX_INT8_SPEED rows=%d k=%d m=2048 fp16_gemm_path %.2f ms (%.1f TFLOPS) int8 fast %.2f ms (%.1f TOPS) -> %.2fx; fast+smooth %.2f ms; exact %.2f ms (%.1f TOPS)\n",
                    rows, k, t16, flop / t16 * 1e-9, t8, flop / t8 * 1e-9, t16 / t8, t8s, t8e, flop / t8e * 1e-9);
    }
    std::puts(ok ? "NAX_INT8_PASS" : "NAX_INT8_FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("NAX_INT8_ERROR %s\n", std::string(e.what()).substr(0, 1500).c_str());
    return 2;
  }
}

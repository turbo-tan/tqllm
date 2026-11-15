// Load check of the M5 matrix-unit kernels against CPU references (issue #24: on macOS 27 an M5 / M5 Max ran them and
// answered one repeated token, while the M1-M4 paths answered correctly on the same files).
#include "matrix_unit.hpp"
#include "attention_exact.hpp"
#include "nax_affine.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace tqllm::mlx_backend {
namespace {
namespace mx = mlx::core;
uint16_t bf16_bits(float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);}
float bf16_value(uint16_t b) {const uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f;}
float bf16_round(float v) {return bf16_value(bf16_bits(v));}
std::vector<float> host(const mx::array& y) {
    auto c = mx::contiguous(mx::astype(y, mx::float32)); mx::eval(c);
    return std::vector<float>(c.data<float>(), c.data<float>() + c.size());
}
// largest |got - want| over the largest |want|
double rel_err(const std::vector<float>& got, const std::vector<double>& want) {
    if (got.size() != want.size()) return INFINITY;
    double worst = 0, peak = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) return INFINITY;
        peak = std::max(peak, std::abs(want[i]));
        worst = std::max(worst, std::abs(double(got[i]) - want[i]));
    }
    return peak > 0 ? worst / peak : worst;
}
}  // namespace

MatrixUnitCheck matrix_unit_check() {
    MatrixUnitCheck out;
    auto note = [&](const char* name, double err, double tol) {
        if (err > out.error) out.error = err;
        if (!(err <= tol) && out.ok) {out.ok = false; out.failed = name;}
    };
    try {
        std::mt19937 rng(24);
        // 4-bit affine weights in the NAX lane order: the decode / verify row kernel (1 and 16 rows) and the prefill GEMM
        const int rows = 128, k = 256, group = 64, m = 64;
        std::vector<uint8_t> codes(size_t(rows) * k);
        for (auto& c : codes) c = uint8_t(rng() & 15u);
        std::uniform_real_distribution<float> sd(0.002f, 0.02f), bd(-0.2f, 0.2f), xd(-2.0f, 2.0f);
        std::vector<uint16_t> s(size_t(rows) * (k / group)), b(s.size());
        for (size_t i = 0; i < s.size(); ++i) {s[i] = bf16_bits(sd(rng)); b[i] = bf16_bits(bd(rng));}
        auto w = nax_affine_load(nax_affine_pack(codes, s, b, rows, k, 4, group), rows, k, 4, group);
        std::vector<float> x(size_t(m) * k);
        for (auto& v : x) v = float(mx::float16_t(xd(rng)));
        std::vector<double> want(size_t(m) * rows);
        for (int r = 0; r < m; ++r)
            for (int n = 0; n < rows; ++n) {
                double acc = 0;
                for (int c = 0; c < k; ++c) {
                    const size_t g = size_t(n) * (k / group) + c / group;
                    acc += double(x[size_t(r) * k + c]) * (double(bf16_value(s[g])) * codes[size_t(n) * k + c] + bf16_value(b[g]));
                }
                want[size_t(r) * rows + n] = acc;
            }
        auto X = mx::array(x.data(), {m, k}, mx::float32);
        auto first = [&](int n) {return std::vector<double>(want.begin(), want.begin() + size_t(n) * rows);};
        note("matmul 1 row", rel_err(host(nax_affine_matmul(w, mx::slice(X, {0, 0}, {1, k}))), first(1)), 1e-2);
        note("matmul 16 rows", rel_err(host(nax_affine_matmul(w, mx::slice(X, {0, 0}, {16, k}))), first(16)), 1e-2);
        note("prefill GEMM", rel_err(host(nax_affine_prefill(w, mx::astype(X, mx::float16))), want), 2e-2);

        // tensor-op exact attention: one row, 6 query heads on one KV head, head dim 256, 100 of 128 cached keys
        const int heads = 6, dim = 256, cap = 128, length = 100;
        const float scale = 0.0625f;
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> q(size_t(heads) * dim), kc(size_t(cap) * dim), vc(kc.size());
        for (auto& v : q) v = bf16_round(nd(rng));
        for (auto& v : kc) v = bf16_round(nd(rng));
        for (auto& v : vc) v = bf16_round(nd(rng));
        std::vector<double> att(size_t(heads) * dim, 0.0);
        for (int h = 0; h < heads; ++h) {
            std::vector<double> sc(length);
            double top = -INFINITY, den = 0;
            for (int j = 0; j < length; ++j) {
                double d = 0;
                for (int e = 0; e < dim; ++e) d += double(q[size_t(h) * dim + e]) * kc[size_t(j) * dim + e];
                sc[j] = d * scale;
                top = std::max(top, sc[j]);
            }
            for (int j = 0; j < length; ++j) den += sc[j] = std::exp(sc[j] - top);
            for (int j = 0; j < length; ++j)
                for (int e = 0; e < dim; ++e) att[size_t(h) * dim + e] += sc[j] / den * vc[size_t(j) * dim + e];
        }
        auto Q = mx::array(q.data(), {1, heads, 1, dim}, mx::float32);
        auto K = mx::astype(mx::array(kc.data(), {1, 1, cap, dim}, mx::float32), mx::bfloat16);
        auto V = mx::astype(mx::array(vc.data(), {1, 1, cap, dim}, mx::float32), mx::bfloat16);
        note("exact attention", rel_err(host(decode_attention_exact(Q, K, V, scale, length)), att), 2e-2);
    } catch (const std::exception&) {
        // a kernel that does not compile or run here: the M1-M4 paths
        out.ok = false; out.failed = "kernel did not run"; out.error = INFINITY;
    }
    return out;
}
}  // namespace tqllm::mlx_backend

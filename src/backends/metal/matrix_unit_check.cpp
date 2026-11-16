// Load check of the M5 matrix-unit kernels against CPU references (issue #24: on macOS 27 an M5 / M5 Max ran them and
// answered one repeated token, while the M1-M4 paths answered correctly on the same files).
#include "matrix_unit.hpp"
#include "attention_exact.hpp"
#include "nax_affine.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <tuple>
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

        // tensor-op exact attention, 24 query heads on 4 KV heads, head dim 256: a 16-row verify over 1000 keys (row t sees
        // keys < 985 + t) and one row over 100 of 128 keys
        for (auto [heads, kvh, cap, length, n] : {std::array{6, 1, 128, 100, 1}, std::array{24, 4, 1024, 1000, 16}}) {
            const int dim = 256, group = heads / kvh;
            const float scale = 0.0625f;
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> q(size_t(heads) * n * dim), kc(size_t(kvh) * cap * dim), vc(kc.size());
            for (auto& v : q) v = bf16_round(nd(rng));
            for (auto& v : kc) v = bf16_round(nd(rng));
            for (auto& v : vc) v = bf16_round(nd(rng));
            std::vector<double> att(q.size(), 0.0);
            for (int h = 0; h < heads; ++h)
                for (int t = 0; t < n; ++t) {
                    const int g = h / group, seen = length - n + t + 1;
                    const float* qr = &q[(size_t(h) * n + t) * dim];
                    std::vector<double> sc(seen);
                    double top = -INFINITY, den = 0;
                    for (int j = 0; j < seen; ++j) {
                        double d = 0;
                        for (int e = 0; e < dim; ++e) d += double(qr[e]) * kc[(size_t(g) * cap + j) * dim + e];
                        top = std::max(top, sc[j] = d * scale);
                    }
                    for (int j = 0; j < seen; ++j) den += sc[j] = std::exp(sc[j] - top);
                    double* out = &att[(size_t(h) * n + t) * dim];
                    for (int j = 0; j < seen; ++j)
                        for (int e = 0; e < dim; ++e) out[e] += sc[j] / den * vc[(size_t(g) * cap + j) * dim + e];
                }
            auto Q = mx::array(q.data(), {1, heads, n, dim}, mx::float32);
            auto K = mx::astype(mx::array(kc.data(), {1, kvh, cap, dim}, mx::float32), mx::bfloat16);
            auto V = mx::astype(mx::array(vc.data(), {1, kvh, cap, dim}, mx::float32), mx::bfloat16);
            note(n == 1 ? "exact attention 1 row" : "exact attention 16 rows", rel_err(host(decode_attention_exact(Q, K, V, scale, length)), att), 2e-2);
        }
    } catch (const std::exception&) {
        // a kernel that does not compile or run here: the M1-M4 paths
        out.ok = false; out.failed = "kernel did not run"; out.error = INFINITY;
    }
    return out;
}

namespace {
// the first `rows` rows of a stored matrix, run with the full matrix's split-K
NaxAffine first_rows(const NaxAffine& w, int rows) {
    if (w.rows <= rows) return w;
    NaxAffine s = w;
    auto cut = [&](const mx::array& a) {
        if (a.ndim() < 1 || a.shape(0) != w.rows / 32) return a;
        mx::Shape start(a.ndim(), 0), stop = a.shape();
        stop[0] = rows / 32;
        return mx::contiguous(mx::slice(a, start, stop));
    };
    s.words = cut(w.words); s.high = cut(w.high); s.scales = cut(w.scales); s.biases = cut(w.biases);
    s.rows = rows; s.split = nax_affine_splits(w);
    return s;
}
std::vector<double> host_double(const mx::array& y) {auto f = host(y); return {f.begin(), f.end()};}
}  // namespace

std::string check_model_matrix_unit(const std::map<std::string, NaxAffine>& nax, const std::map<std::string, NaxAffine>& nax_tq4,
                                    const std::map<std::string, NaxAffine>& packs) {
    try {
        std::mt19937 rng(24);
        std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
        std::set<std::tuple<int, int, int, int, bool, int>> seen;
        auto input = [&](int m, int k) {
            std::vector<float> x(size_t(m) * k);
            for (auto& v : x) v = float(mx::float16_t(xd(rng)));
            return mx::array(x.data(), {m, k}, mx::float32);
        };
        char why[256];
        // each shape / format / split once: 1 and 16 rows on the row kernel, 64 rows on the prefill GEMM (4-bit), against
        // the stored values dequantized into an FP32 GEMM
        for (const auto* map : {&nax, &nax_tq4, &packs})
            for (const auto& [name, full] : *map) {
                if (full.q6) continue;
                const int split = nax_affine_splits(full);
                if (!seen.insert({full.rows, full.columns, full.bits, full.group, full.tq4, split}).second) continue;
                const auto w = first_rows(full, std::min(full.rows, 4096));
                auto X = input(64, w.columns);
                auto ref = mx::matmul(X, mx::transpose(mx::astype(nax_affine_dequant(w), mx::float32)));
                for (int m : {1, 16}) {
                    auto Xm = mx::slice(X, {0, 0}, {m, w.columns});
                    const double e = rel_err(host(nax_affine_matmul(w, Xm)), host_double(mx::slice(ref, {0, 0}, {m, w.rows})));
                    if (!(e <= 1e-2)) {std::snprintf(why, sizeof why, "%s %dx%d %d-bit group %d split %d, %d rows: relative error %.3g", name.c_str(), full.rows, full.columns, full.bits, full.group, split, m, e); return why;}
                }
                if (w.bits == 4 && w.group % 64 == 0 && w.columns % 64 == 0 && w.rows % 64 == 0) {
                    const double e = rel_err(host(nax_affine_prefill(w, mx::astype(X, mx::float16))), host_double(ref));
                    if (!(e <= 2e-2)) {std::snprintf(why, sizeof why, "%s %dx%d prefill GEMM: relative error %.3g", name.c_str(), full.rows, full.columns, e); return why;}
                }
            }
        // the decode FFN: gate and up with SwiGLU in one launch, first layer
        for (const auto* map : {&nax, &nax_tq4}) {
            auto g = map->find("blk.0.ffn_gate.weight"), u = map->find("blk.0.ffn_up.weight");
            if (g == map->end() || u == map->end()) continue;
            const auto G = first_rows(g->second, std::min(g->second.rows, 4096)), U = first_rows(u->second, std::min(u->second.rows, 4096));
            auto X = input(16, G.columns);
            auto gr = host(mx::matmul(X, mx::transpose(mx::astype(nax_affine_dequant(G), mx::float32))));
            auto ur = host(mx::matmul(X, mx::transpose(mx::astype(nax_affine_dequant(U), mx::float32))));
            std::vector<double> want(gr.size());
            for (size_t i = 0; i < gr.size(); ++i) {
                const float gb = bf16_round(gr[i]), act = bf16_round(gb / (1.0f + std::exp(-gb)));
                want[i] = bf16_round(act * bf16_round(ur[i]));
            }
            const double e = rel_err(host(nax_affine_swiglu(G, U, X)), want);
            if (!(e <= 2e-2)) {std::snprintf(why, sizeof why, "blk.0 FFN SwiGLU: relative error %.3g", e); return why;}
        }
    } catch (const std::exception& e) {
        return std::string("a matrix-unit kernel failed: ") + e.what();
    }
    return {};
}
}  // namespace tqllm::mlx_backend

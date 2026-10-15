#include "nax_affine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
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
    bool ok = run(256, 512, 4, 128, rng) && run(128, 256, 4, 64, rng) && run(256, 512, 6, 64, rng) && run_invariance(5120, 17408, rng) && run_invariance(1024, 5120, rng) && run_embedding(rng);
    std::puts(ok ? "MLX_NAX_AFFINE_PASS" : "MLX_NAX_AFFINE_FAIL");
    return ok ? 0 : 1;
}

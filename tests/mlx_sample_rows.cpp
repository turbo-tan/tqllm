// sample_rows draws each row's token from filtered_distribution (temperature, top k, top p): empirical frequencies over
// many draws match the host distribution, no token falls outside it, and a seed repeats its draws.
#include "sampling.hpp"
#include "sampling_topk.hpp"
#include <mlx/transforms.h>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace mx = mlx::core;
using tqllm::mlx_backend::filtered_distribution;
using tqllm::mlx_backend::row_topk;
using tqllm::mlx_backend::sample_rows;

int main() {
    const int rows = 3, vocab = 2000, copies = 4000, seeds = 10;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> v(size_t(rows) * vocab);
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < vocab; ++t) v[size_t(r) * vocab + t] = nd(rng) * (1.0f + 1.5f * r);  // flat to peaked rows
    struct Case {float temperature; int top_k; float top_p;};
    const Case cases[] = {{0.6f, 20, 0.95f}, {1.0f, 0, 1.0f}, {0.7f, 40, 0.8f}};
    bool ok = true;
    for (const auto& c : cases) {
        std::vector<float> tiled(size_t(rows) * copies * vocab);
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < copies; ++j)
                std::copy(v.begin() + r * vocab, v.begin() + (r + 1) * vocab, tiled.begin() + (size_t(r) * copies + j) * vocab);
        auto logits = mx::array(tiled.data(), {rows * copies, vocab}, mx::float32);
        std::vector<std::map<uint32_t, double>> count(rows);
        for (int s = 0; s < seeds; ++s) {
            auto drawn = sample_rows(logits, c.temperature, c.top_k, c.top_p, uint64_t(1000 + s));
            mx::eval(drawn);
            const auto* d = drawn.data<uint32_t>();
            for (int r = 0; r < rows; ++r)
                for (int j = 0; j < copies; ++j) count[r][d[size_t(r) * copies + j]] += 1.0;
        }
        const int k = c.top_k > 0 ? c.top_k : 64;
        for (int r = 0; r < rows; ++r) {
            auto scaled = mx::multiply(mx::array(v.data() + r * vocab, {1, vocab}, mx::float32), mx::array(1.0f / c.temperature));
            auto [idx, val] = row_topk(scaled, k);
            mx::eval(idx, val);
            const auto want = filtered_distribution(val.data<float>(), idx.data<uint32_t>(), k, c.top_p);
            std::map<uint32_t, double> p;
            for (const auto& [prob, id] : want) p[id] = prob;
            const double n = double(copies) * seeds;
            double tv = 0, outside = 0;
            for (const auto& [id, prob] : p) tv += std::fabs(prob - (count[r].count(id) ? count[r][id] / n : 0.0));
            for (const auto& [id, cnt] : count[r]) if (!p.count(id)) outside += cnt;
            tv *= 0.5;
            const bool pass = outside == 0 && tv < 0.02;
            ok &= pass;
            std::printf("SAMPLE T %.1f k %d p %.2f row %d support %zu TV %.4f outside %.0f %s\n", c.temperature, c.top_k, c.top_p, r,
                        p.size(), tv, outside, pass ? "PASS" : "FAIL");
        }
    }
    auto logits = mx::array(v.data(), {rows, vocab}, mx::float32);
    auto a = sample_rows(logits, 0.6f, 20, 0.95f, 42), b = sample_rows(logits, 0.6f, 20, 0.95f, 42);
    mx::eval(a, b);
    bool same = true;
    for (int r = 0; r < rows; ++r) same &= a.data<uint32_t>()[r] == b.data<uint32_t>()[r];
    ok &= same;
    std::printf("SAMPLE seed repeat %s\n", same ? "PASS" : "FAIL");
    std::printf(ok ? "SAMPLE_ROWS_PASS\n" : "SAMPLE_ROWS_FAIL\n");
    return ok ? 0 : 1;
}

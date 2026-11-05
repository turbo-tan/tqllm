// sample_rows draws each row's token from filtered_distribution (temperature, top k, top p): empirical frequencies over
// many draws match the host distribution, no token falls outside it, and a seed repeats its draws.
#include "sampling.hpp"
#include "sampling_topk.hpp"
#include <mlx/transforms.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <map>
#include <random>
#include <vector>

namespace mx = mlx::core;
using tqllm::mlx_backend::filtered_distribution;
using tqllm::mlx_backend::row_topk;
using tqllm::mlx_backend::sample_rows;

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0) {  // device cost of one verify's draws: 16 rows x the Qwen vocab, BF16
        const int n = 16, vocab = 248320;
        auto logits = mx::astype(mx::random::normal({n, vocab}), mx::bfloat16);
        mx::eval(logits);
        for (int w = 0; w < 5; ++w) mx::eval(sample_rows(logits, 0.6f, 20, 0.95f, uint64_t(w)));
        for (int mode = 0; mode < 2; ++mode) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < 50; ++s) mx::eval(mode ? sample_rows(logits, 0.6f, 20, 0.95f, uint64_t(s)) : mx::argmax(logits, -1));
            std::printf("SAMPLE_BENCH %s %.3f ms per call\n", mode ? "sample_rows" : "argmax",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 50);
        }
        return 0;
    }
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
    {
        const int wide = 248320 + 77;  // not a multiple of the block
        auto big = mx::random::normal({rows, wide}, mx::float32, mx::random::key(5));
        bool same_sets = true;
        for (int kk : {1, 20, 64}) {
            auto [i1, v1] = tqllm::mlx_backend::row_topk(big, kk);
            auto [i2, v2] = tqllm::mlx_backend::row_topk_blocked(big, kk);
            mx::eval(i1, i2);
            for (int r = 0; r < rows; ++r) {
                std::vector<uint32_t> a1(i1.data<uint32_t>() + r * kk, i1.data<uint32_t>() + (r + 1) * kk);
                std::vector<uint32_t> a2(i2.data<uint32_t>() + r * kk, i2.data<uint32_t>() + (r + 1) * kk);
                std::sort(a1.begin(), a1.end()); std::sort(a2.begin(), a2.end());
                same_sets &= a1 == a2;
            }
        }
        ok &= same_sets;
        std::printf("SAMPLE blocked top k == row_topk (k 1 / 20 / 64, vocab %d) %s\n", wide, same_sets ? "PASS" : "FAIL");
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

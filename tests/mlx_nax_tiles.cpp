// Sweep of nax_affine_prefill tile shapes (RS, TN, SGS, TB, PF, RF) at the prefill row counts that matter, FFN gate shape
// [17408, 5120], against dequantize (once) + FP16 matmul. Prints ms per call (median of 15) and the best tile per m.
#import <Foundation/Foundation.h>
#include "nax_affine.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
using Clock = std::chrono::steady_clock;
static uint16_t bf(float v) {uint32_t u; std::memcpy(&u, &v, 4); return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);}
static NaxAffine rn(int r, int c, std::mt19937& g) {
    std::vector<uint8_t> co(size_t(r) * c);
    for (auto& x : co) x = g() & 15u;
    std::uniform_real_distribution<float> sd(.002f, .02f), bd(-.05f, .05f);
    std::vector<uint16_t> s(size_t(r) * (c / 64)), b(s.size());
    for (size_t i = 0; i < s.size(); ++i) {s[i] = bf(sd(g)); b[i] = bf(bd(g));}
    return nax_affine_load(nax_affine_pack(co, s, b, r, c, 4, 64), r, c, 4, 64);
}
static double med(std::vector<double>& v) {std::sort(v.begin(), v.end()); return v[v.size() / 2];}
int main() {
    @autoreleasepool {
        const int F = 17408, H = 5120;
        mx::random::seed(3);
        std::mt19937 g(1);
        auto w = rn(F, H, g);
        auto deq = nax_affine_dequant(w); mx::eval(deq);
        for (int m : {512, 1024, 2048}) {
            auto x = mx::astype(mx::random::normal({m, H}), mx::float16); mx::eval(x);
            const auto bench = [&](auto f) {
                auto r = f(); mx::eval(r);
                std::vector<double> t;
                for (int i = 0; i < 15; ++i) {auto s = Clock::now(); auto y = f(); mx::eval(y); t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());}
                return med(t);
            };
            const double base = bench([&] {return mx::matmul(x, mx::transpose(deq));});
            const double withDeq = bench([&] {return mx::matmul(x, mx::transpose(nax_affine_dequant(w)));});
            std::printf("TILES m=%d: fp16 matmul (weights already dequantized) %.2f ms, dequantize + matmul %.2f ms\n", m, base, withDeq);
            double best = 1e9; std::array<int, 6> bt{};
            for (int rs : {16, 32, 64}) for (int tn : {32, 64, 128}) for (int sgs : {2, 4, 8}) for (int tb : {0, 1}) for (int pf : {0, 1}) for (int rf : {0, 1}) {
                if (F % tn) continue;
                double ms = 1e9;
                try {ms = bench([&] {return nax_affine_prefill(w, x, 0, std::nullopt, rs, tn, sgs, tb, pf, rf);});} catch (...) {continue;}
                if (ms < best) {best = ms; bt = {rs, tn, sgs, tb, pf, rf};}
            }
            const double cur = bench([&] {return nax_affine_prefill(w, x, 0, std::nullopt, 32, 64, 4, 1, 1, 1);});
            std::printf("TILES m=%d: current tile 32x64 sg4 tb1 pf1 rf1 %.2f ms; best %.2f ms at RS %d TN %d SGS %d TB %d PF %d RF %d\n",
                        m, cur, best, bt[0], bt[1], bt[2], bt[3], bt[4], bt[5]);
        }
        std::printf("TILES_DONE\n");
    }
    return 0;
}

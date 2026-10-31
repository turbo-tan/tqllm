// Fused GDN verify / decode step (gdn_step_fused) alone on the Qwen3.8-27B shape (16 key heads, 48 value heads, 128 x 128
// state): 48 independent layers per eval, median ms of 20 evals at 1 / 4 / 8 / 16 rows; per-layer us and the state bytes
// moved (read + write) as GB/s. usage: bench-mlx-gdn-step
#include "gdn.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main() {
    const int kh = 16, vh = 48, width = 2 * kh * 128 + vh * 128, layers = 48;
    auto rnd = [](mx::Shape s, float scale) {return mx::multiply(mx::random::normal(s), mx::array(scale));};
    auto weights = rnd({width, 4}, 0.5f), bias = rnd({vh}, 1.0f), nw = rnd({128}, 0.3f);
    auto rate = mx::negative(mx::abs(rnd({vh}, 1.0f)));
    std::vector<mx::array> states;
    for (int l = 0; l < layers; ++l) states.push_back(rnd({1, vh, 128, 128}, 0.1f));
    mx::eval(weights, bias, nw, rate); mx::eval(states);
    for (int m : {1, 4, 8, 16}) {
        auto prev = rnd({3, width}, 1.0f), rows = rnd({m, width}, 1.0f), alpha = rnd({m, vh}, 1.0f), beta = rnd({m, vh}, 1.0f);
        auto z = rnd({m, vh * 128}, 1.0f);
        mx::eval(prev, rows, alpha, beta, z);
        auto run = [&] {
            std::vector<mx::array> outs;
            for (int l = 0; l < layers; ++l) {
                auto s = gdn_step_fused(prev, rows, weights, alpha, beta, bias, rate, z, nw, 1e-6f, 0.0883883f, states[size_t(l)], kh, vh);
                outs.push_back(s.out); outs.push_back(s.state); outs.push_back(s.deltas);
            }
            mx::eval(outs);
        };
        for (int w = 0; w < 3; ++w) run();
        std::vector<double> t;
        for (int r = 0; r < 20; ++r) {
            const auto t0 = std::chrono::steady_clock::now(); run();
            t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        const double ms = t[t.size() / 2], bytes = double(layers) * vh * 128 * 128 * 4 * 2;
        std::printf("GDN_STEP m=%2d %d layers %7.3f ms = %6.1f us per layer, state read+write %.0f GB/s\n", m, layers, ms,
                    ms * 1000 / layers, bytes / ms / 1e6);
    }
}

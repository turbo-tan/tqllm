// Speculative sampling host logic: the filtered distribution is a proper distribution with the top logit most likely
// (the 10-18 bug read the row max after overwriting it -> nan), and speculative_pick reproduces p for any draft token.
#include "sampling.hpp"
#include <cstdio>
#include <map>

using namespace tqllm::mlx_backend;

int main() {
    bool ok = true;
    const float scaled[5] = {2.0f, 9.0f, 8.5f, -1.0f, 7.0f};
    const uint32_t ids[5] = {10, 11, 12, 13, 14};
    auto d = filtered_distribution(scaled, ids, 5, 1.0);
    double total = 0;
    for (const auto& e : d) {total += e.first; ok &= std::isfinite(e.first) && e.first >= 0;}
    ok &= d[0].second == 11 && std::fabs(total - 1.0) < 1e-9 && d[0].first > d[1].first;
    std::printf("SAMPLING distribution top %u p %.4f total %.6f %s\n", d[0].second, d[0].first, total, ok ? "PASS" : "FAIL");
    auto tp = filtered_distribution(scaled, ids, 5, 0.6);
    ok &= tp.size() == 2 && tp[0].second == 11;
    std::printf("SAMPLING top_p 0.6 keeps %zu %s\n", tp.size(), tp.size() == 2 ? "PASS" : "FAIL");
    // Monte Carlo: emitted tokens follow p whichever token is drafted (the likely one, an unlikely one, one outside p)
    std::mt19937_64 rng(5);
    for (uint32_t x : {11u, 14u, 99u}) {
        std::map<uint32_t, int> count;
        const int trials = 400000;
        for (int i = 0; i < trials; ++i) ++count[speculative_pick(d, x, rng).second];
        double worst = 0;
        for (const auto& e : d) worst = std::max(worst, std::fabs(count[e.second] / double(trials) - e.first));
        const bool pass = worst < 4e-3 && count.size() <= d.size();
        ok &= pass;
        std::printf("SAMPLING speculative draft %u: max |freq - p| %.4f %s\n", x, worst, pass ? "PASS" : "FAIL");
    }
    std::printf(ok ? "SAMPLING_PASS\n" : "SAMPLING_FAIL\n");
    return ok ? 0 : 1;
}

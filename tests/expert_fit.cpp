#include "../src/expert_fit.hpp"
#include <cstdio>

int main() {
    int bad = 0;
    auto expect = [&](const char * what, bool ok) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++bad; } };
    const size_t MiB = size_t(1) << 20;
    // two small layers and one large one, as Flash-Next's 1.46 / 2.54 MiB experts
    const std::vector<size_t> block = {size_t(1.5 * MiB), size_t(1.5 * MiB), size_t(2.5 * MiB)};
    const std::vector<int> caps = {100, 200, 300};
    const size_t full = tqllm::bank_bytes(caps, block);
    expect("fits: unchanged", tqllm::fit_layer_slots(caps, block, full, 16) == caps);
    const auto cut = tqllm::fit_layer_slots(caps, block, full - 100 * MiB, 16);
    expect("cut fits the budget", tqllm::bank_bytes(cut, block) <= full - 100 * MiB);
    expect("cut leaves less than one large expert unused", full - 100 * MiB - tqllm::bank_bytes(cut, block) < 3 * 3 * MiB);
    expect("shape kept: order of counts", cut[0] < cut[1] && cut[1] < cut[2]);
    expect("never above the request", cut[0] <= 100 && cut[1] <= 200 && cut[2] <= 300);
    const auto floor_case = tqllm::fit_layer_slots(caps, block, 10 * MiB, 16);
    expect("min_cap floor when nothing fits", floor_case == std::vector<int>({16, 16, 16}));
    std::printf(bad ? "EXPERT_FIT_FAILED\n" : "EXPERT_FIT_OK\n");
    return bad ? 1 : 0;
}

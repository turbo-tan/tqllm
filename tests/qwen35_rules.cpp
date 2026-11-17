// qwen35_rules.hpp: the batched-verify column budget per device class; the host K/V tail graph kinds.
#include "qwen35_rules.hpp"
#include <cstdio>
int main() {
    using tqllm::qwen35::verify_column_budget;
    using tqllm::qwen35::stage_graph_kind;
    int fails = 0;
    auto expect = [&](bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } };
    expect(verify_column_budget(8, false, 64) == 16, "3090 (sm_86): 16 columns");
    expect(verify_column_budget(12, true, 64) == 24, "GB10 (integrated Blackwell): 24 columns (np 8 104.5 -> 135.1 tok/s)");
    expect(verify_column_budget(12, false, 64) == 64, "5060 Ti (discrete Blackwell): the full width");
    expect(verify_column_budget(12, true, 8) == 8 && verify_column_budget(8, false, 8) == 8, "never above the full width");
    // 11-17: a request with nothing in host memory captured placeholder copies; the next request that spilled updated them
    // to the mapped K/V and failed ("generation failed: invalid argument", 27B q8_0 K/V + -fit on, 16 GB)
    expect(stage_graph_kind({0, 0, 0, 0}) != stage_graph_kind({1 << 20, 1 << 20, 0, 0}), "nothing staged vs a host tail: two graphs");
    expect(stage_graph_kind({1 << 20, 0, 0, 0}) != stage_graph_kind({1 << 20, 1 << 20, 0, 0}), "K staged, V not yet: its own graph");
    expect(stage_graph_kind({1 << 20, 1 << 20, 0, 0}) == stage_graph_kind({3 << 20, 2 << 20, 0, 0}), "a growing tail updates in place");
    expect(stage_graph_kind({1 << 20, 1 << 20, 1 << 20, 1 << 20}) == stage_graph_kind({0, 0, 0, 0}), "all compacted: placeholders");
    std::printf("qwen35 rules: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

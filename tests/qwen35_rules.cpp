// qwen35_rules.hpp: the batched-verify column budget per device class.
#include "qwen35_rules.hpp"
#include <cstdio>
int main() {
    using tqllm::qwen35::verify_column_budget;
    int fails = 0;
    auto expect = [&](bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } };
    expect(verify_column_budget(8, false, 64) == 16, "3090 (sm_86): 16 columns");
    expect(verify_column_budget(12, true, 64) == 24, "GB10 (integrated Blackwell): 24 columns (np 8 104.5 -> 135.1 tok/s)");
    expect(verify_column_budget(12, false, 64) == 64, "5060 Ti (discrete Blackwell): the full width");
    expect(verify_column_budget(12, true, 8) == 8 && verify_column_budget(8, false, 8) == 8, "never above the full width");
    std::printf("qwen35 rules: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

// layer ranges of the multi-GPU layer split (src/models/qwen/qwen35/layer_split.hpp)
#include "../src/models/qwen/qwen35/layer_split.hpp"
#include <cstdio>
static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } }
int main() {
    using tqllm::layer_range;
    const auto a = layer_range(64, 0, 2, -1), b = layer_range(64, 1, 2, -1);
    check(a.lo == 0 && a.hi == 32 && b.lo == 32 && b.hi == 64, "64 layers on two GPUs: 32 + 32");
    const auto c = layer_range(65, 0, 2, -1), d = layer_range(65, 1, 2, -1);
    check(c.hi == d.lo && c.lo == 0 && d.hi == 65, "odd count: contiguous, covers every layer");
    const auto e = layer_range(64, 0, 2, 40), f = layer_range(64, 1, 2, 40);
    check(e.hi == 40 && f.lo == 40 && f.hi == 64, "--split-at 40");
    check(layer_range(64, 0, 1, -1).hi == 64, "one GPU: every layer");
    check(a.owns(31) && !a.owns(32) && b.owns(32) && a.first() && !b.first(), "ownership");
    bool threw = false; try { layer_range(64, 0, 2, 64); } catch (...) { threw = true; }
    check(threw, "--split-at out of range refused");
    if (failures) return 1;
    std::puts("layer_split: ok");
    return 0;
}

// Flash-Next decode rules (no CUDA): the F39 verify path choice, the F38 slots=auto keep, the adaptive draft stop.
#include "models/qwen/qwen4exp/decode_rules.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace tqllm::flashnext;
static int failures = 0;
static void check(bool ok, const std::string & what) { if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; } }

int main() {
    const int topk = 10, slots = 150;
    // F39: np 4 verify = 4 sequences x 4 rows; 160 assignments over 60 distinct experts (sequences share experts)
    std::vector<int> routed;
    for (int row = 0; row < 16; ++row) for (int k = 0; k < topk; ++k) routed.push_back((row * 7 + k * 13) % 60);
    check(!layer_takes_waves(routed, 16, topk, slots, true), "np 4 verify block whose distinct experts fit takes the split path");
    check(layer_takes_waves(routed, 16, topk, slots, true, true), "old rule: the same block takes waves (the F39 divergence)");
    // one sequence's block never reaches the waves rule
    std::vector<int> one(routed.begin(), routed.begin() + 4 * topk);
    check(!layer_takes_waves(one, 4, topk, slots, true) && !layer_takes_waves(one, 4, topk, slots, true, true), "np 1 verify block: split either way");
    // more distinct experts than slots: waves are needed
    std::vector<int> wide; for (int i = 0; i < 160; ++i) wide.push_back(i);
    check(layer_takes_waves(wide, 16, topk, slots, true), "verify block with more distinct experts than slots takes waves");
    // prompt chunks always take waves
    check(layer_takes_waves(one, 4, topk, slots, false), "prompt chunk (spec off) takes waves");
    check(!layer_takes_waves({1}, 1, topk, slots, false), "single-token step takes the split path");

    // F38: the keep grows with every sequence
    const size_t state = size_t(900) << 20, reserve = size_t(1200) << 20;
    check(auto_keep_bytes(4, state, reserve) - auto_keep_bytes(2, state, reserve) == 2 * (state + (size_t(256) << 20)),
          "slots=auto keeps each extra sequence's state and working room");
    check(auto_keep_bytes(1, state, reserve) == reserve + (size_t(256) << 20), "one sequence keeps the reserve only (long prompts borrow banks)");
    check(auto_keep_bytes(2, state, reserve) >= 2 * state, "several sequences keep every state");

    // adaptive drafts
    check(keep_drafting(0.f, 0.01f), "tau 0: fixed draft count");
    check(keep_drafting(0.9f, 0.95f) && !keep_drafting(0.9f, 0.5f), "tau 0.9: stop below, continue at or above");
    check(keep_drafting(0.9f, 0.9f), "tau 0.9: equal continues");

    std::printf("decode rules: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

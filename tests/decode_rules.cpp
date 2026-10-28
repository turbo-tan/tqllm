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

    // slot tiers: the floor scales with the slot total, never exceeds the target, never drops below 64 (F39: a 6-row verify
    // routes to at most 60 distinct experts, so the split path holds at every bank size)
    check(slot_floor(157, 144, 157) == 144 && slot_floor(247, 144, 157) == 226, "floor scales each layer's target like the total");
    check(slot_floor(70, 40, 157) == 64 && slot_floor(50, 40, 157) == 50, "floor never below 64, never above the target");
    check(verify_slot_min(5, 10) == 64 && verify_slot_min(3, 10) == 40, "verify floor: 6 rows x top-10 -> 64, 4 rows -> 40");
    check(slot_floor(57, 0, 57, verify_slot_min(3, 10)) == 40, "small card: a 57-slot layer can share down to 40");
    check(slot_tiers(1, false), "one sequence: slot tiers, small cards included (12 GB refill left 4 of 48 layers without a bank)");
    check(!slot_tiers(4, false) && !slot_tiers(1, true), "np > 1 or a keep override: no tiers");
    std::vector<int> verify6; for (int i = 0; i < 60; ++i) verify6.push_back(i);
    check(!layer_takes_waves(verify6, 6, topk, slot_floor(101, 144, 157), true), "6-row verify keeps the split path at the floor");

    // regrow keep: the request's own state on top of the base, capped at the full context
    const size_t base = size_t(768) << 20, full = size_t(4000) << 20;
    check(regrow_keep_bytes(base, full, 65536, 32768) == base + full / 2, "regrow keeps a 32K request's state (half of 64K)");
    check(regrow_keep_bytes(base, full, 65536, 1 << 20) == base + full, "regrow keep never exceeds the full context's state");
    check(regrow_keep_bytes(base, full, 0, 32768) == base, "no state size known: the base only");

    // adaptive drafts
    check(keep_drafting(0.f, 0.01f), "tau 0: fixed draft count");
    check(keep_drafting(0.9f, 0.95f) && !keep_drafting(0.9f, 0.5f), "tau 0.9: stop below, continue at or above");
    check(keep_drafting(0.9f, 0.9f), "tau 0.9: equal continues");

    // one sequence's draft count: a verify block (drafts + 1 rows) must fit the CPU expert pool when there is one
    for (int m : {1, 3, 5}) {
        const DraftPlan small = one_sequence_drafts(m, true);
        check(small.drafts + 1 <= kCpuPoolRows && !small.adaptive, "CPU pool: the verify block fits the pool's rows, fixed length");
        check(small.drafts >= 1 && small.drafts <= m, "CPU pool: at least one draft, never more than the manifest's");
    }
    check(one_sequence_drafts(3, false).drafts == 5 && one_sequence_drafts(3, false).adaptive, "no CPU pool: adaptive, up to 5");
    check(one_sequence_drafts(7, false).drafts == 7, "no CPU pool: a larger manifest count stays");
    check(one_sequence_drafts(0, false).drafts == 0 && !one_sequence_drafts(0, true).adaptive, "no MTP: no drafts");

    // CPU-owned experts: the device an expert runs on does not depend on residency (F26), owned experts stay out of banks
    {
        int owned = 0, same = 1;
        for (int l = 0; l < 48; ++l) for (int e = 0; e < 512; ++e) {
            owned += cpu_owned(l, e);
            same &= runs_on_cpu(l, e, true, true) == runs_on_cpu(l, e, false, true);
            same &= !runs_on_cpu(l, e, false, false);
        }
        check(same != 0, "an expert runs on the same device whether it is resident or not; no pool: always the GPU");
        check(owned > 48 * 512 / 8 * 9 / 10 && owned < 48 * 512 / 8 * 11 / 10, "about an eighth of the experts are owned");
        std::vector<int> cnt(512, 0); for (int e = 0; e < 512; ++e) cnt[size_t(e)] = (e * 37) % 11;
        const auto top = bank_experts(cnt, 7, 57, true); bool clean = top.size() == 57;
        for (int e : top) clean &= !cpu_owned(7, e) && cnt[size_t(e)] > 0;
        for (size_t i = 1; i < top.size(); ++i) clean &= cnt[size_t(top[i - 1])] >= cnt[size_t(top[i])];
        check(clean, "a bank's experts: most routed first, no owned expert when the pool computes them");
        check(bank_experts(cnt, 7, 1000, false).size() == size_t(512 - std::count(cnt.begin(), cnt.end(), 0)), "no pool: every routed expert");
    }

    // packed experts without a pack file: built in memory on a discrete GPU, never on a GPU sharing system memory
    const double GB = 1e9;
    check(build_pack_in_memory(false, false, false, 51 * GB, 128 * GB), "no pack file, discrete GPU, 128 GB RAM: build the pack from the GGUF");
    check(!build_pack_in_memory(true, false, false, 51 * GB, 128 * GB), "a pack file is used as given");
    check(!build_pack_in_memory(false, true, false, 51 * GB, 128 * GB), "GB10 (integrated): no pack built");
    check(!build_pack_in_memory(false, false, true, 51 * GB, 128 * GB), "hot/cold manages expert memory: no pack built");
    check(!build_pack_in_memory(false, false, false, 51 * GB, 64 * GB) && !build_pack_in_memory(false, false, false, 51 * GB, 24 * GB),
          "a 51 GB pack is not pinned on a 64 GB or 24 GB machine");

    {   // the HF manifest's 3090 plan on a GB10 (512 slots = every expert): no layer below 512
        const std::vector<int> plan{314, 291, 269, 160, 189, 219, 248, 239, 208, 135, 212, 160, 128, 145};
        const auto all = scale_layer_slots(plan, 512, 512);
        check(*std::min_element(all.begin(), all.end()) == 512, "every expert fits: each layer keeps all experts");
        const auto some = scale_layer_slots(plan, 200, 512); long sum = 0; for (int c : some) sum += c;
        check(std::abs(sum - 200L * long(plan.size())) <= long(plan.size()) && some[0] > some[9], "smaller card: plan scaled to the budget, shape kept");
    }
    check(bank_budget_bytes(true, 18.0, size_t(120) << 30) == size_t(120) << 30, "slots=auto: banks capped by the device, not budget_gb");
    check(bank_budget_bytes(false, 18.0, size_t(120) << 30) == size_t(18) << 30, "fixed slots: budget_gb as given");
    check(prefill_chunk(16384, 8) == 16384 && prefill_chunk(16384, 8192) == 16384, "prefill chunk: the engine setting, manifest key or not");
    check(prefill_chunk(0, 8192) == 8192, "prefill chunk: the manifest value only when the setting is off");
    check(drop_after_pack("blk.7.ffn_gate_exps.weight") && drop_after_pack("blk.47.ffn_down_exps.weight"), "trunk expert pages may be dropped after the pack");
    check(!drop_after_pack("per_layer_token_embd.weight") && !drop_after_pack("token_embd.weight") && !drop_after_pack("blk.0.attn_qkv.weight"),
          "the per-layer token embedding and dense tensors stay cached");
    check(warm_at_start("per_layer_token_embd.weight", 28.8e9, 128e9) && !warm_at_start("blk.0.ffn_gate_exps.weight", 1e9, 128e9),
          "the per-layer token embedding is read at start-up");
    check(!warm_at_start("per_layer_token_embd.weight", 28.8e9, 24.7e9), "no start-up read of a 28.8 GB embedding on a 23 GiB host");
    // per-layer decode graphs: on unified memory (GB10 +5-6 %) and on PCIe cards before Blackwell (3090: off cost chat
    // 86.5 -> 83.9 tok/s, 32K 75.8 -> 73.9); off on Blackwell PCIe cards (5090: chat 128 -> 154 with graphs off)
    check(layer_graphs(true, 12) && layer_graphs(true, 11), "unified memory (GB10) keeps the decode graphs");
    check(layer_graphs(false, 8) && layer_graphs(false, 9), "RTX 3090 / 4090 (sm_86 / sm_89) keep the decode graphs");
    check(!layer_graphs(false, 12), "RTX 5090 (sm_120, PCIe) launches the kernels directly");
    std::printf("decode rules: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

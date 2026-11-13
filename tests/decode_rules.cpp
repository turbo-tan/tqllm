#include <algorithm>
// Flash-Next decode rules (no CUDA): the F39 verify path choice, the F38 slots=auto keep, the adaptive draft stop.
#include "models/qwen/qwen4exp/decode_rules.hpp"
#include <cstdio>
#include <new>
#include <cstdlib>
#include <string>
#include <vector>

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

    // G2: decode graphs are per model (owner id), not per address
    {
        GraphWarmup w;
        alignas(8) unsigned char storage[64];
        struct Model { uint64_t owner = next_graph_owner(); };
        auto * a = new (storage) Model;
        const uint64_t first = a->owner;
        check(!w.capture(a->owner, 3, 4, false) && !w.capture(a->owner, 3, 4, false) && w.capture(a->owner, 3, 4, false), "calls 1-2 run plain, call 3 is captured");
        w.forget(a->owner); a->~Model();
        auto * b = new (storage) Model;  // the next request's model at the same address
        check(b->owner != first, "a model built at a finished model's address gets its own owner id");
        check(!w.capture(b->owner, 3, 4, false), "its first call runs plain (it inherits no warm-up calls)");
        check(w.size() == 1, "the finished model's warm-up entries are gone");
        check(!w.capture(b->owner, 3, 4, true) && !w.capture(b->owner, 3, 4, false) && w.capture(b->owner, 3, 4, false), "a plain call (failed capture) starts the count over");
        b->~Model();
    }
    // F38: the keep grows with every sequence
    const size_t state = size_t(900) << 20, reserve = size_t(1200) << 20;
    check(auto_keep_bytes(4, state, reserve) - auto_keep_bytes(2, state, reserve) == 2 * state,
          "slots=auto keeps each extra sequence's state (its working buffers are in the measured state)");
    // 3090 np 8 (11-03): measured state 286 MB, reserve 512 MB: kept 4562 MB and 2.9 GB stayed idle with all 8 decoding
    check(auto_keep_bytes(8, size_t(286) << 20, size_t(512) << 20) == size_t(7 * 286 + 512 + 256) << 20,
          "np 8 keeps seven states, the reserve and one working room");
    check(auto_keep_bytes(1, state, reserve) == reserve + (size_t(256) << 20), "one sequence keeps the reserve only (long prompts borrow banks)");
    check(auto_keep_bytes(2, state, reserve) >= state + reserve, "several sequences keep every state beyond the warm-up's");
    // 3090 np 2 (11-02): measured state 1212 MB, reserve 512 MB: kept 3448 MB and 3.35 GB stayed idle with both decoding
    check(auto_keep_bytes(2, size_t(1212) << 20, size_t(512) << 20) == size_t(1212 + 512 + 256) << 20,
          "np 2 keeps one extra sequence's state (the warm-up holds the first while the banks are sized)");

    // prefetch at np > 1 (11-03): a batched block of np sequences x 4 rows predicts like one sequence's 4 rows
    check(predicts(4, 4, 1), "np 1: a 4-row verify block prefetches");
    check(predicts(8, 4, 2), "np 2: two 4-row blocks in one step prefetch");
    check(predicts(32, 4, 8), "np 8: eight 4-row blocks prefetch");
    check(!predicts(1024, 4, 1), "a prompt chunk does not");
    check(prefetch_ranks(1).last == 6 && prefetch_ranks(2).last == 3 && prefetch_ranks(4).last == 3, "np > 1 prefetches each token's 3 best only");
    check(next_cpu_budget(4, true, 0, 4 * 2) == 5 && next_cpu_budget(4, true, 0) == 4, "np 2: the CPU miss budget may grow past one sequence's 4");

    // K/V mapping (3090 np 2, 11-03): np sequences at context C / np must map no more K/V than one sequence at C (the
    // same positions); a fixed 32 MB page made every sequence map 26 x 32 MB whatever its context (832 MB at 4K)
    {
        const size_t MB = size_t(1) << 20, g = 2 * MB, row = 1024;  // Flash-Next: 2 KV heads x 256 x BF16 per position
        const int buffers = 26;                                      // 12 attention layers + MTP, K and V each
        auto span = [&](size_t n, size_t page) { return (n + page - 1) / page * page; };
        auto seq_bytes = [&](int ctx, bool old_rule) {
            const size_t n = size_t(ctx) * row;
            const size_t page = old_rule ? std::max(g, (size_t(32) << 20) / g * g) : kv_page_bytes(n, g);
            return size_t(buffers) * span(n, page);
        };
        for (int C : {32768, 65536, 131072})
            for (int np : {2, 4, 8}) {
                const size_t one = seq_bytes(C, false), many = size_t(np) * seq_bytes(C / np, false);
                check(many <= one + size_t(np * buffers) * g, "np sequences at C / np map no more K/V than one at C (+1 granule per buffer)");
            }
        check(8 * seq_bytes(65536 / 8, true) > 2 * seq_bytes(65536, true), "the old 32 MB page: np 8 at 8K mapped over twice one 64K sequence");
        check(seq_bytes(4096, false) == size_t(buffers) * 4 * MB, "a 4K sequence maps 4 MB per buffer, not 32");
        check(kv_page_bytes(size_t(64) << 20, g) == size_t(32) << 20, "long buffers keep 32 MB pages (host moves)");
        check(kv_page_bytes(1, g) == g && kv_page_bytes(5 * MB, g) == 6 * MB, "short buffers: one granule-rounded page");
        check(kv_page_bytes(5 * MB, 0) == 5 * MB, "no VMM granule: the buffer itself");
    }
    // batched steps at high np (3090, 11-03): fewer drafts per slot instead of one slot at a time
    check(batch_drafts(2, 3, 10, 145, 512) == 3, "np 2, 145 slots: 8 rows x 10 fit, full drafts");
    check(batch_drafts(4, 3, 10, 114, 512) == 1, "np 4, 114 slots: 2 rows per slot (80 routed) fit, 4 would not (160)");
    check(batch_drafts(8, 3, 10, 120, 512) == 0, "np 8, 120 slots: one token per slot (80 routed) batched");
    check(batch_drafts(8, 3, 10, 51, 512) == -1, "np 8, 51 slots: not even one token per slot fits: one at a time");
    // np 8 (3090, 11-03): 122 slots on average but 86 in the smallest layer; sized by the average, 6 slots x 3 rows routed up
    // to 180 experts there: waves path, 1-2 of 8 answers differed from np 1
    check(batch_drafts(6, 3, 10, 86, 512) == 0 && 6 * (0 + 1) * 10 <= 86, "np 6, smallest layer 86: one token per slot");
    check(batch_drafts(8, 3, 10, 86, 512) == 0, "np 8, smallest layer 86: 80 routed fit");
    check(batch_drafts(4, 3, 10, 0, 512) == -1, "a lent bank (0 slots): one slot at a time");
    // 4+ sequences: at most 2 drafts each (GB10 np 8, 11-11: 2 drafts +3..6 % over 3, texts = np 1)
    check(batch_drafts(8, 5, 10, 512, 512) == 2, "every expert resident (GB10), 8 sequences: 2 drafts");
    check(batch_drafts(16, 5, 10, 512, 512) == 2, "16 sequences: 2 drafts (48 rows within 64)");
    check(batch_drafts(4, 3, 10, 512, 512) == 2 && batch_drafts(3, 3, 10, 512, 512) == 3, "4 sequences capped at 2, 3 keep their drafts");
    check(batch_drafts(2, 3, 10, 512, 512) == 3, "2 sequences keep their drafts");
    check(batch_drafts(1, 3, 10, 10, 512) == 3, "one sequence: its own drafts");
    // CPU misses: miss-driven on discrete cards (any non-resident expert), the static eighth on small cards (F26)
    {
        int owned = 0; for (int e = 0; e < 512; ++e) owned += cpu_owned(3, e);
        check(owned > 0 && owned < 512, "static partition: a fixed share of each layer");
        int e_res = 0; while (e_res < 512 && !cpu_owned(3, e_res)) ++e_res;  // an owned expert
        check(runs_on_cpu(3, e_res, true, true) && !runs_on_cpu(3, e_res, true, true, true), "an owned, resident expert: CPU when static, GPU when miss-driven");
        int e_not = 0; while (e_not < 512 && cpu_owned(3, e_not)) ++e_not;      // a not-owned expert
        check(!runs_on_cpu(3, e_not, false, true) && runs_on_cpu(3, e_not, false, true, true), "a not-owned miss: PCIe when static, CPU when miss-driven");
        check(!runs_on_cpu(3, e_not, false, false, true), "no pool for this call: never the CPU");
    }
    // adaptive CPU miss budget: the pool takes only what it finishes inside the GPU's work for the call
    check(next_cpu_budget(4, false, 0) == 3, "the GPU was idle when the pool finished: one expert fewer");
    check(next_cpu_budget(3, true, 0) == 4 && next_cpu_budget(4, true, 0) == 4, "the GPU still busy: one more, capped");
    check(next_cpu_budget(1, false, 0) == 0, "still late at one: every miss over PCIe");
    check(next_cpu_budget(0, false, 10) == 0 && next_cpu_budget(0, false, 64) == 1, "at zero: a probe call every 64 tries one again");
    // np 8 (11-05): ~16 misses per layer call; while the GPU stays busy the share climbs to the batch cap (4 per sequence)
    { int b = 4; for (int k = 0; k < 40; ++k) b = next_cpu_budget(b, true, 0, 4 * 8); check(b == 32, "np 8: the share reaches the batch cap while the GPU stays busy"); }
    { int b = 32; for (int k = 0; k < 40; ++k) b = next_cpu_budget(b, k % 2 == 0, 0, 32); check(b >= 31 && b <= 32, "alternating busy / idle holds the share"); }
    // slot tiers: the floor scales with the slot total, never exceeds the target, never drops below 64 (F39: a 6-row verify
    // routes to at most 60 distinct experts, so the split path holds at every bank size)
    check(slot_floor(157, 144, 157) == 144 && slot_floor(247, 144, 157) == 226, "floor scales each layer's target like the total");
    check(slot_floor(70, 40, 157) == 64 && slot_floor(50, 40, 157) == 50, "floor never below 64, never above the target");
    check(verify_slot_min(5, 10) == 64 && verify_slot_min(3, 10) == 40, "verify floor: 6 rows x top-10 -> 64, 4 rows -> 40");
    check(ep_result_cap(10) == 7 && ep_result_cap(80) == 50 && ep_result_cap(240) == 150 && ep_result_cap(1) == 1 && ep_result_cap(0) == 0,
          "--split experts result cap: 5/8 of the assignments, rounded up, never more than all of them");
    {   // --split experts forward state (3 trunk layers, rows 2)
        int announced = 0; bool answer = true;
        auto ann = [&](int) { ++announced; return answer; };
        auto layer = [&](EpForwardState & st, int l, bool can = true, bool alive = true) { const auto c = ep_before_layer(st, l, 2, can, alive, ann); if (c != EpCall::Fail) ep_after_layer(st, l, 3); return c; };
        auto whole = [&](EpForwardState & st, EpCall want) { bool ok = true; for (int l = 0; l < 3; ++l) ok = ok && layer(st, l) == want; return ok && !st.open; };
        EpForwardState s;
        check(whole(s, EpCall::Exchange) && announced == 1 && whole(s, EpCall::Exchange) && announced == 2, "--split experts: an accepted forward exchanges every layer, announced once");
        answer = false;
        check(whole(s, EpCall::Local) && announced == 3 && !s.off, "a declined forward runs locally; the next is asked again");
        answer = true; s = {}; announced = 0;
        check(ep_before_layer(s, 0, 2, true, true, ann) == EpCall::Exchange && ep_before_layer(s, 0, 2, true, true, ann) == EpCall::Exchange && announced == 1,
              "a layer 0 re-run after a failed capture is not announced twice");
        s = {}; layer(s, 0); layer(s, 1);
        check(layer(s, 2, true, false) == EpCall::Fail && s.off && layer(s, 0) == EpCall::Local, "rank 1 gone inside an accepted forward: that request fails, then local");
        s = {}; layer(s, 0); layer(s, 1);  // the forward failed at layer 2 (its send threw: rank 1 went away)
        check(ep_before_layer(s, 0, 2, true, false, ann) == EpCall::Local && s.off, "rank 1 gone, next forward: local, not a second failed request (c5e139d9)");
        s = {}; layer(s, 0); layer(s, 1);  // an accepted forward left open while rank 1 is alive (it still waits for layer 2)
        announced = 0;
        check(layer(s, 0) == EpCall::Local && s.off && announced == 0, "an accepted forward left open: the next runs locally, nothing announced (no out-of-step exchange)");
        answer = false; s = {}; layer(s, 0); layer(s, 1); answer = true; announced = 0;
        check(layer(s, 0) == EpCall::Exchange && announced == 1 && !s.off, "a declined forward left open: rank 1 waits for nothing, the next forward is asked");
        s = {}; layer(s, 0);
        check(layer(s, 1, false) == EpCall::Fail && s.off, "an accepted forward whose layer cannot exchange fails");
        s = {};
        check(layer(s, 0, false) == EpCall::Local && !s.open && !s.off && layer(s, 1) == EpCall::Local, "layer 0 cannot exchange: the forward is not announced, all local");
        s = {}; layer(s, 0);
        check(layer(s, 2) == EpCall::Fail && s.off, "an accepted forward's layer out of order: fails, parallelism off");
    }
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
        const DraftPlan small = one_sequence_drafts(m, kCpuPoolRows);
        check(small.drafts + 1 <= kCpuPoolRows && !small.adaptive, "CPU pool: the verify block fits the pool's rows, fixed length");
        check(small.drafts >= 1 && small.drafts <= m, "CPU pool: at least one draft, never more than the manifest's");
    }
    check(cpu_pool_rows(0, true) == 0 && cpu_pool_rows(15, true) == kCpuPoolRows && cpu_pool_rows(15, false) == kCpuPoolMaxRows,
          "pool rows: no pool, none; a small card's 4-row block; a discrete card's 64");
    check(one_sequence_drafts(3, 0).drafts == 5 && one_sequence_drafts(3, 0).adaptive, "no CPU pool: adaptive, up to 5");
    check(one_sequence_drafts(7, 0).drafts == 7, "no CPU pool: a larger manifest count stays");
    check(one_sequence_drafts(3, 0, true).drafts == 7 && one_sequence_drafts(3, 0, true).adaptive, "every expert resident: adaptive, up to 7");
    check(one_sequence_drafts(3, kCpuPoolRows, true).drafts + 1 <= kCpuPoolRows, "CPU pool wins over residency");
    {  // copy drafts
        const std::vector<int> ctx = {5, 1, 2, 3, 9, 8, 7, 4, 1, 2, 3, 6, 6, 1, 2, 3};
        check(copy_draft(ctx, {}, 3, 4) == std::vector<int>({6, 6, 1, 2}), "copy: the latest earlier match wins");
        check(copy_draft(ctx, {}, 3, 1) == std::vector<int>({6}), "copy: at most max_len tokens");
        const std::vector<int> fresh = {7, 7, 2, 3, 1};
        check(copy_draft(fresh, {}, 3, 4).empty(), "copy: no earlier match, no draft");
        const std::vector<int> old = {0, 7, 2, 3, 1, 4, 4, 4};
        const std::vector<int> fresh2 = {9, 3, 1, 4};
        check(copy_draft(fresh2, {&old}, 3, 3) == std::vector<int>({4, 4}), "copy: an earlier request continues it");
        check(copy_draft(fresh2, {&old}, 3, 9) == std::vector<int>({4, 4}), "copy: ends at the history's end");
        const std::vector<int> self = {3, 1, 4, 3, 1, 4};
        check(copy_draft(self, {&old}, 3, 2) == std::vector<int>({3, 1}), "copy: the context itself before the history");
        check(copy_draft({1, 2}, {&old}, 3, 2).empty(), "copy: a context shorter than n");
    }
    check(auto_slot_drafts(3, 5, 512, 512, 0) == 7, "slots=auto with every expert resident drafts up to 7 (GB10 drafted 5)");
    check(auto_slot_drafts(3, 5, 159, 512, 0) == 5, "slots=auto on a 24 GB card keeps 5");
    check(auto_slot_drafts(3, 3, 512, 512, kCpuPoolRows) == 3, "CPU pool keeps its block");
    check(auto_slot_drafts(0, 0, 512, 512, 0) == 0, "no MTP: no drafts");
    check(one_sequence_drafts(0, 0).drafts == 0 && !one_sequence_drafts(0, kCpuPoolRows).adaptive, "no MTP: no drafts");
    // a small card that must run exact: the CPU pool never runs, so the pool's row limit must not hold the block to 3
    // drafts (16 GB 3090 IQ3_S, 11-11: decode 78.6 -> 89.3 tok/s at 1.2K, the same 192 greedy tokens)
    check(sequence_drafts(3, true, -1, 32).drafts == 5 && sequence_drafts(3, true, -1, 32).adaptive,
          "--exact on: no CPU pool, so the adaptive 5-draft block (not the pool's 3)");
    check(sequence_drafts(3, false, -1, 32).drafts + 1 <= kCpuPoolRows && !sequence_drafts(3, false, -1, 32).adaptive,
          "--exact off: the pool runs, the block stays inside its rows");
    check(sequence_drafts(5, false, 0, 32).drafts == 5 && sequence_drafts(5, false, 0, 32).adaptive,
          "TQFN_CPU_MISS=0 with --exact off: no pool either, adaptive drafts");
    check(sequence_drafts(3, false, -1, 4).drafts == 3, "--exact off on 4 cores: one pool thread, the 4-row block");
    // prefill chunk of a small card: the prompt's own length, 1 KiB up, never above the cap
    check(small_card_chunk(1218, 32768) == 2048, "1.2K prompt: a 2K chunk, not the engine's 16K");
    check(small_card_chunk(8520, 32768) == 9216, "8.5K prompt: its own length, one chunk");
    check(small_card_chunk(25319, 32768) == 25600, "25.3K prompt: 25.6K, the largest the plan can take");
    check(small_card_chunk(25319, 16384) == 16384 && small_card_chunk(40000, 32768) == 32768, "capped at the max chunk");
    check(small_card_chunk(1, 32768) == 1024 && small_card_chunk(0, 32768) == 0, "tiny and empty prompts");

    // parking the LM heads for a multi-chunk prompt: only when their room can make the whole prompt one chunk (the pin +
    // unpark is ~1.1 s of TTFT, and a 25.3K prompt at 16 GB cannot reach one chunk with 11.1 GB free)
    {
        const size_t mb = size_t(1) << 20;
        check(!park_heads_for_one_chunk(12405 * mb, 11098 * mb, 634 * mb, 0), "16 GB 25.3K: one chunk needs 12.4 GB of 11.7 GB -> no park");
        check(park_heads_for_one_chunk(12405 * mb, 11900 * mb, 634 * mb, 0), "12.5 GB free + the heads: the probe can succeed, park");
        check(park_heads_for_one_chunk(12405 * mb, 11098 * mb, 634 * mb, 700 * mb), "a trim that hands back 700 MB counts too");
        check(park_heads_for_one_chunk(9240 * mb, 9000 * mb, 634 * mb, 0), "an 8.5K prompt fits one chunk with the heads' room");
        check(park_heads_for_one_chunk(12405 * mb, 11098 * mb, 0, 0) && park_heads_for_one_chunk(0, 11098 * mb, 634 * mb, 0),
              "nothing known (non-resident heads, no plan): keep the old probe");
    }

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
        // staging slices: smaller for a prompt the engine chunk covers (the room goes to resident banks), the measured
        // 128-expert slices for a longer prompt (its plan and its text are the measured ones)
        check(small_card_stage(true, false, 128, 1220, 16384) == 64 && small_card_stage(true, false, 128, 8520, 16384) == 64,
              "small card, prompt inside the engine chunk: 64-expert slices");
        check(small_card_stage(true, false, 128, 25319, 16384) == 128 && small_card_stage(false, false, 128, 1220, 16384) == 128,
              "longer prompt or a big card: the engine's slices");
        check(small_card_stage(true, true, 96, 1220, 16384) == 96 && small_card_stage(true, false, 32, 1220, 16384) == 32,
              "--prefill-stage wins; an already smaller setting stays");
        // MTP catch-up piece: the MTP experts stream once per piece, so a 25K prompt takes 4K pieces, a short one its own size
        check(mtp_piece_tokens(25319) == 4096 && mtp_piece_tokens(9000) == 4096, "a long prompt: 4096-position MTP pieces");
        check(mtp_piece_tokens(1000) == 1000 && mtp_piece_tokens(0) == 1, "short or empty: one piece of its own size");
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
    {   // --prompt-cache: an agent's next turn = the last prompt (its tail re-rendered) + a new message
        const std::vector<int> seq{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        const int snap = snapshot_at(10, 0, 2);
        check(snap == 8 && snapshot_at(10, 7, 2) == 0 && snapshot_at(3, 0, 2) == 0, "snapshot kTail before the end, only when the new part is >= 2 x kTail");
        check(reusable_prefix(seq, snap, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 11}) == 8, "next turn: equal through the token at snap -> reuse [0, snap)");
        check(reusable_prefix(seq, snap, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 42, 10}) == 0, "the token at snap differs (MTP pair snap - 1): no reuse");
        check(reusable_prefix(seq, snap, std::vector<uint32_t>{1, 2, 3, 0, 5, 6, 7, 8, 9}) == 0, "an earlier token differs: no reuse");
        check(reusable_prefix(seq, snap, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8}) == 0, "a prompt no longer than snap: no reuse");
        check(reusable_prefix(seq, 0, std::vector<uint32_t>{1, 2, 3}) == 0, "no snapshot: no reuse");
    }
    check(qsa_key_ring_cells(true, 32768) == 32768 && qsa_key_ring_cells(false, 32768) == 0, "QSA key ring + host mirror on discrete GPUs only (GB10: the full array, no mirror)");
    check(keep_prefill_buffers(true, size_t(26) << 30) && !keep_prefill_buffers(false, size_t(26) << 30), "GB10 keeps the engine's prefill buffers between requests; discrete cards hand them to the banks");
    check(!keep_prefill_buffers(true, size_t(16) << 30), "GB10 below its 20 GB margin hands them back too");
    check(dense_pool_for(size_t(600) << 20, 256, 2048) == (size_t(600) << 20) && dense_pool_for(size_t(600) << 20, 256, 120) == 0 &&
          dense_pool_for(size_t(600) << 20, 0, 4096) == 0, "the dense FP16 pool only from --dense-min tokens (a short prompt keeps decode's memory)");
    check(layer_graphs(true, 12) && layer_graphs(true, 11), "unified memory (GB10) keeps the decode graphs");
    check(preload_all_experts(512, 512) && preload_all_experts(600, 512) && !preload_all_experts(168, 512) && !preload_all_experts(512, 0),
          "banks that hold every expert are filled at start-up; smaller banks keep the first-use fill");
    check(layer_graphs(false, 8) && layer_graphs(false, 9), "RTX 3090 / 4090 (sm_86 / sm_89) keep the decode graphs");
    check(!layer_graphs(false, 12), "RTX 5090 (sm_120, PCIe) launches the kernels directly");
    // the pinned residual buffer of a multi-chunk prefill: pinned at start-up when the host easily holds it (pinning it per
    // request cost a 32K prompt 0.72 s of time to first token), else per request as before
    check(pin_residual_at_start(6 * GB, 125 * GB, 40 * GB, false), "3090 desktop, 125 GB RAM, 40 GB available: 6 GB pinned at start");
    check(!pin_residual_at_start(6 * GB, 109 * GB, 18 * GB, false), "pod, 18 GB available: pinned per request");
    check(!pin_residual_at_start(6 * GB, 48 * GB, 30 * GB, false), "48 GB machine: never more than a tenth of RAM pinned for it");
    check(!pin_residual_at_start(0, 125 * GB, 40 * GB, false), "single-chunk contexts need no buffer");
    check(!pin_residual_at_start(6 * GB, 121 * GB, 116 * GB, true), "GB10 (unified memory): pinned memory is the GPU's memory (759 MB left after warm-up): per request");
    // host K/V for old positions: default on for discrete GPUs with VMM (RTX 3090, 256K: 52-56 -> 65-71 tok/s; <= 64K moves
    // nothing), off on unified memory (GB10: host and device are the same memory); TQFN_KV_HOST=0 / 1 forces it
    check(kv_host_on(nullptr, true, false), "3090 / 5090 with VMM: on by default");
    check(!kv_host_on(nullptr, true, true), "GB10 (unified memory): off by default");
    check(!kv_host_on(nullptr, false, false), "no VMM granule: plain allocations");
    check(!kv_host_on("0", true, false) && kv_host_on("1", true, true), "TQFN_KV_HOST=0 / 1 forces it");
    // approximate tail skip (Strata's default route-tail-skip): a token's expert ranked R-th or lower that is not resident is
    // replaced by the token's first choice with weight 0; never when the first choice itself would be skipped; any block size
    {
        const int topk = 4;  // two tokens x top-4: token 0 routes 1,2,3,4; token 1 routes 5,6,7,8
        const std::vector<int> routed{1, 2, 3, 4, 5, 6, 7, 8};
        auto resident = [](int e) { return e != 3 && e != 4 && e != 5 && e != 8; };  // 3, 4, 5, 8 would be misses
        const auto skip = tail_skip_targets(routed, topk, 3, resident);  // R = 3: ranks 3 and 4 (0-based 2, 3) may be skipped
        check(skip == std::vector<int>{-1, -1, 1, 1, -1, -1, -1, 5}, "missing ranks 3-4 go to the token's first choice (5 is loaded "
              "anyway, missing or not)");
        check(tail_skip_targets(std::vector<int>{1, 3, 9, 4, 6, 7, 3, 2}, topk, 3, resident)[6] == -1,
              "expert 3 is rank 2 for token 1 but rank 1 for token 0 (its best rank): kept");
        std::vector<int> big(8 * topk); for (int i = 0; i < int(big.size()); ++i) big[size_t(i)] = i;
        const auto eight = tail_skip_targets(big, topk, 3, [](int e) { return e % topk == 0; });
        check(std::count(eight.begin(), eight.end(), -1) == 8 * topk - 16, "an 8-token verify block skips too (the old rule stopped at 4 tokens)");
        check(tail_skip_targets(routed, topk, 0, resident) == std::vector<int>(routed.size(), -1), "R = 0: off");
    }
    // np 10 / 12 with pinned per-layer slots: every layer at least sequences x topk, total unchanged (11-05)
    {
        std::vector<int> caps = {60, 80, 200, 180, 90, 150};  // total 760, avg 126
        const auto lifted = lift_layer_floor(caps, 100, 512);
        int tot = 0, lo = 1 << 30; for (int c : lifted) { tot += c; lo = std::min(lo, c); }
        check(tot == 760 && lo >= 100, "np 10: every layer at least 100 slots, total kept");
        check(batch_drafts(10, 3, 10, 60, 512) == -1 && batch_drafts(10, 3, 10, lo, 512) == 0, "np 10 batches one row per slot instead of stepping one slot at a time");
        check(lift_layer_floor(caps, 200, 512) == std::vector<int>(6, 126) || [&] { auto x = lift_layer_floor(caps, 200, 512); int t = 0; for (int c : x) t += c; return t == 760; }(), "floor above the average: uniform split");
        check(lift_layer_floor(caps, 0, 512) == caps, "one sequence: caps unchanged");
        const auto big = lift_layer_floor({300, 20, 300}, 40, 512);
        check(big[1] == 40 && big[0] + big[2] == 580, "the slots come from the largest layers");
    }
    {   // CPU miss pool: --exact off alone starts it, and then it must read the pinned copy (11-06 np 8: 68 / 70 vs 76 / 77)
        check(cpu_miss_threads(-1, false, 32) == 15 && cpu_miss_threads(-1, true, 32) == 0 && cpu_miss_threads(14, true, 32) == 14,
              "pool threads: env wins, --exact off min(15, cores / 2 - 1), --exact on none");
        check(cpu_pool_reads_packed(-1, false, 32), "--exact off without TQFN_CPU_MISS: the pool reads the pinned packed experts");
        check(!cpu_pool_reads_packed(-1, true, 32) && !cpu_pool_reads_packed(0, false, 32), "no pool, no packed view");
        check(cpu_miss_threads(-1, false, 4) == 1, "small CPU: at least one thread");
    }
    {   // the built-in Flash-Next routing profile (manifests without layer_slots, e.g. Unsloth files)
        const std::string prof = flashnext_layer_profile(48, 512);
        int n = prof.empty() ? 0 : 1; for (char ch : prof) n += ch == ',';
        check(n == 48, "Flash-Next profile: one entry per layer");
        check(std::string(flashnext_layer_profile(40, 512)).empty() && std::string(flashnext_layer_profile(48, 256)).empty(), "only for 48 layers x 512 experts");
    }
    std::printf("decode rules: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

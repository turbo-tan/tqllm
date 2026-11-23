#include <algorithm>
// Flash-Next decode rules (no CUDA): the F39 verify path choice, the F38 slots=auto keep, the adaptive draft stop.
#include "models/qwen/qwen4exp/decode_rules.hpp"
#include "experts/packed_span.hpp"
#include <cstdio>
#include <memory>
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
        check(kv_page_bytes(1, g) == g && kv_page_bytes(5 * MB, 0) == 5 * MB, "a granule at least; no VMM granule: the buffer itself");
        // 3090 VRAM account 11-19: a 32K request's K/V (36,864 rows: prompt + max tokens + the next turn's room) mapped
        // 2 x 32 MB per buffer, 1664 MB for 936 MB of rows; 2 MB pages map the rows rounded to 2K positions
        const size_t n32 = size_t(36864) * row;
        check(span(n32, kv_page_bytes(n32, g)) == n32, "a 32K request maps its 36 MB per buffer, not 64");
        check(size_t(buffers) * span(n32, kv_page_bytes(n32, g)) <= size_t(936) * MB, "26 buffers: 936 MB at 32K (was 1664)");
        for (size_t n : {size_t(1), 5 * MB, n32, size_t(131072) * row, size_t(262144) * row + 1})
            check(span(n, kv_page_bytes(n, g)) - n < g, "never a page past the buffer's last granule");
        // after a long prompt (kv_to_host): pages wholly older than the newest 32K rows go to host, the device keeps
        // 32K .. 34K rows of each buffer (32 MB pages kept 32K .. 64K: 1792 MB of K/V at 128K for a 32K window)
        const size_t per = kv_page_bytes(size_t(128000) * row, g) / row, len = 123254, keep = 32768;
        const size_t front = kv_host_front(len, keep, per);
        check(per == 2048 && len - front * per >= keep && len - front * per < keep + per, "128K: the device keeps 32K .. 34K rows");
        check(kv_host_front(len, keep, 32768) == 2 && len - 2 * 32768 > 57000, "the old 32 MB page kept 57K of 123K rows on the device");
        check(kv_host_front(20000, keep, per) == 0 && kv_host_front(len, keep, 0) == 0, "shorter than the window: nothing moves");
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
    check(dflash_tau(true) == 0.35f && dflash_tau(false) == 0.f, "DFlash2 chain cut: 0.35 with every expert resident, full chain on discrete cards");

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
    check(one_sequence_drafts(3, kCpuPoolMaxRows).drafts == 5 && one_sequence_drafts(3, kCpuPoolMaxRows).adaptive,
          "discrete card's pool (64 rows): adaptive, up to 5 (was capped at 3 fixed)");
    check(sequence_drafts(3, false, 15, 32, false, false).adaptive && !sequence_drafts(3, false, 15, 32, false, true).adaptive,
          "--exact off: a discrete card keeps the adaptive block, a small card the pool's 4 rows");
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
    if (kCpuMissPoolBuilt) check(sequence_drafts(3, false, -1, 32).drafts + 1 <= kCpuPoolRows && !sequence_drafts(3, false, -1, 32).adaptive,
                                 "--exact off: the pool runs, the block stays inside its rows");
    else check(sequence_drafts(5, false, -1, 32).drafts == 5 && sequence_drafts(5, false, -1, 32).adaptive,
               "arm64 --exact off: no CPU pool, so the adaptive block as with --exact on");
    check(sequence_drafts(5, false, 0, 32).drafts == 5 && sequence_drafts(5, false, 0, 32).adaptive,
          "TQFN_CPU_MISS=0 with --exact off: no pool either, adaptive drafts");
    if (kCpuMissPoolBuilt) check(sequence_drafts(3, false, -1, 4).drafts == 3, "--exact off on 4 cores: one pool thread, the 4-row block");
    // prefill chunk of a small card: the prompt's own length, 1 KiB up, never above the cap
    check(small_card_chunk(1218, 32768) == 2048, "1.2K prompt: a 2K chunk, not the engine's 16K");
    check(small_card_chunk(8520, 32768) == 9216, "8.5K prompt: its own length, one chunk");
    check(small_card_chunk(25319, 32768) == 25600, "25.3K prompt: 25.6K, the largest the plan can take");
    check(small_card_chunk(25319, 16384) == 16384 && small_card_chunk(40000, 32768) == 32768, "capped at the max chunk");
    check(small_card_chunk(1, 32768) == 1024 && small_card_chunk(0, 32768) == 0, "tiny and empty prompts");
    check(short_prompt_chunk(300, 16384) == 2048 && short_prompt_chunk(2049, 16384) == 3072, "short prompt: its own length, at least 2K");
    check(short_prompt_chunk(16384, 16384) == 16384 && short_prompt_chunk(16385, 16384) == 16384 && short_prompt_chunk(0, 16384) == 16384,
          "a prompt of a chunk or more keeps the engine chunk");

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
        {   // experts read from the GGUF on demand (31 GB RAM: no pack): every routed expert goes through the bank, which holds
            // fewer than a long chunk routes (.77 rc170d: any prompt of 256+ tokens failed "active set exceeds slots")
            const std::vector<int> hot{1, 2, 3, 4, 5, 6, 7};
            const auto ps = bank_passes(hot, 3, [](int e) { return e == 5 || e == 6; });
            bool fits = !ps.empty(); std::vector<int> all;
            for (const auto & p : ps) { fits = fits && !p.empty() && p.size() <= 3; all.insert(all.end(), p.begin(), p.end()); }
            std::sort(all.begin(), all.end());
            check(fits && all == hot, "bank passes: every routed expert once, no pass larger than the bank");
            check(ps.size() == 3 && ps[0].size() == 3 && std::count(ps[0].begin(), ps[0].end(), 5) && std::count(ps[0].begin(), ps[0].end(), 6),
                  "bank passes: resident experts in the first pass (no load)");
            check(bank_passes(hot, 7, [](int) { return false; }) == std::vector<std::vector<int>>{hot}, "bank passes: one pass when the bank holds them all");
        }
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
    // a 64 GB scope (68.7 GB limit, 67.8 GB available at load): the pack + 3.8 GB prompt buffers + 1.4 GB MTP + 5 GB reserve fit
    check(build_pack_in_memory(false, false, false, 51 * GB * 1.02, 68.7 * GB, 67.8 * GB, 10.2 * GB), "64 GB scope: the pack fits beside a full prompt");
    check(!build_pack_in_memory(false, false, false, 51 * GB * 1.02, 31 * GB, 29 * GB, 10.2 * GB), "31 GB: no pack (host tier)");
    check(!build_pack_in_memory(false, false, false, 51 * GB * 1.02, 64 * GB, 58 * GB, 10.2 * GB), "a box with 58 GB available: no pack");
    check(warm_prefill_chunk({1}) && warm_prefill_chunk({1, 1, 1}), "warm-up chunk where every rank reads packed experts");
    {   // split RAM cap: .77 (31 GB, ~28 GB available), 20 GB reserve; at 128K the head (last rank) holds 5.0 GB of residual rows
        // and 2.0 GB of embeddings; the K/V tail adds 0.05 byte per pack byte
        const uint64_t GiB = uint64_t(1) << 30;
        const uint64_t head = 7 * GiB, ram = 31 * GiB, avail = 28 * GiB;
        const uint64_t c_head = split_ram_cap(avail, ram, kSplitHostReserve, head, 0.05), c_mid = split_ram_cap(avail, ram, kSplitHostReserve, 0, 0.05);
        check(c_head < GiB && c_head + head + kSplitHostReserve <= avail, "31 GB head rank at 128K: under 1 GB of pack, 20 GB stay");
        check(c_mid > 7 * GiB && c_mid < 8 * GiB, "31 GB middle rank: about 7.6 GB of pack");
        check(split_ram_cap(110 * GiB, 125 * GiB, kSplitHostReserve, head, 0.05) == uint64_t(0.6 * double(125 * GiB) / 1.02), "125 GB box: the pinned pack rule binds");
        check(split_ram_cap(15 * GiB, ram, kSplitHostReserve, 0, 0) == 0, "less available than the reserve: no pack");
        // with the fixed per-rank cost (4 GB + the 1.5 GB MTP pack) the 128K head rank on a 31 GB box has no room at all
        check(split_ram_cap(31 * GiB, ram, kSplitHostReserve, head, 0.05, kSplitHostFixed + 3 * GiB / 2) == 0,
              "31 GB head rank at 128K with the fixed cost: no room (refused at start)");
    }
    check(!warm_prefill_chunk({1, 0}) && !warm_prefill_chunk({0}), "--split layer: one rank without a pack skips it on every rank");
    // 2x RTX 4090, 124 GB (11-18): each --split layer rank packed all 48 layers (2 x 51 GB); a rank holds its own half
    {
        const std::vector<uint64_t> base = {0, 100, 200, 300}, block = {10, 10, 10, 10};
        const auto all = tqllm::packed_span(base, block, 10, 0, 4), r1 = tqllm::packed_span(base, block, 10, 2, 4), r0 = tqllm::packed_span(base, block, 10, 0, 2);
        check(all.first == 0 && all.second == 400, "packed_span: every layer");
        check(r0.first == 0 && r0.second == 200 && r1.first == 200 && r1.second == 400, "packed_span: each rank its own half, no overlap");
        check(build_pack_in_memory(false, false, false, 25.5 * GB, 124 * GB / 2), "2 ranks on 124 GB: each packs its 25.5 GB half");
        check(!build_pack_in_memory(false, false, false, 51 * GB, 124 * GB / 2), "a whole 51 GB pack per rank would not fit its share");
    }
    check(!build_pack_in_memory(false, false, true, 51 * GB, 128 * GB), "hot/cold manages expert memory: no pack built");
    check(!build_pack_in_memory(false, false, false, 51 * GB, 64 * GB) && !build_pack_in_memory(false, false, false, 51 * GB, 24 * GB),
          "a 51 GB pack is not pinned on a 64 GB or 24 GB machine");
    {   // host tier under the banks (experts read from the GGUF on demand), .77: 29 GB available, 2 MB blocks
        const uint64_t G = uint64_t(1) << 30, blk = 2070000;
        const int n = host_tier_blocks(29 * G, 7 * G, G, kHostTierReserve, blk, 16080, 512);
        check(n == int((29 * G - 7 * G - G - kHostTierReserve) / blk), "31 GB box: the tier takes what the prompt buffers, ring and reserve leave");
        check(host_tier_blocks(200 * G, 7 * G, G, kHostTierReserve, blk, 16080, 512) == 16080, "never more blocks than the experts off the card");
        check(host_tier_blocks(13 * G, 7 * G, G, kHostTierReserve, blk, 16080, 512) == 0, "no tier when the RAM is taken (below min blocks)");
        check(host_tier_blocks(10 * G, 0, 0, kHostTierReserve, 0, 16080, 512) == 0, "no block size: no tier");
    }
    check(read_rows_randomly("per_layer_token_embd.weight", 28.8 * GB, 31 * GB) && !read_rows_randomly("per_layer_token_embd.weight", 28.8 * GB, 125 * GB)
          && !read_rows_randomly("token_embd.weight", 2 * GB, 31 * GB), "31 GB box: the 28.8 GB per-layer embedding is read row by row, no read-ahead; 125 GB warms it");
    {   // layer-outer prefill: the chunks before the snapshot boundary, each 256+ tokens, at least two
        using S = std::vector<std::pair<int, int>>;
        check(layer_outer_chunks(S{{0, 16384}, {16384, 31985}, {31985, 32049}}, 31985) == 2, "32K: both 16K chunks, not the 64-token tail");
        check(layer_outer_chunks(S{{31985, 48369}, {48369, 64753}, {64753, 81137}, {81137, 97521}, {97521, 113905}, {113905, 122678}, {122678, 122742}}, 122678) == 6,
              "128K after a 32K turn: six chunks, the tail after the snapshot runs as before");
        check(layer_outer_chunks(S{{0, 16384}, {16384, 20000}}, 0) == 2, "no snapshot: every chunk");
        check(layer_outer_chunks(S{{0, 16384}, {16384, 16500}, {16500, 20000}}, 0) == 0, "a short chunk second: one layer-major chunk only, nothing to gain");
        check(layer_outer_chunks(S{{0, 9000}}, 0) == 0, "one chunk: as before");
    }

    {   // the HF manifest's 3090 plan on a GB10 (512 slots = every expert): no layer below 512
        const std::vector<int> plan{314, 291, 269, 160, 189, 219, 248, 239, 208, 135, 212, 160, 128, 145};
        const auto all = scale_layer_slots(plan, 512, 512);
        check(*std::min_element(all.begin(), all.end()) == 512, "every expert fits: each layer keeps all experts");
        const auto some = scale_layer_slots(plan, 200, 512); long sum = 0; for (int c : some) sum += c;
        check(std::abs(sum - 200L * long(plan.size())) <= long(plan.size()) && some[0] > some[9], "smaller card: plan scaled to the budget, shape kept");
    }
    {   // 2x RTX 4090 --split layer (11-14): the 48-layer profile scaled over all layers left rank 1 (layers 24-47, the colder
        // half) 1,146 slot-layers short of 381 x 24 (2.3 GB idle) and pushed rank 0 over its budget; 5 layers clamped at 512
        const std::vector<int> prof{314,291,269,160,189,219,248,239,208,135,212,160,227,227,172,128,128,145,227,169,164,197,236,141,
                                    152,134,182,137,189,167,167,128,128,135,213,143,153,170,226,128,129,140,161,131,140,140,176,165};
        auto sum = [](const std::vector<int> & v, size_t a, size_t b) { long t = 0; for (size_t i = a; i < b; ++i) t += v[i]; return t; };
        const auto all = scale_layer_slots(prof, 381, 512);
        check(std::abs(sum(all, 0, 48) - 381L * 48) <= 48, "water-filling: clamped layers' slots go to the others (total kept)");
        check(*std::max_element(all.begin(), all.end()) <= 512 && all[0] == 512, "a hot layer keeps every expert, never more");
        for (int r = 0; r < 2; ++r) {
            std::vector<bool> owned(48, false); for (size_t l = size_t(r) * 24; l < size_t(r + 1) * 24; ++l) owned[l] = true;
            const auto mine = scale_owned_layer_slots(prof, owned, 381, 512);
            const long got = sum(mine, size_t(r) * 24, size_t(r + 1) * 24);
            check(std::abs(got - 381L * 24) <= 24, r ? "split rank 1: its own 24 layers get 381 x 24 slots (was 7,998)" : "split rank 0: 381 x 24, not over budget (was 9,935)");
        }
        std::vector<bool> every(48, true);
        check(scale_owned_layer_slots(prof, every, 200, 512) == scale_layer_slots(prof, 200, 512), "one card: same as before");
    }
    check(bank_budget_bytes(true, 18.0, size_t(120) << 30) == size_t(120) << 30, "slots=auto: banks capped by the device, not budget_gb");
    check(bank_budget_bytes(false, 18.0, size_t(120) << 30) == size_t(18) << 30, "fixed slots: budget_gb as given");
    check(prefill_chunk(16384, 8) == 16384 && prefill_chunk(16384, 8192) == 16384, "prefill chunk: the engine setting, manifest key or not");
    check(prefill_chunk(0, 8192) == 8192, "prefill chunk: the manifest value only when the setting is off");
    check(drop_layer_after_pack("blk.7.ffn_gate_exps.weight", 7), "copied layer expert pages may be dropped");
    check(!drop_layer_after_pack("blk.70.ffn_gate_exps.weight", 7) &&
          !drop_layer_after_pack("blk.8.ffn_gate_exps.weight", 7), "other layers remain cached");
    check(!drop_layer_after_pack("blk.7.attn_qkv.weight", 7) &&
          !drop_layer_after_pack("per_layer_token_embd.weight", 7), "dense and embedding pages remain cached");
    check(drop_after_pack("blk.7.ffn_gate_exps.weight") && drop_after_pack("blk.47.ffn_down_exps.weight"), "trunk expert pages may be dropped after the pack");
    check(!drop_after_pack("per_layer_token_embd.weight") && !drop_after_pack("token_embd.weight") && !drop_after_pack("blk.0.attn_qkv.weight"),
          "the per-layer token embedding and dense tensors stay cached");
    check(warm_at_start("per_layer_token_embd.weight", 28.8e9, 128e9) && !warm_at_start("blk.0.ffn_gate_exps.weight", 1e9, 128e9),
          "the per-layer token embedding is read at start-up");
    check(!warm_at_start("per_layer_token_embd.weight", 28.8e9, 24.7e9), "no start-up read of a 28.8 GB embedding on a 23 GiB host");
    // a 64 GB scope after a 51 GB pinned pack: ~14 GB left, the 28.8 GB table cannot stay cached: rows read on demand
    check(!warm_at_start("per_layer_token_embd.weight", 28.8e9, 68.7e9, 14e9) && read_rows_randomly("per_layer_token_embd.weight", 28.8e9, 68.7e9, 14e9),
          "64 GB scope beside the pinned pack: no start-up read, rows read on demand");
    check(warm_at_start("per_layer_token_embd.weight", 28.8e9, 134e9, 70e9), "125 GB box after the pack: warmed as before");
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
        {   // the reply's state: prompt {1..8}, output {9, 10} (10 not fed) -> live 9; the next turn repeats prompt ++ output
            check(reusable_from(seq, 9, 6, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) == 9, "next turn = prompt ++ reply ++ message: reuse [0, live)");
            check(reusable_from(seq, 9, 6, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 99, 11}) == 6, "the last output token differs: the snapshot");
            check(reusable_from(seq, 9, 6, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8}) == 6, "the same prompt again: the snapshot");
            check(reusable_from(seq, 9, 6, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) == 9, "prompt ++ reply and nothing more: reads only the last output token");
            check(reusable_from(seq, 9, 6, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}) == 6, "a prompt no longer than live: the snapshot");
            check(reusable_from(seq, 0, 6, std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) == 6, "no live state (several slots, aborted): the snapshot");
            // a step that accepted 4 rows with room for 2 more output tokens keeps 2 (the state ends with the output)
            check(reused_read_rows(true, 0, 29, 16384) == 8 && reused_read_rows(true, 0, 300, 16384) == 16384,
                  "a short new part after the reply's state reads in decode-size steps; a long one in chunks");
            check(reused_read_rows(false, 0, 64, 16384) == 16384 && reused_read_rows(false, 8, 29, 16384) == 8,
                  "a restored tail reads with the step size of the read that took its snapshot");
            check((refill_rebuilds({3, 7, 9}, [](int l) { return l != 7; }) == std::vector<int>{3, 9}), "lent layers with a bank built mid-prompt are rebuilt by the refill");
            check(committed_rows(4, 2) == 2 && committed_rows(3, 9) == 3 && committed_rows(1, 1) == 1, "rows kept at the max_tokens cut");
        }
        // one slot keeps its conversation too (a chat's next turn read the whole conversation again before)
        check(prompt_cache_keeps(true, 1) && prompt_cache_keeps(true, 8) && !prompt_cache_keeps(false, 1) && !prompt_cache_keeps(false, 8),
              "--prompt-cache on keeps the sequence with one slot as with several");
        // its kept K/V: this request + the reply + a next message, never the whole slot context
        check(cached_sequence_rows(32000, 512, 131072) == 32000 + 512 + 16 + 4096, "kept sequence: the request plus 4096 rows for the next turn");
        check(cached_sequence_rows(32000, 8192, 131072) == 32000 + 8192 + 16 + 8192 + 2048, "kept sequence: room for a max_tokens reply and a message");
        check(cached_sequence_rows(128000, 4096, 131072) == 131072, "kept sequence: never past the slot context");
        check(kept_sequence_fits(33000, 512, 36624) && !kept_sequence_fits(36200, 512, 36624), "a turn that outgrew the kept K/V does not fit it");
        check(kept_sequence_rows(33000, 512, 36864, 131072) == 36864, "a turn that fits keeps the kept K/V as it is");
        // GB10 rc170d gate, 128K chat: turn 1 (119511 + 24) kept 123904 rows, turn 5 (122480 + 1500) needs 123996: it keeps
        // its prefix and the K/V grows (it read the whole prompt again: 64 s to the first token instead of 0.3 s)
        check(kept_sequence_rows(122480, 1500, 123904, 131072) == 131072, "a turn that outgrew the kept K/V grows it, up to the slot context");
        check(kept_sequence_rows(40000, 512, 36864, 131072) == 36864 + 36864 / 4, "the kept K/V grows by a quarter at least (a long chat does not regrow every turn)");
        check(kept_sequence_rows(60000, 512, 36864, 131072) == cached_sequence_rows(60000, 512, 131072), "a large step grows to the request plus room for the next turn");
        // the same prompt again (kTail new tokens) runs as decode steps: its host K/V stays on the host
        check(!kv_back_for_prompt(64, 16384) && !kv_back_for_prompt(255, 16384) && kv_back_for_prompt(256, 16384) && kv_back_for_prompt(32000, 8192) &&
              !kv_back_for_prompt(4096, 128), "host K/V returns to the device only for a layer-major chunk");
    }
    check(qsa_key_ring_cells(true, 32768) == 32768 && qsa_key_ring_cells(false, 32768) == 0, "QSA key ring + host mirror on discrete GPUs only (GB10: the full array, no mirror)");
    check(keep_prefill_buffers(true, size_t(26) << 30) && !keep_prefill_buffers(false, size_t(26) << 30), "GB10 keeps the engine's prefill buffers between requests; discrete cards hand them to the banks");
    check(!keep_prefill_buffers(true, size_t(16) << 30), "GB10 below its 20 GB margin hands them back too");
    check(dense_pool_for(size_t(600) << 20, 256, 2048) == (size_t(600) << 20) && dense_pool_for(size_t(600) << 20, 256, 120) == 0 &&
          dense_pool_for(size_t(600) << 20, 0, 4096) == 0, "the dense FP16 pool only from --dense-min tokens (a short prompt keeps decode's memory)");
    check(layer_graphs(true, 12) && layer_graphs(true, 11), "unified memory (GB10) keeps the decode graphs");
    check(preload_all_experts(512, 512) && preload_all_experts(600, 512) && !preload_all_experts(168, 512) && !preload_all_experts(512, 0),
          "banks that hold every expert are filled at start-up; smaller banks keep the first-use fill");
    check(refill_load_count(153, 153, 512) == 1 && refill_load_count(0, 153, 512) == 0,
          "a discrete card's lent bank is rebuilt at its slot count with one held expert; decode's misses fill the rest");
    check(refill_load_count(512, 512, 512) == 512, "a bank that holds every expert (GB10) reloads them all (device maps)");
    check(layer_graphs(false, 8) && layer_graphs(false, 9), "RTX 3090 / 4090 (sm_86 / sm_89) keep the decode graphs");
    check(!layer_graphs(false, 12), "RTX 5090 (sm_120, PCIe) launches the kernels directly");
    // the pinned residual buffer of a multi-chunk prefill: pinned at start-up when the host easily holds it (pinning it per
    // request cost a 32K prompt 0.72 s of time to first token), else per request as before
    check(pin_residual_at_start(6 * GB, 125 * GB, 40 * GB, false), "3090 desktop, 125 GB RAM, 40 GB available: 6 GB pinned at start");
    check(!pin_residual_at_start(6 * GB, 109 * GB, 18 * GB, false), "pod, 18 GB available: pinned per request");
    check(!pin_residual_at_start(6 * GB, 48 * GB, 30 * GB, false), "48 GB machine: never more than a tenth of RAM pinned for it");
    check(!pin_residual_at_start(0, 125 * GB, 40 * GB, false), "single-chunk contexts need no buffer");
    // GB10 (unified memory), Flash-Next 131K context: 4.3 GB beside 60 GB available is pinned at start (11-18: pinned in the
    // first 128K request it took 2.2 s of TTFT); with less than 20 GB left after it, per request
    check(pin_residual_at_start(size_t(4.3 * GB), 121 * GB, 60 * GB, true), "GB10, 60 GB available: the 131K buffer pinned at start");
    check(!pin_residual_at_start(6 * GB, 121 * GB, 24 * GB, true), "GB10, 24 GB available: 18 GB would be left, per request");
    // the prompt cache without --prompt-cache (owner 11-18: on where memory fits, --exact on too)
    check(prompt_cache_by_fit(false, false, true, 1, size_t(60 * GB), size_t(4.5 * GB)), "GB10, one slot, 60 GB available: on");
    check(prompt_cache_by_fit(false, false, true, 8, size_t(60 * GB), size_t(4.5 * GB)), "GB10, 8 slots x 4.5 GB beside 60 GB: on");
    check(!prompt_cache_by_fit(false, false, true, 1, size_t(24 * GB), size_t(4.5 * GB)), "GB10, 24 GB available: under 20 GB would be left, off");
    check(!prompt_cache_by_fit(false, false, true, 16, size_t(60 * GB), size_t(4.5 * GB)), "GB10, 16 slots x 4.5 GB: does not fit, off");
    check(prompt_cache_by_fit(false, false, false, 1, 0, size_t(4.5 * GB)), "discrete card: on (the kept sequence is the slot's own K/V)");
    check(prompt_cache_by_fit(true, true, true, 16, 0, size_t(9 * GB)) && !prompt_cache_by_fit(true, false, false, 1, 0, 0),
          "--prompt-cache on / off given: it wins");
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
        // the rule: --exact off skips at rank 8, --exact on never
        check(tail_skip_rank(true) == 0 && tail_skip_rank(false) == kTailSkipRank && kTailSkipRank == 8, "--exact off: R = 8; --exact on: off");
        check(tail_skip_targets(routed, topk, tail_skip_rank(true), resident) == std::vector<int>(routed.size(), -1), "--exact on: nothing skipped");
        // a skipped assignment never becomes a CPU pool column, even when its target (the token's first choice) is missed; a
        // second column of the same (token, expert) is wasted CPU work and can pass the pool job's 64 columns
        check(cpu_column(false, true) && !cpu_column(true, true) && !cpu_column(true, false) && !cpu_column(false, false),
              "skipped: no CPU column; kept and missed: a CPU column");
        // with a host tier only what the drive would serve is skipped: 3 sits in the tier (a copy), 4 is on the drive only
        auto tiered = [&](int e) { return tail_skip_keeps(resident(e), e == 3); };
        check(tail_skip_targets(routed, topk, 3, tiered) == std::vector<int>{-1, -1, -1, 1, -1, -1, -1, 5},
              "host tier: expert 3 (in the tier) kept, 4 (drive only) skipped");
        check(tail_skip_keeps(true, false) && tail_skip_keeps(false, true) && !tail_skip_keeps(false, false), "kept when on the card or in the tier");
        // the host tier's elastic chunks: a full 128K prompt's buffers + 2 GB of reserve; a long prompt keeps its buffers +
        // the full reserve, decode and one-chunk prompts the decode reserve
        const uint64_t G = uint64_t(1) << 30, rows = 8192, kv = 24576;
        check(prompt_host_bytes(131072, rows, kv, 32768) == 131072 * rows + 98304 * kv && prompt_host_bytes(32768, rows, kv, 32768) == 32768 * rows &&
              prompt_host_bytes(131072, rows, kv, 0) == 131072 * rows, "prompt host bytes: rows + the K/V tail past 32K");
        check(host_tier_chunks(G * 3) == 10 && host_tier_chunks(0) == 4, "elastic chunks: buffers + 2 GB in 512 MB chunks");
        check(warm_host_tier(24569, 24576) && !warm_host_tier(8441, 24576) && warm_host_tier(22119, 24576) && !warm_host_tier(22118, 24576),
              "the tier is filled at start only when it holds 90 % of the experts (64 GB yes, 31 GB no)");
        check(host_tier_keep(true, G) == G + kHostTierReserve && host_tier_keep(false, G) == kHostTierDecodeReserve,
              "a long prompt keeps its buffers + the reserve; decode the decode reserve");
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
        check(cpu_miss_threads(-1, false, 32, false, true) == 15 && cpu_miss_threads(-1, true, 32, false, true) == 0 &&
              cpu_miss_threads(14, true, 32, false, true) == 14, "pool threads: env wins, --exact off min(15, cores / 2 - 1), --exact on none");
        if (kCpuMissPoolBuilt) check(cpu_pool_reads_packed(-1, false, 32), "--exact off without TQFN_CPU_MISS: the pool reads the pinned packed experts");
        check(!cpu_pool_reads_packed(-1, true, 32) && !cpu_pool_reads_packed(0, false, 32), "no pool, no packed view");
        check(cpu_miss_threads(-1, false, 4, false, true) == 1, "small CPU: at least one thread");
        // G2 (11-13): a small card with --np > 1 runs no pool; one request keeps it; TQFN_CPU_MISS still wins
        check(cpu_miss_threads(-1, false, 32, true, true) == 0 && cpu_miss_threads(-1, false, 32, false, true) == 15 &&
              cpu_miss_threads(6, false, 32, true, true) == 6, "small card, several requests: no CPU expert pool unless asked");
        if (kCpuMissPoolBuilt) check(!cpu_pool_reads_packed(-1, false, 32, true) && cpu_pool_reads_packed(-1, false, 32, false), "the packed view follows the pool");
        // experts read from the GGUF on demand (a host tier): no pool unless asked (its jobs would read the GGUF mapping from
        // the drive), and the draft plan then keeps the adaptive block
        check(cpu_miss_threads(-1, false, 32, false, true, true) == 0 && cpu_miss_threads(6, false, 32, false, true, true) == 6,
              "host tier: no CPU expert pool unless TQFN_CPU_MISS asks");
        check(!cpu_pool_reads_packed(-1, false, 32, false, true), "host tier: no packed view");
        // one sequence on a 24 GB-class card: no pool unless TQFN_CPU_MISS asks (.77: no gain; .188: the fault's trigger);
        // several sequences there and small cards keep their rules
        check(cpu_miss_threads(-1, false, 32, false, true, false, true) == 0 && cpu_miss_threads(6, false, 32, false, true, false, true) == 6 &&
              cpu_miss_threads(-1, false, 32, false, true, false, false) == 15, "big card, one sequence: no pool unless asked; several: the pool");
        check(sequence_drafts(3, false, -1, 32, false, false, false).adaptive, "big card, one sequence, --exact off: the adaptive block (no pool)");
        check(sequence_drafts(3, false, -1, 32, false, false, true).adaptive && sequence_drafts(3, false, -1, 32, false, false, true).drafts == 5,
              "host tier, --exact off: adaptive block, up to 5 drafts");
        // 11-18: the pool's kernels are x86 AVX2; on arm64 (GB10) --exact off alone must not start it (every request failed)
#if defined(__x86_64__)
        check(cpu_miss_threads(-1, false, 20) == 9, "x86: --exact off starts the pool");
#else
        check(cpu_miss_threads(-1, false, 20) == 0 && !cpu_pool_reads_packed(-1, false, 20), "arm64: --exact off runs no CPU pool (AVX2 kernels only)");
#endif
        check(cpu_miss_threads(4, false, 20, false, false) == 4, "TQFN_CPU_MISS set: asked for even without the kernels (the pool then refuses)");
    }
    {   // --np limit: 32 only when the manifest pins every expert resident (slots=512), else 16
        check(np_max(false, 512, 512) == 32, "slots=512 (every expert resident): up to --np 32");
        check(np_max(true, 512, 512) == 16 && np_max(false, 511, 512) == 16 && np_max(false, 150, 512) == 16, "slots=auto or fewer than 512: up to --np 16");
        for (int np = 17; np <= 32; ++np) check(np <= np_max(false, 512, 512) && np > np_max(true, 512, 512), "np 17-32: resident serves, not resident refuses");
        check(np_max(false, 256, 256) == 32 && np_max(false, 384, 512) == 16, "the model's expert count decides, not 512");
    }
    {   // layer graphs: up to 1024 / topk rows with every expert resident (device maps), 8 otherwise; more rows run eager
        check(graph_rows_max(10) == 102 && graph_rows_max(8) == 128 && graph_rows_max(0) == 1024, "device maps: 1024 / topk rows");
        check(kBatchRowsMax * 10 <= 1024 && kBatchRowsMax <= graph_rows_max(10), "a batch's rows fit the device maps (top-10)");
        for (int n = 33; n <= 64; ++n) check(n <= layer_graph_rows(true, 10), "resident, top-10: 33-64 rows run as graphs");
        check(65 > layer_graph_rows(true, 10), "resident, top-10: past a batch's 64 rows: eager");
        int most = 0; for (int a = 2; a <= 32; ++a) most = std::max(most, a * (batch_drafts(a, 5, 10, 512, 512) + 1));
        check(most <= kBatchRowsMax && most > 51, "np 2-32 with every expert resident: up to 64 rows a step, all within the graph rule");
        check(layer_graph_rows(false, 10) == 8, "not resident: up to 8 rows graphed, 9 eager");
    }
    {   // device maps and the waves readback (fn_layer_forward): a 512-slot bank of 512 top-10 experts, verify rows 52-64
        for (int n = 1; n <= 64; ++n)
            check(device_maps(n, 10, 512, 512, true) && !reads_routes_for_waves(false, true, n, 10, 512, true),
                  "every expert resident: " + std::to_string(n) + " rows build their maps on the device, no readback");
        check(!device_maps(2, 10, 150, 512, true) && reads_routes_for_waves(false, false, 16, 10, 150, true), "a partial bank keeps the host maps");
        check(!device_maps(4, 10, 512, 512, false) && device_maps(1, 10, 512, 512, false), "prompt chunks keep the host maps");
        check(!device_maps(graph_rows_max(10) + 1, 10, 512, 512, true), "past 1024 assignments: host maps");
    }
    {   // batched attention workspace (Flash-Next: 24 heads x 256): within 64 MB, else the per-sequence path (blocks of 64 MB)
        check(attention_multi_fits(64, 24, 256, 2048), "64 rows below the indexer budget (2,048): batched");
        check(!attention_multi_fits(64, 24, 256, 131072) && !attention_multi_fits(4, 24, 256, 131072) && attention_multi_fits(1, 24, 256, 131072), "128K without an indexer: 4+ rows take the per-sequence path (one row: 25 MB)");
        check(attention_multi_fits(8, 24, 256, 8192) && !attention_multi_fits(8, 24, 256, 65536), "8 rows: batched to ~16K, not at 64K");
    }
    {   // StepUndo: a captured step that failed after swapping the GDN buffers, advancing len and making a first-use buffer
        struct Rec { std::unique_ptr<int> history, history2, state, state2; int last_n = 3, channels = 8; } r;
        r.history = std::make_unique<int>(1); r.history2 = std::make_unique<int>(2); r.state = std::make_unique<int>(3);
        int len = 40; std::unique_ptr<int> ple;
        const int * h0 = r.history.get(), * h1 = r.history2.get(), * s0 = r.state.get();
        StepUndo<std::unique_ptr<int>> undo;
        undo.swapped(r.history, r.history2); undo.swapped(r.state, r.state2); undo.value(r.last_n); undo.value(len); undo.first_use(ple);
        std::swap(r.history, r.history2);                                        // the conv history swap
        r.state2 = std::make_unique<int>(4); std::swap(r.state, r.state2);       // state2 made on first use, then the swap
        r.last_n = 7; len += 7; ple = std::make_unique<int>(5);
        undo.restore();
        check(r.history.get() == h0 && r.history2.get() == h1, "rerun: history buffers back in place");
        check(r.state.get() == s0 && !r.state2, "rerun: state back, the first-use second buffer dropped");
        check(r.last_n == 3 && len == 40 && !ple, "rerun: lengths back, the first-use PLE buffer dropped (made again, zeroed)");
        StepUndo<std::unique_ptr<int>> none; none.swapped(r.history, r.history2); none.value(len); none.restore();
        check(r.history.get() == h0 && len == 40, "a step that changed nothing: restore keeps it");
    }
    {   // TQFN_CPU_MISS: its value when set, -1 when unset
        unsetenv("TQFN_CPU_MISS"); check(cpu_miss_env() == -1, "TQFN_CPU_MISS unset: -1");
        setenv("TQFN_CPU_MISS", "6", 1); check(cpu_miss_env() == 6, "TQFN_CPU_MISS=6: 6");
        setenv("TQFN_CPU_MISS", "0", 1); check(cpu_miss_env() == 0, "TQFN_CPU_MISS=0: 0 (pool off)");
        unsetenv("TQFN_CPU_MISS");
    }
    {   // the built-in Flash-Next routing profile (manifests without layer_slots, e.g. Unsloth files)
        const std::string prof = flashnext_layer_profile(48, 512);
        int n = prof.empty() ? 0 : 1; for (char ch : prof) n += ch == ',';
        check(n == 48, "Flash-Next profile: one entry per layer");
        check(std::string(flashnext_layer_profile(40, 512)).empty() && std::string(flashnext_layer_profile(48, 256)).empty(), "only for 48 layers x 512 experts");
    }
    {   // drafter pick: DFlash2 warm-up, MTP warm-up, then the faster one, with a probe of the other every kPickEvery steps
        DrafterPick p; std::vector<int> used;
        for (int s = 0; s < 64; ++s) { const int w = pick_drafter(p); used.push_back(w); drafter_done(p, w, w ? 3 : 6, 0.025); }
        check(used[0] == 0 && used[kPickWarm] == 1, "warm-up: DFlash2 first, then MTP");
        int mtp = 0; for (int s = 2 * kPickWarm; s < 64; ++s) mtp += used[size_t(s)];
        check(mtp == 2 * kPickProbe, "code-like rates: DFlash2 drafts, MTP only on probes");
        DrafterPick q; int d = 0;
        for (int s = 0; s < 64; ++s) { const int w = pick_drafter(q); if (s >= 2 * kPickWarm && !w) ++d; drafter_done(q, w, w ? 3 : 2, 0.025); }
        check(d == 2 * kPickProbe, "prose-like rates: MTP drafts, DFlash2 only on probes");
        DrafterPick r; for (int s = 0; s < 8; ++s) drafter_done(r, pick_drafter(r), 5, 0.025);
        for (int s = 0; s < 40; ++s) { const int w = pick_drafter(r); drafter_done(r, w, w ? 5 : 1, 0.025); }
        check(pick_drafter(r) == 1 || r.step % kPickEvery >= kPickEvery - kPickProbe, "DFlash2 falling behind mid-request: MTP takes over");
    }
    {   // side branches: one sequence on device maps only
        check(side_branches(true, false, false, false), "device maps, one sequence: side branches");
        check(!side_branches(false, false, false, false), "host maps (a card that streams experts): none");
        check(!side_branches(true, true, false, false) && !side_branches(true, false, true, false) && !side_branches(true, false, false, true),
              "several sequences, a prefill sink or --split experts: none");
        // the MTP catch-up after a step only when something drafts
        check(!mtp_after_step(0, false), "drafts=0 without DFlash2: no MTP catch-up per step");
        check(mtp_after_step(1, false) && mtp_after_step(3, false) && mtp_after_step(0, true), "drafting servers keep it");
    }
    {   // layer-major prefill without a pack: the bank only while the routed experts fit, else streamed (on-demand GGUF)
        check(!prefill_through_bank(true, false, true, 10, 200), "pack: only resident experts through the bank");
        check(prefill_through_bank(false, false, true, 512, 512) && prefill_through_bank(false, false, false, 512, 512), "no pack, no host blocks (all resident): the bank");
        check(prefill_through_bank(false, true, true, 180, 206), "on demand, routed experts fit the bank: the bank (short prompts unchanged)");
        check(!prefill_through_bank(false, true, true, 512, 206), "on demand, 28K prompt routing to every expert: streamed, not 'active set exceeds slots'");
        check(!prefill_through_bank(false, true, false, 10, 206), "on demand, borrowed layer (no bank): streamed");
    }
    std::printf("decode rules: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

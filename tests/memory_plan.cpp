// Flash-Next start-up memory plan (memory_plan.hpp, no CUDA): pinned to the numbers of real starts.
#include "models/qwen/qwen4exp/memory_plan.hpp"
#include <cstdio>
#include <string>

using namespace tqllm::flashnext;
static int failures = 0;
static void check(bool ok, const std::string & what) { if (!ok) { std::printf("FAIL %s\n", what.c_str()); ++failures; } }
static size_t mb(double x) { return size_t(x * double(kMiB)); }

int main() {
    // Flash-Next: 512 experts, top-10; one slot in every layer of the IQ3_S / tqLLM file is 94.8 MB
    MemShape fn; fn.attention_layers = 12; fn.kv_heads = 2; fn.head_dim = 256; fn.index_dim = 128; fn.experts = 512; fn.topk = 10;
    const size_t slot = mb(94.8);

    // context keep: the reserve (1 GB for one sequence, 256 MB for several) steps to 2 GB past 64K; host K/V lowers the state
    {
        MemRun a; a.context = 65536; MemRun b = a; b.context = 65537; MemRun np2 = a; np2.sequences = 2;
        check(context_keep(fn, a).reserve == 1024 * kMiB && context_keep(fn, b).reserve == 2048 * kMiB,
              "one sequence: reserve 1 GB to 64K (256 MB ran 3090 GSM8K decodes out of memory), 2 GB past it");
        check(context_keep(fn, np2).reserve == 256 * kMiB, "several sequences to 64K: 256 MB (their keep holds a state per extra sequence)");
        MemRun dev; dev.context = 160000; MemRun host = dev; host.kv_host = true;
        check(context_keep(fn, host).state < context_keep(fn, dev).state, "host K/V: old positions leave the device state");
        MemRun shortk = host; shortk.context = 60000;
        check(context_keep(fn, shortk).state == context_keep(fn, MemRun{60000}).state, "host K/V moves nothing up to twice the kept window");
        MemRun small; small.context = 8192;
        check(context_keep(fn, small).state < context_keep(fn, dev).state, "state grows with the context");
    }

    // RTX 3090, 128K server (context 160000), one sequence, 1.4.2 M1 start: 17675 MB free without banks, 2304 MB kept
    // (2 GB reserve + 256 MB) -> 162 slots; the second round measured 235 MB of rebuild extra -> 2539 MB kept, 159 slots
    {
        MemRun r; r.context = 160000; r.kv_host = true;
        const SlotPlan p = auto_slot_plan(fn, r, mb(17675), slot, 0);
        check(p.keep == 2304 * kMiB, "3090 128K keeps 2304 MB (got " + std::to_string(p.keep / kMiB) + ")");
        check(p.slots == 162, "3090 128K first round 162 slots (got " + std::to_string(p.slots) + ")");
        const SlotPlan q = auto_slot_plan(fn, r, mb(17673), slot, 0, 235 * kMiB);
        check(q.keep == 2539 * kMiB && q.slots == 159, "3090 128K second round 2539 MB kept, 159 slots (got " + std::to_string(q.slots) + ")");
        check(q.tiers && q.tier_keep == 256 * kMiB, "one sequence: tiered banks, the refill leaves 256 MB");
        check(q.tier_a < q.slots && q.tier_a > q.verify_min, "tier A below the target and above the verify floor");
        check(q.drafts == 5 && q.adaptive, "3090 one sequence: adaptive drafts up to 5 (not every expert resident)");
    }

    // GB10, one sequence: 106035 MB free, 2304 MB kept -> every expert resident (512); drafts must be 7 as with a pinned
    // slots=512 manifest (1.4.1 drafted 5: 104 vs 122 tok/s, fixed in 1.4.2)
    {
        MemRun r; r.context = 81920;
        const SlotPlan p = auto_slot_plan(fn, r, mb(106035), slot, 0);
        check(p.slots == 512, "GB10: every expert resident");
        check(p.drafts == 7, "GB10 slots=auto: 7 drafts (1.4.1 regression) (got " + std::to_string(p.drafts) + ")");
    }

    // RTX 3090 --np 8, 8192 per slot (64K total), 1.4.1 gate: measured one-sequence state 286 MB -> 7 x 286 + 256 + 256 =
    // 2514 MB kept, 145 slots; several sequences keep the manifest's drafts and no tiers
    {
        MemRun r; r.context = 8192; r.sequences = 8;
        const SlotPlan p = auto_slot_plan(fn, r, mb(16298), slot, 286 * kMiB);
        check(p.keep == 2514 * kMiB, "3090 np 8 keeps 2514 MB, no per-request K/V up front (got " + std::to_string(p.keep / kMiB) + ")");
        check(p.slots == 145, "3090 np 8: 145 slots (keeping every request's 8K K/V cost 19, 11-05)");
        check(!p.tiers && p.drafts == 3, "np 8: no tiers, manifest drafts");
        // the draft cutoff today's batch_drafts applies at 145 slots: 8 sequences x 1 row x top-10 fit, 2 rows do not
        check(batch_drafts(8, p.drafts, fn.topk, 145, fn.experts) == 0, "3090 np 8: no drafts per slot at 145 slots (the np 8 limit)");
        check(batch_drafts(16, p.drafts, fn.topk, p.slots, fn.experts) == -1, "np 16: not even one row per slot fits (serial steps)");
        MemRun cached = r; cached.caching = true;
        // with the prompt cache each slot keeps its full-context estimate even when the one-sequence probe measured less
        check(auto_slot_plan(fn, cached, mb(16298), slot, 1024).keep > auto_slot_plan(fn, r, mb(16298), slot, 1024).keep,
              "prompt cache: every slot keeps its full context, not the short probe");
    }

    // RTX 3090 --np 12 x 8K (11-05): ran out of memory with 84 MB free; a request now makes room at its start
    {
        const size_t kv = sequence_kv_bytes(fn, 3300);
        check(request_shrink_bytes(fn, 3300, kv + 256 * kMiB, 256 * kMiB) == 0, "room for the request's K/V: nothing to give back");
        check(request_shrink_bytes(fn, 3300, 84 * kMiB, 256 * kMiB) == kv + 256 * kMiB - 84 * kMiB, "84 MB free: the banks give back the rest");
        // 16 GB np 4 (11-09): a new sequence's PLE histories (~280 MB) were not counted, the reserve went to them and the
        // running requests' next step failed for memory (HTTP 500)
        const size_t fixed = 284 * kMiB;
        check(request_shrink_bytes(fn, 3300, kv + 256 * kMiB, 256 * kMiB, fixed) == fixed, "a new sequence's fixed state is given back too");
        check(sequence_kv_bytes(fn, 8192) == size_t(fn.attention_layers) * (2 * 8192 * 2 * 256 * 2 + 8192 * 128 * 4 + 2048 * 4), "K/V bytes per sequence");
    }
    // small card (12 GB): the keep is 512 MB; with the CPU pool (--exact off) the drafts stay inside its 4-row block
    {
        MemRun r; r.context = 16384; r.small_card = true; r.pool_rows = kCpuPoolRows;
        const SlotPlan p = auto_slot_plan(fn, r, mb(9000), slot, 0);
        check(p.keep == 512 * kMiB, "small card keeps 512 MB");
        check(p.drafts + 1 <= kCpuPoolRows && !p.adaptive, "small card: fixed drafts within the CPU pool block");
        check(p.verify_min == verify_slot_min(p.drafts, fn.topk), "small card: the verify floor follows the plan's drafts");
        check(p.tier_keep == 512 * kMiB, "small card: the refill keeps 512 MB");
    }
    // the same card running exact (no CPU pool): the plan drafts and floors on the adaptive block the engine runs, not on a
    // 4-row pool block that is not there (11-12: the plan assumed 3 drafts / a 40-slot floor while the engine drafted 5)
    {
        MemRun r; r.context = 16384; r.small_card = true;
        const SlotPlan p = auto_slot_plan(fn, r, mb(9000), slot, 0);
        check(p.drafts == 5 && p.adaptive, "exact small card: the plan's 5-draft adaptive block");
        check(p.verify_min == verify_slot_min(5, fn.topk), "exact small card: the verify floor follows 5 drafts");
    }

    // fit check after the banks: slots=auto borrows, a fixed count that cannot hold the context is refused with a smaller one
    {
        MemRun r; r.context = 160000; r.kv_host = true;
        const ContextKeep k = context_keep(fn, r);
        check(fit_max_slots(k, k.state + k.reserve, false, false, 160, 160 * slot) == -1, "fits: no refusal");
        check(fit_max_slots(k, 1024 * kMiB, true, false, 160, 160 * slot) == -1, "slots=auto: a long prompt borrows banks");
        const int m = fit_max_slots(k, 1024 * kMiB, false, false, 160, 160 * slot);
        check(m >= 0 && m < 160, "fixed slots that cannot hold the context: suggest fewer (got " + std::to_string(m) + ")");
        check(size_t(160 - m) * slot >= k.state + k.reserve - 1024 * kMiB, "the suggestion frees enough for the context");
    }

    // The prefill plan follows the prompt length and the sizes fixed at start, not what the previous request left free.
    // 11-12: at 16 GB the same 25.3K prompt took one chunk of 25600 or two of 21504 depending on the free memory it found
    // (+-1 s of TTFT), and since a chunk boundary picks the top-k of the layer-major prefill, the text followed it.
    {
        // Flash-Next GSQ-RCO IQ3_S as the 16 GB cap leaves it: 5,324,800 B an expert, 64-expert staging slices, a 3800 MB
        // arena, 160 MB dense pool, 2024 MB workspace (--prefill-reserve-mb 1000), a 162 MB MTP bank and a 10496 MB plan
        // room (free + bank bytes at start, pool trimmed, 256 MB steps: the engine's "prefill plan room" line)
        tqllm::PrefillShape s; s.hidden = 2560; s.hc = 4; s.expert_ff = 640; s.topk = 10; s.experts = 512; s.layers = 48;
        s.expert_block_bytes = 5324800; s.wide_bf16 = true;
        const size_t room = mb(10496), mtp = mb(162);
        // the request's K/V state: the engine sizes it for prompt + 192 tokens + 16, in 1 KiB steps (fn_engine_prefill)
        const auto tune_for = [&](size_t prompt) {
            return [&, prompt](int chunk, int block) {
                tqllm::PrefillTuning t; t.chunk = chunk; t.sub = block; t.stage = 64; t.dense_cache_bytes = mb(160);
                t.workspace_bytes = mb(2024); t.mtp_bank_bytes = mtp;
                t.arena_bytes = tqllm::prefill_arena_req(mb(3800), chunk, block, 24);
                const int cap = int(std::min<size_t>(32768, (prompt + 192 + 16 + 1023) / 1024 * 1024));
                t.context_bytes = tqllm::prefill_context_bytes(cap, block, 12, 2, 256, 128);
                return t;
            };
        };
        const auto t25 = tune_for(25318);
        const tqllm::PrefillChoice c = tqllm::prefill_choice(s, t25, 25318, room, mtp, 0, 25600, 2048);
        check(c.chunk == 18432 && c.block == 2048, "25.3K at 16 GB: 2 chunks of 18432, block 2048 (got chunk " +
              std::to_string(c.chunk) + ", block " + std::to_string(c.block) + ")");
        check(!c.park_heads, "25.3K: the heads' 635 MB cannot make it one chunk");
        check(tqllm::prefill_plan(s, t25(25600, 2048), 25318, 0).need > room + mtp, "one chunk of 25600 does not fit");
        check(tqllm::prefill_plan(s, t25(21504, 512), 25318, 0).need > room + mtp, "21504 MB is 45 MB short as well");
        // the leftover state: banks resident or half released, the same room and the same plan
        const tqllm::PlanRoom a = tqllm::plan_room(room, mb(591), mb(10037), mb(200), 0, 0);
        const tqllm::PlanRoom b = tqllm::plan_room(room, mb(1571), mb(9017), mb(200), 0, 0);
        check(a.from_budget && b.from_budget && a.room == room && b.room == room, "both leftover states plan with the fixed room");
        const tqllm::PrefillChoice ca = tqllm::prefill_choice(s, t25, 25318, a.room, mtp, 0, 25600, 2048);
        const tqllm::PrefillChoice cb = tqllm::prefill_choice(s, t25, 25318, b.room, mtp, 0, 25600, 2048);
        check(ca.chunk == cb.chunk && ca.block == cb.block && ca.chunk == c.chunk, "chunk and block do not follow the leftover state");
        // a session that cannot give the room plans with what it has, and is flagged (not silent): 400 + 1000 MB
        const tqllm::PlanRoom thin = tqllm::plan_room(room, mb(400), mb(1000), 0, 0, 0);
        check(!thin.from_budget && thin.room == mb(1400), "a short session plans with what it has (flagged)");
        // a prompt inside the engine chunk gets its own length; more room buys one chunk
        const tqllm::PrefillChoice small = tqllm::prefill_choice(s, tune_for(1217), 1217, room, 0, 0, 2048, 2048);
        check(small.chunk == 2048 && !small.park_heads, "1.2K: its own 2 KiB chunk");
        const tqllm::PrefillChoice big = tqllm::prefill_choice(s, t25, 25318, mb(12416), mtp, 0, 25600, 2048);
        check(big.chunk == 25600, "25.3K with 1.9 GB more room: one chunk of 25600 (got " + std::to_string(big.chunk) + ")");
    }

    std::printf("memory plan: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

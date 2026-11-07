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
        check(sequence_kv_bytes(fn, 8192) == size_t(fn.attention_layers) * (2 * 8192 * 2 * 256 * 2 + 8192 * 128 * 4 + 2048 * 4), "K/V bytes per sequence");
    }
    // small card (12 GB): the keep is 512 MB, drafts limited to the CPU pool's block
    {
        MemRun r; r.context = 16384; r.small_card = true;
        const SlotPlan p = auto_slot_plan(fn, r, mb(9000), slot, 0);
        check(p.keep == 512 * kMiB, "small card keeps 512 MB");
        check(p.drafts + 1 <= kCpuPoolRows && !p.adaptive, "small card: fixed drafts within the CPU pool block");
        check(p.tier_keep == 512 * kMiB, "small card: the refill keeps 512 MB");
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

    std::printf("memory plan: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

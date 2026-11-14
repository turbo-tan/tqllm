#include "fit_context.hpp"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>

int main() {
    const size_t GB = 1000000000ull, MB = size_t(1) << 20;
    int bad = 0;
    auto expect = [&](const char * what, bool ok) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++bad; } };
    // 5060 Ti, TQ3_XS, rc3 f5eaeb1 (2026-10-07): 163840 left 0.38 GB, the halved 81920 left 1.84 GB, margin 768 MiB.
    // Halving alone wasted ~58K tokens of context; the fit must land well above 81920 and below 163840.
    const uint32_t c = tqllm::fit_context(163840, size_t(0.38 * GB), 81920, size_t(1.84 * GB), 768 * MB);
    expect("5060 Ti case above the halved context", c >= 131072);
    expect("5060 Ti case below the short context", c < 163840);
    expect("aligned to 4096", c % 4096 == 0);
    // the predicted context keeps the margin under the linear model
    const double per_tok = (1.84 - 0.38) * GB / (163840 - 81920);
    expect("keeps the margin", 1.84 * GB - (c - 81920) * per_tok >= 768.0 * MB);
    // -fitt 0 must still measure: a spilled WSL load reports ~0 free and has to count as "does not fit"
    expect("fit off: no margin", tqllm::fit_margin(false, -1, false) == 0);
    expect("default margin 768 MiB", tqllm::fit_margin(true, -1, false) == 768 * MB);
    expect("WSL default margin 256 MiB", tqllm::fit_margin(true, -1, true) == 256 * MB);
    expect("-fitt 0 keeps a floor (0 free = spilled)", tqllm::fit_margin(true, 0, true) >= 64 * MB);
    expect("-fitt 0 keeps a floor on Linux too", tqllm::fit_margin(true, 0, false) >= 64 * MB);
    // the graph cache's working set is part of every margin (the fit and GraphLru agree by construction)
    const size_t graphs = tqllm::kGraphCacheFloor * tqllm::kGraphCacheFloorBytes;
    expect("-fitt 300: 300 MiB + the graph working set", tqllm::fit_margin(true, 300, false) == 300 * MB + graphs);
    expect("the default holds the graph working set", tqllm::fit_margin(true, -1, false) >= 64 * MB + graphs);
    // 5060 Ti, TQ_XS 200K (2026-10-09): host tails of 15360 then 21504 positions left 0.05 then 0.19 GB (margin 0.27); the
    // fit halved the context to 102400 instead of moving ~4K more positions to host
    expect("converging host refinement continues", tqllm::fit_host_again(2, size_t(0.19 * GB), size_t(0.05 * GB), 21504, 204800));
    expect("no progress: stop refining", !tqllm::fit_host_again(2, size_t(0.05 * GB), size_t(0.05 * GB), 21504, 204800));
    expect("all KV already on host: stop", !tqllm::fit_host_again(1, size_t(0.10 * GB), 0, 204800, 204800));
    expect("bounded number of tries", !tqllm::fit_host_again(8, size_t(0.20 * GB), size_t(0.19 * GB), 30720, 204800));
    // WSL: the driver reads ~0 near the budget, spilled or not; the fit must judge by its own accounting
    {
        const size_t total = size_t(16311) * MB, before = size_t(900) * MB;
        expect("Linux: driver figure", tqllm::fit_free(false, 300 * MB, total, before, 14 * GB) == 300 * MB);
        // GB10 (integrated): 1.62 GB driver free + 40 GB of page cache the OS would drop -> about 41.6 GB usable
        expect("unified: reclaimable page cache counts as free",
               tqllm::unified_free(size_t(1.62 * GB), 121 * GB, 50 * GB, 10 * GB) == size_t(1.62 * GB) + 40 * GB);
        expect("unified: never above the total", tqllm::unified_free(100 * GB, 121 * GB, 60 * GB, 10 * GB) == 121 * GB);
        expect("unified: no reclaimable memory, the driver figure", tqllm::unified_free(3 * GB, 121 * GB, 5 * GB, 5 * GB) == 3 * GB);
        expect("WSL, fits (driver says 0): accounted room left",
               tqllm::fit_free(true, 0, total, before, size_t(14.6 * GB)) > 400 * MB);
        expect("WSL, spilled (driver says 0.29 GB): accounted 0",
               tqllm::fit_free(true, 290 * MB, total, before, size_t(16.2 * GB)) == 0);
    }
    // no room: stay at the known-good context
    expect("no room above ctx_ok", tqllm::fit_context(65536, size_t(0.10 * GB), 32768, size_t(0.81 * GB), 768 * MB) == 32768);
    // degenerate measurements fall back to ctx_ok
    expect("free_ok below margin", tqllm::fit_context(65536, 100 * MB, 32768, 500 * MB, 768 * MB) == 32768);
    expect("non-increasing free", tqllm::fit_context(65536, 900 * MB, 32768, 900 * MB, 768 * MB) == 32768);
    expect("ctx order", tqllm::fit_context(32768, 100 * MB, 65536, 2000 * MB, 768 * MB) == 65536);
    // the loader's refinement loop against a linear memory model: an attempt that falls short becomes the new upper point.
    // 5060 Ti, -fitt 256 (2026-10-07): 204800 left 0.05 GB, 102400 left 1.57 GB, 188416 left 0.26 GB (10 MB short) - the
    // first fix fell back to 102400; refining must land within a few thousand tokens of the true limit.
    {
        const double free0 = 1.57 * GB, per = (1.57 - 0.05) * GB / (204800 - 102400) * 1.035;  // slightly steeper than measured
        auto free_at = [&](uint32_t c) { return free0 - (double(c) - 102400) * per; };
        const size_t mg = 256 * MB;
        uint32_t cs = 204800, ok = 102400;
        size_t fs = size_t(std::max(0.0, free_at(cs))), fo = size_t(free_at(ok));
        uint32_t got = ok;
        for (int k = 0; k < 3; ++k) {
            const uint32_t b = tqllm::fit_context(cs, fs, ok, fo, mg);
            if (b <= ok) break;
            if (free_at(b) >= double(mg)) { got = b; break; }
            cs = b, fs = size_t(std::max(0.0, free_at(b)));
        }
        const double limit = 102400 + (free0 - mg) / per;
        expect("refinement lands near the true limit", got > 102400 && got <= limit && limit - got < 12288);
        std::printf("refined: %u (true limit %.0f)\n", got, limit);
    }
    expect("zero alignment", tqllm::fit_context(65536, 100 * MB, 32768, 2000 * MB, 768 * MB, 0) == 32768);
    // the loader's refinement loop against a linear memory model: an attempt that falls short becomes the new upper point.
    // 5060 Ti, -fitt 256 (2026-10-07): 204800 left 0.05 GB, 102400 left 1.57 GB, 188416 left 0.26 GB (10 MB short) - the
    // first fix fell back to 102400; refining must land within a few thousand tokens of the true limit.
    {
        const double free0 = 1.57 * GB, per = (1.57 - 0.05) * GB / (204800 - 102400) * 1.035;  // slightly steeper than measured
        auto free_at = [&](uint32_t c) { return free0 - (double(c) - 102400) * per; };
        const size_t mg = 256 * MB;
        uint32_t cs = 204800, ok = 102400;
        size_t fs = size_t(std::max(0.0, free_at(cs))), fo = size_t(free_at(ok));
        uint32_t got = ok;
        for (int k = 0; k < 3; ++k) {
            const uint32_t b = tqllm::fit_context(cs, fs, ok, fo, mg);
            if (b <= ok) break;
            if (free_at(b) >= double(mg)) { got = b; break; }
            cs = b, fs = size_t(std::max(0.0, free_at(b)));
        }
        const double limit = 102400 + (free0 - mg) / per;
        expect("refinement lands near the true limit", got > 102400 && got <= limit && limit - got < 12288);
        std::printf("refined: %u (true limit %.0f)\n", got, limit);
    }
    // host tail: TQ3_XS tq3 KV is 16 layers x 4 heads x 112 B x (K + V) = 14336 B per position
    const size_t tok = 14336;
    expect("enough free: nothing moves", tqllm::fit_host_tokens(262144, 0, 800 * MB, 768 * MB, tok) == 0);
    // WDDM: 0.00 GB free means over budget by an unknown amount (5060 Ti 2026-10-07: moving 10240 positions still read 0.00,
    // the fit halved the context) -> measure with all KV on host instead of stepping from a saturated reading
    expect("saturated reading: measure with all KV on host", tqllm::fit_host_tokens(262144, 0, 0, 64 * MB, tok) == 262144);
    expect("saturated on refine too", tqllm::fit_host_tokens(262144, 10240, 1 * MB, 768 * MB, tok) == 262144);
    // after a measured split, a saturated reading is a small miss (5060 Ti -fitt 64: 0.03 GB after the split, the fit then
    // re-measured and halved): add the margin's worth of positions instead of measuring again
    const uint32_t r = tqllm::fit_host_tokens(262144, 59392, 30 * MB, 64 * MB, tok, 1024, true);
    expect("measured split: refine by the margin", r > 59392 && r < 59392 + 8192);
    const uint32_t h = tqllm::fit_host_tokens(262144, 0, 50 * MB, 768 * MB, tok);
    expect("moves enough positions to free the shortfall", size_t(h) * tok >= 718 * MB);
    expect("moves at most one alignment step more", size_t(h - 1024) * tok < 718 * MB);
    expect("aligned to 1024", h % 1024 == 0);
    expect("refine adds to what is already on host", tqllm::fit_host_tokens(262144, 32768, 700 * MB, 768 * MB, tok) > 32768);
    expect("capped at the context", tqllm::fit_host_tokens(65536, 0, 0, 4096 * MB, tok) == 65536);
    expect("unknown token size: nothing moves", tqllm::fit_host_tokens(65536, 0, 0, 768 * MB, 0) == 0);
    // KV host tail left room on the GPU (5060 Ti, TQ_XS 262144, 2026-10-09: 105472 positions on host, 0.66 GB free vs a
    // 0.27 GB margin): the spare above the margin must bring whole aligned blocks back to the device
    {
        const size_t tok16 = 16 * 1024;  // device bytes per position incl. granule rounding
        const uint32_t back = tqllm::fit_host_return(105472, size_t(0.66 * GB), size_t(0.27 * GB), tok16);
        expect("spare above the margin returns positions to the GPU", back < 105472 && back % 1024 == 0);
        expect("returned positions fit in the spare", size_t(105472 - back) * tok16 <= size_t(0.39 * GB));
        expect("no spare: nothing returns", tqllm::fit_host_return(105472, size_t(0.28 * GB), size_t(0.27 * GB), tok16) == 105472);
        expect("never below zero", tqllm::fit_host_return(2048, size_t(4.0 * GB), size_t(0.27 * GB), tok16) == 0);
        expect("unknown token size: unchanged", tqllm::fit_host_return(4096, size_t(1.0 * GB), size_t(0.27 * GB), 0) == 4096);
    }
    // F1 give-back: tester 3090 256K (10-13) ended 4.05 GB free against a 0.77 GB margin with the LM head in host memory
    expect("give back with 3.3 GB spare", tqllm::fit_try_return(size_t(4.05 * GB), size_t(0.77 * GB)));
    expect("no give-back with 0.18 GB spare", !tqllm::fit_try_return(size_t(0.95 * GB), size_t(0.77 * GB)));
    expect("no give-back with the fit off", !tqllm::fit_try_return(size_t(4.0 * GB), 0));
    // --moe-host-mb / --moe-cache-mb auto: 3090 held at a 16 GB card (15.22 GiB free), MiniCPM-V-4.7 Q5_K_M (24728583072 B):
    // by hand 10000 MiB host was the smallest that loaded; the gated cache arm was 11200 host + 1200 cache
    {
        const uint64_t q5 = 24728583072ull;
        const double GiB = 1073741824.0;
        const auto c16 = tqllm::moe_auto_place(q5, size_t(15.22 * GiB), -1, -1);
        expect("16 GB: auto host near the hand-picked 10000 + cache", c16.host_mb > 10800 && c16.host_mb < 11600);
        expect("16 GB: auto cache ~12 % of the host share", c16.cache_mb > 1100 && c16.cache_mb < 1300);
        const auto f24 = tqllm::moe_auto_place(q5, size_t(23.3 * GiB), -1, -1);
        expect("24 GB: small host share, floor cache", f24.host_mb > 1500 && f24.host_mb < 2500 && f24.cache_mb == 256);
        expect("fits on the device: nothing on host, no cache", tqllm::moe_auto_place(15ull << 30, size_t(23.3 * GiB), -1, -1).host_mb == 0 &&
               tqllm::moe_auto_place(15ull << 30, size_t(23.3 * GiB), -1, -1).cache_mb == 0);
        const auto given = tqllm::moe_auto_place(q5, 0, 10000, -1);
        expect("explicit host + auto cache: host raised by the cache", given.cache_mb == 1200 && given.host_mb == 11200);
        const auto off = tqllm::moe_auto_place(q5, 0, 10000, 0);
        expect("--moe-cache-mb 0: no cache, host as given", off.cache_mb == 0 && off.host_mb == 10000);
        const auto cap = tqllm::moe_auto_place(q5, 0, 40000, -1);
        expect("cache capped at 1536 MiB", cap.cache_mb == 1536 && cap.host_mb == 41536);
        expect("no host experts: an explicit cache is dropped", tqllm::moe_auto_place(q5, 0, 0, 800).cache_mb == 0);
    }
    {   // Qwopus Q4_K_M, 3090, 131072, no drafter (11-02): the load with all KV on host left 5.60 GB, the estimate from it put
        // every position back on the GPU, that load ran out of memory and the fit halved the context twice (32768).
        // Device model: free = 5.60 GB - (ctx - host) x the real per-position cost, which the 0.75 GB / 131072 estimate undershoots.
        const uint32_t ctx = 131072;
        const size_t margin = 810 * MB, real_tb = 46000;  // 6.0 GB for all 131072 positions on the device: out of memory
        auto free_at = [&](uint32_t host) -> long long { return (long long)(5.60 * GB) - (long long)(ctx - host) * (long long)real_tb; };
        expect("unmeasured out of memory: measure with all KV on host",
               tqllm::fit_host_next(ctx, 0, false, 0, margin, 0, false, 0, 0, 0) == ctx);
        expect("measured, then out of memory: try a tail, do not give up context",
               tqllm::fit_host_next(ctx, 0, false, 0, margin, 0, true, ctx, 0, 0) > 0);
        // the step before 11-02: once measured, an out-of-memory load had no host step left, so the context was halved
        auto old_next = [&](uint32_t c, uint32_t h, bool loaded, size_t fr, size_t tb, bool measured, int t, size_t pf) {
            if (t > 0 && !tqllm::fit_host_again(t, fr, pf, h, c)) return h;
            if (!loaded || !tb) return measured ? h : c;
            return tqllm::fit_host_tokens(c, h, fr, margin, tb, 1024, measured);
        };
        // replays the fit's host step from the measuring load's all-on-device estimate; true when a tail meets the margin
        auto replay = [&](auto && next, auto && free_fn, uint32_t & host, int & loads) {
            host = 0, loads = 0;
            int tries = 0;
            size_t prev = 0;
            while (loads < 12) {
                ++loads;
                const long long fr = free_fn(host);
                if (fr >= (long long)margin) return true;
                const bool oom = fr < 0;
                const uint32_t h = next(host, !oom, oom ? 0 : size_t(fr), oom ? 0 : size_t(32768), tries, prev);
                if (h <= host) return false;  // the fit halves the context here
                prev = oom ? 0 : size_t(fr), host = h, ++tries;
            }
            return false;
        };
        auto new_step = [&](uint32_t h, bool l, size_t fr, size_t tb, int t, size_t pf) {
            return tqllm::fit_host_next(ctx, h, l, fr, margin, tb, true, ctx, t, pf);
        };
        auto old_step = [&](uint32_t h, bool l, size_t fr, size_t tb, int t, size_t pf) { return old_next(ctx, h, l, fr, tb, true, t, pf); };
        uint32_t host = 0;
        int loads = 0;
        expect("Qwopus case, old step: gives up the context (the bug)", !replay(old_step, free_at, host, loads));
        const bool fit = replay(new_step, free_at, host, loads);
        expect("Qwopus case: a host tail meets the margin at the full context", fit && host < ctx && free_at(host) >= (long long)margin);
        expect("Qwopus case: bounded reloads", loads <= 8);
        // heavier per-position cost: the first bisected tail still loads short, so the estimate path (capped at the
        // known-good tail) has to finish the fit
        auto free_heavy = [&](uint32_t h) -> long long { return (long long)(5.60 * GB) - (long long)(ctx - h) * 80000; };
        expect("heavy case, old step: gives up the context", !replay(old_step, free_heavy, host, loads));
        const bool fit_heavy = replay(new_step, free_heavy, host, loads);
        expect("heavy case: refined past the first bisection to a fitting tail",
               fit_heavy && host > 65536 && host < ctx && free_heavy(host) >= (long long)margin);
        expect("heavy case: bounded reloads", loads <= 8);
        // bisection: halves toward the known-good tail, aligned, ends on it
        expect("bisect halves the gap", tqllm::fit_host_bisect(0, 131072) == 65536);
        expect("bisect is aligned", tqllm::fit_host_bisect(1000, 131072) % 1024 == 0);
        expect("bisect near the known-good tail takes it", tqllm::fit_host_bisect(128000, 131072) == 131072);
        expect("bisect without a known-good tail: nothing", tqllm::fit_host_bisect(4096, 0) == 4096);
        expect("short estimate is capped at the known-good tail",
               tqllm::fit_host_next(ctx, 0, true, 100 * MB, margin, 1000, true, 65536, 0, 0) == 65536);
        expect("no known-good tail, refinement stalled: give up the step",
               tqllm::fit_host_next(ctx, 21504, true, 50 * MB, margin, 32768, true, 0, 3, 50 * MB) == 21504);
        expect("all KV on host already: nothing left", tqllm::fit_host_next(ctx, ctx, false, 0, margin, 0, true, ctx, 0, 0) == ctx);
    }
    {   // one measuring load decides the placement (Qwopus Q4_K_M 131072 + DFlash2, 3090, 11-02: all KV / head / embeddings in
        // host memory left 4.43 GB; the step fit then took 10 loads, 10.5 min, and forced -ub 512 + 2048-row weight blocks)
        const uint32_t ctx = 131072;
        const size_t margin = 810 * MB, tb = 33280, round_b = 64 * MB, head_b = size_t(1.43 * GB), embed_b = size_t(0.72 * GB);
        const auto p = tqllm::fit_place(ctx, size_t(4.43 * GB), margin, round_b, tb, head_b, embed_b, false, false);
        expect("plan: full context, a KV tail, the margin kept",
               p.kv_host > 0 && p.kv_host < ctx && size_t(ctx - p.kv_host) * tb + round_b + margin <= size_t(4.43 * GB));
        expect("plan: the tail is the smallest that fits (within one block)",
               size_t(ctx - p.kv_host + 1024) * tb + round_b + margin > size_t(4.43 * GB));
        expect("plan: tail aligned", p.kv_host % 1024 == 0);
        expect("plan: head and embeddings stay in host memory while KV is", p.head_host && p.embed_host);
        const auto all = tqllm::fit_place(ctx, size_t(8.0 * GB), margin, round_b, tb, head_b, embed_b, false, false);
        expect("plenty of room: everything on the device", all.kv_host == 0 && !all.head_host && !all.embed_host);
        const auto mid = tqllm::fit_place(ctx, margin + round_b + ctx * tb + head_b + MB, margin, round_b, tb, head_b,
                                          embed_b, false, false);
        expect("room for KV + head only: embeddings stay in host memory", mid.kv_host == 0 && !mid.head_host && mid.embed_host);
        const auto user = tqllm::fit_place(ctx, size_t(8.0 * GB), margin, round_b, tb, head_b, embed_b, true, false);
        expect("asked head in host memory: stays there", user.head_host && !user.embed_host);
        const auto none = tqllm::fit_place(ctx, 500 * MB, margin, round_b, tb, head_b, embed_b, false, false);
        expect("below the margin: nothing comes back", none.kv_host == ctx && none.head_host && none.embed_host);
    }
    {   // 3090 Qwopus Q4_K_M 131072 + DFlash2 (11-02): a failed give-back left the CUDA last error set and the reload without it
        // died on its first stream check ("CUDA kernel launch failed: out of memory"). The fit's loads clear it.
        bool last_error = false;
        int made = 0;
        auto make = [&](bool fail) {
            return [&, fail]() -> std::unique_ptr<int> {
                if (last_error) throw std::runtime_error("CUDA kernel launch failed: out of memory");
                if (fail) { last_error = true; throw std::runtime_error("CUDA error: out of memory"); }
                return std::make_unique<int>(++made);
            };
        };
        auto clear = [&] { last_error = false; };
        expect("out of memory: no engine", !tqllm::fit_try_load(make(true), clear));
        const auto again = tqllm::fit_try_load(make(false), clear);
        expect("reload after an out-of-memory load succeeds", again && *again == 1);
        bool other = false;
        try { tqllm::fit_try_load([]() -> std::unique_ptr<int> { throw std::runtime_error("bad GGUF"); }, clear); }
        catch (const std::exception &) { other = true; }
        expect("other errors propagate", other);
    }
    if (bad) return 1;
    std::printf("PASS fit_context (5060 Ti case -> %u)\n", c);
    return 0;
}

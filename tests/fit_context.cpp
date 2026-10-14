#include "../src/fit_context.hpp"
#include <algorithm>
#include <cstdio>

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
    if (bad) return 1;
    std::printf("PASS fit_context (5060 Ti case -> %u)\n", c);
    return 0;
}

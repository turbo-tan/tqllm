// --split layer memory fit (src/engine/split_fit.hpp): two ranks with a simple memory model (weights + KV per position,
// out of memory past the card) run the fit's loop as the engine does: the context it keeps is the largest that leaves the
// margin free on both ranks (within one 1024 block of it, or the line estimate), it never keeps a context that does not
// fit, it gives up only when the smallest context (-fitc) does not fit, with a per-rank message, and it stays bounded.
#include "split_fit.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string & what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; } }

struct Card { double total, weights, per_tok; };  // bytes
// noise: added to the free reading of load number i (page cache on unified memory moving between loads)
std::vector<double> noise;
int n_load = 0;
tqllm::SplitLoad load(const Card & c, uint32_t ctx) {
    const double need = c.weights + double(ctx) * c.per_tok;
    if (need > c.total) return {false, 0, uint64_t(c.total)};
    const double nz = size_t(n_load) < noise.size() ? noise[n_load] : 0;
    return {true, uint64_t(std::min(c.total, c.total - need + nz)), uint64_t(c.total)};
}
struct Result { tqllm::SplitFitStep last; int loads = 0; std::string error; };
Result run(const std::vector<Card> & cards, uint32_t asked, uint32_t min_ctx, uint64_t margin) {
    tqllm::SplitFit fit(asked, min_ctx, margin);
    Result r;
    n_load = 0;
    uint32_t ctx = asked;
    for (;;) {
        std::vector<tqllm::SplitLoad> seen;
        for (const auto & c : cards) seen.push_back(load(c, ctx));
        ++r.loads;
        ++n_load;
        r.last = fit.next(ctx, seen);
        if (r.last.kind == tqllm::SplitFitStep::fail) { r.error = fit.failure(ctx); return r; }
        if (r.last.kind == tqllm::SplitFitStep::keep) return r;
        ctx = r.last.ctx;
        if (r.loads > 20) return r;
    }
}
// the largest context (1024 multiple) that leaves margin free on every card
uint32_t best(const std::vector<Card> & cards, uint64_t margin) {
    uint32_t b = ~0u;
    for (const auto & c : cards) b = std::min<uint32_t>(b, uint32_t((c.total - c.weights - double(margin)) / c.per_tok) / 1024 * 1024);
    return b;
}
}  // namespace

int main() {
    const uint64_t margin = 768ull << 20;
    // unsloth Q8_0 27B on 2 x 24 GB, one slot: loads at 131,072, out of memory at 196,608 (pod, 11-14)
    const std::vector<Card> q8 = {{24e9, 14.6e9, 60000}, {24e9, 14.4e9, 60000}};
    {
        const auto r = run(q8, 262144, 4096, margin);
        const uint32_t b = best(q8, margin);
        std::printf("Q8_0 2 x 24 GB, asked 262144: keeps %u after %d loads (best %u)\n", r.last.ctx, r.loads, b);
        check(r.last.kind == tqllm::SplitFitStep::keep, "a context fits: keep it");
        check(r.last.ctx >= 131072 && r.last.ctx <= b, "kept context between the measured 131,072 and the largest that fits");
        check(b - r.last.ctx <= 4 * 1024, "within 4 blocks of the largest fitting context");
        check(r.loads <= tqllm::SplitFit::kMaxLoads + 1, "bounded number of loads");
        const auto chk = load(q8[0], r.last.ctx), chk1 = load(q8[1], r.last.ctx);
        check(chk.loaded && chk.free_b >= margin && chk1.loaded && chk1.free_b >= margin, "the kept context fits on both ranks");
    }
    {   // uneven ranks: the fuller card decides
        const std::vector<Card> uneven = {{24e9, 12e9, 60000}, {24e9, 17e9, 60000}};
        const auto r = run(uneven, 196608, 4096, margin);
        const uint32_t b = best(uneven, margin);
        std::printf("uneven ranks, asked 196608: keeps %u after %d loads (best %u)\n", r.last.ctx, r.loads, b);
        check(r.last.kind == tqllm::SplitFitStep::keep && r.last.ctx <= b && b - r.last.ctx <= 4 * 1024, "the smaller rank's limit");
    }
    {   // fits as asked: one load, kept
        const auto r = run(q8, 65536, 4096, margin);
        check(r.last.kind == tqllm::SplitFitStep::keep && r.last.ctx == 65536 && r.loads == 1, "fits as asked: one load");
    }
    {   // a rank short of the margin (loaded) at the asked context: measured, the estimate lands in one more load or two
        const std::vector<Card> tight = {{24e9, 14.6e9, 60000}, {24e9, 14.4e9, 60000}};
        const auto r = run(tight, 150000 / 1024 * 1024, 4096, margin);
        check(r.last.kind == tqllm::SplitFitStep::keep && r.last.ctx <= best(tight, margin), "short of the margin: a smaller context that fits");
    }
    {   // GB10 pair, 27B TQ_L, 11-14: 102.46 GB free per rank after loading at 131,072, KV ~32.8 KB per position per rank
        const double total = 121.7e9, kv = 32.8e3, w = total - 102.46e9 - 131072 * kv;
        const std::vector<Card> gb10 = {{total, w, kv}, {total, w, kv}};
        const uint64_t m2 = uint64_t(102.46e9) + (2048ull << 20);  // the -fitt of the first pair run: limit ~65.5K
        const auto a = run(gb10, 131072, 4096, m2);
        std::printf("GB10, margin free+2 GiB: keeps %u after %d loads (best %u)\n", a.last.ctx, a.loads, best(gb10, m2));
        check(a.last.kind == tqllm::SplitFitStep::keep && a.last.ctx <= best(gb10, m2) && best(gb10, m2) - a.last.ctx <= 4 * 1024,
              "GB10 run 1: the limit sits at the halving, keep it");
        const uint64_t m1 = uint64_t(102.46e9) + (1024ull << 20);  // limit ~98K, between the halving and the asked context
        const auto b = run(gb10, 131072, 4096, m1);
        std::printf("GB10, margin free+1 GiB: keeps %u after %d loads (best %u)\n", b.last.ctx, b.loads, best(gb10, m1));
        check(b.last.kind == tqllm::SplitFitStep::keep && b.last.ctx > 65536 && b.last.ctx <= best(gb10, m1) &&
              best(gb10, m1) - b.last.ctx <= 4 * 1024, "GB10: the estimate lands between the halving and the asked context");
        // page cache shifting between loads (unified memory): one reading off by +-0.5 / 3 GB, the others exact. The fit
        // keeps a context that fits, in bounded loads, and lands near the limit
        for (double nz : {3e9, 0.5e9, -0.5e9}) {
            noise = {0, nz};
            const auto c = run(gb10, 131072, 4096, m1);
            noise.clear();
            std::printf("GB10 + %+.1f GB page-cache noise: keeps %u after %d loads\n", nz / 1e9, c.last.ctx, c.loads);
            check(c.last.kind == tqllm::SplitFitStep::keep && c.last.ctx <= best(gb10, m1) && c.loads <= tqllm::SplitFit::kMaxLoads,
                  "noisy readings: a fitting context, bounded loads");
            check(best(gb10, m1) - c.last.ctx <= 8 * 1024, "noisy readings: within 8 blocks of the limit");
        }
    }
    {   // nothing fits even at -fitc: fail, naming each rank
        const std::vector<Card> big = {{24e9, 23.8e9, 60000}, {24e9, 22e9, 60000}};
        const auto r = run(big, 131072, 4096, margin);
        std::printf("too big: %s\n", r.error.c_str());
        check(r.last.kind == tqllm::SplitFitStep::fail, "nothing fits: give up");
        check(r.error.find("rank 0") != std::string::npos && r.error.find("rank 1") != std::string::npos, "the error names both ranks");
        check(r.error.find("4096") != std::string::npos, "the error names the smallest context tried");
    }
    std::puts(failures ? "split fit: FAILED" : "split fit: OK");
    return failures ? 1 : 0;
}

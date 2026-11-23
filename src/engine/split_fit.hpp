#pragma once
// The memory fit of --split layer (one process per GPU, each holding a range of layers). Both ranks load at the same
// context, swap what they saw ({loaded, free bytes}) over the control socket and run SplitFit::next on the same pair of
// records, so they agree on every step without further messages: keep this load, load again at another context, or give
// up. A rank's load either fits (loaded with at least `margin` free) or not; the context to try next:
//   - nothing fitted yet: half the context (aligned, not below -fitc);
//   - a fitting context below a failing one: free memory falls linearly with the context, so two readings per rank give
//     its bytes per position and the context that leaves the margin; the smaller over the ranks is loaded next. The two
//     readings come from two fitting loads (one more at half the fitting context when only one exists: the failing load
//     ran out of memory and read nothing) or from a fitting load and a failing one that loaded short of the margin;
//     with two measured failing loads, also the line through them, and the smaller estimate above the fitting context;
//     when the estimate fails too, the midpoint, while more than kBisectSpan contexts apart. On unified memory (GB10) a
//     free reading includes page cache the OS reclaims, which moves between loads: a slope at or below kMinPerPos bytes
//     per position is noise, not KV, and counts as no estimate;
//   - the last fitting context again when a later try failed (the engine was dropped).
// Pure host logic (tests/split_fit.cpp); the loads, the memory readings and the socket are the caller's.
#include "fit_context.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace tqllm {

struct SplitLoad {         // one rank's result of one load attempt
    bool loaded = false;   // false: out of memory while loading
    uint64_t free_b = 0;   // free device bytes after the load (0 when not loaded)
    uint64_t total_b = 0;  // device memory
};

struct SplitFitStep {
    enum Kind { keep, load, fail } kind = keep;
    uint32_t ctx = 0;      // load: the context to load at; keep: the context of the engine just loaded
};

class SplitFit {
public:
    static constexpr uint32_t kAlign = 1024;
    static constexpr int kMaxLoads = 8;
    SplitFit(uint32_t asked, uint32_t min_ctx, uint64_t margin) : asked_(asked), min_ctx_(std::max<uint32_t>(min_ctx, 1)), margin_(margin) {}

    bool fits(const std::vector<SplitLoad> & ranks) const {
        return std::all_of(ranks.begin(), ranks.end(), [&](const SplitLoad & r) { return r.loaded && r.free_b >= margin_; });
    }

    // after the load at `ctx` on every rank: what both ranks do next
    SplitFitStep next(uint32_t ctx, const std::vector<SplitLoad> & ranks) {
        ++loads_;
        if (fits(ranks)) {
            oks_.push_back({ctx, ranks});
            if (ctx >= ok_) { ok_ = ctx; ok_free_ = ranks; }
            if (ctx == ok_ && (ctx >= asked_ || !bad_ || loads_ >= kMaxLoads)) return {SplitFitStep::keep, ctx};
            const uint32_t up = loads_ < kMaxLoads ? between() : ok_;
            if (up && up != ok_) return {SplitFitStep::load, up};  // an estimate above ok_, or the measuring probe below it
            return ctx == ok_ ? SplitFitStep{SplitFitStep::keep, ctx} : SplitFitStep{SplitFitStep::load, ok_};
        }
        if (!bad_ || ctx < bad_) { bad_ = ctx; bad_free_ = ranks; }
        if (std::all_of(ranks.begin(), ranks.end(), [](const SplitLoad & r) { return r.loaded; })) fails_.push_back({ctx, ranks});
        last_fail_ = ranks;
        if (ok_) {  // a fitting context is known: try between (or measure below it), or go back to it
            const uint32_t up = loads_ < kMaxLoads ? between() : ok_;
            return {SplitFitStep::load, up ? up : ok_};
        }
        if (ctx <= min_ctx_) return {SplitFitStep::fail, ctx};
        const uint32_t half = std::max(min_ctx_, ctx / 2 / kAlign * kAlign);
        return {SplitFitStep::load, half >= ctx ? min_ctx_ : half};
    }

    // the error when nothing fits even at -fitc: what each rank saw at the smallest context
    std::string failure(uint32_t ctx) const {
        std::string s = "--split layer: the model does not fit on these GPUs even at context " + std::to_string(ctx) + " (-fitc)";
        for (size_t r = 0; r < last_fail_.size(); ++r) {
            const auto & x = last_fail_[r];
            char buf[200];
            if (!x.loaded)
                std::snprintf(buf, sizeof buf, "; rank %zu: out of memory while loading its layers (%.2f GB device)", r, double(x.total_b) / 1e9);
            else if (x.free_b < margin_)
                std::snprintf(buf, sizeof buf, "; rank %zu: %.2f GB free after loading, needs %.2f GB (-fitt margin)", r,
                              double(x.free_b) / 1e9, double(margin_) / 1e9);
            else
                std::snprintf(buf, sizeof buf, "; rank %zu: fits (%.2f GB free)", r, double(x.free_b) / 1e9);
            s += buf;
        }
        return s + ". Use --split-at to move layers to the card with room, fewer slots (--np), or a smaller model file";
    }
    uint32_t ok() const { return ok_; }

private:
    static constexpr uint32_t kBisectSpan = 4 * kAlign;
    static constexpr double kMinPerPos = 1024;  // bytes per position per rank; real KV is several KB
    // a context between the largest fitting one and the smallest failing one, or ok_ when none is worth a load
    uint32_t between() const {
        if (!bad_ || bad_ <= ok_) return ok_;
        auto limit = [&](uint32_t lo, const std::vector<SplitLoad> & at_lo, uint32_t hi, const std::vector<SplitLoad> & at_hi, bool up) {
            // per rank: bytes per position from the two readings, then the context that leaves the margin
            uint64_t c = ~0ull;
            for (size_t r = 0; r < at_lo.size() && r < at_hi.size(); ++r) {
                if (!at_lo[r].loaded || !at_hi[r].loaded || at_lo[r].free_b <= at_hi[r].free_b) return uint64_t(0);
                const double per_tok = double(at_lo[r].free_b - at_hi[r].free_b) / double(hi - lo);
                if (per_tok <= kMinPerPos) return uint64_t(0);
                const auto & base = up ? at_hi[r] : at_lo[r];
                const double x = double(up ? hi : lo) + (double(base.free_b) - double(margin_)) / per_tok;  // below the margin: back down
                if (x <= 0) return uint64_t(0);
                c = std::min<uint64_t>(c, uint64_t(x));
            }
            return c == ~0ull ? uint64_t(0) : c / kAlign * kAlign;
        };
        uint64_t est = 0;
        const bool bad_measured = std::all_of(bad_free_.begin(), bad_free_.end(), [](const SplitLoad & r) { return r.loaded; });
        if (bad_measured) {
            est = limit(ok_, ok_free_, bad_, bad_free_, false);
            // the two smallest measured failing loads: readings taken after ok_'s, so a page-cache shift since then does
            // not repeat the same overshoot; the smaller of the two estimates above ok_ is loaded
            const Ok * b2 = nullptr;
            for (const auto & f : fails_) if (f.ctx > bad_ && (!b2 || f.ctx < b2->ctx)) b2 = &f;
            if (b2) {
                const uint64_t e2 = limit(bad_, bad_free_, b2->ctx, b2->ranks, false);
                if (e2 > ok_ && (est <= ok_ || e2 < est)) est = e2;
            }
        }
        else {
            const Ok * lower = nullptr;  // the largest fitting load below ok_
            for (const auto & o : oks_) if (o.ctx < ok_ && (!lower || o.ctx > lower->ctx)) lower = &o;
            if (!lower) {  // one more reading: half the fitting context (it fits; it only measures)
                const uint32_t probe = std::max(min_ctx_, ok_ / 2 / kAlign * kAlign);
                if (probe < ok_ && !probed_) { probed_ = true; return probe; }
            } else est = limit(lower->ctx, lower->ranks, ok_, ok_free_, true);
        }
        if (est >= bad_) est = uint64_t(bad_ - 1) / kAlign * kAlign;
        if (est > ok_ && uint32_t(est) != tried_est_) { tried_est_ = uint32_t(est); return uint32_t(est); }
        if (est && est <= ok_) return ok_;  // two readings put the limit at ok_ (within one block): no further load
        if (bad_ - ok_ <= kBisectSpan) return ok_;
        const uint32_t mid = (ok_ + (bad_ - ok_) / 2) / kAlign * kAlign;
        return mid > ok_ ? mid : ok_;
    }
    struct Ok { uint32_t ctx; std::vector<SplitLoad> ranks; };
    std::vector<Ok> oks_, fails_;  // fails_: the failing loads every rank finished (measured)
    mutable bool probed_ = false;
    uint32_t asked_, min_ctx_;
    uint64_t margin_;
    uint32_t ok_ = 0, bad_ = 0;
    mutable uint32_t tried_est_ = 0;
    int loads_ = 0;
    std::vector<SplitLoad> ok_free_, bad_free_, last_fail_;
};

}  // namespace tqllm

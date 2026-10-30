#pragma once
// Verify width per step (7 or 15 drafted tokens of the block-16 draft): the width with more expected tokens per unit of
// step time. A 7-token step sees min(a, 7) of the acceptance a a 15-token step would see (same draft), so
//   E7  = 1 + mean(min(a, 7))                      (all steps of the request)
//   E15 = E7 + P(a >= 7) * mean(a - 7 | a >= 7)    (tail from this request's 15-token steps only: other prompts'
//                                                   tails misjudge it, j1024 u2 94.9 -> 84.3)
// P(a >= 7) = 0 gives E15 = E7; full accepts without a tail sample yet take a 15-token step to measure one, and after a
// narrow full accept the next step is wide while tail samples < sqrt(full accepts) (a narrow step cannot see its tail).
// Replayed on per-step traces with tools/width_policy_sim.cpp (j1031).
// Step time per width: this request's steps plus, as one sample, the mean of the served request nearest in context
// length that timed that width (one sample per width flipped doc8k on noise, j1024; the request's own steps soon outweigh it). The first
// step of a request (graph build) is not timed, and the first step of each width in the process (its kernels' JIT) is
// left out of the served time lines (a 3x narrow sample as the only prior kept 8K / 32K requests wide for good, j1086
// -3..-7 %). Exactness does not depend on the choice.
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace tqllm::mlx_backend {
class WidthPolicy {
public:
    static constexpr int kNarrow = 7, kWide = 15;
    // prompts from this length draft the drafter's native 8-row block only (j1340 / j1341 / j1343, same binary, texts equal):
    // 64K 44.7 -> 52.0, 128K 34.2 -> 38.3, 256K 20.2 -> 23.9, 8K 55.2 -> 59.0, 16K 44.3 -> 45.2 tok/s; doc8k (7.9K)
    // (7,884 tokens) equal, the short gate 3-9 % slower on narrow drafts, and mixing native narrow steps into the measured
    // policy below the length loses doc8k 6-8 % (j1331, j1343). The narrow step's attention and weights cost half a wide
    // step's, and the native draft accepts more than the head of a 16-draft (j934)
    static constexpr int kNativeFrom = 8000;
    bool native() const {return req_.context >= kNativeFrom;}
    void begin(int context = 0) {
        if (req_.served[0].n + req_.served[1].n > 0) served_.push_back({req_.context, {req_.served[0], req_.served[1]}});
        req_ = {};
        req_.context = context;
        prior_[0] = prior_[1] = Mean{};
        for (int w = 0; w < 2; ++w) {  // per width: the nearest served context that timed it
            long best = -1;
            for (const auto& s : served_) {
                const long d = std::labs(long(s.context) - long(context));
                if (s.time[w].n > 0 && (best < 0 || d < best)) {best = d; prior_[w] = s.time[w].one();}
            }
        }
    }
    int choose(int) const {
        if (req_.context >= kNativeFrom) return kNarrow;
        const Mean t15 = req_.time[1] + prior_[1], t7 = req_.time[0] + prior_[0];
        if (t15.n == 0 || req_.steps == 0) return kWide;
        if (t7.n == 0) return kNarrow;
        const double e7 = 1.0 + req_.sum_min7 / req_.steps, p7 = req_.full7 / req_.steps;
        if (p7 > 0 && req_.tail.n == 0) return kWide;
        // a narrow full accept hides its tail: keep sampling it (tail samples >= sqrt(full accepts)), or a request whose
        // first tail samples were short stays narrow for good (j1030: u2 / code2 / code3 -8..17 %)
        if (req_.last_full_narrow && req_.tail.n * req_.tail.n < req_.full7) return kWide;
        const double e15 = e7 + p7 * req_.tail.mean();
        return e15 / t15.mean() > e7 / t7.mean() ? kWide : kNarrow;
    }
    void record(int width, int accepted, int, double ms) {
        if (width != kNarrow && width != kWide) return;
        if (req_.steps > 0) {  // the first step of a request builds its graphs
            req_.time[width == kWide].add(ms);
            if (built_[width == kWide]) req_.served[width == kWide].add(ms);
        }
        built_[width == kWide] = true;
        const int m7 = accepted < kNarrow ? accepted : kNarrow;
        req_.steps += 1; req_.sum_min7 += m7; req_.full7 += m7 == kNarrow;
        if (width == kWide && accepted >= kNarrow) req_.tail.add(accepted - kNarrow);
        ++(width == kWide ? req_.wide : req_.narrow);
        req_.last_full_narrow = width == kNarrow && accepted >= kNarrow;
    }
    int narrow_steps() const {return req_.narrow;}
    int wide_steps() const {return req_.wide;}
private:
    struct Mean {
        double n = 0, sum = 0;
        void add(double v) {n += 1; sum += v;}
        double mean() const {return n > 0 ? sum / n : 0.0;}
        Mean operator+(const Mean& o) const {return {n + o.n, sum + o.sum};}
        Mean one() const {return n > 0 ? Mean{1, mean()} : Mean{};}
    };
    struct Request {int context = 0; double steps = 0, sum_min7 = 0, full7 = 0; Mean time[2], served[2], tail; int narrow = 0, wide = 0; bool last_full_narrow = false;};
    struct Served {int context; Mean time[2];};
    Request req_{};
    Mean prior_[2]{};
    bool built_[2]{};
    std::vector<Served> served_;
};
}  // namespace tqllm::mlx_backend

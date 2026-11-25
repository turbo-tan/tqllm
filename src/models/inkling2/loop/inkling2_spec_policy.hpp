#pragma once
// Inkling-2-Mini greedy MTP decode: per request, draft + verify only while it is measured faster than plain steps.
// Both paths give the same tokens (exact greedy), so the choice only changes speed. The request starts with two plain
// steps (their time is the plain reference), then drafts L = round(EMA of accepted drafts) + 1 (2 .. 7) a step. After
// kWarm spec steps (kProbe after a plain run), when the EMA of tokens gained per spec step over the EMA of its time is
// below 1 / plain step time, a run of plain steps follows (kPlainRun, doubling after every losing probe up to
// kPlainMax, back to kPlainRun when a probe wins), then spec is probed again. The gain EMA starts from a prior (3
// tokens a step), so one bad first step does not decide.
// Drafts per spec step: the L that maximises expected tokens per second from this request's own record, so the choice
// follows the device (verify rows cost very differently on a 3090 and on a GB10): the acceptance of draft i given drafts
// 1 .. i - 1 were kept, and the step time per L as a line a + b L fitted to the measured step times (both decaying);
// a neighbour of the best L is tried every kExplore-th spec step. Until two lengths are measured: round(EMA of accepted) + 1.
// With per-draft confidences (DFlash2: the softmax weight of each drafted token among its 16 head candidates), the
// verify window is chosen per step instead (window()): the L whose expected tokens, 1 + sum over i <= L of the product of
// the calibrated chances of drafts 1 .. i, per the step-time line a + b L is largest. The calibration maps a confidence
// to the measured chance that a draft which was verified was kept (kBins bins, decaying, prior = the confidence itself),
// so a drafter that is over- or under-confident on this device and request is corrected from its own record.
#include <algorithm>
#include <array>

namespace tqllm {

class Inkling2SpecPolicy {
public:
    static constexpr int kPlainRun = 8, kPlainMax = 64, kWarm = 4, kProbe = 2, kMaxL = 7, kExplore = 8;
    explicit Inkling2SpecPolicy(int fixed_depth = 0) : fixed_(fixed_depth) {}
    // drafts for the next step, 0 = a plain step
    int next(int room) {
        if (plain_left_ > 0) { --plain_left_; return 0; }
        const int L = fixed_ ? fixed_ : depth();
        return std::min(L, room - 1) >= 1 ? std::min(L, room - 1) : 0;
    }
    void plain_done(double secs) { t_plain_ = t_plain_ > 0 ? 0.7 * t_plain_ + 0.3 * secs : secs; }
    // verify window for drafts with confidences conf[0 .. n) (n >= 1): 1 .. n, or the fixed depth
    int window(const float * conf, int n) {
        if (fixed_) return std::min(fixed_, n);
        double a, b;
        if (!line(a, b)) {  // no line yet: the plain step as the base, a row at a fifth of it (GB10 ~5 ms on ~25 ms)
            if (t_plain_ <= 0) return n;
            a = t_plain_; b = 0.2 * t_plain_;
        }
        int best = 1;
        double e = 1.0, run = 1.0, best_rate = 0.0;
        for (int L = 1; L <= n; ++L) {
            run *= chance(conf[L - 1]); e += run;
            const double rate = e / (a + b * L);
            if (rate > best_rate) { best_rate = rate; best = L; }
        }
        if (++choices_ % kExplore == 0) best = std::clamp(best + ((choices_ / kExplore) % 2 ? 1 : -1), 1, n);  // keep the line fitted
        return best;
    }
    // the calibrated chance that a verified draft of confidence c is kept
    double chance(float c) const {
        const int k = bin(c);
        const double mid = (k + 0.5) / kBins;
        return (ckept_[k] + kPrior * mid) / (creach_[k] + kPrior);
    }
    void spec_done(int accepted, int L, double secs, const float * conf) {
        if (conf) {
            for (int j = 0; j < kBins; ++j) { creach_[j] *= kDecay; ckept_[j] *= kDecay; }
            for (int i = 1; i <= L && i <= accepted + 1; ++i) {  // drafts 1 .. accepted kept, draft accepted + 1 rejected
                const int k = bin(conf[i - 1]);
                creach_[k] += 1.0; ckept_[k] += i <= accepted ? 1.0 : 0.0;
            }
        }
        spec_done(accepted, L, secs);
    }
    void spec_done(int accepted, int L, double secs) {
        if (L >= 1 && L <= kMaxL) {
            for (int i = 1; i <= kMaxL; ++i) { reach_[i] *= kDecay; kept_[i] *= kDecay; }
            for (int i = 1; i <= L && i <= accepted + 1; ++i) { reach_[i] += 1.0; kept_[i] += i <= accepted ? 1.0 : 0.0; }
            t_len_[L] = t_len_[L] > 0 ? 0.7 * t_len_[L] + 0.3 * secs : secs;
        }
        ema_acc_ = 0.7 * ema_acc_ + 0.3 * (accepted == L ? accepted + 1.0 : double(accepted));
        gain_ = 0.7 * gain_ + 0.3 * (accepted + 1.0);
        t_spec_ = t_spec_ > 0 ? 0.7 * t_spec_ + 0.3 * secs : secs;
        ++spec_steps_;
        if (fixed_ || t_plain_ <= 0 || ++since_ < (probing_ ? kProbe : kWarm)) return;
        if (gain_ / t_spec_ < 1.0 / t_plain_) {  // spec loses: a run of plain steps, longer after each losing probe
            run_ = probing_ ? std::min(kPlainMax, run_ * 2) : kPlainRun;
            plain_left_ = run_;
            probing_ = true;
        } else {
            run_ = kPlainRun;
            probing_ = false;
        }
        since_ = 0;
    }
    int spec_steps() const { return spec_steps_; }
    // expected tokens of a spec step with L drafts (the verified token included), from the acceptance record
    double expected_tokens(int L) const {
        double e = 1.0, run = 1.0;
        for (int i = 1; i <= L; ++i) { run *= (kept_[i] + 0.7 * kPrior) / (reach_[i] + kPrior); e += run; }
        return e;
    }

private:
    static constexpr double kDecay = 0.98, kPrior = 2.0;  // record half-life ~35 steps; prior acceptance 0.7 with weight 2
    static constexpr int kBins = 10;
    static int bin(float c) { return std::clamp(int(c * kBins), 0, kBins - 1); }
    // the step-time line a + b L over the measured lengths (false until two lengths are measured)
    bool line(double & a, double & b) const {
        double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int L = 1; L <= kMaxL; ++L)
            if (t_len_[L] > 0) { n += 1; sx += L; sy += t_len_[L]; sxx += double(L) * L; sxy += L * t_len_[L]; }
        if (n < 2) return false;
        b = std::max(0.0, (n * sxy - sx * sy) / std::max(1e-12, n * sxx - sx * sx)); a = std::max(1e-6, (sy - b * sx) / n);
        return true;
    }
    int depth() {
        double a, b;
        if (!line(a, b)) return std::max(2, std::min(kMaxL, int(ema_acc_ + 0.5) + 1));
        int best = 1;
        double best_rate = 0;
        for (int L = 1; L <= kMaxL; ++L) {
            const double rate = expected_tokens(L) / (a + b * L);
            if (rate > best_rate) { best_rate = rate; best = L; }
        }
        if (++choices_ % kExplore == 0) best = std::clamp(best + ((choices_ / kExplore) % 2 ? 1 : -1), 1, kMaxL);  // keep the line fitted
        return best;
    }
    std::array<double, kMaxL + 1> reach_{}, kept_{}, t_len_{};
    std::array<double, kBins> creach_{}, ckept_{};
    long choices_ = 0;
    int fixed_ = 0, plain_left_ = 2, spec_steps_ = 0, since_ = 0, run_ = kPlainRun;
    bool probing_ = false;
    double ema_acc_ = 3.0, gain_ = 3.0, t_spec_ = 0, t_plain_ = 0;
};

}  // namespace tqllm

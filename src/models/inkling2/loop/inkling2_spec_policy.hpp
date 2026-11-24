#pragma once
// Inkling-2-Mini greedy MTP decode: per request, draft + verify only while it is measured faster than plain steps.
// Both paths give the same tokens (exact greedy), so the choice only changes speed. The request starts with two plain
// steps (their time is the plain reference), then drafts L = round(EMA of accepted drafts) + 1 (2 .. 7) a step. After
// kWarm spec steps (kProbe after a plain run), when the EMA of tokens gained per spec step over the EMA of its time is
// below 1 / plain step time, a run of plain steps follows (kPlainRun, doubling after every losing probe up to
// kPlainMax, back to kPlainRun when a probe wins), then spec is probed again. The gain EMA starts from a prior (3
// tokens a step), so one bad first step does not decide.
#include <algorithm>

namespace tqllm {

class Inkling2SpecPolicy {
public:
    static constexpr int kPlainRun = 8, kPlainMax = 64, kWarm = 4, kProbe = 2;
    explicit Inkling2SpecPolicy(int fixed_depth = 0) : fixed_(fixed_depth) {}
    // drafts for the next step, 0 = a plain step
    int next(int room) {
        if (plain_left_ > 0) { --plain_left_; return 0; }
        const int L = fixed_ ? fixed_ : std::max(2, std::min(7, int(ema_acc_ + 0.5) + 1));
        return std::min(L, room - 1) >= 1 ? std::min(L, room - 1) : 0;
    }
    void plain_done(double secs) { t_plain_ = t_plain_ > 0 ? 0.7 * t_plain_ + 0.3 * secs : secs; }
    void spec_done(int accepted, int L, double secs) {
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

private:
    int fixed_ = 0, plain_left_ = 2, spec_steps_ = 0, since_ = 0, run_ = kPlainRun;
    bool probing_ = false;
    double ema_acc_ = 3.0, gain_ = 3.0, t_spec_ = 0, t_plain_ = 0;
};

}  // namespace tqllm

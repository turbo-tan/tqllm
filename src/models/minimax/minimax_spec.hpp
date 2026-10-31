#pragma once
// Draft length per speculative step from the running acceptance rate. A verify step of k drafts costs about
// t0 + c (k + 1) (the routed experts of the k + 1 verified tokens are read from memory; GB10 MiniMax: t0 ~ 70 ms, c ~ 12 ms)
// and yields (1 - a^(k+1)) / (1 - a) tokens on average at a per-draft acceptance a: the length that maximises tokens
// per unit cost, at least 1 so the rate keeps being measured.
#include <algorithm>
#include <cmath>

inline int minimax_draft_len(double accept, int kmax, double cost_per_token = 12.0 / 70.0) {
    const double a = std::clamp(accept, 0.0, 0.99);
    int best = 1;
    double best_rate = -1;
    for (int k = 1; k <= std::max(1, kmax); ++k) {
        const double tokens = (1.0 - std::pow(a, k + 1)) / (1.0 - a);
        const double rate = tokens / (1.0 + cost_per_token * (k + 1));
        if (rate > best_rate + 1e-12) { best_rate = rate; best = k; }
    }
    return best;
}

// running per-draft acceptance: a step that drafted d and kept k made k successful trials, plus one failed trial when
// k < d (drafts past the first miss were never tried); successes / trials over a decaying window
struct MinimaxAcceptRate {
    double s = 3.0, t = 5.0;  // prior: 0.6 (the DSpark mean on accept40)
    double rate() const { return s / t; }
    void update(int drafted, int accepted, double keep = 0.9) {
        if (drafted <= 0) return;
        s = keep * s + accepted;
        t = keep * t + accepted + (accepted < drafted ? 1 : 0);
    }
};

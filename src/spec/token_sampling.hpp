#pragma once
// Token sampling shared by the engines (temperature > 0): a position's top-16 candidates with exact logits (the head's
// top-16 kernel) become a distribution (repeat / frequency / presence penalties, temperature, top-k <= 16, top-p); draws
// come from it, and
// speculative acceptance against it gives exact samples of the target with deterministic drafts (accept draft d with
// p(d), else resample from p without d; bonus token from p).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace tqllm {

// Penalties on the tokens of the last kPresenceLastN outputs (llama.cpp order): repeat (LM Studio / llama.cpp repeat_penalty:
// a positive logit divided by it, a negative one multiplied), then frequency x count + presence subtracted.
struct SamplingParams {
    float temperature = 0.f, top_p = 1.f, presence = 0.f; int top_k = 0;
    float repeat = 1.f, frequency = 0.f;
    bool penalized() const { return presence > 0.f || frequency > 0.f || repeat != 1.f; }
};
constexpr int kPresenceLastN = 64;  // llama.cpp's default penalty window (repeat_last_n)
struct TopDist { uint32_t id[16]; double p[16]; int n = 0; };

// logits16 / ids16: one position's 16 largest logits and their ids (any order); recent: the output before it
inline TopDist top_dist(const float * logits16, const uint32_t * ids16, const SamplingParams & s,
                        const std::vector<uint32_t> & recent = {}) {
    TopDist d;
    int ord[16];
    for (int i = 0; i < 16; ++i) ord[i] = i;
    float lg[16];
    std::copy_n(logits16, 16, lg);
    if (s.penalized()) {
        const size_t r0 = recent.size() > size_t(kPresenceLastN) ? recent.size() - kPresenceLastN : 0;
        for (int i = 0; i < 16; ++i) {
            const long cnt = std::count(recent.begin() + long(r0), recent.end(), ids16[i]);
            if (!cnt) continue;
            if (s.repeat != 1.f) lg[i] = lg[i] > 0.f ? lg[i] / s.repeat : lg[i] * s.repeat;
            lg[i] -= s.frequency * float(cnt) + s.presence;
        }
    }
    std::sort(ord, ord + 16, [&](int a, int b) { return lg[a] > lg[b]; });
    const int k = s.top_k > 0 ? std::min(s.top_k, 16) : 16;
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        d.id[i] = ids16[ord[i]];
        d.p[i] = std::exp(double(lg[ord[i]] - lg[ord[0]]) / s.temperature);
        sum += d.p[i];
    }
    double cum = 0;
    d.n = k;
    for (int i = 0; i < k; ++i) {
        cum += d.p[i] / sum;
        if (cum >= s.top_p) { d.n = i + 1; break; }
    }
    double s2 = 0;
    for (int i = 0; i < d.n; ++i) s2 += d.p[i];
    for (int i = 0; i < d.n; ++i) d.p[i] /= s2;
    return d;
}

template<class Rng> uint32_t draw(const TopDist & d, Rng & rng, uint32_t exclude = UINT32_MAX) {
    double total = 0;
    for (int i = 0; i < d.n; ++i) total += d.id[i] == exclude ? 0 : d.p[i];
    double u = std::uniform_real_distribution<double>(0, total)(rng);
    for (int i = 0; i < d.n; ++i) {
        if (d.id[i] == exclude) continue;
        if ((u -= d.p[i]) <= 0) return d.id[i];
    }
    for (int i = d.n - 1; i >= 0; --i) if (d.id[i] != exclude) return d.id[i];
    return d.id[0];
}

// Speculative acceptance of n drafts: dist_at(j, recent) is position j's distribution (j = n: the bonus position).
// Writes res[0..j] (res[j] the resampled or bonus token), returns j accepted drafts; with penalties (track), recent grows
// by each accepted draft.
template<class DistAt, class Rng>
int accept_drafts(const uint32_t * draft, int n, DistAt && dist_at, Rng & rng, uint32_t * res, std::vector<uint32_t> & recent,
                  bool track) {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int j = 0; j < n; ++j) {
        const TopDist d = dist_at(j, recent);
        double q = 0;
        for (int i = 0; i < d.n; ++i) if (d.id[i] == draft[j]) q = d.p[i];
        if (uni(rng) >= q) {
            res[j] = draw(d, rng, draft[j]);
            return j;
        }
        if (track) recent.push_back(draft[j]);
    }
    res[n] = draw(dist_at(n, recent), rng);
    return n;
}

}  // namespace tqllm

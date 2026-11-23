#pragma once
// Token sampling shared by the engines (temperature > 0): a position's top-16 candidates with exact logits (the head's
// top-16 kernel) become a distribution (repeat / frequency / presence penalties, temperature, top-k <= 16, top-p); draws
// come from it, and
// speculative acceptance against it gives exact samples of the target with deterministic drafts (accept draft d with
// p(d), else resample from p without d; bonus token from p).
#include "gumbel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace tqllm {

// Penalties on the tokens of the last kPresenceLastN outputs (llama.cpp order): repeat (LM Studio / llama.cpp repeat_penalty:
// a positive logit divided by it, a negative one multiplied), then frequency x count + presence subtracted.
struct SamplingParams {
    float temperature = 0.f, top_p = 1.f, presence = 0.f; int top_k = 0;
    float repeat = 1.f, frequency = 0.f;
    uint64_t seed = 0;  // Gumbel noise of exact sampling (gumbel.hpp)
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

// ---- Exact sampling over the whole vocabulary (the qwen35 engine; HeadSampler in head_sampler.hpp) ----
// The target distribution at a position: penalties on the logits of the last kPresenceLastN output tokens, then the
// top_k rows (all rows when top_k <= 0), then the smallest prefix (logits descending, ties by lower id) whose softmax(l / T)
// mass reaches top_p of the top_k mass. The draw is the Gumbel-max argmax over that prefix of l + T g(key, row): an exact
// sample of the prefix's renormalised softmax, and a deterministic function of (seed, position). Speculative acceptance
// is then a plain comparison: a draft is accepted when it equals the target's draw at its position, which keeps every
// output token an exact sample whatever the drafter proposed.

inline double gumbel(uint64_t key, uint32_t row) {
    const double u = (double(gumbel_bits(key, row)) + 0.5) * double(kGumbelUnit);
    return -std::log(-std::log(u));
}
// whole-vocabulary draw: no top-p / top-k cut, every row competes in the Gumbel argmax
inline bool full_vocab_draw(const SamplingParams & s, int rows) { return s.top_p >= 1.f && (s.top_k <= 0 || s.top_k >= rows); }

// (id, count) of the tokens in the penalty window (the last kPresenceLastN of recent), empty without penalties
inline std::vector<std::pair<uint32_t, int>> penalty_counts(const SamplingParams & s, const std::vector<uint32_t> & recent) {
    std::vector<std::pair<uint32_t, int>> c;
    if (!s.penalized()) return c;
    const size_t r0 = recent.size() > size_t(kPresenceLastN) ? recent.size() - kPresenceLastN : 0;
    for (size_t i = r0; i < recent.size(); ++i) {
        auto it = std::find_if(c.begin(), c.end(), [&](const auto & e) { return e.first == recent[i]; });
        if (it == c.end()) c.push_back({recent[i], 1});
        else ++it->second;
    }
    return c;
}
inline float penalized_logit(float lg, int cnt, const SamplingParams & s) {  // top_dist's formula
    if (!cnt) return lg;
    if (s.repeat != 1.f) lg = lg > 0.f ? lg / s.repeat : lg * s.repeat;
    return lg - (s.frequency * float(cnt) + s.presence);
}
inline int penalty_count(const std::vector<std::pair<uint32_t, int>> & pc, uint32_t id) {
    for (const auto & e : pc) if (e.first == id) return e.second;
    return 0;
}

namespace sampling_detail {
// Gumbel argmax over rows ids[0, n) with penalised logits pl (ties: lower id)
inline uint32_t gumbel_argmax(const uint32_t * ids, const float * pl, size_t n, double T, uint64_t key) {
    double best = -std::numeric_limits<double>::infinity();
    uint32_t bi = ids[0];
    for (size_t i = 0; i < n; ++i) {
        const double v = double(pl[i]) + T * gumbel(key, ids[i]);
        if (v > best || (v == best && ids[i] < bi)) best = v, bi = ids[i];
    }
    return bi;
}
}  // namespace sampling_detail

// Reference draw from all `rows` exact logits of one position (TQLLM_DEBUG=full_head, and the screened head's fallback).
inline uint32_t sample_reference(const float * logits, int rows, const SamplingParams & s, const std::vector<uint32_t> & recent,
                                 uint64_t key) {
    std::vector<float> pl(logits, logits + rows);
    for (const auto & [id, cnt] : penalty_counts(s, recent))
        if (id < uint32_t(rows)) pl[id] = penalized_logit(pl[id], cnt, s);
    std::vector<uint32_t> ord(static_cast<size_t>(rows));
    std::iota(ord.begin(), ord.end(), 0u);
    const double T = s.temperature;
    if (full_vocab_draw(s, rows)) return sampling_detail::gumbel_argmax(ord.data(), pl.data(), ord.size(), T, key);
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return pl[a] > pl[b] || (pl[a] == pl[b] && a < b); });
    const int k = s.top_k > 0 ? std::min(s.top_k, rows) : rows;
    const double ref = pl[ord[0]];
    std::vector<double> w(static_cast<size_t>(k));
    double Z = 0;
    for (int i = 0; i < k; ++i) Z += w[size_t(i)] = std::exp((double(pl[ord[size_t(i)]]) - ref) / T);
    int n = k;
    double cum = 0;
    for (int i = 0; i < k; ++i) {
        cum += w[size_t(i)];
        if (cum >= double(s.top_p) * Z) { n = i + 1; break; }
    }
    std::vector<float> plo(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) plo[size_t(i)] = pl[ord[size_t(i)]];
    return sampling_detail::gumbel_argmax(ord.data(), plo.data(), size_t(n), T, key);
}

// What a screened head knows about one position: exact logits of some rows, and bounds for all the others.
struct ExactRows {
    std::vector<uint32_t> id;  // distinct rows with exact (unpenalised) logits; must hold every penalised token
    std::vector<float> lg;
    float m_out = -std::numeric_limits<float>::infinity();  // every other row's exact logit is <= m_out
    double z_lo = 0, z_hi = 0;  // the sum over every other row of exp((l - z_ref) / T) lies in [z_lo, z_hi]
    float z_ref = 0;
    bool gumbel_closed = false;  // full-vocabulary draw: no other row can win the Gumbel argmax
    int rows = 0;               // vocabulary rows
};

// The exact draw from the bounds, or false when they cannot decide it (the caller re-scores more rows). Equal to
// sample_reference on the full logits whenever it returns true (tests/sampling.cpp).
inline bool sample_bounded(const ExactRows & e, const SamplingParams & s, const std::vector<std::pair<uint32_t, int>> & pc,
                           uint64_t key, uint32_t & tok) {
    const size_t nc = e.id.size();
    if (!nc) return false;
    std::vector<float> pl(nc);
    for (size_t i = 0; i < nc; ++i) pl[i] = penalized_logit(e.lg[i], penalty_count(pc, e.id[i]), s);
    const double T = s.temperature;
    if (full_vocab_draw(s, e.rows)) {
        if (!e.gumbel_closed) return false;
        tok = sampling_detail::gumbel_argmax(e.id.data(), pl.data(), nc, T, key);
        return true;
    }
    std::vector<size_t> ord(nc);
    std::iota(ord.begin(), ord.end(), size_t(0));
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
        return pl[a] > pl[b] || (pl[a] == pl[b] && e.id[a] < e.id[b]);
    });
    size_t m = 0;  // the first m rows are the vocabulary's top m: every other row is <= m_out
    while (m < nc && pl[ord[m]] > e.m_out) ++m;
    if (!m) return false;
    const double ref = pl[ord[0]];
    std::vector<double> w(nc);
    double sum_all = 0;
    for (size_t i = 0; i < nc; ++i) sum_all += w[i] = std::exp((double(pl[ord[i]]) - ref) / T);
    const double rescale = std::exp((double(e.z_ref) - ref) / T);
    const double rest_lo = e.z_lo * rescale, rest_hi = e.z_hi * rescale;
    const size_t k = s.top_k > 0 && s.top_k < e.rows ? size_t(s.top_k) : SIZE_MAX;
    double zlo, zhi;
    if (k <= m) {  // the top-k mass is known exactly (the same sum, in the same order, as sample_reference)
        zlo = zhi = 0;
        for (size_t i = 0; i < k; ++i) zlo += w[i];
        zhi = zlo;
    } else {
        zhi = sum_all + rest_hi;
        if (k == SIZE_MAX) zlo = sum_all + rest_lo;
        else { zlo = 0; for (size_t i = 0; i < m; ++i) zlo += w[i]; }
    }
    const size_t lim = std::min(k, m);
    const double tp = s.top_p;
    auto first_reach = [&](double Z) {
        double cum = 0;
        for (size_t i = 0; i < lim; ++i) {
            cum += w[i];
            if (cum >= tp * Z) return i + 1;
        }
        return lim == k ? k : size_t(0);  // top-k reached: the whole top-k (0: the prefix runs past the known rows)
    };
    const size_t n_hi = first_reach(zhi), n_lo = first_reach(zlo);
    if (!n_hi || n_lo != n_hi) return false;
    std::vector<uint32_t> ids(n_hi);
    std::vector<float> plo(n_hi);
    for (size_t i = 0; i < n_hi; ++i) ids[i] = e.id[ord[i]], plo[i] = pl[ord[i]];
    tok = sampling_detail::gumbel_argmax(ids.data(), plo.data(), n_hi, T, key);
    return true;
}

}  // namespace tqllm

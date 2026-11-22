// src/spec/token_sampling.hpp: exact sampling (sample_bounded on screened bounds == sample_reference on all logits), and
// the top-16 helpers against the engines' previous private copies (Qwen35Engine::dist / draw / accept, 10-31): the same
// distributions, the same tokens and the same RNG draws for random top-16 logits, every parameter combination.
#include "spec/token_sampling.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
struct OldDist { uint32_t id[16]; double p[16]; int n = 0; };
OldDist old_dist(const float * slg, const uint32_t * id, const tqllm::SamplingParams & samp, const std::vector<uint32_t> & recent) {
    OldDist d;
    int ord[16];
    for (int i = 0; i < 16; ++i) ord[i] = i;
    float lg[16];
    std::copy_n(slg, 16, lg);
    if (samp.presence > 0.f) {
        const size_t r0 = recent.size() > size_t(64) ? recent.size() - 64 : 0;
        for (int i = 0; i < 16; ++i)
            if (std::find(recent.begin() + long(r0), recent.end(), id[i]) != recent.end()) lg[i] -= samp.presence;
    }
    std::sort(ord, ord + 16, [&](int a, int b) { return lg[a] > lg[b]; });
    const int k = samp.top_k > 0 ? std::min(samp.top_k, 16) : 16;
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        d.id[i] = id[ord[i]];
        d.p[i] = std::exp(double(lg[ord[i]] - lg[ord[0]]) / samp.temperature);
        sum += d.p[i];
    }
    double cum = 0;
    d.n = k;
    for (int i = 0; i < k; ++i) {
        cum += d.p[i] / sum;
        if (cum >= samp.top_p) { d.n = i + 1; break; }
    }
    double s2 = 0;
    for (int i = 0; i < d.n; ++i) s2 += d.p[i];
    for (int i = 0; i < d.n; ++i) d.p[i] /= s2;
    return d;
}
uint32_t old_draw(const OldDist & d, std::mt19937_64 & rng, uint32_t exclude = UINT32_MAX) {
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

// Exact sampling: a noisy first pass a = l + e (|e_r| <= b_r) screened like HeadSampler (top-16 by a, then the rows with the
// largest a + b), against sample_reference on the exact logits. sample_bounded must either abstain or give the reference's
// token; with every row known it must always decide.
int exact_sampling_cases(int & cases) {
    int fails = 0;
    std::mt19937 gen(11);
    const int V = 3000;
    const float temps[] = {0.3f, 0.7f, 1.f}, top_ps[] = {1.f, 0.95f, 0.8f, 0.3f};
    const int top_ks[] = {0, 1, 7, 40};
    int decided_first = 0, total = 0;
    for (float t : temps) for (float tp : top_ps) for (int tk : top_ks) for (int pen = 0; pen < 2; ++pen) for (int rep = 0; rep < 12; ++rep) {
        tqllm::SamplingParams s{t, tp, pen ? 0.7f : 0.f, tk};
        if (pen && rep % 2) s.repeat = 1.3f;
        std::normal_distribution<float> nd(0.f, rep % 3 == 0 ? 1.f : 4.f);  // flat and peaked positions
        std::vector<float> l(V), a(V), b(V);
        for (int r = 0; r < V; ++r) {
            l[size_t(r)] = nd(gen);
            b[size_t(r)] = 0.05f + 0.3f * float(gen() % 1000) / 1000.f;
            a[size_t(r)] = l[size_t(r)] + b[size_t(r)] * (2.f * float(gen() % 1001) / 1000.f - 1.f);
        }
        std::vector<uint32_t> recent;
        for (int i = 0; i < 70; ++i) recent.push_back(uint32_t(gen() % 60));  // ids near the top after the boost below
        for (int r = 0; r < 60; ++r) l[size_t(r)] += 3.f, a[size_t(r)] += 3.f;
        const auto pc = tqllm::penalty_counts(s, recent);
        for (int step = 0; step < 3; ++step) {
            const uint64_t key = tqllm::gumbel_key(gen(), uint32_t(step));
            const uint32_t want = tqllm::sample_reference(l.data(), V, s, recent, key);
            const bool full = tqllm::full_vocab_draw(s, V);
            // screened rounds: K rows by U = a + b (round 1: top-16 by a), plus every penalised id
            bool done = false;
            for (int K : {16, 64, 1024, V}) {
                std::vector<uint32_t> ord(V);
                std::iota(ord.begin(), ord.end(), 0u);
                std::vector<double> key_v(V);
                for (int r = 0; r < V; ++r) key_v[size_t(r)] = K == 16 ? a[size_t(r)] : a[size_t(r)] + b[size_t(r)];
                if (full)  // Gumbel screen: rows whose upper score reaches the best lower score
                    for (int r = 0; r < V; ++r) key_v[size_t(r)] = a[size_t(r)] + b[size_t(r)] + t * tqllm::gumbel(key, uint32_t(r));
                std::sort(ord.begin(), ord.end(), [&](uint32_t x, uint32_t y) { return key_v[x] > key_v[y]; });
                std::vector<char> in(V, 0);
                tqllm::ExactRows e;
                e.rows = V;
                int take = std::min(K, V);
                if (full) {
                    double maxl = -1e30;
                    for (int r = 0; r < V; ++r)
                        if (!tqllm::penalty_count(pc, uint32_t(r)))
                            maxl = std::max(maxl, a[size_t(r)] - b[size_t(r)] + t * tqllm::gumbel(key, uint32_t(r)));
                    take = 0;
                    while (take < V && key_v[ord[size_t(take)]] >= maxl) ++take;
                    e.gumbel_closed = true;
                }
                for (int i = 0; i < take; ++i) in[ord[size_t(i)]] = 1;
                for (const auto & p : pc) in[p.first] = 1;
                float ref = -1e30f;
                for (int r = 0; r < V; ++r) if (in[size_t(r)]) e.id.push_back(uint32_t(r)), e.lg.push_back(l[size_t(r)]), ref = std::max(ref, l[size_t(r)]);
                e.z_ref = ref;
                for (int r = 0; r < V; ++r) {
                    if (in[size_t(r)]) continue;
                    e.m_out = std::max(e.m_out, a[size_t(r)] + b[size_t(r)]);
                    e.z_hi += std::exp((double(a[size_t(r)] + b[size_t(r)]) - ref) / t);
                    e.z_lo += std::exp((double(a[size_t(r)] - b[size_t(r)]) - ref) / t);
                }
                uint32_t tok = 0;
                if (tqllm::sample_bounded(e, s, pc, key, tok)) {
                    ++cases;
                    if (tok != want) { ++fails; std::printf("EXACT FAIL: T %.2f top_p %.2f top_k %d pen %d K %d: %u vs %u\n", t, tp, tk, pen, K, tok, want); }
                    decided_first += K == 16;
                    done = true;
                    break;
                }
                if (K == V) break;
            }
            ++total;
            if (!done) { ++fails; std::printf("EXACT FAIL: undecided with every row known (T %.2f top_p %.2f top_k %d)\n", t, tp, tk); }
        }
    }
    std::printf("exact sampling: %d positions, %d decided by the first round\n", total, decided_first);
    return fails;
}
}  // namespace

int main() {
    std::mt19937 gen(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    int fails = 0, cases = 0;
    const float temps[] = {0.3f, 0.7f, 1.f, 1.5f}, top_ps[] = {1.f, 0.95f, 0.5f, 0.05f}, presences[] = {0.f, 0.5f, 2.f};
    const int top_ks[] = {0, 1, 5, 16, 40};
    for (float t : temps) for (float tp : top_ps) for (float pr : presences) for (int tk : top_ks) for (int rep = 0; rep < 8; ++rep) {
        const tqllm::SamplingParams s{t, tp, pr, tk};
        const int n = int(gen() % 6), cols = n + 1;
        std::vector<float> lg(size_t(cols) * 16); std::vector<uint32_t> id(size_t(cols) * 16), draft(static_cast<size_t>(n)), recent0(static_cast<size_t>(gen() % 90));
        for (auto & v : lg) v = nd(gen);
        for (int c = 0; c < cols; ++c) for (int i = 0; i < 16; ++i) id[size_t(c) * 16 + i] = uint32_t(c * 100 + (i * 7 + rep) % 16);
        for (int j = 0; j < n; ++j) draft[size_t(j)] = id[size_t(j) * 16 + gen() % 16];  // a candidate of its column
        for (auto & r : recent0) r = uint32_t(gen() % 300);  // ids overlap the candidates (c * 100 + 0..15)
        // distributions
        for (int c = 0; c < cols; ++c) {
            const auto a = tqllm::top_dist(&lg[size_t(c) * 16], &id[size_t(c) * 16], s, recent0);
            const auto b = old_dist(&lg[size_t(c) * 16], &id[size_t(c) * 16], s, recent0);
            ++cases;
            if (a.n != b.n || std::memcmp(a.id, b.id, sizeof(uint32_t) * size_t(a.n)) || std::memcmp(a.p, b.p, sizeof(double) * size_t(a.n))) ++fails;  // entries past n are unset
        }
        // acceptance: new helper vs the old loop, same seed
        const uint64_t seed = gen();
        std::mt19937_64 r1(seed), r2(seed);
        std::vector<uint32_t> rec1 = recent0, rec2 = recent0, res1(static_cast<size_t>(cols)), res2(static_cast<size_t>(cols));
        const int j1 = tqllm::accept_drafts(draft.data(), n, [&](int i, const std::vector<uint32_t> & r) { return tqllm::top_dist(&lg[size_t(i) * 16], &id[size_t(i) * 16], s, r); },
                                            r1, res1.data(), rec1, pr > 0.f);
        int j2 = 0;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (; j2 < n; ++j2) {
            const OldDist d = old_dist(&lg[size_t(j2) * 16], &id[size_t(j2) * 16], s, rec2);
            double q = 0;
            for (int i = 0; i < d.n; ++i) if (d.id[i] == draft[size_t(j2)]) q = d.p[i];
            if (uni(r2) >= q) { res2[size_t(j2)] = old_draw(d, r2, draft[size_t(j2)]); break; }
            if (pr > 0.f) rec2.push_back(draft[size_t(j2)]);
        }
        if (j2 == n) res2[size_t(j2)] = old_draw(old_dist(&lg[size_t(j2) * 16], &id[size_t(j2) * 16], s, rec2), r2);
        ++cases;
        if (j1 != j2 || res1[size_t(j1)] != res2[size_t(j2)] || rec1 != rec2 || r1() != r2()) ++fails;
    }
    // repeat / frequency penalties (LM Studio / llama.cpp semantics) on a hand-made position: token 1 (logit +4) and token 2
    // (logit -1) are in the window, token 1 twice; token 0 (+3) is not
    {
        float lg16[16]; uint32_t ids[16];
        for (int i = 0; i < 16; ++i) { lg16[i] = -8.f - float(i); ids[i] = uint32_t(i); }
        lg16[0] = 3.f; lg16[1] = 4.f; lg16[2] = -1.f;
        const std::vector<uint32_t> recent{1, 2, 1, 9};
        auto logp = [&](const tqllm::SamplingParams & s, uint32_t id) {
            const auto d = tqllm::top_dist(lg16, ids, s, recent);
            for (int i = 0; i < d.n; ++i) if (d.id[i] == id) return std::log(d.p[i]);
            return -1e9;
        };
        auto expect = [&](bool ok, const char * what) { ++cases; if (!ok) { ++fails; std::printf("SAMPLING FAIL: %s\n", what); } };
        const tqllm::SamplingParams base{1.f, 1.f, 0.f, 0};
        tqllm::SamplingParams rep = base; rep.repeat = 2.f;  // token 1: 4 / 2 = 2 (below token 0's 3); token 2: -1 x 2 = -2
        tqllm::SamplingParams frq = base; frq.frequency = 1.f;  // token 1: 4 - 2 = 2; token 2: -1 - 1 = -2
        expect(logp(base, 1) > logp(base, 0), "no penalty: token 1 leads");
        expect(std::fabs((logp(rep, 0) - logp(rep, 1)) - 1.0) < 1e-6, "repeat 2: a positive logit is halved (3 vs 2)");
        expect(std::fabs((logp(rep, 0) - logp(rep, 2)) - 5.0) < 1e-6, "repeat 2: a negative logit is doubled (3 vs -2)");
        expect(std::fabs((logp(frq, 0) - logp(frq, 1)) - 1.0) < 1e-6, "frequency 1: minus the count (4 - 2)");
        expect(std::fabs((logp(frq, 0) - logp(frq, 2)) - 5.0) < 1e-6, "frequency 1: count 1 (-1 - 1)");
        tqllm::SamplingParams off = base; off.repeat = 1.f; off.frequency = 0.f;
        expect(!off.penalized() && rep.penalized() && frq.penalized(), "penalized() flags only active penalties");
    }
    fails += exact_sampling_cases(cases);
    std::printf("SAMPLING %s (%d cases, %d mismatches)\n", fails ? "FAIL" : "PASS", cases, fails);
    return fails ? 1 : 0;
}

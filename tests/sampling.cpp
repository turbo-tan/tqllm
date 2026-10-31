// src/spec/token_sampling.hpp against the engines' previous private copies (Qwen35Engine::dist / draw / accept, 10-31): the same
// distributions, the same tokens and the same RNG draws for random top-16 logits, every parameter combination.
#include "spec/token_sampling.hpp"
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
    std::printf("SAMPLING %s (%d cases, %d mismatches)\n", fails ? "FAIL" : "PASS", cases, fails);
    return fails ? 1 : 0;
}

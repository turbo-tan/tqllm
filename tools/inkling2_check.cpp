// inkling2-check MODEL GOLDEN_BIN [--chunk N] [--greedy-only | --tf-only]: the Inkling-2-Mini text model
// (src/models/inkling2/loop) against golden references from the transformers reference, converted to plain binaries:
//   tf_ids.i32 [T], tf_top_ids.i32 [T][K], tf_top_lp.f32 [T][K] (the reference's top-K log-probs per position)
//   greedy.txt: one prompt per line: name n_in ids... n_gen ids...
// Teacher-forced: mean KL(ref || ours) over the reference's top-K support and top-1 agreement over all T positions.
// Greedy: the first 64 generated tokens (or the reference's length if shorter) must match. Pass: KLD <= 0.01, top-1 >= 98 %,
// greedy on >= 4 of 5 prompts.
// --mt REF (GOLDEN_BIN = golden/mt): multi-text full-vocabulary gate; --onestep: per-visit check;
// --bench 1024,8192 [--decode N] [--profile decode|prefill|greedy] (GOLDEN_BIN = golden/mt): prefill / decode tok/s, graph == eager;
// --spec GOLDEN_BIN CTXLIST [--decode N] (GOLDEN_BIN arg 2 = golden/mt): MTP spec decode vs plain on the greedy prompts
// and bench-text contexts, spec == plain and tokens per step (--spec-trace: every adaptive step on stderr);
// --reuse CTXLIST (GOLDEN_BIN = golden/mt): exact conversation reuse vs a fresh prefill (first logits, tokens, TTFT);
// --exact off: the device fast dense format (INT8 W8A16 on Ampere), gated by KLD / PPL
// --expert-kernels auto|tensor|warp: the routed-expert family (default auto: the load's rule)
// --expert-budget-mb N: device memory for the routed experts (the rest stream from mapped host memory);
// --handover (golden/mt): prompt -> decode hand-over KLD; --greedy-debug NAME (golden/bin): one prompt's paths per step.
#include "inkling2_model.hpp"
#include "inkling2_spec_policy.hpp"

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace tqllm;

namespace {
int g_kernels = -1;  // --expert-kernels auto|tensor|warp
size_t g_budget = SIZE_MAX;  // --expert-budget-mb: device bytes for the routed experts (default: what fits)
bool g_exact = true;           // --exact off: the device fast dense format (INT8 W8A16 on Ampere)
bool g_spec_trace = false;    // --spec-trace: every adaptive spec / plain step (drafts, accepted, ms) on stderr
std::string g_drafter;          // --drafter DIR
template <class T> std::vector<T> load(const std::string & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p);
    const std::string b((std::istreambuf_iterator<char>(in)), {});
    std::vector<T> v(b.size() / sizeof(T));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(T));
    return v;
}
double secs(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

// Multi-text golden (golden/mt): texts.bin (count, then n + ids per text), lp_<ref>.f16 full log-softmax rows [sum T][V],
// top1_<ref>.i32, route_<ref>.i16 [text][48][T][8]. Full-vocabulary KL(ref || ours) per position (the reference row
// renormalised from its F16 values), top-1 agreement, routing agreement per visit (full forward, both sides compounding).
struct Texts { std::vector<std::vector<int>> ids; std::vector<std::string> names; };
Texts read_texts(const std::string & dir) {
    const auto v = load<int32_t>(dir + "/texts.bin");
    Texts t; size_t o = 1;
    for (int i = 0; i < v[0]; ++i) { const int n = v[o++]; t.ids.emplace_back(v.begin() + o, v.begin() + o + n); o += n; }
    return t;
}
double route_agree(const int * ours, const int16_t * ref, int n, int k) {
    double a = 0;
    for (int i = 0; i < n; ++i)
        for (int x = 0; x < k; ++x)
            for (int y = 0; y < k; ++y) if (ours[i * k + x] == ref[i * k + y]) { a += 1; break; }
    return a / (double(n) * k);
}
const char * visit_type(int v) {
    static const char * names[48];
    static char buf[48][32];
    if (!names[0])
        for (int i = 0; i < 48; ++i) {
            int l = i < 6 ? i : i < 42 ? 6 + (i - 6) % 12 : 18 + (i - 42);
            const int pass = i >= 6 && i < 42 ? (i - 6) / 12 + 1 : 0;
            const bool full = (l + 1) % 6 == 0;
            std::snprintf(buf[i], sizeof buf[i], "%s %s", i < 6 ? "prelude" : i < 42 ? (pass == 1 ? "pass1" : pass == 2 ? "pass2" : "pass3") : "coda",
                          full ? "full" : "sliding");
            names[i] = buf[i];
        }
    return names[v];
}
void print_route_table(const std::vector<double> & sum, const std::vector<double> & cnt, const char * what) {
    std::map<std::string, std::pair<double, double>> by;
    for (int v = 0; v < int(sum.size()); ++v) if (cnt[v] > 0) { auto & b = by[visit_type(v)]; b.first += sum[v]; b.second += cnt[v]; }
    std::printf("routing agreement (%s) per visit type:\n", what);
    for (const auto & [k, b] : by) std::printf("  %-16s %.2f %%\n", k.c_str(), 100.0 * b.first / b.second);
    std::printf("  per visit:");
    for (int v = 0; v < int(sum.size()); ++v) std::printf(" %.1f", cnt[v] > 0 ? 100.0 * sum[v] / cnt[v] : -1.0);
    std::printf("\n");
}
int run_mt(const std::string & model, const std::string & dir, const std::string & ref, int chunk, bool hf) {
    const Texts tx = read_texts(dir);
    size_t total = 0; int maxn = 0;
    for (const auto & v : tx.ids) { total += v.size(); maxn = std::max(maxn, int(v.size())); }
    Inkling2Model m(model, maxn + 64, chunk, hf, g_budget, g_exact, 0, g_kernels);
    const int V = m.vocab();
    std::ifstream lpf(dir + "/lp_" + ref + ".f16", std::ios::binary);
    const auto top1 = load<int32_t>(dir + "/top1_" + ref + ".i32");
    const auto routes = load<int16_t>(dir + "/route_" + ref + ".i16");
    if (top1.size() != total) throw std::runtime_error("mt: top1 size");
    std::vector<__half> row(static_cast<size_t>(V));
    std::vector<float> lg;
    std::vector<std::vector<int>> log;
    std::vector<double> rsum(48, 0), rcnt(48, 0);
    double kl_all = 0, nll = 0; long agree_all = 0, n_all = 0, nll_n = 0; size_t roff = 0, toff = 0;
    for (size_t ti = 0; ti < tx.ids.size(); ++ti) {
        const auto & ids = tx.ids[ti];
        const int T = int(ids.size());
        m.reset();
        log.clear();
        m.log_routes(&log);
        double kl = 0; long agree = 0;
        for (int p = 0; p < T; p += chunk) {
            const int n = std::min(chunk, T - p);
            m.forward(ids.data() + p, n, n, lg);
            for (int i = 0; i < n; ++i) {
                lpf.read(reinterpret_cast<char *>(row.data()), std::streamsize(row.size() * 2));
                const float * x = lg.data() + size_t(i) * V;
                double mq = -1e30, mr = -1e30; int am = 0;
                for (int v = 0; v < V; ++v) { if (x[v] > mq) { mq = x[v]; am = v; } mr = std::max(mr, double(__half2float(row[v]))); }
                double zq = 0, zr = 0;
                for (int v = 0; v < V; ++v) { zq += std::exp(double(x[v]) - mq); zr += std::exp(double(__half2float(row[v])) - mr); }
                const double lzq = mq + std::log(zq), lzr = mr + std::log(zr);
                double k = 0;
                for (int v = 0; v < V; ++v) {
                    const double lr = double(__half2float(row[v])) - lzr;
                    if (lr > -80) k += std::exp(lr) * (lr - (double(x[v]) - lzq));
                }
                kl += k;
                agree += am == top1[toff + p + i];
                if (p + i + 1 < T) { nll += lzq - double(x[ids[p + i + 1]]); ++nll_n; }
            }
        }
        m.log_routes(nullptr);
        // log: chunk-major, 48 visits per chunk; reference: [48][T][8]
        const int nch = (T + chunk - 1) / chunk;
        if (int(log.size()) != 48 * nch) throw std::runtime_error("mt: route log size");
        for (int c = 0; c < nch; ++c) {
            const int p = c * chunk, n = std::min(chunk, T - p);
            for (int v = 0; v < 48; ++v) {
                rsum[v] += route_agree(log[size_t(c) * 48 + v].data(), routes.data() + roff + (size_t(v) * T + p) * 8, n, 8) * n;
                rcnt[v] += n;
            }
        }
        roff += size_t(48) * T * 8; toff += T;
        kl_all += kl; agree_all += agree; n_all += T;
        std::printf("text %2zu (%4d tokens): KLD %.5f top-1 %.2f %%\n", ti, T, kl / T, 100.0 * agree / T);
    }
    std::printf("ALL vs %s: %ld positions, full-vocabulary KLD %.5f, top-1 %.2f %%, PPL of the texts %.4f (%ld next tokens)\n", ref.c_str(), n_all,
                kl_all / n_all, 100.0 * agree_all / n_all, std::exp(nll / nll_n), nll_n);
    print_route_table(rsum, rcnt, ("ours vs " + ref + ", full forward").c_str());
    return 0;
}
// One step per visit on the reference's own hidden states (hs_<name>.f32, [49][512][D]) of the 4 texts that have them;
// routing agreement against the reference's routes of those tokens.
int run_onestep(const std::string & model, const std::string & dir, bool hf) {
    const Texts tx = read_texts(dir);
    const char * names[] = {"wiki00", "wiki01", "code_textwrap", "code_heapq"};
    const int idx[] = {0, 1, 12, 14};
    const int N = 512;
    Inkling2Model m(model, N + 64, N, hf, g_budget, g_exact, 0, g_kernels);
    const auto routes = load<int16_t>(dir + "/route_bf16.i16");
    std::vector<size_t> roff(tx.ids.size() + 1, 0);
    for (size_t i = 0; i < tx.ids.size(); ++i) roff[i + 1] = roff[i] + size_t(48) * tx.ids[i].size() * 8;
    std::vector<double> rsum(48, 0), rcnt(48, 0);
    std::vector<std::vector<int>> log;
    for (int k = 0; k < 4; ++k) {
        const auto hs = load<float>(dir + "/hs_" + names[k] + ".f32");
        const int T = int(tx.ids[idx[k]].size()), n = std::min(N, T);
        log.clear();
        m.log_routes(&log);
        std::printf("one-step %s:\n", names[k]);
        m.debug_visits(hs, n);
        m.log_routes(nullptr);
        for (int v = 0; v < int(log.size()) && v < 48; ++v) {
            rsum[v] += route_agree(log[v].data(), routes.data() + roff[idx[k]] + size_t(v) * T * 8, n, 8) * n;
            rcnt[v] += n;
        }
    }
    print_route_table(rsum, rcnt, "ours one-step vs bf16 reference");
    return 0;
}
// Speed: for each context length, prefill that many tokens of the multi-text golden (repeated) in chunks, then greedy
// decode `ndec` tokens; prefill and decode tok/s. The first length runs twice, the second time without the decode graph:
// the decoded tokens and every step's logits must be bitwise equal (graph == eager, and plain decode is deterministic).
// --reuse CTXLIST (GOLDEN_BIN = golden/mt): exact conversation reuse, as the engine does it. Turn 1: prompt A (ctx tokens
// of the bench text) prefilled in chunks aligned from 0, the state at the start of its final chunk kept (snapshot), a
// greedy reply (32 tokens, and once 2 chunks + 100 tokens). Turn 2: B = A + reply + 64 new tokens, (1) restored to the
// snapshot and prefilled from there, (2) fresh from 0 (1.7.0): first-token logits bitwise, 32 greedy tokens, TTFT.
int run_reuse(const std::string & model, const std::string & dir, const std::string & list, int chunk) {
    std::vector<int> ctxs;
    for (std::stringstream ss(list); ss.good();) { std::string x; std::getline(ss, x, ','); if (!x.empty()) ctxs.push_back(std::atoi(x.c_str())); }
    const Texts tx = read_texts(dir);
    std::vector<int> text;
    for (const auto & v : tx.ids) text.insert(text.end(), v.begin(), v.end());
    const int maxc = *std::max_element(ctxs.begin(), ctxs.end()), longr = 2 * chunk + 100;
    std::vector<int> fill;
    while (int(fill.size()) < maxc + longr + 400) fill.insert(fill.end(), text.begin(), text.end());
    Inkling2Model m(model, maxc + longr + 512, chunk, false, g_budget, g_exact, 0, g_kernels);
    std::vector<float> lg;
    const int U = 64, G = 32;
    int keep = 0;
    auto prefill = [&](const std::vector<int> & p, size_t from) {  // aligned chunks; snapshot at the final chunk's start
        for (size_t i = from; i < p.size(); i += size_t(chunk)) {
            const int n = int(std::min<size_t>(size_t(chunk), p.size() - i));
            if (i + size_t(chunk) >= p.size() && i > 0) { m.snapshot(); keep = int(i); }
            m.forward(p.data() + i, n, 1, lg);
        }
        return int(std::max_element(lg.begin(), lg.end()) - lg.begin());
    };
    auto gen = [&](int first, int n) {
        std::vector<int> out{first};
        for (int i = 1; i < n; ++i) out.push_back(m.forward_greedy(out.back()));
        return out;
    };
    bool all = true;
    for (const int ctx : ctxs)
        for (const int R : {32, longr}) {
            if (R == longr && ctx != ctxs.front()) continue;
            std::vector<int> A(fill.begin(), fill.begin() + ctx);
            m.reset(); keep = 0;
            const auto reply = gen(prefill(A, 0), R);
            std::vector<int> B = A;
            B.insert(B.end(), reply.begin(), reply.end());
            B.insert(B.end(), fill.begin() + ctx + R, fill.begin() + ctx + R + U);
            const int k1 = keep;
            auto t0 = std::chrono::steady_clock::now();
            const bool restored = keep > 0 && m.restore();
            const int c1 = prefill(B, restored ? size_t(k1) : 0);
            const double ttft1 = secs(t0);
            const auto lg1 = lg;
            const auto o1 = gen(c1, G);
            m.reset();
            t0 = std::chrono::steady_clock::now();
            const int c2 = prefill(B, 0);
            const double ttft2 = secs(t0);
            const bool lg_eq = lg1.size() == lg.size() && std::memcmp(lg1.data(), lg.data(), lg.size() * 4) == 0;
            const auto o2 = gen(c2, G);
            const bool ok = restored && lg_eq && o1 == o2;
            all = all && ok;
            std::printf("reuse ctx %6d (+%d reply, +%d new): turn-2 TTFT reuse %.3f s (from %d), fresh (1.7.0) %.3f s; first logits %s, tokens %s\n", ctx, R,
                        U, ttft1, k1, ttft2, lg_eq ? "bitwise equal" : "DIFFER", o1 == o2 ? "== fresh" : "DIFFER");
        }
    return all ? 0 : 1;
}

int run_bench(const std::string & model, const std::string & dir, const std::string & list, int chunk, int ndec, const std::string & profile) {
    std::vector<int> ctxs;
    for (std::stringstream ss(list); ss.good();) { std::string x; std::getline(ss, x, ','); if (!x.empty()) ctxs.push_back(std::atoi(x.c_str())); }
    const Texts tx = read_texts(dir);
    std::vector<int> text;
    for (const auto & v : tx.ids) text.insert(text.end(), v.begin(), v.end());
    const int maxc = *std::max_element(ctxs.begin(), ctxs.end());
    std::vector<int> fill;
    while (int(fill.size()) < maxc) fill.insert(fill.end(), text.begin(), text.end());
    Inkling2Model m(model, maxc + ndec + 8, chunk, false, g_budget, g_exact, 0, g_kernels);
    std::vector<float> lg;
    // --profile decode | prefill: cudaProfilerStart / Stop around the first run's decode steps or prompt chunks (nsys
    // --capture-range=cudaProfilerApi)
    bool prof = profile == "decode", prof_pf = profile == "prefill";
    auto run = [&](int ctx, std::vector<int> & got, uint64_t & hash, double & tp, double & td) {
        m.reset();
        if (prof_pf) cudaProfilerStart();
        const auto t0 = std::chrono::steady_clock::now();
        for (int p = 0; p < ctx; p += chunk) { const int n = std::min(chunk, ctx - p); m.forward(fill.data() + p, n, 1, lg); }
        tp = secs(t0);
        if (prof_pf) { cudaProfilerStop(); prof_pf = false; }
        if (const int moved = m.replan_experts())  // partial residency: the prompt's most-routed experts onto the device
            std::printf("bench: %d experts per layer resident, %d moved after the prompt\n", m.resident_experts(), moved);
        got.clear(); hash = 1469598103934665603ull;
        if (prof) cudaProfilerStart();  // nsys --capture-range=cudaProfilerApi: the decode steps only
        const auto t1 = std::chrono::steady_clock::now();
        double tf = 0;  // inside forward() (graph launch, GPU, logits copy); the rest is the host argmax and hash
        for (int i = 0; i < ndec; ++i) {
            const int am = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
            got.push_back(am);
            const auto t2 = std::chrono::steady_clock::now();
            m.forward(&am, 1, 1, lg);
            tf += secs(t2);
            for (size_t j = 0; j < lg.size(); j += 97) { uint32_t u; std::memcpy(&u, &lg[j], 4); hash = (hash ^ u) * 1099511628211ull; }
        }
        td = secs(t1);
        std::printf("bench ctx %6d decode split: %.3f ms / token in forward(), %.3f ms host argmax + hash\n", ctx, tf / ndec * 1e3, (td - tf) / ndec * 1e3);
        std::fprintf(stderr, "bench ids:");  // diagnostic: the generated ids, so a serving run can be compared exactly
        for (const int x : got) std::fprintf(stderr, " %d", x);
        std::fprintf(stderr, "\n");
        if (prof) { cudaProfilerStop(); prof = false; }
    };
    for (size_t ci = 0; ci < ctxs.size(); ++ci) {
        const int ctx = ctxs[ci];
        std::vector<int> a; uint64_t ha; double tp, td;
        run(ctx, a, ha, tp, td);
        std::printf("bench ctx %6d: prefill %.1f tok/s (chunk %d, %.1f s), decode %.2f tok/s (%d tokens), logits hash %016llx\n", ctx, ctx / tp, chunk,
                    tp, ndec / td, ndec, (unsigned long long)ha);
        if (ci == 0) {  // run 2 without the decode graph (a route log forces eager launches): must equal run 1 bitwise
            std::vector<int> b; uint64_t hb; double tp2, td2;
            std::vector<std::vector<int>> routes;
            m.log_routes(&routes);
            run(ctx, b, hb, tp2, td2);
            m.log_routes(nullptr);
            {   // distinct routed experts per visit over windows of W consecutive decode tokens (what a W-row verify reads)
                std::vector<const std::vector<int> *> one;
                for (const auto & r : routes) if (r.size() == 8) one.push_back(&r);
                const int visits = 48, steps = int(one.size()) / visits;
                for (int W : {1, 3, 5, 8}) {
                    double sum = 0; int cnt = 0;
                    for (int s0 = 0; s0 + W <= steps; s0 += W)
                        for (int v = 0; v < visits; ++v) {
                            std::vector<int> e;
                            for (int k = 0; k < W; ++k) e.insert(e.end(), one[size_t((s0 + k) * visits + v)]->begin(), one[size_t((s0 + k) * visits + v)]->end());
                            std::sort(e.begin(), e.end());
                            sum += double(std::unique(e.begin(), e.end()) - e.begin()); ++cnt;
                        }
                    std::printf("bench routes: %d-token windows, %.1f distinct experts per visit (of %d assignments)\n", W, cnt ? sum / cnt : 0.0, 8 * W);
                }
            }
            std::printf("bench ctx %6d run 2 (eager, route log): decode %.2f tok/s; graph == eager: tokens %s, logits %s\n", ctx, ndec / td2,
                        a == b ? "equal" : "DIFFER", ha == hb ? "equal" : "DIFFER");
            if (a != b || ha != hb) return 1;
            {   // run 3: greedy decode without logits (forward_greedy): the same tokens
                m.reset();
                for (int p = 0; p < ctx; p += chunk) { const int n = std::min(chunk, ctx - p); m.forward(fill.data() + p, n, 1, lg); }
                std::vector<int> g;
                int tok = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
                if (profile == "greedy") cudaProfilerStart();  // nsys --capture-range=cudaProfilerApi: run 3's greedy steps
                const auto t3 = std::chrono::steady_clock::now();
                for (int i = 0; i < ndec; ++i) { g.push_back(tok); tok = m.forward_greedy(tok); }
                const double td3 = secs(t3);
                if (profile == "greedy") cudaProfilerStop();
                std::printf("bench ctx %6d run 3 (greedy, no logits): decode %.2f tok/s; tokens %s\n", ctx, ndec / td3, g == a ? "equal" : "DIFFER");
                if (g != a) return 1;
            }
        }
    }
    return 0;
}

// One greedy prompt (by name) along the reference's own tokens: per step the top-2 of (P) the prompt prefilled then
// single steps (the graph), (E) the same without the graph, (S) every token a single step, (F) the prefix prefilled.
int run_greedy_debug(const std::string & model, const std::string & dir, const std::string & want_name, int chunk, int steps) {
    std::ifstream g(dir + "/greedy.txt");
    std::string line;
    while (std::getline(g, line)) {
        std::istringstream in(line);
        std::string name; int nin, ngen;
        in >> name >> nin;
        std::vector<int> pin(nin);
        for (auto & x : pin) in >> x;
        in >> ngen;
        std::vector<int> ref(ngen);
        for (auto & x : ref) in >> x;
        if (name != want_name) continue;
        Inkling2Model m(model, nin + steps + 64, chunk, false, g_budget, g_exact, 0, g_kernels);
        std::vector<int> seq(pin);
        seq.insert(seq.end(), ref.begin(), ref.begin() + steps);
        std::vector<std::vector<float>> P(steps), E(steps), S(steps), F(steps);
        std::vector<float> lg;
        std::vector<std::vector<int>> routes;
        for (int mode = 0; mode < 2; ++mode) {
            m.reset();
            if (mode == 1) m.log_routes(&routes);
            for (int p = 0; p < nin; p += chunk) { const int n = std::min(chunk, nin - p); m.forward(pin.data() + p, n, 1, lg); }
            for (int i = 0; i < steps; ++i) {
                (mode ? E : P)[i] = lg;
                m.forward(&ref[i], 1, 1, lg);
            }
            m.log_routes(nullptr);
        }
        m.reset();
        for (int p = 0; p < nin + steps; ++p) { m.forward(&seq[p], 1, 1, lg); if (p >= nin - 1 && p - nin + 1 < steps) S[p - nin + 1] = lg; }
        for (int i = 0; i < steps; ++i) {
            m.reset();
            const int L = nin + i;
            for (int p = 0; p < L; p += chunk) { const int n = std::min(chunk, L - p); m.forward(seq.data() + p, n, 1, lg); }
            F[i] = lg;
        }
        auto top2 = [&](const std::vector<float> & x) {
            int a = int(std::max_element(x.begin(), x.end()) - x.begin());
            int b = -1;
            for (int v = 0; v < int(x.size()); ++v) if (v != a && (b < 0 || x[v] > x[b])) b = v;
            char buf[96];
            std::snprintf(buf, sizeof buf, "%6d %.3f / %6d %.3f", a, x[a], b, x[b]);
            return std::string(buf);
        };
        for (int i = 0; i < steps; ++i)
            std::printf("step %2d ref %6d | P %s | E %s | S %s | F %s | ref-token logit P %.3f S %.3f F %.3f\n", i, ref[i], top2(P[i]).c_str(), top2(E[i]).c_str(),
                        top2(S[i]).c_str(), top2(F[i]).c_str(), P[i][ref[i]], S[i][ref[i]], F[i][ref[i]]);
        return 0;
    }
    std::fprintf(stderr, "no prompt %s\n", want_name.c_str());
    return 1;
}

// Prompt -> decode hand-over: text 0 of the multi-text golden, split at s: (A) prefill s tokens in chunks, then 8 single
// steps; (B) every token a single step; (C) the s + 8 tokens prefilled. Full-vocabulary KLD between the 8 positions' logits
// of each pair; all three must agree to the noise of the two kernel families (KLD ~1e-4), a broken hand-over shows up
// as A far from both B and C.
int run_handover(const std::string & model, const std::string & dir, int chunk) {
    const Texts tx = read_texts(dir);
    const auto & ids = tx.ids[0];
    Inkling2Model m(model, int(ids.size()) + 64, chunk, false, g_budget, g_exact, 0, g_kernels);
    const int V = m.vocab(), K = 8;
    auto kld = [&](const float * p, const float * q) {
        double mp = -1e30, mq = -1e30;
        for (int v = 0; v < V; ++v) { mp = std::max(mp, double(p[v])); mq = std::max(mq, double(q[v])); }
        double zp = 0, zq = 0;
        for (int v = 0; v < V; ++v) { zp += std::exp(p[v] - mp); zq += std::exp(q[v] - mq); }
        double k = 0;
        for (int v = 0; v < V; ++v) { const double lp = p[v] - mp - std::log(zp), lq = q[v] - mq - std::log(zq); k += std::exp(lp) * (lp - lq); }
        return k;
    };
    int bad = 0;
    for (int s : {37, 100, 300}) {
        std::vector<float> A, B, C, lg;
        m.reset();
        for (int p = 0; p < s; p += chunk) { const int n = std::min(chunk, s - p); m.forward(ids.data() + p, n, 1, lg); }
        for (int i = 0; i < K; ++i) { m.forward(ids.data() + s + i, 1, 1, lg); A.insert(A.end(), lg.begin(), lg.end()); }
        m.reset();
        for (int p = 0; p < s + K; ++p) { m.forward(ids.data() + p, 1, 1, lg); if (p >= s) B.insert(B.end(), lg.begin(), lg.end()); }
        m.reset();
        for (int p = 0; p < s; p += chunk) { const int n = std::min(chunk, s - p); m.forward(ids.data() + p, n, 1, lg); }
        m.forward(ids.data() + s, K, K, C);
        double ab = 0, ac = 0, bc = 0;
        for (int i = 0; i < K; ++i) {
            const size_t o = size_t(i) * V;
            ab += kld(A.data() + o, B.data() + o); ac += kld(A.data() + o, C.data() + o); bc += kld(B.data() + o, C.data() + o);
        }
        ab /= K; ac /= K; bc /= K;
        const bool ok = ab < 0.01 && ac < 0.01;
        bad += !ok;
        std::printf("hand-over at %3d: KLD prefill+steps vs steps %.2e, vs prefill %.2e; steps vs prefill %.2e %s\n", s, ab, ac, bc, ok ? "ok" : "FAIL");
    }
    return bad ? 1 : 0;
}

// Speculative decoding with the MTP drafter (block of 8: anchor + 7 greedy drafts, verified in one target block):
// for each context length (the bench text) and each greedy prompt, N tokens plain and N tokens spec; the two token
// streams must be identical (spec == plain); tokens per verify step and tok/s of both.
struct GenStats { std::vector<int> tok; int steps = 0; double secs = 0; };
int argmax_row(const std::vector<float> & lg, int row, int V) {
    return int(std::max_element(lg.begin() + size_t(row) * V, lg.begin() + size_t(row + 1) * V) - (lg.begin() + size_t(row) * V));
}
GenStats gen_plain(Inkling2Model & m, const std::vector<int> & prompt, int chunk, int N) {
    std::vector<float> lg;
    m.reset();
    for (int p = 0; p < int(prompt.size()); p += chunk) { const int n = std::min(chunk, int(prompt.size()) - p); m.forward(prompt.data() + p, n, 1, lg); }
    GenStats g;
    g.tok.push_back(argmax_row(lg, 0, m.vocab()));
    const auto t0 = std::chrono::steady_clock::now();
    while (int(g.tok.size()) < N) { g.tok.push_back(m.forward_greedy(g.tok.back())); ++g.steps; }  // greedy without logits (bounded head)
    g.secs = secs(t0);
    return g;
}
// adaptive: verify L drafts with L = round(EMA of the tokens a step gained) + 1 (2 .. 7): an MoE verify reads every
// distinct expert of the block, so drafts that will not be accepted cost expert bandwidth; else all 7. (Measured against
// an expected-tokens-per-time policy over per-depth acceptance and per-length step times, with and without exploration:
// the EMA rule was as good or better on these prompts.)
GenStats gen_spec(Inkling2Model & m, const std::vector<int> & prompt, int chunk, int N, bool adaptive, bool prof = false) {
    std::vector<float> lg;
    m.reset();
    for (int p = 0; p < int(prompt.size()); p += chunk) { const int n = std::min(chunk, int(prompt.size()) - p); m.forward(prompt.data() + p, n, 1, lg); }
    GenStats g;
    const int V = m.vocab();
    g.tok.push_back(argmax_row(lg, 0, V));
    std::vector<int> d, blk(8);
    if (prof) cudaProfilerStart();
    const auto t0 = std::chrono::steady_clock::now();
    tqllm::Inkling2SpecPolicy pol(adaptive ? 0 : 7);  // the engine's choice of spec / plain steps (adaptive), or always 7 drafts
    while (int(g.tok.size()) < N) {
        const int anchor = g.tok.back();
        const int L = pol.next(1 << 20);
        const auto ts = std::chrono::steady_clock::now();
        if (L == 0) {
            g.tok.push_back(m.forward_greedy(anchor)); pol.plain_done(secs(ts)); ++g.steps;
            if (g_spec_trace && adaptive) std::fprintf(stderr, "  step %3d plain %.2f ms\n", g.steps, secs(ts) * 1e3);
            continue;
        }
        m.draft(anchor, d, L);
        const double tdraft = secs(ts);
        blk[0] = anchor;
        for (int i = 0; i < L; ++i) blk[size_t(i + 1)] = d[size_t(i)];
        int nx[8];
        m.verify_greedy(blk.data(), L + 1, nx);  // the verify rows' greedy tokens (bounded LM head), no logits
        int a = 0;
        while (a < L && nx[a] == d[size_t(a)]) ++a;
        m.commit(a + 1);
        pol.spec_done(a, L, secs(ts));
        if (g_spec_trace && adaptive) std::fprintf(stderr, "  step %3d spec L %d accepted %d %.2f ms (draft %.2f)\n", g.steps + 1, L, a, secs(ts) * 1e3, tdraft * 1e3);
        for (int i = 0; i < a; ++i) g.tok.push_back(d[size_t(i)]);
        g.tok.push_back(nx[a]);
        ++g.steps;
    }
    g.secs = secs(t0);
    if (prof) cudaProfilerStop();
    g.tok.resize(size_t(N));
    return g;
}
// one decode step after a prompt, launched as the CUDA graph and eagerly (a route log forces eager launches), each twice:
// all four logits rows must be equal bitwise
int run_graph_debug(const std::string & model, const std::string & mt, int ctx, int chunk) {
    const Texts tx = read_texts(mt);
    std::vector<int> text;
    for (const auto & v : tx.ids) text.insert(text.end(), v.begin(), v.end());
    std::vector<int> fill;
    while (int(fill.size()) < ctx + 8) fill.insert(fill.end(), text.begin(), text.end());
    Inkling2Model m(model, ctx + 64, chunk, false, g_budget, g_exact, 0, g_kernels);
    std::vector<float> lg;
    auto one = [&](bool eager, int steps) {
        std::vector<std::vector<int>> routes;
        m.reset();
        for (int p = 0; p < ctx; p += chunk) { const int n = std::min(chunk, ctx - p); m.forward(fill.data() + p, n, 1, lg); }
        if (eager) m.log_routes(&routes);
        std::vector<std::vector<float>> out{lg};  // [0]: the prompt's last logits
        for (int i = 0; i < steps; ++i) { m.forward(&fill[size_t(ctx + i)], 1, 1, lg); out.push_back(lg); }
        m.log_routes(nullptr);
        return out;
    };
    const auto g1 = one(false, 4), g2 = one(false, 4), e1 = one(true, 4), e2 = one(true, 4);
    for (int i = 0; i < 5; ++i) {
        auto diff = [&](const std::vector<float> & a, const std::vector<float> & b) {
            double mx = 0; int nd = 0;
            for (size_t k = 0; k < a.size(); ++k) if (a[k] != b[k]) { ++nd; mx = std::max(mx, double(std::fabs(a[k] - b[k]))); }
            char buf[64]; std::snprintf(buf, sizeof buf, "%d differ (max %.3g)", nd, mx); return std::string(buf);
        };
        std::printf("%s %d: graph vs graph %s, eager vs eager %s, graph vs eager %s\n", i ? "step" : "prompt", i ? i - 1 : 0, diff(g1[size_t(i)], g2[size_t(i)]).c_str(),
                    diff(e1[size_t(i)], e2[size_t(i)]).c_str(), diff(g1[size_t(i)], e1[size_t(i)]).c_str());
    }
    return 0;
}
int run_spec(const std::string & model, const std::string & mt, const std::string & gb, const std::string & list, int chunk, int N, bool prof) {
    std::vector<int> ctxs;
    for (std::stringstream ss(list); ss.good();) { std::string x; std::getline(ss, x, ','); if (!x.empty()) ctxs.push_back(std::atoi(x.c_str())); }
    const Texts tx = read_texts(mt);
    std::vector<int> text;
    for (const auto & v : tx.ids) text.insert(text.end(), v.begin(), v.end());
    const int maxc = ctxs.empty() ? 0 : *std::max_element(ctxs.begin(), ctxs.end());
    std::vector<int> fill;
    while (int(fill.size()) < maxc) fill.insert(fill.end(), text.begin(), text.end());
    std::vector<std::pair<std::string, std::vector<int>>> prompts;
    std::map<std::string, std::vector<int>> refs;  // the reference's greedy tokens
    {
        std::ifstream g(gb + "/greedy.txt");
        std::string line;
        while (std::getline(g, line)) {
            std::istringstream in(line);
            std::string name; int nin, ngen = 0;
            in >> name >> nin;
            std::vector<int> pin(nin);
            for (auto & x : pin) in >> x;
            if (in >> ngen) { std::vector<int> r(ngen); for (auto & x : r) in >> x; refs[name] = r; }
            prompts.emplace_back(name, pin);
        }
    }
    for (int c : ctxs) prompts.emplace_back("ctx" + std::to_string(c), std::vector<int>(fill.begin(), fill.begin() + c));
    Inkling2Model m(model, std::max(maxc, 64) + N + 64, chunk, false, g_budget, g_exact, Inkling2Model::drafter_bytes(g_drafter.empty() ? model : g_drafter), g_kernels);
    m.enable_mtp(g_drafter.empty() ? model : g_drafter);
    bool all = true;
    for (const auto & [name, pin] : prompts) {
        const auto p = gen_plain(m, pin, chunk, N);
        const auto q = gen_spec(m, pin, chunk, N, false, prof && name == prompts.front().first);
        const auto r = gen_spec(m, pin, chunk, N, true);
        const bool same = p.tok == q.tok && p.tok == r.tok;
        all = all && same;
        int first = -1;
        for (int i = 0; i < N && !same; ++i) if (p.tok[size_t(i)] != q.tok[size_t(i)] || p.tok[size_t(i)] != r.tok[size_t(i)]) { first = i; break; }
        std::printf("spec %-12s prompt %6zu: plain %.2f tok/s | block 8: %.2f tok/s, %.2f tokens / step | adaptive: %.2f tok/s, %.2f tokens / step | "
                    "spec == plain: %s", name.c_str(), pin.size(), (N - 1) / p.secs, (N - 1) / q.secs, double(N - 1) / q.steps, (N - 1) / r.secs,
                    double(N - 1) / r.steps, same ? "yes\n" : "NO, first difference at ");
        if (!same) std::printf("%d\n", first);
        uint64_t h = 1469598103934665603ull;
        for (const int t : p.tok) h = (h ^ uint64_t(uint32_t(t))) * 1099511628211ull;
        int agree = 0;
        if (refs.count(name)) { const auto & r = refs[name]; while (agree < int(r.size()) && agree < int(p.tok.size()) && r[size_t(agree)] == p.tok[size_t(agree)]) ++agree; }
        std::printf("  plain %-12s tokens hash %016llx, first %d of %d equal to the reference's greedy\n", name.c_str(), (unsigned long long)h, agree,
                    refs.count(name) ? int(std::min(refs[name].size(), p.tok.size())) : 0);
    }
    std::printf("spec: %s\n", all ? "PASS (every spec stream equals plain)" : "FAIL");
    return all ? 0 : 1;
}

// the DFlash2 drafter against the sglang-math reference (tools/inkling2/dflash2_reference.py, tests/data/inkling2_dflash2_ref.bin):
// the final normed block rows, the top-16 candidates and the 7-token path. The drafter amplifies rounding (the reference
// moves 10 % hidden RMS between BF16 and FP32 activations), so it is held to the Metal port's bands
int run_dflash2_ref(const std::string & model, const std::string & drafter, const std::string & fixture) {
    static const int kTokens[] = {1719, 220, 46389, 7, 77, 8, 25, 198, 149862, 150003, 149934, 196982, 220, 17, 25, 198, 149860,
                                  149862, 150583, 149934, 198, 149862, 150583, 220};
    std::ifstream in(fixture, std::ios::binary);
    int32_t h[8];
    if (!in.read(reinterpret_cast<char *>(h), sizeof(h)) || h[0] != 0x32464C44) throw std::runtime_error("dflash2 fixture header");
    const int M = h[1], anchor = h[2], block = h[4], D = h[5], K = h[6];
    if (M != int(std::size(kTokens)) || K != 16) throw std::runtime_error("dflash2 fixture shape");
    std::vector<uint32_t> ref_tok(size_t(block - 1)), ref_ids(size_t(block * K));
    std::vector<float> ref_lg(ref_ids.size()), ref_hid(size_t(block * D));
    in.read(reinterpret_cast<char *>(ref_tok.data()), ref_tok.size() * 4);
    in.read(reinterpret_cast<char *>(ref_ids.data()), ref_ids.size() * 4);
    in.read(reinterpret_cast<char *>(ref_lg.data()), ref_lg.size() * 4);
    in.read(reinterpret_cast<char *>(ref_hid.data()), ref_hid.size() * 4);
    if (!in) throw std::runtime_error("dflash2 fixture truncated");
    Inkling2Model m(model, 4096, 64, false, g_budget, g_exact, Inkling2Model::drafter_bytes(drafter), g_kernels);
    m.enable_mtp(drafter);
    std::vector<float> hid;
    std::vector<int> ids, path;
    m.draft_trace(std::vector<int>(std::begin(kTokens), std::end(kTokens)), anchor, hid, ids, path);
    double err = 0, mag = 0;
    for (size_t i = size_t(D); i < ref_hid.size(); ++i) { const double e = hid[i] - ref_hid[i]; err += e * e; mag += double(ref_hid[i]) * ref_hid[i]; }
    int overlap = K;
    for (int r = 1; r < block; ++r) {
        int same = 0;
        for (int j = 0; j < K; ++j) same += std::count(ref_ids.begin() + r * K, ref_ids.begin() + (r + 1) * K, uint32_t(ids[size_t(r * K + j)])) > 0;
        overlap = std::min(overlap, same);
    }
    int matched = 0;
    while (matched < block - 1 && uint32_t(path[size_t(matched)]) == ref_tok[size_t(matched)]) ++matched;
    const double rms = std::sqrt(err / mag);
    std::printf("DFLASH2 head %s hidden_rel_rms %.5f top16_overlap_min %d/%d path %d/%d tokens", g_exact ? "int8-copy" : "bf16", rms, overlap, K, matched,
                block - 1);
    for (int t : path) std::printf(" %d", t);
    std::printf("\nreference tokens");
    for (auto t : ref_tok) std::printf(" %u", t);
    const bool pass = rms < 0.12 && overlap >= 12 && matched >= 1;
    std::printf("\nDFLASH2 %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: inkling2-check MODEL GOLDEN_BIN [--chunk N] [--greedy-only|--tf-only]\n"); return 2; }
    const std::string model = argv[1], gold = argv[2];
    int chunk = 64;
    bool do_tf = true, do_greedy = true, first64 = false, hf = false;
    std::string visits, mt_ref, bench, gdebug, profile, spec_bin, spec_ctx, reuse, df_ref;
    bool onestep = false, handover = false;
    int ndec = 64, graph_debug = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--chunk" && i + 1 < argc) chunk = std::atoi(argv[++i]);
        else if (a == "--greedy-only") do_tf = false;
        else if (a == "--tf-only") do_greedy = false;
        else if (a == "--hf-attention") hf = true;
        else if (a == "--mt" && i + 1 < argc) mt_ref = argv[++i];
        else if (a == "--onestep") onestep = true;
        else if (a == "--handover") handover = true;
        else if (a == "--spec" && i + 2 < argc) { spec_bin = argv[++i]; spec_ctx = argv[++i]; }
        else if (a == "--greedy-debug" && i + 1 < argc) gdebug = argv[++i];
        else if (a == "--bench" && i + 1 < argc) bench = argv[++i];
        else if (a == "--reuse" && i + 1 < argc) reuse = argv[++i];
        else if (a == "--decode" && i + 1 < argc) ndec = std::atoi(argv[++i]);
        else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (a == "--expert-budget-mb" && i + 1 < argc) g_budget = size_t(std::atoll(argv[++i])) << 20;
        else if (a == "--exact" && i + 1 < argc) g_exact = std::string(argv[++i]) != "off";
        else if (a == "--expert-kernels" && i + 1 < argc) { const std::string v = argv[++i]; g_kernels = v == "tensor" ? 1 : v == "warp" ? 0 : -1; }
        else if (a == "--spec-trace") g_spec_trace = true;
        else if (a == "--drafter" && i + 1 < argc) g_drafter = argv[++i];  // a DFlash2 folder for --spec (default: the checkpoint's MTP)
        else if (a == "--dflash2-ref" && i + 1 < argc) df_ref = argv[++i];  // fixture; needs --drafter
        else if (a == "--graph-debug" && i + 1 < argc) graph_debug = std::atoi(argv[++i]);  // prompt length: graph vs eager decode steps
        else if (a == "--visits" && i + 1 < argc) visits = argv[++i];
        else if (a == "--first64") { do_greedy = false; first64 = true; }
    }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (!df_ref.empty()) return run_dflash2_ref(model, g_drafter, df_ref);
        if (graph_debug > 0) return run_graph_debug(model, gold, graph_debug, chunk);
        if (!mt_ref.empty()) return run_mt(model, gold, mt_ref, chunk, hf);
        if (onestep) return run_onestep(model, gold, hf);
        if (handover) return run_handover(model, gold, chunk);
        if (!spec_bin.empty()) return run_spec(model, gold, spec_bin, spec_ctx, chunk, ndec, profile == "spec");
        if (!gdebug.empty()) return run_greedy_debug(model, gold, gdebug, chunk, 12);
        if (!bench.empty()) return run_bench(model, gold, bench, chunk, ndec, profile);
        if (!reuse.empty()) return run_reuse(model, gold, reuse, chunk);
        const auto t0 = std::chrono::steady_clock::now();
        const auto ids = load<int32_t>(gold + "/tf_ids.i32");
        Inkling2Model m(model, int(ids.size()) + 64, chunk, hf, g_budget, g_exact, 0, g_kernels);
        std::printf("loaded in %.0f s\n", secs(t0));
        const int V = m.vocab();
        if (!visits.empty()) { const auto hs = load<float>(visits); m.debug_visits(hs, 64); return 0; }
        bool pass = true;
        std::vector<float> lg;
        if (do_tf) {
            const auto top = load<int32_t>(gold + "/tf_top_ids.i32");
            const auto lp = load<float>(gold + "/tf_top_lp.f32");
            const int T = first64 ? 64 : int(ids.size()), Kt = int(top.size() / ids.size());
            double kld = 0, kmax = 0;
            int agree = 0;
            std::vector<double> bk((T + 127) / 128, 0.0);
            std::vector<int> ba((T + 127) / 128, 0);
            double nll_ours = 0, nll_ref = 0; int nll_n = 0;  // next-token NLL where the reference's top-K holds it
            const auto t1 = std::chrono::steady_clock::now();
            for (int p = 0; p < T; p += chunk) {
                const int n = std::min(chunk, T - p);
                m.forward(ids.data() + p, n, n, lg);
                for (int i = 0; i < n; ++i) {
                    const float * x = lg.data() + size_t(i) * V;
                    double mx = -1e30; int am = 0;
                    for (int v = 0; v < V; ++v) if (x[v] > mx) { mx = x[v]; am = v; }
                    double z = 0;
                    for (int v = 0; v < V; ++v) z += std::exp(double(x[v]) - mx);
                    const double lz = mx + std::log(z);
                    const int t = p + i;
                    double k = 0;
                    for (int j = 0; j < Kt; ++j) {
                        const double lr = lp[size_t(t) * Kt + j], lq = double(x[top[size_t(t) * Kt + j]]) - lz;
                        k += std::exp(lr) * (lr - lq);
                    }
                    if (t + 1 < int(ids.size()))
                        for (int j = 0; j < Kt; ++j)
                            if (top[size_t(t) * Kt + j] == ids[t + 1]) {
                                nll_ref -= lp[size_t(t) * Kt + j]; nll_ours -= double(x[ids[t + 1]]) - lz; ++nll_n; break;
                            }
                    kld += k; kmax = std::max(kmax, k);
                    agree += am == top[size_t(t) * Kt];
                    bk[t / 128] += k; ba[t / 128] += am == top[size_t(t) * Kt];
                }
            }
            for (size_t b = 0; b < bk.size(); ++b) {
                const int cnt = std::min(128, T - int(b) * 128);
                std::printf("  positions %4zu-%4zu: KLD %.5f top-1 %.1f %%\n", b * 128, b * 128 + cnt - 1, bk[b] / cnt, 100.0 * ba[b] / cnt);
            }
            std::printf("next-token NLL over %d positions: ours %.4f, reference %.4f\n", nll_n, nll_ours / nll_n, nll_ref / nll_n);
            const double mk = kld / T, a = 100.0 * agree / T;
            const bool ok = mk <= 0.01 && a >= 98.0 && T >= 2000;
            pass = pass && ok;
            std::printf("teacher-forced: %d positions in %.1f s (%.0f tok/s): mean KLD %.5f (max %.4f), top-1 %.2f %% %s\n", T, secs(t1), T / secs(t1),
                        mk, kmax, a, ok ? "PASS" : "FAIL");
        }
        if (do_greedy) {
            std::ifstream g(gold + "/greedy.txt");
            std::string line;
            int prompts = 0, same = 0, flips_all = 0, steps_all = 0;
            while (std::getline(g, line)) {
                std::istringstream in(line);
                std::string name; int nin, ngen;
                in >> name >> nin;
                std::vector<int> pin(nin);
                for (auto & x : pin) in >> x;
                in >> ngen;
                std::vector<int> ref(ngen);
                for (auto & x : ref) in >> x;
                m.reset();
                const auto t1 = std::chrono::steady_clock::now();
                for (int p = 0; p < nin; p += chunk) { const int n = std::min(chunk, nin - p); m.forward(pin.data() + p, n, 1, lg); }
                const int want = std::min(64, ngen);
                std::vector<int> got;
                while (int(got.size()) < want) {
                    int am = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
                    got.push_back(am);
                    if (int(got.size()) == want) break;
                    m.forward(&am, 1, 1, lg);
                }
                int first = -1;
                for (int i = 0; i < want; ++i) if (got[i] != ref[i]) { first = i; break; }
                // teacher-forced along the reference's tokens (prompt prefilled, then single steps): positions whose argmax
                // differs from the reference's next token, a less luck-bound view than the first divergence
                m.reset();
                for (int p = 0; p < nin; p += chunk) { const int n = std::min(chunk, nin - p); m.forward(pin.data() + p, n, 1, lg); }
                int flips = 0;
                for (int i = 0; i < want; ++i) {
                    flips += int(std::max_element(lg.begin(), lg.end()) - lg.begin()) != ref[i];
                    if (i + 1 < want) m.forward(&ref[i], 1, 1, lg);
                }
                flips_all += flips; steps_all += want;
                std::printf("  [teacher-forced: %d of %d steps' argmax differ from the reference] ", flips, want);
                ++prompts; same += first < 0;
                std::printf("greedy %-12s prompt %3d, %d tokens in %.1f s: %s", name.c_str(), nin, want, secs(t1),
                            first < 0 ? "identical\n" : "differs at ");
                if (first >= 0) std::printf("%d (ours %d, ref %d)\n", first, got[first], ref[first]);
            }
            const bool ok = prompts >= 5 && same >= 4;
            pass = pass && ok;
            std::printf("greedy: %d of %d prompts identical over the first 64 tokens %s; teacher-forced argmax differs at %d of %d steps\n", same,
                        prompts, ok ? "PASS" : "FAIL", flips_all, steps_all);
        }
        std::printf(pass ? "inkling2-check: PASS\n" : "inkling2-check: FAIL\n");
        return pass ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "inkling2-check: %s\n", e.what());
        return 1;
    }
}

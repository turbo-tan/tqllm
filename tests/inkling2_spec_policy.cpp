// Inkling-2-Mini spec / plain choice (src/models/inkling2/loop/inkling2_spec_policy.hpp): two plain steps first; spec
// steps that keep gaining fewer tokens per second than plain steps (measured, after kWarm of them) switch to kPlainRun
// plain steps, then kProbe spec steps; each losing probe doubles the plain run; one bad step alone does not switch;
// winning spec stays; a fixed depth always drafts.
#include "inkling2_spec_policy.hpp"

#include <cstdio>

int main() {
    bool ok = true;
    auto check = [&](bool c, const char * what) { if (!c) { std::printf("FAIL: %s\n", what); ok = false; } };
    {   // losing: plain 10 ms / token, spec 25 ms for 2 tokens (80 vs 100 tok/s)
        tqllm::Inkling2SpecPolicy p;
        check(p.next(100) == 0 && p.next(100) == 0, "two plain steps first");
        p.plain_done(0.010); p.plain_done(0.010);
        for (int i = 0; i < tqllm::Inkling2SpecPolicy::kWarm; ++i) {
            const int L = p.next(100);
            check(L >= 1 && L <= 7, "spec steps while warming up");
            p.spec_done(1, L, 0.025);
        }
        int plain = 0;
        for (int i = 0; i < tqllm::Inkling2SpecPolicy::kPlainRun; ++i) plain += p.next(100) == 0;
        check(plain == tqllm::Inkling2SpecPolicy::kPlainRun, "losing spec steps: plain steps follow");
        for (int i = 0; i < tqllm::Inkling2SpecPolicy::kProbe; ++i) { const int L = p.next(100); check(L > 0, "then spec probes"); p.spec_done(1, L, 0.025); }
        plain = 0;
        for (int i = 0; i < 2 * tqllm::Inkling2SpecPolicy::kPlainRun; ++i) plain += p.next(100) == 0;
        check(plain == 2 * tqllm::Inkling2SpecPolicy::kPlainRun, "a losing probe doubles the plain run");
        for (int i = 0; i < tqllm::Inkling2SpecPolicy::kProbe; ++i) { const int L = p.next(100); check(L > 0, "probe again"); p.spec_done(6, L, 0.020); }
        for (int i = 0; i < 6; ++i) { const int L = p.next(100); check(L > 0, "a winning probe returns to spec"); p.spec_done(6, L, 0.020); }
    }
    {   // one bad step among good ones: stays spec (plain 10 ms; spec 20 ms, 4 tokens, once 1 token)
        tqllm::Inkling2SpecPolicy p;
        p.next(100); p.next(100); p.plain_done(0.010); p.plain_done(0.010);
        for (int i = 0; i < 12; ++i) { const int L = p.next(100); check(L > 0, "one bad step does not switch"); p.spec_done(i == 1 ? 0 : 3, L, 0.020); }
    }
    {   // winning: plain 10 ms, spec 20 ms for 4 tokens (200 vs 100 tok/s)
        tqllm::Inkling2SpecPolicy p;
        p.next(100); p.next(100); p.plain_done(0.010); p.plain_done(0.010);
        for (int i = 0; i < 20; ++i) { const int L = p.next(100); check(L > 0, "a winning spec stays"); p.spec_done(3, L, 0.020); }
    }
    {   // fixed depth: always drafts after the two plain steps, never falls back
        tqllm::Inkling2SpecPolicy p(7);
        p.next(100); p.next(100); p.plain_done(0.010); p.plain_done(0.010);
        for (int i = 0; i < 20; ++i) { const int L = p.next(100); check(L == 7, "fixed depth"); p.spec_done(0, L, 0.050); }
        check(p.next(3) == 2, "depth capped by the room left");
    }
    // drafts per step follow the device's step-time line and the acceptance record
    auto converge = [&](double a_ms, double b_ms, double p) {  // step time a + b L, each draft kept with probability p (fixed sequence)
        tqllm::Inkling2SpecPolicy pol;
        pol.next(1000); pol.next(1000); pol.plain_done(0.030); pol.plain_done(0.030);
        unsigned r = 12345;
        int last = 0, hist[8] = {};
        for (int i = 0; i < 400; ++i) {
            const int L = pol.next(1000);
            if (L == 0) { pol.plain_done(0.030); continue; }
            int acc = 0;
            while (acc < L) { r = r * 1103515245u + 12345u; if (double((r >> 8) & 0xffff) / 65536.0 >= p) break; ++acc; }
            pol.spec_done(acc, L, (a_ms + b_ms * L) / 1000.0);
            if (i >= 300) ++hist[L];
            last = L;
        }
        int mode = 1;
        for (int L = 1; L <= 7; ++L) if (hist[L] > hist[mode]) mode = L;
        (void)last;
        return mode;
    };
    // (the EMA rule alone picks about 3-4 and 2 here)
    const int cheap = converge(10.0, 0.4, 0.75), costly = converge(10.0, 6.0, 0.6);
    std::printf("drafts per step: cheap rows, 75 %% acceptance -> %d; costly rows, 60 %% -> %d\n", cheap, costly);
    check(cheap >= 6, "cheap verify rows: long drafts");
    check(costly == 1, "costly verify rows and lower acceptance: one draft");
    {   // per-step verify windows from DFlash2 confidences (GB10-like line: plain 25 ms, a + b L = 25 + 5 L ms)
        tqllm::Inkling2SpecPolicy p;
        p.next(100); p.next(100); p.plain_done(0.025); p.plain_done(0.025);
        const float sure[7] = {0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.97f, 0.97f}, cut[7] = {0.95f, 0.95f, 0.03f, 0.95f, 0.95f, 0.95f, 0.95f};
        check(p.window(sure, 7) == 7, "no line yet, confident drafts: the whole block");
        check(p.window(cut, 7) <= 3, "no line yet, a 3 % draft at position 3: the window stops at or right after it");
        // a measured line: 30 ms at L 1, 60 ms at L 7 (5 ms a row)
        for (int i = 0; i < 6; ++i) { p.spec_done(1, 1, 0.030, sure); p.spec_done(7, 7, 0.060, sure); }
        int w = 0; for (int i = 0; i < 6; ++i) w = std::max(w, p.window(sure, 7));
        check(w == 7, "confident drafts that keep being accepted: the whole block");
        int wc = 99; for (int i = 0; i < 6; ++i) wc = std::min(wc, p.window(cut, 7));
        check(wc <= 2, "a 3 % draft: verify up to it at most");
        check(p.window(sure, 3) <= 3, "never more than the drafts");
    }
    {   // calibration: an over-confident drafter (0.95 drafts rejected at the first position) is corrected by its record
        tqllm::Inkling2SpecPolicy p;
        p.next(100); p.next(100); p.plain_done(0.025); p.plain_done(0.025);
        const float c[7] = {0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f};
        const double before = p.chance(0.95f);
        for (int i = 0; i < 40; ++i) p.spec_done(0, 4, 0.045, c);
        check(before > 0.8 && p.chance(0.95f) < 0.2, "rejected 0.95 drafts: their calibrated chance falls");
        check(p.chance(0.15f) > 0.1 && p.chance(0.15f) < 0.2, "other bins keep their prior");
        int w = 99; for (int i = 0; i < 6; ++i) w = std::min(w, p.window(c, 7));
        check(w <= 2, "then the window shrinks to the first draft or two");
    }
    {   // #106: a step that also re-planned the resident experts reports no time (-1): acceptance counts, timing does not, so
        // copies in a cold start's steps do not make spec look slower than plain (plain 10 ms, spec 10 + 2.5 L ms, all kept;
        // every 8th step re-planned). Before: -1 entered the time averages and the request fell into plain runs.
        tqllm::Inkling2SpecPolicy p;
        const float c[7] = {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        int plain = 0, steps = 0;
        for (int toks = 0; toks < 2000; ++steps) {
            const bool replan = steps % 8 == 7;
            if (p.next(4096) == 0) { ++plain; ++toks; p.plain_done(replan ? -1.0 : 0.010); continue; }
            const int L = p.window(c, 7);
            toks += L + 1; p.spec_done(L, L, replan ? -1.0 : 0.010 + 0.0025 * L, c);
        }
        check(plain == 2, "re-planned steps (time -1) never send a winning request to plain steps");
        int w = 0; for (int i = 0; i < 4; ++i) w = std::max(w, p.window(c, 7));
        check(w >= 5, "and the window stays wide");
        // losing spec (plain 10 ms / token, spec 25 ms for 2 tokens) with a re-planned plain step: plain steps still win
        tqllm::Inkling2SpecPolicy q;
        q.next(100); q.next(100); q.plain_done(0.010); q.plain_done(-1.0);
        for (int i = 0; i < tqllm::Inkling2SpecPolicy::kWarm; ++i) { const int L = q.next(100); q.spec_done(1, L, i == 1 ? -1.0 : 0.025); }
        int qp = 0;
        for (int i = 0; i < tqllm::Inkling2SpecPolicy::kPlainRun; ++i) qp += q.next(100) == 0;
        check(qp == tqllm::Inkling2SpecPolicy::kPlainRun, "a re-planned plain step keeps the plain reference: losing spec still yields plain steps");
    }
    {   // a fixed depth ignores the confidences
        tqllm::Inkling2SpecPolicy p(5);
        const float low[7] = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
        check(p.window(low, 7) == 5 && p.window(low, 3) == 3, "fixed depth: min(depth, drafts)");
    }
    std::printf(ok ? "inkling2-spec-policy: PASS\n" : "inkling2-spec-policy: FAIL\n");
    return ok ? 0 : 1;
}

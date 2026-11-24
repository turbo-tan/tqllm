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
            check(L >= 2 && L <= 7, "spec steps while warming up");
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
    std::printf(ok ? "inkling2-spec-policy: PASS\n" : "inkling2-spec-policy: FAIL\n");
    return ok ? 0 : 1;
}

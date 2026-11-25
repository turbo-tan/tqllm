// Inkling-2 partial residency: the resident experts are re-planned every kEvery decode steps, counting plain tokens and
// drafted verify blocks alike (#106: verify blocks did not count, so a drafted reply kept the prompt's plan to the end).
#include "inkling2_residency.hpp"

#include <cstdio>

int main() {
    int fail = 0;
    auto check = [&](bool c, const char * what) { if (!c) { std::printf("FAIL: %s\n", what); ++fail; } };
    auto replans = [](int n, bool verifying, bool partial, int steps) {
        Inkling2ReplanClock c;
        int r = 0;
        for (int i = 0; i < steps; ++i) r += c.step(n, verifying, partial);
        return r;
    };
    const int k = Inkling2ReplanClock::kEvery;
    check(replans(1, false, true, 3 * k) == 3, "plain decode: a re-plan every kEvery tokens");
    check(replans(4, true, true, 3 * k) == 3, "drafted decode: a re-plan every kEvery verify blocks (#106)");
    check(replans(8, true, true, k - 1) == 0, "no re-plan before kEvery steps");
    check(replans(512, false, true, 3 * k) == 0, "prompt chunks do not count");
    check(replans(1, false, false, 3 * k) == 0 && replans(4, true, false, 3 * k) == 0, "all experts resident: never");
    {   // a reply that mixes plain and drafted steps counts both
        Inkling2ReplanClock c;
        int r = 0;
        for (int i = 0; i < k; ++i) r += c.step(i % 2 ? 1 : 5, i % 2 == 0, true);
        check(r == 1, "mixed plain and verify steps share one count");
    }
    {   // a settled plan backs off; a re-plan that moves experts, or a new prompt, restarts at kEvery
        Inkling2ReplanClock c;
        int when[4] = {}, r = 0;
        for (int i = 1; r < 4 && i < 10 * k * 8; ++i)
            if (c.step(4, true, true)) { when[r++] = i; c.planned(0, 72); }
        check(when[0] == k && when[1] == 3 * k && when[2] == 7 * k && when[3] == 15 * k, "settled re-plans: 32, 64, 128, 256 steps apart");
        c.planned(500, 72);
        check(c.interval == k, "a re-plan that moved experts: back to kEvery");
        for (int i = 0; i < 20; ++i) c.planned(0, 72);
        check(c.interval == Inkling2ReplanClock::kMax, "back-off capped at kMax");
        c.step(300, false, true);
        check(c.interval == k && c.steps == 0, "a prompt chunk restarts the schedule");
    }
    std::printf(fail ? "inkling2-residency: FAIL\n" : "inkling2-residency: PASS\n");
    return fail ? 1 : 0;
}

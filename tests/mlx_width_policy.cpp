#include "width_policy.hpp"
#include <algorithm>
#include <cstdio>

using tqllm::mlx_backend::WidthPolicy;
namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    failures += !ok;
}
// one request of `steps` steps: acceptance accept(width), step ms time(width) (the first step 3x: graph build)
template <class Accept, class Time>
int serve(WidthPolicy& p, int steps, Accept accept, Time time) {
    p.begin();
    int narrow = 0;
    for (int s = 0; s < steps; ++s) {
        const int w = p.choose(0);
        narrow += w == WidthPolicy::kNarrow;
        p.record(w, accept(w), 0, time(w) * (s == 0 ? 3.0 : 1.0));
    }
    return narrow;
}
}  // namespace

int main() {
    auto short_ctx = [](int w) {return w == WidthPolicy::kWide ? 75.0 : 71.0;};
    auto long_ctx = [](int w) {return w == WidthPolicy::kWide ? 102.0 : 90.0;};
    {
        WidthPolicy p;
        p.begin();
        check(p.choose(0) == WidthPolicy::kWide, "first step: wide (untimed)");
        p.record(15, 3, 0, 300);
        check(p.choose(0) == WidthPolicy::kWide, "second step: wide, timed");
        p.record(15, 3, 0, 75);
        check(p.choose(0) == WidthPolicy::kNarrow, "third step: narrow, timed");
    }
    {
        WidthPolicy p;
        const int narrow = serve(p, 100, [](int w) {return w;}, short_ctx);
        check(narrow <= 1, "every drafted token accepted: 16 / 75 ms beats 8 / 71 ms, wide after one narrow probe");
    }
    {
        WidthPolicy p;
        const int narrow = serve(p, 100, [](int) {return 2;}, short_ctx);
        check(narrow >= 97, "acceptance 2: equal tokens, the cheaper narrow step");
    }
    {
        // long context, acceptance 3 (never a full 7): no tail data at all, still narrow (E15 = E7)
        WidthPolicy p;
        const int narrow = serve(p, 100, [](int) {return 3;}, long_ctx);
        check(narrow >= 97, "long context, no full accepts: narrow without any tail sample (j1021 stayed wide)");
    }
    {
        // a = 9 at width 15 (7 at width 7): E7 = 8, E15 = 10. Short: 10/75 > 8/71 -> wide; long: 10/102 < 8/90? no:
        // 0.098 vs 0.089 -> wide; long and slow wide step 130 ms: 0.077 < 0.089 -> narrow
        auto accept = [](int w) {return w == WidthPolicy::kWide ? 9 : 7;};
        WidthPolicy p;
        check(serve(p, 50, accept, short_ctx) <= 1, "short context: the wide step's extra tokens pay for its time");
        // the short request before is one prior sample per width (75 / 71 ms): a few steps to learn the slow wide step
        check(serve(p, 50, accept, [](int w) {return w == WidthPolicy::kWide ? 130.0 : 90.0;}) >= 40,
              "slow wide step: narrow wins once measured");
    }
    {
        // the first (graph-build) step is not timed: a 3x first wide step must not push the request to narrow
        WidthPolicy p;
        const int narrow = serve(p, 60, [](int w) {return w == WidthPolicy::kWide ? 12 : 7;}, short_ctx);
        check(narrow <= 1, "graph-build step excluded from the wide time");
    }
    {
        // j1024 u2: earlier low-acceptance requests (tail 0..1) must not set this request's tail; its own wide steps do
        WidthPolicy p;
        for (int r = 0; r < 4; ++r) serve(p, 40, [](int w) {return w == WidthPolicy::kWide ? 8 : 7;}, short_ctx);
        const int narrow = serve(p, 80, [](int w) {return w == WidthPolicy::kWide ? 14 : 7;}, short_ctx);
        check(narrow <= 2, "a high-acceptance request after low-tail ones stays wide (own tail only)");
    }
    {
        // step times start from the nearest served context: a noisy first wide sample (110 ms) in a request at 4K does
        // not flip it to narrow when the 4K request before measured 85 vs 83 ms (below kNativeFrom: measured choice)
        WidthPolicy p;
        auto t8k = [](int w) {return w == WidthPolicy::kWide ? 85.0 : 83.0;};
        auto accept = [](int w) {return w == WidthPolicy::kWide ? 9 : 7;};
        serve(p, 12, [](int) {return 3;}, short_ctx);  // warm process: both widths used (first-use JIT) before
        p.begin(4096);  // a 4K request that timed both widths
        for (int s = 0; s < 40; ++s) {const int w = s % 2 ? WidthPolicy::kNarrow : WidthPolicy::kWide; p.record(w, accept(w), 0, s == 0 ? 300.0 : t8k(w));}
        p.begin(4100);
        p.record(p.choose(0), 9, 0, 300.0);  // graph build
        p.record(WidthPolicy::kWide, 9, 0, 110.0);  // one noisy sample
        check(p.choose(0) == WidthPolicy::kWide, "noisy single sample smoothed by the nearest context's times");
    }
    {
        // j1030 u2 / code2 / code3: the first tail samples are short (a = 7 exactly), then acceptance runs long. Narrow
        // full accepts hide the tail, so without further wide samples the request stays narrow (v3: 7 x 78 of 80)
        WidthPolicy p;
        int s = 0;
        const int narrow = serve(p, 80, [&s](int w) {const int a = s++ < 4 ? 7 : 14; return w == WidthPolicy::kWide ? a : 7;}, short_ctx);
        check(narrow <= 40, "short first tails do not lock a long-acceptance request into narrow steps");
    }
    {
        // the process's first narrow step pays its kernels' JIT (250 ms); as the only narrow sample it must not become the
        // served prior that keeps a later long request (narrow 90 vs wide 102 ms, no tails) wide for good
        WidthPolicy p;
        int narrow_steps = 0;
        auto jit_short = [&](int w) {
            if (w == WidthPolicy::kNarrow && narrow_steps++ == 0) return 250.0;
            return short_ctx(w);
        };
        p.begin();
        for (int s = 0; s < 12; ++s) {const int w = p.choose(0); p.record(w, w == WidthPolicy::kWide ? 9 : 7, 0, s == 0 ? 300.0 : jit_short(w));}
        const int narrow = serve(p, 40, [](int) {return 3;}, long_ctx);
        check(narrow >= 30, "a first-use JIT narrow sample does not keep a later request wide");
    }
    {
        // j1161 248K on int8 KV: wide 246 ms, narrow 155 ms, acceptance ~2.9 with a rare long accept (1 step in 15 accepts
        // 10 when wide); 12 of 14 drafted per step were wide (11.8 tok/s vs 21.0 capped narrow, j1154)
        WidthPolicy p;
        serve(p, 40, [](int w) {return w == WidthPolicy::kWide ? 9 : 7;}, short_ctx);  // warm process, a short request first
        int s = 0;
        p.begin(248000);
        int narrow = 0;
        for (int k = 0; k < 88; ++k) {
            const int w = p.choose(0), a = s++ % 15 == 0 ? 10 : 2;
            narrow += w == WidthPolicy::kNarrow;
            p.record(w, w == WidthPolicy::kWide ? a : std::min(a, 7), 0, (w == WidthPolicy::kWide ? 246.0 : 155.0) * (k == 0 ? 3.0 : 1.0));
        }
        std::printf("INFO 248K: %d of 88 steps narrow\n", narrow);
        check(narrow >= 70, "very long context, slow wide step, rare long accepts: narrow");
    }
    {
        // j1340 / j1341: from kNativeFrom the request drafts the native block on every step, whatever the time lines
        // say (a 16-draft's head would be verified on the narrow steps the policy already prefers there); below it the
        // measured choice stands
        WidthPolicy p;
        auto accept = [](int w) {return w == WidthPolicy::kWide ? 14 : 7;};
        p.begin(WidthPolicy::kNativeFrom);
        int narrow = 0;
        for (int s = 0; s < 40; ++s) {const int w = p.choose(0); narrow += w == WidthPolicy::kNarrow; p.record(w, accept(w), 0, s == 0 ? 300.0 : short_ctx(w));}
        check(narrow == 40, "a prompt at kNativeFrom drafts the native block on every step");
        p.begin(WidthPolicy::kNativeFrom - 1);
        check(p.choose(0) == WidthPolicy::kWide, "below kNativeFrom the first step is wide (measured choice)");
    }
    std::printf(failures ? "WIDTH_POLICY_FAIL\n" : "WIDTH_POLICY_PASS\n");
    return failures ? 1 : 0;
}

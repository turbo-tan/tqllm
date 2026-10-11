// Prefill dispatch. TQ3 / TQ4 have no native GEMM, so the old fixed rule sent every chunk wider than the matvec kernel
// through a per-chunk FP16 dequantisation of all weights (5060 Ti: ~254 ms per chunk; 28-token prompt 340 ms vs 93 ms
// on TQ3_4S's native kernel). The crossover is now measured per GPU; costs below are from the 3090 / 5060 Ti runs.
#include "prefill_path.hpp"
#include <cstdio>
#include <cstdlib>
using namespace tqllm;
static int fails = 0;
static void expect(bool ok, const char * what) { if (!ok) { std::printf("FAIL: %s\n", what); ++fails; } }
int main() {
    // costs fitted to the A/B runs (whole 27B, ms): batched pass of 32 columns, native, dequant + cuBLAS
    const PrefillCosts tq4_3090{60.0, 32, 0, 0, 140.0, 0.40};      // 3090: batched 86 ms @33, 199 @91; dequant 155 @33, 160 @91
    const PrefillCosts tq4_5060{90.0, 32, 0, 0, 260.0, 0.70};      // 5060 Ti: dequant 340 @28 .. 473 @330
    const PrefillCosts tq34s_5060{90.0, 32, 58.0, 1.06, 260.0, 0.70};  // 5060 Ti TQ3_4S native: 93 @33, 187 @122
    const auto c3090 = prefill_crossover(tq4_3090, 16, 512), c5060 = prefill_crossover(tq4_5060, 16, 512);
    expect(c3090.cross >= 33 && c3090.cross < 91, "3090 TQ4: batched wins at 33 columns, loses at 91 (measured)");
    expect(c5060.cross > c3090.cross, "5060 Ti (lower bandwidth) keeps batched to a wider chunk than the 3090");
    expect(!c5060.native, "TQ4 has no native kernel");
    const auto n34 = prefill_crossover(tq34s_5060, 12, 512);
    expect(n34.native && n34.cross > 400, "5060 Ti TQ3_4S: native beats dequant past 400 columns");
    PrefillFormat tq4{16, 32, false}, tq34s{12, 32, true};
    // uncalibrated: exactly the old rule
    expect(prefill_path(8, tq4, true, false) == PrefillPath::matvec, "n <= matvec width -> matvec");
    expect(prefill_path(40, tq4, true, true) == PrefillPath::batched, "batched verify stays batched");
    expect(prefill_path(28, tq4, true, false) == PrefillPath::dequant, "uncalibrated TQ4 n=28 -> dequant (old rule)");
    expect(prefill_path(100, tq34s, true, false) == PrefillPath::native, "uncalibrated TQ3_4S n=100 -> native (old rule)");
    expect(prefill_path(200, tq34s, true, false) == PrefillPath::dequant, "uncalibrated TQ3_4S n=200 -> dequant (old rule)");
    // calibrated (the fix)
    tq4.cross = c5060.cross; tq4.cross_native = c5060.native;
    expect(prefill_path(28, tq4, true, false) == PrefillPath::batched, "5060 Ti TQ4 n=28 -> batched, not dequant");
    expect(prefill_path(1000, tq4, true, false) == PrefillPath::dequant, "5060 Ti TQ4 n=1000 -> dequant");
    tq34s.cross = n34.cross; tq34s.cross_native = n34.native;
    expect(prefill_path(250, tq34s, true, false) == PrefillPath::native, "5060 Ti TQ3_4S n=250 -> native");
    PrefillFormat tq4_3090f{16, 32, false, c3090.cross, false};
    expect(prefill_path(91, tq4_3090f, true, false) == PrefillPath::dequant, "3090 TQ4 n=91 -> dequant (measured faster)");
    expect(prefill_path(33, tq4_3090f, true, false) == PrefillPath::batched, "3090 TQ4 n=33 -> batched (measured -44 %)");
    expect(prefill_path(300, PrefillFormat{16, 32, false}, false, false) == PrefillPath::batched, "no FP16 buffer -> batched");
    std::printf("prefill_path: %s (3090 TQ4 cross %d, 5060 Ti TQ4 cross %d, 5060 Ti TQ3_4S cross %d native)\n",
                fails ? "FAIL" : "PASS", c3090.cross, c5060.cross, n34.cross);
    return fails ? EXIT_FAILURE : EXIT_SUCCESS;
}

// The qwen35 engine's grouped-expert guard used to reject every non-Plain basis, which blocked
// our own TQ3_4S MoE files ("unsupported grouped expert format: TQ3_4S"). The forward path calls
// the format's own expert_matvec / expert_gemm, which handle any basis, so the guard only needs
// has_expert_matvec(). This test calls the real guard (src/models/qwen/qwen35/qwen35_guard.hpp)
// with a mock Mat for each format: TQ3_4S (Tq3Rot32, has_expert_matvec) must pass; a format
// without expert matvec must still be refused. On the old code the TQ3_4S case threw.
// The second half pins the same engine's Gated DeltaNet mix staging (mix_staging), which the old
// code skipped for Basis::Tq3Rot32 and which the hub's TQ3_4S files need (BF16 [b|a] rows).
#include "qwen35_guard.hpp"
#include "weight_format.hpp"

#include <cstdio>
#include <memory>
#include <string>

using namespace tqllm;

static int failures = 0;
static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static bool throws(const WeightFormat * fmt) {
    Mat m;
    m.p = (const void *)1;  // non-null: the guard checks the format
    m.fmt = fmt;
    try { check_grouped_expert_format(m, "blk.0."); } catch (const std::exception &) { return true; }
    return false;
}

int main() {
    const auto * tq3 = weight_format(46);  // GGML_TQ3_4S
    check(tq3 != nullptr, "TQ3_4S is registered");
    if (!tq3) { std::printf("FAIL: %d failure(s)\n", ++failures); return 1; }

    check(tq3->has_expert_matvec(), "TQ3_4S has grouped expert matvec");
    check(tq3->basis() == Basis::Tq3Rot32, "TQ3_4S basis is Tq3Rot32 (not Plain)");
    check(!throws(tq3), "the guard accepts TQ3_4S (the old code threw here: regression pin)");

    // A format without expert matvec is still refused.
    const auto * f32 = weight_format(0);
    if (f32) {
        check(!f32->has_expert_matvec(), "F32 has no grouped expert matvec");
        check(throws(f32), "the guard still refuses F32 experts");
    }

    // K4 (the loaded form of Q4_K, engine id 1003) has expert matvec and Plain basis: accepted.
    const auto * k4 = weight_format(1003);
    if (k4) {
        check(k4->has_expert_matvec(), "K4 has grouped expert matvec");
        check(k4->basis() == Basis::Plain, "K4 basis is Plain");
        check(!throws(k4), "the guard accepts K4");
    }

    // The Gated DeltaNet mix staging of the verify path. The hub's TQ3_4S 35B-A3B files keep the GDN [b|a]
    // projections BF16 next to a rotated [qkv|z]; the old code skipped the staging for Basis::Tq3Rot32, which
    // left the buffers null and made the verify path write through a null output at 13..NGEMM_BLAS columns.
    check(mix_staging(Basis::Tq3Rot32, true, true).quantized, "TQ3_4S with BF16 GDN b/a gets the mix staging (crash pin)");
    check(mix_staging(Basis::Tq3Rot32, true, true).unrotated, "and an un-rotated FP16 copy for the BF16 rows");
    check(!mix_staging(Basis::Tq3Rot32, false, true).quantized, "TQ3_4S with b/a fused into the mix needs no staging");
    check(mix_staging(Basis::Tq4Rot128, true, true).quantized, "TQ4 keeps its staging");
    // our TQ4 files store the BF16 b/a rotated: the un-rotated copy took Qwen3.8-27B TQ_L MMLU-Pro 37/40 -> 23/40
    check(!mix_staging(Basis::Tq4Rot128, true, true).unrotated, "TQ4 with BF16 b/a reads the rotated activation");
    check(mix_staging(Basis::Plain, true, true).quantized, "Plain (W4A16) keeps its staging");
    check(!mix_staging(Basis::Plain, true, true).unrotated, "Plain feeds the BF16 rows the same FP16 copy as before");
    check(bf16_ba_unrotated(Basis::Tq3Rot32), "hub TQ3_4S: BF16 b/a read the un-rotated activation");
    check(!bf16_ba_unrotated(Basis::Tq4Rot128), "our TQ4: BF16 b/a read the rotated activation");
    check(!bf16_ba_unrotated(Basis::Plain), "Plain: nothing to un-rotate");
    check(!mix_staging(Basis::Tq3Rot32, true, false).quantized, "no quantized target matrices: no staging");

    std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

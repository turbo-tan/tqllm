// Layout regression + real-file cross-check for gguf_block_layout / tensor_bytes.
//
// Two independent gates:
//
// 1. REFACTOR SAFETY. tensor_bytes() was rewritten to go through gguf_block_layout(). This
//    re-derives the ORIGINAL hardcoded per-type byte sizes (the values that shipped before the
//    MoE work, for the dense qwen35/qwen2 path) and requires exact equality, so adding the
//    2-bit types cannot silently change how an existing checkpoint is sized.
//
// 2. REAL-FILE CROSS-CHECK. The block sizes were verified against the only MoE checkpoint on
//    the fleet (GLM-5.3-Flash TQ3_4S, 4 shards, .55). That verification is encoded here as
//    data: the expert tensor shapes observed in shard 1, and the measured per-type byte totals
//    from the same shard. If (shape / block_elems * block_bytes * count) does not reproduce the
//    measured total, the block size is wrong. This is a much stronger check than synthetic
//    blocks because it is the actual file.
//
// Host only: no CUDA, no GPU, no model download (the numbers are recorded constants).
#include <cstdio>
#include <cstdint>

#include "gguf.hpp"

using tqllm::GgmlType;

static int failures = 0;

// The tensor_bytes() table as it stood before the refactor, INCLUDING the five formats added
// on feat/multi-slot after the MoE work branched (Q3_K, Q5_K, IQ3_XXS, IQ3_S, IQ4_XS). This
// list is what the refactor must reproduce byte-for-byte; if a format is added upstream and not
// here, gate 1 will not catch its loss, so update both together.
struct Legacy { uint32_t type; uint64_t block; uint64_t per_block; };
static const Legacy LEGACY[] = {
    {GgmlType::GGML_F32,     1,   4},
    {GgmlType::GGML_F16,     1,   2},
    {GgmlType::GGML_BF16,    1,   2},
    {GgmlType::GGML_Q4_0,    32,  18},
    {GgmlType::GGML_Q8_0,    32,  34},
    {GgmlType::GGML_Q2_K,    256, 84},
    {GgmlType::GGML_Q3_K,    256, 110},
    {GgmlType::GGML_Q4_K,    256, 144},
    {GgmlType::GGML_Q5_K,    256, 176},
    {GgmlType::GGML_Q6_K,    256, 210},
    {GgmlType::GGML_IQ2_XXS, 256, 66},
    {GgmlType::GGML_IQ2_XS,  256, 74},
    {GgmlType::GGML_IQ1_S,   256, 50},
    {GgmlType::GGML_IQ4_NL,  32,  18},
    {GgmlType::GGML_IQ1_M,   256, 56},
    {GgmlType::GGML_IQ3_XXS, 256, 98},
    {GgmlType::GGML_IQ3_S,   256, 110},
    {GgmlType::GGML_IQ2_S,   256, 82},
    {GgmlType::GGML_IQ4_XS,  256, 136},
    {GgmlType::GGML_TQ3_4S,  32,  16},
    {GgmlType::GGML_TQ4,     128, 66},
    {GgmlType::GGML_TQ3,     128, 50},
};

static void check_layout_matches_legacy() {
    printf("-- gate 1: refactor safety (existing dense-path types unchanged) --\n");
    for (const auto & L : LEGACY) {
        uint64_t elems = 0, bytes = 0;
        const bool ok = tqllm::gguf_block_layout(L.type, elems, bytes);
        if (!ok) {
            printf("   FAIL type %u: gguf_block_layout returned false (was supported)\n", L.type);
            ++failures;
            continue;
        }
        if (elems != L.block || bytes != L.per_block) {
            printf("   FAIL type %u: got (%llu,%llu) expected (%llu,%llu)\n", L.type,
                   (unsigned long long) elems, (unsigned long long) bytes,
                   (unsigned long long) L.block, (unsigned long long) L.per_block);
            ++failures;
            continue;
        }
        // And the byte count for a realistic tensor size must match the legacy formula.
        const uint64_t n = L.block * 37;  // an arbitrary whole number of blocks
        const uint64_t expect = n / L.block * L.per_block;
        uint64_t blk, bpb;
        tqllm::gguf_block_layout(L.type, blk, bpb);
        const uint64_t got = n / blk * bpb;
        if (got != expect) {
            printf("   FAIL type %u: n=%llu bytes got %llu expected %llu\n", L.type,
                   (unsigned long long) n, (unsigned long long) got,
                   (unsigned long long) expect);
            ++failures;
        }
    }
    printf("   %zu legacy types checked\n", sizeof(LEGACY) / sizeof(LEGACY[0]));
}

// Observed in GLM-5.3-Flash-TQ3_4S.gguf-00001-of-00004.gguf (read-only header parse,
// 2026-09-29, script ~/.hermes/scripts/verify_block_sizes.py + gguf_tensors.py).
struct Observed {
    const char * what;
    uint32_t type;
    uint64_t d0, d1, d2;   // expert tensor dims [cols, rows, n_experts]
    uint64_t count;        // how many such tensors in shard 1
    double measured_gib;   // measured per-type total for shard 1, in GiB
};

static void check_real_file() {
    printf("-- gate 2: real-file cross-check (GLM-5.3-Flash shard 1/4) --\n");

    // IQ2_S in shard 1 is exactly ffn_gate_exps (11) + ffn_up_exps (11), both [4096,2048,288].
    // Q2_K  in shard 1 is exactly ffn_down_exps (11), [2048,4096,288].
    const Observed IQ2S{"IQ2_S gate+up experts", GgmlType::GGML_IQ2_S, 4096, 2048, 288, 22, 15.86};
    const Observed Q2K {"Q2_K down experts",      GgmlType::GGML_Q2_K,  2048, 4096, 288, 11,  8.12};

    for (const auto & o : {IQ2S, Q2K}) {
        uint64_t elems = 0, bpb = 0;
        if (!tqllm::gguf_block_layout(o.type, elems, bpb)) {
            printf("   FAIL %s: type %u unsupported\n", o.what, o.type);
            ++failures;
            continue;
        }
        const uint64_t n = o.d0 * o.d1 * o.d2;
        if (n % elems) {
            printf("   FAIL %s: n=%llu not divisible by block %llu\n", o.what,
                   (unsigned long long) n, (unsigned long long) elems);
            ++failures;
            continue;
        }
        const double per_tensor = double(n / elems * bpb);
        const double total = per_tensor * double(o.count);
        const double total_gib = total / double(1024ull * 1024ull * 1024ull);
        const double rel = (total_gib > o.measured_gib)
                               ? (total_gib - o.measured_gib) / o.measured_gib
                               : (o.measured_gib - total_gib) / o.measured_gib;
        printf("   %-26s per_tensor=%llu B x %llu = %.3f GiB, measured %.2f GiB, rel diff %.4f%%\n",
               o.what, (unsigned long long) per_tensor, (unsigned long long) o.count,
               total_gib, o.measured_gib, rel * 100.0);
        // The measured totals are printed to 2 decimals, so allow 0.5% - far tighter than a
        // wrong block size (84 vs 82 bytes is a 2.4% error and would be caught).
        if (rel > 0.005) {
            printf("   FAIL %s: predicted bytes do not reproduce the measured file total\n", o.what);
            ++failures;
        }
    }

    // Negative control: swapping the two block sizes must FAIL the check, proving gate 2 has
    // discriminating power rather than passing for any plausible number.
    const uint64_t n_iq2s = 4096ull * 2048ull * 288ull;
    const double wrong = double(n_iq2s / 256 * 84) * 22.0 / double(1024ull*1024ull*1024ull);
    const double rel_wrong = (wrong - 15.86) / 15.86;
    printf("   negative control: IQ2_S sized as Q2_K (84 B) would predict %.3f GiB (%.2f%% off) -> %s\n",
           wrong, rel_wrong * 100.0, rel_wrong > 0.005 ? "correctly rejected" : "NOT REJECTED");
    if (rel_wrong <= 0.005) {
        printf("   FAIL: negative control did not discriminate\n");
        ++failures;
    }
}

int main() {
    check_layout_matches_legacy();
    check_real_file();
    if (failures) {
        printf("\nFAIL: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASS: layout table matches the pre-refactor sizes and reproduces the real file\n");
    return 0;
}

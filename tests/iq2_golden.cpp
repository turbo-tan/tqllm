// Golden test: our host decoders must match ggml's own reference decoders EXACTLY (bit-for-bit
// in FP32 output) on random blocks. Exactness is required because these are lossless decodes of
// a fixed bit layout - any difference means a layout mistake, which would silently corrupt every
// weight read through that format (and every MoE expert stored in it).
//
// Formats covered: Q2_K, IQ2_S, IQ2_XXS, IQ2_XS, IQ4_NL.
//
// Random bytes are a SAFE and deliberately harsh input here: every grid/sign index in these
// formats is masked to the table's range by construction, so no implementation can read out of
// bounds, and random data exercises branches a real quantizer might never emit.
//
// Each format also gets a non-vacuity guard, because a decoder that returned all zeros would
// "match" an all-zero reference and pass for the wrong reason.
//
// Build (host only, no CUDA, no GPU, no model download): tools/run_iq2_golden.sh
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dequant.hpp"

extern "C" {
#include "ggml.h"
#include "ggml-quants.h"
}

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t xrng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

// FP16 bits from a float (round-to-nearest) so we can write valid block scales.
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return uint16_t(sign);            // flush subnormals to zero
    if (exp >= 31) return uint16_t(sign | 0x7C00u); // inf
    return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
}
// A scale that is neither zero nor inf, so decoded output is genuinely non-trivial.
static uint16_t rand_scale() {
    return f32_to_f16(float(double(xrng() % 1000) / 500.0 + 0.05));
}

// --------------------------------------------------------------- block fillers ----
// Each fills exactly one block with arbitrary-but-structurally-valid bytes.

static void fill_q2k(uint8_t * b) {           // 84 B: scales[16] qs[64] d dmin
    std::memset(b, 0, 84);
    for (int i = 0; i < 16; ++i) b[i] = uint8_t(xrng());          // scales
    for (int i = 16; i < 80; ++i) b[i] = uint8_t(xrng());         // qs (2-bit codes)
    uint16_t d = rand_scale(), dm = rand_scale();
    std::memcpy(b + 80, &d, 2);
    std::memcpy(b + 82, &dm, 2);
}

static void fill_iq2s(uint8_t * b) {          // 82 B: d qs[64] qh[8] scales[8]
    std::memset(b, 0, 82);
    uint16_t d = rand_scale();
    std::memcpy(b, &d, 2);
    for (int i = 2; i < 66; ++i) b[i] = uint8_t(xrng());          // qs[64]
    for (int i = 66; i < 74; ++i) b[i] = uint8_t(xrng() & 0xFF);  // qh[8]
    for (int i = 74; i < 82; ++i) b[i] = uint8_t(xrng());         // scales[8]
}

static void fill_iq2xxs(uint8_t * b) {        // 66 B: d qs[64]
    std::memset(b, 0, 66);
    uint16_t d = rand_scale();
    std::memcpy(b, &d, 2);
    for (int i = 2; i < 66; ++i) b[i] = uint8_t(xrng());
}

static void fill_iq2xs(uint8_t * b) {         // 74 B: d qs[64] scales[8]
    std::memset(b, 0, 74);
    uint16_t d = rand_scale();
    std::memcpy(b, &d, 2);
    for (int i = 2; i < 66; ++i) b[i] = uint8_t(xrng());          // qs[64] (32 uint16)
    for (int i = 66; i < 74; ++i) b[i] = uint8_t(xrng());         // scales[8]
}

static void fill_iq4nl(uint8_t * b) {         // 18 B: d qs[16]
    std::memset(b, 0, 18);
    uint16_t d = rand_scale();
    std::memcpy(b, &d, 2);
    for (int i = 2; i < 18; ++i) b[i] = uint8_t(xrng());          // 32 4-bit codes
}

// ------------------------------------------------- ggml reference adapters ----
static void ref_q2k   (const void * p, float * y, int64_t k) { dequantize_row_q2_K   (reinterpret_cast<const block_q2_K    *>(p), y, k); }
static void ref_iq2s  (const void * p, float * y, int64_t k) { dequantize_row_iq2_s  (reinterpret_cast<const block_iq2_s   *>(p), y, k); }
static void ref_iq2xxs(const void * p, float * y, int64_t k) { dequantize_row_iq2_xxs(reinterpret_cast<const block_iq2_xxs *>(p), y, k); }
static void ref_iq2xs (const void * p, float * y, int64_t k) { dequantize_row_iq2_xs (reinterpret_cast<const block_iq2_xs  *>(p), y, k); }
static void ref_iq4nl (const void * p, float * y, int64_t k) { dequantize_row_iq4_nl (reinterpret_cast<const block_iq4_nl  *>(p), y, k); }

struct FmtCase {
    const char * name;
    uint32_t gguf_type;      // id passed to tqllm::dequantize_host
    int64_t block_elems;     // weights per block
    int64_t block_bytes;     // encoded bytes per block
    void (*ggml_ref)(const void *, float *, int64_t);
    void (*fill)(uint8_t *);
};

int main() {
    // Block sizes are ggml's own (blck_size / type_size from the type table):
    // Q2_K 256/84, IQ2_S 256/82, IQ2_XXS 256/66, IQ2_XS 256/74, IQ4_NL 32/18.
    const FmtCase CASES[] = {
        {"Q2_K",    10, 256, 84, ref_q2k,    fill_q2k},
        {"IQ2_S",   22, 256, 82, ref_iq2s,   fill_iq2s},
        {"IQ2_XXS", 16, 256, 66, ref_iq2xxs, fill_iq2xxs},
        {"IQ2_XS",  17, 256, 74, ref_iq2xs,  fill_iq2xs},
        {"IQ4_NL",  20,  32, 18, ref_iq4nl,  fill_iq4nl},
    };

    const int BLOCKS = 64;
    int total_failures = 0;
    int formats_checked = 0;

    for (const FmtCase & c : CASES) {
        const int64_t raw_bytes = int64_t(BLOCKS) * c.block_bytes;
        const int64_t n = int64_t(BLOCKS) * c.block_elems;
        // Explicit resize, not `vec(size_t(n))` - that form is parsed as a function declaration
        // (most vexing parse) and silently makes the name a function pointer.
        std::vector<uint8_t> raw;
        raw.resize(size_t(raw_bytes));
        for (int i = 0; i < BLOCKS; ++i) c.fill(&raw[size_t(i) * size_t(c.block_bytes)]);

        std::vector<float> ours, want;
        ours.resize(size_t(n));
        want.resize(size_t(n));
        tqllm::dequantize_host(c.gguf_type, raw.data(), ours.data(), n);
        c.ggml_ref(raw.data(), want.data(), n);

        int mismatches = 0;
        double max_abs = 0;
        int nonzero = 0;
        double sum_abs = 0;
        for (int64_t i = 0; i < n; ++i) {
            double d = double(ours[i]) - double(want[i]);
            if (d < 0) d = -d;
            if (d > max_abs) max_abs = d;
            if (ours[i] != want[i]) ++mismatches;
            if (want[i] != 0.f) ++nonzero;
            sum_abs += double(want[i] < 0 ? -want[i] : want[i]);
        }

        const double nonzero_frac = double(nonzero) / double(n);
        printf("%-8s %6lld weights  exact-mismatch=%-6d max_abs_diff=%.3e  ref nonzero=%.1f%% mean|x|=%.3f\n",
               c.name, (long long) n, mismatches, max_abs, nonzero_frac * 100.0, sum_abs / double(n));

        if (nonzero_frac < 0.5) {
            printf("         FAIL: reference output mostly zero - comparison would be vacuous\n");
            ++total_failures;
        }
        if (mismatches) {
            printf("         FAIL: %d weight(s) differ from the ggml reference\n", mismatches);
            ++total_failures;
        }
        ++formats_checked;
    }

    if (total_failures) {
        printf("\nFAIL: %d problem(s) across %d formats\n", total_failures, formats_checked);
        return 1;
    }
    printf("\nPASS: all %d host decoders bit-exact vs their ggml reference\n", formats_checked);
    return 0;
}

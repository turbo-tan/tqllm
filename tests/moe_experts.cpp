// Gate for the 3-D MoE expert slicer (gguf_expert_slice / gguf_expert_count).
//
// The failure mode this must catch is SILENT: a wrong per-expert stride reads the neighbouring
// expert's weights, and the model still produces fluent text - it is just the wrong network.
// Arithmetic self-consistency is not enough, so this test writes a real synthetic GGUF file
// whose experts carry distinguishable values, reads it back through the actual loader, and
// requires each slice to decode to ITS OWN expert's values.
//
// Host only: no CUDA, no GPU, no model download.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

using namespace tqllm;

static int failures = 0;
static void check(bool ok, const std::string & what) {
    printf("   %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// ---------------------------------------------------------------- synthetic GGUF writer ----
static void put_u8(std::vector<char> & b, uint8_t v) { b.push_back(char(v)); }
static void put_u32(std::vector<char> & b, uint32_t v) {
    b.insert(b.end(), reinterpret_cast<const char *>(&v), reinterpret_cast<const char *>(&v) + 4);
}
static void put_u64(std::vector<char> & b, uint64_t v) {
    b.insert(b.end(), reinterpret_cast<const char *>(&v), reinterpret_cast<const char *>(&v) + 8);
}
static void put_str(std::vector<char> & b, const std::string & s) {
    put_u64(b, uint64_t(s.size()));
    b.insert(b.end(), s.begin(), s.end());
}

// Writes a GGUF v3 file with two tensors:
//   experts.f32  [COLS][ROWS][N_EXP]  F32      - value(e,r,c) = e*10000 + r*100 + c
//   experts.q2k  [COLS][ROWS][N_EXP]  Q2_K     - opaque payload, layout checks only
// Returns the path.
static const uint64_t COLS = 32, ROWS = 8, N_EXP = 5;

static std::string write_synthetic(const std::string & path) {
    std::vector<char> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put_u32(b, 3);          // version
    put_u64(b, 2);          // n_tensors
    put_u64(b, 1);          // n_kv

    // one metadata kv: general.alignment = 32
    put_str(b, "general.alignment");
    put_u32(b, 4);          // UINT32
    put_u32(b, 32);

    // --- tensor info 1: F32 experts
    const uint64_t f32_bytes = COLS * ROWS * N_EXP * 4;
    put_str(b, "experts.f32");
    put_u32(b, 3);                       // n_dims
    put_u64(b, COLS); put_u64(b, ROWS); put_u64(b, N_EXP);
    put_u32(b, 0);                       // GGML_F32
    put_u64(b, 0);                       // offset within data section

    // --- tensor info 2: Q2_K experts (256 elems / 84 bytes per block)
    put_str(b, "experts.q2k");
    put_u32(b, 3);
    put_u64(b, COLS); put_u64(b, ROWS); put_u64(b, N_EXP);
    put_u32(b, 10);                      // GGML_Q2_K
    put_u64(b, f32_bytes);               // offset within data section

    // pad to alignment (data_offset logic in gguf_open rounds up after the tensor table)
    while (b.size() % 32) put_u8(b, 0);

    // --- data: F32 with per-expert recognisable values
    for (uint64_t e = 0; e < N_EXP; ++e)
        for (uint64_t r = 0; r < ROWS; ++r)
            for (uint64_t c = 0; c < COLS; ++c) {
                float v = float(e * 10000 + r * 100 + c);
                b.insert(b.end(), reinterpret_cast<const char *>(&v),
                         reinterpret_cast<const char *>(&v) + 4);
            }
    // --- data: Q2_K payload, filled with a per-expert marker byte
    for (uint64_t e = 0; e < N_EXP; ++e) {
        const uint64_t per_expert = COLS * ROWS / 256 * 84;
        for (uint64_t i = 0; i < per_expert; ++i) put_u8(b, uint8_t(0xE0 + e));
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(b.data(), std::streamsize(b.size()));
    out.close();
    return path;
}

// ------------------------------------------------------------------------------- tests ----
static void test_synthetic_f32_experts(const std::string & path) {
    printf("-- synthetic F32 experts: each slice must decode to ITS OWN expert --\n");
    GgufFile f = gguf_open(path);
    const GgufTensor & t = f.tensors.at("experts.f32");

    check(gguf_expert_count(t) == N_EXP, "expert count == 5");
    check(t.ne.size() == 3 && t.ne[0] == COLS && t.ne[1] == ROWS && t.ne[2] == N_EXP,
          "tensor dims are [COLS][ROWS][N_EXP]");

    // Whole-tensor read: verify the file really holds distinct experts (non-vacuity).
    std::vector<char> whole = f.read(t);
    check(whole.size() == COLS * ROWS * N_EXP * 4, "whole tensor byte count");
    const float * wf = reinterpret_cast<const float *>(whole.data());
    bool distinct = true;
    for (uint64_t e = 1; e < N_EXP && distinct; ++e)
        if (wf[0] == wf[e * COLS * ROWS]) distinct = false;
    check(distinct, "experts hold distinguishable values (test is not vacuous)");

    // Per-expert slice must equal the corresponding span of the whole tensor exactly.
    bool all_ok = true;
    for (uint64_t e = 0; e < N_EXP; ++e) {
        GgufTensor s = gguf_expert_slice(t, e, "experts.f32");
        if (s.ne.size() != 2 || s.ne[0] != COLS || s.ne[1] != ROWS) { all_ok = false; break; }
        if (s.bytes != COLS * ROWS * 4) { all_ok = false; break; }
        std::vector<char> got = f.read(s);
        if (got.size() != s.bytes) { all_ok = false; break; }
        if (std::memcmp(got.data(), whole.data() + e * s.bytes, s.bytes) != 0) { all_ok = false; break; }
    }
    check(all_ok, "every slice is byte-identical to its own expert span");

    // Decode check: slice(e)[0] must be e*10000 (the value we wrote).
    bool values_ok = true;
    for (uint64_t e = 0; e < N_EXP; ++e) {
        GgufTensor s = gguf_expert_slice(t, e, "experts.f32");
        std::vector<char> got = f.read(s);
        const float v0 = *reinterpret_cast<const float *>(got.data());
        const float vlast = reinterpret_cast<const float *>(got.data())[COLS * ROWS - 1];
        if (v0 != float(e * 10000)) values_ok = false;
        if (vlast != float(e * 10000 + (ROWS - 1) * 100 + (COLS - 1))) values_ok = false;
    }
    check(values_ok, "sliced values identify the correct expert (first and last element)");

    // An off-by-one stride must be DETECTED by the value check, not just by the memcmp above.
    GgufTensor s1 = gguf_expert_slice(t, 1, "experts.f32");
    std::vector<char> g1 = f.read(s1);
    check(*reinterpret_cast<const float *>(g1.data()) == 10000.f,
          "expert 1 reads 10000, not expert 0's 0 or expert 2's 20000");
}

static void test_synthetic_q2k_layout(const std::string & path) {
    printf("-- synthetic Q2_K experts: block-math slicing --\n");
    GgufFile f = gguf_open(path);
    const GgufTensor & t = f.tensors.at("experts.q2k");
    check(gguf_expert_count(t) == N_EXP, "Q2_K expert count == 5");

    const uint64_t per_expert = COLS * ROWS / 256 * 84;
    bool ok = true;
    for (uint64_t e = 0; e < N_EXP; ++e) {
        GgufTensor s = gguf_expert_slice(t, e, "experts.q2k");
        if (s.bytes != per_expert) { ok = false; break; }
        std::vector<char> got = f.read(s);
        if (got.size() != per_expert) { ok = false; break; }
        // every byte of expert e was written as 0xE0 + e
        for (char c : got) if (uint8_t(c) != uint8_t(0xE0 + e)) { ok = false; break; }
        if (!ok) break;
    }
    check(ok, "each Q2_K slice reads exactly its own expert's bytes");
}

static void test_fail_closed() {
    printf("-- fail-closed behaviour (must THROW, not return something plausible) --\n");
    auto throws = [](const GgufTensor & t, uint64_t idx) {
        try { gguf_expert_slice(t, idx, "t"); return false; }
        catch (const std::runtime_error &) { return true; }
    };

    GgufTensor good;
    good.type = 0; good.ne = {COLS, ROWS, N_EXP}; good.offset = 0;
    good.bytes = COLS * ROWS * N_EXP * 4;

    check(throws(good, N_EXP), "index == n_experts throws");
    check(throws(good, N_EXP + 100), "index far out of range throws");

    GgufTensor two_d = good; two_d.ne = {COLS, ROWS};
    check(throws(two_d, 0), "2-D tensor throws (not an expert tensor)");

    GgufTensor zero_exp = good; zero_exp.ne = {COLS, ROWS, 0};
    check(throws(zero_exp, 0), "zero experts throws");

    // A tensor whose recorded byte count disagrees with the block math must throw: this is the
    // guard against a wrong layout assumption silently mis-reading every expert.
    GgufTensor bad_bytes = good; bad_bytes.bytes = good.bytes + 4;
    check(throws(bad_bytes, 0), "byte-count mismatch throws");

    // An expert whose element count does not divide into whole quant blocks cannot be sliced.
    GgufTensor bad_block; bad_block.type = 10;  // Q2_K, 256-element blocks
    bad_block.ne = {100, 7, N_EXP}; bad_block.offset = 0;
    bad_block.bytes = 100 * 7 * N_EXP;
    check(throws(bad_block, 0), "per-expert count not block-aligned throws");

    // 4-D is not an expert tensor in this scheme.
    GgufTensor four_d = good; four_d.ne = {COLS, ROWS, N_EXP, 2};
    check(throws(four_d, 0), "4-D tensor throws");
}

static void test_real_glm_shapes() {
    printf("-- real GLM-5.3-Flash expert shapes (constants, no file needed) --\n");
    // From the shard-1 tensor table: ffn_gate_exps [4096,2048,288] IQ2_S,
    // ffn_down_exps [2048,4096,288] Q2_K.
    struct Case { const char * name; uint32_t type; uint64_t c, r, e; uint64_t expect; };
    const Case cases[] = {
        {"ffn_gate_exps IQ2_S", 22, 4096, 2048, 288, 4096ull * 2048ull / 256ull * 82ull},
        {"ffn_up_exps   IQ2_S", 22, 4096, 2048, 288, 4096ull * 2048ull / 256ull * 82ull},
        {"ffn_down_exps Q2_K ", 10, 2048, 4096, 288, 2048ull * 4096ull / 256ull * 84ull},
    };
    for (const auto & c : cases) {
        GgufTensor t;
        t.type = c.type; t.ne = {c.c, c.r, c.e}; t.offset = 1000;
        t.bytes = c.expect * c.e;
        bool ok = true;
        for (uint64_t i : {0ull, 1ull, 143ull, 287ull}) {
            GgufTensor s = gguf_expert_slice(t, i, c.name);
            if (s.bytes != c.expect) ok = false;
            if (s.offset != 1000 + i * c.expect) ok = false;
            if (s.ne.size() != 2 || s.ne[0] != c.c || s.ne[1] != c.r) ok = false;
        }
        printf("   %-22s per_expert=%llu B, last offset=%llu\n", c.name,
               (unsigned long long) c.expect,
               (unsigned long long) (1000 + 287 * c.expect));
        check(ok, std::string(c.name) + " slices consistently across all 288 experts");
    }
    // Sanity against the measured file: 22 IQ2_S expert tensors in shard 1 totalled 15.86 GiB.
    // Each tensor holds 288 experts, so the tensor total is per_expert * 288 * 22.
    const double per_expert = double(4096ull * 2048ull / 256ull * 82ull);
    const double per_tensor = per_expert * 288.0;
    const double total = per_tensor * 22.0 / double(1024ull * 1024ull * 1024ull);
    printf("   per_expert=%.0f B, per_tensor=%.3f GiB, 22 tensors = %.3f GiB (measured 15.86)\n",
           per_expert, per_tensor / double(1024ull * 1024ull * 1024ull), total);
    check(std::fabs(total - 15.86) / 15.86 < 0.005, "reproduces the measured shard-1 total");
}

int main() {
    // The scratch GGUF is written to the current directory with a plain filename: under ctest
    // the working directory is the build tree, not the repo root, so a path prefixed with
    // build-hostonly/ does not exist there. /tmp is banned for artifacts on this fleet.
    const std::string path = "test_moe_experts.gguf";
    write_synthetic(path);
    test_synthetic_f32_experts(path);
    test_synthetic_q2k_layout(path);
    test_fail_closed();
    test_real_glm_shapes();
    std::remove(path.c_str());
    if (failures) {
        printf("\nFAIL: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASS: expert slicing verified against real file bytes and fail-closed on bad input\n");
    return 0;
}

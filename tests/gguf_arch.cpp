// Host-only gate for the C1 arch-detection plumbing added to src/core/gguf.cpp:
//   GgufFile::str(), GgufFile::has(), gguf_shape_string()
//
// Why this is a separate test from gguf_layout.cpp: that one checks byte sizing. This one checks
// the string/presence accessors that config_from_gguf now depends on to derive the metadata
// namespace from `general.architecture` instead of a hardcoded "qwen35." prefix.
//
// The bug this guards against is silent and severe: before C1, an arch was never read at all. A
// qwen35moe file was parsed with the dense prefix and died as "GGUF missing key
// qwen35.block_count" - an error naming a key that legitimately does not exist, pointing the
// reader at the wrong cause. Worse, if a fallback had existed it could have loaded a MoE file as
// dense and produced fluent wrong text.
//
// GgufFile::kv is a public member, so a file can be synthesized in memory: no model download, no
// GPU, no CUDA. Fail-closed throughout - every case asserts a specific outcome.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "gguf.hpp"

using namespace tqllm;

static int failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        printf("   FAIL: %s\n", what);
        ++failures;
    } else {
        printf("   ok: %s\n", what);
    }
}

// Expect a throw whose message contains `needle`. A throw with a different message, or no throw at
// all, is a failure: the caller branches on these messages.
template <typename Fn>
static void expect_throw(Fn && fn, const std::string & needle, const char * what) {
    try {
        fn();
        printf("   FAIL: %s (no exception thrown)\n", what);
        ++failures;
    } catch (const std::exception & e) {
        const std::string m = e.what();
        if (m.find(needle) == std::string::npos) {
            printf("   FAIL: %s (threw, but message lacks '%s'): %s\n", what, needle.c_str(), m.c_str());
            ++failures;
        } else {
            printf("   ok: %s -> \"%s\"\n", what, m.c_str());
        }
    }
}

// ---- str() / has() ------------------------------------------------------------------------
static void check_str_and_has() {
    printf("\n[str()/has()]\n");
    GgufFile f;
    f.path = "synthetic.gguf";

    GgufValue arch;
    arch.str = "qwen35moe";
    f.kv.emplace("general.architecture", arch);

    GgufValue numeric;  // numbers leave .str empty by construction (value() case 8 is the only
    numeric.num = 40;   // writer of .str), which str() must treat as "not a string"
    f.kv.emplace("qwen35moe.block_count", numeric);

    check(f.has("general.architecture"), "has() true for a present key");
    check(!f.has("qwen35.block_count"), "has() false for an absent key");
    check(f.str("general.architecture") == "qwen35moe", "str() returns the arch string");

    // Absent key: must throw, not return "". Returning "" would let a caller branch on an empty
    // arch as if the file declared none, which is exactly the silent path C1 removes.
    expect_throw([&] { f.str("general.architecture_missing"); }, "missing key",
                 "str() on an absent key throws 'missing key'");

    // Present-but-numeric key: must throw distinctly rather than returning "".
    expect_throw([&] { f.str("qwen35moe.block_count"); }, "no string value",
                 "str() on a present non-string key throws 'no string value'");

    // Negative control: an empty-string value is indistinguishable from "not a string" by design.
    GgufValue empty;
    f.kv.emplace("general.architecture_empty", empty);
    expect_throw([&] { f.str("general.architecture_empty"); }, "no string value",
                 "str() on an empty string throws (negative control: cannot fake success)");
}

// ---- the arch-prefix derivation that config_from_gguf now performs -------------------------
static void check_arch_prefix_derivation() {
    printf("\n[arch prefix derivation - the C1 fix]\n");
    // The dense and MoE variants use different metadata namespaces and neither contains the
    // other's keys. Verify that deriving the prefix from the arch string reaches the right keys.
    for (const char * arch : {"qwen35", "qwen35moe"}) {
        GgufFile f;
        GgufValue a;
        a.str = arch;
        f.kv.emplace("general.architecture", a);
        GgufValue n;
        n.num = 40;
        f.kv.emplace(std::string(arch) + ".block_count", n);

        const std::string A = f.str("general.architecture") + ".";
        check(f.has(A + "block_count"),
              (std::string("prefix '") + A + "' reaches block_count for arch " + arch).c_str());
        // The wrong prefix must NOT reach it - this is the regression that caused the confusing
        // "missing key qwen35.block_count" on a qwen35moe file.
        const std::string wrong = std::string(arch) == "qwen35" ? "qwen35moe." : "qwen35.";
        check(!f.has(wrong + "block_count"),
              (std::string("the other arch's prefix '") + wrong + "' does not match").c_str());
    }
}

// ---- gguf_shape_string() ------------------------------------------------------------------
static void check_shape_string() {
    printf("\n[gguf_shape_string()]\n");
    GgufTensor t2;
    t2.ne = {2048, 512};
    check(gguf_shape_string(t2) == "[2048,512]", "2-D shape formats as [2048,512]");

    GgufTensor t3;
    t3.ne = {2048, 512, 256};
    check(gguf_shape_string(t3) == "[2048,512,256]", "3-D expert shape formats as [2048,512,256]");

    GgufTensor t0;
    check(gguf_shape_string(t0) == "[]", "empty shape formats as [] (no crash, no trailing comma)");

    GgufTensor t1;
    t1.ne = {7};
    check(gguf_shape_string(t1) == "[7]", "1-D shape formats as [7]");
}

// ---- gguf_expert_count / slice interplay with the MoE dims --------------------------------
static void check_expert_count() {
    printf("\n[gguf_expert_count() on the verified MoE dims]\n");
    // The real file declares blk.0.ffn_gate_exps.weight as [hidden, expert_ff, n_expert] =
    // [2048, 512, 256] (verified by tools/gguf_truth.py against the actual GGUF).
    GgufTensor gate;
    gate.ne = {2048, 512, 256};
    check(gguf_expert_count(gate) == 256, "3-D expert tensor reports 256 experts");

    GgufTensor shared;  // the shared expert is 2-D, so it must report 0
    shared.ne = {2048, 512};
    check(gguf_expert_count(shared) == 0, "2-D shared expert reports 0 (not an expert tensor)");
}

int main() {
    printf("=== C1 arch-detection host gate (no GPU, no model, no CUDA) ===\n");
    check_str_and_has();
    check_arch_prefix_derivation();
    check_shape_string();
    check_expert_count();

    if (failures) {
        printf("\nFAIL: %d check(s)\n", failures);
        return 1;
    }
    printf("\nPASS: str()/has()/shape_string() behave as config_from_gguf requires\n");
    return 0;
}

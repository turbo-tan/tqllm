#pragma once
// Verify-matvec kernel selection as data: one ordered rule table for every weight format with more than one kernel.
// A rule matches on (format bits, compute capability, k in 32-weight blocks, rows, columns); the first match wins and
// names the kernel and its prefetch mode. Each rule cites the measurement that justified it.
//
// Adding a format or a GPU: implement the kernel(s), make sure the candidates a rule chooses between are tested for
// width independence (and for bit-identity where the comment says "same numbers"), then add rows here - nothing else in
// the launcher changes. Overrides without rebuilding:
//   TQLLM_MATVEC_RULE="<bits>:<legacy|split16>[:<pf>]"  (several separated by ','), e.g. "3:split16:2,4:legacy"
//   TQLLM_DEBUG=matvec  print each distinct pick once (bits, cc, rows x k, columns -> kernel / pf, rule)
#include "../../src/debug.hpp"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

namespace tqllm_cuda::mvr {

// Legacy: the format's older kernel (TQ3 / TQ4: tq4_matvec_k MT / variant path; IQ3: the u4h kernel); Split16: tq3b_matvec_h_k
enum class Kern : int { Legacy = 0, Split16 = 1 };
struct Pick {
    Kern kern;
    int pf;  // Split16 prefetch mode: 0 none, 1 next step, 2 two steps per iteration (same results in every mode)
};
struct Rule {
    int bits;                    // tq3b_matvec_h_k BITS: 3 (TQ3, GGUF 301), 4 (TQ4, GGUF 300), 14 (IQ4N), 21 (IQ3)
    int cc_min, cc_max;          // compute capability as major * 10 + minor (86 = 3090, 120 = 5060 Ti, 121 = GB10)
    int kb_min, kb_max;          // k / 32
    int rows_min;
    Pick pick;
    const char * why;
};

constexpr int ANY = 1 << 30;
// TQ3: Split16 and Legacy differ in MMA grouping (within 0.6%, tests/quant.cu tq3_verify_kernel), so the choice is fixed
// per (cc, shape) - never timed at run time. TQ4: Split16 and Legacy give the same numbers (tq4_verify_kernel).
inline constexpr Rule RULES[] = {
    {3, 121, 121, 0, ANY, 0, {Kern::Legacy, 0}, "TQ3 every shape, GB10: TQ_XS decode 14.65 -> 18.2 tok/s no drafter, 72.8 -> 89.4 with DFlash2; KLD 0.07925 vs 0.07932"},
    {3, 100, ANY, 257, ANY, 0, {Kern::Legacy, 0}, "TQ3 long k, Blackwell: 5060 Ti 20480x17408 legacy 416 us vs split16 476-543"},
    {3, 0, ANY, 257, ANY, 0, {Kern::Split16, 1}, "TQ3 long k: 3090 5120x17408 split16+pf1 64.2 us vs legacy 78.9"},
    {3, 0, ANY, 0, 256, 0, {Kern::Split16, 0}, "TQ3 short k: 3090 17408x5120 53.4 vs 70.0 us; 5060 Ti 336 vs 346 (4x rows)"},
    {4, 0, ANY, 0, 256, 8192, {Kern::Split16, 0}, "TQ4 short k, many rows: 3090 17408x5120 59.7 vs 64.3 us"},
    {4, 0, ANY, 0, ANY, 0, {Kern::Legacy, 0}, "TQ4 small rows / long k: 3090 5120x6144 legacy 31.5 vs 36.1; 5120x17408 59.5 vs 73"},
    {14, 0, ANY, 0, ANY, 0, {Kern::Split16, 0}, "IQ4N: split16 only; pf 0 best on 3090 (17408x5120 63.4 us, 5120x17408 71.8)"},
    {21, 0, ANY, 257, ANY, 0, {Kern::Legacy, 0}, "IQ3 long k: 3090 5120x17408 u4h 83.4 vs split16 98.8 us"},
    {21, 0, ANY, 0, 256, 0, {Kern::Split16, 0}, "IQ3 short k: 3090 17408x5120 split16 73.3 vs u4h 80.3 us"},
};
constexpr int MAX_BITS = 32;

inline int device_cc() {
    static const int cc = [] {
        int dev = 0, ma = 0, mi = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
        return ma * 10 + mi;
    }();
    return cc;
}

struct Override {
    bool set = false;
    Pick pick{Kern::Legacy, 0};
};
inline Override env_override(int bits) {
    static const auto table = [] {
        struct T { Override o[MAX_BITS]; } t;
        if (const char * e = std::getenv("TQLLM_MATVEC_RULE")) {
            std::string s(e);
            size_t p = 0;
            while (p < s.size()) {
                const size_t q = s.find(',', p);
                const std::string item = s.substr(p, q == std::string::npos ? std::string::npos : q - p);
                int b = 0, pf = 0;
                char kn[16] = {};
                if (std::sscanf(item.c_str(), "%d:%15[a-z0-9]:%d", &b, kn, &pf) >= 2 && b >= 0 && b < MAX_BITS)
                    t.o[b] = {true, {std::strcmp(kn, "split16") == 0 ? Kern::Split16 : Kern::Legacy, pf}};
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }
        return t;
    }();
    return bits >= 0 && bits < MAX_BITS ? table.o[bits] : Override{};
}

// Pure lookup (unit-tested): first matching rule, Legacy when none matches or for > 16 columns (Split16 holds 16).
inline Pick lookup(int bits, int cc, int rows, int k, int ncols, const char ** why = nullptr) {
    if (ncols > 16) {
        if (why) *why = "> 16 columns: legacy";
        return {Kern::Legacy, 0};
    }
    const int kb = k / 32;
    for (const auto & r : RULES)
        if (r.bits == bits && cc >= r.cc_min && cc <= r.cc_max && kb >= r.kb_min && kb <= r.kb_max && rows >= r.rows_min) {
            if (why) *why = r.why;
            return r.pick;
        }
    if (why) *why = "no rule: legacy";
    return {Kern::Legacy, 0};
}

inline Pick choose(int bits, int rows, int k, int ncols) {
    const char * why = "";
    Pick p = lookup(bits, device_cc(), rows, k, ncols, &why);
    if (const auto o = env_override(bits); o.set && (o.pick.kern == Kern::Legacy || ncols <= 16)) p = o.pick, why = "env override";
    static const bool log = tqllm::debug_on("matvec");
    if (log) {
        static std::mutex mu;
        static std::set<std::tuple<int, int, int, int>> seen;
        std::lock_guard<std::mutex> lock(mu);
        if (seen.insert({bits, rows, k, ncols <= 8 ? 8 : 16}).second)
            std::fprintf(stderr, "matvec pick: bits %d cc %d %d x %d cols<=%d -> %s pf %d (%s)\n", bits, device_cc(), rows, k,
                         ncols <= 8 ? 8 : 16, p.kern == Kern::Split16 ? "split16" : "legacy", p.pf, why);
    }
    return p;
}

}  // namespace tqllm_cuda::mvr

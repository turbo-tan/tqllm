// int8 prefill vs the DFlash2 drafter's inputs: the target features the drafter reads (captured during prefill) must
// stay as close to FP16 as int8 rounding allows. Reference: FP16 prefill; noise floor: FP16 with another chunk size
// (same maths, different summation order); test: int8 on (Ampere). Model-in-the-loop: set TQLLM_TEST_MODEL (qwen35
// GGUF) and TQLLM_TEST_DRAFTER (DFlash2 GGUF); optional TQLLM_TEST_TOKENS (one id per line, >= 2048 ids). Exit 77 = skip.
#include "tqllm/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {
float bf(uint16_t b) {
    const uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
struct Err {
    double rel = 0, worst = 0;  // relative L2 over all positions; worst single position
    int worst_pos = -1;
};
// per feature layer: error of b against a
std::vector<Err> compare(const std::vector<uint16_t> & a, const std::vector<uint16_t> & b, int P, int nf, int H) {
    std::vector<Err> out(static_cast<size_t>(nf));
    for (int f = 0; f < nf; ++f) {
        double e = 0, r = 0;
        for (int p = 0; p < P; ++p) {
            double ep = 0, rp = 0;
            const size_t o = (size_t(p) * nf + f) * H;
            for (int h = 0; h < H; ++h) {
                const double x = bf(a[o + h]), d = bf(b[o + h]) - x;
                ep += d * d, rp += x * x;
            }
            e += ep, r += rp;
            const double wp = std::sqrt(ep / std::max(rp, 1e-30));
            if (wp > out[size_t(f)].worst) out[size_t(f)].worst = wp, out[size_t(f)].worst_pos = p;
        }
        out[size_t(f)].rel = std::sqrt(e / std::max(r, 1e-30));
    }
    return out;
}
}  // namespace

int main() {
    const char * model = std::getenv("TQLLM_TEST_MODEL");
    const char * drafter = std::getenv("TQLLM_TEST_DRAFTER");
    if (!model || !drafter) { std::puts("int8_features: skipped (set TQLLM_TEST_MODEL and TQLLM_TEST_DRAFTER)"); return 77; }
    std::vector<uint32_t> tokens;
    if (const char * tf = std::getenv("TQLLM_TEST_TOKENS")) {
        std::ifstream in(tf);
        for (uint32_t t; in >> t;) tokens.push_back(t);
    }
    if (tokens.size() < 2048) {  // deterministic stand-in text
        std::mt19937 rng(7);
        tokens.clear();
        for (int i = 0; i < 2048; ++i) tokens.push_back(1000 + rng() % 100000);
    }
    tokens.resize(2048);
    const int P = int(tokens.size());
    auto run = [&](int int8, int ubatch, std::vector<uint16_t> & feats) {
        tqllm::EngineOptions o;
        o.draft_model = drafter;
        o.prefill_int8 = int8;
        o.ubatch = ubatch;
        o.kv = "tq3";
        tqllm::Model m(model, model, 4096, 1, o);  // the tokenizer is read from the GGUF
        return m.drafter_features(tokens, feats);
    };
    std::vector<uint16_t> ref, floor_, i8;
    const int nf = run(0, 0, ref);
    run(0, 1024, floor_);
    run(1, 0, i8);
    const int H = int(ref.size() / (size_t(P) * nf));
    const auto ef = compare(ref, floor_, P, nf, H), ei = compare(ref, i8, P, nf, H);
    int bad = 0;
    std::printf("%-8s %14s %14s %16s %16s\n", "feature", "FP16 floor rel", "int8 rel", "FP16 worst pos", "int8 worst pos");
    for (int f = 0; f < nf; ++f) {
        const auto & a = ef[size_t(f)];
        const auto & b = ei[size_t(f)];
        std::printf("%-8d %14.5f %14.5f %9.4f @%-5d %9.4f @%-5d\n", f, a.rel, b.rel, a.worst, a.worst_pos, b.worst, b.worst_pos);
        // int8 rounding: a few times the summation-order floor, never a localised blow-up
        if (b.rel > std::max(0.05, 10 * a.rel) || b.worst > std::max(0.25, 10 * a.worst)) ++bad;
    }
    if (bad) { std::printf("FAIL: int8 drafter features off on %d of %d layers\n", bad, nf); return 1; }
    std::puts("int8_features: ok");
    return 0;
}

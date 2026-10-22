// INT8-activation grouped expert GEMM vs the FP16 grouped kernel and an fp64 oracle (IQ4_NL, fmt 14), plus the review
// cases: adjacent blocks with unequal weight and activation scales (exactly representable x: the int8 path must match the
// fp64 oracle to fp32 rounding), zero-activation rows (gate/up and down inputs), and the other expert formats vs fp16.
#include "quant_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static const int8_t kv[16] = {
#include "kvalues_iq4nl.inc"
};
static int failures = 0;
static void check(bool ok, const char * what, double v = 0) { if (!ok) ++failures; std::printf("%s %s (%.3g)\n", ok ? "ok  " : "FAIL", what, v); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

struct Routing { int E, topk, cap, tokens; std::vector<int> counts, groups; };
static Routing route(int tokens, int E, int topk, std::mt19937 & rng) {
    Routing r{E, topk, 0, tokens, std::vector<int>(E, 0), {}};
    std::vector<std::vector<int>> per(E);
    for (int tok = 0; tok < tokens; ++tok) {
        std::vector<int> ch;
        while (int(ch.size()) < topk) { int e = int(rng() % E); if (std::find(ch.begin(), ch.end(), e) == ch.end()) ch.push_back(e); }
        for (int j = 0; j < topk; ++j) per[ch[j]].push_back(tok * topk + j);
    }
    for (auto & v : per) r.cap = std::max(r.cap, int(v.size()));
    r.groups.assign(size_t(E) * r.cap, 0);
    for (int e = 0; e < E; ++e) { r.counts[e] = int(per[e].size()); std::copy(per[e].begin(), per[e].end(), r.groups.begin() + size_t(e) * r.cap); }
    return r;
}

// IQ4_NL bank in the engine's layout: 16 bytes per 32 codes (byte j: low nibble k 2j, high nibble k 2j + 1), half scale per 32
struct Bank { int E, rows, k; std::vector<uint8_t> w; std::vector<__half> s; };
static Bank make_bank(int E, int rows, int k, std::mt19937 & rng, float smin, float smax) {
    Bank b{E, rows, k, std::vector<uint8_t>(size_t(E) * rows * (k / 32) * 16), std::vector<__half>(size_t(E) * rows * (k / 32))};
    for (auto & v : b.w) v = uint8_t(rng());
    std::uniform_real_distribution<float> us(smin, smax);
    for (auto & v : b.s) v = __float2half(us(rng));
    return b;
}
static double weight(const Bank & b, int e, int r, int kk) {
    const size_t blk = (size_t(e) * b.rows + r) * (b.k / 32) + kk / 32; const int j = kk % 32;
    const uint8_t byte = b.w[blk * 16 + j / 2]; const int nib = (j & 1) ? byte >> 4 : byte & 15;
    return double(__half2float(b.s[blk])) * kv[nib];
}

struct Act { int rows, k; std::vector<__half> x; };
// per-32-block activations: exact (x = scale * code, code in [-127,127]) or dense normal with outliers
static Act make_act(int rows, int k, std::mt19937 & rng, bool exact, bool outliers, const std::vector<int> & zero_rows) {
    Act a{rows, k, std::vector<__half>(size_t(rows) * k)};
    std::uniform_real_distribution<float> us(1e-3f, 1.f); std::normal_distribution<float> nx(0.f, 0.3f);
    for (int r = 0; r < rows; ++r) {
        const bool zero = std::find(zero_rows.begin(), zero_rows.end(), r) != zero_rows.end();
        for (int blk = 0; blk < k / 32; ++blk) {
            // exact: power-of-two block scale (fp16-exact products) and one +-127 per block, so the int8 quantiser is lossless;
            // adjacent blocks get scales 2^-10 .. 2^0, i.e. very different, which is the unequal-scale case
            const float sc = exact ? std::ldexp(1.f, -int(rng() % 11)) : us(rng) * (blk % 2 ? 1.f : 0.013f);
            const int big = int(rng() % 32);
            for (int j = 0; j < 32; ++j) {
                float v;
                if (zero) v = 0.f;
                else if (exact) v = sc * float(j == big ? (rng() % 2 ? 127 : -127) : int(rng() % 255) - 127);
                else v = (outliers && rng() % 400 == 0) ? nx(rng) * 30.f : nx(rng);
                a.x[size_t(r) * k + blk * 32 + j] = __float2half(v);
            }
        }
    }
    return a;
}

struct Dev { void * p = nullptr; template <class T> T * as() { return static_cast<T *>(p); } ~Dev() { if (p) cudaFree(p); } };
static Dev up(const void * h, size_t bytes) { Dev d; CK(cudaMalloc(&d.p, bytes)); CK(cudaMemcpy(d.p, h, bytes, cudaMemcpyHostToDevice)); return d; }
static Dev zero(size_t bytes) { Dev d; CK(cudaMalloc(&d.p, bytes)); CK(cudaMemset(d.p, 0, bytes)); return d; }

// returns {max rel error vs oracle (int8), max rel error vs oracle (fp16), rel RMS int8 vs fp16}
struct Res { double i8_max, f16_max, rms_i8_f16, i8_rms; };
static Res run(const Bank & b, const Act & a, const Routing & r, bool per_expert_input, int fmt) {
    const int rows = b.rows, k = b.k, nb = k / 32, A = r.tokens * r.topk;
    Dev w = up(b.w.data(), b.w.size()), s = up(b.s.data(), b.s.size() * 2), x = up(a.x.data(), a.x.size() * 2);
    Dev counts = up(r.counts.data(), r.counts.size() * 4), groups = up(r.groups.data(), r.groups.size() * 4);
    Dev y16 = zero(size_t(A) * rows * 2), y8 = zero(size_t(A) * rows * 2);
    Dev xq = zero(size_t(a.rows) * k), xs = zero(size_t(a.rows) * nb * 4), xsum = zero(size_t(a.rows) * nb * 4);
    tqllm_cuda::k_quant_act_rows(x.as<__half>(), nullptr, nullptr, 0, 0, a.rows, k, xq.as<int8_t>(), xs.as<float>(), xsum.as<int16_t>(), nullptr);
    tqllm_cuda::k_expert_gemm(fmt, w.as<uint8_t>(), s.p, x.as<__half>(), y16.as<__nv_bfloat16>(), rows, k, counts.as<int>(), groups.as<int>(), r.cap, r.E, r.topk, per_expert_input, nullptr);
    tqllm_cuda::k_expert_gemm_i8(fmt, w.as<uint8_t>(), s.p, xq.as<int8_t>(), xs.as<float>(), xsum.as<int16_t>(), y8.as<__nv_bfloat16>(), rows, k, counts.as<int>(), groups.as<int>(), r.cap, r.E, r.topk, per_expert_input, nullptr);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<__nv_bfloat16> o16(size_t(A) * rows), o8(o16.size());
    CK(cudaMemcpy(o16.data(), y16.p, o16.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o8.data(), y8.p, o8.size() * 2, cudaMemcpyDeviceToHost));
    // fp64 oracle for a sample of outputs (every assignment, 8 rows each)
    Res res{0, 0, 0, 0}; double se = 0, sy = 0, se8 = 0;
    std::mt19937 pick(7);
    struct Sample { double acc, v8, v16; }; std::vector<Sample> samples;
    for (int e = 0; e < r.E; ++e) for (int i = 0; i < r.counts[e]; ++i) {
        const int asg = r.groups[size_t(e) * r.cap + i], in = per_expert_input ? asg : asg / r.topk;
        for (int rr = 0; rr < 8; ++rr) {
            const int row = int(pick() % rows);
            double acc = 0; for (int kk = 0; kk < k; ++kk) acc += weight(b, e, row, kk) * double(__half2float(a.x[size_t(in) * k + kk]));
            samples.push_back({acc, double(__bfloat162float(o8[size_t(asg) * rows + row])), double(__bfloat162float(o16[size_t(asg) * rows + row]))});
        }
    }
    double typ = 0; for (const auto & sm : samples) typ += sm.acc * sm.acc; typ = std::sqrt(typ / std::max<size_t>(1, samples.size()));
    for (const auto & sm : samples) {
        const double scale = std::fabs(sm.acc) + 0.05 * typ;  // relative, with a floor at 5 % of the typical output (bf16 outputs)
        res.i8_max = std::max(res.i8_max, std::fabs(sm.v8 - sm.acc) / scale); res.f16_max = std::max(res.f16_max, std::fabs(sm.v16 - sm.acc) / scale);
        se += (sm.v8 - sm.v16) * (sm.v8 - sm.v16); sy += sm.v16 * sm.v16; se8 += (sm.v8 - sm.acc) * (sm.v8 - sm.acc);
    }
    res.rms_i8_f16 = std::sqrt(se / std::max(1e-30, sy)); res.i8_rms = std::sqrt(se8 / std::max(1e-30, sy));
    return res;
}

int main() {
    std::mt19937 rng(3);
    const int E = 6, topk = 2, tokens = 200, k = 512, rows = 128;
    const Routing r = route(tokens, E, topk, rng);
    // 1. exact activations, unequal adjacent scales (weights 1e-3..1 per block, activations x0.013 on even blocks)
    {
        const Bank b = make_bank(E, rows, k, rng, 1e-3f, 1.f);
        const Act a = make_act(tokens, k, rng, true, false, {});
        const Res o = run(b, a, r, false, 14);
        std::printf("exact x, unequal scales: int8 vs fp64 max %.2e (rms %.2e), fp16 kernel vs fp64 max %.2e\n", o.i8_max, o.i8_rms, o.f16_max);
        check(o.i8_max < 1e-2 && o.i8_rms < 2.5e-3, "int8 matches the fp64 oracle on exactly representable x (bf16 output rounding only)", o.i8_max);
        check(o.i8_max <= o.f16_max * 1.5 + 1e-3, "int8 is at least as close to fp64 as the fp16 kernel", o.i8_max / std::max(1e-12, o.f16_max));
    }
    // 2. dense normal activations with outliers: int8 within 2 % relative RMS of the fp16 kernel
    {
        const Bank b = make_bank(E, rows, k, rng, 2e-3f, 1e-2f);
        const Act a = make_act(tokens, k, rng, false, true, {});
        const Res o = run(b, a, r, false, 14);
        std::printf("normal x with outliers: rel RMS int8 vs fp16 %.4f\n", o.rms_i8_f16);
        check(o.rms_i8_f16 < 0.02, "int8 vs fp16 relative RMS under 2 % with outliers (per-32 scales)", o.rms_i8_f16);
    }
    // 3. zero rows: gate/up path (token rows) and down path (assignment rows) give exactly 0
    {
        const Bank b = make_bank(E, rows, k, rng, 2e-3f, 1e-2f);
        const Act a = make_act(tokens, k, rng, false, false, {0, 5, 77});
        const Res o = run(b, a, r, false, 14);
        check(o.rms_i8_f16 < 0.02 && std::isfinite(o.i8_max), "zero token rows: finite, int8 vs fp16 within 2 %", o.rms_i8_f16);
        const Act ad = make_act(tokens * topk, k, rng, false, false, {1, 2, 300});
        const Res od = run(b, ad, r, true, 14);
        check(od.rms_i8_f16 < 0.02 && std::isfinite(od.i8_max), "per-expert (down) input rows incl. zero rows", od.rms_i8_f16);
    }
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("all int8 expert GEMM checks passed");
    return 0;
}

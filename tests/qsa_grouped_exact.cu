// Grouped selected attention (the prefill path) vs qsa_attention_direct (decode / verify): bitwise equal outputs on the
// Flash-Next shape (24 query heads, 2 kv heads, head size 256, selection width 2051) with invalid rows, rows past the
// query's context (valid but pos >= cells: score 0, no value) and negative positions. TQFN_QSA_FIXED=0 runs the
// generic kernels instead of the fixed-shape ones.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template<class T> static T * up(const std::vector<T> & h) { T * d; check(cudaMalloc(&d, h.size() * sizeof(T))); check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
static bool run(int tokens, int width, int first_cells, int heads, int kv_heads, int dim) {
    std::mt19937 rng(tokens * 7 + width); std::normal_distribution<float> nd(0.f, 1.f);
    const int cells = first_cells + tokens + 4;  // query t sees positions < first_cells + 1 + t; the rest of the cache is "future"
    std::vector<__nv_bfloat16> k(size_t(cells) * kv_heads * dim), v(k.size());
    for (auto & x : k) x = __float2bfloat16_rn(nd(rng));
    for (auto & x : v) x = __float2bfloat16_rn(nd(rng));
    std::vector<float> q(size_t(tokens) * heads * dim);
    for (auto & x : q) x = 0.3f * nd(rng);
    std::vector<int32_t> pos(size_t(tokens) * width); std::vector<uint8_t> valid(pos.size());
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < width; ++r) {
            const size_t i = size_t(t) * width + r; const unsigned u = rng() % 100; const int seen = first_cells + 1 + t;
            pos[i] = int32_t(rng() % unsigned(seen)); valid[i] = 1;
            if (u < 5) valid[i] = 0;                                    // not selected
            else if (u < 8) pos[i] = seen + int(rng() % 3);             // valid, but past this query's context
            else if (u < 10) pos[i] = -1;
        }
    const auto & kpad = k; const auto & vpad = v;
    auto * dk = up(kpad); auto * dv = up(vpad); auto * dq = up(q); auto * dp = up(pos); auto * dvalid = up(valid);
    float * ref, * out, * sc, * inv;
    const size_t on = size_t(tokens) * heads * dim;
    check(cudaMalloc(&ref, on * 4)); check(cudaMalloc(&out, on * 4));
    check(cudaMalloc(&sc, size_t(tokens) * heads * width * 4)); check(cudaMalloc(&inv, size_t(tokens) * heads * 4));
    tqllm_cuda::QsaDeviceSelection sel; sel.positions = dp; sel.valid = dvalid; sel.width = width;
    tqllm_cuda::qsa_attention_direct(dq, dk, dv, sel, ref, tokens, first_cells + 1, heads, kv_heads, dim, nullptr);
    tqllm_cuda::qsa_attention_grouped(dq, dk, dv, sel, sc, inv, out, tokens, first_cells + 1, heads, kv_heads, dim, nullptr);
    check(cudaDeviceSynchronize());
    if (std::getenv("QSA_BENCH")) {  // ms per grouped call (one prefill chunk of one attention layer)
        cudaEvent_t e0, e1; check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1)); check(cudaEventRecord(e0));
        for (int i = 0; i < 5; ++i) tqllm_cuda::qsa_attention_grouped(dq, dk, dv, sel, sc, inv, out, tokens, first_cells + 1, heads, kv_heads, dim, nullptr);
        check(cudaEventRecord(e1)); check(cudaEventSynchronize(e1)); float ms = 0; check(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("  grouped %.2f ms per call\n", ms / 5);
    }
    std::vector<float> a(on), b(on);
    check(cudaMemcpy(a.data(), ref, on * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(b.data(), out, on * 4, cudaMemcpyDeviceToHost));
    size_t diff = 0; double worst = 0;
    for (size_t i = 0; i < on; ++i) if (std::memcmp(&a[i], &b[i], 4)) { ++diff; worst = std::max(worst, double(std::fabs(a[i] - b[i]))); }
    for (void * p : {(void *) dk, (void *) dv, (void *) dq, (void *) dp, (void *) dvalid, (void *) ref, (void *) out, (void *) sc, (void *) inv}) cudaFree(p);
    std::printf("tokens %d width %d heads %d/%d dim %d: %zu of %zu outputs differ (max %.3g)\n", tokens, width, heads, kv_heads, dim, diff, on, worst);
    return diff == 0;
}
int main() {
    if (std::getenv("QSA_BENCH")) return run(1024, 2051, 14000, 24, 2, 256) ? 0 : 1;
    bool ok = run(64, 2051, 4096, 24, 2, 256);
    ok = run(33, 2051, 2100, 24, 2, 256) && ok;
    ok = run(40, 700, 900, 24, 2, 256) && ok;   // narrower selection: fewer row blocks
    ok = run(32, 300, 400, 8, 2, 128) && ok;    // other shape: generic kernels
    std::puts(ok ? "qsa_grouped_exact PASS" : "qsa_grouped_exact FAIL");
    return ok ? 0 : 1;
}

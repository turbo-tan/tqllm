// Runtime LoRA kernels (src/formats/weight/lora.cuh) against the CPU reference (lora.hpp): y = base + scale * B (A x) for
// verify widths (1, 8, 16 columns) and a prefill chunk, ranks below / above one kernel pass, scale 0 leaving y
// bit-identical, and A stored in the TQ4 rotated basis (the rows of A rotated like an activation) reading the rotated x.
#include "formats/weight/lora.cuh"
#include "formats/weight/lora.hpp"
#include "formats/weight/quant_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tqllm;
static void check(cudaError_t x) {
    if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x));
}
template <class T> static T * up(const std::vector<T> & h) {
    T * d = nullptr;
    check(cudaMalloc(&d, h.size() * sizeof(T)));
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> down(const T * d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

// one case: max |gpu - ref| relative to max |delta|; false on failure
static bool run(int rows, int cols, int rank, int n, float scale, bool rotate, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    LoraTensor t;
    t.rows = rows, t.cols = cols, t.rank = rank;
    t.a.resize(size_t(rank) * cols), t.b.resize(size_t(rows) * rank);
    for (auto & v : t.a) v = 0.02f * nd(rng);
    for (auto & v : t.b) v = 0.05f * nd(rng);
    const float alpha = 2.f * rank, factor = lora_factor(alpha, rank);  // factor 2
    std::vector<__half> xh(size_t(n) * cols);
    std::vector<float> xf(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = __float2half(nd(rng)), xf[i] = __half2float(xh[i]);
    std::vector<__nv_bfloat16> y0(size_t(n) * rows);
    std::vector<float> ref(y0.size());
    for (size_t i = 0; i < y0.size(); ++i) y0[i] = __float2bfloat16(nd(rng)), ref[i] = __bfloat162float(y0[i]);
    lora_reference(t, factor, scale, xf.data(), ref.data(), n);

    std::vector<float> a = t.a, b = t.b;
    for (auto & v : b) v *= factor;
    __half * dx = up(xh);
    if (rotate) {  // x and the rows of A through the same TQ4 rotation (BF16 in, as the engine's load path does it)
        std::vector<__nv_bfloat16> xb(xh.size()), ab(a.size());
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = __float2bfloat16(xf[i]);
        for (size_t i = 0; i < ab.size(); ++i) ab[i] = __float2bfloat16(a[i] * 256.f);
        __nv_bfloat16 *dxb = up(xb), *dab = up(ab);
        __half * dah = nullptr;
        check(cudaMalloc(&dah, a.size() * 2));
        tqllm_cuda::tq4_rotate_h(dxb, dx, cols, n, nullptr);
        tqllm_cuda::tq4_rotate_h(dab, dah, cols, rank, nullptr);
        const auto ah = down(dah, a.size());
        for (size_t i = 0; i < a.size(); ++i) a[i] = __half2float(ah[i]) / 256.f;
        cudaFree(dxb), cudaFree(dab), cudaFree(dah);
    }
    float *da = up(a), *db = up(b), *dt = nullptr, *ds = up(std::vector<float>{scale});
    check(cudaMalloc(&dt, tqllm_cuda::lora_scratch_floats(n, cols, rank) * 4));
    __nv_bfloat16 * dy = up(y0);
    tqllm_cuda::lora_apply(da, db, rows, cols, rank, dx, dy, n, rows, ds, dt, nullptr);
    check(cudaDeviceSynchronize());
    const auto y = down(dy, y0.size());
    double err = 0, mag = 0;
    bool same = true;
    for (size_t i = 0; i < y.size(); ++i) {
        err = std::max(err, std::fabs(double(__bfloat162float(y[i])) - ref[i]));
        mag = std::max(mag, std::fabs(double(ref[i]) - __bfloat162float(y0[i])));
        same &= __bfloat16_as_ushort(y[i]) == __bfloat16_as_ushort(y0[i]);
    }
    // BF16 output rounding: half an ulp of values up to ~4 (0.016), plus the rotation's FP16 / BF16 rounding of x and A
    const double tol = 0.02 + (rotate ? 0.02 * mag : 1e-3 * mag);
    const bool ok = scale == 0.f ? same : err <= tol;
    std::printf("rows %5d cols %5d rank %2d n %4d scale %4.1f %s: max |delta| %.4f, max err %.5f%s -> %s\n", rows, cols, rank,
                n, double(scale), rotate ? "tq4-rot" : "plain  ", mag, err, scale == 0.f ? (same ? " (bit-identical)" : " (changed)") : "",
                ok ? "ok" : "FAIL");
    cudaFree(dx), cudaFree(da), cudaFree(db), cudaFree(dt), cudaFree(ds), cudaFree(dy);
    return ok;
}

int main() {
    std::mt19937 rng(7);
    int fails = 0;
    for (const int n : {1, 8, 16, 512}) {
        fails += !run(640, 2176, 3, n, 1.f, false, rng);
        fails += !run(640, 2176, 11, n, -0.7f, false, rng);  // two rank passes of the down kernel
    }
    fails += !run(512, 1024, 64, 5, 1.f, false, rng);    // the largest rank
    fails += !run(640, 2176, 3, 16, 0.f, false, rng);    // scale 0: y unchanged
    fails += !run(640, 2176, 3, 2048, 0.f, false, rng);
    for (const int n : {1, 16, 300}) fails += !run(640, 2176, 3, n, 1.f, true, rng);
    std::printf("%s\n", fails ? "FAILED" : "all ok");
    return fails ? 1 : 0;
}

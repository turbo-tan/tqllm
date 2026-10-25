// Hyper-connection kernels with several streams against a scalar reference of the same math: hc_init copies the
// embedding into every stream, hc_norm is an RMS norm per (token, stream) row with that stream's gamma, hc_mix averages
// the streams weighted by sigmoid(logit), hc_combine adds the block output scaled by 2 sigmoid(inject / streams).
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template <typename T> static T * upload(const std::vector<T> & v) {
    T * d; check(cudaMalloc(&d, v.size() * sizeof(T))); check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
static std::vector<float> download(const float * d, size_t n) {
    std::vector<float> v(n); check(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost)); return v;
}
static double worst(const std::vector<float> & got, const std::vector<double> & want) {
    double w = 0;
    for (size_t i = 0; i < got.size(); ++i) w = std::max(w, std::fabs(got[i] - want[i]) / (1.0 + std::fabs(want[i])));  // O(1) inputs
    return w;
}

int main() {
    const int h = 384, c = 4, n = 5; const float eps = 1e-6f;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    auto rand = [&](size_t k) { std::vector<float> v(k); for (auto & x : v) x = nd(rng); return v; };
    const auto emb = rand(size_t(n) * h), gamma = rand(size_t(c) * h), logits = rand(size_t(n) * c * h), block = rand(size_t(n) * h),
               inject = rand(size_t(n) * c), res = rand(size_t(n) * c * h);
    float * d_emb = upload(emb), * d_gamma = upload(gamma), * d_logits = upload(logits), * d_block = upload(block),
          * d_inject = upload(inject), * d_res = upload(res), * d_out;
    check(cudaMalloc(&d_out, size_t(n) * c * h * sizeof(float)));
    int failed = 0;
    auto report = [&](const char * what, double err) { std::printf("%-12s max err %.3g\n", what, err); failed += !(err < 1e-5); };

    tqllm_cuda::hc_init(d_emb, d_out, h, c, n, nullptr);
    std::vector<double> want(size_t(n) * c * h);
    for (int t = 0; t < n; ++t) for (int s = 0; s < c; ++s) for (int j = 0; j < h; ++j) want[(size_t(t) * c + s) * h + j] = emb[size_t(t) * h + j];
    report("init", worst(download(d_out, want.size()), want));

    tqllm_cuda::hc_norm(d_res, d_gamma, d_out, h, c, n, eps, nullptr);
    for (int r = 0; r < n * c; ++r) {
        double ss = 0; for (int j = 0; j < h; ++j) ss += double(res[size_t(r) * h + j]) * res[size_t(r) * h + j];
        const double inv = 1.0 / std::sqrt(ss / h + eps);
        for (int j = 0; j < h; ++j) want[size_t(r) * h + j] = res[size_t(r) * h + j] * inv * gamma[size_t(r % c) * h + j];
    }
    report("norm", worst(download(d_out, want.size()), want));

    tqllm_cuda::hc_mix(d_res, d_logits, d_out, h, c, n, nullptr);
    std::vector<double> mixed(size_t(n) * h);
    for (int t = 0; t < n; ++t) for (int j = 0; j < h; ++j) {
        double sum = 0;
        for (int s = 0; s < c; ++s) { const size_t at = (size_t(t) * c + s) * h + j; sum += res[at] / (1.0 + std::exp(-double(logits[at]))); }
        mixed[size_t(t) * h + j] = sum / c;
    }
    report("mix", worst(download(d_out, mixed.size()), mixed));

    tqllm_cuda::hc_combine(d_res, d_block, d_inject, d_out, h, c, n, nullptr);
    for (int t = 0; t < n; ++t) for (int s = 0; s < c; ++s) {
        const double scatter = 2.0 / (1.0 + std::exp(-double(inject[size_t(t) * c + s]) / c));
        for (int j = 0; j < h; ++j) { const size_t at = (size_t(t) * c + s) * h + j; want[at] = res[at] + block[size_t(t) * h + j] * scatter; }
    }
    report("combine", worst(download(d_out, want.size()), want));

    for (float * p : {d_emb, d_gamma, d_logits, d_block, d_inject, d_res, d_out}) cudaFree(p);
    std::printf(failed ? "FAIL\n" : "PASS\n");
    return failed ? 1 : 0;
}

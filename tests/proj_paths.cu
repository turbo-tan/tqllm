// Dense prefill projection paths (Stage::project) on a Q8_0 [512][2560] matrix (the MTP layer's attn_k shape): the matvec
// path (<= 64 tokens), the u4h-family GEMM (65..255 tokens) and the fp16 cuBLAS GEMM over the dequantised weight (>= 256
// tokens) against a CPU fp64 reference on the same fp16 inputs. Each path's per-row error must stay at fp16/bf16 rounding
// level, and a row's result must not depend on how many tokens the batch holds beyond the path choice.
#include "formats/weight/weight_format.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }

static int run(int rows, int cols) {
    const int nb = cols / 32;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<char> raw(size_t(rows) * nb * 34);
    std::vector<float> wref(size_t(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            char * blk = raw.data() + (size_t(r) * nb + b) * 34;
            const __half d = __float2half(0.0004f * (1.f + 0.25f * std::fabs(nd(rng))));
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) {
                const int q = std::clamp(int(std::lround(40.f * nd(rng))), -127, 127);
                blk[2 + i] = char(q);
                wref[size_t(r) * cols + b * 32 + i] = __half2float(d) * float(q);
            }
        }
    const auto * fmt = tqllm::weight_format(8);
    if (!fmt) { std::printf("FAIL: no Q8_0 format\n"); return 1; }
    std::vector<void *> keep;
    tqllm::Upload up = [&](const void * h, size_t bytes) { void * p; check(cudaMalloc(&p, bytes)); check(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice)); keep.push_back(p); return static_cast<const void *>(p); };
    const tqllm::Mat m = fmt->load(raw, rows, cols, up, nullptr);
    fmt = m.fmt;  // Q8_0 loads into the engine's I8H matrices
    __half * wh; check(cudaMalloc(&wh, size_t(rows) * cols * 2)); fmt->dequant_h(m, wh, nullptr);
    cublasHandle_t blas; cublasCreate(&blas);

    const int nmax = 2220;
    std::vector<__half> x(size_t(nmax) * cols);
    for (auto & v : x) v = __float2half(nd(rng));
    __half * dx; float * dy; check(cudaMalloc(&dx, x.size() * 2)); check(cudaMalloc(&dy, size_t(nmax) * rows * 4));
    check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    const int probe = 40;  // rows checked against the CPU reference (the first ones and the last ones of each batch)
    auto ref_row = [&](int t, std::vector<double> & out) {
        out.assign(rows, 0.0);
        for (int r = 0; r < rows; ++r) { double s = 0; for (int c = 0; c < cols; ++c) s += double(wref[size_t(r) * cols + c]) * double(__half2float(x[size_t(t) * cols + c])); out[r] = s; }
    };
    std::vector<std::vector<double>> ref(nmax);
    int fails = 0;
        for (int n : {48, 65, 172, 255, 256, 2048, 2220}) {
        check(cudaMemset(dy, 0xff, size_t(nmax) * rows * 4));
        const char * path;
        if (n >= 256) {
            path = "cublas-fp16";
            const float alpha = 1.f, beta = 0.f;
            cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols, &alpha, wh, CUDA_R_16F, cols, dx, CUDA_R_16F, cols, &beta, dy, CUDA_R_32F, rows,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        } else if (n > 64 && fmt->gemm_f32(m, dx, dy, n, nullptr)) {
            path = "u4h-gemm";
        } else {
            path = "matvec";
            const int step = fmt->max_matvec_cols();
            for (int b = 0; b < n; b += step) fmt->matvec_f32(m, dx + size_t(b) * cols, dy + size_t(b) * rows, std::min(step, n - b), nullptr);
        }
        check(cudaDeviceSynchronize());
        std::vector<float> y(size_t(n) * rows); check(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0; int worst_t = -1;
        for (int k = 0; k < probe; ++k) {
            const int t = k < probe / 2 ? k : n - probe + k; if (t < 0 || t >= n) continue;
            if (ref[size_t(t)].empty()) ref_row(t, ref[size_t(t)]);
            double e2 = 0, r2 = 0;
            for (int r = 0; r < rows; ++r) { const double d = double(y[size_t(t) * rows + r]) - ref[size_t(t)][r]; e2 += d * d; r2 += ref[size_t(t)][r] * ref[size_t(t)][r]; }
            const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
            if (rel > worst) { worst = rel; worst_t = t; }
        }
        const bool ok = worst < 4e-3;  // fp16 activations + bf16/fp32 outputs: ~1e-3; a wrong tail or tile shows far above
        std::printf("[%5d x %5d] n=%-5d %-12s worst row rms rel err %.2e (token %d): %s\n", rows, cols, n, path, worst, worst_t, ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    cublasDestroy(blas);
    for (void * p : keep) cudaFree(p);
    cudaFree(wh), cudaFree(dx), cudaFree(dy);
    return fails;
}

int main() {
    int fails = 0;
    // MTP / attention layer shapes: attn_k / attn_v, hc_attn_down, hc_attn_up, hc_attn_inject, eh_proj
    for (auto rc : {std::pair{512, 2560}, {320, 10240}, {10240, 320}, {4, 10240}, {2560, 5120}}) fails += run(rc.first, rc.second);
    return fails ? 1 : 0;
}

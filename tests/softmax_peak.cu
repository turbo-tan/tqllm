// fn_softmax_peak (the MTP draft confidence) vs a double-precision host reference, and the build's default stream.
#include "qwen4exp_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#error "CUDA builds use the per-thread default stream (no legacy stream): CUDA_API_PER_THREAD_DEFAULT_STREAM must be defined"
#endif

int main() {
    int failures = 0;
    std::mt19937 rng(7);
    for (int n : {1, 31, 1024, 65536, 151936}) for (float spread : {0.5f, 4.f, 40.f}) {
        std::normal_distribution<float> d(0.f, spread);
        std::vector<float> x(n); for (auto & v : x) v = d(rng);
        if (n > 1) x[n / 3] += 3 * spread;  // a clear winner on some rows
        double mx = -INFINITY; for (float v : x) mx = std::max(mx, double(v));
        double s = 0; for (float v : x) s += std::exp(double(v) - mx);
        const double want = 1.0 / s;
        float * dx = nullptr, * dout = nullptr; cudaMalloc(&dx, n * 4); cudaMalloc(&dout, 4);
        cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::fn_softmax_peak(dx, n, dout, nullptr);
        float got = 0; cudaMemcpy(&got, dout, 4, cudaMemcpyDeviceToHost); cudaFree(dx); cudaFree(dout);
        if (cudaGetLastError() != cudaSuccess || !(std::fabs(got - want) <= 1e-5 + 1e-4 * want)) {
            std::printf("FAIL n %d spread %.1f: peak %.8f, reference %.8f\n", n, spread, got, want); ++failures;
        }
    }
    std::printf("softmax peak: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

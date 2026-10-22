// Dense prefill GEMM on the 3090: cuBLAS fp16 with fp32 accumulate (CUBLAS_COMPUTE_32F, the engine today) vs fp16
// accumulate (CUBLAS_COMPUTE_16F) at the Flash-Next dense shapes (M = 2048-token sub-chunk). GeForce Ampere runs fp16
// tensor math at half rate with fp32 accumulate; this prints the measured ratio and the max |diff| of the outputs.
// usage: dense_acc_bench [M=2048]
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CK(x) do { if ((x) != cudaSuccess) { std::fprintf(stderr, "%s:%d cuda error\n", __FILE__, __LINE__); std::exit(1); } } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s:%d cublas error\n", __FILE__, __LINE__); std::exit(1); } } while (0)
int main(int argc, char ** argv) {
    const int M = argc > 1 ? std::atoi(argv[1]) : 2048;
    struct Shape { const char * name; int n, k; } shapes[] = {{"gdn in_proj", 18432, 2560}, {"gdn out", 2560, 6144}, {"attn qkv", 7168, 2560},
                                                               {"attn out", 2560, 6144}, {"shared gate/up", 640, 2560}, {"hc mix", 2560, 10240}};
    cublasHandle_t h; CB(cublasCreate(&h));
    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    for (const auto & s : shapes) {
        std::vector<__half> a(size_t(s.n) * s.k), x(size_t(M) * s.k);
        for (auto & v : a) v = __float2half(nd(rng) * 0.02f); for (auto & v : x) v = __float2half(nd(rng));
        __half * da, * dx, * y16; float * y32;
        CK(cudaMalloc(&da, a.size() * 2)); CK(cudaMalloc(&dx, x.size() * 2)); CK(cudaMalloc(&y32, size_t(M) * s.n * 4)); CK(cudaMalloc(&y16, size_t(M) * s.n * 2));
        CK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        const float af = 1.f, bf = 0.f; const __half ah = __float2half(1.f), bh = __float2half(0.f);
        auto run32 = [&] { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, M, s.k, &af, da, CUDA_R_16F, s.k, dx, CUDA_R_16F, s.k, &bf, y32, CUDA_R_32F, s.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)); };
        auto run16 = [&] { CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.n, M, s.k, &ah, da, CUDA_R_16F, s.k, dx, CUDA_R_16F, s.k, &bh, y16, CUDA_R_16F, s.n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT)); };
        auto time = [&](auto f) { f(); CK(cudaDeviceSynchronize()); cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                                  for (int r = 0; r < 20; ++r) f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / 20; };
        const float t32 = time(run32), t16 = time(run16);
        std::vector<float> o32(size_t(M) * s.n); std::vector<__half> o16(o32.size());
        CK(cudaMemcpy(o32.data(), y32, o32.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(o16.data(), y16, o16.size() * 2, cudaMemcpyDeviceToHost));
        double maxd = 0, maxv = 0; for (size_t i = 0; i < o32.size(); ++i) { maxd = std::max(maxd, double(std::fabs(o32[i] - __half2float(o16[i])))); maxv = std::max(maxv, double(std::fabs(o32[i]))); }
        const double tflop = 2.0 * M * s.n * s.k / 1e12;
        std::printf("%-16s M %d N %5d K %5d: fp32-acc %.3f ms (%.1f TFLOPS)  fp16-acc %.3f ms (%.1f TFLOPS)  %.2fx  max|diff| %.3g (max|y| %.3g)\n",
                    s.name, M, s.n, s.k, t32, tflop / t32 * 1e3, t16, tflop / t16 * 1e3, t32 / t16, maxd, maxv);
        cudaFree(da); cudaFree(dx); cudaFree(y32); cudaFree(y16);
    }
    cublasDestroy(h);
    return 0;
}

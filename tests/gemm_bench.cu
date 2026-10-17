// Prefill GEMM bench: fused u4h-family GEMM (WeightFormat::gemm) vs dequant_h + cuBLAS (the prefill path), per GGUF type.
// Random blocks with sane FP16 scales; error of both vs an FP32-accumulate cuBLAS reference on the dequantised weights.
// usage: gemm-bench [rows 17408] [cols 5120] [iters 10] [type 0 = all]
#include "weight_format.hpp"
#include "gguf.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 17408, cols = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 10;
    const unsigned only = argc > 4 ? unsigned(std::atoi(argv[4])) : 0;
    struct T { unsigned ty; const char * nm; int be, bb, doff, dmoff; };
    const T types[] = {{12, "Q4_K", 256, 144, 0, 2}, {13, "Q5_K", 256, 176, 0, 2}, {14, "Q6_K", 256, 210, 208, -1},
                       {11, "Q3_K", 256, 110, 108, -1}, {21, "IQ3_S", 256, 110, 0, -1}, {18, "IQ3_XXS", 256, 98, 0, -1},
                       {16, "IQ2_XXS", 256, 66, 0, -1}, {17, "IQ2_XS", 256, 74, 0, -1}, {22, "IQ2_S", 256, 82, 0, -1},
                       {19, "IQ1_S", 256, 50, 0, -1}, {10, "Q2_K", 256, 84, 80, 82},
                       {23, "IQ4_XS", 256, 136, 0, -1}, {20, "IQ4_NL", 32, 18, 0, -1}, {8, "Q8_0", 32, 34, 0, -1}};
    std::mt19937 g(7);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd;
    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int ns[] = {64, 128, 344, 1024, 2048};
    const int nmax = 2048;
    __half *x, *wh;
    __nv_bfloat16 *yf;
    __half *yc;
    float * yr;
    cudaMalloc(&x, size_t(nmax) * cols * 2);
    cudaMalloc(&wh, size_t(rows) * cols * 2);
    cudaMalloc(&yf, size_t(nmax) * rows * 2);
    cudaMalloc(&yc, size_t(nmax) * rows * 2);
    cudaMalloc(&yr, size_t(nmax) * rows * 4);
    {
        std::vector<__half> hx(size_t(nmax) * cols);
        for (auto & v : hx) v = __float2half(nd(g));
        cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    }
    std::printf("%-8s %6s %10s %10s %7s %10s %10s\n", "type", "n", "fused_us", "cublas_us", "speedup", "err_fused", "err_cublas");
    for (const auto & t : types) {
        if (only && t.ty != only) continue;
        std::vector<char> raw(size_t(rows) * cols / t.be * t.bb);
        for (size_t b = 0; b < raw.size() / t.bb; ++b) {
            auto * blk = reinterpret_cast<uint8_t *>(raw.data() + b * t.bb);
            for (int i = 0; i < t.bb; ++i) blk[i] = uint8_t(byte(g));
            if (t.ty == 14 || t.ty == 11)
                for (int i = t.ty == 14 ? 192 : 96; i < (t.ty == 14 ? 208 : 108); ++i) blk[i] = uint8_t(byte(g) % 64);
            const __half d = __float2half(0.0004f * (1 + byte(g) % 40));
            std::memcpy(blk + t.doff, &d, 2);
            if (t.dmoff >= 0) std::memcpy(blk + t.dmoff, &d, 2);
        }
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto m = tqllm::weight_format(t.ty)->load(raw, rows, cols, up, nullptr);
        for (int n : ns) {
            m.fmt->dequant_h(m, wh, nullptr);
            const float a32 = 1.f, b32 = 0.f;
            cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols, &a32, wh, CUDA_R_16F, cols, x, CUDA_R_16F, cols, &b32, yr,
                         CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            const __half a16 = __float2half(1.f), b16 = __float2half(0.f);
            auto run_cublas = [&] {
                m.fmt->dequant_h(m, wh, nullptr);
                cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, cols, &a16, wh, CUDA_R_16F, cols, x, CUDA_R_16F, cols, &b16,
                             yc, CUDA_R_16F, rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            };
            auto run_fused = [&] {
                if (!m.fmt->gemm(m, x, yf, n, nullptr)) std::printf("no fused gemm for %s\n", t.nm);
            };
            float us[2];
            for (int which = 0; which < 2; ++which) {
                auto f = [&] { which ? run_cublas() : run_fused(); };
                f();
                cudaEventRecord(e0);
                for (int i = 0; i < iters; ++i) f();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                us[which] = ms * 1000.f / iters;
            }
            std::vector<float> r(size_t(n) * rows);
            std::vector<__nv_bfloat16> hf(r.size());
            std::vector<__half> hc(r.size());
            cudaMemcpy(r.data(), yr, r.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hf.data(), yf, hf.size() * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hc.data(), yc, hc.size() * 2, cudaMemcpyDeviceToHost);
            double se = 0, sf = 0, sc = 0;
            for (size_t i = 0; i < r.size(); ++i) {
                se += double(r[i]) * r[i];
                const double df = __bfloat162float(hf[i]) - r[i], dc = __half2float(hc[i]) - r[i];
                sf += df * df;
                sc += dc * dc;
            }
            std::printf("%-8s %6d %10.1f %10.1f %6.2fx %10.2e %10.2e\n", t.nm, n, us[0], us[1], us[1] / us[0], std::sqrt(sf / se),
                        std::sqrt(sc / se));
        }
        for (auto * p : keep) cudaFree(p);
    }
    return 0;
}

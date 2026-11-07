// fn_gemv_f32_tokens (F32 file weights, Unsloth's ssm_alpha / ssm_beta / hc_*_inject): close to a double reference, and the
// same bits when captured in a CUDA graph on the per-thread stream (the cuBLAS SGEMM it replaces broke the decode graphs)
#include "qwen4exp_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    bool ok = true;
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    const int shapes[][2] = {{48, 2560}, {4, 10240}, {512, 2560}, {7, 1000}};
    for (const auto & sh : shapes) {
        const int rows = sh[0], cols = sh[1], tokens = 8;
        std::vector<float> w(size_t(rows) * cols), x(size_t(cols) * tokens);
        for (auto & v : w) v = 0.05f * nd(rng);
        for (auto & v : x) v = nd(rng);
        float * dw, * dx, * dy, * dg;
        check(cudaMalloc(&dw, w.size() * 4)); check(cudaMalloc(&dx, x.size() * 4));
        check(cudaMalloc(&dy, size_t(rows) * tokens * 4)); check(cudaMalloc(&dg, size_t(rows) * tokens * 4));
        check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        for (int n = 1; n <= tokens; n += 3) {
            tqllm_cuda::fn_gemv_f32_tokens(dw, dx, dy, rows, cols, n, nullptr);
            cudaGraph_t graph; cudaGraphExec_t exec;
            check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            tqllm_cuda::fn_gemv_f32_tokens(dw, dx, dg, rows, cols, n, nullptr);
            check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            check(cudaGraphInstantiate(&exec, graph, 0)); check(cudaGraphLaunch(exec, cudaStreamPerThread)); check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<float> y(size_t(rows) * n), g(y.size());
            check(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(g.data(), dg, g.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0;
            for (int t = 0; t < n; ++t)
                for (int r = 0; r < rows; ++r) {
                    double ref = 0, mag = 0;
                    for (int c = 0; c < cols; ++c) { ref += double(w[size_t(r) * cols + c]) * x[size_t(t) * cols + c]; mag += std::fabs(double(w[size_t(r) * cols + c]) * x[size_t(t) * cols + c]); }
                    worst = std::max(worst, std::fabs(y[size_t(t) * rows + r] - ref) / (mag + 1e-30));
                }
            const bool same = std::memcmp(y.data(), g.data(), y.size() * 4) == 0, close = worst < 1e-6;
            std::printf("%d x %d, tokens %d: graph %s, worst relative error %.2e %s\n", rows, cols, n, same ? "bitwise equal" : "DIFFERS", worst, close ? "ok" : "TOO LARGE");
            ok = ok && same && close;
            {   // strided output (the router's [n][experts + 1] logits): row t at y + t * (rows + 1), same values
                float * dl; check(cudaMalloc(&dl, size_t(rows + 1) * n * 4));
                tqllm_cuda::fn_gemv_f32_tokens_ld(dw, dx, dl, rows, cols, n, rows + 1, nullptr);
                std::vector<float> l(size_t(rows + 1) * n); check(cudaMemcpy(l.data(), dl, l.size() * 4, cudaMemcpyDeviceToHost));
                bool eq = true;
                for (int t = 0; t < n; ++t) eq = eq && std::memcmp(&l[size_t(t) * (rows + 1)], &y[size_t(t) * rows], size_t(rows) * 4) == 0;
                std::printf("   strided output %s\n", eq ? "bitwise equal" : "DIFFERS");
                ok = ok && eq; cudaFree(dl);
            }
            check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
        }
        cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dg);
    }
    std::printf("gemv f32 %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

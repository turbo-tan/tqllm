// fn_gemv_bf16_tokens_ld (router written straight into the [n][experts + 1] logits) must equal fn_gemv_bf16_tokens
// bit for bit on the router shapes (256 x 2560 and 1 x 2560), 1..5 tokens.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    const int experts = 256, cols = 2560; bool ok = true;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> wr(size_t(experts) * cols), ws(cols); std::vector<float> x(size_t(cols) * 5);
    for (auto & v : wr) v = __float2bfloat16_rn(0.05f * nd(rng));
    for (auto & v : ws) v = __float2bfloat16_rn(0.05f * nd(rng));
    for (auto & v : x) v = nd(rng);
    __nv_bfloat16 * dwr, * dws; float * dx, * dr, * ds, * dl;
    check(cudaMalloc(&dwr, wr.size() * 2)); check(cudaMalloc(&dws, ws.size() * 2)); check(cudaMalloc(&dx, x.size() * 4));
    check(cudaMalloc(&dr, size_t(experts) * 5 * 4)); check(cudaMalloc(&ds, 5 * 4)); check(cudaMalloc(&dl, size_t(experts + 1) * 5 * 4));
    check(cudaMemcpy(dwr, wr.data(), wr.size() * 2, cudaMemcpyHostToDevice)); check(cudaMemcpy(dws, ws.data(), ws.size() * 2, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    for (int n = 1; n <= 5; ++n) {
        check(cudaMemset(dl, 0xff, size_t(experts + 1) * 5 * 4));
        tqllm_cuda::fn_gemv_bf16_tokens(dwr, dx, dr, experts, cols, n, nullptr);
        tqllm_cuda::fn_gemv_bf16_tokens(dws, dx, ds, 1, cols, n, nullptr);
        tqllm_cuda::fn_gemv_bf16_tokens_ld(dwr, dx, dl, experts, cols, n, experts + 1, nullptr);
        tqllm_cuda::fn_gemv_bf16_tokens_ld(dws, dx, dl + experts, 1, cols, n, experts + 1, nullptr);
        std::vector<float> r(size_t(experts) * n), s(n), l(size_t(experts + 1) * n);
        check(cudaMemcpy(r.data(), dr, r.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(s.data(), ds, s.size() * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(l.data(), dl, l.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int t = 0; t < n; ++t) {
            bad += std::memcmp(&l[size_t(t) * (experts + 1)], &r[size_t(t) * experts], size_t(experts) * 4) != 0;
            bad += std::memcmp(&l[size_t(t) * (experts + 1) + experts], &s[t], 4) != 0;
        }
        std::printf("router logits strided, tokens %d: %s\n", n, bad ? "DIFFER" : "bitwise equal");
        ok = ok && !bad;
    }
    std::printf("gemv ld %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// hc_combine_norm_rs_bf16 (one pass) must equal hc_combine followed by hc_norm_rs_bf16 bit for bit: the combined rows,
// the row scales and the bf16 image, including the in-place case (out aliases residual) the prefill path uses.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template <typename T> static T * upload(const std::vector<T> & v) { T * d; check(cudaMalloc(&d, v.size() * sizeof(T))); check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <typename T> static std::vector<T> download(const T * d, size_t n) { std::vector<T> v(n); check(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
int main() {
    const int h = 2560, c = 4; bool ok = true;
    for (int n : {1, 37, 300}) for (bool in_place : {false, true}) {
        std::mt19937 rng(n * 2 + in_place); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> residual(size_t(n) * c * h), block(size_t(n) * h), inject(size_t(n) * c), gamma(size_t(c) * h);
        for (auto & v : residual) v = 3.f * nd(rng); for (auto & v : block) v = nd(rng); for (auto & v : inject) v = 2.f * nd(rng); for (auto & v : gamma) v = 1.f + 0.1f * nd(rng);
        float * dres = upload(residual), * dres2 = upload(residual), * dblock = upload(block), * dinject = upload(inject), * dgamma = upload(gamma);
        float * out_a, * out_b, * rs_a, * rs_b; __nv_bfloat16 * img_a, * img_b;
        check(cudaMalloc(&out_a, residual.size() * 4)); check(cudaMalloc(&out_b, residual.size() * 4));
        check(cudaMalloc(&rs_a, size_t(n) * c * 4)); check(cudaMalloc(&rs_b, size_t(n) * c * 4));
        check(cudaMalloc(&img_a, residual.size() * 2)); check(cudaMalloc(&img_b, residual.size() * 2));
        float * oa = in_place ? dres : out_a, * ob = in_place ? dres2 : out_b;
        tqllm_cuda::hc_combine(dres, dblock, dinject, oa, h, c, n, nullptr);
        tqllm_cuda::hc_norm_rs_bf16(oa, dgamma, rs_a, img_a, h, c, n, 1e-6f, nullptr);
        tqllm_cuda::hc_combine_norm_rs_bf16(dres2, dblock, dinject, ob, dgamma, rs_b, img_b, h, c, n, 1e-6f, nullptr);
        check(cudaDeviceSynchronize());
        const auto A = download(oa, residual.size()), B = download(ob, residual.size()), ra = download(rs_a, size_t(n) * c), rb = download(rs_b, size_t(n) * c);
        const auto ia = download(img_a, residual.size()), ib = download(img_b, residual.size());
        const bool same = std::memcmp(A.data(), B.data(), A.size() * 4) == 0 && std::memcmp(ra.data(), rb.data(), ra.size() * 4) == 0 &&
                          std::memcmp(ia.data(), ib.data(), ia.size() * 2) == 0;
        if (!same) std::printf("n %d in_place %d: differs\n", n, int(in_place));
        ok = ok && same;
        cudaFree(dres); cudaFree(dres2); cudaFree(dblock); cudaFree(dinject); cudaFree(dgamma); cudaFree(out_a); cudaFree(out_b); cudaFree(rs_a); cudaFree(rs_b); cudaFree(img_a); cudaFree(img_b);
    }
    std::printf(ok ? "HC_COMBINE_NORM PASS\n" : "HC_COMBINE_NORM FAIL\n");
    return ok ? 0 : 1;
}

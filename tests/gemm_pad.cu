// Prefill GEMM padding: rows [n, padded) of the GEMM input are zeroed before the FP8 activation quantisation reads them,
// so no padding row carries stale data (NaN here) into the codes or scales; the real rows are untouched.
#include "src/gemm_pad.hpp"
#include "fp8_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int cols = 256, cap = 512;
    int fails = 0;
    for (int n : {1, 13, 26, 63}) {
        const int padded = tqllm::gemm_pad_rows(n, cap);
        if (padded % 8 || padded < n || padded - n >= 8) { std::printf("FAIL n=%d: padded rows %d\n", n, padded); ++fails; continue; }
        std::vector<__half> h(size_t(padded) * cols, __ushort_as_half(0x7E00));  // stale data: NaN everywhere
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < cols; ++c) h[size_t(r) * cols + c] = __float2half(float((r * 7 + c) % 23 - 11) / 8.f);
        __half * x = nullptr;
        uint8_t * q = nullptr;
        float * sc = nullptr;
        cudaMalloc(&x, h.size() * 2);
        cudaMalloc(&q, size_t(padded) * cols);
        cudaMalloc(&sc, size_t(padded) * 4);
        cudaMemcpy(x, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
#ifndef GEMM_PAD_SKIP_ZERO  // build with -DGEMM_PAD_SKIP_ZERO to see the behaviour before the fix (fails)
        tqllm::zero_gemm_padding(x, n, padded, cols, nullptr);
#endif
        tqllm_cuda::fp8_quant_rows(x, padded, cols, cols, q, sc, nullptr);
        cudaDeviceSynchronize();
        std::vector<uint8_t> hq(size_t(padded) * cols);
        std::vector<float> hs(padded);
        std::vector<__half> hx(h.size());
        cudaMemcpy(hq.data(), q, hq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(hs.data(), sc, hs.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hx.data(), x, hx.size() * 2, cudaMemcpyDeviceToHost);
        bool ok = std::memcmp(hx.data(), h.data(), size_t(n) * cols * 2) == 0;  // real rows untouched
        for (int r = n; r < padded; ++r) {
            ok &= std::isfinite(hs[r]);
            for (int c = 0; c < cols; ++c) ok &= hq[size_t(r) * cols + c] == 0 && __half2float(hx[size_t(r) * cols + c]) == 0.f;
        }
        for (int r = 0; r < n; ++r) ok &= std::isfinite(hs[r]) && hs[r] > 0.f;
        std::printf("n=%-3d padded=%-3d padding zeroed, codes 0, scales finite, real rows unchanged: %s\n", n, padded, ok ? "PASS" : "FAIL");
        fails += !ok;
        cudaFree(x), cudaFree(q), cudaFree(sc);
    }
    if (cudaGetLastError() != cudaSuccess) { std::printf("FAIL: CUDA error\n"); return 1; }
    return fails ? 1 : 0;
}

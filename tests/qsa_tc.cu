// QSA prefill attention on the tensor cores against the grouped kernels at the Flash-Next verify shape
// (3 rows, 24 heads, 2 kv heads, dim 256, 2048+ selected positions, some invalid / out of range). Outputs must be
// at the Flash-Next prefill shape; max relative error <= 2e-2 (bf16 Q and P). Prints the time per call of both.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
int main() {
    const int heads = 24, kvh = 2, dim = 256, cells = 32768;  // 12 query heads per kv head
    bool ok = true;
    for (int width : {2048, 2560}) for (int tokens : {64, 256}) {
        std::mt19937 rng(width + tokens);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<__nv_bfloat16> k(size_t(cells) * kvh * dim), v(k.size());
        for (auto & x : k) x = __float2bfloat16(nd(rng)); for (auto & x : v) x = __float2bfloat16(nd(rng));
        std::vector<float> q(size_t(tokens) * heads * dim); for (auto & x : q) x = 0.2f * nd(rng);
        std::vector<int32_t> pos(size_t(tokens) * width); std::vector<uint8_t> val(pos.size());
        std::uniform_int_distribution<int> pp(-4, cells + 4), vv(0, 9);
        for (size_t i = 0; i < pos.size(); ++i) { pos[i] = pp(rng); val[i] = vv(rng) != 0; }
        __nv_bfloat16 * dk, * dv; float * dq, * dout; int32_t * dp; uint8_t * dval;
        cudaMalloc(&dk, k.size() * 2); cudaMalloc(&dv, v.size() * 2); cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dout, q.size() * 4);
        cudaMalloc(&dp, pos.size() * 4); cudaMalloc(&dval, val.size());
        cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dp, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dval, val.data(), val.size(), cudaMemcpyHostToDevice);
        tqllm_cuda::QsaDeviceSelection sel; sel.positions = dp; sel.valid = dval; sel.width = width;
        std::vector<float> out[2]; float us[2]; float * dsc, * dinv; cudaMalloc(&dsc, size_t(tokens) * heads * width * 4); cudaMalloc(&dinv, size_t(tokens) * heads * 4);
        for (int mode = 0; mode < 2; ++mode) {
            tqllm_cuda::qsa_set_tc(mode);
            tqllm_cuda::qsa_attention_grouped(dq, dk, dv, sel, dsc, dinv, dout, tokens, cells - tokens, heads, kvh, dim, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("CUDA error\n"); return 1; }
            out[mode].resize(q.size()); cudaMemcpy(out[mode].data(), dout, q.size() * 4, cudaMemcpyDeviceToHost);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
            for (int r = 0; r < 50; ++r) tqllm_cuda::qsa_attention_grouped(dq, dk, dv, sel, dsc, dinv, dout, tokens, cells - tokens, heads, kvh, dim, nullptr);
            cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1); us[mode] = 1000.f * ms / 50;
        }
        double num = 0, den = 0;
        for (size_t i = 0; i < q.size(); ++i) { num = std::max(num, double(std::fabs(out[1][i] - out[0][i]))); den = std::max(den, double(std::fabs(out[0][i]))); }
        const double rel = num / std::max(den, 1e-30);
        ok = ok && rel < 2e-2;
        std::printf("QSA prefill width %d tokens %d: grouped %.1f us, tensor-core %.1f us (%.2fx), max rel err %.2e\n",
                    width, tokens, us[0], us[1], us[0] / us[1], rel);
        cudaFree(dsc); cudaFree(dinv);
        cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dout); cudaFree(dp); cudaFree(dval);
    }
    std::printf(ok ? "QSA_TC PASS\n" : "QSA_TC FAIL\n");
    return ok ? 0 : 1;
}

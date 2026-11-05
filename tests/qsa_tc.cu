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
        std::vector<float> out[4]; float us[4]; float * dsc, * dinv; cudaMalloc(&dsc, size_t(tokens) * heads * width * 4); cudaMalloc(&dinv, size_t(tokens) * heads * 4);
        for (int mode = 0; mode < 4; ++mode) {
            tqllm_cuda::qsa_set_tc(mode);  // 0 grouped, 1 tensor cores 4 warps, 2 tensor cores 8 warps
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
        const bool same = std::memcmp(out[1].data(), out[2].data(), q.size() * 4) == 0;  // 8 warps: the same values as 4
        const bool pipe = std::memcmp(out[2].data(), out[3].data(), q.size() * 4) == 0;  // overlapped gathers: the same values
        ok = ok && same && pipe;
        std::printf("QSA prefill width %d tokens %d: overlapped gathers %.1f us vs %.1f us (%.2fx), %s\n", width, tokens, us[3], us[2], us[2] / us[3], pipe ? "bit-identical" : "DIFFERENT");
        std::printf("QSA prefill width %d tokens %d: tensor-core 4 warps %.1f us, 8 warps %.1f us (%.2fx), %s\n", width, tokens, us[1], us[2], us[1] / us[2], same ? "bit-identical" : "DIFFERENT");
        std::printf("QSA prefill width %d tokens %d: grouped %.1f us, tensor-core %.1f us (%.2fx), max rel err %.2e\n",
                    width, tokens, us[0], us[1], us[0] / us[1], rel);
        cudaFree(dsc); cudaFree(dinv);
        cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dout); cudaFree(dp); cudaFree(dval);
    }
    // dense mode (no selection lists): bit-identical to the same kernel on an explicit identity selection with the causal
    // cut as invalid rows, and within bf16 error of the split-K dense kernel it replaces in prefill
    for (int pos : {0, 1000}) for (int tokens : {64, 1048}) {
        const int width = pos + tokens, kcells = width;
        std::mt19937 rng(pos + tokens + 7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<__nv_bfloat16> k(size_t(kcells) * kvh * dim), v(k.size());
        for (auto & x : k) x = __float2bfloat16(nd(rng)); for (auto & x : v) x = __float2bfloat16(nd(rng));
        std::vector<float> q(size_t(tokens) * heads * dim); for (auto & x : q) x = 0.2f * nd(rng);
        std::vector<int32_t> ip(size_t(tokens) * width); std::vector<uint8_t> iv(ip.size());
        for (int t = 0; t < tokens; ++t) for (int i = 0; i < width; ++i) { ip[size_t(t) * width + i] = i; iv[size_t(t) * width + i] = i <= pos + t; }
        __nv_bfloat16 * dk, * dv; float * dq, * dout; int32_t * dp; uint8_t * dval; float * dsc, * dinv;
        cudaMalloc(&dk, k.size() * 2); cudaMalloc(&dv, v.size() * 2); cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dout, q.size() * 4);
        cudaMalloc(&dp, ip.size() * 4); cudaMalloc(&dval, iv.size());
        cudaMalloc(&dsc, size_t(tokens) * heads * width * 4); cudaMalloc(&dinv, size_t(tokens) * heads * 4);
        cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dp, ip.data(), ip.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dval, iv.data(), iv.size(), cudaMemcpyHostToDevice);
        tqllm_cuda::qsa_set_tc(3);
        std::vector<float> dense3(q.size());
        tqllm_cuda::qsa_attention_dense_tc(dq, dk, dv, dout, tokens, pos + 1, heads, kvh, dim, nullptr);
        cudaDeviceSynchronize(); cudaMemcpy(dense3.data(), dout, q.size() * 4, cudaMemcpyDeviceToHost);
        tqllm_cuda::qsa_set_tc(2);
        std::vector<float> dense(q.size()), ident(q.size()), split(q.size());
        float us_dense = 0, us_split = 0;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const bool took = tqllm_cuda::qsa_attention_dense_tc(dq, dk, dv, dout, tokens, pos + 1, heads, kvh, dim, nullptr);
        cudaDeviceSynchronize(); cudaMemcpy(dense.data(), dout, q.size() * 4, cudaMemcpyDeviceToHost);
        cudaEventRecord(e0); for (int r = 0; r < 20; ++r) tqllm_cuda::qsa_attention_dense_tc(dq, dk, dv, dout, tokens, pos + 1, heads, kvh, dim, nullptr);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&us_dense, e0, e1); us_dense *= 1000.f / 20;
        tqllm_cuda::QsaDeviceSelection sel; sel.positions = dp; sel.valid = dval; sel.width = width;
        tqllm_cuda::qsa_attention_grouped(dq, dk, dv, sel, dsc, dinv, dout, tokens, pos + 1, heads, kvh, dim, nullptr);
        cudaDeviceSynchronize(); cudaMemcpy(ident.data(), dout, q.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<float> ws(tqllm_cuda::fn_dense_attention_workspace(heads, dim, tokens, pos)); float * dws; cudaMalloc(&dws, ws.size() * 4);
        tqllm_cuda::fn_dense_attention(dq, dk, dv, dout, heads, kvh, dim, tokens, pos, nullptr, dws);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("CUDA error\n"); return 1; }
        cudaMemcpy(split.data(), dout, q.size() * 4, cudaMemcpyDeviceToHost);
        cudaEventRecord(e0); for (int r = 0; r < 20; ++r) tqllm_cuda::fn_dense_attention(dq, dk, dv, dout, heads, kvh, dim, tokens, pos, nullptr, dws);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&us_split, e0, e1); us_split *= 1000.f / 20;
        double num = 0, den = 0;
        for (size_t i = 0; i < q.size(); ++i) { num = std::max(num, double(std::fabs(dense[i] - split[i]))); den = std::max(den, double(std::fabs(split[i]))); }
        const double rel = num / std::max(den, 1e-30);
        const bool same = took && std::memcmp(dense.data(), ident.data(), q.size() * 4) == 0 && std::memcmp(dense.data(), dense3.data(), q.size() * 4) == 0;
        ok = ok && same && rel < 2e-2;
        std::printf("QSA dense pos %d tokens %d: tensor-core %.1f us, split-K %.1f us (%.2fx), max rel err %.2e, identity selection %s\n",
                    pos, tokens, us_dense, us_split, us_split / us_dense, rel, same ? "bit-identical" : "DIFFERENT");
        cudaFree(dws); cudaFree(dsc); cudaFree(dinv); cudaFree(dk); cudaFree(dv); cudaFree(dq); cudaFree(dout); cudaFree(dp); cudaFree(dval);
    }
    tqllm_cuda::qsa_set_tc(3);
    std::printf(ok ? "QSA_TC PASS\n" : "QSA_TC FAIL\n");
    return ok ? 0 : 1;
}

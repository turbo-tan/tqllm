// Split decode QSA attention, grouped (K / V rows read once per kv head for 4 query heads) vs per query head
// (attention_split): bitwise equal outputs at the Flash-Next shape (24 heads, 2 kv heads, dim 256) for 1-8 rows, the
// selection widths a decode meets (2051 = top-k 2048 + ratio 4 - 1, shorter, the 8192 cap) and 8 / 16 slices, with
// invalid rows, rows past the query's context and negative positions. QSA_BENCH=1 prints the time per call of both.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
static bool ok_cuda(cudaError_t e) { if (e != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(e)); return false; } return true; }
int main() {
    const int heads = 24, kvh = 2, dim = 256, cells = 32768;
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__nv_bfloat16> k(size_t(cells) * kvh * dim), v(k.size());
    for (auto & x : k) x = __float2bfloat16(nd(rng));
    for (auto & x : v) x = __float2bfloat16(nd(rng));
    __nv_bfloat16 * dk, * dv; cudaMalloc(&dk, k.size() * 2); cudaMalloc(&dv, v.size() * 2);
    cudaMemcpy(dk, k.data(), k.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dv, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
    bool ok = true;
    for (int splits : {16, 8}) for (int width : {2051, 2048, 1337, 512, 8192}) for (int tokens : {1, 2, 3, 5, 8}) {
        std::vector<float> q(size_t(tokens) * heads * dim); for (auto & x : q) x = 0.2f * nd(rng);
        std::vector<int32_t> pos(size_t(tokens) * width); std::vector<uint8_t> val(pos.size());
        for (int t = 0; t < tokens; ++t)
            for (int r = 0; r < width; r += 4) {  // the selection's 4-position blocks
                const int b = int(rng() % unsigned(cells / 4));
                for (int j = 0; j < 4 && r + j < width; ++j) {
                    const size_t i = size_t(t) * width + r + j; const unsigned u = rng() % 100;
                    pos[i] = 4 * b + j; val[i] = 1;
                    if (u < 4) val[i] = 0;
                    else if (u < 6) pos[i] = cells - tokens + t + int(rng() % 3);  // valid, at or past the query's context end
                    else if (u < 7) pos[i] = -1;
                }
            }
        float * dq, * da, * db; int32_t * dp; uint8_t * dval;
        cudaMalloc(&dq, q.size() * 4); cudaMalloc(&da, q.size() * 4); cudaMalloc(&db, q.size() * 4);
        cudaMalloc(&dp, pos.size() * 4); cudaMalloc(&dval, val.size());
        cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dp, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dval, val.data(), val.size(), cudaMemcpyHostToDevice);
        tqllm_cuda::QsaDeviceSelection sel; sel.positions = dp; sel.valid = dval; sel.width = width;
        const int first = cells - tokens;
        tqllm_cuda::qsa_set_split(splits);
        float us[2] = {0, 0};
        for (int mode = 0; mode < 2; ++mode) {
            tqllm_cuda::qsa_set_split_grouped(mode == 1);
            float * out = mode ? db : da;
            tqllm_cuda::qsa_attention_direct(dq, dk, dv, sel, out, tokens, first, heads, kvh, dim, nullptr);
            if (!ok_cuda(cudaDeviceSynchronize())) return 1;
            if (std::getenv("QSA_BENCH")) {
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                for (int r = 0; r < 50; ++r) tqllm_cuda::qsa_attention_direct(dq, dk, dv, sel, out, tokens, first, heads, kvh, dim, nullptr);
                cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1); us[mode] = 1000.f * ms / 50;
            }
        }
        std::vector<float> a(q.size()), b(q.size());
        cudaMemcpy(a.data(), da, a.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), db, b.size() * 4, cudaMemcpyDeviceToHost);
        size_t diff = 0; for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
        ok = ok && diff == 0;
        std::printf("splits %2d width %4d rows %d: %zu of %zu outputs differ", splits, width, tokens, diff, a.size());
        if (us[0] > 0) std::printf("; per head %.1f us, grouped %.1f us (%.2fx)", us[0], us[1], us[0] / us[1]);
        std::printf("\n");
        cudaFree(dq); cudaFree(da); cudaFree(db); cudaFree(dp); cudaFree(dval);
    }
    tqllm_cuda::qsa_set_split(-1); tqllm_cuda::qsa_set_split_grouped(true);
    cudaFree(dk); cudaFree(dv);
    std::puts(ok ? "QSA_SPLIT_GROUPED PASS" : "QSA_SPLIT_GROUPED FAIL");
    return ok ? 0 : 1;
}

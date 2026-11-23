// Decode norm / RoPE from the position table (fn_rope_positions) vs the per-element FP64 cosine / sine: bitwise equal for
// the Flash-Next decode calls (attention q 24 x 256 interleaved with its gate, k 2 x 256, indexer q 4 x 128, rotary 64,
// theta 1e7; another rotary / theta) at 1-8 rows (and 63) and positions up to past the table's first size (growth), and for
// the QSA indexer's key blocks (qsa_norm_rope_blocks). QSA_BENCH=1 prints the time per decode call of both.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
static bool same(const float * a, const float * b, size_t n, size_t & diff) {
    std::vector<float> x(n), y(n);
    cudaMemcpy(x.data(), a, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), b, n * 4, cudaMemcpyDeviceToHost);
    diff = 0; for (size_t i = 0; i < n; ++i) diff += std::memcmp(&x[i], &y[i], 4) != 0;
    return diff == 0;
}
int main() {
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    auto upload = [&](size_t n, float scale) { std::vector<float> h(n); for (auto & v : h) v = scale * nd(rng); float * d; cudaMalloc(&d, n * 4); cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice); return d; };
    bool ok = true;
    struct Shape { int heads, dim, rot; float theta; bool inter; };
    const Shape shapes[] = {{24, 256, 64, 1e7f, true}, {2, 256, 64, 1e7f, false}, {4, 128, 64, 1e7f, false}, {8, 256, 128, 1e6f, false}};
    for (const auto & sh : shapes) for (int n : {1, 2, 3, 5, 8, 63}) for (int pos : {0, 5, 2047, 32374, 131069, 250001}) {
        const size_t in_count = size_t(n) * sh.heads * sh.dim * (sh.inter ? 2 : 1), out_count = size_t(n) * sh.heads * sh.dim;
        float * x = upload(in_count, 1.f), * w = upload(sh.dim, 1.f), * a, * b;
        cudaMalloc(&a, out_count * 4); cudaMalloc(&b, out_count * 4);
        tqllm_cuda::fn_set_rope_positions(false);
        tqllm_cuda::fn_norm_rope(x, w, a, sh.heads, sh.dim, n, pos, sh.rot, sh.theta, 1e-6f, sh.inter, nullptr);
        tqllm_cuda::fn_set_rope_positions(true);
        tqllm_cuda::fn_norm_rope(x, w, b, sh.heads, sh.dim, n, pos, sh.rot, sh.theta, 1e-6f, sh.inter, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("CUDA error"); return 1; }
        size_t diff = 0;
        if (!same(a, b, out_count, diff)) { ok = false; std::printf("FAIL heads %d dim %d rot %d rows %d position %d: %zu of %zu differ\n", sh.heads, sh.dim, sh.rot, n, pos, diff, out_count); }
        if (std::getenv("QSA_BENCH") && n == 1 && pos == 32374) {
            float us[2];
            for (int m = 0; m < 2; ++m) {
                tqllm_cuda::fn_set_rope_positions(m == 1);
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                for (int r = 0; r < 100; ++r) tqllm_cuda::fn_norm_rope(x, w, b, sh.heads, sh.dim, n, pos, sh.rot, sh.theta, 1e-6f, sh.inter, nullptr);
                cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1); us[m] = 10.f * ms;
            }
            std::printf("heads %d dim %d rows 1: per element %.1f us, table %.1f us\n", sh.heads, sh.dim, us[0], us[1]);
        }
        cudaFree(x); cudaFree(w); cudaFree(a); cudaFree(b);
    }
    for (int first : {0, 3, 8093, 40000, 62499}) for (int blocks : {1, 3, 64}) {  // indexer keys: block b sits at position 4 b
        const int dim = 128, ratio = 4, total = first + blocks;
        float * pooled = upload(size_t(total) * dim, 1.f), * w = upload(dim, 1.f), * a, * b;
        cudaMalloc(&a, size_t(total) * dim * 4); cudaMalloc(&b, size_t(total) * dim * 4);
        cudaMemset(a, 0, size_t(total) * dim * 4); cudaMemset(b, 0, size_t(total) * dim * 4);
        tqllm_cuda::fn_set_rope_positions(false);
        tqllm_cuda::qsa_norm_rope_blocks(pooled, w, a, first, blocks, ratio, dim, 64, 1e7f, 1e-6f, nullptr);
        tqllm_cuda::fn_set_rope_positions(true);
        tqllm_cuda::qsa_norm_rope_blocks(pooled, w, b, first, blocks, ratio, dim, 64, 1e7f, 1e-6f, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("CUDA error"); return 1; }
        size_t diff = 0;
        if (!same(a, b, size_t(total) * dim, diff)) { ok = false; std::printf("FAIL indexer blocks %d + %d: %zu differ\n", first, blocks, diff); }
        cudaFree(pooled); cudaFree(w); cudaFree(a); cudaFree(b);
    }
    std::puts(ok ? "NORM_ROPE_POSITIONS PASS" : "NORM_ROPE_POSITIONS FAIL");
    return ok ? 0 : 1;
}

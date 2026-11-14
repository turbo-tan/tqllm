// QsaRawKeyCache::append_multi (several sequences' indexer keys in one copy / pool / norm-RoPE launch each) must leave every
// cache as append + normalize_new per cache does: raw ring, pooled and normalized blocks bit for bit, over rounds of mixed
// block sizes (1..5 tokens), a ring small enough to wrap (with the pinned host mirror), and a speculative truncate.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
int main() {
    const int S = 5, max_cells = 512, ratio = 4, dim = 128, ring = 96, rotary = 64; const float theta = 1e7f, eps = 1e-6f;
    std::vector<std::unique_ptr<tqllm_cuda::QsaRawKeyCache>> a, b;
    for (int i = 0; i < S; ++i) { a.push_back(std::make_unique<tqllm_cuda::QsaRawKeyCache>(max_cells, ratio, dim, ring));
                                  b.push_back(std::make_unique<tqllm_cuda::QsaRawKeyCache>(max_cells, ratio, dim, ring)); }
    std::vector<float> w(dim); for (int d = 0; d < dim; ++d) w[d] = 1.f + .25f * std::sin(float(d));
    float * dw, * dk; cudaMalloc(&dw, dim * 4); cudaMemcpy(dw, w.data(), dim * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dk, size_t(S) * 8 * dim * 4);
    bool ok = true;
    for (int round = 0; round < 40 && ok; ++round) {
        std::vector<int> off, n; int rows = 0;
        for (int i = 0; i < S; ++i) { const int t = 1 + (round * 3 + i * 7) % 5; off.push_back(rows); n.push_back(t); rows += t; }
        std::vector<float> keys(size_t(rows) * dim);
        for (size_t i = 0; i < keys.size(); ++i) keys[i] = std::sin(float(i) * .013f + round) * (1.f + .1f * float(i % 7));
        cudaMemcpy(dk, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice);
        if (round % 9 == 8) for (int i = 0; i < S; ++i) { const int to = std::max(0, a[i]->size() - 3); a[i]->truncate(to); b[i]->truncate(to); }
        for (int i = 0; i < S; ++i) { a[i]->append(dk + size_t(off[i]) * dim, n[i], nullptr); a[i]->normalize_new(dw, rotary, theta, eps, nullptr); }
        std::vector<tqllm_cuda::QsaRawKeyCache *> bs; for (auto & c : b) bs.push_back(c.get());
        tqllm_cuda::QsaRawKeyCache::append_multi(bs, dk, off, n, dw, rotary, theta, eps, nullptr);
        cudaDeviceSynchronize();
        for (int i = 0; i < S; ++i) {
            auto get = [](const float * p, size_t count) { std::vector<float> v(count); cudaMemcpy(v.data(), p, count * 4, cudaMemcpyDeviceToHost); return v; };
            const int cells = std::min(a[i]->size(), a[i]->ring()), blocks = a[i]->complete_blocks();
            if (a[i]->size() != b[i]->size() || a[i]->ring() != b[i]->ring()) { std::printf("FAIL: round %d cache %d sizes differ\n", round, i); ok = false; break; }
            std::vector<float> ra(size_t(cells) * dim), rb(ra.size());
            for (int c = a[i]->size() - cells, j = 0; c < a[i]->size(); ++c, ++j) {  // the ring's newest cells
                cudaMemcpy(ra.data() + size_t(j) * dim, a[i]->raw() + size_t(c % a[i]->ring()) * dim, dim * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(rb.data() + size_t(j) * dim, b[i]->raw() + size_t(c % b[i]->ring()) * dim, dim * 4, cudaMemcpyDeviceToHost);
            }
            const auto pa = get(a[i]->pooled(), size_t(blocks) * dim), pb = get(b[i]->pooled(), pa.size());
            const auto na = get(a[i]->normalized(), size_t(blocks) * dim), nb = get(b[i]->normalized(), na.size());
            if (std::memcmp(ra.data(), rb.data(), ra.size() * 4) || std::memcmp(pa.data(), pb.data(), pa.size() * 4) || std::memcmp(na.data(), nb.data(), na.size() * 4)) {
                std::printf("FAIL: round %d cache %d: raw / pooled / normalized differ from per-cache append\n", round, i); ok = false; break;
            }
        }
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ok = false; }
    std::puts(ok ? "qsa append multi: PASS (one batch for several caches bit-identical to per-cache append + normalize)" : "qsa append multi: FAIL");
    return ok ? 0 : 1;
}

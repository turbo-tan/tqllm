// QSA indexer block scores (qsa_score_pack): the fixed 4-head x 128 kernel vs the generic one, sort keys bitwise equal,
// including blocks not yet complete for a query (key 0), a last group of fewer than 16 blocks, a partial group of
// queries and negative dots.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static bool run(int blocks, int tokens, int first_position, int ratio) {
    constexpr int heads = 4, dim = 128;
    std::mt19937 rng(blocks + tokens * 3 + first_position); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> k(size_t(blocks) * dim), q(size_t(tokens) * heads * dim);
    for (auto & x : k) x = nd(rng);
    for (auto & x : q) x = nd(rng);
    float * dk, * dq; uint64_t * a, * b;
    check(cudaMalloc(&dk, k.size() * 4)); check(cudaMalloc(&dq, q.size() * 4));
    check(cudaMalloc(&a, size_t(tokens) * blocks * 8)); check(cudaMalloc(&b, size_t(tokens) * blocks * 8));
    check(cudaMemcpy(dk, k.data(), k.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemset(a, 0x5a, size_t(tokens) * blocks * 8)); check(cudaMemset(b, 0xa5, size_t(tokens) * blocks * 8));
    tqllm_cuda::qsa_score_pack(dk, dq, a, blocks, first_position, ratio, heads, dim, tokens, nullptr, 0);
    tqllm_cuda::qsa_score_pack(dk, dq, b, blocks, first_position, ratio, heads, dim, tokens, nullptr, 1);
    check(cudaDeviceSynchronize());
    if (std::getenv("QSA_BENCH")) {
        for (int f : {0, 1}) {
            cudaEvent_t e0, e1; check(cudaEventCreate(&e0)); check(cudaEventCreate(&e1)); check(cudaEventRecord(e0));
            for (int i = 0; i < 5; ++i) tqllm_cuda::qsa_score_pack(dk, dq, f ? b : a, blocks, first_position, ratio, heads, dim, tokens, nullptr, f);
            check(cudaEventRecord(e1)); check(cudaEventSynchronize(e1)); float ms = 0; check(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("  %s %.2f ms per call\n", f ? "fixed" : "generic", ms / 5);
        }
    }
    std::vector<uint64_t> ha(size_t(tokens) * blocks), hb(ha.size());
    check(cudaMemcpy(ha.data(), a, ha.size() * 8, cudaMemcpyDeviceToHost)); check(cudaMemcpy(hb.data(), b, hb.size() * 8, cudaMemcpyDeviceToHost));
    size_t diff = 0, zero = 0;
    for (size_t i = 0; i < ha.size(); ++i) { diff += ha[i] != hb[i]; zero += ha[i] == 0; }
    cudaFree(dk); cudaFree(dq); cudaFree(a); cudaFree(b);
    std::printf("blocks %d tokens %d first %d: %zu of %zu keys differ (%zu incomplete)\n", blocks, tokens, first_position, diff, ha.size(), zero);
    return diff == 0;
}
int main() {
    if (std::getenv("QSA_BENCH")) return run(23413, 1024, 92631, 4) ? 0 : 1;
    bool ok = run(23413, 64, 93590, 4);   // 93.7K prompt, last chunk
    ok = run(1000, 64, 3970, 4) && ok;    // early blocks incomplete for the first queries
    ok = run(23413, 37, 93590, 4) && ok;  // a partial group of 16 queries
    ok = run(5000, 100, 1000, 4) && ok;   // a query group that crosses the completion boundary of many block groups
    ok = run(37, 16, 120, 4) && ok;       // last group of 5 blocks
    ok = run(1, 4, 3, 4) && ok;
    std::puts(ok ? "qsa_pack_exact PASS" : "qsa_pack_exact FAIL");
    return ok ? 0 : 1;
}

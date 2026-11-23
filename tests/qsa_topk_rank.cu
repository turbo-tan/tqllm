// QSA indexer top-k on score bits (qsa_topk_bits) against a host full sort of the same keys (score << 32 | ~index,
// descending): the first min(complete blocks, cap) keys in order, for decode / verify rows at 2K-128K positions, with
// many tied scores (coarse values and zeros) and the decode cap (513) and the full cap (1024).
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>
int main() {
    std::mt19937 rng(3); int bad = 0, cases = 0; const int ratio = 4;
    for (int first : {2100, 8190, 32373, 131000}) for (int tokens : {1, 3, 8}) for (int cap : {513, 1024}) for (int coarse : {0, 1}) {
        const int blocks = (first + tokens) / ratio;
        std::vector<uint32_t> sc(size_t(tokens) * blocks);
        std::uniform_real_distribution<float> u(0.f, 4.f);
        for (auto & v : sc) { float f = u(rng); if (coarse) f = float(int(f * 8)) / 8; if (rng() % 10 == 0) f = 0.f; std::memcpy(&v, &f, 4); }
        uint32_t * ds; uint64_t * dout; constexpr int kCap = 1024;
        cudaMalloc(&ds, sc.size() * 4); cudaMalloc(&dout, size_t(tokens) * kCap * 8);
        cudaMemcpy(ds, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice); cudaMemset(dout, 0, size_t(tokens) * kCap * 8);
        tqllm_cuda::qsa_topk_bits(ds, dout, blocks, tokens, first, ratio, cap, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("CUDA error"); return 1; }
        std::vector<uint64_t> out(size_t(tokens) * kCap); cudaMemcpy(out.data(), dout, out.size() * 8, cudaMemcpyDeviceToHost);
        for (int t = 0; t < tokens; ++t) {
            const int complete = (first + t + 1) / ratio, need = std::min(complete, cap);
            std::vector<uint64_t> keys;
            for (int i = 0; i < complete; ++i) keys.push_back((uint64_t(sc[size_t(t) * blocks + i]) << 32) | uint32_t(~uint32_t(i)));
            std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());
            if (std::memcmp(keys.data(), out.data() + size_t(t) * kCap, size_t(need) * 8)) { ++bad; std::printf("FAIL first %d row %d cap %d coarse %d\n", first, t, cap, coarse); }
            ++cases;
        }
        cudaFree(ds); cudaFree(dout);
    }
    std::printf("qsa_topk_rank: %d rows, %d differ\n%s\n", cases, bad, bad ? "QSA_TOPK_RANK FAIL" : "QSA_TOPK_RANK PASS");
    return bad ? 1 : 0;
}

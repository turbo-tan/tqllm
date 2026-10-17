// topk_warp_k must give the same ids, weights and shared weights, bit for bit, as topk_k.
#include "moe_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
int main() {
    std::mt19937 rng(7); int bad = 0, cases = 0;
    for (int e : {512, 384, 1000}) for (int n : {1, 3, 8}) for (int trial = 0; trial < 50; ++trial) {
        const int k = 10; std::vector<float> lg(size_t(n) * (e + 1));
        std::normal_distribution<float> nd(0.f, trial % 3 ? 2.f : 0.05f);
        for (auto & v : lg) v = nd(rng);
        if (trial % 5 == 0) for (int t = 0; t < n; ++t) for (int j = 0; j < e; j += 7) lg[size_t(t) * (e + 1) + j] = 1.5f;  // ties
        float * dl; int * id[2]; float * w[2], * sw[2];
        cudaMalloc(&dl, lg.size() * 4); cudaMemcpy(dl, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice);
        std::vector<int> hi[2]; std::vector<float> hw[2], hs[2];
        for (int v = 0; v < 2; ++v) {
            cudaMalloc(&id[v], n * k * 4); cudaMalloc(&w[v], n * k * 4); cudaMalloc(&sw[v], n * 4);
            tqllm_cuda::moe_topk_variant(dl, id[v], w[v], sw[v], e, k, n, 1.f, v == 1, nullptr);
            hi[v].resize(n * k); hw[v].resize(n * k); hs[v].resize(n);
            cudaMemcpy(hi[v].data(), id[v], n * k * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hw[v].data(), w[v], n * k * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hs[v].data(), sw[v], n * 4, cudaMemcpyDeviceToHost);
            cudaFree(id[v]); cudaFree(w[v]); cudaFree(sw[v]);
        }
        cudaFree(dl); ++cases;
        if (hi[0] != hi[1] || std::memcmp(hw[0].data(), hw[1].data(), hw[0].size() * 4) || std::memcmp(hs[0].data(), hs[1].data(), hs[0].size() * 4)) ++bad;
    }
    if (cudaGetLastError() != cudaSuccess) { std::printf("topk_warp: CUDA error\n"); return 1; }
    std::printf("topk_warp: %d cases, %d differ\n", cases, bad);
    std::printf(bad ? "topk warp FAIL\n" : "topk warp PASS\n"); return bad ? 1 : 0;
}

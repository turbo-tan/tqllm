// moe_topk2 (this layer's and the predicted next layer's routing in one launch) must equal two moe_topk calls bit for
// bit: ids, weights and shared-expert gates, for 256 / 512 / 1024 experts, top-10, 1..8 tokens.
#include "moe_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    bool ok = true; const int k = 10;
    for (int e : {256, 512, 1024}) for (int n = 1; n <= 8; ++n) {  // up to a 6-token verify block and past it
        std::mt19937 rng(e + n); std::normal_distribution<float> nd(0.f, 2.f);
        std::vector<float> l1(size_t(n) * (e + 1)), l2(l1.size());
        for (auto & v : l1) v = nd(rng);
        for (auto & v : l2) v = nd(rng);
        l1[3] = l1[7];  // a tie: lower index wins in both paths
        float * d1, * d2, * w[4], * s[4]; int * id[4];
        check(cudaMalloc(&d1, l1.size() * 4)); check(cudaMalloc(&d2, l2.size() * 4));
        check(cudaMemcpy(d1, l1.data(), l1.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(d2, l2.data(), l2.size() * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < 4; ++i) { check(cudaMalloc(&w[i], size_t(n) * k * 4)); check(cudaMalloc(&s[i], size_t(n) * 4)); check(cudaMalloc(&id[i], size_t(n) * k * 4)); }
        tqllm_cuda::moe_topk(d1, id[0], w[0], s[0], e, k, n, 1.f, nullptr);
        tqllm_cuda::moe_topk(d2, id[1], w[1], s[1], e, k, n, 1.f, nullptr);
        tqllm_cuda::moe_topk2(d1, id[2], w[2], s[2], d2, id[3], w[3], s[3], e, k, n, 1.f, nullptr);
        check(cudaDeviceSynchronize());
        auto get = [&](const void * p, size_t bytes) { std::vector<char> v(bytes); check(cudaMemcpy(v.data(), p, bytes, cudaMemcpyDeviceToHost)); return v; };
        const size_t ib = size_t(n) * k * 4, sb = size_t(n) * 4;
        const bool same = get(id[0], ib) == get(id[2], ib) && get(id[1], ib) == get(id[3], ib) && get(w[0], ib) == get(w[2], ib) &&
                          get(w[1], ib) == get(w[3], ib) && get(s[0], sb) == get(s[2], sb) && get(s[1], sb) == get(s[3], sb);
        std::printf("experts %4d tokens %d: %s\n", e, n, same ? "bitwise equal" : "DIFFER");
        ok = ok && same;
        cudaFree(d1); cudaFree(d2); for (int i = 0; i < 4; ++i) { cudaFree(w[i]); cudaFree(s[i]); cudaFree(id[i]); }
    }
    std::printf("topk2 %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

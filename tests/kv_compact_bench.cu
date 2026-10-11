// Time of one TqCache compaction block at serving size (8K tq3 rows, 4 KV heads, 24 query heads, 256 ring queries) for
// m = nb / 8 and nb / 4 stand-in rows and several NNLS iteration counts. Not a test: prints ms per (layer, block).
#include "qwen35_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace tqllm_cuda;

int main() {
    constexpr int NB = 8192, KVH = 4, H = 24, D = 256, R = 256;
    cudaStream_t st;
    cudaStreamCreate(&st);
    std::mt19937 g(5);
    std::normal_distribution<float> nd;
    const int ts = 2 * KVH * D;
    std::vector<__nv_bfloat16> hkv(size_t(NB) * ts);
    for (auto & x : hkv) x = __float2bfloat16(nd(g));
    __nv_bfloat16 *dkv, *ring;
    Tq3KV256 *kc, *vc;
    int * dpos;
    cudaMalloc(&dkv, hkv.size() * 2);
    cudaMalloc(&kc, size_t(NB) * KVH * sizeof(Tq3KV256));
    cudaMalloc(&vc, size_t(NB) * KVH * sizeof(Tq3KV256));
    cudaMalloc(&dpos, 8);
    cudaMemcpy(dkv, hkv.data(), hkv.size() * 2, cudaMemcpyHostToDevice);
    for (int p = 0; p < NB; p += 512) {
        const int hp[2] = {p, p + 1};
        cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
        cache_write_multi_tq3kv(dkv + size_t(p) * ts, dkv + size_t(p) * ts + KVH * D, ts, 512, kc, vc, dpos, KVH, D, st);
    }
    std::vector<__nv_bfloat16> hr(size_t(R) * H * D);
    for (auto & x : hr) x = __float2bfloat16(nd(g) * 0.5f);
    cudaMalloc(&ring, hr.size() * 2);
    cudaMemcpy(ring, hr.data(), hr.size() * 2, cudaMemcpyHostToDevice);
    KvStandIn256 *ck, *cv;
    float * cb;
    cudaMalloc(&ck, size_t(NB) * KVH * sizeof(KvStandIn256));
    cudaMalloc(&cv, size_t(NB) * KVH * sizeof(KvStandIn256));
    cudaMalloc(&cb, size_t(NB) * KVH * 4);
    cublasHandle_t blas;
    cusolverDnHandle_t sol;
    cublasCreate(&blas);
    cusolverDnCreate(&sol);
    void * scratch;
    cudaMalloc(&scratch, kv_compact_scratch_bytes(NB, R * (H / KVH), NB / 4, KVH));
    for (int rr : {R, 128, 32})
    for (int m : {NB / 8, NB / 4})
        for (int solve : {1, 0}) {
            const int iters = 10;
            auto once = [&] {
                kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, NB, KVH, H, ring, rr, 1.f / 16, m, 0, ck, cv, cb, scratch, blas,
                                 sol, st, iters, 1e-3f, 0, 0, 0, solve);
                cudaStreamSynchronize(st);
            };
            once();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 3; ++i) once();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 3;
            std::printf("kv_compact_bench: ring rows %3d (nq %4d) m %d %s solve: %.1f ms per layer block\n", rr, rr * (H / KVH), m,
                        solve == 1 ? "m x m " : "auto  ", ms);
        }
    return 0;
}

// Bench (not a ctest): TQ3 / TQ4 matvec time by column count (1..16) and row tiles per block (MT 1 / 2 / 4 on the 16-row
// split-K kernel), weights out of L2 as decode reads them; GB/s of weight bytes per cell. Also checks that every MT
// gives the same output as MT 1 (bit-identical by construction).
// usage: test-matvec-cols [reps]
#include "quant_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)

template <class F>
static float cold(F && launch, int reps, void * evict, size_t eb) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float total = 0.f;
    for (int i = 0; i < reps; ++i) {
        cudaMemsetAsync(evict, i & 0xff, eb, 0);
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        total += ms;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return total / float(reps);
}

using namespace tqllm_cuda;

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
    struct Shape { const char * name; int rows, k; };
    // 27B verify shapes: FFN gate+up, FFN down, GDN [qkv|z] mixer
    const Shape shapes[] = {{"ffn_gu", 34816, 5120}, {"ffn_down", 5120, 17408}, {"gdn_mix", 16384, 5120}};
    const int cols[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 16};
    int l2 = 0;
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t eb = std::max<size_t>(size_t(2) * l2, size_t(8) << 20);
    void * evict;
    CK(cudaMalloc(&evict, eb));
    std::mt19937 rng(7);
    int bad = 0;
    std::printf("bits shape rows x k | cols | default | MT1 | MT2 | MT4  (GB/s of weight bytes; us in brackets)\n");
    for (int bits : {3, 4}) {
        for (const auto & sh : shapes) {
            const size_t wbytes = size_t(sh.rows) * sh.k * bits / 8, nsc = size_t(sh.rows) * sh.k / 32;
            std::vector<uint8_t> hw(wbytes);
            for (auto & b : hw) b = uint8_t(rng());
            std::vector<__half> hs(nsc), hx(size_t(16) * sh.k);
            for (auto & v : hs) v = __float2half(0.001f + 0.01f * float(rng() % 100) / 100.f);
            for (auto & v : hx) v = __float2half(float(int(rng() % 200) - 100) / 100.f);
            uint8_t * w;
            __half * sc, * x;
            __nv_bfloat16 * y, * y1;
            CK(cudaMalloc(&w, wbytes));
            CK(cudaMalloc(&sc, nsc * 2));
            CK(cudaMalloc(&x, hx.size() * 2));
            CK(cudaMalloc(&y, size_t(16) * sh.rows * 2));
            CK(cudaMalloc(&y1, size_t(16) * sh.rows * 2));
            CK(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(sc, hs.data(), nsc * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
            const double gb = double(wbytes + size_t(sh.rows) * sh.k / 128 * 2) / 1e9;  // codes + one FP16 scale per 128
            for (const int nc : cols) {
                auto run = [&](int v, __nv_bfloat16 * out) {
                    if (bits == 3) tq4b3_matvec_variant(w, sc, x, out, sh.rows, sh.k, nc, v, 0);
                    else tq4_matvec_variant(w, sc, x, out, sh.rows, sh.k, nc, v, 0);
                };
                std::printf("%d %-8s %5d x %5d | %2d |", bits, sh.name, sh.rows, sh.k, nc);
                run(20, y1);
                CK(cudaDeviceSynchronize());
                for (const int v : {0, 20, 23, 26}) {
                    const float ms = cold([&] { run(v, y); }, reps, evict, eb);
                    CK(cudaGetLastError());
                    std::printf(" %6.0f (%5.1f)", gb / (ms * 1e-3), ms * 1e3);
                    if (v != 0) {
                        CK(cudaDeviceSynchronize());
                        std::vector<uint16_t> a(size_t(nc) * sh.rows), b(a.size());
                        CK(cudaMemcpy(a.data(), y, a.size() * 2, cudaMemcpyDeviceToHost));
                        CK(cudaMemcpy(b.data(), y1, b.size() * 2, cudaMemcpyDeviceToHost));
                        if (std::memcmp(a.data(), b.data(), a.size() * 2)) { std::printf(" !DIFF"); ++bad; }
                    }
                }
                std::printf("\n");
            }
            cudaFree(w); cudaFree(sc); cudaFree(x); cudaFree(y); cudaFree(y1);
        }
    }
    std::printf("matvec_cols: %s\n", bad ? "MT outputs DIFFER" : "all MT outputs identical to MT 1");
    return bad ? 1 : 0;
}

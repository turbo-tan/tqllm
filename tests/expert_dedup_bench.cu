// Bench only: grouped dedup expert matvec at the Flash-Next decode shape (G distinct experts from a 64-slot bank,
// 1-2 token columns each), isolated from copies; prints us and effective GB/s per format.
// usage: expert-dedup-bench [groups=16] [cols_per_group=2] [iters=200]
#include "weight_format.hpp"
#include "quant_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <functional>
int main(int argc, char ** argv) {
    const int G = argc > 1 ? std::atoi(argv[1]) : 16, C = argc > 2 ? std::atoi(argv[2]) : 2, iters = argc > 3 ? std::atoi(argv[3]) : 200;
    const int slots = 64, ntok = 4, DS = tqllm_cuda::kDedupStride;
    std::mt19937 g(7);
    struct Case { unsigned ty; int bb, rows, cols; const char * what; };
    for (const Case cs : {Case{22, 82, 640, 2560, "gate/up IQ2_S"}, Case{18, 98, 640, 2560, "gate/up IQ3_XXS"}, Case{21, 110, 640, 2560, "gate/up IQ3_S"},
                          Case{20, 144, 2560, 640, "down IQ4_NL"}}) {
        std::vector<char> raw(size_t(cs.rows) * slots * cs.cols / 256 * cs.bb);
        for (auto & c : raw) c = char(g() & 255);
        const __half d = __float2half(0.001f);
        for (size_t b = 0; b < raw.size() / cs.bb; ++b) {
            if (cs.ty == 20) for (int q = 0; q < 8; ++q) std::memcpy(raw.data() + b * cs.bb + q * 18, &d, 2);
            else std::memcpy(raw.data() + b * cs.bb, &d, 2);
        }
        const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
        const auto m = tqllm::weight_format(cs.ty)->load(raw, cs.rows * slots, cs.cols, up, nullptr);
        std::vector<__half> x(size_t(ntok) * cs.cols, __float2half(0.01f));
        __half * dx; cudaMalloc(&dx, x.size() * 2); cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        __nv_bfloat16 * dy; cudaMalloc(&dy, size_t(ntok) * 10 * cs.rows * 2);
        std::vector<int> map(size_t(3) * 40 * DS, -1);
        for (int q = 0; q < G; ++q) {
            int * e = map.data() + q * DS; e[0] = (q * 37) % slots; e[1] = C;
            for (int c = 0; c < C; ++c) { e[2 + c] = (q + c) % ntok; e[2 + tqllm_cuda::kDedupCols + c] = (q * C + c) % 40; }
        }
        int * dm; cudaMalloc(&dm, map.size() * 4); cudaMemcpy(dm, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
        auto call = [&] { m.fmt->expert_matvec_dedup(m, dx, dy, cs.rows, dm, G, nullptr); };
        for (int i = 0; i < 20; ++i) call();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) call();
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        const double bytes = double(raw.size()) / slots * G;
        std::printf("dedup %-16s G=%d cols=%d: %7.2f us  %6.0f GB/s\n", cs.what, G, C, 1000.f * ms / iters, bytes * iters / (ms / 1000) / 1e9);
        {  // TQFN_TILED layout: same output bits, timing of the lane-contiguous code tiles
            const int bpb = tqllm_cuda::tiled_block_bytes(m.fmt->u4h_id());
            if (bpb) {
                const size_t bytes = size_t(cs.rows) * slots * (cs.cols / 32) * bpb;
                std::vector<char> h(bytes), t(bytes);
                cudaMemcpy(h.data(), m.p, bytes, cudaMemcpyDeviceToHost);
                tqllm_cuda::tile_expert_rows(h.data(), t.data(), cs.rows * slots, cs.cols, bpb);
                void * dt; cudaMalloc(&dt, bytes); cudaMemcpy(dt, t.data(), bytes, cudaMemcpyHostToDevice);
                tqllm::Mat mt = m; mt.p = dt;
                std::vector<__nv_bfloat16> ref(size_t(ntok) * 10 * cs.rows), got(ref.size());
                cudaMemset(dy, 0, ref.size() * 2); call(); cudaDeviceSynchronize(); cudaMemcpy(ref.data(), dy, ref.size() * 2, cudaMemcpyDeviceToHost);
                tqllm_cuda::set_tiled_experts(true);
                auto callt = [&] { mt.fmt->expert_matvec_dedup(mt, dx, dy, cs.rows, dm, G, nullptr); };
                cudaMemset(dy, 0, got.size() * 2); callt(); cudaDeviceSynchronize(); cudaMemcpy(got.data(), dy, got.size() * 2, cudaMemcpyDeviceToHost);
                for (int i = 0; i < 20; ++i) callt();
                cudaEventRecord(e0); for (int i = 0; i < iters; ++i) callt(); cudaEventRecord(e1); cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                std::printf("   tiled: %7.2f us  bit-identical: %s\n", 1000.f * ms / iters, std::memcmp(ref.data(), got.data(), ref.size() * 2) == 0 ? "yes" : "NO");
                tqllm_cuda::set_tiled_experts(false); cudaFree(dt);
            }
        }
        if (cs.ty != 20) {  // paired gate + up (same matrix twice stands in for up): one launch vs two
            __nv_bfloat16 * dy2; cudaMalloc(&dy2, size_t(ntok) * 10 * cs.rows * 2);
            auto two = [&] { call(); call(); };
            auto pair = [&] { m.fmt->expert_matvec_dedup2(m, m, dx, dy, dy2, cs.rows, dm, G, nullptr); };
            for (auto [nm, fn] : {std::pair<const char *, std::function<void()>>{"two launches", two}, {"paired", pair}}) {
                for (int i = 0; i < 20; ++i) fn();
                cudaEventRecord(e0); for (int i = 0; i < iters; ++i) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1); std::printf("   gate+up %-12s %7.2f us\n", nm, 1000.f * ms / iters);
            }
            pair(); cudaDeviceSynchronize(); std::vector<__nv_bfloat16> a1(size_t(ntok) * 10 * cs.rows), a2(a1.size()), b1(a1.size());
            call(); cudaDeviceSynchronize(); cudaMemcpy(a1.data(), dy, a1.size() * 2, cudaMemcpyDeviceToHost);
            pair(); cudaDeviceSynchronize(); cudaMemcpy(b1.data(), dy, b1.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(a2.data(), dy2, a2.size() * 2, cudaMemcpyDeviceToHost);
            std::printf("   paired bit-identical: %s\n", std::memcmp(a1.data(), b1.data(), a1.size() * 2) == 0 && std::memcmp(a1.data(), a2.data(), a1.size() * 2) == 0 ? "yes" : "NO");
            cudaFree(dy2);
        }
        cudaFree(dx); cudaFree(dy); cudaFree(dm);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

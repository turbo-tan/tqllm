// Bench only: one MoE verify block's expert projections at the Qwen3.6-35B-A3B shape (256 experts, hidden 2048, expert
// FFN 512; gate / up Q4_K, down Q5_K as in UD-Q4_K_M), dedup maps of G distinct experts with C columns each, the expert
// sets rotating so the weights come from DRAM. Times each row tiling (set_dedup_mt) and the prefetching paired launch,
// checks that every variant gives the same bits, prints us and GB/s of weight bytes.
// usage: moe-verify-bench [groups=33] [cols=2] [iters=300]
#include "weight_format.hpp"
#include "quant_kernels.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int G = argc > 1 ? std::atoi(argv[1]) : 33, C = argc > 2 ? std::atoi(argv[2]) : 2, iters = argc > 3 ? std::atoi(argv[3]) : 300;
    const int E = 256, H = 2048, FE = 512, ntok = 8, DS = tqllm_cuda::kDedupStride, R = E / G;
    std::mt19937 g(11);
    const tqllm::Upload up = [](const void * h, size_t n) {
        void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p);
    };
    auto bank = [&](unsigned ty, int bb, int rows, int cols, std::vector<char> & raw) {  // random codes, small fp16 scales
        raw.assign(size_t(rows) * E * cols / 256 * bb, 0);
        for (auto & c : raw) c = char(g() & 255);
        const __half d = __float2half(0.001f);
        for (size_t b = 0; b < raw.size() / bb; ++b) std::memcpy(raw.data() + b * bb, &d, 2), std::memcpy(raw.data() + b * bb + 2, &d, 2);
        return tqllm::weight_format(ty)->load(raw, rows * E, cols, up, nullptr);
    };
    std::vector<char> rg, ru, rd;
    const auto mg = bank(12, 144, FE, H, rg), mu = bank(12, 144, FE, H, ru), md = bank(13, 176, H, FE, rd);
    std::vector<__half> x(size_t(ntok) * H), a(size_t(G) * C * FE);
    for (auto & v : x) v = __float2half(float(int(g() % 200) - 100) / 1000.f);
    for (auto & v : a) v = __float2half(float(int(g() % 200) - 100) / 1000.f);
    __half *dx, *da; cudaMalloc(&dx, x.size() * 2); cudaMalloc(&da, a.size() * 2);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice);
    const size_t outn = size_t(G) * C;
    __nv_bfloat16 *yg, *yu, *yd; cudaMalloc(&yg, outn * FE * 2); cudaMalloc(&yu, outn * FE * 2); cudaMalloc(&yd, outn * H * 2);
    // R maps over disjoint expert sets; group q: expert, C columns (token input for gate/up, assignment input for down)
    std::vector<int> gu(size_t(R) * G * DS, -1), dn(gu.size(), -1);
    for (int r = 0; r < R; ++r)
        for (int q = 0; q < G; ++q) {
            int * e = gu.data() + (size_t(r) * G + q) * DS, * f = dn.data() + (size_t(r) * G + q) * DS;
            e[0] = f[0] = r * G + q; e[1] = f[1] = C;
            for (int c = 0; c < C; ++c) {
                e[2 + c] = (q + c) % ntok, f[2 + c] = q * C + c;
                e[2 + tqllm_cuda::kDedupCols + c] = f[2 + tqllm_cuda::kDedupCols + c] = q * C + c;
            }
        }
    int *dgu, *ddn; cudaMalloc(&dgu, gu.size() * 4); cudaMalloc(&ddn, dn.size() * 4);
    cudaMemcpy(dgu, gu.data(), gu.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(ddn, dn.data(), dn.size() * 4, cudaMemcpyHostToDevice);
    int rot = 0;
    auto pair = [&] { mg.fmt->expert_matvec_dedup2(mg, mu, dx, yg, yu, FE, dgu + size_t(rot++ % R) * G * DS, G, nullptr); };
    auto down = [&] { md.fmt->expert_matvec_dedup(md, da, yd, H, ddn + size_t(rot++ % R) * G * DS, G, nullptr); };
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time = [&](const std::function<void()> & f) {
        for (int i = 0; i < 30; ++i) f();
        cudaEventRecord(e0); for (int i = 0; i < iters; ++i) f(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); return 1000.f * ms / iters;
    };
    auto snap = [&](__nv_bfloat16 * p, size_t n) { std::vector<__nv_bfloat16> v(n); rot = 0; cudaDeviceSynchronize(); cudaMemcpy(v.data(), p, n * 2, cudaMemcpyDeviceToHost); return v; };
    const double gu_bytes = double(rg.size() + ru.size()) / E * G, dn_bytes = double(rd.size()) / E * G;
    const int fg = mg.fmt->u4h_id(), fd = md.fmt->u4h_id();
    std::printf("G=%d cols=%d: gate/up fmt %d %.1f MB, down fmt %d %.1f MB per call\n", G, C, fg, gu_bytes / 1e6, fd, dn_bytes / 1e6);
    std::vector<__nv_bfloat16> ref_g, ref_u, ref_d;
    for (int pm : {1, 2}) for (int pf : {0, 1}) {
        if (pm == 2 && pf) continue;
        tqllm_cuda::set_dedup_mt(fg, pm, 0); tqllm_cuda::set_dedup_pf_min(fg, pf ? 1 : 1 << 30);
        const float us = time(pair);
        rot = 0; pair(); const auto og = snap(yg, outn * FE), ou = snap(yu, outn * FE);
        if (ref_g.empty()) ref_g = og, ref_u = ou;
        const bool same = std::memcmp(og.data(), ref_g.data(), og.size() * 2) == 0 && std::memcmp(ou.data(), ref_u.data(), ou.size() * 2) == 0;
        std::printf("  gate+up MT %d%s: %7.1f us %5.0f GB/s  bits %s\n", pm, pf ? " PF" : "   ", us, gu_bytes / us / 1e3, same ? "same" : "DIFFER");
    }
    tqllm_cuda::set_dedup_pf_min(fg, 0);
    for (int sm : {1, 2, 4}) {
        tqllm_cuda::set_dedup_mt(fd, 0, sm);
        const float us = time(down);
        rot = 0; down(); const auto od = snap(yd, outn * H);
        if (ref_d.empty()) ref_d = od;
        std::printf("  down    MT %d%s: %7.1f us %5.0f GB/s  bits %s\n", sm, "    ", us, dn_bytes / us / 1e3,
                    std::memcmp(od.data(), ref_d.data(), od.size() * 2) == 0 ? "same" : "DIFFER");
    }
    tqllm_cuda::set_dedup_mt(fd, 0, 0);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

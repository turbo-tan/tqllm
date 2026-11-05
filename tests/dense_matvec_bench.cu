// Bench only: matvec_f32 of the Flash-Next dense projections at decode width (1..6 rows: one sequence's verify block)
// for each weight format the files use, as achieved GB/s of weight bytes (the GB10 peaks near 273, a 3090 near 936).
// usage: dense-matvec-bench [iters=200]
#include "formats/weight/weight_format.hpp"
#include "core/gguf.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <vector>
using namespace tqllm;

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    struct Shape { int rows, cols; const char * what; };
    const Shape shapes[] = {{10240, 2560, "GDN qkv"}, {6144, 2560, "GDN gate z"}, {2560, 6144, "GDN out"},
                            {12288, 2560, "attn q"}, {640, 2560, "shared gate/up"}, {2560, 640, "shared down"}};
    const uint32_t types[] = {14 /* Q6_K */, 12 /* Q4_K */, 23 /* IQ4_XS */, 20 /* IQ4_NL */, 3 /* Q4_1 */};
    std::mt19937 rng(1);
    __half * dx; float * dy; cudaMalloc(&dx, size_t(12288) * 32 * 2); cudaMalloc(&dy, size_t(12288) * 32 * 4);
    std::vector<__half> xh(size_t(12288) * 32, __float2half(0.01f)); cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::printf("%-16s %-8s", "shape", "format"); for (int n : {1, 4, 8, 16, 32}) std::printf("  n=%d us   GB/s", n); std::printf("\n");
    for (const auto & sh : shapes)
        for (uint32_t ty : types) {
            const auto * f = weight_format(ty); if (!f) continue;
            // copies of the matrix past the caches (>= 512 MB), cycled, so every launch streams its weights from memory
            std::vector<void *> keep; std::vector<Mat> ms;
            try {
                std::vector<char> raw(gguf_tensor_bytes(ty, {uint64_t(sh.cols), uint64_t(sh.rows)}, "w"));
                std::uniform_int_distribution<int> bd(0, 0x3B); for (auto & c : raw) c = char(bd(rng));
                Upload up = [&](const void * h, size_t n) { void * p; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); keep.push_back(p); return (const void *) p; };
                const int copies = int(std::max<size_t>(2, (size_t(512) << 20) / raw.size()));
                for (int c = 0; c < copies; ++c) ms.push_back(f->load(raw, sh.rows, sh.cols, up, nullptr));
            } catch (const std::exception &) { for (void * p : keep) cudaFree(p); continue; }
            const Mat & m = ms[0];
            const double bytes = double(m.fmt->row_bytes(sh.cols) + m.fmt->aux_row_bytes(sh.cols)) * sh.rows;
            std::printf("%-16s %-8s", sh.what, f->name());
            for (int n : {1, 4, 8, 16, 32}) {
                for (int i = 0; i < 10; ++i) ms[size_t(i) % ms.size()].fmt->matvec_f32(ms[size_t(i) % ms.size()], dx, dy, n, nullptr);
                cudaEventRecord(a);
                for (int i = 0; i < iters; ++i) { const Mat & mi = ms[size_t(i) % ms.size()]; mi.fmt->matvec_f32(mi, dx, dy, n, nullptr); }
                cudaEventRecord(b); cudaEventSynchronize(b);
                float ms = 0; cudaEventElapsedTime(&ms, a, b); const double us = 1e3 * ms / iters;
                std::printf("  %7.1f %6.0f", us, bytes > 0 ? bytes / us / 1e3 : 0.0);
            }
            std::printf("\n");
            for (void * p : keep) cudaFree(p);
        }
    return 0;
}

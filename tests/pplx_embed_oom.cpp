// A failed activation allocation must leave the embedder usable: embed a short document, fill the GPU, ask for a
// document long enough to grow the activation buffers (it must fail with an exception, not a crash), free the memory,
// and the short document must embed again with the same vectors. Needs TQLLM_PPLX_GGUF (any pplx-embed-v2-context
// GGUF); exit 77 = skipped, also on an integrated GPU (GB10), where filling the device fills the host.
#include "pplx_embed.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
tqllm::EmbedInput synthetic(size_t n, uint32_t seed) {
    tqllm::EmbedInput in;
    for (size_t i = 0; i < n; ++i) in.ids.push_back(1000 + uint32_t((i * 2654435761u + seed) % 20000));
    in.spans = {{1, uint32_t(n / 2)}, {uint32_t(n / 2), uint32_t(n)}};
    return in;
}
}  // namespace

int main() {
    const char * gguf = std::getenv("TQLLM_PPLX_GGUF");
    if (!gguf) { std::puts("pplx embed oom: skipped (set TQLLM_PPLX_GGUF)"); return 77; }
    int integrated = 0;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0);
    if (integrated) {  // GB10 / DGX Spark: device memory is the system's, filling it takes the host down with it
        std::puts("pplx embed oom: skipped (integrated GPU: unified memory)");
        return 77;
    }
    auto model = tqllm::PplxEmbedder::load(gguf, 0, tqllm::PplxEmbedder::Mode::fast);
    const std::vector<tqllm::EmbedInput> small{synthetic(300, 1)};
    const auto ref = model->embed_many(small);

    std::vector<void *> hog;  // fill the device in 256 MB blocks, leave about 64 MB
    for (;;) {
        size_t free_b = 0, total = 0;
        cudaMemGetInfo(&free_b, &total);
        if (free_b < (size_t(320) << 20)) break;
        void * p = nullptr;
        if (cudaMalloc(&p, size_t(256) << 20) != cudaSuccess) { cudaGetLastError(); break; }
        hog.push_back(p);
    }
    const size_t big = std::min<size_t>(model->max_tokens(), 60000);
    bool threw = false;
    try { model->embed_many({synthetic(big, 2)}); } catch (const std::exception & e) { threw = true; std::printf("long document refused: %s\n", e.what()); }
    for (void * p : hog) cudaFree(p);
    cudaGetLastError();

    int failures = 0;
    if (!threw) { std::puts("FAIL: the long document did not run out of memory (device too large for this test?)"); ++failures; }
    try {
        const auto again = model->embed_many(small);
        double worst = 1;
        for (size_t i = 0; i < ref[0].size(); i += model->dim()) {
            double d = 0, a = 0, b = 0;
            for (int j = 0; j < model->dim(); ++j) { d += double(ref[0][i + j]) * again[0][i + j]; a += double(ref[0][i + j]) * ref[0][i + j]; b += double(again[0][i + j]) * again[0][i + j]; }
            worst = std::min(worst, d / std::sqrt(a * b));
        }
        std::printf("after the failure: cosine vs before %.6f\n", worst);
        if (!(worst > 0.9999)) { std::puts("FAIL: vectors changed after a failed allocation"); ++failures; }
    } catch (const std::exception & e) {
        std::printf("FAIL: the embedder is broken after a failed allocation: %s\n", e.what());
        ++failures;
    }
    std::puts(failures ? "pplx embed oom: FAILED" : "pplx embed oom: OK");
    return failures ? 1 : 0;
}

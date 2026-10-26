// TQ3_4S grouped experts at the 35B-A3B shape (256 experts + the shared expert appended to the stack for
// decode, expert FFN 512, hidden 2048, top-8). The qwen35 engine only reached this path once its
// grouped-expert guard stopped rejecting non-Plain bases, and both grouped entry points were wrong:
//   * expert_gemm refused the stack ("TQ3_4S: invalid stacked experts") because the engine appends the shared
//     expert to the routed stack, so rows * experts != m.rows;
//   * nothing had ever compared the grouped matvec / GEMM with the dense ones at this shape.
// References: expert_matvec == the dense matvec of the row-sliced expert (bit for bit), and expert_gemm ==
// the dense GEMM of the gathered assignments (bit for bit), as tests/moe_device.cu does for the u4h family.
#include "moe_kernels.cuh"
#include "quant_kernels.cuh"
#include "weight_format.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

static int failures = 0;
static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
static void require(bool ok, const char * why) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", why);
    if (!ok) ++failures;
}
// per-assignment / per-expert checks: only reported when they fail
static void pin(bool ok, const char * why) { if (!ok) { std::printf("FAIL %s\n", why); ++failures; } }

template <class T> struct Device {
    T * p = nullptr;
    Device() = default;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    // a pageable upload may return before its DMA lands, and the build's per-thread default stream does not order it
    // before kernels on the test's own stream
    void put(const std::vector<T> & v) {
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        ck(cudaStreamSynchronize(cudaStreamPerThread));
    }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

int main() {
    try {
        const int E = 256, rows = 512, cols = 2048, topk = 8;
        const int slots = E + 1;  // the shared expert appended to the routed stack (the engine's decode fusion)
        const auto * fmt = tqllm::weight_format(46);  // GGML_TQ3_4S
        require(fmt != nullptr, "TQ3_4S is registered");
        if (!fmt) return 1;

        // raw TQ3_4S blocks: 16 bytes per 32 weights, bytes 0..3 are the four E3M5 scales (kept small and
        // finite), bytes 4..15 the 3-bit codes.
        std::mt19937 gen(350001u);
        std::vector<char> raw(size_t(slots) * rows * (cols / 32) * 16);
        for (size_t b = 0; b < raw.size() / 16; ++b) {
            raw[b * 16 + 0] = char(8 + gen() % 24);
            raw[b * 16 + 1] = char(8 + gen() % 24);
            raw[b * 16 + 2] = char(8 + gen() % 24);
            raw[b * 16 + 3] = char(8 + gen() % 24);
            for (int j = 4; j < 16; ++j) raw[b * 16 + j] = char(gen() % 256);
        }
        std::vector<void *> allocs;
        tqllm::Upload up = [&](const void * h, size_t bytes) {
            void * p = nullptr; ck(cudaMalloc(&p, bytes)); ck(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice)); ck(cudaStreamSynchronize(cudaStreamPerThread));
            allocs.push_back(p); return static_cast<const void *>(p);
        };
        cudaStream_t stream; ck(cudaStreamCreate(&stream));
        const auto m = fmt->load(raw, slots * rows, cols, up, stream);
        ck(cudaStreamSynchronize(stream));
        require(m.fmt == fmt, "the loaded stack keeps the TQ3_4S format");

        // ---- decode: one token, top-8 plus the appended shared slot (ids up to E) ----
        {
            const int n = 1, kv = topk + 1;
            std::vector<__half> x(size_t(n) * cols);
            for (auto & v : x) v = __float2half(float(int(gen() % 201) - 100) / 100.f);
            std::vector<int> ids(size_t(n) * kv);
            for (int j = 0; j < kv; ++j) ids[j] = j == kv - 1 ? E : int(gen() % E);
            Device<__half> dx(x.size()); dx.put(x);
            Device<int> di(ids.size()); di.put(ids);
            Device<__nv_bfloat16> out(size_t(n) * kv * rows), ref(size_t(n) * kv * rows);
            fmt->expert_matvec(m, dx.p, out.p, rows, di.p, n, kv, false, stream);
            for (int a = 0; a < n * kv; ++a) {
                tqllm::Mat v = m;
                v.p = static_cast<const char *>(m.p) + size_t(ids[a]) * rows * fmt->row_bytes(cols);
                v.rows = rows;
                fmt->matvec(v, dx.p, ref.p + size_t(a) * rows, 1, stream);
            }
            ck(cudaStreamSynchronize(stream));
            const auto y = out.get(size_t(n) * kv * rows), yr = ref.get(size_t(n) * kv * rows);
            require(std::memcmp(y.data(), yr.data(), y.size() * 2) == 0,
                    "expert_matvec at the 35B-A3B shape == the dense matvec of each row-sliced expert");
        }

        // ---- verify / prefill: grouped GEMM over the stack that holds one slot more than the map addresses ----
        {
            const int n = 13;  // > the format's 12-column matvec width: the engine's grouped GEMM path
            std::vector<__half> x(size_t(n) * cols);
            for (auto & v : x) v = __float2half(float(int(gen() % 201) - 100) / 100.f);
            std::vector<int> ids(size_t(n) * topk);
            for (auto & v : ids) v = int(gen() % E);
            Device<__half> dx(x.size()); dx.put(x);
            Device<int> di(ids.size()); di.put(ids);
            Device<int> counts(E), groups(size_t(E) * n);
            Device<__nv_bfloat16> out(size_t(n) * topk * rows), ref(size_t(n) * topk * rows);
            tqllm_cuda::moe_group(di.p, counts.p, groups.p, E, n, topk, stream);
            bool threw = false;
            try {
                fmt->expert_gemm(m, dx.p, out.p, rows, counts.p, groups.p, n, E, topk, false, stream);
            } catch (const std::exception & e) {
                threw = true;
                std::printf("      expert_gemm threw: %s\n", e.what());
            }
            require(!threw, "expert_gemm accepts a stack with more slots than the map addresses (crash pin)");
            // the test's copies run on the per-thread default stream (--default-stream per-thread): wait for `stream`
            ck(cudaStreamSynchronize(stream));
            if (!threw) {
                const auto gc = counts.get(E), gg = groups.get(size_t(E) * n);
                std::vector<int> seen(size_t(n) * topk, 0);
                for (int e = 0; e < E; ++e) {
                    if (!gc[e]) continue;
                    std::vector<__half> gathered(size_t(gc[e]) * cols);
                    for (int c = 0; c < gc[e]; ++c) {
                        const int g = gg[size_t(e) * n + c];
                        pin(g >= 0 && g < n * topk && ids[size_t(g)] == e, "invalid grouped assignment");
                        ++seen[size_t(g)];
                        std::copy_n(x.data() + size_t(g / topk) * cols, cols, gathered.data() + size_t(c) * cols);
                    }
                    Device<__half> dg(gathered.size()); dg.put(gathered);
                    Device<__nv_bfloat16> exp(size_t(gc[e]) * rows);
                    tqllm::Mat v = m;
                    v.p = static_cast<const char *>(m.p) + size_t(e) * rows * fmt->row_bytes(cols);
                    v.rows = rows;
                    pin(fmt->gemm(v, dg.p, exp.p, gc[e], stream), "TQ3_4S dense GEMM reference missing");
                    for (int c = 0; c < gc[e]; ++c)
                        ck(cudaMemcpyAsync(ref.p + size_t(gg[size_t(e) * n + c]) * rows, exp.p + size_t(c) * rows,
                                            size_t(rows) * 2, cudaMemcpyDeviceToDevice, stream));
                    ck(cudaStreamSynchronize(stream));
                    cudaFree(dg.p); dg.p = nullptr;
                    cudaFree(exp.p); exp.p = nullptr;
                }
                require(std::all_of(seen.begin(), seen.end(), [](int v) { return v == 1; }),
                        "every assignment written exactly once");
                const auto y = out.get(size_t(n) * topk * rows), yr = ref.get(size_t(n) * topk * rows);
                require(std::memcmp(y.data(), yr.data(), y.size() * 2) == 0,
                        "expert_gemm == the dense GEMM of the gathered assignments (bit for bit)");
            }
        }

        ck(cudaStreamDestroy(stream));
        for (void * p : allocs) ck(cudaFree(p));
        std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}

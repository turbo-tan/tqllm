// Dense projections of a several-sequence verify block (9..32 tokens): one matvec_f32 of all columns must equal the
// launches of 8 columns that Checkpoint::project ran (bit for bit, per format), and read the weights once instead of once
// per 8 columns. Prints every registered format; fails only on a format that claims wide exact columns
// (max_exact_cols() > 8) and differs.
#include "formats/weight/weight_format.hpp"
#include "core/gguf.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }

int main() {
    // rows: a dense projection, and an LM-head height (>= 65536 rows: the 9..16-column kernels pick another split there)
    int fails = 0;
    for (const int rows : {2048, 81920}) {
    constexpr int cols = 2560, maxn = 32;
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<__half> xh(size_t(cols) * maxn); for (auto & v : xh) v = __float2half(nd(rng));
    __half * dx; float * y1, * y2; check(cudaMalloc(&dx, xh.size() * 2)); check(cudaMalloc(&y1, size_t(rows) * maxn * 4)); check(cudaMalloc(&y2, size_t(rows) * maxn * 4));
    check(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
    cudaEvent_t a, b; check(cudaEventCreate(&a)); check(cudaEventCreate(&b));
    for (uint32_t type = 0; type < 64; ++type) {
        const auto * f = weight_format(type);
        if (!f || f->basis() != Basis::Plain) continue;
        std::vector<void *> keep; Mat m;
        try {
            std::vector<char> raw(gguf_tensor_bytes(type, {uint64_t(cols), uint64_t(rows)}, "w"));
            std::uniform_int_distribution<int> bd(0, 0x3B); for (auto & c : raw) c = char(bd(rng));
            Upload up = [&](const void * h, size_t n) { void * p; check(cudaMalloc(&p, n)); check(cudaMemcpy(p, h, n, cudaMemcpyHostToDevice)); keep.push_back(p); return (const void *) p; };
            m = f->load(raw, rows, cols, up, nullptr);
        } catch (const std::exception & e) { for (void * p : keep) cudaFree(p); continue; }
        const int step = m.fmt->max_matvec_cols(), claim = m.fmt->max_exact_cols();
        auto per_step = [&](int n) { for (int b0 = 0; b0 < n; b0 += step) m.fmt->matvec_f32(m, dx + size_t(b0) * cols, y2 + size_t(b0) * rows, std::min(step, n - b0), nullptr); };
        std::printf("%-10s type %2u rows %6d step %2d claims %2d:", f->name(), type, rows, step, claim);
        bool all_same = true;
        for (int n : {12, 16, 24, 32}) {
            bool same = false;
            try {
                check(cudaMemset(y1, 0, size_t(rows) * maxn * 4)); check(cudaMemset(y2, 0, size_t(rows) * maxn * 4));
                m.fmt->matvec_f32(m, dx, y1, n, nullptr); per_step(n); check(cudaDeviceSynchronize());
                std::vector<float> r1(size_t(rows) * n), r2(r1.size());
                check(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(r2.data(), y2, r2.size() * 4, cudaMemcpyDeviceToHost));
                same = std::memcmp(r1.data(), r2.data(), r1.size() * 4) == 0;
            } catch (const std::exception &) { cudaGetLastError(); }
            std::printf(" %d:%s", n, same ? "same" : "DIFF");
            if (n <= claim) all_same &= same;
            if (n == 24 && same) {
                float t1 = 0, t2 = 0; constexpr int reps = 30;
                for (int k = 0; k < 3; ++k) { m.fmt->matvec_f32(m, dx, y1, n, nullptr); per_step(n); }
                check(cudaEventRecord(a)); for (int k = 0; k < reps; ++k) m.fmt->matvec_f32(m, dx, y1, n, nullptr); check(cudaEventRecord(b)); check(cudaEventSynchronize(b)); check(cudaEventElapsedTime(&t1, a, b));
                check(cudaEventRecord(a)); for (int k = 0; k < reps; ++k) per_step(n); check(cudaEventRecord(b)); check(cudaEventSynchronize(b)); check(cudaEventElapsedTime(&t2, a, b));
                std::printf(" (24 cols: one launch %.1f us, per %d %.1f us)", 1e3f * t1 / reps, step, 1e3f * t2 / reps);
            }
        }
        std::printf("%s\n", claim > step && !all_same ? "  FAIL" : "");
        fails += claim > step && !all_same;
        for (void * p : keep) cudaFree(p);
    }
    cudaFree(dx); cudaFree(y1); cudaFree(y2);
    }
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}

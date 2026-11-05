// TASK_ALPHA_GEMV: the BF16 GEMVs one Flash-Next verify step runs, at the shapes the 3090 step trace shows (4.67 ms of
// ~25 ms per step): hc mixers (down 320 x 10240, up 10240 x 320, inject 4 x 10240), the fused routers + shared gates in
// one segmented launch (512 + 1 + 512 + 1 rows x 2560), the GDN alpha / beta (48 x 2560) and the indexer k_proj
// (128 x 2560). Every shape at T = 1, 2, 4, 6, 8 token columns: us per call and GB/s for the kernels the trace shows
// (old) and for the optimized ones (new), and the new output must equal the old bit for bit (memcmp). The last lines
// project ms per step with the trace's calls per step (100.9 hc down, 100.9 hc up, 46.4 router segments, 190 tiny).
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {
constexpr int kMaxT = 8;

std::vector<__nv_bfloat16> weights(int rows, int cols, std::mt19937 & rng) {
    std::vector<__nv_bfloat16> w(size_t(rows) * cols);
    for (auto & v : w) v = __float2bfloat16_rn(std::normal_distribution<float>(0.f, 0.05f)(rng));
    return w;
}
__nv_bfloat16 * device_weights(int rows, int cols, std::mt19937 & rng) {
    const auto w = weights(rows, cols, rng);
    __nv_bfloat16 * d = nullptr; cudaMalloc(&d, w.size() * 2);
    cudaMemcpy(d, w.data(), w.size() * 2, cudaMemcpyHostToDevice); return d;
}
struct Buf {  // one shape: weights, kMaxT x cols activations, the old and the new output
    __nv_bfloat16 * w = nullptr; float * x = nullptr; float * yo = nullptr; float * yn = nullptr;
};
Buf alloc(int rows, int cols, std::mt19937 & rng) {
    Buf b; b.w = device_weights(rows, cols, rng);
    std::vector<float> x(size_t(kMaxT) * cols);
    for (auto & v : x) v = std::normal_distribution<float>(0.f, 1.f)(rng);
    cudaMalloc(&b.x, x.size() * 4); cudaMemcpy(b.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&b.yo, size_t(kMaxT) * rows * 4); cudaMalloc(&b.yn, size_t(kMaxT) * rows * 4);
    return b;
}
void free_buf(Buf & b) { cudaFree(b.w); cudaFree(b.x); cudaFree(b.yo); cudaFree(b.yn); b = Buf{}; }

// us per call: three windows of back-to-back launches, the best window (launch gaps included, as the trace counts them)
double bench(const std::function<void()> & f, int reps) {
    for (int i = 0; i < 30; ++i) f();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    double best = 1e30;
    for (int w = 0; w < 3; ++w) {
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) f();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        best = std::min(best, double(ms) * 1000.0 / reps);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}
// GB/s counting the bytes a perfect kernel moves: every weight row once, x once
double gbs(int rows, int cols, int t, double us) { return (double(rows) * cols * 2.0 + double(t) * cols * 4.0) / (us * 1e3); }

// old = the kernels the step trace shows today (one row per block); new = the dispatch's choice (R rows per block)
void run_old(const Buf & b, int rows, int cols, int t) { tqllm_cuda::fn_gemv_bf16_tokens(b.w, b.x, b.yo, rows, cols, t, nullptr, 1); }
void run_new(const Buf & b, int rows, int cols, int t) { tqllm_cuda::fn_gemv_bf16_tokens(b.w, b.x, b.yn, rows, cols, t, nullptr, 0); }

// report the first differing element (row, token) so a bit-identity break points at a kernel
int cmp(const char * what, int t, const std::vector<float> & o, const std::vector<float> & n, int rows) {
    if (std::memcmp(o.data(), n.data(), o.size() * 4) == 0) return 0;
    int shown = 0;
    for (size_t i = 0; i < o.size() && shown < 4; ++i)
        if (std::memcmp(&o[i], &n[i], 4) != 0) {
            std::printf("  MISMATCH %s T=%d row %d token %d: old %.9g new %.9g\n", what, t, int(i % rows), int(i / rows), o[i], n[i]);
            ++shown;
        }
    return 1;
}

struct Shape { const char * name; int rows, cols; double per_step; };
const Shape kShapes[] = {
    {"hc_down   few   320x10240", 320, 10240, 100.9},
    {"hc_up   short 10240x320 ", 10240, 320, 100.9},
    {"hc_inject few     4x10240", 4, 10240, 100.0},
    {"ssm_ab    few    48x2560 ", 48, 2560, 75.0},
    {"index_k   few   128x2560", 128, 2560, 12.5},
};
}  // namespace

int main() {
    std::mt19937 rng(20261028);
    int bad = 0;
    double old_step[5] = {}, new_step[5] = {};  // projected ms per verify step, per T (1, 2, 4, 6, 8)
    const int kTs[] = {1, 2, 4, 6, 8};
    for (const auto & s : kShapes) {
        Buf b = alloc(s.rows, s.cols, rng);
        for (int ti = 0; ti < 5; ++ti) {
            const int t = kTs[ti];
            run_old(b, s.rows, s.cols, t); run_new(b, s.rows, s.cols, t);
            std::vector<float> o(size_t(t) * s.rows), n(o.size());
            cudaMemcpy(o.data(), b.yo, o.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(n.data(), b.yn, n.size() * 4, cudaMemcpyDeviceToHost);
            if (cmp(s.name, t, o, n, s.rows)) ++bad;
            const double uo = bench([&] { run_old(b, s.rows, s.cols, t); }, 200), un = bench([&] { run_new(b, s.rows, s.cols, t); }, 200);
            old_step[ti] += s.per_step * uo / 1000.0; new_step[ti] += s.per_step * un / 1000.0;
            std::printf("%s T=%d  old %7.2f us %7.0f GB/s | new %7.2f us %7.0f GB/s | x%.2f  %.2f -> %.2f ms/step\n",
                        s.name, t, uo, gbs(s.rows, s.cols, t, uo), un, gbs(s.rows, s.cols, t, un),
                        uo > 0 ? un / uo : 1.0, s.per_step * uo / 1000.0, s.per_step * un / 1000.0);
        }
        free_buf(b);
    }
    // the fused router launch of one layer call: this router + shared gate and the next layer's in one launch (1026 rows)
    {
        const int cols = 2560, E = 512, rows[4] = {E, 1, E, 1}; int total = 0;
        __nv_bfloat16 * w[4]; float * yo[2] = {}; float * yn[2] = {};
        for (int i = 0; i < 4; ++i) { w[i] = device_weights(rows[i], cols, rng); total += rows[i]; }
        for (int j = 0; j < 2; ++j) {
            cudaMalloc(&yo[j], size_t(kMaxT) * (E + 1) * 4); cudaMalloc(&yn[j], size_t(kMaxT) * (E + 1) * 4);
        }
        tqllm_cuda::GemvSegs g, gn; int r = 0;
        for (int i = 0; i < 4; ++i) {
            g.w[i] = w[i]; g.y[i] = yo[i / 2] + (i % 2 ? E : 0); g.ld[i] = E + 1; g.start[i] = r; r += rows[i]; g.start[i + 1] = r;
            gn.w[i] = w[i]; gn.y[i] = yn[i / 2] + (i % 2 ? E : 0); gn.ld[i] = E + 1; gn.start[i] = g.start[i];
        }
        g.n = gn.n = 4; gn.start[4] = g.start[4];
        Buf b = alloc(E, cols, rng);  // x only (its weights are the first router, reused)
        float * rf = nullptr; cudaMalloc(&rf, size_t(kMaxT) * (E + 1) * 4);
        const char * seg_name[4] = {"seg0 512", "seg1 shexp", "seg2 next512", "seg3 shexp1"};
        const double per_step = 46.4;
        for (int ti = 0; ti < 5; ++ti) {
            const int t = kTs[ti];
            auto old = [&] { tqllm_cuda::fn_gemv_bf16_segments(g, b.x, cols, t, nullptr, 1); };
            auto nw = [&] { tqllm_cuda::fn_gemv_bf16_segments(gn, b.x, cols, t, nullptr, 0); };
            old(); nw();
            std::vector<float> o(size_t(t) * (E + 1) * 2), n(o.size());
            for (int j = 0; j < 2; ++j) {
                cudaMemcpy(o.data() + size_t(t) * (E + 1) * j, yo[j], size_t(t) * (E + 1) * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(n.data() + size_t(t) * (E + 1) * j, yn[j], size_t(t) * (E + 1) * 4, cudaMemcpyDeviceToHost);
                const std::vector<float> oj(o.begin() + size_t(t) * (E + 1) * j, o.begin() + size_t(t) * (E + 1) * (j + 1));
                const std::vector<float> nj(n.begin() + size_t(t) * (E + 1) * j, n.begin() + size_t(t) * (E + 1) * (j + 1));
                if (cmp(j ? "router+1" : "router", t, oj, nj, E + 1)) ++bad;
            }
            // and the one segmented launch must equal the same rows run as separate fn_gemv_bf16_tokens calls
            for (int i = 0; i < 4; ++i) {
                tqllm_cuda::fn_gemv_bf16_tokens(w[i], b.x, rf, rows[i], cols, t, nullptr, 0);
                std::vector<float> a(size_t(t) * rows[i]), c(size_t(t) * (E + 1)), d(a.size());
                cudaMemcpy(a.data(), rf, a.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(c.data(), yn[i / 2], c.size() * 4, cudaMemcpyDeviceToHost);
                for (int tt = 0; tt < t; ++tt)
                    for (int k = 0; k < rows[i]; ++k) d[size_t(tt) * rows[i] + k] = c[size_t(tt) * (E + 1) + (i % 2 ? E : 0) + k];
                if (cmp(seg_name[i], t, a, d, rows[i])) ++bad;
            }
            const double uo = bench(old, 200), un = bench(nw, 200);
            old_step[ti] += per_step * uo / 1000.0; new_step[ti] += per_step * un / 1000.0;
            std::printf("router  seg %dx%d  T=%d  old %7.2f us %7.0f GB/s | new %7.2f us %7.0f GB/s | x%.2f  %.2f -> %.2f ms/step\n",
                        total, cols, t, uo, gbs(total, cols, t, uo), un, gbs(total, cols, t, un),
                        uo > 0 ? un / uo : 1.0, per_step * uo / 1000.0, per_step * un / 1000.0);
        }
        for (int i = 0; i < 4; ++i) cudaFree(w[i]);
        cudaFree(yo[0]); cudaFree(yo[1]); cudaFree(yn[0]); cudaFree(yn[1]); cudaFree(rf); free_buf(b);
    }
    for (int ti = 0; ti < 5; ++ti)
        std::printf("T=%d projected per verify step: old %.2f ms, new %.2f ms (target <= 2.5)\n", kTs[ti], old_step[ti], new_step[ti]);
    if (cudaGetLastError() != cudaSuccess) { std::printf("gemv_bf16_perf: CUDA error\n"); return 1; }
    std::printf("gemv_bf16_perf: %s (%d mismatching outputs)\n", bad ? "gemv bf16 perf FAIL" : "gemv bf16 perf PASS", bad);
    return bad ? 1 : 0;
}

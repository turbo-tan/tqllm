// MXFP8 routed experts through the WeightFormat (GGML_MXFP8) at widths that are not a multiple of 512 (Inkling-2-Mini:
// 2304 and 1152): load() accepts them, dense matvec at k = 1152, expert_matvec (assignments, per-expert inputs, skipped
// negative ids) and expert_gemm (moe_group's grouped layout, experts with 0, 1 and more than 64 assignments) against a
// host decode of E4M3 x 2^(E8M0 - 127). Before the change load() refused cols % 512 != 0 and MXFP8 had no expert kernels.
#include "weight_format.hpp"
#include "gguf.hpp"
#include "moe_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tqllm;
namespace {
bool ok = true;
void check(bool c, const std::string & w) { if (!c) { std::fprintf(stderr, "FAIL: %s\n", w.c_str()); ok = false; } }
void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); std::exit(e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver ? 77 : 2); }
}
uint32_t g = 7u;
uint32_t lcg() { g = g * 1664525u + 1013904223u; return g; }
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1 + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}
struct Stack { std::vector<char> raw; std::vector<float> W; int rows, cols, experts; };
Stack make(int experts, int rows, int cols) {
    Stack st{{}, {}, rows, cols, experts};
    const size_t rb = size_t(cols) + cols / 32, R = size_t(experts) * rows;
    st.raw.resize(R * rb); st.W.resize(R * cols);
    for (size_t r = 0; r < R; ++r) {
        for (int c = 0; c < cols; ++c) { uint8_t b = uint8_t(lcg() >> 24); if ((b & 0x7f) == 0x7f) b &= 0xfe; st.raw[r * rb + c] = char(b); }
        for (int j = 0; j < cols / 32; ++j) st.raw[r * rb + cols + j] = char(uint8_t(113 + lcg() % 4));  // the checkpoint's scale range
        for (int c = 0; c < cols; ++c)
            st.W[r * cols + c] = e4m3(uint8_t(st.raw[r * rb + c])) * std::ldexp(1.0f, int(uint8_t(st.raw[r * rb + cols + c / 32])) - 127);
    }
    return st;
}
template <class T> T * up(const std::vector<T> & v) {
    T * p = nullptr; ck(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)));
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
std::vector<float> down_bf(const __nv_bfloat16 * p, size_t n) {
    std::vector<__nv_bfloat16> h(n); ck(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> o(n); for (size_t i = 0; i < n; ++i) o[i] = __bfloat162float(h[i]); return o;
}
double rel(double got, double ref, double mag) { return std::fabs(got - ref) / (std::max(std::fabs(ref), mag) + 1e-9); }
}  // namespace

int main() try {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
    const WeightFormat * wf = weight_format(GGML_MXFP8);
    check(wf && wf->has_expert_matvec(), "MXFP8 has expert kernels");
    cudaStream_t s; ck(cudaStreamCreate(&s));
    const auto upload = [](const void * h, size_t b) -> const void * {
        void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice));
        ck(cudaStreamSynchronize(cudaStreamPerThread)); return d;
    };
    const int E = 6, topk = 4, n = 40;
    for (const auto & [rows, cols] : {std::pair{192, 2304}, std::pair{2304, 1152}}) {
        const Stack st = make(E, rows, cols);
        Mat m;
        try { m = wf->load(st.raw, E * rows, cols, upload, s); } catch (const std::exception & e) { check(false, std::string("load cols ") + std::to_string(cols) + ": " + e.what()); continue; }
        // x [n * topk][cols] (per-expert inputs use all rows; token inputs the first n)
        std::vector<__half> xh(size_t(n) * topk * cols);
        std::vector<float> xf(xh.size());
        for (size_t i = 0; i < xh.size(); ++i) { xh[i] = __float2half(float(int(lcg() % 2001) - 1000) / 500.0f); xf[i] = __half2float(xh[i]); }
        // routing: expert 0 on every token (40 assignments), expert 5 on none, expert 4 once, one id skipped (-1)
        std::vector<int> ids(size_t(n) * topk);
        for (int t = 0; t < n; ++t) {
            ids[t * topk] = 0;
            for (int k = 1; k < topk; ++k) ids[t * topk + k] = 1 + (t + k) % 3;  // distinct within a token, as top-k routing
        }
        ids[5 * topk + 3] = 4; ids[9 * topk + 2] = -1;
        __half * dx = up(xh);
        int * dids = up(ids);
        __nv_bfloat16 * dy = nullptr; ck(cudaMalloc(&dy, size_t(n) * topk * rows * 2));
        for (int per = 0; per < 2; ++per) {
            auto ref = [&](int a, int r) {
                const int e = ids[a]; const float * xr = xf.data() + size_t(per ? a : a / topk) * cols;
                double v = 0, mag = 0;
                for (int c = 0; c < cols; ++c) { const double p = double(st.W[(size_t(e) * rows + r) * cols + c]) * xr[c]; v += p; mag += std::fabs(p); }
                return std::pair{v, mag / std::sqrt(double(cols))};
            };
            ck(cudaMemsetAsync(dy, 0, size_t(n) * topk * rows * 2, s));
            wf->expert_matvec(m, dx, dy, rows, dids, n, topk, per != 0, s);
            ck(cudaStreamSynchronize(s));
            auto y = down_bf(dy, size_t(n) * topk * rows);
            double em = 0;
            for (int a = 0; a < n * topk; ++a)
                for (int r = 0; r < rows; ++r) {
                    if (ids[a] < 0) { em = std::max(em, double(std::fabs(y[size_t(a) * rows + r]))); continue; }  // skipped: untouched (0)
                    const auto [v, mag] = ref(a, r);
                    em = std::max(em, rel(y[size_t(a) * rows + r], v, mag));
                }
            // grouped GEMM: moe_group's layout ([experts][n] assignment ids), skipped ids left out
            int * counts = nullptr, * groups = nullptr;
            ck(cudaMalloc(&counts, E * 4)); ck(cudaMalloc(&groups, size_t(E) * n * topk * 4));
            tqllm_cuda::moe_group(dids, counts, groups, E, n, topk, s);
            ck(cudaMemsetAsync(dy, 0, size_t(n) * topk * rows * 2, s));
            check(wf->expert_gemm(m, dx, dy, rows, counts, groups, n, E, topk, per != 0, s), "expert_gemm supported");
            ck(cudaStreamSynchronize(s));
            y = down_bf(dy, size_t(n) * topk * rows);
            double eg = 0;
            int shown = 0;
            for (int a = 0; a < n * topk; ++a)
                for (int r = 0; r < rows; ++r) {
                    if (ids[a] < 0) continue;
                    const auto [v, mag] = ref(a, r);
                    const double er = rel(y[size_t(a) * rows + r], v, mag);
                    if (er > 1e-2 && shown < 6) { std::printf("  gemm a %d (expert %d) row %d: got %.5f ref %.5f\n", a, ids[a], r, y[size_t(a) * rows + r], v); ++shown; }
                    eg = std::max(eg, er);
                }
            std::printf("MXFP8 experts %4d x %4d, %s inputs: expert_matvec worst rel err %.2e, expert_gemm %.2e\n", rows, cols, per ? "per-expert" : "token",
                        em, eg);
            check(em < 1e-2, "expert_matvec accuracy");
            check(eg < 1e-2, "expert_gemm accuracy");
            cudaFree(counts); cudaFree(groups);
        }
        // dense matvec at this width (k % 512 != 0): the stack as one matrix, 3 columns
        __nv_bfloat16 * dd = nullptr; ck(cudaMalloc(&dd, size_t(3) * E * rows * 2));
        wf->matvec(m, dx, dd, 3, s);
        ck(cudaStreamSynchronize(s));
        const auto yd = down_bf(dd, size_t(3) * E * rows);
        double ed = 0;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < E * rows; ++r) {
                double v = 0, mag = 0;
                for (int k = 0; k < cols; ++k) { const double p = double(st.W[size_t(r) * cols + k]) * xf[size_t(c) * cols + k]; v += p; mag += std::fabs(p); }
                ed = std::max(ed, rel(yd[size_t(c) * E * rows + r], v, mag / std::sqrt(double(cols))));
            }
        std::printf("MXFP8 dense matvec k %d: worst rel err %.2e\n", cols, ed);
        check(ed < 1e-2, "dense matvec accuracy");
        cudaFree(dd); cudaFree(dx); cudaFree(dids); cudaFree(dy);
    }
    std::printf(ok ? "mxfp8_experts: PASS\n" : "mxfp8_experts: FAIL\n");
    return ok ? 0 : 1;
} catch (const std::exception & e) {
    std::fprintf(stderr, "mxfp8_experts: %s\n", e.what());
    return 1;
}

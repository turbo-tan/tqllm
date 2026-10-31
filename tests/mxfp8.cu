// MXFP8 weight format (GGML_MXFP8): matvec (bf16 and f32 out, 1..8 columns, strided) and dequant_h against a host
// decode of E4M3 x 2^(E8M0 - 127), on random codes and scales including the extremes.
#include "weight_format.hpp"
#include "gguf.hpp"
#include "dequant.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tqllm;
namespace {
bool ok = true;
void check(bool c, const std::string & w) { if (!c) { std::fprintf(stderr, "FAIL: %s\n", w.c_str()); ok = false; } }
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); std::exit(2); } }
uint32_t g = 99u;
uint32_t lcg() { g = g * 1664525u + 1013904223u; return g; }
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 15 && m == 7 ? NAN : e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1 + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}
}  // namespace

int main() try {
    const int rows = 96, cols = 1024;
    const size_t rb = size_t(cols) + cols / 32;  // a row: cols E4M3, then its cols / 32 scales
    std::vector<char> raw(size_t(rows) * rb);
    std::vector<float> W(size_t(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            uint8_t b = uint8_t(lcg() >> 24);
            if ((b & 0x7f) == 0x7f) b &= 0xfe;  // no NaN codes
            raw[r * rb + c] = char(b);
        }
        for (int j = 0; j < cols / 32; ++j)
            raw[r * rb + cols + j] = char(r == 0 && j == 0 ? 1 : r == 0 && j == 1 ? 140 : uint8_t(120 + lcg() % 14));  // extremes + typical
    }
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            W[size_t(r) * cols + c] = e4m3(uint8_t(raw[r * rb + c])) * std::ldexp(1.0f, int(uint8_t(raw[r * rb + cols + c / 32])) - 127);
    const WeightFormat * wf = weight_format(GGML_MXFP8);
    check(wf != nullptr, "MXFP8 registered");
    cudaStream_t s; ck(cudaStreamCreate(&s));
    // a pageable upload may return before its DMA lands, and the build's per-thread default stream does not order it
    // before kernels on the test's own stream
    const auto upload = [](const void * h, size_t b) -> const void * {
        void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice));
        ck(cudaStreamSynchronize(cudaStreamPerThread));
        return d;
    };
    const Mat m = wf->load(raw, rows, cols, upload, s);
    for (int n : {1, 3, 8}) {
        std::vector<__half> x(size_t(n) * cols);
        for (auto & v : x) v = __float2half((int(lcg() % 2001) - 1000) / 1000.0f);
        __half * dx; ck(cudaMalloc(&dx, x.size() * 2)); ck(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice)); ck(cudaStreamSynchronize(cudaStreamPerThread));
        float * dy; ck(cudaMalloc(&dy, size_t(n) * rows * 4));
        wf->matvec_f32(m, dx, dy, n, s); ck(cudaStreamSynchronize(s));
        std::vector<float> y(size_t(n) * rows); ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        double e2 = 0, r2 = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < rows; ++r) {
                double a = 0;
                for (int k = 0; k < cols; ++k) a += double(W[size_t(r) * cols + k]) * __half2float(x[size_t(c) * cols + k]);
                const double d = y[size_t(c) * rows + r] - a; e2 += d * d; r2 += a * a;
            }
        std::printf("mxfp8 matvec n=%d rel_rms %.2e\n", n, std::sqrt(e2 / r2));
        check(std::sqrt(e2 / r2) < 1e-5, "matvec vs host decode, n = " + std::to_string(n));
        cudaFree(dx); cudaFree(dy);
    }
    __half * dh; ck(cudaMalloc(&dh, size_t(rows) * cols * 2));
    wf->dequant_h(m, dh, s); ck(cudaStreamSynchronize(s));
    std::vector<__half> h(size_t(rows) * cols); ck(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < h.size(); ++i) if (__half2float(h[i]) != __half2float(__float2half(W[i]))) ++bad;
    check(bad == 0, "dequant_h == FP16 of the host decode (" + std::to_string(bad) + " differ)");
    {   // host decode (dequantize_host, the engine's small-matrix F32 path) == the reference
        std::vector<float> hd(W.size());
        dequantize_host_rows(GGML_MXFP8, raw.data(), hd.data(), rows, cols);
        check(std::memcmp(hd.data(), W.data(), W.size() * 4) == 0, "dequantize_host_rows == host decode");
    }
    {   // two matrices stacked by rows (the engine's k|v and gate|up) run as one: rows are self-contained
        std::vector<char> two(raw);
        two.insert(two.end(), raw.begin(), raw.end());
        const Mat m2 = wf->load(two, 2 * rows, cols, upload, s);
        __half * d2; ck(cudaMalloc(&d2, size_t(2) * rows * cols * 2));
        wf->dequant_h(m2, d2, s); ck(cudaStreamSynchronize(s));
        std::vector<__half> h2(size_t(2) * rows * cols); ck(cudaMemcpy(h2.data(), d2, h2.size() * 2, cudaMemcpyDeviceToHost));
        check(std::memcmp(h2.data(), h.data(), h.size() * 2) == 0 && std::memcmp(h2.data() + h.size(), h.data(), h.size() * 2) == 0,
              "row-stacked matrices decode as each one");
    }
    std::printf(ok ? "mxfp8: OK\n" : "mxfp8: FAILED\n");
    return ok ? 0 : 1;
} catch (const std::exception & e) { std::fprintf(stderr, "mxfp8: %s\n", e.what()); return 1; }

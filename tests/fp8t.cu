// FP8T weight format (GGML_FP8T, modelopt FP8 projections as stored: per 128 weights the F32 scale, then 128 E4M3): load,
// matvec (bf16 / f32 out, 1..16 columns, strided), dequant_h and the host decode against a host reference, on random codes
// over the whole E4M3 range, row counts that are not multiples of 16 and both split-K shapes; two tensors with different
// scales row-concatenated keep their own; and the batched-verify invariant (a column's bits do not depend on the width).
#include "weight_format.hpp"
#include "gguf.hpp"
#include "dequant.hpp"
#include "quant_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
uint32_t g = 7u;
uint32_t lcg() { g = g * 1664525u + 1013904223u; return g; }
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 15 && m == 7 ? NAN : e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1 + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}
const void * upload(const void * h, size_t b) {
    void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice));
    ck(cudaStreamSynchronize(cudaStreamPerThread));
    return d;
}
// GGUF bytes of a [rows][cols] FP8T tensor with scale sc; W gets the decoded values
std::vector<char> make(int rows, int cols, float sc, std::vector<float> & W) {
    std::vector<char> raw(size_t(rows) * (cols / 128) * 132);
    W.resize(size_t(rows) * cols);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < cols / 128; ++b) {
            char * blk = raw.data() + (size_t(r) * (cols / 128) + b) * 132;
            std::memcpy(blk, &sc, 4);
            for (int i = 0; i < 128; ++i) {
                uint8_t q = uint8_t(lcg() >> 24);
                if ((q & 0x7f) == 0x7f) q &= 0xfe;  // no NaN codes
                if (r == 0 && b == 0 && i < 2) q = i ? 0x7e : 0x01;  // 448 and the smallest subnormal
                blk[4 + i] = char(q);
                W[size_t(r) * cols + b * 128 + i] = e4m3(q) * sc;
            }
        }
    return raw;
}
double ref_dot(const std::vector<float> & W, const std::vector<__half> & x, int r, int c, int cols) {
    double a = 0;
    for (int k = 0; k < cols; ++k) a += double(W[size_t(r) * cols + k]) * __half2float(x[size_t(c) * cols + k]);
    return a;
}
}  // namespace

int main() try {
    const WeightFormat * wf = weight_format(GGML_FP8T);
    check(wf != nullptr, "FP8T registered");
    if (!wf) { std::printf("fp8t: FAILED\n"); return 1; }
    uint64_t el = 0, by = 0;
    check(gguf_block_layout(GGML_FP8T, el, by) && el == 128 && by == 132, "FP8T block layout 128 / 132");
    check(wf->max_batch_cols() >= 16 && wf->max_matvec_cols() == 8, "FP8T: 16 columns per batched launch, dispatch cap 8");
    cudaStream_t s; ck(cudaStreamCreate(&s));
    for (auto [rows, cols, sc] : {std::tuple{40, 512, 0.00731f}, std::tuple{2064, 256, 1.5e-4f}}) {
        std::vector<float> W;
        const auto raw = make(rows, cols, sc, W);
        const Mat m = wf->load(raw, rows, cols, upload, s);
        const std::string tag = std::to_string(rows) + "x" + std::to_string(cols) + " (split-K " + std::to_string(tqllm_cuda::fp8t_split_k(rows)) + ")";
        const int NC = 16;
        std::vector<__half> x(size_t(NC) * cols);
        for (auto & v : x) v = __float2half((int(lcg() % 2001) - 1000) / 250.0f);
        __half * dx = static_cast<__half *>(const_cast<void *>(upload(x.data(), x.size() * 2)));
        float * dy; ck(cudaMalloc(&dy, size_t(NC) * rows * 4));
        std::vector<std::vector<float>> ywide;
        for (int n : {1, 3, 8, 9, 16}) {
            wf->matvec_f32(m, dx, dy, n, s); ck(cudaStreamSynchronize(s));
            std::vector<float> y(size_t(n) * rows); ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0;
            for (int c = 0; c < n; ++c) {
                double mx = 0;
                std::vector<double> ref(rows);
                for (int r = 0; r < rows; ++r) { ref[r] = ref_dot(W, x, r, c, cols); mx = std::max(mx, std::fabs(ref[r])); }
                for (int r = 0; r < rows; ++r) worst = std::max(worst, std::fabs(y[size_t(c) * rows + r] - ref[r]) / mx);
            }
            std::printf("fp8t %s matvec n=%2d worst |err| / max|ref| %.2e\n", tag.c_str(), n, worst);
            check(worst < 1e-6, tag + " matvec vs host, n = " + std::to_string(n));
            if (n == 16) ywide.push_back(y);
        }
        {   // a column's bits: 16 in one launch == each alone; bf16 out == bf16(f32 out); strided output
            size_t diff = 0;
            for (int c = 0; c < NC; ++c) {
                wf->matvec_f32(m, dx + size_t(c) * cols, dy, 1, s); ck(cudaStreamSynchronize(s));
                std::vector<float> y1(rows); ck(cudaMemcpy(y1.data(), dy, rows * 4, cudaMemcpyDeviceToHost));
                diff += std::memcmp(y1.data(), ywide[0].data() + size_t(c) * rows, rows * 4) != 0;
            }
            check(diff == 0, tag + ": a column's bits depend on the launch width (" + std::to_string(diff) + " columns)");
            const int ldy = rows + 24;
            __nv_bfloat16 * db; ck(cudaMalloc(&db, size_t(NC) * ldy * 2));
            check(wf->matvec_ld(m, dx, db, NC, ldy, s), "matvec_ld supported");
            ck(cudaStreamSynchronize(s));
            std::vector<__nv_bfloat16> yb(size_t(NC) * ldy); ck(cudaMemcpy(yb.data(), db, yb.size() * 2, cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (int c = 0; c < NC; ++c)
                for (int r = 0; r < rows; ++r)
                    bad += __bfloat162float(yb[size_t(c) * ldy + r]) != __bfloat162float(__float2bfloat16(ywide[0][size_t(c) * rows + r]));
            check(bad == 0, tag + ": strided bf16 output == bf16 of the f32 output (" + std::to_string(bad) + " differ)");
            cudaFree(db);
        }
        __half * dh; ck(cudaMalloc(&dh, size_t(rows) * cols * 2));
        wf->dequant_h(m, dh, s); ck(cudaStreamSynchronize(s));
        std::vector<__half> h(size_t(rows) * cols); ck(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < h.size(); ++i) bad += __half2float(h[i]) != __half2float(__float2half(W[i]));
        check(bad == 0, tag + ": dequant_h == FP16 of the host decode (" + std::to_string(bad) + " differ)");
        std::vector<float> hd(W.size());
        dequantize_host(GGML_FP8T, raw.data(), hd.data(), int64_t(hd.size()));
        check(std::memcmp(hd.data(), W.data(), W.size() * 4) == 0, tag + ": dequantize_host == host decode");
        cudaFree(dh); cudaFree(dx); cudaFree(dy);
    }
    {   // q | k row-concatenated with different scales (the engine's g_cat): each part keeps its scale
        std::vector<float> Wa, Wb;
        auto a = make(32, 256, 0.01f, Wa);
        const auto b = make(16, 256, 3.0f, Wb);
        a.insert(a.end(), b.begin(), b.end());
        const Mat m = wf->load(a, 48, 256, upload, s);
        __half * dh; ck(cudaMalloc(&dh, size_t(48) * 256 * 2));
        wf->dequant_h(m, dh, s); ck(cudaStreamSynchronize(s));
        std::vector<__half> h(size_t(48) * 256); ck(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost));
        Wa.insert(Wa.end(), Wb.begin(), Wb.end());
        size_t bad = 0;
        for (size_t i = 0; i < h.size(); ++i) bad += __half2float(h[i]) != __half2float(__float2half(Wa[i]));
        check(bad == 0, "row-concatenated parts keep their own scale (" + std::to_string(bad) + " differ)");
        auto mixed = a;
        float other = 2.0f; std::memcpy(mixed.data() + 132, &other, 4);  // second block of row 0: another scale
        bool threw = false;
        try { wf->load(mixed, 48, 256, upload, s); } catch (const std::exception &) { threw = true; }
        check(threw, "a row with two scales is refused");
    }
    std::printf(ok ? "fp8t: OK\n" : "fp8t: FAILED\n");
    return ok ? 0 : 1;
} catch (const std::exception & e) { std::fprintf(stderr, "fp8t: %s\n", e.what()); return 1; }

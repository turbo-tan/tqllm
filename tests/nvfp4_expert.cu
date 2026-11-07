// NVFP4 experts (GGUF NVFP4G: per 128 weights F32 global, 8 E4M3 group scales, E2M1 nibbles with low = even weight) through the
// NVFP4G loader on the u4h kernels (FMT 16): decode expert matvec and prefill grouped GEMM against an fp64 reference of the file
// values, and the host dequant (the CPU miss pool's fallback) equal to them.
#include "weight_format.hpp"
#include "gguf.hpp"
#include "dequant.hpp"
#include "quant_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void require(bool ok, const std::string & msg) { if (!ok) throw std::runtime_error(msg); }
template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); check(cudaMemset(p, 0, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) { std::vector<T> x(n); check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
static const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

static bool run(int rows, int cols) {
    const int experts = 5, n = 70, topk = 3, A = n * topk;
    uint64_t elems = 0, bytes = 0; require(gguf_block_layout(GGML_NVFP4G, elems, bytes) && elems == 128 && bytes == 76, "NVFP4G block layout");
    std::mt19937 rng(uint32_t(rows * 7 + cols));
    const int64_t total_rows = int64_t(experts) * rows, nb = cols / 128;
    std::vector<char> raw(size_t(total_rows * nb * 76));
    std::vector<float> w(size_t(total_rows) * cols);
    for (int64_t r = 0; r < total_rows; ++r) {
        const float g = 0.0005f + 0.002f * float(rng() % 1000) / 1000.f;  // one global per row (modelopt: per matrix)
        for (int64_t b = 0; b < nb; ++b) {
            char * blk = raw.data() + (r * nb + b) * 76; std::memcpy(blk, &g, 4);
            for (int grp = 0; grp < 8; ++grp) {
                const uint8_t sc = uint8_t(0x28 + rng() % 0x30);  // E4M3 0.125 .. 16
                blk[4 + grp] = char(sc);
                const float d = e4m3_to_float(sc) * g;
                for (int i = 0; i < 8; ++i) {
                    const uint8_t q = uint8_t(rng()); blk[12 + grp * 8 + i] = char(q);
                    float * o = &w[size_t(r) * cols + size_t(b) * 128 + grp * 16 + 2 * i];
                    o[0] = kE2M1[q & 15] * d; o[1] = kE2M1[q >> 4] * d;
                }
            }
        }
    }
    {   // host dequant (the CPU miss pool's fallback for NVFP4G rows) gives the file values
        std::vector<float> h(w.size()); dequantize_host(GGML_NVFP4G, raw.data(), h.data(), int64_t(h.size()));
        require(std::memcmp(h.data(), w.data(), w.size() * sizeof(float)) == 0, "dequantize_host NVFP4G differs from the file values");
    }
    std::vector<std::unique_ptr<Device<char>>> owned;
    Upload up = [&](const void * h, size_t nb) { owned.push_back(std::make_unique<Device<char>>(nb)); check(cudaMemcpy(owned.back()->p, h, nb, cudaMemcpyHostToDevice)); return (const void *)owned.back()->p; };
    const auto * format = weight_format(GGML_NVFP4G); require(format, "NVFP4G not registered");
    const Mat m = format->load(raw, experts * rows, cols, up, nullptr);
    require(m.fmt && m.fmt->has_expert_matvec() && m.fmt->u4h_id() == 16, "NVFP4G must load as NVF4 (u4h 16)");

    std::vector<int> ids(A); for (int a = 0; a < A; ++a) ids[a] = (a % topk == 0) ? 0 : int(1 + rng() % (experts - 1));
    std::vector<__half> x(size_t(cols) * n); std::vector<float> xf(x.size());
    for (size_t i = 0; i < x.size(); ++i) { x[i] = __float2half_rn(float(int(rng() % 2001) - 1000) / 1000.f); xf[i] = __half2float(x[i]); }
    std::vector<double> ref(size_t(A) * rows), mag(ref.size());
    for (int a = 0; a < A; ++a)
        for (int r = 0; r < rows; ++r) {
            double s = 0, sa = 0; const float * wr = &w[(size_t(ids[a]) * rows + r) * cols]; const float * xr = &xf[size_t(a / topk) * cols];
            for (int c = 0; c < cols; ++c) { s += double(wr[c]) * xr[c]; sa += std::fabs(double(wr[c]) * xr[c]); }
            ref[size_t(a) * rows + r] = s; mag[size_t(a) * rows + r] = sa;
        }
    Device<__half> dx(x.size()); dx.put(x);
    Device<int> dids(static_cast<size_t>(A)); dids.put(ids);
    Device<__nv_bfloat16> ym(size_t(A) * rows), yg(size_t(A) * rows);
    m.fmt->expert_matvec(m, dx.p, ym.p, rows, dids.p, n, topk, false, nullptr);
    std::vector<int> cnt(experts, 0); int cap = 1;
    for (int a = 0; a < A; ++a) cap = std::max(cap, ++cnt[ids[a]]);
    std::vector<int> grp(size_t(experts) * cap, 0), fill(experts, 0);
    for (int a = 0; a < A; ++a) grp[size_t(ids[a]) * cap + fill[ids[a]]++] = a;
    Device<int> dcnt(experts), dgrp(grp.size()); dcnt.put(cnt); dgrp.put(grp);
    require(m.fmt->expert_gemm(m, dx.p, yg.p, rows, dcnt.p, dgrp.p, cap, experts, topk, false, nullptr), "NVF4 expert GEMM missing");
    check(cudaDeviceSynchronize());
    const auto hm = ym.get(ref.size()), hg = yg.get(ref.size());
    double wm = 0, wg = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double den = mag[i] + 1e-9;
        wm = std::max(wm, std::fabs(double(__bfloat162float(hm[i])) - ref[i]) / den);
        wg = std::max(wg, std::fabs(double(__bfloat162float(hg[i])) - ref[i]) / den);
    }
    // FP16 activations are exact here; the error is the BF16 output rounding plus fp16/fp32 accumulation (< 1 % of sum |w x|)
    const bool ok = wm < 1e-2 && wg < 1e-2;
    std::printf("NVFP4 %d x %d: decode matvec worst %.2e, prefill GEMM worst %.2e (of sum |w x|) %s\n", rows, cols, wm, wg, ok ? "ok" : "TOO LARGE");
    return ok;
}
int main() {
    try {
        bool ok = run(640, 2560);  // gate / up
        ok = run(2560, 640) && ok; // down
        std::printf("nvfp4 expert %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) { std::printf("nvfp4 expert FAIL: %s\n", e.what()); return 1; }
}

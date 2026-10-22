// BF16 / F16 / F32 GGUF matrices through the weight-format registry: matvec (BF16 and FP32 output) and dequant_h
// against a CPU reference on the stored values (F32 is stored as BF16).
#include "formats/weight/weight_format.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static uint16_t to_bf16(float v) { uint32_t u; std::memcpy(&u, &v, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float from_bf16(uint16_t b) { const uint32_t u = uint32_t(b) << 16; float v; std::memcpy(&v, &u, 4); return v; }

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 0.05f);
    int fails = 0;
    for (uint32_t type : {30u, 1u, 0u}) {
        const auto * f = tqllm::weight_format(type);
        if (!f) { std::printf("FAIL type %u: no weight format registered\n", type); ++fails; continue; }
        for (const auto & sh : std::vector<std::pair<int, int>>{{96, 320}, {130, 5120}, {64, 72}}) {
            const int rows = sh.first, cols = sh.second;
            std::vector<float> w(size_t(rows) * cols), stored(w.size());
            for (auto & v : w) v = nd(rng);
            const size_t es = type == 0 ? 4 : 2;
            std::vector<char> raw(w.size() * es);
            for (size_t i = 0; i < w.size(); ++i) {
                if (type == 30) { const uint16_t b = to_bf16(w[i]); std::memcpy(&raw[2 * i], &b, 2); stored[i] = from_bf16(b); }
                else if (type == 1) { const __half h = __float2half(w[i]); std::memcpy(&raw[2 * i], &h, 2); stored[i] = __half2float(h); }
                else { std::memcpy(&raw[4 * i], &w[i], 4); stored[i] = from_bf16(to_bf16(w[i])); }
            }
            std::vector<void *> keep;
            tqllm::Upload up = [&](const void * h, size_t bytes) { void * p; check(cudaMalloc(&p, bytes)); check(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice)); keep.push_back(p); return static_cast<const void *>(p); };
            const tqllm::Mat m = f->load(raw, rows, cols, up, nullptr);
            const int n = std::min(8, m.fmt->max_matvec_cols());
            std::vector<__half> x(size_t(n) * cols);
            for (auto & v : x) v = __float2half(nd(rng) * 20);
            __half * dx; check(cudaMalloc(&dx, x.size() * 2)); check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
            __nv_bfloat16 * dy; check(cudaMalloc(&dy, size_t(n) * rows * 2));
            float * dyf; check(cudaMalloc(&dyf, size_t(n) * rows * 4));
            __half * dh; check(cudaMalloc(&dh, w.size() * 2));
            m.fmt->matvec(m, dx, dy, n, nullptr); m.fmt->matvec_f32(m, dx, dyf, n, nullptr); m.fmt->dequant_h(m, dh, nullptr);
            check(cudaDeviceSynchronize());
            std::vector<__nv_bfloat16> y(size_t(n) * rows); std::vector<float> yf(size_t(n) * rows); std::vector<__half> h(w.size());
            check(cudaMemcpy(y.data(), dy, y.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(yf.data(), dyf, yf.size() * 4, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost));
            double e16 = 0, e32 = 0, mag = 0; size_t hbad = 0;
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < rows; ++r) {
                    double ref = 0;
                    for (int k = 0; k < cols; ++k) ref += double(stored[size_t(r) * cols + k]) * __half2float(x[size_t(c) * cols + k]);
                    mag = std::max(mag, std::fabs(ref));
                    e16 = std::max(e16, std::fabs(ref - __bfloat162float(y[size_t(c) * rows + r])));
                    e32 = std::max(e32, std::fabs(ref - yf[size_t(c) * rows + r]));
                }
            for (size_t i = 0; i < w.size(); ++i) hbad += __half2float(h[i]) != __half2float(__float2half(stored[i]));
            const bool ok = e16 <= 1e-2 * mag && e32 <= 1e-4 * mag && hbad == 0;
            std::printf("%s type %u %s rows %d cols %d n %d: bf16-out err %.3g, f32-out err %.3g (max |y| %.3g), dequant_h mismatches %zu\n",
                        ok ? "PASS" : "FAIL", type, f->name(), rows, cols, n, e16, e32, mag, hbad);
            fails += !ok;
            cudaFree(dx); cudaFree(dy); cudaFree(dyf); cudaFree(dh);
            for (void * p : keep) cudaFree(p);
        }
    }
    return fails ? 1 : 0;
}

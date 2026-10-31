// The qwen35 verify path keeps the Gated DeltaNet [b|a] projections BF16 next to a quantized, rotated [qkv|z]
// (the hub's TQ3_4S 35B-A3B files do: attn_qkv is TQ3_4S, ssm_beta / ssm_alpha stay BF16). BF16 weights are
// stored un-rotated, so those rows must read the normalised activation un-rotated. The engine takes that copy
// from norm_h, which is norm_rotate_h without the rotation, and widens it to BF16 for bf16_matvec_cols.
// This test pins both halves:
//   (a) norm_h + tq3_rotate_h == norm_rotate_h bit for bit: the un-rotated copy is the same activation the
//       quantized [qkv|z] rows see, so the two groups stay consistent;
//   (b) bf16_matvec_cols on the un-rotated copy against a CPU reference of W x. Feeding it the rotated copy
//       (what the code did before, through mm_h's BF16 branch) is far from that reference, so (b) fails on the
//       old input and cannot pass vacuously.
//   (c) our TQ4 files store the BF16 [b|a] rows rotated (W R^T, R the per-128 TQ4 rotation): the copy the engine
//       picks for Basis::Tq4Rot128 (bf16_ba_unrotated) must give W x. The un-rotated pick (72209a7) is far from
//       it - that pick made Qwen3.8-27B TQ_L greedy decode loop (MMLU-Pro 37/40 -> 23/40).
#include "quant_kernels.cuh"
#include "qwen35_guard.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template <class T> struct Device {
    T * p = nullptr;
    Device() = default;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { if (p) cudaFree(p); }
    void put(const std::vector<T> & v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

int main() {
    try {
        const int k = 2048, ncols = 13, rows = 32;  // hidden, the verify width that crashed, [b|a] rows
        const float eps = 1e-6f;
        std::mt19937 gen(20261020u);
        std::uniform_real_distribution<float> u(-1.f, 1.f);

        std::vector<__nv_bfloat16> x(size_t(ncols) * k);
        for (auto & v : x) v = __float2bfloat16_rn(u(gen) * 4.f);
        std::vector<float> gain(k);
        for (auto & v : gain) v = 0.5f + 0.01f * float(gen() % 100);
        const int nparts = k / 256;
        std::vector<float> part(size_t(ncols) * nparts, 0.f);
        for (int c = 0; c < ncols; ++c)
            for (int i = 0; i < nparts; ++i) {
                float t = 0.f;
                for (int j = 0; j < 256; ++j) {
                    const float v = __bfloat162float(x[size_t(c) * k + i * 256 + j]);
                    t += v * v;
                }
                part[size_t(c) * nparts + i] = t;
            }
        std::vector<__nv_bfloat16> wba(size_t(rows) * k);
        for (auto & v : wba) v = __float2bfloat16_rn(u(gen) * .05f);

        Device<__nv_bfloat16> dx(x.size()), dw(wba.size());
        Device<float> dpart(part.size()), dgain(gain.size());
        Device<__half> dr(size_t(ncols) * k), du(size_t(ncols) * k), dr2(size_t(ncols) * k);
        Device<__nv_bfloat16> dxu(size_t(ncols) * k), dxr(size_t(ncols) * k);
        Device<__nv_bfloat16> dy(size_t(ncols) * rows), dyr(size_t(ncols) * rows);
        dx.put(x); dw.put(wba); dpart.put(part); dgain.put(gain);

        tqllm_cuda::norm_rotate_h(dx.p, dgain.p, dpart.p, eps, dr.p, k, ncols, nullptr);
        tqllm_cuda::norm_h(dx.p, dgain.p, dpart.p, eps, du.p, k, ncols, nullptr);
        // the un-rotated FP16 copy widened to BF16, as h_to_bf16 does (exact: norm_h wrote FP16 of a BF16 value)
        std::vector<__nv_bfloat16> xu(size_t(ncols) * k), xr(size_t(ncols) * k);
        const auto un = du.get(size_t(ncols) * k), rot = dr.get(size_t(ncols) * k);
        for (size_t i = 0; i < xu.size(); ++i) {
            xu[i] = __float2bfloat16_rn(__half2float(un[i]));
            xr[i] = __float2bfloat16_rn(__half2float(rot[i]));
        }
        dxu.put(xu); dxr.put(xr);

        tqllm_cuda::tq3_rotate_h(dxu.p, dr2.p, k, ncols, nullptr);
        const auto rot2 = dr2.get(size_t(ncols) * k);
        require(std::memcmp(rot.data(), rot2.data(), rot.size() * 2) == 0,
                "norm_h + tq3_rotate_h == norm_rotate_h bit for bit (the un-rotated copy is the same activation)");

        auto worst = [&](const __nv_bfloat16 * dx_in, Device<__nv_bfloat16> & out) {
            tqllm_cuda::bf16_matvec_cols(dw.p, dx_in, out.p, rows, k, ncols, nullptr);
            const auto y = out.get(size_t(ncols) * rows);
            double worst_rel = 0.;
            for (int c = 0; c < ncols; ++c)
                for (int r = 0; r < rows; ++r) {
                    double ref = 0., scale = 0.;
                    for (int j = 0; j < k; ++j) {
                        const double t = double(__bfloat162float(wba[size_t(r) * k + j])) * __bfloat162float(xu[size_t(c) * k + j]);
                        ref += t;
                        scale += std::abs(t);
                    }
                    const double got = __bfloat162float(y[size_t(c) * rows + r]);
                    if (!std::isfinite(got) || !std::isfinite(ref)) return 1e9;
                    worst_rel = std::max(worst_rel, std::abs(got - ref) / std::max(1e-6, scale));
                }
            return worst_rel;
        };
        const double e_un = worst(dxu.p, dy), e_rot = worst(dxr.p, dyr);
        std::printf("worst rel err: un-rotated input %.2e, rotated input %.2e\n", e_un, e_rot);
        require(e_un < 1e-2, "the BF16 [b|a] rows match W x with the un-rotated activation (vs a CPU reference)");
        require(e_rot > 10. * e_un, "the rotated copy is far from W x: the check fails on the old input");

        // (c) TQ4: rows stored rotated. R w_r is tq4_rotate_h of the row (R = H D / sqrt(128) per 128, so the row
        // of W R^T is R applied to w_r); rows widened back to BF16.
        Device<__half> dw4h(wba.size()), d4r(size_t(ncols) * k);
        tqllm_cuda::tq4_rotate_h(dw.p, dw4h.p, k, rows, nullptr);
        const auto w4h = dw4h.get(wba.size());
        std::vector<__nv_bfloat16> w4(wba.size());
        for (size_t i = 0; i < w4.size(); ++i) w4[i] = __float2bfloat16_rn(__half2float(w4h[i]));
        Device<__nv_bfloat16> dw4(w4.size()), dx4r(size_t(ncols) * k), dy4(size_t(ncols) * rows);
        dw4.put(w4);
        tqllm_cuda::tq4_norm_rotate_h(dx.p, dgain.p, dpart.p, eps, d4r.p, k, ncols, nullptr);
        const auto r4 = d4r.get(size_t(ncols) * k);
        std::vector<__nv_bfloat16> x4r(r4.size());
        for (size_t i = 0; i < x4r.size(); ++i) x4r[i] = __float2bfloat16_rn(__half2float(r4[i]));
        dx4r.put(x4r);
        auto err4 = [&](const __nv_bfloat16 * dx_in) {
            tqllm_cuda::bf16_matvec_cols(dw4.p, dx_in, dy4.p, rows, k, ncols, nullptr);
            const auto y = dy4.get(size_t(ncols) * rows);
            double worst_rel = 0.;
            for (int c = 0; c < ncols; ++c)
                for (int r = 0; r < rows; ++r) {
                    double ref = 0., scale = 0.;
                    for (int j = 0; j < k; ++j) {
                        const double t = double(__bfloat162float(wba[size_t(r) * k + j])) * __bfloat162float(xu[size_t(c) * k + j]);
                        ref += t;
                        scale += std::abs(t);
                    }
                    const double got = __bfloat162float(y[size_t(c) * rows + r]);
                    if (!std::isfinite(got)) return 1e9;
                    worst_rel = std::max(worst_rel, std::abs(got - ref) / std::max(1e-6, scale));
                }
            return worst_rel;
        };
        const bool un4 = tqllm::bf16_ba_unrotated(tqllm::Basis::Tq4Rot128);
        const double e4_pick = err4(un4 ? dxu.p : dx4r.p), e4_other = err4(un4 ? dx4r.p : dxu.p);
        std::printf("TQ4 worst rel err: engine's pick (%s) %.2e, the other copy %.2e\n", un4 ? "un-rotated" : "rotated",
                    e4_pick, e4_other);
        require(e4_pick < 1e-2, "TQ4 (rows stored rotated): the engine's input pick gives W x");
        require(e4_other > 10. * e4_pick, "and the other copy is far from W x");

        std::printf("%s: %d failure(s)\n", failures ? "FAIL" : "PASS", failures);
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}

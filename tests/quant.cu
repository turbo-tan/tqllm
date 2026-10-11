// TQ3_4S / Q6_K kernels vs a CPU reference that follows llama.cpp-tq3's dequantize_row_tq3_4s / dequantize_row_q6_K.
#include "quant_kernels.cuh"
#include "formats/matvec_rules.hpp"
#include "qwen35_kernels.cuh"
#include "../src/dequant.hpp"
#include "../src/formats/weight_format.hpp"
#include <cuda_fp8.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;
int failures = 0;

const float CB[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f, 0.230106f, 0.725222f, 1.277503f, 1.988943f};
const float SG[32] = {+1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
                      -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1};

float e3m5(uint8_t b) { return b ? std::ldexp(1.f + (b & 31) / 32.f, (b >> 5) - 9) : 0.f; }

void deq_tq3(const uint8_t * blk, float * out) {  // one 32-weight block
    float rot[32];
    for (int g = 0; g < 4; ++g) {
        const uint8_t * qp = blk + 4 + g * 3;
        const uint8_t idx[8] = {uint8_t(qp[0] & 7), uint8_t((qp[0] >> 3) & 7), uint8_t(((qp[0] >> 6) | (qp[1] << 2)) & 7),
                                uint8_t((qp[1] >> 1) & 7), uint8_t((qp[1] >> 4) & 7),
                                uint8_t(((qp[1] >> 7) | (qp[2] << 1)) & 7), uint8_t((qp[2] >> 2) & 7),
                                uint8_t((qp[2] >> 5) & 7)};
        for (int j = 0; j < 8; ++j) rot[g * 8 + j] = CB[idx[j]] * e3m5(blk[g]);
    }
    for (int step = 1; step < 32; step <<= 1)
        for (int i = 0; i < 32; i += step << 1)
            for (int j = i; j < i + step; ++j) {
                const float a = rot[j], b = rot[j + step];
                rot[j] = a + b;
                rot[j + step] = a - b;
            }
    for (int i = 0; i < 32; ++i) out[i] = rot[i] / std::sqrt(32.f) * SG[i];
}

void deq_q6k(const uint8_t * blk, float * y) {
    const uint8_t *ql = blk, *qh = blk + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    __half hd;
    std::memcpy(&hd, blk + 208, 2);
    const float d = __half2float(hd);
    for (int n = 0; n < 2; ++n, y += 128, ql += 64, qh += 32, sc += 8)
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            y[l] = d * sc[is] * (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = d * sc[is + 2] * (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = d * sc[is + 4] * (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = d * sc[is + 6] * (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
}

template <class T> T * dev(const std::vector<T> & h) {
    T * p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T * p, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

void check(const char * name, const std::vector<float> & got, const std::vector<double> & ref, double tol_rel) {
    double se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = got[i] - ref[i];
        se += e * e;
        sr += ref[i] * ref[i];
        mx = std::fmax(mx, std::fabs(e));
    }
    const double rel = std::sqrt(se / (sr + 1e-30));
    const bool ok = rel <= tol_rel;
    std::printf("%-16s rel_rms %.3e  max_abs %.3e  %s\n", name, rel, mx, ok ? "OK" : "FAIL");
    failures += !ok;
}

} // namespace

int main() {
    std::mt19937 g(99);
    std::uniform_int_distribution<int> byte(0, 255), sexp(3, 6), smant(0, 31);
    std::normal_distribution<float> nd;
    const int rows = std::getenv("TQ_ROWS") ? std::atoi(std::getenv("TQ_ROWS")) : 300;
    const int k = std::getenv("TQ_K") ? std::atoi(std::getenv("TQ_K")) : 3104;  // 97 blocks: 3 full tiles + partial
    const int nblk = k / 32;
    std::vector<uint8_t> w(size_t(rows) * nblk * 16);
    for (size_t b = 0; b < size_t(rows) * nblk; ++b) {
        for (int s = 0; s < 4; ++s) w[b * 16 + s] = uint8_t((sexp(g) << 5) | smant(g));
        for (int q = 4; q < 16; ++q) w[b * 16 + q] = uint8_t(byte(g));
    }
    std::vector<float> wd(size_t(rows) * k);
    for (size_t b = 0; b < size_t(rows) * nblk; ++b) deq_tq3(&w[b * 16], &wd[b * 32]);
    auto * dw = dev(w);

    for (int ncols : {1, 3, 8}) {
        std::vector<bf16> x(size_t(ncols) * k);
        for (auto & v : x) v = __float2bfloat16_rn(nd(g));
        std::vector<double> ref(size_t(ncols) * rows);
        for (int n = 0; n < ncols; ++n)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int i = 0; i < k; ++i) acc += double(wd[size_t(r) * k + i]) * __bfloat162float(x[size_t(n) * k + i]);
                ref[size_t(n) * rows + r] = acc;
            }
        auto * dx = dev(x);
        float * dxr = nullptr;
        bf16 * dy = nullptr;
        cudaMalloc(&dxr, x.size() * 4);
        cudaMalloc(&dy, ref.size() * 2);
        tqllm_cuda::tq3_rotate(dx, dxr, k, ncols, nullptr);
        tqllm_cuda::tq3_matvec(dw, dxr, dy, rows, k, ncols, nullptr);
        const auto yb = host(dy, ref.size());
        std::vector<float> y(yb.size());
        for (size_t i = 0; i < y.size(); ++i) y[i] = __bfloat162float(yb[i]);
        char name[32];
        std::snprintf(name, sizeof(name), "tq3_matvec_n%d", ncols);
        check(name, y, ref, 5e-3);  // BF16 output rounding dominates
    }
    { // int8 fast path: repack in place, then compare against the CPU reference (int8 activation error ~0.6%)
        std::vector<bf16> x(k);
        for (auto & v : x) v = __float2bfloat16_rn(nd(g));
        std::vector<double> ref(rows);
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int i = 0; i < k; ++i) acc += double(wd[size_t(r) * k + i]) * __bfloat162float(x[i]);
            ref[r] = acc;
        }
        auto * dw8 = dev(w);
        tqllm_cuda::tq3_repack(dw8, int64_t(rows) * nblk, nullptr);
        auto * dx = dev(x);
        int8_t * dq = nullptr;
        float * ds = nullptr;
        bf16 * dy = nullptr;
        cudaMalloc(&dq, k);
        cudaMalloc(&ds, nblk * 4);
        cudaMalloc(&dy, rows * 2);
        tqllm_cuda::tq3_rotate_i8(dx, dq, ds, k, nullptr);
        tqllm_cuda::tq3_matvec_i8(dw8, dq, ds, dy, rows, k, nullptr);
        const auto yb = host(dy, rows);
        std::vector<float> y(rows);
        for (int i = 0; i < rows; ++i) y[i] = __bfloat162float(yb[i]);
        check("tq3_matvec_i8", y, ref, 1.5e-2);
    }
    for (int ncols : {1, 5, 8, 12, 64, 100}) { // FP16 tensor-core verify path (<= 12) / prefill GEMM (> 12)
        std::vector<bf16> x(size_t(ncols) * k);
        for (auto & v : x) v = __float2bfloat16_rn(nd(g));
        std::vector<double> ref(size_t(ncols) * rows);
        for (int n = 0; n < ncols; ++n)
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int i = 0; i < k; ++i) acc += double(wd[size_t(r) * k + i]) * __bfloat162float(x[size_t(n) * k + i]);
                ref[size_t(n) * rows + r] = acc;
            }
        auto * dw8 = dev(w);
        tqllm_cuda::tq3_repack(dw8, int64_t(rows) * nblk, nullptr);
        auto * dx = dev(x);
        __half * dh = nullptr;
        bf16 * dy = nullptr;
        cudaMalloc(&dh, x.size() * 2);
        cudaMalloc(&dy, ref.size() * 2);
        tqllm_cuda::tq3_rotate_h(dx, dh, k, ncols, nullptr);
        if (ncols <= 12) tqllm_cuda::tq3_matvec_h(dw8, dh, dy, rows, k, ncols, nullptr);
        else tqllm_cuda::tq3_gemm_h(dw8, dh, dy, rows, k, ncols, nullptr);
        const auto yb = host(dy, ref.size());
        std::vector<float> y(yb.size());
        for (size_t i = 0; i < y.size(); ++i) y[i] = __bfloat162float(yb[i]);
        char name[32];
        std::snprintf(name, sizeof(name), "tq3_matvec_h_n%d", ncols);
        check(name, y, ref, 1e-2);  // int8 codebook + fp16 activation rounding
    }
    { // batched-verify width (13..32 columns): every column bit-identical to a narrow (<= 8 column) call
        const int wrows = 1024, wk = 2048, wnb = wk / 32;
        std::vector<uint8_t> ww(size_t(wrows) * wnb * 16);
        for (auto & v : ww) v = uint8_t(byte(g));
        for (size_t b = 0; b < size_t(wrows) * wnb; ++b) ww[b * 16] = uint8_t(1 + byte(g) % 200);
        auto * dww = dev(ww);
        tqllm_cuda::tq3_repack(dww, int64_t(wrows) * wnb, nullptr);
        bool same = true;
        for (int ncols : {13, 16, 20, 24, 29, 32}) {
            std::vector<bf16> x(size_t(ncols) * wk);
            for (auto & v : x) v = __float2bfloat16_rn(nd(g));
            auto * dx = dev(x);
            __half * dh = nullptr;
            bf16 *dyw = nullptr, *dyn = nullptr;
            cudaMalloc(&dh, x.size() * 2);
            cudaMalloc(&dyw, size_t(ncols) * wrows * 2);
            cudaMalloc(&dyn, size_t(ncols) * wrows * 2);
            tqllm_cuda::tq3_rotate_h(dx, dh, wk, ncols, nullptr);
            tqllm_cuda::tq3_matvec_h(dww, dh, dyw, wrows, wk, ncols, nullptr);
            for (int c0 = 0; c0 < ncols; c0 += 8)
                tqllm_cuda::tq3_matvec_h(dww, dh + size_t(c0) * wk, dyn + size_t(c0) * wrows, wrows, wk,
                                         std::min(8, ncols - c0), nullptr);
            const auto a = host(dyw, size_t(ncols) * wrows), b = host(dyn, size_t(ncols) * wrows);
            same &= std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
            cudaFree(dh);
            cudaFree(dyw);
            cudaFree(dyn);
        }
        {  // narrow variants (default / row-tiled / prefetching) are bit-identical per column
            const int nk = 13056;  // >= the long-K threshold
            std::vector<uint8_t> w2(size_t(256) * (nk / 32) * 16);
            for (auto & v : w2) v = uint8_t(byte(g));
            for (size_t b = 0; b < size_t(256) * (nk / 32); ++b) w2[b * 16] = uint8_t(1 + byte(g) % 200);
            auto * dw2 = dev(w2);
            tqllm_cuda::tq3_repack(dw2, int64_t(256) * (nk / 32), nullptr);
            std::vector<bf16> x(size_t(8) * nk);
            for (auto & v : x) v = __float2bfloat16_rn(nd(g));
            auto * dx = dev(x);
            __half * dh = nullptr;
            bf16 * dy = nullptr;
            cudaMalloc(&dh, x.size() * 2);
            cudaMalloc(&dy, size_t(8) * 256 * 2);
            tqllm_cuda::tq3_rotate_h(dx, dh, nk, 8, nullptr);
            std::vector<std::vector<bf16>> outs;
            for (const char * cfg : {"0", "1", "2", "4"}) {
                tqllm_cuda::tq3_matvec_h_variant(dw2, dh, dy, 256, nk, 8, std::atoi(cfg), nullptr);
                outs.push_back(host(dy, size_t(8) * 256));
            }
            for (size_t i = 1; i < outs.size(); ++i) same &= std::memcmp(outs[0].data(), outs[i].data(), outs[0].size() * 2) == 0;
        }
        {  // u4h row tiling (MT 1 / 2 / 4) is bit-identical per column
            const int ur = 1000, uk = 2048;
            std::vector<uint8_t> uw(size_t(ur) * uk / 2);
            for (auto & v : uw) v = uint8_t(byte(g));
            std::vector<__half2> usm(size_t(ur) * uk / 32);
            for (auto & v : usm) v = __floats2half2_rn(0.01f * (1 + byte(g) % 7), -0.05f);
            std::vector<__half> ux(size_t(16) * uk);
            for (auto & v : ux) v = __float2half_rn(nd(g));
            auto *duw = dev(uw);
            auto *dsm = dev(usm);
            auto *dux = dev(ux);
            bf16 * duy = nullptr;
            cudaMalloc(&duy, size_t(16) * ur * 2);
            std::vector<std::vector<bf16>> outs;
            for (int mt : {1, 2, 4})
                for (int nc : {8, 5, 16, 12}) {
                    tqllm_cuda::u4h_matvec_mt(duw, dsm, dux, duy, ur, uk, nc, mt, nullptr);
                    outs.push_back(host(duy, size_t(nc) * ur));
                }
            for (size_t i = 4; i < outs.size(); ++i)
                same &= std::memcmp(outs[i % 4].data(), outs[i].data(), outs[i].size() * 2) == 0;
            same &= std::memcmp(outs[0].data(), outs[2].data(), outs[0].size() * 2) == 0;  // 16 columns: same first 8
        }
        std::printf("%-24s %s\n", "tq3_matvec_h_wide_exact", same ? "PASS" : "FAIL");
        if (!same) ++failures;
    }
    {  // TQ4 matvec variants (row tiles 1 / 2 / 4, prefetch) are bit-identical per column
        const int tr = 1000, tk = 5120;
        std::vector<uint8_t> w(size_t(tr) * tk / 2);
        for (auto & v : w) v = uint8_t(byte(g));
        std::vector<__half> sc(size_t(tr) * tk / 128);
        for (auto & v : sc) v = __float2half_rn(0.001f * (1 + byte(g) % 50));
        auto * dw = dev(w);
        auto * ds = dev(sc);
        bool same = true;
        for (const int nc : {8, 5, 16}) {
            std::vector<__half> x(size_t(nc) * tk);
            for (auto & v : x) v = __float2half_rn(nd(g));
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, size_t(nc) * tr * 2);
            std::vector<bf16> ref;
            for (int v = 0; v < 10; ++v) {
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dy, tr, tk, nc, v, nullptr);
                const auto out = host(dy, size_t(nc) * tr);
                if (v == 0) ref = out;
                else same &= std::memcmp(ref.data(), out.data(), out.size() * 2) == 0;
            }
            cudaFree(dx), cudaFree(dy);
        }
        // width independence: a column's result at 10 / 12 / 16 columns equals its result from an 8-column (or 2-column) call
        {
            std::vector<__half> x(size_t(16) * tk);
            for (auto & v : x) v = __float2half_rn(nd(g));
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, size_t(16) * tr * 2);
            tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dy, tr, tk, 8, 0, nullptr);
            const auto lo = host(dy, size_t(8) * tr);
            tqllm_cuda::tq4_matvec_variant(dw, ds, dx + size_t(8) * tk, dy, tr, tk, 8, 0, nullptr);
            const auto hi = host(dy, size_t(8) * tr);
            for (const int nc : {10, 12, 16}) {
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dy, tr, tk, nc, 0, nullptr);
                const auto wide = host(dy, size_t(nc) * tr);
                same &= std::memcmp(wide.data(), lo.data(), size_t(8) * tr * 2) == 0;
                same &= std::memcmp(wide.data() + size_t(8) * tr, hi.data(), size_t(nc - 8) * tr * 2) == 0;
            }
            cudaFree(dx), cudaFree(dy);
        }
        // new TQ4 verify kernel (variants 20 / 21 = without / with weight prefetch): bit-identical to each other and width
        // independent at 1..16 columns; close to the previous MT 2 kernel (variant 2 here: different MMA grouping)
        {
            std::vector<__half> x(size_t(16) * tk);
            for (auto & v : x) v = __float2half_rn(nd(g));
            auto * dx = dev(x);
            bf16 *dy = nullptr, *dz = nullptr;
            cudaMalloc(&dy, size_t(16) * tr * 2), cudaMalloc(&dz, size_t(16) * tr * 2);
            std::vector<bf16> one(size_t(16) * tr);
            for (int c = 0; c < 16; ++c) {
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx + size_t(c) * tk, dy, tr, tk, 1, 20, nullptr);
                const auto col = host(dy, size_t(tr));
                std::memcpy(one.data() + size_t(c) * tr, col.data(), size_t(tr) * 2);
            }
            double vs_old = 0;
            bool same4 = true;
            for (int nc = 1; nc <= 16; ++nc) {
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dy, tr, tk, nc, 20, nullptr);
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dz, tr, tk, nc, 21, nullptr);
                const auto a = host(dy, size_t(nc) * tr), b = host(dz, size_t(nc) * tr);
                same4 &= std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dz, tr, tk, nc, 22, nullptr);
                same4 &= std::memcmp(host(dz, size_t(nc) * tr).data(), a.data(), a.size() * 2) == 0;
                same4 &= std::memcmp(a.data(), one.data(), a.size() * 2) == 0;
                tqllm_cuda::tq4_matvec_variant(dw, ds, dx, dz, tr, tk, nc, 2, nullptr);
                const auto o = host(dz, size_t(nc) * tr);
                for (size_t i = 0; i < o.size(); ++i) {
                    const double r = __bfloat162float(o[i]), q = __bfloat162float(a[i]);
                    vs_old = std::max(vs_old, std::fabs(r - q) / (std::fabs(r) + 1e-2));
                }
            }
            cudaFree(dx), cudaFree(dy), cudaFree(dz);
            const bool ok4 = same4 && vs_old < 2e-2;
            std::printf("%-24s %s (prefetch / width bit-identical %s, vs previous kernel worst rel %.2e)\n", "tq4_verify_kernel",
                        ok4 ? "PASS" : "FAIL", same4 ? "yes" : "NO", vs_old);
            if (!ok4) ++failures;
        }
        if (std::getenv("TQ_BENCH_TQ4")) {  // wide-column matvec timing at FFN shapes, variant 0 and 1 (prefetch)
            for (auto [br, bk] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
                std::vector<uint8_t> bw(size_t(br) * bk / 2);
                for (auto & v : bw) v = uint8_t(byte(g));
                std::vector<__half> bs(size_t(br) * bk / 128, __float2half_rn(0.001f));
                auto * dbw = dev(bw);
                auto * dbs = dev(bs);
                std::vector<__half> bx(size_t(16) * bk, __float2half_rn(0.01f));
                auto * dbx = dev(bx);
                bf16 * dby = nullptr;
                cudaMalloc(&dby, size_t(16) * br * 2);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0), cudaEventCreate(&e1);
                for (const int nc : {8, 10, 12, 16})
                    for (const int v : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) {
                        if (nc > 8 && v > 1) continue;
                        for (int i = 0; i < 10; ++i) tqllm_cuda::tq4_matvec_variant(dbw, dbs, dbx, dby, br, bk, nc, v, nullptr);
                        cudaEventRecord(e0);
                        for (int i = 0; i < 50; ++i) tqllm_cuda::tq4_matvec_variant(dbw, dbs, dbx, dby, br, bk, nc, v, nullptr);
                        cudaEventRecord(e1), cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        std::printf("bench tq4 %5d x %5d nc=%2d v=%d: %7.1f us\n", br, bk, nc, v, ms * 1000.f / 50.f);
                    }
                cudaFree(dbw), cudaFree(dbs), cudaFree(dbx), cudaFree(dby);
            }
        }
        cudaFree(dw), cudaFree(ds);
        std::printf("%-24s %s\n", "tq4_matvec_variants", same ? "PASS" : "FAIL");
        if (!same) ++failures;
    }
    {  // matvec rule table: the measured picks per (format bits, compute capability, shape), host-only lookups
        using tqllm_cuda::mvr::Kern;
        using tqllm_cuda::mvr::lookup;
        struct Case { int bits, cc, rows, k, nc; Kern kern; int pf; };
        const Case cases[] = {
            {3, 86, 34816, 5120, 8, Kern::Split16, 0},   // TQ3 gate/up, 3090
            {3, 86, 5120, 17408, 8, Kern::Split16, 1},   // TQ3 down, 3090
            {3, 120, 34816, 5120, 8, Kern::Split16, 0},  // TQ3 gate/up, 5060 Ti
            {3, 120, 5120, 17408, 8, Kern::Legacy, 0},   // TQ3 down, 5060 Ti
            {3, 121, 5120, 17408, 8, Kern::Legacy, 0},   // TQ3 down, GB10 (Blackwell rule)
            {3, 121, 34816, 5120, 8, Kern::Legacy, 0},   // TQ3 gate/up, GB10: legacy on every TQ3 shape
            {3, 121, 16384, 5120, 8, Kern::Legacy, 0},   // TQ3 short k, GB10
            {3, 121, 5120, 6144, 8, Kern::Legacy, 0},    // TQ3 o-proj / GDN out, GB10
            {4, 86, 34816, 5120, 8, Kern::Split16, 0},   // TQ4 gate/up
            {4, 86, 5120, 6144, 8, Kern::Legacy, 0},     // TQ4 o-proj / GDN out (few rows)
            {4, 86, 5120, 17408, 8, Kern::Legacy, 0},    // TQ4 down
            {14, 86, 34816, 5120, 8, Kern::Split16, 0},  // IQ4N gate/up
            {14, 86, 5120, 17408, 8, Kern::Split16, 0},  // IQ4N down
            {21, 86, 34816, 5120, 8, Kern::Split16, 0},  // IQ3 gate/up
            {21, 86, 5120, 17408, 8, Kern::Legacy, 0},   // IQ3 down: u4h kernel
            {3, 86, 34816, 5120, 24, Kern::Legacy, 0},   // > 16 columns: batched path
            {5, 86, 34816, 5120, 8, Kern::Legacy, 0},    // unknown format: no rule
        };
        bool ok = true;
        for (const auto & c : cases) {
            const auto p = lookup(c.bits, c.cc, c.rows, c.k, c.nc);
            const bool hit = p.kern == c.kern && (p.kern == Kern::Legacy || p.pf == c.pf);
            if (!hit) std::printf("   rule miss: bits %d cc %d %d x %d nc %d\n", c.bits, c.cc, c.rows, c.k, c.nc);
            ok &= hit;
        }
        std::printf("%-24s %s\n", "matvec_rules", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }
    {  // TQ3 (nibble-sliced 3-bit planes A0 / A1 / C per 32 weights, levels +-{14,45,79,127}) dequant + matvec vs FP64 CPU
        const int tr = 300, tk = 5120, nb = tk / 32;
        std::vector<uint32_t> w(size_t(tr) * nb * 3);
        std::vector<uint8_t> codes(size_t(tr) * tk);
        for (auto & c : codes) c = uint8_t(byte(g) % 8);
        for (int r = 0; r < tr; ++r)
            for (int b = 0; b < nb; ++b) {
                uint32_t a[2] = {0, 0}, cs = 0;  // nibble-sliced planes (kernels/formats/tq4.cu)
                for (int i = 0; i < 32; ++i) {
                    const uint32_t c = codes[size_t(r) * tk + b * 32 + i];
                    a[i / 16] |= (c & 3u) << (4 * (i % 8) + 2 * ((i / 8) % 2));
                    cs |= ((c >> 2) & 1u) << (4 * (i % 8) + i / 8);
                }
                uint32_t * o = &w[size_t(r) * nb * 3 + b];  // device layout: planes [A0 x nb][A1 x nb][C x nb]
                o[0] = a[0], o[nb] = a[1], o[2 * nb] = cs;
            }
        std::vector<__half> sc(size_t(tr) * tk / 128);
        for (auto & v : sc) v = __float2half_rn(0.001f * (1 + byte(g) % 50));
        static const int lv[8] = {14, 45, 79, 127, -14, -45, -79, -127};
        auto * dw = dev(w);
        auto * ds = dev(sc);
        __half * dq = nullptr;
        cudaMalloc(&dq, size_t(tr) * tk * 2);
        tqllm_cuda::tq4b3_dequant_h(reinterpret_cast<const uint8_t *>(dw), ds, dq, tr, tk, nullptr);
        const auto deq = host(dq, size_t(tr) * tk);
        bool ok = true;
        for (size_t i = 0; i < deq.size() && ok; ++i)
            ok = __half2float(deq[i]) == __half2float(__float2half_rn(float(lv[codes[i]]) * __half2float(sc[i / 128])));
        double worst = 0;
        for (const int nc : {1, 5, 8, 16, 32}) {
            std::vector<__half> x(size_t(nc) * tk);
            for (auto & v : x) v = __float2half_rn(nd(g));
            auto * dx = dev(x);
            float * dy = nullptr;
            cudaMalloc(&dy, size_t(nc) * tr * 4);
            tqllm_cuda::tq4b3_matvec(reinterpret_cast<const uint8_t *>(dw), ds, dx, dy, tr, tk, nc, nullptr);
            const auto y = host(dy, size_t(nc) * tr);
            for (int c = 0; c < nc; ++c)
                for (int r = 0; r < tr; ++r) {
                    double ref = 0, mag = 0;
                    for (int j = 0; j < tk; ++j) {
                        const double wv = __half2float(deq[size_t(r) * tk + j]), xv = __half2float(x[size_t(c) * tk + j]);
                        ref += wv * xv, mag += std::fabs(wv * xv);
                    }
                    worst = std::max(worst, std::fabs(double(y[size_t(c) * tr + r]) - ref) / (mag + 1e-9));
                }
            cudaFree(dx), cudaFree(dy);
        }
        ok = ok && worst < 2e-3;
        std::printf("%-24s %s (dequant exact, matvec worst rel %.2e)\n", "tq3_matvec_vs_cpu", ok ? "PASS" : "FAIL", worst);
        if (!ok) ++failures;
        // TQ3 verify kernel (tq3b_matvec_h_k): (a) the weight prefetch (variant 21) gives the same bits as no prefetch
        // (variant 20) at 1..16 columns; (b) width independence: a column's result does not depend on how many columns
        // are in the call (tree verify relies on it); (c) the default dispatch (auto prefetch) equals one of them;
        // (d) close to the previous block-per-lane kernel (variant 1; different MMA grouping, so not bit-equal).
        {
            using bf16 = __nv_bfloat16;
            bool same = true;
            double vs_old = 0;
            std::vector<__half> x(size_t(16) * tk);
            for (auto & v : x) v = __float2half_rn(nd(g));
            auto * dx = dev(x);
            bf16 *dy = nullptr, *dz = nullptr;
            cudaMalloc(&dy, size_t(16) * tr * 2), cudaMalloc(&dz, size_t(16) * tr * 2);
            const auto * w8 = reinterpret_cast<const uint8_t *>(dw);
            std::vector<bf16> one(size_t(16) * tr);  // every column computed alone (1-column calls)
            for (int c = 0; c < 16; ++c) {
                tqllm_cuda::tq4b3_matvec_variant(w8, ds, dx + size_t(c) * tk, dy, tr, tk, 1, 20, nullptr);
                const auto col = host(dy, size_t(tr));
                std::memcpy(one.data() + size_t(c) * tr, col.data(), size_t(tr) * 2);
            }
            for (int nc = 1; nc <= 16; ++nc) {
                tqllm_cuda::tq4b3_matvec_variant(w8, ds, dx, dy, tr, tk, nc, 20, nullptr);
                tqllm_cuda::tq4b3_matvec_variant(w8, ds, dx, dz, tr, tk, nc, 21, nullptr);
                const auto a = host(dy, size_t(nc) * tr), b = host(dz, size_t(nc) * tr);
                same &= std::memcmp(a.data(), b.data(), a.size() * 2) == 0;              // (a)
                tqllm_cuda::tq4b3_matvec_variant(w8, ds, dx, dz, tr, tk, nc, 22, nullptr);
                same &= std::memcmp(host(dz, size_t(nc) * tr).data(), a.data(), a.size() * 2) == 0;  // (a) two-step mode
                same &= std::memcmp(a.data(), one.data(), a.size() * 2) == 0;            // (b)
                tqllm_cuda::tq4b3_matvec_bf16_ld(w8, ds, dx, dz, tr, tk, nc, tr, nullptr);
                same &= std::memcmp(host(dz, size_t(nc) * tr).data(), a.data(), a.size() * 2) == 0;  // (c)
                tqllm_cuda::tq4b3_matvec_variant(w8, ds, dx, dz, tr, tk, nc, 1, nullptr);
                const auto o = host(dz, size_t(nc) * tr);
                for (size_t i = 0; i < o.size(); ++i) {
                    const double r = __bfloat162float(o[i]), q = __bfloat162float(a[i]);
                    vs_old = std::max(vs_old, std::fabs(r - q) / (std::fabs(r) + 1e-2));
                }
            }
            const bool close = vs_old < 2e-2;  // bf16 outputs: a few ulps of the output
            cudaFree(dx), cudaFree(dy), cudaFree(dz);
            std::printf("%-24s %s (prefetch / width / dispatch bit-identical %s, vs previous kernel worst rel %.2e)\n",
                        "tq3_verify_kernel", same && close ? "PASS" : "FAIL", same ? "yes" : "NO", vs_old);
            if (!(same && close)) ++failures;
        }
        cudaFree(dw), cudaFree(ds), cudaFree(dq);
    }
    {  // NVFP4 (modelopt) matvec vs FP64 CPU: E2M1 codes, E4M3 scale per 16, FP32 global scale
        const int nr = 300, nk = 2560;
        std::vector<uint8_t> w(size_t(nr) * nk / 2), sc(size_t(nr) * nk / 16);
        for (auto & v : w) v = uint8_t(byte(g));
        std::vector<float> scf(sc.size());
        for (size_t i = 0; i < sc.size(); ++i) {
            const __nv_fp8_e4m3 f(0.002f * (1 + byte(g) % 200));
            sc[i] = f.__x;
            scf[i] = float(f);
        }
        const float gs = 0.37f;
        static const float lv[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        bool ok = true;
        double worst = 0;
        for (const int nc : {1, 3, 8}) {
            std::vector<bf16> x(size_t(nc) * nk);
            for (auto & v : x) v = __float2bfloat16_rn(nd(g));
            auto * dw = dev(w);
            auto * ds = dev(sc);
            auto * dx = dev(x);
            float * dy = nullptr;
            cudaMalloc(&dy, size_t(nc) * nr * 4);
            tqllm_cuda::nvfp4_matvec(dw, ds, gs, dx, dy, nr, nk, nc, nullptr);
            const auto y = host(dy, size_t(nc) * nr);
            for (int c = 0; c < nc; ++c)
                for (int r = 0; r < nr; ++r) {
                    double ref = 0;
                    for (int i = 0; i < nk; ++i) {
                        const uint8_t b = w[size_t(r) * nk / 2 + i / 2];
                        const int code = (i & 1) ? b >> 4 : b & 15;
                        ref += double(lv[code]) * scf[size_t(r) * nk / 16 + i / 16] * __bfloat162float(x[size_t(c) * nk + i]);
                    }
                    ref *= gs;
                    const double e = std::fabs(y[size_t(c) * nr + r] - ref) / std::fmax(1.0, std::fabs(ref));
                    worst = std::fmax(worst, e);
                }
            ok &= worst < 1e-4;
            cudaFree(dw), cudaFree(ds), cudaFree(dx), cudaFree(dy);
        }
        std::printf("%-24s max_rel %.2e  %s\n", "nvfp4_matvec", worst, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }
    { // embedding row
        bf16 * dy = nullptr;
        cudaMalloc(&dy, k * 2);
        tqllm_cuda::tq3_row(dw, 123, dy, k, nullptr);
        const auto yb = host(dy, k);
        std::vector<float> y(k);
        std::vector<double> ref(k);
        for (int i = 0; i < k; ++i) {
            y[i] = __bfloat162float(yb[i]);
            ref[i] = wd[size_t(123) * k + i];
        }
        check("tq3_row", y, ref, 5e-3);
    }
    { // Q6_K
        const int qrows = 257, qk = 1024, qnb = qk / 256;
        std::vector<uint8_t> q(size_t(qrows) * qnb * 210);
        for (size_t b = 0; b < size_t(qrows) * qnb; ++b) {
            uint8_t * blk = &q[b * 210];
            for (int i = 0; i < 192; ++i) blk[i] = uint8_t(byte(g));
            for (int i = 0; i < 16; ++i) blk[192 + i] = uint8_t(int8_t(byte(g) % 64 - 32));
            const __half hd = __float2half(0.001f * (1 + b % 7));
            std::memcpy(blk + 208, &hd, 2);
        }
        std::vector<float> qd(size_t(qrows) * qk);
        for (size_t b = 0; b < size_t(qrows) * qnb; ++b) deq_q6k(&q[b * 210], &qd[b * 256]);
        auto * dq = dev(q);
        for (int ncols : {1, 5, 12}) {
            std::vector<float> x(size_t(ncols) * qk);
            for (auto & v : x) v = nd(g);
            std::vector<double> ref(size_t(ncols) * qrows);
            for (int n = 0; n < ncols; ++n)
                for (int r = 0; r < qrows; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qk; ++i) acc += double(qd[size_t(r) * qk + i]) * x[size_t(n) * qk + i];
                    ref[size_t(n) * qrows + r] = acc;
                }
            auto * dx = dev(x);
            float * dy = nullptr;
            cudaMalloc(&dy, ref.size() * 4);
            tqllm_cuda::q6k_matvec(dq, dx, dy, qrows, qk, ncols, nullptr);
            char name[32];
            std::snprintf(name, sizeof(name), "q6k_matvec_n%d", ncols);
            check(name, host(dy, ref.size()), ref, 1e-5);
        }
        { // int8 + fp16-scale LM head (tensor cores) vs the CPU reference, 1 and 8 columns
            int8_t * w8 = nullptr;
            __half * sc = nullptr;
            cudaMalloc(&w8, size_t(qrows) * qk);
            cudaMalloc(&sc, size_t(qrows) * qk / 16 * 2);
            tqllm_cuda::q6k_to_i8(dq, w8, sc, qrows, qk, nullptr);
            for (int ncols : {1, 8, 12}) {
                std::vector<bf16> x(size_t(ncols) * qk);
                for (auto & v : x) v = __float2bfloat16_rn(nd(g));
                std::vector<double> ref(size_t(ncols) * qrows);
                for (int n = 0; n < ncols; ++n)
                    for (int r = 0; r < qrows; ++r) {
                        double acc = 0;
                        for (int i = 0; i < qk; ++i) acc += double(qd[size_t(r) * qk + i]) * __bfloat162float(x[size_t(n) * qk + i]);
                        ref[size_t(n) * qrows + r] = acc;
                    }
                auto * dx = dev(x);
                __half * dh = nullptr;
                float * dy = nullptr;
                cudaMalloc(&dh, x.size() * 2);
                cudaMalloc(&dy, ref.size() * 4);
                tqllm_cuda::bf16_to_h(dx, dh, int(x.size()), nullptr);
                tqllm_cuda::i8h_matvec(w8, sc, dh, dy, qrows, qk, ncols, nullptr);
                char name[32];
                std::snprintf(name, sizeof(name), "i8h_lmhead_n%d", ncols);
                check(name, host(dy, ref.size()), ref, 2e-3);
            }
        }
        bf16 * dr = nullptr;  // embedding row
        cudaMalloc(&dr, qk * 2);
        tqllm_cuda::q6k_row(dq, 77, dr, qk, nullptr);
        const auto rb = host(dr, qk);
        std::vector<float> rv(qk);
        std::vector<double> rr(qk);
        for (int i = 0; i < qk; ++i) {
            rv[i] = __bfloat162float(rb[i]);
            rr[i] = qd[size_t(77) * qk + i];
        }
        check("q6k_row", rv, rr, 5e-3);
    }
    {  // Q4_K loader (GGUF 12 -> U4H) vs the host Q4_K dequantiser
        const int qr = 48, qc = 512;
        std::vector<char> raw(size_t(qr) * qc / 256 * 144);
        for (size_t b = 0; b < raw.size() / 144; ++b) {
            char * blk = raw.data() + b * 144;
            const __half d = __float2half(0.001f * (1 + byte(g) % 50)), mn = __float2half(0.002f * (1 + byte(g) % 50));
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &mn, 2);
            for (int i = 4; i < 144; ++i) blk[i] = char(byte(g));
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(12, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto * wf = tqllm::weight_format(12);
        const auto m = wf->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("q4k_loader_k4", got, rd, 2e-3);
        for (int nc : {1, 3, 8, 12}) {  // matvec through the loaded format vs the host dequant
            std::vector<__half> x(size_t(nc) * qc);
            for (auto & v : x) v = __float2half(nd(g));
            std::vector<double> ref2(size_t(nc) * qr);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < qr; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qc; ++i) acc += double(ref[size_t(r) * qc + i]) * __half2float(x[size_t(n) * qc + i]);
                    ref2[size_t(n) * qr + r] = acc;
                }
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, ref2.size() * 2);
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto yb = host(dy, ref2.size());
            std::vector<float> yv(yb.size());
            for (size_t i = 0; i < yv.size(); ++i) yv[i] = __bfloat162float(yb[i]);
            char nm[40];
            std::snprintf(nm, sizeof(nm), "q4k_%s_matvec_n%d", m.fmt->name(), nc);
            check(nm, yv, ref2, 5e-3);
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // Q5_K loader (GGUF 13 -> K5) vs the host Q5_K dequantiser: exact dequant, matvec n = 1..12
        const int qr = 48, qc = 768;
        std::vector<char> raw(size_t(qr) * qc / 256 * 176);
        for (size_t b = 0; b < raw.size() / 176; ++b) {
            char * blk = raw.data() + b * 176;
            const __half d = __float2half(0.001f * (1 + byte(g) % 50)), mn = __float2half(0.002f * (1 + byte(g) % 50));
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &mn, 2);
            for (int i = 4; i < 176; ++i) blk[i] = char(byte(g));
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(13, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto * wf = tqllm::weight_format(13);
        const auto m = wf->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("q5k_loader_k5", got, rd, 2e-3);
        for (int nc : {1, 3, 8, 12}) {  // matvec through the loaded format vs the host dequant
            std::vector<__half> x(size_t(nc) * qc);
            for (auto & v : x) v = __float2half(nd(g));
            std::vector<double> ref2(size_t(nc) * qr);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < qr; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qc; ++i) acc += double(ref[size_t(r) * qc + i]) * __half2float(x[size_t(n) * qc + i]);
                    ref2[size_t(n) * qr + r] = acc;
                }
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, ref2.size() * 2);
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto yb = host(dy, ref2.size());
            std::vector<float> yv(yb.size());
            for (size_t i = 0; i < yv.size(); ++i) yv[i] = __bfloat162float(yb[i]);
            char nm[40];
            std::snprintf(nm, sizeof(nm), "q5k_%s_matvec_n%d", m.fmt->name(), nc);
            check(nm, yv, ref2, 5e-3);
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // Q3_K loader (GGUF 11 -> K3): dequant and matvec vs the host Q3_K dequantiser
        const int qr = 40, qc = 768;
        std::vector<char> raw(size_t(qr) * qc / 256 * 110);
        for (size_t b = 0; b < raw.size() / 110; ++b) {
            char * blk = raw.data() + b * 110;
            for (int i = 0; i < 108; ++i) blk[i] = char(byte(g));
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
            std::memcpy(blk + 108, &d, 2);
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(11, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto m = tqllm::weight_format(11)->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("q3k_loader_k3", got, rd, 2e-3);
        for (int nc : {1, 3, 8, 12}) {
            std::vector<__half> x(size_t(nc) * qc);
            for (auto & v : x) v = __float2half(nd(g));
            std::vector<double> ref2(size_t(nc) * qr);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < qr; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qc; ++i) acc += double(ref[size_t(r) * qc + i]) * __half2float(x[size_t(n) * qc + i]);
                    ref2[size_t(n) * qr + r] = acc;
                }
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, ref2.size() * 2);
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto yb = host(dy, ref2.size());
            std::vector<float> yv(yb.size());
            for (size_t i = 0; i < yv.size(); ++i) yv[i] = __bfloat162float(yb[i]);
            char nm[40];
            std::snprintf(nm, sizeof(nm), "q3k_K3_matvec_n%d", nc);
            check(nm, yv, ref2, 5e-3);
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // IQ4_XS loader (GGUF 23 -> IQ4): dequant and matvec vs the host IQ4_XS dequantiser
        const int qr = 40, qc = 768;
        std::vector<char> raw(size_t(qr) * qc / 256 * 136);
        for (size_t b = 0; b < raw.size() / 136; ++b) {
            char * blk = raw.data() + b * 136;
            for (int i = 2; i < 136; ++i) blk[i] = char(byte(g));
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
            std::memcpy(blk, &d, 2);
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(23, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto m = tqllm::weight_format(23)->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("iq4xs_loader_iq4", got, rd, 2e-3);
        for (int nc : {1, 3, 8, 12}) {
            std::vector<__half> x(size_t(nc) * qc);
            for (auto & v : x) v = __float2half(nd(g));
            std::vector<double> ref2(size_t(nc) * qr);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < qr; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qc; ++i) acc += double(ref[size_t(r) * qc + i]) * __half2float(x[size_t(n) * qc + i]);
                    ref2[size_t(n) * qr + r] = acc;
                }
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, ref2.size() * 2);
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto yb = host(dy, ref2.size());
            std::vector<float> yv(yb.size());
            for (size_t i = 0; i < yv.size(); ++i) yv[i] = __bfloat162float(yb[i]);
            char nm[40];
            std::snprintf(nm, sizeof(nm), "iq4xs_IQ4_matvec_n%d", nc);
            check(nm, yv, ref2, 5e-3);
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // IQ3_S loader (GGUF 21 -> IQ3): dequant and matvec vs the host IQ3_S dequantiser
        const int qr = 40, qc = 768;
        std::vector<char> raw(size_t(qr) * qc / 256 * 110);
        for (size_t b = 0; b < raw.size() / 110; ++b) {
            char * blk = raw.data() + b * 110;
            for (int i = 2; i < 110; ++i) blk[i] = char(byte(g));
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
            std::memcpy(blk, &d, 2);
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(21, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto m = tqllm::weight_format(21)->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("iq3s_loader_iq3", got, rd, 2e-3);
        for (int nc : {1, 3, 8, 12}) {
            std::vector<__half> x(size_t(nc) * qc);
            for (auto & v : x) v = __float2half(nd(g));
            std::vector<double> ref2(size_t(nc) * qr);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < qr; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qc; ++i) acc += double(ref[size_t(r) * qc + i]) * __half2float(x[size_t(n) * qc + i]);
                    ref2[size_t(n) * qr + r] = acc;
                }
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, ref2.size() * 2);
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto yb = host(dy, ref2.size());
            std::vector<float> yv(yb.size());
            for (size_t i = 0; i < yv.size(); ++i) yv[i] = __bfloat162float(yb[i]);
            char nm[40];
            std::snprintf(nm, sizeof(nm), "iq3s_IQ3_matvec_n%d", nc);
            check(nm, yv, ref2, 5e-3);
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // IQ3_XXS loader (GGUF 18 -> IQ3X): dequant and matvec vs the host IQ3_XXS dequantiser
        const int qr = 40, qc = 768;
        std::vector<char> raw(size_t(qr) * qc / 256 * 98);
        for (size_t b = 0; b < raw.size() / 98; ++b) {
            char * blk = raw.data() + b * 98;
            for (int i = 2; i < 98; ++i) blk[i] = char(byte(g));
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
            std::memcpy(blk, &d, 2);
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(18, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto m = tqllm::weight_format(18)->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("iq3xxs_loader_iq3x", got, rd, 2e-3);
        for (int nc : {1, 3, 8, 12}) {
            std::vector<__half> x(size_t(nc) * qc);
            for (auto & v : x) v = __float2half(nd(g));
            std::vector<double> ref2(size_t(nc) * qr);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < qr; ++r) {
                    double acc = 0;
                    for (int i = 0; i < qc; ++i) acc += double(ref[size_t(r) * qc + i]) * __half2float(x[size_t(n) * qc + i]);
                    ref2[size_t(n) * qr + r] = acc;
                }
            auto * dx = dev(x);
            bf16 * dy = nullptr;
            cudaMalloc(&dy, ref2.size() * 2);
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto yb = host(dy, ref2.size());
            std::vector<float> yv(yb.size());
            for (size_t i = 0; i < yv.size(); ++i) yv[i] = __bfloat162float(yb[i]);
            char nm[40];
            std::snprintf(nm, sizeof(nm), "iq3xxs_IQ3X_matvec_n%d", nc);
            check(nm, yv, ref2, 5e-3);
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // Q6_K loader (GGUF 14 -> I8H) + I8H prefill dequant vs the host Q6_K dequantiser
        const int qr = 40, qc = 768;
        std::vector<char> raw(size_t(qr) * qc / 256 * 210);
        for (size_t b = 0; b < raw.size() / 210; ++b) {
            char * blk = raw.data() + b * 210;
            for (int i = 0; i < 192; ++i) blk[i] = char(byte(g));
            for (int i = 192; i < 208; ++i) blk[i] = char(byte(g) % 128 - 64);
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
            std::memcpy(blk + 208, &d, 2);
        }
        std::vector<float> ref(size_t(qr) * qc);
        tqllm::dequantize_host(14, raw.data(), ref.data(), int64_t(ref.size()));
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        const auto m = tqllm::weight_format(14)->load(raw, qr, qc, up, nullptr);
        __half * out = nullptr;
        cudaMalloc(&out, ref.size() * 2);
        m.fmt->dequant_h(m, out, nullptr);
        const auto hv = host(out, ref.size());
        std::vector<float> got(ref.size());
        std::vector<double> rd(ref.begin(), ref.end());
        for (size_t i = 0; i < got.size(); ++i) got[i] = __half2float(hv[i]);
        check("q6k_loader_i8h", got, rd, 2e-3);
        cudaFree(out);
        for (auto * p : keep) cudaFree(p);
    }
    {  // strided output (matvec_ld) of every Plain loader format + composite fused groups: bitwise vs plain matvec
        const int qc = 768;
        std::vector<void *> keep;
        const tqllm::Upload up = [&](const void * h, size_t n) {
            void * p = nullptr;
            cudaMalloc(&p, n);
            cudaMemcpy(p, h, n, cudaMemcpyHostToDevice);
            keep.push_back(p);
            return static_cast<const void *>(p);
        };
        auto make = [&](uint32_t ty, int r) {  // random blocks with sane FP16 scales
            const int bs = ty == 12 ? 144 : ty == 11 ? 110 : ty == 23 ? 136 : ty == 21 ? 110 : ty == 18 ? 98 : 210;
            const int doff = ty == 11 ? 108 : ty == 14 ? 208 : 0;
            std::vector<char> raw(size_t(r) * qc / 256 * bs);
            for (size_t b = 0; b < raw.size() / bs; ++b) {
                char * blk = raw.data() + b * bs;
                for (int i = 0; i < bs; ++i) blk[i] = char(byte(g));
                if (ty == 14)
                    for (int i = 192; i < 208; ++i) blk[i] = char(byte(g) % 128 - 64);
                const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
                std::memcpy(blk + doff, &d, 2);
                if (ty == 12) std::memcpy(blk + 2, &d, 2);  // dmin
            }
            return tqllm::weight_format(ty)->load(raw, r, qc, up, nullptr);
        };
        auto bits = [](const bf16 * p, size_t n) {
            const auto h = host(p, n);
            std::vector<uint16_t> b(n);
            std::memcpy(b.data(), h.data(), n * 2);
            return b;
        };
        int bad = 0;
        for (uint32_t ty : {12u, 11u, 23u, 21u, 18u, 14u}) {
            const int qr = 40, ld = qr + 24;
            const auto m = make(ty, qr);
            if (!m.fmt->has_matvec_ld()) { std::printf("FAIL ld_%u: no matvec_ld\n", ty); ++bad; continue; }
            for (int nc : {1, 3, 8, 12, 32}) {
                if (nc > m.fmt->max_batch_cols()) continue;
                std::vector<__half> x(size_t(nc) * qc);
                for (auto & v : x) v = __float2half(nd(g));
                auto * dx = dev(x);
                bf16 *y0 = nullptr, *y1 = nullptr;
                cudaMalloc(&y0, size_t(nc) * qr * 2);
                cudaMalloc(&y1, size_t(nc) * ld * 2);
                cudaMemset(y1, 0x7F, size_t(nc) * ld * 2);
                m.fmt->matvec(m, dx, y0, nc, nullptr);
                m.fmt->matvec_ld(m, dx, y1, nc, ld, nullptr);
                const auto a = bits(y0, size_t(nc) * qr), b = bits(y1, size_t(nc) * ld);
                int e = 0;
                for (int n = 0; n < nc; ++n)
                    for (int r = 0; r < ld; ++r)
                        e += r < qr ? b[size_t(n) * ld + r] != a[size_t(n) * qr + r] : b[size_t(n) * ld + r] != 0x7F7F;
                if (e) { std::printf("FAIL ld_%s_n%d: %d mismatches\n", m.fmt->name(), nc, e); ++bad; }
                cudaFree(dx);
                cudaFree(y0);
                cudaFree(y1);
            }
        }
        {  // composite K4 (Q4_K) + IQ3 (IQ3_S) + I8H (Q6_K): matvec / dequant equal to the parts', in row order
            const std::vector<tqllm::Mat> parts = {make(12, 48), make(21, 32), make(14, 16)};
            auto c = tqllm::composite_format(parts);
            const auto cm = tqllm::composite_mat(*c);
            const int R = cm.rows;
            if (R != 96 || c->max_batch_cols() != 8 || std::string(c->name()) != "K4+IQ3+I8H") {
                std::printf("FAIL composite shape: rows %d batch %d name %s\n", R, c->max_batch_cols(), c->name());
                ++bad;
            }
            for (int nc : {1, 3, 8}) {
                std::vector<__half> x(size_t(nc) * qc);
                for (auto & v : x) v = __float2half(nd(g));
                auto * dx = dev(x);
                bf16 *yc = nullptr, *yp = nullptr;
                cudaMalloc(&yc, size_t(nc) * R * 2);
                cudaMalloc(&yp, size_t(nc) * 48 * 2);
                c->matvec(cm, dx, yc, nc, nullptr);
                const auto bc = bits(yc, size_t(nc) * R);
                int e = 0, r0 = 0;
                for (const auto & p : parts) {
                    p.fmt->matvec(p, dx, yp, nc, nullptr);
                    const auto bp = bits(yp, size_t(nc) * p.rows);
                    for (int n = 0; n < nc; ++n)
                        for (int r = 0; r < p.rows; ++r) e += bc[size_t(n) * R + r0 + r] != bp[size_t(n) * p.rows + r];
                    r0 += p.rows;
                }
                if (e) { std::printf("FAIL composite_matvec_n%d: %d mismatches\n", nc, e); ++bad; }
                cudaFree(dx);
                cudaFree(yc);
                cudaFree(yp);
            }
            __half *dc = nullptr, *dp = nullptr;
            cudaMalloc(&dc, size_t(R) * qc * 2);
            cudaMalloc(&dp, size_t(48) * qc * 2);
            c->dequant_h(cm, dc, nullptr);
            const auto hc = host(dc, size_t(R) * qc);
            int e = 0, r0 = 0;
            for (const auto & p : parts) {
                p.fmt->dequant_h(p, dp, nullptr);
                const auto hp = host(dp, size_t(p.rows) * qc);
                e += std::memcmp(hc.data() + size_t(r0) * qc, hp.data(), size_t(p.rows) * qc * 2) != 0;
                r0 += p.rows;
            }
            if (e) { std::printf("FAIL composite_dequant: %d parts differ\n", e); ++bad; }
            cudaFree(dc);
            cudaFree(dp);
            const std::vector<tqllm::Mat> mixed_cols = {make(12, 16), tqllm::Mat{parts[1].p, 1006, 16, 512, parts[1].aux, parts[1].fmt}};
            if (tqllm::composite_format(mixed_cols)) { std::printf("FAIL composite accepted mismatched cols\n"); ++bad; }
        }
        std::printf("%s matvec_ld / composite (%d failures)\n", bad ? "FAIL" : "ok  ", bad);
        if (bad) failures += bad;
        for (auto * p : keep) cudaFree(p);
    }
    {  // Q3_K embedding rows (exact vs host) and Q5_K head -> int8 (within half an int8 step of the host Q5_K values)
        const int qc = 1024, nr = 12;
        int bad = 0;
        std::vector<char> q3(size_t(nr) * qc / 256 * 110), q5(size_t(nr) * qc / 256 * 176);
        for (size_t b = 0; b < q3.size() / 110; ++b) {
            for (int i = 0; i < 110; ++i) q3[b * 110 + i] = char(byte(g));
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40));
            std::memcpy(q3.data() + b * 110 + 108, &d, 2);
        }
        for (size_t b = 0; b < q5.size() / 176; ++b) {
            for (int i = 0; i < 176; ++i) q5[b * 176 + i] = char(byte(g));
            const __half d = __float2half(0.0005f * (1 + byte(g) % 40)), dm = __float2half(0.0003f * (1 + byte(g) % 40));
            std::memcpy(q5.data() + b * 176, &d, 2);
            std::memcpy(q5.data() + b * 176 + 2, &dm, 2);
        }
        std::vector<float> r3(size_t(nr) * qc), r5(size_t(nr) * qc);
        tqllm::dequantize_host(11, q3.data(), r3.data(), int64_t(r3.size()));
        tqllm::dequantize_host(13, q5.data(), r5.data(), int64_t(r5.size()));
        const std::vector<uint32_t> toks = {7, 0, 11, 3};
        auto * d3 = dev(q3);
        auto * dt = dev(toks);
        bf16 * ye = nullptr;
        cudaMalloc(&ye, toks.size() * qc * 2);
        tqllm_cuda::embed_multi_q3k(d3, dt, int(toks.size()), ye, qc, nullptr);
        const auto he = host(ye, toks.size() * qc);
        for (size_t n = 0; n < toks.size(); ++n)
            for (int i = 0; i < qc; ++i) {
                const float want = __bfloat162float(__float2bfloat16_rn(r3[size_t(toks[n]) * qc + i]));
                if (__bfloat162float(he[n * qc + i]) != want) ++bad;
            }
        auto * d5 = dev(q5);
        int8_t * w8 = nullptr;
        __half * s8 = nullptr;
        cudaMalloc(&w8, r5.size());
        cudaMalloc(&s8, r5.size() / 16 * 2);
        tqllm_cuda::q5k_to_i8(d5, w8, s8, nr, qc, nullptr);
        const auto hw = host(w8, r5.size());
        const auto hs = host(s8, r5.size() / 16);
        double worst = 0;
        for (size_t i = 0; i < r5.size(); ++i) {
            const float st = __half2float(hs[i / 16]);
            const double err = std::fabs(double(hw[i]) * st - r5[i]);
            worst = std::max(worst, err / std::max(1e-12, double(st)));
        }
        if (worst > 0.51) ++bad;
        std::printf("%s q3k_embed / q5k_to_i8 (%d failures, worst %.3f int8 steps)\n", bad ? "FAIL" : "ok  ", bad, worst);
        failures += bad;
        cudaFree(d3);
        cudaFree(dt);
        cudaFree(ye);
        cudaFree(d5);
        cudaFree(w8);
        cudaFree(s8);
    }
    {  // interleaved M-RoPE: t = h = w = p bitwise equal to the 1-D rotation; image positions vs a host reference
        const int n = 5, heads = 3, dim = 256, rot = 64, ts = heads * dim;
        const int3 sec = {11, 11, 10};
        std::vector<bf16> x(size_t(n) * ts);
        for (auto & v : x) v = __float2bfloat16_rn(nd(g));
        const int p0 = 4321;
        std::vector<int> pos = {p0, p0 + 1}, tab(size_t(n) * 4), tab2(size_t(n) * 4);
        for (int i = 0; i < n; ++i) {
            tab[4 * i] = -1;
            tab[4 * i + 1] = tab[4 * i + 2] = tab[4 * i + 3] = p0 + i;
            tab2[4 * i] = -1, tab2[4 * i + 1] = 100, tab2[4 * i + 2] = 100 + i, tab2[4 * i + 3] = 107 + 2 * i;
        }
        auto *a = dev(x), *b = dev(x), *c = dev(x);
        auto *dp = dev(pos), *dt = dev(tab), *dt2 = dev(tab2);
        tqllm_cuda::rope_partial_multi(a, ts, n, heads, dim, rot, dp, 1e7f, nullptr);
        tqllm_cuda::rope_partial_multi(b, ts, n, heads, dim, rot, dp, 1e7f, nullptr, nullptr, dt, sec);
        tqllm_cuda::rope_partial_multi(c, ts, n, heads, dim, rot, dp, 1e7f, nullptr, nullptr, dt2, sec);
        const auto ha = host(a, x.size()), hb = host(b, x.size()), hc = host(c, x.size());
        int bad = std::memcmp(ha.data(), hb.data(), x.size() * 2) != 0;
        double worst = 0;
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < heads; ++h)
                for (int j = 0; j < rot / 2; ++j) {
                    const int comp = (j % 3 == 1 && j < 33) ? 2 : (j % 3 == 2 && j < 30) ? 3 : 1;
                    const double ang = double(tab2[4 * i + comp]) / std::pow(1e7, 2.0 * j / rot);
                    const size_t o = size_t(i) * ts + size_t(h) * dim;
                    const double x0 = __bfloat162float(x[o + j]), x1 = __bfloat162float(x[o + j + rot / 2]);
                    worst = std::max(worst, std::fabs(__bfloat162float(hc[o + j]) - (x0 * std::cos(ang) - x1 * std::sin(ang))));
                    worst = std::max(worst, std::fabs(__bfloat162float(hc[o + j + rot / 2]) - (x0 * std::sin(ang) + x1 * std::cos(ang))));
                }
        if (worst > 0.05) ++bad;
        std::printf("%s mrope (text == 1-D bitwise, image worst %.3g)\n", bad ? "FAIL" : "ok  ", worst);
        failures += bad;
        for (void * p : {(void *) a, (void *) b, (void *) c, (void *) dp, (void *) dt, (void *) dt2}) cudaFree(p);
    }
    if (std::getenv("TQ_BENCH")) {  // micro-benchmark: n = 8 matvec, K4 vs IQ4 vs K3 (random blocks), two FFN shapes
        for (auto [rows, cols] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
            for (auto [ty, bb] : {std::pair{46u, 128}, std::pair{300u, 132}, std::pair{12u, 144}, std::pair{23u, 136}, std::pair{11u, 110}, std::pair{21u, 110}, std::pair{18u, 98}}) {
                std::vector<char> raw(size_t(rows) * cols / 256 * bb);
                for (auto & c : raw) c = char(byte(g));
                for (size_t b = 0; b < raw.size() / bb; ++b) {
                    const __half d = __float2half(0.001f);
                    std::memcpy(raw.data() + b * bb + (ty == 11 ? 108 : 0), &d, 2);
                }
                const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
                const auto m = tqllm::weight_format(ty)->load(raw, rows, cols, up, nullptr);
                std::vector<__half> x(size_t(8) * cols, __float2half(0.01f));
                auto * dx = dev(x);
                bf16 * dy = nullptr;
                cudaMalloc(&dy, size_t(8) * rows * 2);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0); cudaEventCreate(&e1);
                for (int i = 0; i < 20; ++i) m.fmt->matvec(m, dx, dy, 8, nullptr);
                cudaEventRecord(e0);
                for (int i = 0; i < 200; ++i) m.fmt->matvec(m, dx, dy, 8, nullptr);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                std::printf("bench %-4s %5d x %5d n=8: %7.1f us\n", m.fmt->name(), rows, cols, ms * 1000.f / 200.f);
            }
        }
    }
    {  // IQ4N: IQ4_XS on the 16-row split-K kernel vs the CPU IQ4_XS dequant; every width and prefetch mode bit-identical
        const int ir = 300, ik = 4096;
        const int64_t nsb = int64_t(ir) * ik / 256;
        std::uniform_int_distribution<int> u8(0, 255);
        std::vector<char> raw(size_t(nsb) * 136);
        for (int64_t b = 0; b < nsb; ++b) {
            auto * blk = reinterpret_cast<uint8_t *>(raw.data()) + b * 136;
            for (int i = 2; i < 136; ++i) blk[i] = uint8_t(u8(g));
            const __half d = __float2half(0.0005f + 0.0001f * float(b % 7));
            std::memcpy(blk, &d, 2);
        }
        std::vector<float> wd(size_t(ir) * ik);
        tqllm::dequantize_host(23, raw.data(), wd.data(), int64_t(wd.size()));
        const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
        const auto m = tqllm::weight_format(23)->load(raw, ir, ik, up, nullptr);
        const bool is_new = std::string(m.fmt->name()) == "IQ4N";
        std::vector<__half> x(size_t(16) * ik);
        for (auto & v : x) v = __float2half(0.5f * nd(g));
        std::vector<double> ref(size_t(16) * ir);
        for (int n = 0; n < 16; ++n)
            for (int r = 0; r < ir; ++r) {
                double acc = 0;
                for (int i = 0; i < ik; ++i) acc += double(wd[size_t(r) * ik + i]) * __half2float(x[size_t(n) * ik + i]);
                ref[size_t(n) * ir + r] = acc;
            }
        auto * dx = dev(x);
        bf16 * dy = nullptr;
        cudaMalloc(&dy, size_t(16) * ir * 2);
        m.fmt->matvec(m, dx, dy, 16, nullptr);
        const auto full = host(dy, size_t(16) * ir);
        std::vector<float> y(full.size());
        for (size_t i = 0; i < y.size(); ++i) y[i] = __bfloat162float(full[i]);
        check("iq4n_matvec_vs_cpu", y, ref, 5e-3);
        bool same = true;
        for (int nc = 1; nc <= 16; ++nc) {  // a column's result must not depend on the batch width
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto part = host(dy, size_t(nc) * ir);
            same &= std::memcmp(part.data(), full.data(), part.size() * sizeof(bf16)) == 0;
        }
        if (is_new)
            for (int pf : {0, 1, 2}) {
                tqllm_cuda::iq4n_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), dx, dy, ir,
                                             ik, 16, pf, nullptr);
                const auto o = host(dy, size_t(16) * ir);
                same &= std::memcmp(o.data(), full.data(), o.size() * sizeof(bf16)) == 0;
            }
        __half * dh = nullptr;
        cudaMalloc(&dh, size_t(ir) * ik * 2);
        m.fmt->dequant_h(m, dh, nullptr);
        const auto hd = host(dh, size_t(ir) * ik);
        double worst = 0;
        for (size_t i = 0; i < hd.size(); ++i)
            worst = std::max(worst, std::fabs(double(__half2float(hd[i])) - wd[i]) / (std::fabs(wd[i]) + 1e-3));
        const bool deq_ok = worst < 2e-3;
        if (!same || !deq_ok) ++failures;
        std::printf("%-24s %s (format %s, widths / prefetch bit-identical %s, dequant worst rel %.2e)\n", "iq4n_kernel",
                    same && deq_ok ? "PASS" : "FAIL", m.fmt->name(), same ? "yes" : "NO", worst);
        cudaFree(dx), cudaFree(dy), cudaFree(dh);
    }
    {  // IQ3 (IQ3_S) on the 16-row split-K kernel vs the CPU IQ3_S dequant; widths, prefetch modes and the old kernel
        const int ir = 300, ik = 4096;
        const int64_t nsb = int64_t(ir) * ik / 256;
        std::uniform_int_distribution<int> u8(0, 255);
        std::vector<char> raw(size_t(nsb) * 110);
        for (int64_t b = 0; b < nsb; ++b) {
            auto * blk = reinterpret_cast<uint8_t *>(raw.data()) + b * 110;
            for (int i = 2; i < 110; ++i) blk[i] = uint8_t(u8(g));
            const __half d = __float2half(0.0005f + 0.0001f * float(b % 7));
            std::memcpy(blk, &d, 2);
        }
        std::vector<float> wd(size_t(ir) * ik);
        tqllm::dequantize_host(21, raw.data(), wd.data(), int64_t(wd.size()));
        const tqllm::Upload up = [](const void * h, size_t n) { void * p = nullptr; cudaMalloc(&p, n); cudaMemcpy(p, h, n, cudaMemcpyHostToDevice); return static_cast<const void *>(p); };
        const auto m = tqllm::weight_format(21)->load(raw, ir, ik, up, nullptr);
        const bool is_new = true;
        std::vector<__half> x(size_t(16) * ik);
        for (auto & v : x) v = __float2half(0.5f * nd(g));
        std::vector<double> ref(size_t(16) * ir);
        for (int n = 0; n < 16; ++n)
            for (int r = 0; r < ir; ++r) {
                double acc = 0;
                for (int i = 0; i < ik; ++i) acc += double(wd[size_t(r) * ik + i]) * __half2float(x[size_t(n) * ik + i]);
                ref[size_t(n) * ir + r] = acc;
            }
        auto * dx = dev(x);
        bf16 * dy = nullptr;
        cudaMalloc(&dy, size_t(16) * ir * 2);
        m.fmt->matvec(m, dx, dy, 16, nullptr);
        const auto full = host(dy, size_t(16) * ir);
        std::vector<float> y(full.size());
        for (size_t i = 0; i < y.size(); ++i) y[i] = __bfloat162float(full[i]);
        check("iq3_split_vs_cpu", y, ref, 5e-3);
        bool same = true;
        for (int nc = 1; nc <= 16; ++nc) {  // a column's result must not depend on the batch width
            m.fmt->matvec(m, dx, dy, nc, nullptr);
            const auto part = host(dy, size_t(nc) * ir);
            same &= std::memcmp(part.data(), full.data(), part.size() * sizeof(bf16)) == 0;
        }
        if (is_new)
            for (int pf : {0, 1, 2}) {
                tqllm_cuda::iq3_matvec_split_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), dx, dy, ir,
                                             ik, 16, pf, nullptr);
                const auto o = host(dy, size_t(16) * ir);
                same &= std::memcmp(o.data(), full.data(), o.size() * sizeof(bf16)) == 0;
            }
        {  // the previous (u4h) kernel computes the same products in a different order: close, not identical
            tqllm_cuda::iq3_matvec_bf16(static_cast<const uint8_t *>(m.p), static_cast<const uint8_t *>(m.aux), dx, dy, ir, ik, 16,
                                        nullptr);
            const auto o = host(dy, size_t(16) * ir);
            std::vector<float> yo(o.size());
            for (size_t i = 0; i < yo.size(); ++i) yo[i] = __bfloat162float(o[i]);
            check("iq3_old_vs_cpu", yo, ref, 5e-3);
        }
        __half * dh = nullptr;
        cudaMalloc(&dh, size_t(ir) * ik * 2);
        m.fmt->dequant_h(m, dh, nullptr);
        const auto hd = host(dh, size_t(ir) * ik);
        double worst = 0;
        for (size_t i = 0; i < hd.size(); ++i)
            worst = std::max(worst, std::fabs(double(__half2float(hd[i])) - wd[i]) / (std::fabs(wd[i]) + 1e-3));
        const bool deq_ok = worst < 2e-3;
        if (!same || !deq_ok) ++failures;
        std::printf("%-24s %s (format %s, widths / prefetch bit-identical %s, dequant worst rel %.2e)\n", "iq3_split_kernel",
                    same && deq_ok ? "PASS" : "FAIL", m.fmt->name(), same ? "yes" : "NO", worst);
        cudaFree(dx), cudaFree(dy), cudaFree(dh);
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::printf("%s\n", failures ? "QUANT TESTS FAILED" : "all quant tests passed");
    return failures ? 1 : 0;
}

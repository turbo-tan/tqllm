// hc_combine_norm_rs_bf16 (one pass) must equal hc_combine followed by hc_norm_rs_bf16 bit for bit: the combined rows,
// the row scales and the bf16 image, including the in-place case (out aliases residual) the prefill path uses.
// hc_mix_rs_h (FP16 gate logits) must equal hc_mix_rs on the same logits widened to fp32, bit for bit, and its FP16 / BF16
// images the conversions of that output. The BF16-row forms (prefill wide rows) equal the fp32 kernels on the widened
// rows, their written rows rounded to BF16. hc_low_split_h must equal hc_low_silu then the FP16 conversion, plus the inject columns.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
template <typename T> static T * upload(const std::vector<T> & v) { T * d; check(cudaMalloc(&d, v.size() * sizeof(T))); check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <typename T> static std::vector<T> download(const T * d, size_t n) { std::vector<T> v(n); check(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
int main() {
    const int h = 2560, c = 4; bool ok = true;
    for (int n : {1, 37, 300}) for (bool in_place : {false, true}) {
        std::mt19937 rng(n * 2 + in_place); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> residual(size_t(n) * c * h), block(size_t(n) * h), inject(size_t(n) * c), gamma(size_t(c) * h);
        for (auto & v : residual) v = 3.f * nd(rng); for (auto & v : block) v = nd(rng); for (auto & v : inject) v = 2.f * nd(rng); for (auto & v : gamma) v = 1.f + 0.1f * nd(rng);
        float * dres = upload(residual), * dres2 = upload(residual), * dblock = upload(block), * dinject = upload(inject), * dgamma = upload(gamma);
        float * out_a, * out_b, * rs_a, * rs_b; __nv_bfloat16 * img_a, * img_b;
        check(cudaMalloc(&out_a, residual.size() * 4)); check(cudaMalloc(&out_b, residual.size() * 4));
        check(cudaMalloc(&rs_a, size_t(n) * c * 4)); check(cudaMalloc(&rs_b, size_t(n) * c * 4));
        check(cudaMalloc(&img_a, residual.size() * 2)); check(cudaMalloc(&img_b, residual.size() * 2));
        float * oa = in_place ? dres : out_a, * ob = in_place ? dres2 : out_b;
        tqllm_cuda::hc_combine(dres, dblock, dinject, oa, h, c, n, nullptr);
        tqllm_cuda::hc_norm_rs_bf16(oa, dgamma, rs_a, img_a, h, c, n, 1e-6f, nullptr);
        tqllm_cuda::hc_combine_norm_rs_bf16(dres2, dblock, dinject, ob, dgamma, rs_b, img_b, h, c, n, 1e-6f, nullptr);
        check(cudaDeviceSynchronize());
        const auto A = download(oa, residual.size()), B = download(ob, residual.size()), ra = download(rs_a, size_t(n) * c), rb = download(rs_b, size_t(n) * c);
        const auto ia = download(img_a, residual.size()), ib = download(img_b, residual.size());
        const bool same = std::memcmp(A.data(), B.data(), A.size() * 4) == 0 && std::memcmp(ra.data(), rb.data(), ra.size() * 4) == 0 &&
                          std::memcmp(ia.data(), ib.data(), ia.size() * 2) == 0;
        if (!same) std::printf("n %d in_place %d: differs\n", n, int(in_place));
        ok = ok && same;
        cudaFree(dres); cudaFree(dres2); cudaFree(dblock); cudaFree(dinject); cudaFree(dgamma); cudaFree(out_a); cudaFree(out_b); cudaFree(rs_a); cudaFree(rs_b); cudaFree(img_a); cudaFree(img_b);
    }
    for (int n : {1, 300}) {
        std::mt19937 rng(n + 11); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(size_t(n) * c * h), rs(size_t(n) * c), gamma(size_t(c) * h), lw(x.size());
        std::vector<__half> lh(x.size());
        for (auto & v : x) v = 3.f * nd(rng); for (auto & v : rs) v = 0.3f + 0.01f * nd(rng); for (auto & v : gamma) v = 1.f + 0.1f * nd(rng);
        for (size_t i = 0; i < x.size(); ++i) { lh[i] = __float2half_rn(4.f * nd(rng)); lw[i] = __half2float(lh[i]); }
        float * dx = upload(x), * drs = upload(rs), * dg = upload(gamma), * dlw = upload(lw), * oa, * ob; __half * dlh = upload(lh);
        check(cudaMalloc(&oa, size_t(n) * h * 4)); check(cudaMalloc(&ob, size_t(n) * h * 4));
        tqllm_cuda::hc_mix_rs(dx, drs, dg, dlw, oa, h, c, n, nullptr);
        __half * mh; check(cudaMalloc(&mh, size_t(n) * h * 2)); __nv_bfloat16 * mb; check(cudaMalloc(&mb, size_t(n) * h * 2));
        tqllm_cuda::hc_mix_rs_h(dx, drs, dg, dlh, ob, h, c, n, nullptr, mh, mb);
        check(cudaDeviceSynchronize());
        const auto A = download(oa, size_t(n) * h), B = download(ob, size_t(n) * h); const auto MH = download(mh, size_t(n) * h);
        const auto MB = download(mb, size_t(n) * h);
        bool same = std::memcmp(A.data(), B.data(), A.size() * 4) == 0;
        for (size_t i = 0; i < A.size() && same; ++i) {  // the images a conversion gives
            const __half e = __float2half_rn(A[i]); const __nv_bfloat16 eb = __float2bfloat16_rn(A[i]);
            same = std::memcmp(&e, &MH[i], 2) == 0 && std::memcmp(&eb, &MB[i], 2) == 0;
        }
        cudaFree(mh); cudaFree(mb);
        if (!same) std::printf("mix_rs_h n %d: differs\n", n);
        ok = ok && same;
        cudaFree(dx); cudaFree(drs); cudaFree(dg); cudaFree(dlw); cudaFree(dlh); cudaFree(oa); cudaFree(ob);
    }
    {   // fn_moe_hc_combine == fn_moe_combine then hc_combine in place
        const int n = 300, k = 10; std::mt19937 rng(17); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<__nv_bfloat16> r(size_t(n) * k * h); std::vector<float> w(size_t(n) * k), sh(size_t(n) * h), sw(n), inj(size_t(n) * c), wide(size_t(n) * c * h);
        for (auto & v : r) v = __float2bfloat16(nd(rng)); for (auto & v : w) v = 0.1f * nd(rng); for (auto & v : sh) v = nd(rng);
        for (auto & v : sw) v = nd(rng); for (auto & v : inj) v = 2.f * nd(rng); for (auto & v : wide) v = 3.f * nd(rng);
        __nv_bfloat16 * dr = upload(r); float * dw = upload(w), * dsh = upload(sh), * dsw = upload(sw), * dinj = upload(inj), * wa = upload(wide), * wb = upload(wide), * fo;
        check(cudaMalloc(&fo, size_t(n) * h * 4));
        tqllm_cuda::fn_moe_combine(dr, dw, dsh, dsw, fo, h, k, n, nullptr);
        tqllm_cuda::hc_combine(wa, fo, dinj, wa, h, c, n, nullptr);
        tqllm_cuda::fn_moe_hc_combine(dr, dw, dsh, dsw, dinj, wb, h, k, c, n, nullptr);
        check(cudaDeviceSynchronize());
        const auto A = download(wa, wide.size()), B = download(wb, wide.size());
        const bool same = std::memcmp(A.data(), B.data(), A.size() * 4) == 0;
        if (!same) std::printf("moe + hc combine: differs\n");
        ok = ok && same;
        cudaFree(dr); cudaFree(dw); cudaFree(dsh); cudaFree(dsw); cudaFree(dinj); cudaFree(wa); cudaFree(wb); cudaFree(fo);
    }
    for (int n : {1, 300}) {  // hc_low_split_h == hc_low_silu + FP16 conversion on the low columns, inject = the next c columns
        const int low = 36, ld = (low + c + 7) / 8 * 8; std::mt19937 rng(n + 29); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> y(size_t(n) * ld), lowr(size_t(n) * low);
        for (auto & v : y) v = 4.f * nd(rng);
        for (int t = 0; t < n; ++t) for (int j = 0; j < low; ++j) lowr[size_t(t) * low + j] = y[size_t(t) * ld + j];
        float * dy = upload(y), * dl = upload(lowr), * dinj; __half * dlh;
        check(cudaMalloc(&dinj, size_t(n) * c * 4)); check(cudaMalloc(&dlh, size_t(n) * low * 2));
        tqllm_cuda::hc_low_silu(dl, n * low, c, nullptr);
        tqllm_cuda::hc_low_split_h(dy, ld, low, c, dlh, dinj, n, nullptr);
        check(cudaDeviceSynchronize());
        const auto L = download(dl, lowr.size()); const auto LH = download(dlh, lowr.size()); const auto I = download(dinj, size_t(n) * c);
        bool same = true;
        for (size_t i = 0; i < L.size() && same; ++i) { const __half e = __float2half_rn(L[i]); same = std::memcmp(&e, &LH[i], 2) == 0; }
        for (int t = 0; t < n && same; ++t) for (int s = 0; s < c; ++s) same = same && I[size_t(t) * c + s] == y[size_t(t) * ld + low + s];
        if (!same) std::printf("low_split_h n %d: differs\n", n);
        ok = ok && same;
        cudaFree(dy); cudaFree(dl); cudaFree(dinj); cudaFree(dlh);
    }
    {   // BF16 wide rows: each kernel equals its fp32 form on the widened rows, the written rows are that result rounded
        const int n = 300, k = 10; std::mt19937 rng(41); std::normal_distribution<float> nd(0.f, 1.f);
        const size_t W = size_t(n) * c * h;
        std::vector<__nv_bfloat16> xb(W); std::vector<float> xf(W), gamma(size_t(c) * h), block(size_t(n) * h), inj(size_t(n) * c), rsv(size_t(n) * c);
        for (size_t i = 0; i < W; ++i) { xb[i] = __float2bfloat16_rn(3.f * nd(rng)); xf[i] = __bfloat162float(xb[i]); }
        for (auto & v : gamma) v = 1.f + 0.1f * nd(rng); for (auto & v : block) v = nd(rng); for (auto & v : inj) v = 2.f * nd(rng);
        for (auto & v : rsv) v = 0.3f + 0.01f * nd(rng);
        std::vector<__half> lg(W); for (auto & v : lg) v = __float2half_rn(4.f * nd(rng));
        __nv_bfloat16 * dxb = upload(xb); float * dxf = upload(xf), * dg = upload(gamma), * dblk = upload(block), * dinj = upload(inj), * drs = upload(rsv);
        __half * dlg = upload(lg);
        auto same = [](const void * a, const void * b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; };
        bool good = true;
        {   float * r1, * r2; __half * i1, * i2; check(cudaMalloc(&r1, size_t(n) * c * 4)); check(cudaMalloc(&r2, size_t(n) * c * 4));
            check(cudaMalloc(&i1, W * 2)); check(cudaMalloc(&i2, W * 2));
            tqllm_cuda::hc_norm_rs_h(dxf, dg, r1, i1, h, c, n, 1e-6f, nullptr); tqllm_cuda::hc_norm_rs_h(dxb, dg, r2, i2, h, c, n, 1e-6f, nullptr);
            check(cudaDeviceSynchronize());
            const bool a = same(download(r1, size_t(n) * c).data(), download(r2, size_t(n) * c).data(), size_t(n) * c * 4) && same(download(i1, W).data(), download(i2, W).data(), W * 2);
            if (!a) std::printf("bf16 norm_rs_h differs\n"); good = good && a;
            cudaFree(r1); cudaFree(r2); cudaFree(i1); cudaFree(i2); }
        {   float * m1, * m2; check(cudaMalloc(&m1, size_t(n) * h * 4)); check(cudaMalloc(&m2, size_t(n) * h * 4));
            tqllm_cuda::hc_mix_rs_h(dxf, drs, dg, dlg, m1, h, c, n, nullptr); tqllm_cuda::hc_mix_rs_h(dxb, drs, dg, dlg, m2, h, c, n, nullptr);
            check(cudaDeviceSynchronize());
            const bool a = same(download(m1, size_t(n) * h).data(), download(m2, size_t(n) * h).data(), size_t(n) * h * 4);
            if (!a) std::printf("bf16 mix_rs_h differs\n"); good = good && a;
            cudaFree(m1); cudaFree(m2); }
        {   // combine + norm into BF16 rows (in place, and from fp32 rows): rows = bf16(fp32 combine); rs / image = the norm of those rows
            float * of, * r1, * r2, * r3, * r4; __half * i1, * i2, * i3, * i4; __nv_bfloat16 * ob = upload(xb), * oc;
            check(cudaMalloc(&of, W * 4)); check(cudaMalloc(&oc, W * 2));
            for (float ** p : {&r1, &r2, &r3, &r4}) check(cudaMalloc(p, size_t(n) * c * 4));
            for (__half ** p : {&i1, &i2, &i3, &i4}) check(cudaMalloc(p, W * 2));
            tqllm_cuda::hc_combine_norm_rs_h(dxf, dblk, dinj, of, dg, r1, i1, h, c, n, 1e-6f, nullptr);
            const auto F = download(of, W); std::vector<__nv_bfloat16> Fb(W); std::vector<float> Fr(W);
            for (size_t i = 0; i < W; ++i) { Fb[i] = __float2bfloat16_rn(F[i]); Fr[i] = __bfloat162float(Fb[i]); }
            float * dfr = upload(Fr); tqllm_cuda::hc_norm_rs_h(dfr, dg, r2, i2, h, c, n, 1e-6f, nullptr);
            tqllm_cuda::hc_combine_norm_rs_h(ob, dblk, dinj, ob, dg, r3, i3, h, c, n, 1e-6f, nullptr);
            tqllm_cuda::hc_combine_norm_rs_h(dxf, dblk, dinj, oc, dg, r4, i4, h, c, n, 1e-6f, nullptr);
            check(cudaDeviceSynchronize());
            const auto R2 = download(r2, size_t(n) * c); const auto I2 = download(i2, W);
            bool a = same(download(ob, W).data(), Fb.data(), W * 2) && same(download(oc, W).data(), Fb.data(), W * 2);
            a = a && same(download(r3, size_t(n) * c).data(), R2.data(), R2.size() * 4) && same(download(i3, W).data(), I2.data(), W * 2);
            a = a && same(download(r4, size_t(n) * c).data(), R2.data(), R2.size() * 4) && same(download(i4, W).data(), I2.data(), W * 2);
            if (!a) std::printf("bf16 combine_norm_rs_h differs\n"); good = good && a;
            cudaFree(of); cudaFree(ob); cudaFree(oc); cudaFree(dfr);
            for (float * p : {r1, r2, r3, r4}) cudaFree(p); for (__half * p : {i1, i2, i3, i4}) cudaFree(p); }
        {   // moe + hc combine over BF16 rows == bf16(fp32 form on the widened rows); pack / unpack
            std::vector<__nv_bfloat16> r(size_t(n) * k * h); std::vector<float> w(size_t(n) * k), sh(size_t(n) * h), sw(n);
            for (auto & v : r) v = __float2bfloat16(nd(rng)); for (auto & v : w) v = 0.1f * nd(rng); for (auto & v : sh) v = nd(rng); for (auto & v : sw) v = nd(rng);
            __nv_bfloat16 * dr = upload(r), * wb = upload(xb), * pk; float * dw = upload(w), * dsh = upload(sh), * dsw = upload(sw), * wf = upload(xf), * up;
            check(cudaMalloc(&pk, W * 2)); check(cudaMalloc(&up, W * 4));
            tqllm_cuda::fn_moe_hc_combine(dr, dw, dsh, dsw, dinj, wf, h, k, c, n, nullptr);
            tqllm_cuda::fn_moe_hc_combine(dr, dw, dsh, dsw, dinj, wb, h, k, c, n, nullptr);
            tqllm_cuda::wide_pack(wf, pk, int64_t(W), nullptr); tqllm_cuda::wide_unpack(dxb, up, int64_t(W), nullptr);
            check(cudaDeviceSynchronize());
            const auto WF = download(wf, W); std::vector<__nv_bfloat16> e(W); for (size_t i = 0; i < W; ++i) e[i] = __float2bfloat16_rn(WF[i]);
            const bool a = same(download(wb, W).data(), e.data(), W * 2) && same(download(pk, W).data(), e.data(), W * 2) && same(download(up, W).data(), xf.data(), W * 4);
            if (!a) std::printf("bf16 moe_hc_combine / pack differs\n"); good = good && a;
            cudaFree(dr); cudaFree(wb); cudaFree(pk); cudaFree(dw); cudaFree(dsh); cudaFree(dsw); cudaFree(wf); cudaFree(up); }
        ok = ok && good;
        cudaFree(dxb); cudaFree(dxf); cudaFree(dg); cudaFree(dblk); cudaFree(dinj); cudaFree(drs); cudaFree(dlg);
    }
    std::printf(ok ? "HC_COMBINE_NORM PASS\n" : "HC_COMBINE_NORM FAIL\n");
    return ok ? 0 : 1;
}

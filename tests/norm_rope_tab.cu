// fn_norm_rope on a prefill block (n >= 64: the angle table once per token) must equal the per-row kernel (n < 64) on the
// same rows bit for bit: q (24 heads, interleaved with the gate) and k (2 heads), positions up to 100K.
#include "qwen4exp_kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
int main() {
    const int dim = 256, rot = 64; const float theta = 10000000.f, eps = 1e-6f; bool ok = true;
    for (int heads : {24, 2}) for (int pos : {0, 4093, 100000}) {
        const bool inter = heads == 24; const int n = 300;
        std::mt19937 rng(heads + pos); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(size_t(n) * heads * dim * (inter ? 2 : 1)), w(dim);
        for (auto & v : x) v = nd(rng); for (auto & v : w) v = 1.f + 0.1f * nd(rng);
        float * dx, * dw, * oa, * ob; const size_t out_n = size_t(n) * heads * dim;
        check(cudaMalloc(&dx, x.size() * 4)); check(cudaMalloc(&dw, w.size() * 4)); check(cudaMalloc(&oa, out_n * 4)); check(cudaMalloc(&ob, out_n * 4));
        check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        tqllm_cuda::fn_norm_rope(dx, dw, oa, heads, dim, n, pos, rot, theta, eps, inter, nullptr);
        for (int t = 0; t < n; t += 32) {
            const int m = std::min(32, n - t);
            tqllm_cuda::fn_norm_rope(dx + size_t(t) * heads * dim * (inter ? 2 : 1), dw, ob + size_t(t) * heads * dim, heads, dim, m, pos + t, rot, theta, eps, inter, nullptr);
        }
        check(cudaDeviceSynchronize());
        std::vector<float> A(out_n), B(out_n);
        check(cudaMemcpy(A.data(), oa, out_n * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(B.data(), ob, out_n * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(A.data(), B.data(), out_n * 4) == 0;
        std::printf("heads %d pos %d: %s\n", heads, pos, same ? "bit-identical" : "DIFFERENT");
        ok = ok && same;
        cudaFree(dx); cudaFree(dw); cudaFree(oa); cudaFree(ob);
    }
    {   // fn_gated_rms at 8 rows per block (n >= 64) == the per-row kernel (n < 64)
        const int heads = 32, d = 128, n = 300; std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(size_t(n) * heads * d), g(x.size()), w(d);
        for (auto & v : x) v = nd(rng); for (auto & v : g) v = nd(rng); for (auto & v : w) v = 1.f + 0.1f * nd(rng);
        float * dx, * dg, * dw, * oa, * ob;
        check(cudaMalloc(&dx, x.size() * 4)); check(cudaMalloc(&dg, g.size() * 4)); check(cudaMalloc(&dw, w.size() * 4));
        check(cudaMalloc(&oa, x.size() * 4)); check(cudaMalloc(&ob, x.size() * 4));
        check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        tqllm_cuda::fn_gated_rms(dx, dg, dw, oa, heads, d, n, 1e-6f, nullptr);
        for (int t = 0; t < n; t += 32) { const int m = std::min(32, n - t); const size_t o = size_t(t) * heads * d;
            tqllm_cuda::fn_gated_rms(dx + o, dg + o, dw, ob + o, heads, d, m, 1e-6f, nullptr); }
        check(cudaDeviceSynchronize());
        std::vector<float> A(x.size()), B(x.size());
        check(cudaMemcpy(A.data(), oa, A.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(B.data(), ob, B.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(A.data(), B.data(), A.size() * 4) == 0;
        std::printf("gated rms 8 rows per block: %s\n", same ? "bit-identical" : "DIFFERENT"); ok = ok && same;
        cudaFree(dx); cudaFree(dg); cudaFree(dw); cudaFree(oa); cudaFree(ob);
    }
    {   // fn_gated_rms_h / fn_attention_gate_h == the FP16 conversion of the fp32 kernels' outputs
        const int heads = 48, d = 128, n = 300; std::mt19937 rng(9); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(size_t(n) * heads * d), g(x.size()), w(d), qg(size_t(n) * heads * d * 2);
        for (auto & v : x) v = nd(rng); for (auto & v : g) v = nd(rng); for (auto & v : w) v = 1.f + 0.1f * nd(rng); for (auto & v : qg) v = nd(rng);
        float * dx, * dg, * dw, * dqg, * of; __half * oh;
        check(cudaMalloc(&dx, x.size() * 4)); check(cudaMalloc(&dg, g.size() * 4)); check(cudaMalloc(&dw, w.size() * 4)); check(cudaMalloc(&dqg, qg.size() * 4));
        check(cudaMalloc(&of, x.size() * 4)); check(cudaMalloc(&oh, x.size() * 2));
        check(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice)); check(cudaMemcpy(dqg, qg.data(), qg.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> F(x.size()); std::vector<__half> H(x.size()); bool same = true;
        tqllm_cuda::fn_gated_rms(dx, dg, dw, of, heads, d, n, 1e-6f, nullptr); tqllm_cuda::fn_gated_rms_h(dx, dg, dw, oh, heads, d, n, 1e-6f, nullptr);
        check(cudaDeviceSynchronize()); check(cudaMemcpy(F.data(), of, F.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(H.data(), oh, H.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < F.size() && same; ++i) { const __half e = __float2half_rn(F[i]); same = std::memcmp(&e, &H[i], 2) == 0; }
        std::printf("gated rms fp16 image: %s\n", same ? "bit-identical" : "DIFFERENT"); ok = ok && same;
        check(cudaMemcpy(of, dx, x.size() * 4, cudaMemcpyDeviceToDevice));
        tqllm_cuda::fn_attention_gate_h(of, dqg, oh, heads, d, n, nullptr); tqllm_cuda::fn_attention_gate(of, dqg, heads, d, n, nullptr);
        check(cudaDeviceSynchronize()); check(cudaMemcpy(F.data(), of, F.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(H.data(), oh, H.size() * 2, cudaMemcpyDeviceToHost));
        same = true; for (size_t i = 0; i < F.size() && same; ++i) { const __half e = __float2half_rn(F[i]); same = std::memcmp(&e, &H[i], 2) == 0; }
        std::printf("attention gate fp16 image: %s\n", same ? "bit-identical" : "DIFFERENT"); ok = ok && same;
        cudaFree(dx); cudaFree(dg); cudaFree(dw); cudaFree(dqg); cudaFree(of); cudaFree(oh);
    }
    {   // ple_conv / fn_gated_rms_h on FP16 inputs == the fp32 kernels on the same values widened
        const int ch = 10240, n = 300, taps = 4, hd = 48, d = 128; std::mt19937 rng(21); std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<__half> xh(size_t(n) * ch), gh(size_t(n) * hd * d); std::vector<float> xw(xh.size()), gw(gh.size()), hist(size_t(taps - 1) * ch), k(size_t(ch) * taps), r(gh.size()), w(d);
        for (size_t i = 0; i < xh.size(); ++i) { xh[i] = __float2half_rn(nd(rng)); xw[i] = __half2float(xh[i]); }
        for (size_t i = 0; i < gh.size(); ++i) { gh[i] = __float2half_rn(nd(rng)); gw[i] = __half2float(gh[i]); }
        for (auto & v : hist) v = nd(rng); for (auto & v : k) v = 0.3f * nd(rng); for (auto & v : r) v = nd(rng); for (auto & v : w) v = 1.f + 0.1f * nd(rng);
        auto up = [](const auto & v) { using T = typename std::decay_t<decltype(v)>::value_type; T * d; check(cudaMalloc(&d, v.size() * sizeof(T))); check(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; };
        __half * dxh = up(xh), * dgh = up(gh); float * dxw = up(xw), * dgw = up(gw), * dhist = up(hist), * dk = up(k), * dr = up(r), * dw = up(w);
        float * oa, * ob, * na, * nb; __half * ha, * hb;
        check(cudaMalloc(&oa, xw.size() * 4)); check(cudaMalloc(&ob, xw.size() * 4)); check(cudaMalloc(&na, hist.size() * 4)); check(cudaMalloc(&nb, hist.size() * 4));
        check(cudaMalloc(&ha, gh.size() * 2)); check(cudaMalloc(&hb, gh.size() * 2));
        tqllm_cuda::ple_conv(dxw, dhist, dk, oa, na, ch, n, taps, 1, nullptr); tqllm_cuda::ple_conv(dxh, dhist, dk, ob, nb, ch, n, taps, 1, nullptr);
        tqllm_cuda::fn_gated_rms_h(dr, dgw, dw, ha, hd, d, n, 1e-6f, nullptr); tqllm_cuda::fn_gated_rms_h(dr, dgh, dw, hb, hd, d, n, 1e-6f, nullptr);
        check(cudaDeviceSynchronize());
        std::vector<float> A(xw.size()), B(xw.size()), NA(hist.size()), NB(hist.size()); std::vector<__half> HA(gh.size()), HB(gh.size());
        check(cudaMemcpy(A.data(), oa, A.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(B.data(), ob, B.size() * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(NA.data(), na, NA.size() * 4, cudaMemcpyDeviceToHost)); check(cudaMemcpy(NB.data(), nb, NB.size() * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(HA.data(), ha, HA.size() * 2, cudaMemcpyDeviceToHost)); check(cudaMemcpy(HB.data(), hb, HB.size() * 2, cudaMemcpyDeviceToHost));
        const bool same = !std::memcmp(A.data(), B.data(), A.size() * 4) && !std::memcmp(NA.data(), NB.data(), NA.size() * 4) && !std::memcmp(HA.data(), HB.data(), HA.size() * 2);
        std::printf("conv + gated rms on fp16 inputs: %s\n", same ? "bit-identical" : "DIFFERENT"); ok = ok && same;
        cudaFree(dxh); cudaFree(dgh); cudaFree(dxw); cudaFree(dgw); cudaFree(dhist); cudaFree(dk); cudaFree(dr); cudaFree(dw); cudaFree(oa); cudaFree(ob); cudaFree(na); cudaFree(nb); cudaFree(ha); cudaFree(hb);
    }
    std::printf(ok ? "NORM_ROPE_TAB PASS\n" : "NORM_ROPE_TAB FAIL\n");
    return ok ? 0 : 1;
}

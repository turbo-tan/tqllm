// MiniMax kernels against the same math on the CPU: swigluoai, partial NeoX RoPE, causal GQA attention.
#include "minimax_kernels.cuh"
#include <cuda_fp8.h>
#include "minimax_memory.hpp"
#include "minimax_spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <class T> static T * dev(const std::vector<T> & h) {
    T * p = nullptr; ck(cudaMalloc(&p, h.size() * sizeof(T))); ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template <class T> static std::vector<T> host(const T * p, size_t n) {
    std::vector<T> h(n); ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }

int main() try {
    {  // memory guard: the KV it counts is the KV the engine allocates (K + V per KV head, indexer keys; FP16) - the
       // guard counted 2 x 1152 halves per token per layer, twice the 4 x 128 x 2 + 128 the buffers hold (10-24)
        const size_t ctx = 16384, kvh = 4, hd = 128, idim = 128;
        const size_t allocated = ctx * kvh * hd * 2 + ctx * kvh * hd * 2 + ctx * idim * 2;  // kc + vc + ikc
        require(minimax_kv_layer_bytes(ctx, kvh, hd, idim) == allocated, "guard KV bytes != allocated KV bytes");
        require(kMiniMaxHostReserve == size_t(2) << 30, "GB10 host reserve");
        {  // adaptive draft length: short at low acceptance, long at high, never 0, never past the block; the rate
           // estimate counts the failed trial of a step that stopped early
            require(minimax_draft_len(0.2, 7) <= 2 && minimax_draft_len(0.9, 7) >= 5 && minimax_draft_len(0.0, 7) == 1, "draft length vs acceptance");
            int prev = 0; bool mono = true;
            for (int i = 0; i <= 98; ++i) { const int k = minimax_draft_len(i / 100.0, 7); mono = mono && k >= prev && k <= 7; prev = k; }
            require(mono, "draft length grows with acceptance and stays within the block");
            MinimaxAcceptRate hi, lo, half;
            for (int i = 0; i < 60; ++i) { hi.update(3, 3); lo.update(3, 0); half.update(2, 1); }
            require(hi.rate() > 0.95 && lo.rate() < 0.05 && std::fabs(half.rate() - 0.5) < 0.02, "acceptance estimate (k of d, + 1 failed trial)");
        }
        {  // the expert-load guard: FP16 KV when it fits, FP8 when only that fits (GB10 32K), banks when neither
            const size_t G = size_t(1) << 30;
            require(minimax_expert_kv_elem(99 * G, 110 * G, 4 * G, 4 * G, 2 * G, false) == 2, "expert guard: FP16 KV fits");
            require(minimax_expert_kv_elem(99 * G, 106 * G, 4 * G, 4 * G, 2 * G, false) == 1, "expert guard: FP8 KV when FP16 does not fit");
            require(minimax_expert_kv_elem(99 * G, 110 * G, 4 * G, 4 * G, 2 * G, true) == 1, "expert guard: FP8 stays FP8");
            require(minimax_expert_kv_elem(99 * G, 104 * G, 4 * G, 4 * G, 2 * G, false) == 0, "expert guard: banks when neither fits");
        }
        require(!minimax_resident("token_embd.weight") && minimax_resident("output.weight") &&
                minimax_resident("blk.3.ffn_gate_exps.weight"), "the mapped embedding is not a resident tensor");
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    {  // swigluoai: both clamps and the +1 on up
        std::vector<float> g(4096), u(4096);
        for (auto & x : g) x = nd(rng);
        for (auto & x : u) x = nd(rng);
        g[0] = 9.f; u[0] = -9.f;  // clamped: (-7 + 1) * 7 * sigmoid(1.702 * 7)
        float * dg = dev(g), * du = dev(u); __half * dout = nullptr; ck(cudaMalloc(&dout, g.size() * 2));
        tqllm_cuda::mm_swigluoai_f32(dg, du, dout, int(g.size()), 1.702f, 7.f, nullptr);
        const auto o = host(dout, g.size());
        double worst = 0;
        for (size_t i = 0; i < g.size(); ++i) {
            const float gg = std::min(g[i], 7.f), uu = std::max(-7.f, std::min(u[i], 7.f));
            const float ref = (uu + 1.f) * gg / (1.f + std::exp(-1.702f * gg));
            worst = std::max(worst, double(std::fabs(__half2float(o[i]) - ref) / (std::fabs(ref) + 1e-2)));
        }
        require(worst < 2e-3, "swigluoai differs from the CPU math");
        require(std::fabs(__half2float(o[0]) - (-6.f * 7.f / (1.f + std::exp(-1.702f * 7.f)))) < 0.05f, "swigluoai clamps");
    }
    {  // RoPE: rotates dims [0, 64) of each 128-dim head in halves (j, j + 32); dims [64, 128) untouched
        const int T = 3, Hh = 2, D = 128, R = 64, pos0 = 5;
        const float theta = 1e7f;
        std::vector<float> x(size_t(T) * Hh * D);
        for (auto & v : x) v = nd(rng);
        float * dx = dev(x);
        tqllm_cuda::mm_rope(dx, T, Hh, D, R, pos0, theta, nullptr);
        const auto o = host(dx, x.size());
        double worst = 0;
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < Hh; ++h) {
                const float * a = &x[(size_t(t) * Hh + h) * D];
                const float * b = &o[(size_t(t) * Hh + h) * D];
                for (int i = 0; i < R / 2; ++i) {
                    const double f = double(pos0 + t) * std::pow(double(theta), -2.0 * i / R);
                    const double r0 = a[i] * std::cos(f) - a[i + R / 2] * std::sin(f), r1 = a[i + R / 2] * std::cos(f) + a[i] * std::sin(f);
                    worst = std::max({worst, std::fabs(b[i] - r0), std::fabs(b[i + R / 2] - r1)});
                }
                for (int i = R; i < D; ++i) require(b[i] == a[i], "rope touched a pass-through dim");
            }
        require(worst < 1e-4, "rope differs from the CPU math");
    }
    {  // causal GQA: 8 q heads over 2 kv heads, 3 new tokens at positions 4..6 over a 7-row cache
        const int n = 3, hq = 8, hk = 2, D = 128, pos0 = 4, rows = pos0 + n;
        std::vector<float> q(size_t(n) * hq * D), k(size_t(rows) * hk * D), v(k.size());
        for (auto & x : q) x = nd(rng) * 0.2f;
        for (auto & x : k) x = nd(rng) * 0.2f;
        for (auto & x : v) x = nd(rng);
        float * dq = dev(q), * dk = dev(k), * dv = dev(v), * dout = nullptr; ck(cudaMalloc(&dout, q.size() * 4));
        const float scale = 1.f / std::sqrt(float(D));
        tqllm_cuda::mm_attn(dq, dk, dv, dout, n, hq, hk, D, pos0, scale, nullptr);
        const auto o = host(dout, q.size());
        double worst = 0;
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < hq; ++h) {
                const int kh = h / (hq / hk), keys = pos0 + t + 1;
                std::vector<double> s(static_cast<size_t>(keys));
                double m = -1e30, z = 0;
                for (int j = 0; j < keys; ++j) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += double(q[(size_t(t) * hq + h) * D + i]) * k[(size_t(j) * hk + kh) * D + i];
                    s[size_t(j)] = d * scale; m = std::max(m, s[size_t(j)]);
                }
                for (auto & x : s) { x = std::exp(x - m); z += x; }
                for (int i = 0; i < D; ++i) {
                    double r = 0;
                    for (int j = 0; j < keys; ++j) r += s[size_t(j)] / z * v[(size_t(j) * hk + kh) * D + i];
                    worst = std::max(worst, std::fabs(o[(size_t(t) * hq + h) * D + i] - r));
                }
            }
        require(worst < 1e-4, "attention differs from the CPU math (or reads a future row)");
    }
    {  // tensor-core indexer scores (mm_index_scores_mma) against mm_index_scores: 37 tokens (not a multiple of 16), FP16
       // and FP8 keys; the same +-inf blocks, finite scores within FP16 rounding
        const int n = 37, ih = 4, D = 128, B = 128, pos0 = 2600, rows = pos0 + n, nb = (rows + B - 1) / B;
        std::vector<float> iq(size_t(n) * ih * D);
        std::vector<__half> ik(size_t(rows) * D);
        for (auto & x : iq) x = nd(rng) * 0.5f;
        for (auto & x : ik) x = __float2half(nd(rng) * 0.5f);
        std::vector<__nv_fp8_e4m3> i8(ik.size());
        for (size_t i = 0; i < ik.size(); ++i) i8[i] = __nv_fp8_e4m3(__half2float(ik[i]));
        float * diq = dev(iq), * da = nullptr, * db = nullptr;
        __half * dik = dev(ik);
        void * di8 = nullptr; ck(cudaMalloc(&di8, i8.size())); ck(cudaMemcpy(di8, i8.data(), i8.size(), cudaMemcpyHostToDevice));
        ck(cudaMalloc(&da, size_t(n) * nb * 4)); ck(cudaMalloc(&db, size_t(n) * nb * 4));
        for (const bool fp8 : {false, true}) {
            const void * k = fp8 ? di8 : static_cast<const void *>(dik);
            tqllm_cuda::mm_index_scores(diq, k, da, n, ih, D, pos0, B, nb, 1, nullptr, fp8);
            tqllm_cuda::mm_index_scores_mma(diq, k, db, n, ih, D, pos0, B, nb, 1, nullptr, fp8);
            const auto a = host(da, size_t(n) * nb), b = host(db, size_t(n) * nb);
            double w = 0; bool inf_ok = true;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::isinf(a[i]) || std::isinf(b[i])) { inf_ok = inf_ok && a[i] == b[i]; continue; }
                w = std::max(w, double(std::fabs(a[i] - b[i])) / std::max(1.0, double(std::fabs(a[i]))));
            }
            std::printf("mm_index_scores_mma %s: max rel |mma - scalar| %.2e, inf blocks %s\n", fp8 ? "fp8" : "fp16", w, inf_ok ? "same" : "DIFFER");
            require(inf_ok && w < 5e-3, "tensor-core indexer scores differ from index_scores_k");
        }
        for (void * pp : {static_cast<void *>(diq), static_cast<void *>(da), static_cast<void *>(db), static_cast<void *>(dik), di8}) ck(cudaFree(pp));
    }
    {  // tensor-core prompt attention (mm_attn_mma) against mm_attn_hg: 21 tokens (not a multiple of the 8-token group),
       // overlapping per-token block picks, dense and sparse, FP16 and FP8 KV; a token alone == the token in the chunk
        const int D = 128, B = 128, K = 16, hq = 32, hk = 2;
        for (const int dense : {0, 1}) {
            const int n = 21, pos0 = dense ? 100 : 2600, rows = pos0 + n, nb = (rows + B - 1) / B;
            std::vector<float> q(size_t(n) * hq * D);
            std::vector<__half> kc(size_t(rows) * hk * D), vc(kc.size());
            for (auto & x : q) x = nd(rng) * 0.1f;
            for (auto & x : kc) x = __float2half(nd(rng));
            for (auto & x : vc) x = __float2half(nd(rng));
            std::vector<int> ids(size_t(n) * K, -1);
            for (int t = 0; t < n && !dense; ++t) {  // own block, a shared core of 10 and 5 of its own (unsorted on purpose)
                const int qb = (pos0 + t) / B;
                std::vector<int> pick{qb};
                for (int c = 0; c < 10; ++c) pick.push_back((c * 2 + 1) % qb);
                for (int c = 0; pick.size() < size_t(K) && c < 1000; ++c) {
                    const int b = int(rng() % unsigned(qb));
                    if (std::find(pick.begin(), pick.end(), b) == pick.end()) pick.push_back(b);
                }
                std::sort(pick.begin(), pick.end()); pick.erase(std::unique(pick.begin(), pick.end()), pick.end());
                std::reverse(pick.begin(), pick.end());
                for (size_t r = 0; r < pick.size() && r < size_t(K); ++r) ids[size_t(t) * K + r] = pick[r];
            }
            float * dq = dev(q), * da = nullptr, * db = nullptr, * d1 = nullptr;
            __half * dk = dev(kc), * dv = dev(vc);
            int * dids = dev(ids);
            const int * sp = dense ? nullptr : dids;
            ck(cudaMalloc(&da, q.size() * 4)); ck(cudaMalloc(&db, q.size() * 4)); ck(cudaMalloc(&d1, size_t(hq) * D * 4));
            const float scale = 1.f / std::sqrt(float(D));
            tqllm_cuda::mm_attn_hg(dq, dk, dv, da, n, hq, hk, D, pos0, scale, sp, K, B, nullptr);
            tqllm_cuda::mm_attn_mma(dq, dk, dv, db, n, hq, hk, D, pos0, scale, sp, K, B, nullptr);
            const auto a = host(da, q.size()), b = host(db, q.size());
            double w = 0;
            for (size_t i = 0; i < a.size(); ++i) w = std::max(w, double(std::fabs(a[i] - b[i])));
            std::printf("mm_attn_mma %s: max |mma - hg| %.2e\n", dense ? "dense" : "sparse", w);
            require(w < 5e-3, dense ? "tensor-core dense attention differs from attn_hg" : "tensor-core sparse attention differs from attn_hg");
            for (const int t : {0, 9, n - 1}) {
                tqllm_cuda::mm_attn_mma(dq + size_t(t) * hq * D, dk, dv, d1, 1, hq, hk, D, pos0 + t, scale,
                                        sp ? dids + size_t(t) * K : nullptr, K, B, nullptr);
                const auto one = host(d1, size_t(hq) * D);
                require(std::memcmp(one.data(), b.data() + size_t(t) * hq * D, one.size() * 4) == 0,
                        "tensor-core attention: a token alone differs from the same token in the chunk");
            }
            {  // FP8 E4M3 KV == FP16 holding the same values, bit for bit
                std::vector<__nv_fp8_e4m3> k8(kc.size()), v8(vc.size());
                std::vector<__half> kr(kc.size()), vr(vc.size());
                for (size_t i = 0; i < kc.size(); ++i) { k8[i] = __nv_fp8_e4m3(__half2float(kc[i])); kr[i] = __float2half(float(k8[i])); }
                for (size_t i = 0; i < vc.size(); ++i) { v8[i] = __nv_fp8_e4m3(__half2float(vc[i])); vr[i] = __float2half(float(v8[i])); }
                void * dk8 = nullptr, * dv8 = nullptr;
                ck(cudaMalloc(&dk8, k8.size())); ck(cudaMalloc(&dv8, v8.size()));
                ck(cudaMemcpy(dk8, k8.data(), k8.size(), cudaMemcpyHostToDevice)); ck(cudaMemcpy(dv8, v8.data(), v8.size(), cudaMemcpyHostToDevice));
                __half * dkr = dev(kr), * dvr = dev(vr);
                tqllm_cuda::mm_attn_mma(dq, dk8, dv8, da, n, hq, hk, D, pos0, scale, sp, K, B, nullptr, true);
                tqllm_cuda::mm_attn_mma(dq, dkr, dvr, db, n, hq, hk, D, pos0, scale, sp, K, B, nullptr, false);
                require(std::memcmp(host(da, q.size()).data(), host(db, q.size()).data(), q.size() * 4) == 0,
                        "tensor-core FP8 attention differs from FP16 of the same values");
                for (void * pp : {dk8, dv8, static_cast<void *>(dkr), static_cast<void *>(dvr)}) ck(cudaFree(pp));
            }
            for (void * pp : {static_cast<void *>(dq), static_cast<void *>(da), static_cast<void *>(db), static_cast<void *>(d1),
                              static_cast<void *>(dk), static_cast<void *>(dv), static_cast<void *>(dids)}) ck(cudaFree(pp));
        }
    }
    {  // indexer + top-k + sparse FP16 attention at 20+ blocks of 128 against the HF semantics on the CPU
        const int n = 2, ih = 4, D = 128, B = 128, pos0 = 2600, rows = pos0 + n, nb = (rows + B - 1) / B, K = 16, hq = 32, hk = 2;  // 16 query heads per KV head, as MiniMax
        std::vector<float> iq(size_t(n) * ih * D), q(size_t(n) * hq * D);
        std::vector<__half> ik(size_t(rows) * D), kc(size_t(rows) * hk * D), vc(kc.size());
        for (auto & x : iq) x = nd(rng) * 0.1f;
        for (auto & x : q) x = nd(rng) * 0.1f;
        for (auto & x : ik) x = __float2half(nd(rng) * 0.1f);
        for (auto & x : kc) x = __float2half(nd(rng) * 0.1f);
        for (auto & x : vc) x = __float2half(nd(rng));
        float * diq = dev(iq), * dq = dev(q), * dsc = nullptr, * dout = nullptr; int * dids = nullptr;
        __half * dik = dev(ik), * dk = dev(kc), * dv = dev(vc);
        ck(cudaMalloc(&dsc, size_t(n) * nb * 4)); ck(cudaMalloc(&dids, size_t(n) * K * 4)); ck(cudaMalloc(&dout, q.size() * 4));
        tqllm_cuda::mm_index_scores(diq, dik, dsc, n, ih, D, pos0, B, nb, 1, nullptr);
        tqllm_cuda::mm_topk_blocks(dsc, dids, n, nb, K, nullptr);
        const float scale = 1.f / std::sqrt(float(D));
        tqllm_cuda::mm_attn_h(dq, dk, dv, dout, n, hq, hk, D, pos0, scale, dids, K, B, nullptr);
        {  // the prompt-chunk kernel (query heads of a KV head share staged K / V): same result, sparse and dense
            float * dg = nullptr; ck(cudaMalloc(&dg, q.size() * 4));
            for (const int * sparse : {static_cast<const int *>(dids), static_cast<const int *>(nullptr)}) {
                tqllm_cuda::mm_attn_h(dq, dk, dv, dout, n, hq, hk, D, pos0, scale, sparse, K, B, nullptr);
                tqllm_cuda::mm_attn_hg(dq, dk, dv, dg, n, hq, hk, D, pos0, scale, sparse, K, B, nullptr);
                const auto a = host(dout, q.size()), g = host(dg, q.size());
                double w = 0;
                for (size_t i = 0; i < a.size(); ++i) w = std::max(w, double(std::fabs(a[i] - g[i])));
                require(w < 1e-5, sparse ? "grouped sparse attention differs from attn_h" : "grouped dense attention differs from attn_h");
            }
            {  // grouped attention also serves decode / verify: the last token alone (n = 1) == that token in the batch
                tqllm_cuda::mm_attn_hg(dq, dk, dv, dg, n, hq, hk, D, pos0, scale, dids, K, B, nullptr);
                float * d1 = nullptr; ck(cudaMalloc(&d1, size_t(hq) * D * 4));
                tqllm_cuda::mm_attn_hg(dq + size_t(n - 1) * hq * D, dk, dv, d1, 1, hq, hk, D, pos0 + n - 1, scale,
                                       dids + size_t(n - 1) * K, K, B, nullptr);
                const auto all = host(dg, q.size()), one = host(d1, size_t(hq) * D);
                require(std::memcmp(one.data(), all.data() + size_t(n - 1) * hq * D, one.size() * 4) == 0,
                        "grouped attention: a token alone differs from the same token in a batch");
                ck(cudaFree(d1));
            }
            {  // FP8 E4M3 KV cache (long contexts): the same result as FP16 holding the FP8-rounded values, bit for bit
                std::vector<__nv_fp8_e4m3> k8(kc.size()), v8(vc.size()), i8(ik.size());
                std::vector<__half> kr(kc.size()), vr(vc.size()), ir(ik.size());
                for (size_t i = 0; i < kc.size(); ++i) { k8[i] = __nv_fp8_e4m3(__half2float(kc[i])); kr[i] = __float2half(float(k8[i])); }
                for (size_t i = 0; i < vc.size(); ++i) { v8[i] = __nv_fp8_e4m3(__half2float(vc[i])); vr[i] = __float2half(float(v8[i])); }
                for (size_t i = 0; i < ik.size(); ++i) { i8[i] = __nv_fp8_e4m3(__half2float(ik[i])); ir[i] = __float2half(float(i8[i])); }
                void * dk8 = nullptr, * dv8 = nullptr, * di8 = nullptr;
                ck(cudaMalloc(&dk8, k8.size())); ck(cudaMalloc(&dv8, v8.size())); ck(cudaMalloc(&di8, i8.size()));
                ck(cudaMemcpy(dk8, k8.data(), k8.size(), cudaMemcpyHostToDevice)); ck(cudaMemcpy(dv8, v8.data(), v8.size(), cudaMemcpyHostToDevice));
                ck(cudaMemcpy(di8, i8.data(), i8.size(), cudaMemcpyHostToDevice));
                __half * dkr = dev(kr), * dvr = dev(vr), * dir = dev(ir);
                float * s8 = nullptr, * sr = nullptr, * o8 = nullptr, * orr = nullptr;
                ck(cudaMalloc(&s8, size_t(n) * nb * 4)); ck(cudaMalloc(&sr, size_t(n) * nb * 4));
                ck(cudaMalloc(&o8, q.size() * 4)); ck(cudaMalloc(&orr, q.size() * 4));
                tqllm_cuda::mm_index_scores(diq, di8, s8, n, ih, D, pos0, B, nb, 1, nullptr, true);
                tqllm_cuda::mm_index_scores(diq, dir, sr, n, ih, D, pos0, B, nb, 1, nullptr, false);
                require(std::memcmp(host(s8, size_t(n) * nb).data(), host(sr, size_t(n) * nb).data(), size_t(n) * nb * 4) == 0,
                        "FP8 index scores differ from FP16 of the same values");
                for (int grouped = 0; grouped < 2; ++grouped) {
                    if (grouped) {
                        tqllm_cuda::mm_attn_hg(dq, dk8, dv8, o8, n, hq, hk, D, pos0, scale, dids, K, B, nullptr, true);
                        tqllm_cuda::mm_attn_hg(dq, dkr, dvr, orr, n, hq, hk, D, pos0, scale, dids, K, B, nullptr, false);
                    } else {
                        tqllm_cuda::mm_attn_h(dq, dk8, dv8, o8, n, hq, hk, D, pos0, scale, dids, K, B, nullptr, true);
                        tqllm_cuda::mm_attn_h(dq, dkr, dvr, orr, n, hq, hk, D, pos0, scale, dids, K, B, nullptr, false);
                    }
                    require(std::memcmp(host(o8, q.size()).data(), host(orr, q.size()).data(), q.size() * 4) == 0,
                            "FP8 attention differs from FP16 of the same values");
                }
                for (void * pp : {dk8, dv8, di8}) ck(cudaFree(pp));
                for (void * pp : {static_cast<void *>(dkr), static_cast<void *>(dvr), static_cast<void *>(dir), static_cast<void *>(s8),
                                  static_cast<void *>(sr), static_cast<void *>(o8), static_cast<void *>(orr)}) ck(cudaFree(pp));
            }
            ck(cudaFree(dg));
            tqllm_cuda::mm_attn_h(dq, dk, dv, dout, n, hq, hk, D, pos0, scale, dids, K, B, nullptr);
        }
        const auto sc = host(dsc, size_t(n) * nb);
        const auto ids = host(dids, size_t(n) * K);
        const auto o = host(dout, q.size());
        double worst_s = 0, worst_o = 0;
        for (int t = 0; t < n; ++t) {
            const int pos = pos0 + t, qb = pos / B;
            std::vector<double> bs(static_cast<size_t>(nb), -INFINITY);
            for (int j = 0; j <= pos; ++j)
                for (int h = 0; h < ih; ++h) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += double(iq[(size_t(t) * ih + h) * D + i]) * __half2float(ik[size_t(j) * D + i]);
                    bs[size_t(j / B)] = std::max(bs[size_t(j / B)], d);
                }
            bs[size_t(qb)] = INFINITY;
            for (int b = 0; b < nb; ++b)
                if (std::isfinite(bs[size_t(b)])) worst_s = std::max(worst_s, std::fabs(sc[size_t(t) * nb + b] - bs[size_t(b)]));
                else require(sc[size_t(t) * nb + b] == float(bs[size_t(b)]), "indexer: infinite block scores differ");
            std::vector<int> order(static_cast<size_t>(nb));
            for (int b = 0; b < nb; ++b) order[size_t(b)] = b;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return bs[size_t(a)] > bs[size_t(b)]; });
            std::vector<int> want(order.begin(), order.begin() + K), got(ids.begin() + t * K, ids.begin() + (t + 1) * K);
            std::sort(want.begin(), want.end()); std::sort(got.begin(), got.end());
            require(want == got, "top-k blocks differ from the CPU ranking");
            require(std::find(got.begin(), got.end(), qb) != got.end(), "the query's own block is not selected");
            for (int h = 0; h < hq; ++h) {
                const int kh = h / (hq / hk);
                std::vector<int> keys;
                for (int b : want) for (int j = b * B; j < (b + 1) * B && j <= pos; ++j) keys.push_back(j);
                std::vector<double> w(keys.size()); double m = -1e30, z = 0;
                for (size_t a = 0; a < keys.size(); ++a) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += double(q[(size_t(t) * hq + h) * D + i]) * __half2float(kc[(size_t(keys[a]) * hk + kh) * D + i]);
                    w[a] = d * scale; m = std::max(m, w[a]);
                }
                for (auto & x : w) { x = std::exp(x - m); z += x; }
                for (int i = 0; i < D; ++i) {
                    double r = 0;
                    for (size_t a = 0; a < keys.size(); ++a) r += w[a] / z * __half2float(vc[(size_t(keys[a]) * hk + kh) * D + i]);
                    worst_o = std::max(worst_o, std::fabs(o[(size_t(t) * hq + h) * D + i] - r));
                }
            }
        }
        require(worst_s < 1e-4, "indexer block scores differ from the CPU math");
        require(worst_o < 1e-4, "sparse attention differs from the CPU math");
    }
    std::printf("minimax kernels: PASS\n");
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}

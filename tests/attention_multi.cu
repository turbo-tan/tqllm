// fn_rope_table_multi + fn_norm_rope_table and fn_dense_attention_multi (several decode blocks in one launch) must give the
// same bits as fn_norm_rope and fn_dense_attention per sequence: query / key rows and attention outputs, mixed block sizes
// and positions (split counts 1..17, a sequence at position 0), at the Flash-Next shape.
#include "qwen4exp_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    const int heads = 24, kvheads = 2, dim = 256, rot = 64; const float theta = 1e7f, eps = 1e-6f;
    const size_t qrow = size_t(heads) * dim, krow = size_t(kvheads) * dim;
    const std::vector<std::vector<std::pair<int, int>>> cases = {  // (position, tokens) per sequence
        {{0, 1}, {5, 3}}, {{127, 1}, {128, 3}, {300, 2}, {1000, 4}}, {{2045, 3}, {0, 3}, {64, 1}}, {{17, 2}}, {{511, 3}, {512, 3}, {513, 3}, {1, 3}, {2000, 2}}};
    auto up = [](const std::vector<float> & v) { float * p; cudaMalloc(&p, v.size() * 4); cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice); return p; };
    auto down = [](const float * p, size_t n) { std::vector<float> v(n); cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost); return v; };
    std::vector<float> wq(dim), wk(dim);
    for (int i = 0; i < dim; ++i) { wq[i] = 1.f + .3f * std::sin(float(i)); wk[i] = .9f + .2f * std::cos(float(i) * .7f); }
    float * dwq = up(wq), * dwk = up(wk);
    bool ok = true;
    for (const auto & cs : cases) {
        int rows = 0; for (const auto & [p, n] : cs) rows += n;
        std::vector<float> qg(size_t(rows) * qrow * 2), k(size_t(rows) * krow);
        for (size_t i = 0; i < qg.size(); ++i) qg[i] = .8f * std::sin(float(i) * .0131f) + .1f * std::cos(float(i) * .00077f);
        for (size_t i = 0; i < k.size(); ++i) k[i] = .7f * std::cos(float(i) * .0219f);
        float * dqg = up(qg), * dk = up(k), * q1, * q2, * k1, * k2, * o1, * o2;
        for (float ** p : {&q1, &q2, &o1, &o2}) cudaMalloc(p, size_t(rows) * qrow * 4);
        for (float ** p : {&k1, &k2}) cudaMalloc(p, size_t(rows) * krow * 4);
        tqllm_cuda::AttnSeqs seqs; int off = 0;
        std::vector<__nv_bfloat16 *> caches;
        for (const auto & [pos, n] : cs) {
            const size_t len = size_t(pos + n) * krow;
            std::vector<__nv_bfloat16> kc(len), vc(len);
            for (size_t i = 0; i < len; ++i) { kc[i] = __float2bfloat16(std::sin(float(i) * .0071f + off)); vc[i] = __float2bfloat16(std::cos(float(i) * .0043f - off)); }
            __nv_bfloat16 * dkc, * dvc; cudaMalloc(&dkc, len * 2); cudaMalloc(&dvc, len * 2);
            cudaMemcpy(dkc, kc.data(), len * 2, cudaMemcpyHostToDevice); cudaMemcpy(dvc, vc.data(), len * 2, cudaMemcpyHostToDevice);
            caches.push_back(dkc); caches.push_back(dvc);
            tqllm_cuda::fn_norm_rope(dqg + size_t(off) * qrow * 2, dwq, q1 + size_t(off) * qrow, heads, dim, n, pos, rot, theta, eps, true, nullptr);
            tqllm_cuda::fn_norm_rope(dk + size_t(off) * krow, dwk, k1 + size_t(off) * krow, kvheads, dim, n, pos, rot, theta, eps, false, nullptr);
            float * ws; cudaMalloc(&ws, std::max<size_t>(1, tqllm_cuda::fn_dense_attention_workspace(heads, dim, n, pos)) * 4);
            tqllm_cuda::fn_dense_attention(q1 + size_t(off) * qrow, dkc, dvc, o1 + size_t(off) * qrow, heads, kvheads, dim, n, pos, nullptr, ws);
            cudaDeviceSynchronize(); cudaFree(ws);
            const int i = seqs.count++;
            seqs.off[i] = off; seqs.n[i] = n; seqs.pos[i] = pos; seqs.k[i] = dkc; seqs.v[i] = dvc; off += n;
        }
        float2 * tab; cudaMalloc(&tab, size_t(rows) * (rot / 2) * sizeof(float2));
        tqllm_cuda::fn_rope_table_multi(tab, seqs, rot, theta, nullptr);
        tqllm_cuda::fn_norm_rope_table(dqg, dwq, q2, tab, heads, dim, rows, rot, eps, true, nullptr);
        tqllm_cuda::fn_norm_rope_table(dk, dwk, k2, tab, kvheads, dim, rows, rot, eps, false, nullptr);
        float * ws; cudaMalloc(&ws, tqllm_cuda::fn_dense_attention_multi_workspace(heads, dim, seqs) * 4);
        tqllm_cuda::fn_dense_attention_multi(q2, seqs, o2, heads, kvheads, dim, ws, nullptr);
        cudaDeviceSynchronize();
        const auto a1 = down(q1, size_t(rows) * qrow), a2 = down(q2, a1.size()), b1 = down(k1, size_t(rows) * krow), b2 = down(k2, b1.size());
        const auto c1 = down(o1, size_t(rows) * qrow), c2 = down(o2, c1.size());
        const bool same_q = !std::memcmp(a1.data(), a2.data(), a1.size() * 4), same_k = !std::memcmp(b1.data(), b2.data(), b1.size() * 4),
                   same_o = !std::memcmp(c1.data(), c2.data(), c1.size() * 4);
        if (!same_q || !same_k || !same_o) {
            std::printf("FAIL: %d sequences, %d rows: query %s, key %s, attention %s\n", seqs.count, rows, same_q ? "same" : "differs",
                        same_k ? "same" : "differs", same_o ? "same" : "differs");
            ok = false;
        }
        for (auto * p : caches) cudaFree(p);
        for (float * p : {dqg, dk, q1, q2, k1, k2, o1, o2, ws}) cudaFree(p);
        cudaFree(tab);
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ok = false; }
    std::puts(ok ? "attention multi: PASS (one launch for several sequences bit-identical to per-sequence norm / RoPE and attention)" : "attention multi: FAIL");
    return ok ? 0 : 1;
}

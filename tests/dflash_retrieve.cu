// DFlash retrieval window (dflash_retrieve_build): (1) planted far chunks are selected, in ascending order;
// (2) retrieved keys equal keys RoPE-rotated directly to their new (scratch) positions; (3) when retrieval picks the
// chunks right before the recent part, attention over the scratch equals windowed attention over the full cache.
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char * what, double v = 0) {
    std::printf("%-70s %.6f %s\n", what, v, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}
constexpr int KVH = 8, HEADS = 32, HD = 128, KD = KVH * HD, QD = HEADS * HD;
constexpr float THETA = 1e7f;
void rope_cpu(float * x, int p) {  // NeoX halves, as rope_partial_multi with rot = dim = 128 and no YaRN
    for (int d = 0; d < 64; ++d) {
        const float ang = float(p) * (1.f / std::pow(THETA, float(2 * d) / 128.f));
        const float c = __bfloat162float(__float2bfloat16_rn(std::cos(ang))), s = __bfloat162float(__float2bfloat16_rn(std::sin(ang)));
        const float a = x[d], b = x[d + 64];
        x[d] = a * c - b * s;
        x[d + 64] = b * c + a * s;
    }
}
}  // namespace

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.f, 1.f);
    const int pos = 6000, nq = 1, recent = 1536, F = pos - recent, nc = F / 16, P = pos + nq;
    std::vector<float> qf(QD);
    for (auto & v : qf) v = g(rng);
    std::vector<float> qm(KVH * HD, 0.f);  // anchor query mean per KV head
    for (int h = 0; h < KVH; ++h)
        for (int gq = 0; gq < HEADS / KVH; ++gq)
            for (int d = 0; d < HD; ++d) qm[h * HD + d] += qf[(h * 4 + gq) * HD + d] / 4.f;
    auto build = [&](const std::vector<int> & planted, std::vector<float> & raw) {
        raw.assign(size_t(P) * KD, 0.f);
        for (auto & v : raw) v = g(rng) * 0.3f;
        for (int c : planted)
            for (int t = 0; t < 16; ++t)
                for (int h = 0; h < KVH; ++h)
                    for (int d = 0; d < HD; ++d) raw[size_t(c * 16 + t) * KD + h * HD + d] += 3.f * qm[h * HD + d];
    };
    auto upload = [&](const std::vector<float> & raw, std::vector<__nv_bfloat16> & kh, std::vector<__nv_bfloat16> & vh) {
        kh.resize(raw.size()); vh.resize(raw.size());
        std::vector<float> row(HD);
        for (int p = 0; p < P; ++p)
            for (int h = 0; h < KVH; ++h) {
                for (int d = 0; d < HD; ++d) row[d] = raw[size_t(p) * KD + h * HD + d];
                rope_cpu(row.data(), p);  // cache keys are stored rotated at their positions
                for (int d = 0; d < HD; ++d) {
                    kh[size_t(p) * KD + h * HD + d] = __float2bfloat16_rn(row[d]);
                    vh[size_t(p) * KD + h * HD + d] = __float2bfloat16_rn(g(rng));
                }
            }
    };
    __nv_bfloat16 *q, *kc, *vc, *sk, *sv, *o1, *o2;
    float *score, *part;
    int *sel, *dp;
    const int rows = 128 * 16 + recent + nq;
    cudaMalloc(&q, QD * 2); cudaMalloc(&kc, size_t(P) * KD * 2); cudaMalloc(&vc, size_t(P) * KD * 2);
    cudaMalloc(&sk, size_t(rows) * KD * 2); cudaMalloc(&sv, size_t(rows) * KD * 2);
    cudaMalloc(&o1, QD * 2); cudaMalloc(&o2, QD * 2); cudaMalloc(&score, KVH * 4096 * 4); cudaMalloc(&sel, KVH * 128 * 4);
    cudaMalloc(&dp, 8);
    const int pf = tqllm_cuda::attention_gen_partial_floats(HEADS, P + 64, nq);
    cudaMalloc(&part, size_t(pf) * 4);
    std::vector<__nv_bfloat16> qh(QD);
    for (int i = 0; i < QD; ++i) qh[i] = __float2bfloat16_rn(qf[i]);
    cudaMemcpy(q, qh.data(), QD * 2, cudaMemcpyHostToDevice);

    // (1) + (2): far planted chunks
    {
        const std::vector<int> planted = {7, 40, 120, 200};
        std::vector<float> raw;
        std::vector<__nv_bfloat16> kh, vh;
        build(planted, raw);
        upload(raw, kh, vh);
        cudaMemcpy(kc, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(vc, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice);
        const int R = tqllm_cuda::dflash_retrieve_build(q, kc, vc, KD, HEADS, KVH, pos, nq, recent, 4, THETA, score, sel, sk, sv, nullptr);
        cudaDeviceSynchronize();
        std::vector<int> hs(KVH * 4);
        cudaMemcpy(hs.data(), sel, hs.size() * 4, cudaMemcpyDeviceToHost);
        bool all = R == 64;
        for (int h = 0; h < KVH; ++h)
            for (int j = 0; j < 4; ++j) all &= hs[h * 4 + j] == planted[size_t(j)];
        check(all, "planted far chunks selected per head, ascending");
        std::vector<__nv_bfloat16> skh(size_t(rows) * KD);
        cudaMemcpy(skh.data(), sk, skh.size() * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        std::vector<float> row(HD);
        for (int i = 0; i < R; ++i)
            for (int h = 0; h < KVH; ++h) {
                const int src = planted[size_t(i / 16)] * 16 + i % 16, vp = F - R + i;
                for (int d = 0; d < HD; ++d) row[d] = raw[size_t(src) * KD + h * HD + d];
                rope_cpu(row.data(), vp);
                for (int d = 0; d < HD; ++d)
                    maxd = std::max(maxd, double(std::fabs(__bfloat162float(skh[size_t(i) * KD + h * HD + d]) - row[d])) / (std::fabs(row[d]) + 1.0));
            }
        check(maxd < 0.03, "re-rotated keys == keys rotated directly to the new position (max rel)", maxd);
        bool recent_ok = true;
        for (int i = R; i < R + recent + nq; ++i)
            for (int k = 0; k < KD; ++k)
                recent_ok &= __bfloat162float(skh[size_t(i) * KD + k]) == __bfloat162float(kh[size_t(F + i - R) * KD + k]);
        check(recent_ok, "recent part copied unchanged");
    }
    // (3): contiguous case equals windowed attention on the full cache
    {
        std::vector<float> raw;
        std::vector<__nv_bfloat16> kh, vh;
        build({nc - 4, nc - 3, nc - 2, nc - 1}, raw);
        upload(raw, kh, vh);
        cudaMemcpy(kc, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(vc, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice);
        const int R = tqllm_cuda::dflash_retrieve_build(q, kc, vc, KD, HEADS, KVH, pos, nq, recent, 4, THETA, score, sel, sk, sv, nullptr);
        const int hp[2] = {R + recent, pos};
        cudaMemcpy(dp, hp, 8, cudaMemcpyHostToDevice);
        const int W = R + recent + 1;
        const float scale = 1.f / std::sqrt(float(HD));
        tqllm_cuda::attention_gen128(q, sk, sv, o1, part, dp, nq, (R + recent + nq + 63) / 64 * 64, HEADS, KVH, scale, false, W, nullptr, 0);
        tqllm_cuda::attention_gen128(q, kc, vc, o2, part, dp + 1, nq, (P + 63) / 64 * 64, HEADS, KVH, scale, false, W, nullptr, 0);
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> a(QD), b(QD);
        cudaMemcpy(a.data(), o1, QD * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), o2, QD * 2, cudaMemcpyDeviceToHost);
        double maxd = 0;
        for (int i = 0; i < QD; ++i) maxd = std::max(maxd, double(std::fabs(__bfloat162float(a[i]) - __bfloat162float(b[i]))));
        check(R == 64 && maxd < 2e-2, "contiguous retrieval == windowed attention on the full cache (max abs diff)", maxd);
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); return 1; }
    if (failures) return 1;
    std::puts("dflash_retrieve: ok");
    return 0;
}

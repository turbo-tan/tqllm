// HeadSampler (src/spec/head_sampler.hpp) on a synthetic two-stage head: a random int8 head, its 4-bit copy (u4h_from_i8h)
// as the first pass, the engine's top-16 + int8 re-score. Every screened draw must equal sample_reference on all logits
// (computed with rescore_all, whose bits must match i8h_rescore_argmax's), for greedy-like, top-p, top-k, whole-vocabulary
// and penalised parameters; also with an exact first pass (no error table, logits read instead of re-scored).
#include "formats/weight/quant_kernels.cuh"
#include "spec/head_sampler.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const int rows = 30000, k = 1024, nc = 6;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> w8(size_t(rows) * k);
    std::vector<__half> sc(size_t(rows) * k / 16);
    for (auto & v : w8) v = int8_t(std::max(-127.f, std::min(127.f, std::round(nd(rng) * 40.f))));
    for (auto & s : sc) s = __float2half(0.0008f * (0.5f + std::abs(nd(rng))));
    int8_t * dw8; __half * dsc, * dx; uint8_t * dw4; __half2 * dsm; float * da, * dfull, * dex, * dvals; uint32_t * did, * dtok; void * scr;
    cudaMalloc(&dw8, w8.size()); cudaMalloc(&dsc, sc.size() * 2); cudaMalloc(&dw4, size_t(rows) * k / 2);
    cudaMalloc(&dsm, size_t(rows) * k / 32 * 4); cudaMalloc(&dx, size_t(nc) * k * 2); cudaMalloc(&da, size_t(nc) * rows * 4);
    cudaMalloc(&dfull, size_t(nc) * rows * 4); cudaMalloc(&dex, nc * 16 * 4); cudaMalloc(&did, nc * 16 * 4);
    cudaMalloc(&dtok, nc * 4); cudaMalloc(&dvals, nc * 16 * 4); cudaMalloc(&scr, size_t(nc) * 64 * 16 * 8);
    cudaMemcpy(dw8, w8.data(), w8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dsc, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice);
    tqllm_cuda::u4h_from_i8h(dw8, dsc, dw4, dsm, int64_t(rows) * k, nullptr);
    cudaStream_t st;
    cudaStreamCreate(&st);
    tqllm::HeadSampler screened, exact;
    screened.init(rows, k, nc, tqllm::HeadSampler::FirstPass::U4h, dsm, st);
    exact.init(rows, k, nc, tqllm::HeadSampler::FirstPass::Exact, nullptr, st);
    int fails = 0, cases = 0;
    const float temps[] = {0.2f, 0.7f, 1.f, 1.5f};
    const float top_ps[] = {0.5f, 0.8f, 0.95f, 1.f};
    const int top_ks[] = {0, 5, 40};
    for (int trial = 0; trial < 6; ++trial) {
        // inputs: column c's logit spread grows with c (peaked to flat distributions)
        std::vector<__half> x(size_t(nc) * k);
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < k; ++i) x[size_t(c) * k + size_t(i)] = __float2half(nd(rng) * (0.6f + 0.5f * float(c + trial % 3)));
        cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        tqllm_cuda::u4h_matvec(dw4, dsm, dx, da, rows, k, nc, st);
        tqllm_cuda::topk16_big(da, rows, nc, dvals, did, scr, st);
        tqllm_cuda::i8h_rescore_argmax(dw8, dsc, dx, did, k, nc, dtok, st, dex);
        screened.save_x(dx, nc, st);
        screened.rescore_all(dw8, dsc, dx, nc, dfull, st);
        std::vector<float> full(size_t(nc) * rows), ex16(nc * 16);
        std::vector<uint32_t> id16(nc * 16);
        cudaMemcpyAsync(full.data(), dfull, full.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(ex16.data(), dex, ex16.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(id16.data(), did, id16.size() * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        for (int c = 0; c < nc; ++c)
            for (int j = 0; j < 16; ++j)
                if (std::memcmp(&ex16[size_t(c) * 16 + j], &full[size_t(c) * rows + id16[size_t(c) * 16 + j]], 4)) {
                    std::printf("FAIL: rescore_all bits differ from i8h_rescore_argmax (col %d)\n", c);
                    ++fails;
                }
        // the exact sampler's "first pass" is the exact logits themselves, with their top-16
        std::vector<uint32_t> eid(nc * 16);
        std::vector<float> eex(nc * 16);
        for (int c = 0; c < nc; ++c) {
            std::vector<uint32_t> o(rows);
            for (int r = 0; r < rows; ++r) o[size_t(r)] = uint32_t(r);
            const float * f = full.data() + size_t(c) * rows;
            std::partial_sort(o.begin(), o.begin() + 16, o.end(), [&](uint32_t a, uint32_t b) { return f[a] > f[b]; });
            for (int j = 0; j < 16; ++j) eid[size_t(c) * 16 + j] = o[size_t(j)], eex[size_t(c) * 16 + j] = f[o[size_t(j)]];
        }
        uint32_t * deid; float * deex;
        cudaMalloc(&deid, eid.size() * 4); cudaMalloc(&deex, eex.size() * 4);
        cudaMemcpy(deid, eid.data(), eid.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(deex, eex.data(), eex.size() * 4, cudaMemcpyHostToDevice);
        for (float t : temps) for (float tp : top_ps) for (int tk : top_ks) for (int pen = 0; pen < 2; ++pen) {
            tqllm::SamplingParams sp{t, tp, pen ? 0.8f : 0.f, tk};
            sp.repeat = pen ? 1.2f : 1.f;
            sp.seed = rng();
            std::vector<float> tv(nc, t);
            screened.queue_stats(da, did, dex, 0, nc, tv.data(), st);
            exact.queue_stats(dfull, deid, deex, 0, nc, tv.data(), st);
            cudaStreamSynchronize(st);
            std::vector<tqllm::HeadSampleCol> cols;
            for (int c = 0; c < nc; ++c) {
                tqllm::HeadSampleCol hc{c, uint32_t(100 + c + trial * 7), {}};
                if (pen) for (int j = 0; j < 30; ++j) hc.recent.push_back(id16[size_t(c) * 16 + size_t(j % 8)] + uint32_t(j / 8));
                cols.push_back(hc);
            }
            const auto a = screened.sample(cols, sp, da, dw8, dsc, st);
            const auto b = exact.sample(cols, sp, dfull, nullptr, nullptr, st);
            for (int c = 0; c < nc; ++c) {
                const uint32_t want = tqllm::sample_reference(full.data() + size_t(c) * rows, rows, sp, cols[size_t(c)].recent,
                                                              tqllm::gumbel_key(sp.seed, cols[size_t(c)].pos));
                cases += 2;
                if (a[size_t(c)] != want || b[size_t(c)] != want) {
                    ++fails;
                    std::printf("FAIL: T %.1f top_p %.2f top_k %d pen %d col %d: screened %u exact %u reference %u\n", t, tp,
                                tk, pen, c, a[size_t(c)], b[size_t(c)], want);
                }
            }
        }
        cudaFree(deid); cudaFree(deex);
    }
    if (cudaGetLastError() != cudaSuccess) { std::puts("FAIL: CUDA error"); ++fails; }
    for (const auto * h : {&screened}) {
        const auto & s = h->stats;
        std::printf("  %s first pass: %llu columns, %llu by the top-16, %llu x 64, %llu x 1024, %llu x 8192, %llu Gumbel, %llu full\n",
                    "u4h", (unsigned long long) s.cols, (unsigned long long) s.round1,
                    (unsigned long long) s.rounds64, (unsigned long long) s.rounds1024, (unsigned long long) s.rounds8192,
                    (unsigned long long) s.gumbel, (unsigned long long) s.full);
    }
    std::printf("head sampler: %s (%d draws, %d mismatches)\n", fails ? "FAIL" : "PASS", cases, fails);
    return fails ? 1 : 0;
}

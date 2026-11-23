// QSA indexer scores on tensor cores (score_bits_tc_k, bf16 hi + lo) against the fp32 CUDA-core kernel (score_bits_k):
//   - every complete block's score within 1e-4 relative plus 2e-5 absolute (a dot of terms summing to ~3) of the fp32 one, incomplete blocks untouched
//   - the top-k (1024 blocks, the prefill's need cap) differs only where the two scores at the boundary tie within that
//     tolerance; the count of queries whose selected set differs is printed
#include "qwen4exp_kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }
template <class T> struct Device {
    T * p = nullptr; size_t n;
    explicit Device(size_t n_) : n(n_) { ck(cudaMalloc(&p, std::max<size_t>(1, n) * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { ck(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> x(n); ck(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
static float f(uint32_t b) { float v; std::memcpy(&v, &b, 4); return v; }

static void run(int blocks, int tokens, int first, int ratio, uint32_t seed, int & diff_sets, int & queries, double & worst) {
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> keys(size_t(blocks) * 128), q(size_t(tokens) * 512);
    for (int b = 0; b < blocks; ++b) {  // normalized keys, as keys_.normalized() holds them
        double s = 0; for (int i = 0; i < 128; ++i) { keys[size_t(b) * 128 + i] = nd(rng); s += double(keys[size_t(b) * 128 + i]) * keys[size_t(b) * 128 + i]; }
        for (int i = 0; i < 128; ++i) keys[size_t(b) * 128 + i] = float(keys[size_t(b) * 128 + i] / std::sqrt(s));
    }
    for (auto & v : q) v = nd(rng) * 0.3f;
    Device<float> dk(keys.size()), dq(q.size()); dk.put(keys); dq.put(q);
    const uint32_t poison = 0xdeadbeefu;
    Device<uint32_t> s0(size_t(tokens) * blocks), s1(size_t(tokens) * blocks);
    ck(cudaMemset(s0.p, 0xef, s0.n * 4)); ck(cudaMemset(s1.p, 0xef, s1.n * 4));
    tqllm_cuda::qsa_score_bits(dk.p, dq.p, s0.p, blocks, tokens, first, ratio, nullptr, 0);
    tqllm_cuda::qsa_score_bits(dk.p, dq.p, s1.p, blocks, tokens, first, ratio, nullptr, 1);
    ck(cudaDeviceSynchronize());
    const auto a = s0.get(), b = s1.get();
    for (int t = 0; t < tokens; ++t) {
        const int complete = std::min(blocks, (first + t + 1) / ratio);
        for (int k = 0; k < blocks; ++k) {
            const size_t i = size_t(t) * blocks + k;
            if (k >= complete) { require(a[i] == 0xefefefefu && b[i] == 0xefefefefu, "an incomplete block was written"); continue; }
            const double x = f(a[i]), y = f(b[i]), e = std::fabs(x - y);
            worst = std::max(worst, e);
            require(std::fabs(x - y) <= 1e-4 * std::fabs(x) + 2e-5, "score off: t " + std::to_string(t) + " block " + std::to_string(k) +
                    " fp32 " + std::to_string(x) + " tc " + std::to_string(y));
        }
    }
    (void) poison;
    constexpr int cap = 1024;
    Device<uint64_t> o0(size_t(tokens) * cap), o1(size_t(tokens) * cap);
    ck(cudaMemset(o0.p, 0, o0.n * 8)); ck(cudaMemset(o1.p, 0, o1.n * 8));
    tqllm_cuda::qsa_topk_bits(s0.p, o0.p, blocks, tokens, first, ratio, cap, nullptr);
    tqllm_cuda::qsa_topk_bits(s1.p, o1.p, blocks, tokens, first, ratio, cap, nullptr);
    ck(cudaDeviceSynchronize());
    const auto ka = o0.get(), kb = o1.get();
    for (int t = 0; t < tokens; ++t) {
        const int need = std::min((first + t + 1) / ratio, cap);
        std::vector<uint32_t> x, y;
        for (int i = 0; i < need; ++i) { x.push_back(~uint32_t(ka[size_t(t) * cap + i])); y.push_back(~uint32_t(kb[size_t(t) * cap + i])); }
        std::sort(x.begin(), x.end()); std::sort(y.begin(), y.end());
        ++queries;
        if (x == y) continue;
        ++diff_sets;
        // a different set is a near tie at the boundary: the smallest fp32 score taken by each side within tolerance
        std::vector<uint32_t> only;
        std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(only));
        double lo = 1e30, hi = -1e30;
        for (uint32_t k : only) { const double v = f(a[size_t(t) * blocks + k]); lo = std::min(lo, v); hi = std::max(hi, v); }
        require(hi - lo <= 2e-4 * std::fabs(hi) + 4e-5, "top-k differs beyond a near tie at query " + std::to_string(t));
    }
}

int main() {
    int dev = 0, major = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 8) { std::printf("qsa score tc: skipped (sm < 80)\n"); return 77; }
    int diff = 0, queries = 0; double worst = 0;
    run(1037, 48, 0, 4, 1, diff, queries, worst);          // ragged tiles, first blocks incomplete
    run(8192, 64, 32700, 4, 2, diff, queries, worst);      // a 32K-position prefill block
    run(30710, 32, 122800, 4, 3, diff, queries, worst);    // the end of a 128K prompt
    // speed at the 128K end, 2048 queries (one prefill sub-chunk)
    {
        const int blocks = 30710, tokens = 2048, first = 122838 - 2048;
        Device<float> dk(size_t(blocks) * 128), dq(size_t(tokens) * 512); Device<uint32_t> s(size_t(tokens) * blocks);
        ck(cudaMemset(dk.p, 0, dk.n * 4)); ck(cudaMemset(dq.p, 0, dq.n * 4));
        for (int tc = 0; tc < 2; ++tc) {
            tqllm_cuda::qsa_score_bits(dk.p, dq.p, s.p, blocks, tokens, first, 4, nullptr, tc); ck(cudaDeviceSynchronize());
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 5; ++r) tqllm_cuda::qsa_score_bits(dk.p, dq.p, s.p, blocks, tokens, first, 4, nullptr, tc);
            ck(cudaDeviceSynchronize());
            std::printf("qsa score %s: %.2f ms per 2048 queries x %d blocks\n", tc ? "tc" : "fp32",
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5, blocks);
        }
    }
    std::printf("qsa score tc: PASS (worst absolute %.2e, top-k sets differing at near ties: %d of %d queries)\n", worst, diff, queries);
    return 0;
}

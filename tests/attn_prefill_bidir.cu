// attention_mq256_prefill (causal) and attention_mq256_prefill_bidir against an FP32 CPU reference over the same BF16
// inputs, at nq = 1, 33, 65 and 200 (row counts that are not multiples of the 64-row block, partial key tiles).
// The causal output hash at nq = 200 is pinned per GPU architecture: the CAUSAL template must not change the causal
// instance (its SASS was compared with the pre-template kernel, identical).
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char * what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; } }
float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// reference: query i sees keys < (causal ? i + 1 : n); inputs already BF16-rounded
std::vector<float> ref(const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v, int n,
                       int h, int kh, float scale, bool causal) {
    const int D = 256, g = h / kh;
    std::vector<float> y(size_t(n) * h * D);
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i)
        for (int hh = 0; hh < h; ++hh) {
            const int lim = causal ? i + 1 : n, c = hh / g;
            double m = -1e300;
            for (int j = 0; j < lim; ++j) {
                double d = 0;
                for (int e = 0; e < D; ++e) d += double(q[(size_t(i) * h + hh) * D + e]) * k[(size_t(j) * kh + c) * D + e];
                s[j] = d * scale;
                m = std::max(m, s[j]);
            }
            double z = 0;
            for (int j = 0; j < lim; ++j) z += (s[j] = std::exp(s[j] - m));
            for (int e = 0; e < D; ++e) {
                double o = 0;
                for (int j = 0; j < lim; ++j) o += s[j] * v[(size_t(j) * kh + c) * D + e];
                y[(size_t(i) * h + hh) * D + e] = float(o / z);
            }
        }
    return y;
}

uint64_t fnv(const std::vector<uint16_t> & x) {
    uint64_t hsh = 1469598103934665603ull;
    for (auto u : x) { hsh ^= u; hsh *= 1099511628211ull; }
    return hsh;
}
}  // namespace

int main() {
    const int h = 8, kh = 2, D = 256;
    const float scale = 1.f / 16.f;
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    int * dseq0 = nullptr;
    cudaMalloc(&dseq0, sizeof(int));
    const int one = 1;
    cudaMemcpy(dseq0, &one, sizeof(int), cudaMemcpyHostToDevice);
    uint64_t causal_hash = 0;
    for (int n : {1, 33, 65, 200}) {
        uint32_t st = 12345u + n;
        auto rnd = [&] { st = st * 1664525u + 1013904223u; return bf(float(int(st >> 9) % 2001 - 1000) / 400.f); };
        std::vector<float> q(size_t(n) * h * D), k(size_t(n) * kh * D), v(size_t(n) * kh * D);
        for (auto * a : {&q, &k, &v}) for (auto & x : *a) x = rnd();
        auto up = [](const std::vector<float> & a) {
            std::vector<__nv_bfloat16> b(a.size());
            for (size_t i = 0; i < a.size(); ++i) b[i] = __float2bfloat16_rn(a[i]);
            __nv_bfloat16 * d = nullptr;
            cudaMalloc(&d, b.size() * 2);
            cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
            return d;
        };
        __nv_bfloat16 * dq = up(q), * dk = up(k), * dv = up(v), * dy = nullptr;
        cudaMalloc(&dy, q.size() * 2);
        for (bool causal : {true, false}) {
            cudaMemset(dy, 0, q.size() * 2);
            if (causal) tqllm_cuda::attention_mq256_prefill(dq, dk, dv, dy, nullptr, 0, dseq0, n, h, kh, scale, 1, 0);
            else tqllm_cuda::attention_mq256_prefill_bidir(dq, dk, dv, dy, dseq0, n, h, kh, scale, 0);
            check(cudaDeviceSynchronize() == cudaSuccess, "kernel ran");
            std::vector<uint16_t> raw(q.size());
            cudaMemcpy(raw.data(), dy, raw.size() * 2, cudaMemcpyDeviceToHost);
            const auto want = ref(q, k, v, n, h, kh, scale, causal);
            double err = 0;
            for (size_t i = 0; i < raw.size(); ++i) {
                const uint32_t u = uint32_t(raw[i]) << 16;
                float f; std::memcpy(&f, &u, 4);
                err = std::max(err, double(std::fabs(f - want[i])));
            }
            std::printf("n %3d %-6s max |err| %.4f\n", n, causal ? "causal" : "bidir", err);
            check(err < 0.03, causal ? "causal matches the reference" : "bidirectional matches the reference");
            if (causal && n == 200) causal_hash = fnv(raw);
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy);
    }
    std::printf("sm_%d%d causal hash %016llx\n", major, minor, (unsigned long long)causal_hash);
    // pinned from the pre-template kernel (same SASS); other architectures print theirs
    const struct { int cc; uint64_t hash; } pinned[] = {{121, 0x38fa3ac06769b599ull}, {86, 0x83af35a4a5daea80ull}};
    for (const auto & p : pinned)
        if (p.cc == major * 10 + minor && p.hash) check(causal_hash == p.hash, "causal output unchanged (pinned hash)");
    std::printf(failures ? "attn prefill bidir: FAILED\n" : "attn prefill bidir: OK\n");
    return failures ? 1 : 0;
}

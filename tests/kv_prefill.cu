// Prefill attention on the split KV cache (attention_kv_split with a chunk wider than a verify block): a chunk of nq
// queries at the end of a cache of `ctx` tokens, 24 query heads over 4 KV heads, head_dim 256 (Qwen3.8-27B shape).
// Cases: chunk rows equal single queries (short prefix, bit-identical) and stay close when a long prefix is split.
// --bench CTX NQ K V [REPS]: time one attention call and print an output hash (base vs branch comparisons).
#include "qwen35_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

using tqllm_cuda::KvCodec;

namespace {
constexpr int H = 24, KH = 4, D = 256;

size_t row_bytes(KvCodec c) {
    switch (c) {
        case KvCodec::Tq3: return sizeof(tqllm_cuda::Tq3KV256);
        case KvCodec::Q8: return sizeof(tqllm_cuda::Q8KV256);
        case KvCodec::Q4: return sizeof(tqllm_cuda::Q4KV256);
        case KvCodec::Tcq3: return sizeof(tqllm_cuda::Tcq3K256);
        case KvCodec::Tcq2: return sizeof(tqllm_cuda::Tcq2V256);
        default: return sizeof(tqllm_cuda::Tq2V256);
    }
}
KvCodec codec(const std::string & s) {
    if (s == "tq3") return KvCodec::Tq3;
    if (s == "q8") return KvCodec::Q8;
    if (s == "q4") return KvCodec::Q4;
    if (s == "tcq3") return KvCodec::Tcq3;
    if (s == "tcq2") return KvCodec::Tcq2;
    return KvCodec::Tq2s;
}

__global__ void fill_k(__nv_bfloat16 * x, size_t n, uint32_t seed, float sd) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = uint32_t(i) * 2654435761u ^ seed;
        h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
        const uint32_t g = h * 747796405u + 2891336453u;
        const float u1 = (h & 0xFFFFFF) / 16777216.f + 1e-7f, u2 = (g & 0xFFFFFF) / 16777216.f;
        x[i] = __float2bfloat16(sd * sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2));
    }
}

struct Cache {
    void * k = nullptr, * v = nullptr;
    int ctx = 0;
};

// the cache rows from deterministic Gaussian K / V (K scaled so the softmax is peaked rather than flat)
Cache make_cache(KvCodec kc, KvCodec vc, int ctx) {
    Cache c;
    c.ctx = ctx;
    cudaMalloc(&c.k, size_t(ctx) * KH * row_bytes(kc));
    cudaMalloc(&c.v, size_t(ctx) * KH * row_bytes(vc));
    const int B = 4096, ts = 2 * KH * D;
    __nv_bfloat16 * x;
    int * dp;
    cudaMalloc(&x, size_t(B) * ts * 2), cudaMalloc(&dp, sizeof(int));
    for (int p = 0; p < ctx; p += B) {
        const int n = std::min(B, ctx - p);
        fill_k<<<256, 256>>>(x, size_t(n) * ts, 0x9E3779B9u * uint32_t(p / B + 1), 1.f);
        cudaMemcpy(dp, &p, sizeof(int), cudaMemcpyHostToDevice);
        tqllm_cuda::cache_write_kv_side(kc, x, ts, n, c.k, dp, KH, D, 0);
        tqllm_cuda::cache_write_kv_side(vc, x + KH * D, ts, n, c.v, dp, KH, D, 0);
    }
    cudaDeviceSynchronize();
    cudaFree(x), cudaFree(dp);
    return c;
}

uint64_t fnv(const std::vector<__nv_bfloat16> & y) {
    uint64_t h = 1469598103934665603ull;
    const auto * b = reinterpret_cast<const unsigned char *>(y.data());
    for (size_t i = 0; i < y.size() * 2; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// one prefill attention call: queries ctx - nq .. ctx - 1 (each sees the keys up to itself)
std::vector<__nv_bfloat16> run(KvCodec kc, KvCodec vc, const Cache & c, int nq, int reps, double * ms) {
    __nv_bfloat16 * q, * y;
    float * part;
    int * dpos;
    cudaMalloc(&q, size_t(nq) * H * D * 2), cudaMalloc(&y, size_t(nq) * H * D * 2);
    const size_t pf = size_t(nq) * H * 8 * (D + 2);
    cudaMalloc(&part, pf * 4);
    cudaMalloc(&dpos, 2 * sizeof(int));
    fill_k<<<256, 256>>>(q, size_t(nq) * H * D, 12345u, 0.35f);
    const int pos[2] = {c.ctx - nq, c.ctx - nq + 1};
    cudaMemcpy(dpos, pos, sizeof(pos), cudaMemcpyHostToDevice);
    const float scale = 1.f / 16.f;
    tqllm_cuda::attention_kv_split(kc, vc, q, c.k, c.v, y, part, dpos + 1, nq, H, KH, scale, 1, 0, nullptr, nullptr, nullptr, nullptr, 0, false, pf);
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) tqllm_cuda::attention_kv_split(kc, vc, q, c.k, c.v, y, part, dpos + 1, nq, H, KH, scale, 1, 0, nullptr, nullptr, nullptr, nullptr, 0, false, pf);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float t = 0;
    cudaEventElapsedTime(&t, a, b);
    *ms = t / reps;
    std::vector<__nv_bfloat16> out(size_t(nq) * H * D);
    cudaMemcpy(out.data(), y, out.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(q), cudaFree(y), cudaFree(part), cudaFree(dpos);
    return out;
}
// queries pos0 .. pos0 + nq - 1 from the rows q of a query table (row z = position z), one attention call
std::vector<__nv_bfloat16> attend(KvCodec kc, KvCodec vc, const Cache & c, const __nv_bfloat16 * qtab, int pos0, int nq) {
    __nv_bfloat16 * y;
    float * part;
    int * dpos;
    const size_t pf = size_t(nq) * H * 8 * (D + 2);
    cudaMalloc(&y, size_t(nq) * H * D * 2), cudaMalloc(&part, pf * 4), cudaMalloc(&dpos, 2 * sizeof(int));
    const int pos[2] = {pos0, pos0 + 1};
    cudaMemcpy(dpos, pos, sizeof(pos), cudaMemcpyHostToDevice);
    tqllm_cuda::attention_kv_split(kc, vc, qtab + size_t(pos0) * H * D, c.k, c.v, y, part, dpos + 1, nq, H, KH, 1.f / 16.f, 1, 0,
                                   nullptr, nullptr, nullptr, nullptr, 0, false, pf);
    std::vector<__nv_bfloat16> out(size_t(nq) * H * D);
    cudaMemcpy(out.data(), y, out.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(y), cudaFree(part), cudaFree(dpos);
    return out;
}

// rows q0 .. q0 + n - 1 of a prefill chunk at `from` against the same queries attended as a small block (the verify-width
// kernel, no key split): max |difference| relative to the largest output value
double chunk_vs_rows(KvCodec kc, KvCodec vc, int ctx, int from, int nq, int q0, int n) {
    const Cache c = make_cache(kc, vc, ctx);
    __nv_bfloat16 * qtab;
    cudaMalloc(&qtab, size_t(ctx) * H * D * 2);
    fill_k<<<256, 256>>>(qtab, size_t(ctx) * H * D, 777u, 0.35f);
    const auto big = attend(kc, vc, c, qtab, from, nq), small = attend(kc, vc, c, qtab, from + q0, n);
    double d = 0, m = 0;
    for (size_t i = 0; i < small.size(); ++i) {
        const float a = __bfloat162float(big[size_t(q0) * H * D + i]), b = __bfloat162float(small[i]);
        d = std::max(d, double(std::fabs(a - b))), m = std::max(m, double(std::fabs(b)));
    }
    cudaFree(qtab), cudaFree(c.k), cudaFree(c.v);
    return m > 0 ? d / m : 1.0;
}

// A short prefix is one key range: every chunk row is bit-identical to the same query attended alone (q8 / q4 and the
// trellis K pair table, tcq3 / q4).
void chunk_rows_exact() {
    for (const auto & kv : {std::pair{KvCodec::Q8, KvCodec::Q4}, std::pair{KvCodec::Tcq3, KvCodec::Q4}, std::pair{KvCodec::Tq3, KvCodec::Tq3}}) {
        const double r = chunk_vs_rows(kv.first, kv.second, 6000, 3000, 512, 200, 8);
        if (r != 0.0) throw std::runtime_error("chunk row differs from the single query: " + std::to_string(r));
    }
}
// A long prefix is split over key ranges and merged: the same rows within FP32 merge rounding (BF16 output).
void long_prefix_split_close() {
    for (const auto & kv : {std::pair{KvCodec::Q8, KvCodec::Q4}, std::pair{KvCodec::Tcq3, KvCodec::Q4}}) {
        const double r = chunk_vs_rows(kv.first, kv.second, 70000, 69000, 512, 300, 8);
        if (!(r < 2e-2)) throw std::runtime_error("split prefix too far from the single query: " + std::to_string(r));
    }
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 6 && std::string(argv[1]) == "--bench") {
        const int ctx = std::atoi(argv[2]), nq = std::atoi(argv[3]), reps = argc > 6 ? std::atoi(argv[6]) : 3;
        const KvCodec kc = codec(argv[4]), vc = codec(argv[5]);
        const Cache c = make_cache(kc, vc, ctx);
        double ms = 0;
        const auto y = run(kc, vc, c, nq, reps, &ms);
        const double mac = 2.0 * H * D * nq * (ctx - nq / 2.0);
        std::printf("ctx %d nq %d %s/%s: %.3f ms  %.1f TMAC/s  hash %016llx\n", ctx, nq, argv[4], argv[5], ms,
                    mac / ms / 1e9, (unsigned long long)fnv(y));
        if (cudaGetLastError() != cudaSuccess) return 1;
        return 0;
    }
    const struct { const char * name; void (*run)(); } cases[] = {{"chunk_rows_exact", chunk_rows_exact},
                                                                  {"long_prefix_split_close", long_prefix_split_close}};
    int failed = 0;
    for (const auto & c : cases) {
        try {
            c.run();
            if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA error");
            std::printf("PASS %s\n", c.name);
        } catch (const std::exception & e) {
            std::printf("FAIL %s: %s\n", c.name, e.what());
            ++failed;
        }
    }
    return failed ? 1 : 0;
}

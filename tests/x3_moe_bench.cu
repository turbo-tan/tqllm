// X3 routed-expert window GEMV at MiniMax-M3.1 shapes (128 experts, gate / up 6144 -> 3072, down 3072 -> 6144, K2):
// decode (n = 1, top-4) and verify blocks (n = 2..8 tokens, distinct experts read once). Prints us per call and the
// GB/s of the distinct experts' trellis bytes. Synthetic trellis (timing only); no ctest entry.
#include "weight_format.hpp"
#include "x3_record.hpp"
#include "gguf.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

using namespace tqllm;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); std::exit(2); } }
uint32_t g = 4242u;
uint32_t lcg() { g = g * 1664525u + 1013904223u; return g; }

Mat bank(int E, int rows, int cols, int K, cudaStream_t s) {
    std::vector<char> raw;
    for (int e = 0; e < E; ++e) {
        const uint32_t words = 16 * uint32_t(K);
        const X3Record rec = x3_record_make(uint32_t(rows), uint32_t(cols), words, X3_CODEC_MUL1);
        raw.insert(raw.end(), reinterpret_cast<const char *>(&rec), reinterpret_cast<const char *>(&rec) + sizeof rec);
        const size_t tb = size_t(rows / 16) * (cols / 16) * words * 2;
        const size_t at = raw.size();
        raw.resize(at + tb + size_t(cols + rows) * 2);
        for (size_t i = 0; i < tb; i += 4) { const uint32_t w = lcg(); std::memcpy(raw.data() + at + i, &w, 4); }
        for (int i = 0; i < cols + rows; ++i) { const __half h = __float2half(0.7f); std::memcpy(raw.data() + at + tb + 2 * i, &h, 2); }
    }
    return weight_format(GGML_X3)->load(raw, rows * E, cols, [](const void * h, size_t b) -> const void * {
        void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice)); return d; }, s);
}
}  // namespace

int main() {
    const int E = 128, topk = 4, H = 6144, F = 3072;
    cudaStream_t s; ck(cudaStreamCreate(&s));
    struct Proj { const char * name; int rows, cols; bool per_slot; } projs[] = {{"gate/up 6144->3072", F, H, false}, {"down 3072->6144", H, F, true}};
    const WeightFormat * wf = weight_format(GGML_X3);
    for (const Proj & p : projs) {
        const Mat m = bank(E, p.rows, p.cols, 2, s);
        const double expert_bytes = double(p.rows / 16) * (p.cols / 16) * 32 * 2;
        for (int n : {1, 2, 4, 8}) {
            const int count = n * topk;
            std::vector<int> ids(static_cast<size_t>(count));
            for (int t = 0; t < n; ++t) {  // distinct within a token, shared across tokens like real routing (~60 % overlap)
                std::set<int> used;
                for (int j = 0; j < topk; ++j) {
                    int e;
                    do e = (lcg() >> 8) % (t == 0 ? E : 24); while (used.count(e));
                    used.insert(e);
                    ids[size_t(t) * topk + j] = e;
                }
            }
            const std::set<int> distinct(ids.begin(), ids.end());
            int * di; ck(cudaMalloc(&di, ids.size() * 4)); ck(cudaMemcpy(di, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            __half * x; ck(cudaMalloc(&x, size_t(count) * p.cols * 2)); ck(cudaMemset(x, 0, size_t(count) * p.cols * 2));
            __nv_bfloat16 * y; ck(cudaMalloc(&y, size_t(count) * p.rows * 2));
            for (int w = 0; w < 5; ++w) wf->expert_matvec(m, x, y, p.rows, di, n, topk, p.per_slot, s);
            cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
            const int reps = 50;
            ck(cudaEventRecord(a, s));
            for (int r = 0; r < reps; ++r) wf->expert_matvec(m, x, y, p.rows, di, n, topk, p.per_slot, s);
            ck(cudaEventRecord(b, s)); ck(cudaEventSynchronize(b));
            float ms = 0; ck(cudaEventElapsedTime(&ms, a, b));
            const double us = 1000.0 * ms / reps, read = n == 1 ? count * expert_bytes : distinct.size() * expert_bytes;
            std::printf("%-20s n=%d slots=%2d distinct=%2zu  %7.1f us  %6.1f GB/s\n", p.name, n, count, n == 1 ? size_t(count) : distinct.size(),
                        us, read / us / 1e3);
            cudaFree(di); cudaFree(x); cudaFree(y);
        }
    }
    return 0;
}

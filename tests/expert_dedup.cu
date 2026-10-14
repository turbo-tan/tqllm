// Dedup grouped expert matvec (one block per unique expert) must equal the per-assignment grouped matvec bit for bit.
#include "../src/formats/weight_format.hpp"
#include "../src/gguf.hpp"
#include "quant_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void require(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); check(cudaMemset(p, 0, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) { std::vector<T> x(n); check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
static std::vector<int> dedup_map(const std::vector<int> & slots, int topk, bool per_expert, int & groups) {
    std::map<int, std::vector<int>> by;
    for (int a = 0; a < int(slots.size()); ++a) by[slots[a]].push_back(a);
    std::vector<int> map;
    for (auto & [slot, as] : by)
        for (size_t c0 = 0; c0 < as.size(); c0 += tqllm_cuda::kDedupCols) {
            std::vector<int> g(tqllm_cuda::kDedupStride, 0);
            const int nc = int(std::min<size_t>(tqllm_cuda::kDedupCols, as.size() - c0));
            g[0] = slot; g[1] = nc;
            for (int c = 0; c < nc; ++c) { const int a = as[c0 + c]; g[2 + c] = per_expert ? a : a / topk; g[6 + c] = a; }
            map.insert(map.end(), g.begin(), g.end());
        }
    groups = int(map.size() / tqllm_cuda::kDedupStride);
    return map;
}
static void run(uint32_t type, int rows, int cols, bool per_expert) {
    const int experts = 6, n = 3, topk = 4;
    uint64_t elems = 0, bytes = 0; require(gguf_block_layout(type, elems, bytes), "block layout");
    std::mt19937 rng(type * 31 + cols + per_expert);
    std::vector<char> raw(size_t(experts) * rows * cols / elems * bytes);
    for (auto & b : raw) b = char(rng());
    const auto half = __float2half_rn(0.02f);
    for (size_t b = 0; b < raw.size(); b += bytes) std::memcpy(raw.data() + b, &half, 2);  // sane d at block start
    std::vector<std::unique_ptr<Device<char>>> owned;
    Upload up = [&](const void * h, size_t nb) { owned.push_back(std::make_unique<Device<char>>(nb)); check(cudaMemcpy(owned.back()->p, h, nb, cudaMemcpyHostToDevice)); return (const void *)owned.back()->p; };
    const auto * format = weight_format(type); require(format, "format");
    const Mat m = format->load(raw, experts * rows, cols, up, nullptr);
    require(m.fmt->has_expert_matvec() && m.fmt->has_expert_dedup(), "native format lacks grouped / dedup kernels");
    // shared experts across the 3 tokens, one expert used 3 times
    const std::vector<int> slots = {0, 1, 2, 3,  1, 0, 4, 5,  2, 1, 5, 3};
    const int A = int(slots.size());
    Device<int> ids(A); ids.put(slots);
    std::vector<__half> x(size_t(cols) * (per_expert ? A : n));
    for (size_t j = 0; j < x.size(); ++j) x[j] = __float2half_rn(float(int(rng() % 2001) - 1000) / 1000.f);
    Device<__half> dx(x.size()); dx.put(x);
    Device<__nv_bfloat16> ref(size_t(A) * rows), got(size_t(A) * rows);
    m.fmt->expert_matvec(m, dx.p, ref.p, rows, ids.p, n, topk, per_expert, nullptr);
    int groups = 0; const auto map = dedup_map(slots, topk, per_expert, groups);
    Device<int> dm(map.size()); dm.put(map);
    m.fmt->expert_matvec_dedup(m, dx.p, got.p, rows, dm.p, groups, nullptr);
    check(cudaDeviceSynchronize());
    const auto a = ref.get(size_t(A) * rows), b = got.get(size_t(A) * rows);
    require(std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0, "dedup differs from per-assignment grouped matvec");
    if (std::string(m.fmt->name()) == "I8H") {  // grouped fmt 15 vs the per-assignment row-sliced matvec MTP used before
        Device<__nv_bfloat16> sl(size_t(A) * rows);
        for (int a = 0; a < A; ++a) {
            Mat v = m; v.rows = rows;
            v.p = static_cast<const char *>(m.p) + size_t(slots[a]) * rows * m.fmt->row_bytes(cols);
            v.aux = static_cast<const char *>(m.aux) + size_t(slots[a]) * rows * m.fmt->aux_row_bytes(cols);
            v.fmt->matvec(v, dx.p + size_t(per_expert ? a : a / topk) * cols, sl.p + size_t(a) * rows, 1, nullptr);
        }
        check(cudaDeviceSynchronize()); const auto c = sl.get(size_t(A) * rows);
        require(std::memcmp(a.data(), c.data(), a.size() * sizeof(a[0])) == 0, "grouped I8H differs from the row-sliced matvec");
    }
    std::printf("dedup %s rows %d cols %d per_expert %d: %d groups for %d assignments, bit-identical\n", m.fmt->name(), rows, cols, per_expert, groups, A);
}
int main() {
    try {
        for (bool pe : {false, true}) {
            run(GGML_IQ4_NL, 40, 256, pe);
            run(GGML_IQ3_XXS, 40, 512, pe);
            run(GGML_IQ2_S, 24, 256, pe);
            run(GGML_IQ4_NL, 17, 640, pe);
            run(GGML_Q8_0, 40, 640, pe);   // loads as I8H (MTP experts), 640-column down projection
            run(GGML_Q8_0, 24, 2560, pe);
        }
        std::puts("expert dedup PASS");
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

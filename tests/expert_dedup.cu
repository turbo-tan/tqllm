// Dedup grouped expert matvec (one block per unique expert) must equal the per-assignment grouped matvec bit for bit.
#include "weight_format.hpp"
#include "gguf.hpp"
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
            for (int c = 0; c < nc; ++c) { const int a = as[c0 + c]; g[2 + c] = per_expert ? a : a / topk; g[2 + tqllm_cuda::kDedupCols + c] = a; }
            map.insert(map.end(), g.begin(), g.end());
        }
    groups = int(map.size() / tqllm_cuda::kDedupStride);
    return map;
}
// shared experts across 3 tokens, one used 3 times; heavy: 10 tokens all on slot 0 and 6 on slot 1 (groups of more than
// kDedupCols columns split, the verify blocks of up to 8 rows put a hot expert on every token)
static const std::vector<int> kLight = {0, 1, 2, 3,  1, 0, 4, 5,  2, 1, 5, 3};
static std::vector<int> heavy() {
    std::vector<int> s;
    for (int t = 0; t < 10; ++t) { s.push_back(0); s.push_back(t < 6 ? 1 : 2 + t % 4); s.push_back(2 + t % 3); s.push_back(5); }
    return s;
}
// 300 tokens over 6 experts: ~150 groups, past the launch grid's kDedupGridGroups (blocks walk several real groups)
static std::vector<int> many() {
    std::vector<int> s; std::mt19937 g(5);
    for (int t = 0; t < 300; ++t) for (int q = 0; q < 4; ++q) s.push_back(int(g() % 6));
    return s;
}
static void run(uint32_t type, int rows, int cols, bool per_expert, const std::vector<int> & slots = kLight) {
    const int experts = 6, topk = 4, n = int(slots.size()) / topk;
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
    {  // the device path's map: its groups, then 200 groups of slot -1, launched for all of them (grid smaller than the count)
        std::vector<int> padded(map); padded.resize(map.size() + size_t(200) * tqllm_cuda::kDedupStride, -1);
        Device<int> dp(padded.size()); dp.put(padded);
        Device<__nv_bfloat16> gp(size_t(A) * rows);
        m.fmt->expert_matvec_dedup(m, dx.p, gp.p, rows, dp.p, groups + 200, nullptr);
        check(cudaDeviceSynchronize());
        const auto c = gp.get(size_t(A) * rows);
        require(std::memcmp(a.data(), c.data(), a.size() * sizeof(a[0])) == 0, "padded dedup map differs from per-assignment grouped matvec");
        if (!per_expert) {
            Device<__nv_bfloat16> ga(size_t(A) * rows), gb(size_t(A) * rows);
            m.fmt->expert_matvec_dedup2(m, m, dx.p, ga.p, gb.p, rows, dp.p, groups + 200, nullptr);
            check(cudaDeviceSynchronize());
            const auto d = ga.get(size_t(A) * rows), e = gb.get(size_t(A) * rows);
            require(std::memcmp(a.data(), d.data(), a.size() * sizeof(a[0])) == 0 && std::memcmp(a.data(), e.data(), a.size() * sizeof(a[0])) == 0,
                    "padded paired dedup differs from per-assignment grouped matvec");
        }
    }
    if (!per_expert) {  // the paired gate / up launch (one matrix twice here) against the same reference
        Device<__nv_bfloat16> ga(size_t(A) * rows), gb(size_t(A) * rows);
        m.fmt->expert_matvec_dedup2(m, m, dx.p, ga.p, gb.p, rows, dm.p, groups, nullptr);
        check(cudaDeviceSynchronize());
        const auto c = ga.get(size_t(A) * rows), d = gb.get(size_t(A) * rows);
        require(std::memcmp(a.data(), c.data(), a.size() * sizeof(a[0])) == 0 && std::memcmp(a.data(), d.data(), a.size() * sizeof(a[0])) == 0,
                "paired dedup differs from per-assignment grouped matvec");
    }
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
            run(GGML_Q4_1, 40, 2560, pe);  // loads as u4h (an MLX affine 4-bit checkpoint's routed experts): gate / up
            run(GGML_Q4_1, 24, 640, pe);   // down
            run(GGML_GSQ_Q2_0, 40, 640, pe);   // the ISTA files' 2-bit experts (dot_warp)
            run(GGML_GSQ_Q2_0, 136, 640, pe, heavy());  // two full 64-row blocks and a partial one
            run(GGML_GSQ_Q2_0, 24, 2560, pe, heavy());
            run(GGML_GSQ_Q2_0, 16, 3072, pe, heavy());  // wide rows (dot_wide)
            run(GGML_IQ3_XXS, 40, 2560, pe, heavy());
            run(GGML_IQ3_XXS, 40, 256, pe, many());
            run(GGML_GSQ_Q2_0, 72, 640, pe, many());
            run(GGML_IQ2_S, 24, 2560, pe, heavy());
            run(GGML_IQ4_NL, 40, 640, pe, heavy());
            run(GGML_Q4_1, 40, 2560, pe, heavy());
        }
        std::puts("expert dedup PASS");
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

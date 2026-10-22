// Grouped expert GEMM (prefill) on plain and TQFN_TILED banks: every u4h expert format must implement it, the tiled bank
// must give the same bytes as the plain one, and the result must match the dedup matvec within FP16 / BF16 rounding.
#include "weight_format.hpp"
#include "gguf.hpp"
#include "quant_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void require(bool ok, const std::string & msg) { if (!ok) throw std::runtime_error(msg); }
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
static float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

static void run(uint32_t type, int rows, int cols, bool per_expert) {
    const int experts = 6, n = 90, topk = 4, A = n * topk;
    uint64_t elems = 0, bytes = 0; require(gguf_block_layout(type, elems, bytes), "block layout");
    std::mt19937 rng(type * 31 + cols + per_expert);
    std::vector<char> raw(size_t(experts) * rows * cols / elems * bytes);
    for (auto & b : raw) b = char(rng());
    const auto half = __float2half_rn(0.02f);
    for (size_t b = 0; b < raw.size(); b += bytes) std::memcpy(raw.data() + b, &half, 2);
    std::vector<std::unique_ptr<Device<char>>> owned;
    Upload up = [&](const void * h, size_t nb) { owned.push_back(std::make_unique<Device<char>>(nb)); check(cudaMemcpy(owned.back()->p, h, nb, cudaMemcpyHostToDevice)); return (const void *)owned.back()->p; };
    const auto * format = weight_format(type); require(format, "format");
    const Mat m = format->load(raw, experts * rows, cols, up, nullptr);
    require(m.fmt->has_expert_dedup(), std::string(m.fmt->name()) + ": no dedup kernel");
    const int bpb = tqllm_cuda::tiled_block_bytes(m.fmt->u4h_id());

    // skewed routing: expert 0 gets more than 64 tokens (two column tiles), expert 5 a few
    std::vector<int> slots(A);
    for (int a = 0; a < A; ++a) slots[a] = (a % topk == 0) ? 0 : int(1 + rng() % (experts - 1));
    std::vector<int> cnt(experts, 0); int cap = 1;
    for (int a = 0; a < A; ++a) cap = std::max(cap, ++cnt[slots[a]]);
    std::vector<int> grp(size_t(experts) * cap, 0), fill(experts, 0);
    for (int a = 0; a < A; ++a) grp[size_t(slots[a]) * cap + fill[slots[a]]++] = a;
    Device<int> dcnt(experts), dgrp(grp.size()); dcnt.put(cnt); dgrp.put(grp);

    std::vector<__half> x(size_t(cols) * (per_expert ? A : n));
    for (auto & v : x) v = __float2half_rn(float(int(rng() % 2001) - 1000) / 1000.f);
    Device<__half> dx(x.size()); dx.put(x);
    Device<__nv_bfloat16> yd(size_t(A) * rows), yg(size_t(A) * rows), yt(size_t(A) * rows);

    int groups = 0; const auto map = dedup_map(slots, topk, per_expert, groups);
    Device<int> dm(map.size()); dm.put(map);
    m.fmt->expert_matvec_dedup(m, dx.p, yd.p, rows, dm.p, groups, nullptr);
    require(m.fmt->expert_gemm(m, dx.p, yg.p, rows, dcnt.p, dgrp.p, cap, experts, topk, per_expert, nullptr),
            std::string(m.fmt->name()) + ": expert_gemm not implemented (prefill would skip this projection)");
    check(cudaDeviceSynchronize());
    const auto d = yd.get(size_t(A) * rows), g = yg.get(size_t(A) * rows);
    double num = 0, den = 0;
    for (size_t i = 0; i < d.size(); ++i) { const double e = bf(g[i]) - bf(d[i]); num += e * e; den += double(bf(d[i])) * bf(d[i]); }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    require(rel < 1e-2, std::string(m.fmt->name()) + ": grouped GEMM vs dedup rel err " + std::to_string(rel));

    bool tiled = false;
    if (bpb && rows % 16 == 0 && (cols / 32) % 4 == 0) {  // same bank, code bytes in the TQFN_TILED layout
        const size_t code = size_t(experts) * rows * (cols / 32) * bpb;
        std::vector<char> plain(code), t(code);
        check(cudaMemcpy(plain.data(), m.p, code, cudaMemcpyDeviceToHost));
        for (int e = 0; e < experts; ++e)
            tqllm_cuda::tile_expert_rows(plain.data() + size_t(e) * rows * (cols / 32) * bpb, t.data() + size_t(e) * rows * (cols / 32) * bpb, rows, cols, bpb);
        Device<char> dt(code); check(cudaMemcpy(dt.p, t.data(), code, cudaMemcpyHostToDevice));
        Mat mt = m; mt.p = dt.p;
        tqllm_cuda::set_tiled_experts(true);
        mt.fmt->expert_gemm(mt, dx.p, yt.p, rows, dcnt.p, dgrp.p, cap, experts, topk, per_expert, nullptr);
        check(cudaDeviceSynchronize());
        tqllm_cuda::set_tiled_experts(false);
        const auto tt = yt.get(size_t(A) * rows);
        require(std::memcmp(tt.data(), g.data(), g.size() * sizeof(g[0])) == 0, std::string(m.fmt->name()) + ": tiled grouped GEMM differs from plain");
        tiled = true;
    }
    std::printf("expert gemm %s (u4h %d) rows %d cols %d per_expert %d: rel err vs dedup %.2e%s\n", m.fmt->name(), m.fmt->u4h_id(),
                rows, cols, per_expert, rel, tiled ? ", tiled bit-identical" : "");
}
int main() {
    try {
        for (bool pe : {false, true}) {
            run(GGML_IQ4_NL, 64, 640, pe);   // Flash-Next down projection (IQ4L, 640 columns)
            run(GGML_IQ4_NL, 80, 256, pe);
            run(GGML_IQ3_XXS, 64, 512, pe);
            run(GGML_IQ3_S, 64, 512, pe);
            run(GGML_IQ2_S, 48, 256, pe);
        }
        std::puts("expert gemm tiled PASS");
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

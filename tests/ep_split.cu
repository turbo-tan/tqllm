// Expert parallelism over two ranks (simulated on one GPU): each rank computes the assignments of the experts it owns for
// its own rows and the peer's (one dedup launch over both), the result rows cross by a fixed-size exchange, and the combine
// runs in the unsplit order. The MoE output of every rank must equal the unsplit layer (the full_slots path) bit for bit,
// for idle ranks, verify blocks up to 24 rows per rank, hot experts with more than kDedupCols columns, and result caps that
// overflow (overflowing assignments are computed locally).
#include "experts/moe_kernels.cuh"
#include "models/qwen/qwen4exp/qwen4exp_kernels.cuh"
#include "weight_format.hpp"
#include "gguf.hpp"
#include "quant_kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static void require(bool ok, const std::string & msg) { if (!ok) throw std::runtime_error(msg); }
template<class T> struct Device {
    T * p = nullptr; size_t n = 0;
    explicit Device(size_t c) : n(std::max<size_t>(c, 1)) { check(cudaMalloc(&p, n * sizeof(T))); check(cudaMemset(p, 0, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & x) { if (!x.empty()) check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t c) { std::vector<T> x(c); if (c) check(cudaMemcpy(x.data(), p, c * sizeof(T), cudaMemcpyDeviceToHost)); return x; }
};
constexpr int kExperts = 32, kTopk = 6, kH = 512, kFf = 256, DS = tqllm_cuda::kDedupStride;

struct Bank { Mat gate, up, down; std::vector<std::unique_ptr<Device<char>>> owned; };
static Mat make(Bank & b, uint32_t type, int rows, int cols, uint32_t seed) {
    uint64_t elems = 0, bytes = 0; require(gguf_block_layout(type, elems, bytes), "block layout");
    std::mt19937 rng(seed);
    std::vector<char> raw(size_t(kExperts) * rows * cols / elems * bytes);
    for (auto & c : raw) c = char(rng());
    const auto d = __float2half_rn(0.02f);
    for (size_t o = 0; o < raw.size(); o += bytes) {
        if (type == GGML_NVFP4G) {  // per 128 weights (76 B): F32 global scale, 8 E4M3 group scales (0x38 = 1.0), 64 B E2M1
            for (uint64_t q = 0; q + 76 <= bytes; q += 76) { const float gs = 0.02f; std::memcpy(raw.data() + o + q, &gs, 4); std::memset(raw.data() + o + q + 4, 0x38, 8); }
        } else {
            std::memcpy(raw.data() + o, &d, 2);
            if (type == GGML_Q4_1) std::memcpy(raw.data() + o + 2, &d, 2);
        }
    }
    Upload up = [&](const void * h, size_t nb) { b.owned.push_back(std::make_unique<Device<char>>(nb)); check(cudaMemcpy(b.owned.back()->p, h, nb, cudaMemcpyHostToDevice)); return (const void *) b.owned.back()->p; };
    const auto * f = weight_format(type); require(f, "format");
    Mat m = f->load(raw, kExperts * rows, cols, up, nullptr);
    require(m.fmt->has_expert_matvec() && m.fmt->has_expert_dedup(), std::string("no dedup kernels: ") + m.fmt->name());
    return m;
}

// the full_slots decode path of fn_layer_forward: maps on the device, paired gate / up, SwiGLU, down; ed = A rows of h.
// The split launch always takes the dedup kernels (its ids hold -1 for the assignments the other rank computes; the
// one-row grouped matvec has no skip): dedup == grouped matvec bit for bit (test expert_dedup).
static void experts(const Bank & b, const __half * x, const int * ids, const int * slot_of, int n, __nv_bfloat16 * ed, bool split) {
    const int A = n * kTopk;
    if (A == 0) return;
    const bool dedup_gu = n > 1 || split, dedup_dn = n > 1 || split;
    Device<int> maps(size_t(A) * (1 + 2 * DS));
    int * slots = maps.p, * gu = maps.p + A, * dn = gu + size_t(A) * DS;
    tqllm_cuda::moe_dedup_maps(ids, slot_of, A, kTopk, slots, dedup_gu ? gu : nullptr, dedup_dn ? dn : nullptr, A, nullptr);
    Device<__nv_bfloat16> eg(size_t(A) * kFf), eu(eg.n); Device<__half> ax(eg.n);
    if (dedup_gu) b.gate.fmt->expert_matvec_dedup2(b.gate, b.up, x, eg.p, eu.p, kFf, gu, A, nullptr);
    else { b.gate.fmt->expert_matvec(b.gate, x, eg.p, kFf, slots, n, kTopk, false, nullptr); b.up.fmt->expert_matvec(b.up, x, eu.p, kFf, slots, n, kTopk, false, nullptr); }
    tqllm_cuda::fn_swiglu_bf16_h(eg.p, eu.p, ax.p, int64_t(A) * kFf, nullptr);
    if (dedup_dn) b.down.fmt->expert_matvec_dedup(b.down, ax.p, ed, kH, dn, A, nullptr);
    else b.down.fmt->expert_matvec(b.down, ax.p, ed, kH, slots, n, kTopk, true, nullptr);
}

struct Rank { int n; std::vector<int> ids; std::vector<__half> x; std::vector<float> w, sh, sw; };
static Rank make_rank(int n, int pool, std::mt19937 & rng) {
    Rank r{n};
    for (int t = 0; t < n; ++t) {  // distinct experts per token (top-k); pool < kExperts makes hot experts
        std::vector<int> e(kExperts); std::iota(e.begin(), e.end(), 0); std::shuffle(e.begin(), e.begin() + pool, rng);
        for (int q = 0; q < kTopk; ++q) r.ids.push_back(e[size_t(q % pool)]);
    }
    for (int i = 0; i < n * kH; ++i) r.x.push_back(__float2half_rn(float(int(rng() % 2001) - 1000) / 1000.f));
    for (int i = 0; i < n * kTopk; ++i) r.w.push_back(float(rng() % 1000) / 1000.f);
    for (int i = 0; i < n * kH; ++i) r.sh.push_back(float(int(rng() % 2001) - 1000) / 500.f);
    for (int i = 0; i < n; ++i) r.sw.push_back(float(rng() % 1000) / 1000.f);
    return r;
}

static int run(const Bank & b, int n0, int n1, int pool, int cap, uint32_t seed) {
    std::mt19937 rng(seed);
    const Rank R[2] = {make_rank(n0, pool, rng), make_rank(n1, pool, rng)};
    std::vector<int> ident(kExperts); std::iota(ident.begin(), ident.end(), 0);
    Device<int> slot_of(kExperts); slot_of.put(ident);
    // per rank: the unsplit reference, then the split launch over (own rows, peer rows)
    std::vector<std::unique_ptr<Device<__nv_bfloat16>>> ed_cat(2), send(2);
    std::vector<std::vector<float>> ref(2);
    std::vector<std::unique_ptr<Device<int>>> ids(2);
    for (int r = 0; r < 2; ++r) {
        const Rank & me = R[r], & pe = R[1 - r]; const int A = me.n * kTopk, Ap = pe.n * kTopk;
        ids[r] = std::make_unique<Device<int>>(A); ids[r]->put(me.ids);
        Device<__half> x(size_t(me.n) * kH); x.put(me.x);
        Device<float> w(A), sh(size_t(me.n) * kH), sw(me.n), fo(size_t(me.n) * kH);
        w.put(me.w); sh.put(me.sh); sw.put(me.sw);
        Device<__nv_bfloat16> ed(size_t(A) * kH);
        experts(b, x.p, ids[r]->p, slot_of.p, me.n, ed.p, false);
        if (me.n) tqllm_cuda::fn_moe_combine(ed.p, w.p, sh.p, sw.p, fo.p, kH, kTopk, me.n, nullptr);
        check(cudaDeviceSynchronize()); ref[r] = fo.get(size_t(me.n) * kH);

        std::vector<int> cat_ids(me.ids); cat_ids.insert(cat_ids.end(), pe.ids.begin(), pe.ids.end());
        std::vector<__half> cat_x(me.x); cat_x.insert(cat_x.end(), pe.x.begin(), pe.x.end());
        Device<int> all(A + Ap), local(A + Ap); all.put(cat_ids);
        Device<__half> xc(cat_x.size()); xc.put(cat_x);
        tqllm_cuda::ep_local_ids(all.p, A, all.p + A, Ap, r, cap, local.p, nullptr);
        ed_cat[r] = std::make_unique<Device<__nv_bfloat16>>(size_t(A + Ap) * kH);
        experts(b, xc.p, local.p, slot_of.p, me.n + pe.n, ed_cat[r]->p, true);
        send[r] = std::make_unique<Device<__nv_bfloat16>>(size_t(std::max(cap, 1)) * kH);
        tqllm_cuda::ep_gather(ed_cat[r]->p, A, all.p + A, Ap, r, cap, send[r]->p, kH, nullptr);
        // every assignment computed exactly once over the two ranks: own-local + peer-local covers each assignment
        check(cudaDeviceSynchronize());
        const auto lid = local.get(size_t(A + Ap));
        for (int a = 0; a < A + Ap; ++a) require(lid[a] < 0 || lid[a] == cat_ids[a], "local id is not the routed id");
    }
    int bad = 0;
    for (int r = 0; r < 2; ++r) {
        const Rank & me = R[r]; const int A = me.n * kTopk;
        if (!me.n) continue;
        tqllm_cuda::ep_scatter(send[1 - r]->p, ids[r]->p, A, r, cap, ed_cat[r]->p, kH, nullptr);
        Device<float> w(A), sh(size_t(me.n) * kH), sw(me.n), fo(size_t(me.n) * kH);
        w.put(me.w); sh.put(me.sh); sw.put(me.sw);
        tqllm_cuda::fn_moe_combine(ed_cat[r]->p, w.p, sh.p, sw.p, fo.p, kH, kTopk, me.n, nullptr);
        check(cudaDeviceSynchronize());
        const auto got = fo.get(size_t(me.n) * kH);
        if (std::memcmp(got.data(), ref[r].data(), got.size() * sizeof(float)) != 0) ++bad;
    }
    return bad;
}

int main() {
    try {
        int cases = 0, fails = 0;
        for (uint32_t type : {uint32_t(GGML_NVFP4G), uint32_t(GGML_IQ4_NL), uint32_t(GGML_IQ3_XXS), uint32_t(GGML_Q4_1)}) {
            Bank b;
            b.gate = make(b, type, kFf, kH, type * 7 + 1); b.up = make(b, type, kFf, kH, type * 7 + 2); b.down = make(b, type, kH, kFf, type * 7 + 3);
            for (auto [n0, n1] : std::vector<std::pair<int, int>>{{1, 0}, {0, 1}, {3, 0}, {0, 8}, {1, 1}, {4, 3}, {8, 8}, {24, 24}, {24, 21}})
                for (int pool : {kExperts, 7})
                    for (int cap : {1 << 20, 5, 0}) {
                        const int f = run(b, n0, n1, pool, cap, uint32_t(cases * 131 + 17));
                        if (f) std::printf("FAIL %s n %d / %d pool %d cap %d: %d rank(s) differ\n", b.gate.fmt->name(), n0, n1, pool, cap, f);
                        fails += f != 0; ++cases;
                    }
            std::printf("%s: done\n", b.gate.fmt->name());
        }
        std::printf("%s: %d cases\n", fails ? "ep split FAILED" : "ep split PASS", cases);
        return fails ? 1 : 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

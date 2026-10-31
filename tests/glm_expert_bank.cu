// GLM-5.3-Flash routed experts from the shared expert bank (src/models/glm/glm_experts.hpp): a layer's experts in a
// MixedExpertCache of `slots` slots, the grouped expert kernels on the bank's slot ids, waves when a chunk routes past
// the slots. It must give the whole-expert-stack output bit for bit (same rows, same bytes, another row index) and both
// must match a CPU reference of the same layer (dequantise + multiply, the math of tools/glm_ref.cpp). Mock routing
// (moe_route, the router glm-ref uses), the expert type mixes of both GLM files (GSQ Q4_K / Q3_K / Q2_K and TQ3_4S
// IQ2_S + Q2_K), and the packed expert file round trip (a miss is a copy, not a loader repack).
#include "dequant.hpp"
#include "glm5next_kernels.cuh"
#include "glm_experts.hpp"
#include "expert_bank_plan.hpp"
#include "moe.hpp"
#include "packed_experts.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace tqllm;
int failures = 0;
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(e)); }
void ok(bool pass, const char * what) {
    if (!pass) ++failures;
    std::printf("%s %s\n", pass ? "PASS" : "FAIL", what);
}

template <class T> struct Dev {
    T * p = nullptr;
    size_t count = 0;
    explicit Dev(size_t n) : count(n) { check(cudaMalloc(&p, n * sizeof(T))); check(cudaMemset(p, 0, n * sizeof(T))); }
    ~Dev() { if (p) cudaFree(p); }
    void put(const std::vector<T> & v) { check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

const int kLayers = 3, kExperts = 24, kHidden = 512, kFf = 256, kTopk = 8;
const uint32_t kType[kLayers][3] = {{GGML_Q4_K, GGML_Q4_K, GGML_Q4_K},
                                    {GGML_Q3_K, GGML_Q3_K, GGML_Q3_K},
                                    {GGML_IQ2_S, GGML_IQ2_S, GGML_Q2_K}};
int rows_of(int p) { return p == 2 ? kHidden : kFf; }
int cols_of(int p) { return p == 2 ? kFf : kHidden; }

// One expert of one projection: random codes with the block's scale fields set to a small finite half (random half bits
// would give inf / NaN, and a comparison against NaN proves nothing). The scale fields sit where this tree's dequantisers
// read them: Q4_K d/dmin at 0/2, Q3_K d at 108, Q2_K d/dmin at 80/82, IQ2_S d at 0 and its 8 4-bit scale pairs at 74.
std::vector<char> expert_bytes(uint32_t type, int rows, int cols, int seed) {
    uint64_t block = 0, bytes = 0;
    if (!gguf_block_layout(type, block, bytes) || cols % int(block)) throw std::runtime_error("expert type has no block layout here");
    std::vector<char> d(size_t(rows) * size_t(cols) / size_t(block) * size_t(bytes));
    std::mt19937 r(seed);
    for (auto & b : d) b = char(r() & 255);
    const uint16_t h = __half_as_ushort(__float2half_rn(0.001f + 0.0001f * float(seed % 7)));
    auto patch = [&](size_t off, uint16_t v) {
        if (off + 2 <= d.size()) std::memcpy(d.data() + off, &v, 2);
    };
    for (size_t b = 0; b + bytes <= d.size(); b += bytes) {
        switch (type) {
            case GGML_Q4_K: patch(b, h); patch(b + 2, h); break;
            case GGML_Q3_K: patch(b + 108, h); break;
            case GGML_Q2_K: patch(b + 80, h); patch(b + 82, h); break;
            case GGML_IQ2_S:
                patch(b, h);
                for (int k = 0; k < 8; ++k) d[b + 74 + size_t(k)] = char(0x11);  // small, finite per-sub-block scales
                break;
            default: throw std::runtime_error("test fixture: unhandled expert type");
        }
    }
    return d;
}

// raw GGUF bytes of every expert of every layer, and the same weights dequantised on the CPU (the reference's view)
struct Fixture {
    std::vector<std::array<std::vector<char>, 3>> raw;    // [layer * experts + expert][projection]
    std::vector<std::array<std::vector<float>, 3>> deq;   // the same weights dequantised on the CPU
    double max_abs = 0;  // the largest weight magnitude: the fixture must stay far from fp16 / bf16 overflow
    double per_layer[3] = {0, 0, 0};
    bool finite = true;
    Fixture() : raw(size_t(kLayers) * kExperts), deq(size_t(kLayers) * kExperts) {
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e)
                for (int p = 0; p < 3; ++p) {
                    auto & b = raw[size_t(l) * kExperts + e][p];
                    b = expert_bytes(kType[l][p], rows_of(p), cols_of(p), 17 + 31 * e + 101 * l);
                    auto & f = deq[size_t(l) * kExperts + e][p];
                    f.resize(size_t(rows_of(p)) * cols_of(p));
                    dequantize_host(kType[l][p], b.data(), f.data(), int64_t(f.size()));
                    for (float v : f) {
                        if (!std::isfinite(v)) finite = false;
                        per_layer[l] = std::max(per_layer[l], double(std::fabs(v)));
                        max_abs = std::max(max_abs, double(std::fabs(v)));
                    }
                }
    }
    std::array<ExpertPayload, 3> payload(int layer, int expert) const {
        const auto & r = raw[size_t(layer) * kExperts + expert];
        std::array<ExpertPayload, 3> out;
        for (int p = 0; p < 3; ++p) {
            out[p].type = kType[layer][p];
            out[p].bytes = {r[p].data(), r[p].size()};
        }
        return out;
    }
    // the whole expert stack of one projection, as the GGUF 3-D tensor holds it: experts back to back
    std::vector<char> stack(int layer, int p) const {
        std::vector<char> all;
        for (int e = 0; e < kExperts; ++e) { const auto & b = raw[size_t(layer) * kExperts + e][p]; all.insert(all.end(), b.begin(), b.end()); }
        return all;
    }
};

// the whole expert stack of every layer on the device (the path the engine used before the bank)
struct Stack {
    std::vector<std::array<Mat, 3>> m;
    std::vector<void *> owned;
    const void * put(const void * h, size_t bytes) {
        void * p = nullptr;
        check(cudaMalloc(&p, bytes));
        check(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice));
        owned.push_back(p);
        return p;
    }
    ~Stack() { for (void * p : owned) cudaFree(p); }
    explicit Stack(const Fixture & fx) {
        m.resize(size_t(kLayers));
        for (int l = 0; l < kLayers; ++l)
            for (int p = 0; p < 3; ++p) {
                const auto raw = fx.stack(l, p);
                const Mat mm = weight_format(kType[l][p])->load(raw, kExperts * rows_of(p), cols_of(p),
                                                                [this](const void * h, size_t b) { return put(h, b); }, nullptr);
                if (!mm.fmt || !mm.fmt->has_expert_matvec() || mm.fmt->basis() != Basis::Plain)
                    throw std::runtime_error("test fixture: expert type has no grouped expert kernel");
                m[size_t(l)][p] = mm;
            }
    }
};

// one MoE layer on the GPU: gate / up from x FP16 [n][hidden], SwiGLU with the model's clamp, down, weighted sum.
// `d_ids` are row groups of the stacked matrices: expert ids for the whole stack, bank slots for a bank.
std::vector<float> moe_layer(const std::array<Mat, 3> & m, const int * d_ids, const __half * d_x, const float * d_w, int n,
                             float clamp) {
    const int A = n * kTopk;
    Dev<__nv_bfloat16> yg(size_t(A) * kFf), yu(size_t(A) * kFf), yd(size_t(A) * kHidden);
    Dev<__half> act(size_t(A) * kFf);
    Dev<float> out(size_t(n) * kHidden);
    m[0].fmt->expert_matvec(m[0], d_x, yg.p, kFf, d_ids, n, kTopk, false, nullptr);
    m[1].fmt->expert_matvec(m[1], d_x, yu.p, kFf, d_ids, n, kTopk, false, nullptr);
    tqllm_cuda::glm_swiglu(yg.p, yu.p, act.p, A * kFf, clamp, nullptr);
    m[2].fmt->expert_matvec(m[2], act.p, yd.p, kHidden, d_ids, n, kTopk, true, nullptr);
    tqllm_cuda::glm_moe_sum(yd.p, d_w, nullptr, out.p, kTopk, kHidden, nullptr, n);
    return out.get(size_t(n) * kHidden);
}

// the same layer on the CPU: moe_route + moe_expert_ffn + moe_combine (the math tools/glm_ref.cpp runs), FP32
std::vector<float> moe_cpu(const Fixture & fx, int layer, const std::vector<float> & x, const std::vector<int> & ids,
                           const std::vector<float> & w, float clamp) {
    const int n = int(ids.size()) / kTopk;
    std::vector<float> out(size_t(n) * kHidden, 0.f), expert(size_t(kTopk) * kHidden);
    for (int t = 0; t < n; ++t) {
        MoeSelection sel;
        for (int k = 0; k < kTopk; ++k) {
            sel.ids.push_back(ids[size_t(t) * kTopk + k]);
            sel.weights.push_back(w[size_t(t) * kTopk + k]);
        }
        for (int k = 0; k < kTopk; ++k) {
            const auto & e = fx.deq[size_t(layer) * kExperts + sel.ids[size_t(k)]];
            moe_expert_ffn(x.data() + size_t(t) * kHidden, kHidden, e[0].data(), e[1].data(), e[2].data(), kFf, clamp, true,
                           expert.data() + size_t(k) * kHidden);
        }
        moe_combine(sel, expert.data(), kHidden, out.data() + size_t(t) * kHidden);
    }
    return out;
}

double rel_err(const std::vector<float> & got, const std::vector<float> & want) {
    double worst = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        // a NaN / inf on either side makes the comparison vacuous: report it as the worst error instead
        if (!std::isfinite(want[i]) || !std::isfinite(got[i])) return 1e30;
        worst = std::max(worst, std::fabs(got[i] - want[i]) / std::max(1.0, double(std::fabs(want[i]))));
    }
    return worst;
}
bool finite(const std::vector<float> & v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}
bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// one layer from a bank: the assignments in waves, prepare() per wave, the grouped kernels on the wave's slot ids
std::vector<float> moe_bank(MixedExpertCache & cache, int layer, const std::vector<int> & ids, const __half * d_x,
                            const float * d_w, float clamp) {
    const int n = int(ids.size()) / kTopk;
    std::vector<float> out(size_t(n) * kHidden);
    for (const auto & wv : glm_expert_waves(ids, kTopk, cache.slots(layer))) {
        const auto physical = cache.prepare(layer, wv.ids);
        std::vector<int> slot_of(size_t(kExperts), -1);
        for (size_t i = 0; i < wv.ids.size(); ++i) slot_of[size_t(wv.ids[i])] = physical[i];
        std::vector<int> slots;
        slots.reserve(size_t(wv.tokens) * kTopk);
        for (int a = wv.first * kTopk; a < (wv.first + wv.tokens) * kTopk; ++a) {
            if (slot_of[size_t(ids[size_t(a)])] < 0) throw std::runtime_error("a wave does not cover its own expert");
            slots.push_back(slot_of[size_t(ids[size_t(a)])]);
        }
        Dev<int> d_slots(slots.size());
        d_slots.put(slots);
        const auto part = moe_layer({cache.gate(), cache.up(), cache.down()}, d_slots.p, d_x + size_t(wv.first) * kHidden,
                                    d_w + size_t(wv.first) * kTopk, wv.tokens, clamp);
        std::copy(part.begin(), part.end(), out.begin() + size_t(wv.first) * kHidden);
    }
    return out;
}

void run() {
    Fixture fx;
    {  // the fixture must stay in a range where a comparison means something: fp16 / bf16 overflow gives NaN, and
       // comparing against NaN passes every check below.
        char what[160];
        std::snprintf(what, sizeof what, "fixture weights are finite and small (max |w| per layer %.3g / %.3g / %.3g)", fx.per_layer[0],
                          fx.per_layer[1], fx.per_layer[2]);
        ok(fx.finite && fx.max_abs < 4.0, what);
    }
    const Stack stack(fx);
    const float clamp = 10.f;
    const int n = 6;
    std::mt19937 r(20261019);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> x(size_t(n) * kHidden, 0.f), w(size_t(n) * kTopk, 0.f), logits(size_t(kExperts), 0.f),
        bias(size_t(kExperts), 0.f);
    std::vector<int> ids(size_t(n) * kTopk);
    for (auto & v : x) v = u(r);
    MoeConfig mc;
    mc.n_expert = kExperts;
    mc.n_expert_used = kTopk;
    mc.gating = MoeGating::Sigmoid;
    mc.norm_weights = true;
    mc.weight_scale = 2.5f;
    for (int t = 0; t < n; ++t) {  // mock routing: the CPU router glm-ref uses
        for (auto & v : logits) v = u(r) * 6.f;
        for (auto & v : bias) v = u(r) * 0.5f;
        const auto sel = moe_route(mc, logits.data(), bias.data());
        for (int k = 0; k < kTopk; ++k) {
            ids[size_t(t) * kTopk + k] = sel.ids[size_t(k)];
            w[size_t(t) * kTopk + k] = sel.weights[size_t(k)];
        }
    }
    std::vector<__half> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) xh[i] = __float2half_rn(x[i]);
    Dev<__half> dx(xh.size());
    dx.put(xh);
    Dev<float> dw(w.size());
    dw.put(w);
    Dev<int> d_ids(ids.size());
    d_ids.put(ids);
    const size_t budget = 64u << 20;
    auto source = [&](int l, int e) { return fx.payload(l, e); };
    std::vector<int> used_ids = ids;  // the routing's distinct experts: a bank smaller than this must reload
    std::sort(used_ids.begin(), used_ids.end());
    used_ids.erase(std::unique(used_ids.begin(), used_ids.end()), used_ids.end());
    const int used = int(used_ids.size());

    for (int cap : {kTopk, 16, kExperts}) {  // waves cover every assignment exactly once, in token order, within the slots
        const auto waves = glm_expert_waves(ids, kTopk, cap);
        std::vector<char> seen(ids.size(), 0);
        bool cover = true, fits = true, order = true, sets = true;
        int at = 0;
        for (const auto & wv : waves) {
            if (wv.first != at) order = false;
            at += wv.tokens;
            if (int(wv.ids.size()) > cap) fits = false;
            std::vector<int> uniq;
            for (int a = wv.first * kTopk; a < (wv.first + wv.tokens) * kTopk; ++a) {
                if (seen[size_t(a)]) cover = false;
                seen[size_t(a)] = 1;
                uniq.push_back(ids[size_t(a)]);
            }
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            if (uniq != wv.ids) sets = false;
        }
        size_t covered = 0;
        for (char c : seen)
            if (c) ++covered;
        char what[160];
        std::snprintf(what, sizeof what, "waves of %d slots cover the assignments in order (%zu waves)", cap, waves.size());
        ok(cover && fits && order && sets && at == n && covered == seen.size(), what);
    }

    for (int l = 0; l < kLayers; ++l) {
        char what[160];
        const auto want = moe_cpu(fx, l, x, ids, w, clamp);
        const auto whole = moe_layer(stack.m[size_t(l)], d_ids.p, dx.p, dw.p, n, clamp);
        std::snprintf(what, sizeof what, "layer %d whole stack (%s gate/up, %s down) == the CPU reference (rel err %.2e)", l,
                      weight_format(kType[l][0])->name(), weight_format(kType[l][2])->name(), rel_err(whole, want));
        ok(rel_err(whole, want) < 3e-2, what);
        // the engine's guard before it opens a bank: the *loaded* format (Q4_K loads as K4, Q2_K / IQ2_S as Udq) must have
        // the row layout the bank uses - asking the GGUF type's loader instead rejects every K-quant layer.
        std::array<ExpertFormat, 3> fmt{};
        repack_expert(fx.payload(l, 3), kHidden, kFf, false, &fmt);
        std::snprintf(what, sizeof what, "layer %d: every expert type has a bank row layout (u4h ids %d / %d / %d)", l,
                      fmt[0].u4h_id, fmt[1].u4h_id, fmt[2].u4h_id);
        ok(fmt[0].bank_row_layout && fmt[1].bank_row_layout && fmt[2].bank_row_layout, what);
        for (int slots : {kExperts, 16, 8}) {
            MixedExpertCache cache(kLayers, kExperts, kHidden, kFf, slots, budget, source, nullptr);
            const auto got = moe_bank(cache, l, ids, dx.p, dw.p, clamp);
            std::snprintf(what, sizeof what, "layer %d bank of %d slots == whole stack bit for bit", l, slots);
            ok(same_bits(got, whole), what);
            std::snprintf(what, sizeof what, "layer %d bank of %d slots == the CPU reference (rel err %.2e)", l, slots,
                          rel_err(got, want));
            ok(rel_err(got, want) < 3e-2, what);
            if (slots < kExperts) {
                std::snprintf(what, sizeof what, "layer %d bank of %d slots reloaded past its slots (misses %llu / %d used, evictions %llu)",
                              l, slots, (unsigned long long)cache.stats().misses, used,
                              (unsigned long long)cache.stats().evictions);
                ok(cache.stats().misses >= size_t(used) && (used <= slots || cache.stats().evictions > 0), what);
            }
        }
        MixedExpertCache tight(kLayers, kExperts, kHidden, kFf, kTopk - 1, budget, source, nullptr);
        bool refused = false;
        try { tight.prepare(l, ids); } catch (const std::invalid_argument &) { refused = true; }
        ok(refused, "a bank with fewer slots than the model's top-k was accepted");
    }

    {  // the packed expert file: written from the same payloads, read back, and a miss is a copy, not a loader repack
        const std::string path = "tqllm-glm-expert-bank.tqfnexp";  // scratch in the test's own build dir: never outside
        write_packed_experts(path, kLayers, kExperts, false,
                             [&](int l, int e) { return repack_expert(fx.payload(l, e), kHidden, kFf, false); });
        const PackedExperts pk = open_packed(path);
        ok(pk.layers == kLayers && pk.experts == kExperts && !pk.tiled, "packed expert file header");
        bool parts = true;
        for (int l = 0; l < kLayers && parts; ++l) {
            const auto of = repack_expert(fx.payload(l, 3), kHidden, kFf, false);
            const char * at = pk.at(l, 3);
            size_t off = 0;
            for (int k = 0; k < 6; ++k) {
                if (std::memcmp(at + off, of[size_t(k)].data(), of[size_t(k)].size()) != 0) parts = false;
                off += of[size_t(k)].size();
            }
            if (off != pk.block[size_t(l)]) parts = false;
        }
        ok(parts, "packed expert bytes are the layout the cache banks hold");
        for (int l = 0; l < kLayers; ++l) {
            const auto whole = moe_layer(stack.m[size_t(l)], d_ids.p, dx.p, dw.p, n, clamp);
            MixedExpertCache cache(kLayers, kExperts, kHidden, kFf, 8, budget, source, nullptr);
            cache.set_prepacked([&](int l, int e) { return pk.at(l, e); }, false);
            const auto got = moe_bank(cache, l, ids, dx.p, dw.p, clamp);
            ok(same_bits(got, whole), "packed experts: bank of 8 slots == whole stack bit for bit");
        }
        ::munmap(const_cast<char *>(pk.map), pk.bytes);
        std::remove(path.c_str());
    }

    {  // the byte budget, not a slot count, sizes the banks (src/experts/expert_bank_plan.hpp): the routing spread the
        // engine measures per layer, each layer's bytes per slot (its loaded row layout) and the device budget give the
        // per-layer slots - a layer whose routing spreads further gets more - and the bank still matches the whole stack.
        std::vector<size_t> slot_bytes(size_t(kLayers), 0);
        for (int l = 0; l < kLayers; ++l) {
            const auto of = repack_expert(fx.payload(l, 0), kHidden, kFf, false);
            for (const auto & part : of) slot_bytes[size_t(l)] += part.size();
        }
        // mock routing spread: the distinct experts of the mock prompt, skewed per layer (layer 1 reaches twice as many
        // as layer 0, layer 2 half) - what the engine counts from the ids its own router produced.
        const double distinct = double(std::set<int>(ids.begin(), ids.end()).size());
        const std::vector<double> spread = {distinct, 2.0 * distinct, 0.5 * distinct};
        const size_t plan_budget = 12 * (slot_bytes[0] + slot_bytes[1] + slot_bytes[2]);
        const auto plan = expert_layer_slots(plan_budget, slot_bytes, spread, kTopk, kExperts);
        char what[200];
        std::snprintf(what, sizeof what, "the budget sizes every layer at least at top-k (%d / %d / %d slots, %.1f of %.1f MB)",
                      plan.slots[0], plan.slots[1], plan.slots[2], plan.bytes / 1e6, plan_budget / 1e6);
        ok(plan.fits && plan.bytes <= plan_budget && plan.slots[0] >= kTopk && plan.slots[1] >= kTopk && plan.slots[2] >= kTopk,
           what);
        std::snprintf(what, sizeof what, "the layer spreading over twice as many experts gets more slots (%d vs %d / %d)",
                      plan.slots[1], plan.slots[0], plan.slots[2]);
        ok(plan.slots[1] > plan.slots[0] && plan.slots[2] < plan.slots[1], what);
        MixedExpertCache cache(kLayers, kExperts, kHidden, kFf, *std::max_element(plan.slots.begin(), plan.slots.end()),
                               plan_budget, source, nullptr);
        cache.set_layer_slots(plan.slots);
        ok(cache.slots(1) == plan.slots[1] && cache.slots(2) == plan.slots[2], "the bank takes the plan's per-layer slots");
        for (int l = 0; l < kLayers; ++l) {
            const auto want = moe_cpu(fx, l, x, ids, w, clamp);
            const auto whole = moe_layer(stack.m[size_t(l)], d_ids.p, dx.p, dw.p, n, clamp);
            const auto got = moe_bank(cache, l, ids, dx.p, dw.p, clamp);
            std::snprintf(what, sizeof what, "layer %d bank of %d planned slots == whole stack bit for bit", l, cache.slots(l));
            ok(same_bits(got, whole), what);
            std::snprintf(what, sizeof what, "layer %d bank of %d planned slots == the CPU reference (rel err %.2e)", l,
                          cache.slots(l), rel_err(got, want));
            ok(rel_err(got, want) < 3e-2, what);
        }
        // the engine's re-plan between prompts: release every bank, apply new caps, run again. Placement only: the same
        // bits, and no call is refused because a bank existed.
        for (int l : cache.banks_lru()) cache.release_bank(l);
        cache.set_layer_slots(expert_layer_slots(plan_budget, slot_bytes, {1.0, 1.0, 1.0}, kTopk, kExperts).slots);
        for (int l = 0; l < kLayers; ++l) {
            const auto whole = moe_layer(stack.m[size_t(l)], d_ids.p, dx.p, dw.p, n, clamp);
            const auto got = moe_bank(cache, l, ids, dx.p, dw.p, clamp);
            std::snprintf(what, sizeof what, "layer %d re-planned to %d slots == whole stack bit for bit", l, cache.slots(l));
            ok(same_bits(got, whole), what);
        }
    }

    {  // the misses of the next token overlap the layer's attention: the packed experts live in page-locked host memory,
        // prefetch_released() copies them on a copy stream once the bank's last reader passed, and the prepare() that
        // follows finds them resident. Placement only: the same bits, and the prefetch was used instead of a miss.
        const std::string path = "tqllm-glm-expert-bank-pf.tqfnexp";  // scratch in the test's own build dir
        write_packed_experts(path, kLayers, kExperts, false,
                             [&](int l, int e) { return repack_expert(fx.payload(l, e), kHidden, kFf, false); });
        const PackedExperts pk = open_packed(path);
        char * pinned = nullptr;
        check(cudaMallocHost(reinterpret_cast<void **>(&pinned), pk.bytes));
        std::memcpy(pinned, pk.map, pk.bytes);
        ::munmap(const_cast<char *>(pk.map), pk.bytes);
        auto pinned_at = [&](int l, int e) { return pinned + pk.base[size_t(l)] + uint64_t(e) * pk.block[size_t(l)]; };
        // one token's assignments: two disjoint expert sets of top-k ids, so the second is a clean miss or a clean prefetch
        std::vector<int> tok_a(size_t(kTopk), 0), tok_b(size_t(kTopk), 0);
        for (int k = 0; k < kTopk; ++k) { tok_a[size_t(k)] = k; tok_b[size_t(k)] = kTopk + k; }
        auto run_token = [&](MixedExpertCache & c, int layer, const std::vector<int> & tok) {
            std::vector<int> uniq = tok;
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            const auto physical = c.prepare(layer, uniq);
            c.wait_misses();  // split mode: the wave's misses landed on the copy stream
            std::vector<int> slot_of(size_t(kExperts), -1), slots;
            for (size_t i = 0; i < uniq.size(); ++i) slot_of[size_t(uniq[i])] = physical[i];
            for (int id : tok) slots.push_back(slot_of[size_t(id)]);
            Dev<int> d_slots(slots.size());
            d_slots.put(slots);
            return moe_layer({c.gate(), c.up(), c.down()}, d_slots.p, dx.p, dw.p, 1, clamp);  // token 0's row and weights
        };
        cudaStream_t copy = nullptr;
        check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        {  // the caches outlive the stream and the pinned source: they free their banks through the miss stream
            MixedExpertCache plain(kLayers, kExperts, kHidden, kFf, kTopk, budget, source, nullptr);
            MixedExpertCache ahead(kLayers, kExperts, kHidden, kFf, kTopk, budget, source, nullptr);
            plain.set_prepacked(pinned_at, true);  // registered: the cache may DMA / kernel-read the blocks straight
            ahead.set_prepacked(pinned_at, true);
            plain.set_miss_stream(copy);  // split mode: a miss uploads on the copy stream, and a released bank can be prefetched
            ahead.set_miss_stream(copy);
            const auto first_a = run_token(plain, 0, tok_a), first_b = run_token(ahead, 0, tok_a);
            ok(same_bits(first_a, first_b), "the same token on both caches");
            (void) run_token(plain, 1, tok_a);  // the layer after: bank 0's last reader passed, so it can be prefetched
            (void) run_token(ahead, 1, tok_a);
            const bool took = ahead.prefetch_released(0, tok_b, copy);
            ok(took, "prefetch_released() takes the next token's experts for the layer about to run");
            const auto second_a = run_token(plain, 0, tok_b), second_b = run_token(ahead, 0, tok_b);
            ok(same_bits(second_a, second_b), "a prefetched expert is the same bytes as a demand miss");
            char what[200];
            std::snprintf(what, sizeof what,
                          "the prefetch served the next token (used %llu of %llu prefetched, misses %llu vs %llu)",
                          (unsigned long long)ahead.prefetch_used(), (unsigned long long)ahead.prefetched(),
                          (unsigned long long)ahead.stats().misses, (unsigned long long)plain.stats().misses);
            ok(ahead.prefetch_used() >= unsigned(kTopk) && ahead.stats().misses < plain.stats().misses, what);
        }
        check(cudaStreamSynchronize(copy));
        check(cudaStreamDestroy(copy));
        check(cudaFreeHost(pinned));
        std::remove(path.c_str());
    }

    {  // the engine's paired gate + up launch (its `paired` probe) matches the two separate launches, on the whole stack
        // (expert ids) and on a bank (slot ids): same rows, another index, so the same bits.

        auto paired_diff = [&](const std::array<Mat, 3> & m, const int * d_idx, const char * what) {
            Dev<__nv_bfloat16> g1(size_t(n) * kTopk * kFf), u1(g1.count), g2(g1.count), u2(g1.count);
            m[0].fmt->expert_matvec(m[0], dx.p, g1.p, kFf, d_idx, n, kTopk, false, nullptr);
            m[1].fmt->expert_matvec(m[1], dx.p, u1.p, kFf, d_idx, n, kTopk, false, nullptr);
            m[0].fmt->expert_matvec2(m[0], m[1], dx.p, g2.p, u2.p, kFf, d_idx, n, kTopk, nullptr);
            const auto a = g1.get(g1.count), b = g2.get(g2.count), c = u1.get(u1.count), d = u2.get(u2.count);
            // bit comparison (NaN != NaN would make a value comparison vacuous), plus a finiteness guard
            const bool bits = std::memcmp(a.data(), b.data(), a.size() * sizeof(__nv_bfloat16)) == 0 &&
                              std::memcmp(c.data(), d.data(), c.size() * sizeof(__nv_bfloat16)) == 0;
            bool fin = true;
            for (auto v : a) fin = fin && std::isfinite(float(v));
            for (auto v : c) fin = fin && std::isfinite(float(v));
            if (!bits || !fin)
                std::printf("  %s: bits=%d finite=%d (first out %.6g)\n", what, int(bits), int(fin), float(a[0]));
            ok(bits && fin, what);
        };
        paired_diff(stack.m[2], d_ids.p, "layer 2 whole stack: paired gate + up == two launches");
        MixedExpertCache cache(kLayers, kExperts, kHidden, kFf, kExperts, budget, source, nullptr);
        std::vector<int> uniq = ids;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        const auto physical = cache.prepare(2, uniq);
        std::vector<int> slot_of(size_t(kExperts), -1), slots;
        for (size_t i = 0; i < uniq.size(); ++i) slot_of[size_t(uniq[i])] = physical[i];
        for (int id : ids) slots.push_back(slot_of[size_t(id)]);
        Dev<int> d_slots(slots.size());
        d_slots.put(slots);
        paired_diff({cache.gate(), cache.up(), cache.down()}, d_slots.p, "layer 2 bank: paired gate + up == two launches");
    }
}
}  // namespace

int main() {
    try {
        run();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: CUDA error\n");
        return 1;
    }
    std::printf("%s\n", failures ? "glm expert bank: FAILED" : "glm expert bank: PASS");
    return failures ? 1 : 0;
}

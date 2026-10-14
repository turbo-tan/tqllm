#include "moe_kernels.cuh"
#include "qwen2_kernels.cuh"
// Declares tqllm_cuda::iq3_matvec_bf16, used below as the decode reference for GGUF 21.
// The target's include dir is ${CMAKE_CURRENT_SOURCE_DIR}/kernels, so a bare name resolves.
#include "quant_kernels.cuh"
#include "../src/formats/weight_format.hpp"
#include "../src/moe.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
template<class T> struct Device {
    T * p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

static void route_test() {
    const int h = 256, e = 256, k = 8, n = 5;
    std::vector<float> w(e * h), shared(h);
    std::vector<__half> x(n * h);
    for (int j = 0; j < h; ++j) {
        shared[j] = std::sin(float(j)) * .02f;
        for (int t = 0; t < n; ++t) x[t * h + j] = __float2half(std::cos(float(j + t)));
        for (int r = 0; r < e; ++r) w[r * h + j] = std::sin(float(j * 3 + r * 5)) * .02f;
    }
    Device<float> dw(w.size()), ds(h), logits(n * (e + 1)), weights(n * k), sw(n);
    Device<__half> dx(x.size());
    Device<int> ids(n * k);
    dw.put(w); ds.put(shared); dx.put(x);
    tqllm_cuda::moe_router(dw.p, ds.p, dx.p, logits.p, h, e, n, nullptr);
    auto got = logits.get(n * (e + 1));
    for (int t = 0; t < n; ++t) for (int r = 0; r <= e; ++r) {
        double ref = 0.;
        for (int j = 0; j < h; ++j) ref += double(r == e ? shared[j] : w[r * h + j]) * __half2float(x[t * h + j]);
        require(std::abs(got[t * (e + 1) + r] - ref) < 3e-6, "F32 router dot differs from reference");
    }
    // Ties, extreme finite logits, and nontrivial ordinary rows.
    for (int r = 0; r < e; ++r) got[r] = 0.f;
    for (int r = 0; r < e; ++r) got[e + 1 + r] = r == 99 ? 10000.f : -10000.f;
    for (int t = 2; t < n; ++t) for (int r = 0; r < e; ++r)
        got[t * (e + 1) + r] = .75f * std::sin(float(r * 3 + t)) + .017f * r;
    logits.put(got);
    tqllm_cuda::moe_topk(logits.p, ids.p, weights.p, sw.p, e, k, n, 1.3f, nullptr);
    const auto gi = ids.get(n * k); const auto gw = weights.get(n * k); const auto gs = sw.get(n);
    tqllm::MoeConfig cfg;
    cfg.n_expert = e; cfg.n_expert_used = k; cfg.gating = tqllm::MoeGating::Softmax;
    cfg.norm_weights = true; cfg.weight_scale = 1.3f;
    for (int t = 0; t < n; ++t) {
        auto ref = tqllm::moe_route(cfg, got.data() + t * (e + 1), nullptr);
        for (int j = 0; j < k; ++j) {
            if (gi[t * k + j] != ref.ids[j])
                std::fprintf(stderr, "route t=%d slot=%d gpu=%d cpu=%d wgpu=%g wcpu=%g\n", t, j, gi[t*k+j], ref.ids[j], gw[t*k+j], ref.weights[j]);
            require(gi[t * k + j] == ref.ids[j], "top-k ID differs from CPU reference");
            require(std::abs(gw[t * k + j] - ref.weights[j]) < 3e-6, "normalized weight differs");
        }
        require(std::abs(gs[t] - tqllm::moe_sigmoid(got[t * (e + 1) + e])) < 3e-6, "shared gate differs");
    }
    std::puts("F32 router + softmax/top-k/normalization/shared sigmoid: PASS");
}

// Every GGUF type whose runtime format now has grouped-expert support, so the identity suite
// covers the whole u4h family instead of only the three formats that were wired first.
//   bytes        = encoded size of one block (src/gguf.cpp gguf_block_layout)
//   scale_off    = half-scale offset(s) inside the block that the loader memcpys; the generator
//                  writes a varying non-zero scale there (never 0 -> no inf/nan reciprocals)
//   clamp        = byte range the loader uses as a divisor/table index, forced to 1..32
//                  (Q6_K only; matches the pre-existing generator)
//   legacy_iq3   = GGUF 21: take the decode reference from the legacy dense kernel, because
//                  Iq3::matvec() picks Split16 at short k and no grouped Split16 exists.
struct FmtSpec {
    unsigned type;
    const char * name;
    int bytes;
    int clamp_lo, clamp_hi;   // 0,0 = no clamp
    bool legacy_iq3;
};

static const FmtSpec kSpecs[] = {
    { 2u, "Q4_0->U4h",   18,  0,   0,  false},
    { 8u, "Q8_0->I8h",   34,  0,   0,  false},
    {10u, "Q2_K->Udq",   84,  0,   0,  false},
    {11u, "Q3_K->K3",   110,  0,   0,  false},
    {12u, "Q4_K->K4",   144,  0,   0,  false},
    {13u, "Q5_K->K5",   176,  0,   0,  false},
    {14u, "Q6_K->K6",   210, 192, 208, false},
    {16u, "IQ2_XXS->Udq", 66, 0,   0,  false},
    {17u, "IQ2_XS->Udq",  74, 0,   0,  false},
    {18u, "IQ3_XXS->Iq3x",98, 0,   0,  false},
    {19u, "IQ1_S->Udq",   50, 0,   0,  false},
    {20u, "IQ4_NL->Iq4l", 18, 0,   0,  false},
    {21u, "IQ3_S->Iq3",  110, 0,   0,  true },
    {22u, "IQ2_S->Udq",   82, 0,   0,  false},
    {23u, "IQ4_XS->Iq4n",136, 0,   0,  false},
};
// blocks per 256-column row: Q4_0, Q8_0 and IQ4_NL blocks hold 32 weights, the others 256
static int row_blocks(unsigned type) { return type == 2u || type == 8u || type == 20u ? 8 : 1; }
static const FmtSpec & spec_of(unsigned type) {
    for (const auto & sp : kSpecs) if (sp.type == type) return sp;
    throw std::runtime_error("grouped_test: type not in kSpecs");
}

static void grouped_test(unsigned type, int n, bool per_expert, bool use_gemm = false) {
    const FmtSpec & sp = spec_of(type);
    const int e = 4, rows = 19, cols = 256, k = 3, bytes = sp.bytes;
    std::mt19937 gen(71 + type);
    std::vector<char> raw(size_t(e) * rows * row_blocks(type) * bytes);
    for (size_t b = 0; b < raw.size() / bytes; ++b) {
        for (int j = 0; j < bytes; ++j) raw[b * bytes + j] = char(gen() % 256);
        // Varying non-zero scale so no loader takes a zero/inf reciprocal, and so Q4_0's
        // 128-group test (memcmp of 4 consecutive block scales) stays false -> U4h, not W4S.
        // BOTH fp16 scale slots (offset 0 and offset 2) get a valid positive value: several
        // K-quant blocks carry a second fp16 there (Q6_K's dmin, and the K/IQ superblock
        // dmin), and random bytes in an fp16 field can land on Inf/NaN, which would fail the
        // identity check for a reason unrelated to the grouped path. Bit-identity does not
        // depend on the data values, so overwriting payload bytes here is harmless.
        const __half d = __float2half(.001f * (1 + b % 7));
        std::memcpy(raw.data() + b * bytes, &d, 2);
        std::memcpy(raw.data() + b * bytes + 2, &d, 2);
        if (sp.clamp_hi > sp.clamp_lo) {
            std::memcpy(raw.data() + b * bytes + 208, &d, 2);
            for (int j = sp.clamp_lo; j < sp.clamp_hi; ++j) raw[b * bytes + j] = char(1 + gen() % 32);
        }
    }
    std::vector<void *> allocations;
    tqllm::Upload upload = [&](const void * h, size_t bytes) {
        void * p = nullptr; ck(cudaMalloc(&p, bytes)); ck(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice));
        allocations.push_back(p); return static_cast<const void *>(p);
    };
    const auto m = tqllm::weight_format(type)->load(raw, e * rows, cols, upload, nullptr);
    const int nx = per_expert ? n * k : n;
    std::vector<__half> x(nx * cols);
    for (auto & v : x) v = __float2half(float(int(gen() % 101) - 50) / 100.f);
    std::vector<int> ids(n * k);
    for (int t = 0; t < n; ++t) for (int s = 0; s < k; ++s) ids[t * k + s] = (use_gemm ? s : t + s) % e;
    Device<__half> dx(x.size()); dx.put(x);
    Device<int> di(ids.size()); di.put(ids);
    Device<int> counts(e), groups(e * n);
    Device<__nv_bfloat16> out(n * k * rows), ref(n * k * rows);
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    // Device IDs must remain device-side and valid during CUDA graph replay.
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (use_gemm) {
        tqllm_cuda::moe_group(di.p, counts.p, groups.p, e, n, k, stream);
        require(m.fmt->expert_gemm(m, dx.p, out.p, rows, counts.p, groups.p, n, e, k, per_expert, stream),
                "format does not support grouped GEMM");
    } else m.fmt->expert_matvec(m, dx.p, out.p, rows, di.p, n, k, per_expert, stream);
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    if (use_gemm) {
        const auto gc = counts.get(e), gg = groups.get(e * n);
        std::vector<int> seen(n * k, 0);
        for (int expert = 0; expert < e; ++expert) {
            require(gc[expert] >= 0 && gc[expert] <= n, "expert assignment count exceeds capacity");
            if (!gc[expert]) continue;
            std::vector<__half> gathered(gc[expert] * cols);
            for (int c = 0; c < gc[expert]; ++c) {
                const int g = gg[expert * n + c];
                require(g >= 0 && g < n * k && ids[g] == expert, "invalid grouped assignment");
                ++seen[g];
                const int input_col = per_expert ? g : g / k;
                std::copy_n(x.data() + input_col * cols, cols, gathered.data() + c * cols);
            }
            Device<__half> gathered_device(gathered.size()); gathered_device.put(gathered);
            Device<__nv_bfloat16> expected(gc[expert] * rows);
            auto v = m; v.rows = rows;
            v.p = static_cast<const char *>(m.p) + size_t(expert) * rows * m.fmt->row_bytes(cols);
            v.aux = static_cast<const char *>(m.aux) + size_t(expert) * rows * m.fmt->aux_row_bytes(cols);
            require(m.fmt->gemm(v, gathered_device.p, expected.p, gc[expert], stream), "dense GEMM reference missing");
            for (int c = 0; c < gc[expert]; ++c)
                ck(cudaMemcpyAsync(ref.p + gg[expert * n + c] * rows, expected.p + c * rows,
                                    rows * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
            ck(cudaStreamSynchronize(stream));
        }
        require(std::all_of(seen.begin(), seen.end(), [](int v) { return v == 1; }), "lost or duplicated assignment");
    } else {
    for (int t = 0; t < n; ++t) for (int s = 0; s < k; ++s) {
        auto v = m;
        v.rows = rows;
        v.p = static_cast<const char *>(m.p) + size_t(ids[t * k + s]) * rows * m.fmt->row_bytes(cols);
        v.aux = static_cast<const char *>(m.aux) + size_t(ids[t * k + s]) * rows * m.fmt->aux_row_bytes(cols);
        if (sp.legacy_iq3) {
            // Iq3::matvec() would pick Split16 here (short k), which has no grouped variant;
            // the grouped path IS the legacy u4h kernel, so reference against legacy dense to
            // keep the assertion bit-exact rather than toleranced.
            tqllm_cuda::iq3_matvec_bf16(static_cast<const uint8_t *>(v.p),
                                        static_cast<const uint8_t *>(v.aux),
                                        dx.p + size_t(per_expert ? t * k + s : t) * cols,
                                        ref.p + size_t(t * k + s) * rows, rows, cols, 1, stream);
        } else {
            m.fmt->matvec(v, dx.p + size_t(per_expert ? t * k + s : t) * cols,
                          ref.p + size_t(t * k + s) * rows, 1, stream);
        }
    }
    }
    ck(cudaStreamSynchronize(stream));
    const auto y = out.get(n * k * rows), yr = ref.get(n * k * rows);
    require(std::memcmp(y.data(), yr.data(), y.size() * 2) == 0, "grouped expert differs from row-sliced reference");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    for (void * p : allocations) ck(cudaFree(p));
    std::printf("GGUF %2u (%s) grouped %s n=%d per_expert=%d graph replay + tail rows: bit-identical%s\n", sp.type,
                sp.name, use_gemm ? "GEMM" : "matvec", n, per_expert, sp.legacy_iq3 ? " (vs legacy dense)" : "");
}

// gate + up in one launch (expert_matvec2) against two expert_matvec calls: bit-identical
static void pair_test(unsigned type, int n) {
    const FmtSpec & sp = spec_of(type);
    const int e = 4, rows = 19, cols = 256, k = 3, bytes = sp.bytes;
    std::mt19937 gen(97 + type);
    std::vector<void *> allocations;
    tqllm::Upload upload = [&](const void * h, size_t nbytes) {
        void * p = nullptr; ck(cudaMalloc(&p, nbytes)); ck(cudaMemcpy(p, h, nbytes, cudaMemcpyHostToDevice));
        allocations.push_back(p); return static_cast<const void *>(p);
    };
    auto make = [&] {
        std::vector<char> raw(size_t(e) * rows * row_blocks(type) * bytes);
        for (size_t b = 0; b < raw.size() / bytes; ++b) {
            for (int j = 0; j < bytes; ++j) raw[b * bytes + j] = char(gen() % 256);
            const __half d = __float2half(.001f * (1 + b % 7));
            std::memcpy(raw.data() + b * bytes, &d, 2);
            std::memcpy(raw.data() + b * bytes + 2, &d, 2);
            if (sp.clamp_hi > sp.clamp_lo) {
                std::memcpy(raw.data() + b * bytes + 208, &d, 2);
                for (int j = sp.clamp_lo; j < sp.clamp_hi; ++j) raw[b * bytes + j] = char(1 + gen() % 32);
            }
        }
        return tqllm::weight_format(type)->load(raw, e * rows, cols, upload, nullptr);
    };
    const auto a = make(), b = make();
    if (a.fmt->u4h_id() < 0 || a.fmt != b.fmt) {
        for (void * p : allocations) ck(cudaFree(p));
        return;
    }
    std::vector<__half> x(size_t(n) * cols);
    for (auto & v : x) v = __float2half(float(int(gen() % 101) - 50) / 100.f);
    std::vector<int> ids(n * k);
    for (int t = 0; t < n; ++t) for (int s = 0; s < k; ++s) ids[t * k + s] = (t + 2 * s) % e;
    Device<__half> dx(x.size()); dx.put(x);
    Device<int> di(ids.size()); di.put(ids);
    Device<__nv_bfloat16> ya(n * k * rows), yb(n * k * rows), ra(n * k * rows), rb(n * k * rows);
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    a.fmt->expert_matvec2(a, b, dx.p, ya.p, yb.p, rows, di.p, n, k, stream);
    a.fmt->expert_matvec(a, dx.p, ra.p, rows, di.p, n, k, false, stream);
    b.fmt->expert_matvec(b, dx.p, rb.p, rows, di.p, n, k, false, stream);
    ck(cudaStreamSynchronize(stream));
    const auto y1 = ya.get(n * k * rows), y2 = yb.get(n * k * rows), r1 = ra.get(n * k * rows), r2 = rb.get(n * k * rows);
    require(std::memcmp(y1.data(), r1.data(), y1.size() * 2) == 0, "paired expert (gate) differs from expert_matvec");
    require(std::memcmp(y2.data(), r2.data(), y2.size() * 2) == 0, "paired expert (up) differs from expert_matvec");
    ck(cudaStreamDestroy(stream));
    for (void * p : allocations) ck(cudaFree(p));
    std::printf("GGUF %2u (%s) paired gate + up n=%d: bit-identical\n", sp.type, sp.name, n);
}

static void attention_group_test() {
    const int h = 16, kh = 2, dim = 256, n = 8, context = 64, start = 13, split = 1;
    std::mt19937 rng(123);
    std::vector<__nv_bfloat16> q(n * h * dim), k(context * kh * dim), v(k.size());
    for (auto * vec : {&q, &k, &v}) for (auto & x : *vec)
        x = __float2bfloat16(float(int(rng() % 101) - 50) / 100.f);
    Device<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size()), dy(q.size()), ref(q.size());
    Device<float> part(tqllm_cuda::attention_mq_partial_floats(h, n, split));
    Device<int> pos(1); pos.put({start + 1}); dq.put(q); dk.put(k); dv.put(v);
    tqllm_cuda::attention_mq256(dq.p, dk.p, dv.p, dy.p, part.p, pos.p, n, h, kh, 1.f / 16.f, split, nullptr);
    for (int off = 0; off < n; ++off)
        tqllm_cuda::attention_mq256(dq.p + off * h * dim, dk.p, dv.p, ref.p + off * h * dim, part.p, pos.p,
                                   1, h, kh, 1.f / 16.f, split, nullptr, off);
    const auto y = dy.get(q.size()), r = ref.get(q.size());
    require(std::memcmp(y.data(), r.data(), y.size() * 2) == 0,
            "group-8 attention batch differs from sequential queries");
    std::puts("Group-8 attention verify(8) vs sequential queries: bit-identical");
}

static void combine_test() {
    const int h = 19, n = 3, k = 4, count = h * n;
    std::vector<__nv_bfloat16> r(count * k), sh(count), gate(count), up(count);
    std::vector<float> w(n * k), sw(n);
    for (int t = 0; t < n; ++t) {
        sw[t] = .2f + .1f * t;
        for (int s = 0; s < k; ++s) w[t * k + s] = .1f + .03f * s;
    }
    for (size_t i = 0; i < r.size(); ++i) r[i] = __float2bfloat16(float(int(i % 23) - 11) / 7.f);
    for (int i = 0; i < count; ++i) {
        sh[i] = __float2bfloat16(float(i) / 17.f);
        gate[i] = __float2bfloat16(float(i - 25) / 5.f);
        up[i] = __float2bfloat16(float(i - 16) / 7.f);
    }
    Device<__nv_bfloat16> dr(r.size()), ds(sh.size()), dg(count), du(count), out(count);
    Device<float> dw(w.size()), dsw(sw.size()); Device<__half> act(count);
    dr.put(r); ds.put(sh); dg.put(gate); du.put(up); dw.put(w); dsw.put(sw);
    tqllm_cuda::moe_sum(dr.p, ds.p, dw.p, dsw.p, out.p, h, k, n, nullptr);
    auto y = out.get(count);
    for (int i = 0; i < count; ++i) {
        const int t = i / h, j = i % h;
        float ref = 0.f;
        for (int s = 0; s < k; ++s) ref += w[t * k + s] * __bfloat162float(r[(t * k + s) * h + j]);
        ref += sw[t] * __bfloat162float(sh[i]);
        require(std::abs(__bfloat162float(y[i]) - ref) < .008f, "expert/shared combination differs");
    }
    tqllm_cuda::moe_swiglu(dg.p, du.p, act.p, count, nullptr);
    const auto a = act.get(count);
    for (int i = 0; i < count; ++i) {
        const float g = __bfloat162float(gate[i]), u = __bfloat162float(up[i]);
        const auto ref = __float2bfloat16(g / (1.f + std::exp(-g)) * u);
        require(std::abs(__half2float(a[i]) - __bfloat162float(ref)) < .002f, "SwiGLU differs");
    }
    std::puts("SwiGLU + weighted routed/shared combination: PASS");
}
// Hot / cold split (experts of one layer partly in VRAM, partly in host memory): one launch per set over the split ids
// must equal the single launch over all experts, bit for bit, for decode matvec and prefill GEMM.
static void split_test(unsigned type, int n, bool per_expert, bool use_gemm, const std::vector<int> & cold = {1, 3, 4}) {
    const FmtSpec & sp = spec_of(type);
    const int e = 6, rows = 19, cols = 256, k = 3, bytes = sp.bytes, eb = rows * row_blocks(type) * bytes;
    std::mt19937 gen(97 + type);
    std::vector<char> raw(size_t(e) * eb);
    for (size_t b = 0; b < raw.size() / bytes; ++b) {
        for (int j = 0; j < bytes; ++j) raw[b * bytes + j] = char(gen() % 256);
        const __half d = __float2half(.001f * (1 + b % 7));
        std::memcpy(raw.data() + b * bytes, &d, 2);
        std::memcpy(raw.data() + b * bytes + 2, &d, 2);
        if (sp.clamp_hi > sp.clamp_lo) {
            std::memcpy(raw.data() + b * bytes + 208, &d, 2);
            for (int j = sp.clamp_lo; j < sp.clamp_hi; ++j) raw[b * bytes + j] = char(1 + gen() % 32);
        }
    }
    std::vector<int> hot;  // the experts not in cold; a set may be smaller than k (a layer with few cold experts)
    for (int i = 0; i < e; ++i) if (std::find(cold.begin(), cold.end(), i) == cold.end()) hot.push_back(i);
    std::vector<int> hot_slot(e, -1), cold_slot(e, -1);
    std::vector<char> raw_hot, raw_cold;
    for (size_t i = 0; i < hot.size(); ++i) { hot_slot[hot[i]] = int(i); raw_hot.insert(raw_hot.end(), raw.begin() + hot[i] * eb, raw.begin() + (hot[i] + 1) * eb); }
    for (size_t i = 0; i < cold.size(); ++i) { cold_slot[cold[i]] = int(i); raw_cold.insert(raw_cold.end(), raw.begin() + cold[i] * eb, raw.begin() + (cold[i] + 1) * eb); }
    std::vector<void *> allocations;
    tqllm::Upload upload = [&](const void * h, size_t b) {
        void * p = nullptr; ck(cudaMalloc(&p, b)); ck(cudaMemcpy(p, h, b, cudaMemcpyHostToDevice));
        allocations.push_back(p); return static_cast<const void *>(p);
    };
    const auto * wf = tqllm::weight_format(type);
    const auto m = wf->load(raw, e * rows, cols, upload, nullptr);
    const auto mh = wf->load(raw_hot, int(hot.size()) * rows, cols, upload, nullptr);
    const auto mc = wf->load(raw_cold, int(cold.size()) * rows, cols, upload, nullptr);
    const int nx = per_expert ? n * k : n;
    std::vector<__half> x(nx * cols);
    for (auto & v : x) v = __float2half(float(int(gen() % 101) - 50) / 100.f);
    std::vector<int> ids(n * k);
    for (int t = 0; t < n; ++t) {  // k distinct experts per token
        std::vector<int> pick(e); for (int i = 0; i < e; ++i) pick[i] = i;
        std::shuffle(pick.begin(), pick.end(), gen);
        for (int s = 0; s < k; ++s) ids[t * k + s] = pick[s];
    }
    Device<__half> dx(x.size()); dx.put(x);
    Device<int> di(ids.size()), dhs(e), dcs(e), dh(ids.size()), dc(ids.size());
    di.put(ids); dhs.put(hot_slot); dcs.put(cold_slot);
    Device<int> counts(e), groups(e * n), counts2(e), groups2(e * n);
    Device<__nv_bfloat16> ref(n * k * rows), out(n * k * rows);
    cudaStream_t stream; ck(cudaStreamCreate(&stream));
    auto launch = [&](const tqllm::Mat & mm, const int * id, int experts, int * cn, int * gr, __nv_bfloat16 * y) {
        if (use_gemm) {
            tqllm_cuda::moe_group(id, cn, gr, experts, n, k, stream);
            require(mm.fmt->expert_gemm(mm, dx.p, y, rows, cn, gr, n, experts, k, per_expert, stream), "no grouped GEMM");
        } else mm.fmt->expert_matvec(mm, dx.p, y, rows, id, n, k, per_expert, stream);
    };
    launch(m, di.p, e, counts.p, groups.p, ref.p);
    tqllm_cuda::moe_split_ids(di.p, dhs.p, dcs.p, dh.p, dc.p, n * k, stream);
    launch(mh, dh.p, int(hot.size()), counts.p, groups.p, out.p);
    launch(mc, dc.p, int(cold.size()), counts2.p, groups2.p, out.p);
    ck(cudaStreamSynchronize(stream));
    const auto y = out.get(n * k * rows), yr = ref.get(n * k * rows);
    if (std::memcmp(y.data(), yr.data(), y.size() * 2) != 0) {
        size_t i = 0; while (std::memcmp(&y[i], &yr[i], 2) == 0) ++i;
        ck(cudaGetLastError());
        const auto hh = dh.get(n * k), cc = dc.get(n * k);
        for (int q = 0; q < n * k; ++q) {
            int bad = 0; for (int r = 0; r < rows; ++r) bad += std::memcmp(&y[q * rows + r], &yr[q * rows + r], 2) != 0;
            std::fprintf(stderr, "split: assignment %d expert %d hot %d cold %d: %d of %d rows differ, row0 %g vs %g\n", q, ids[q], hh[q], cc[q], bad, rows,
                         __bfloat162float(y[q * rows]), __bfloat162float(yr[q * rows]));
        }
        std::fprintf(stderr, "split: type %u formats %s / %s / %s, first difference at %zu (assignment %zu row %zu): %g vs %g\n", type,
                     m.fmt->name(), mh.fmt->name(), mc.fmt->name(), i, i / rows, i % rows,
                     __bfloat162float(y[i]), __bfloat162float(yr[i]));
        throw std::runtime_error("hot / cold split differs from the single launch");
    }
    ck(cudaStreamDestroy(stream));
    for (void * p : allocations) ck(cudaFree(p));
    std::printf("GGUF %2u (%s) hot/cold split %s n=%d per_expert=%d cold=%zu: bit-identical\n", sp.type, sp.name,
                use_gemm ? "GEMM" : "matvec", n, per_expert, cold.size());
}

int main() {
    try {
        route_test();
        // Decode (grouped matvec) and prefill (grouped GEMM) for every format with grouped-expert
        // support -- not just the three that were wired first.
        for (const auto & sp : kSpecs) {
            for (int n : {1, 5, 17}) for (bool per : {false, true}) grouped_test(sp.type, n, per);
            for (int n : {1, 5, 65, 129}) for (bool per : {false, true}) grouped_test(sp.type, n, per, true);
            for (int n : {1, 5, 17}) for (bool per : {false, true}) split_test(sp.type, n, per, false);
            for (int n : {5, 65, 129}) for (bool per : {false, true}) split_test(sp.type, n, per, true);
            for (bool gemm : {false, true}) split_test(sp.type, 5, false, gemm, {4});  // cold set smaller than k
            for (int n : {1, 5}) pair_test(sp.type, n);
        }
        attention_group_test();
        combine_test();
        ck(cudaDeviceSynchronize());
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

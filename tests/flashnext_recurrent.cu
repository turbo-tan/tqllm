#include "qwen4exp_kernels.cuh"
#include "qwen35_kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template<class T> struct Device {
    T * p = nullptr; size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T> & v) {
        require(v.size() == n, "upload size differs");
        ck(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
};
static void compare(const std::vector<float> & a, const std::vector<double> & b, double atol = 2e-6, double rtol = 2e-5) {
    require(a.size() == b.size(), "oracle size differs");
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) {
            std::fprintf(stderr, "element=%zu actual=%.9g expected=%.12g\n", i, a[i], b[i]);
            throw std::runtime_error("Flash-Next recurrence differs from double oracle");
        }
}
static void gate_test(bool legacy) {
    constexpr int dim = 128;
    const float eps = 1e-6f;
    if (legacy) {
        Device<__nv_bfloat16> x(dim), z(dim), y(dim); Device<float> w(dim);
        x.put(std::vector<__nv_bfloat16>(dim, __float2bfloat16(1.f)));
        z.put(std::vector<__nv_bfloat16>(dim, __float2bfloat16(0.f))); w.put(std::vector<float>(dim, 1.f));
        tqllm_cuda::gated_rms(x.p, z.p, w.p, y.p, 1, dim, eps, nullptr);
        const auto got = y.get();
        require(std::abs(__bfloat162float(got[0]) - .5f) < .001f,
                "Qwen35 SiLU gate returns zero at z=0; Flash-Next requires sigmoid(z)=0.5");
        return;
    }
    constexpr int heads = 3, tokens = 5;
    const int count = dim * heads * tokens;
    std::vector<float> x(count), z(count), w(dim); std::vector<double> expected(count);
    for (int i = 0; i < count; ++i) {
        x[i] = 1.f + .05f * std::sin(float(i));
        z[i] = i < dim ? 0.f : i < 2 * dim ? 100.f : i < 3 * dim ? -100.f : std::cos(float(i));
    }
    for (int i = 0; i < dim; ++i) w[i] = .7f + .03f * std::cos(float(i));
    for (int r = 0; r < heads * tokens; ++r) {
        double ss = 0.; for (int j = 0; j < dim; ++j) ss += double(x[r * dim + j]) * x[r * dim + j];
        for (int j = 0; j < dim; ++j) {
            const int i = r * dim + j;
            expected[i] = x[i] / std::sqrt(ss / dim + eps) * w[j] / (1. + std::exp(-double(z[i])));
        }
    }
    Device<float> dx(count), dz(count), dw(dim), dy(count); dx.put(x); dz.put(z); dw.put(w);
    tqllm_cuda::fn_gated_rms(dx.p, dz.p, dw.p, dy.p, heads, dim, tokens, eps, nullptr);
    const auto got = dy.get(); compare(got, expected);
    tqllm_cuda::fn_gated_rms(dx.p, dz.p, dw.p, dx.p, heads, dim, tokens, eps, nullptr);
    require(dx.get() == got, "in-place gated RMS differs");
    std::puts("Flash-Next sigmoid output gate (zero/saturated/mixed/in-place): PASS");
}
static void legacy_normalization_test() {
    constexpr int dim = 128;
    Device<__nv_bfloat16> x(3 * dim), a(1), b(1), y(dim);
    Device<float> neg(1), bias(1), state(dim * dim), next(dim * dim);
    std::vector<__nv_bfloat16> qkv(3 * dim, __float2bfloat16(1e-8f));
    std::fill(qkv.begin() + 2 * dim, qkv.end(), __float2bfloat16(1.f));
    x.put(qkv); a.put({__float2bfloat16(0.f)}); b.put({__float2bfloat16(0.f)});
    neg.put({-1.f}); bias.put({0.f}); state.put(std::vector<float>(dim * dim, 0.f));
    tqllm_cuda::gdn_step2(x.p, a.p, b.p, neg.p, bias.p, state.p, next.p, y.p, 1, 1, dim, dim, true, nullptr);
    const double expected = .5 * dim * 1e-4 / std::sqrt(double(dim));
    require(std::abs(__bfloat162float(y.get()[0]) - expected) < 1e-6,
            "Qwen35 sum-plus-1e-6 L2 normalization does not implement Flash-Next max(norm,epsilon)");
}
static void attention_test(int heads, int kvheads, int dim, int pos, int n) {
    const int count = heads * dim * n, rot = 64, total = pos + n;
    const float eps = 1e-6f, theta = 10000000.f;
    std::vector<float> qg(size_t(count) * 2), gamma(dim);
    for (size_t i = 0; i < qg.size(); ++i) qg[i] = .3f * std::sin(float(i) * .071f);
    for (int d = 0; d < dim; ++d) gamma[d] = .8f + .13f * std::cos(float(d));
    Device<float> dx(qg.size()), dw(dim), dq(count), out(count); dx.put(qg); dw.put(gamma);
    tqllm_cuda::fn_norm_rope(dx.p, dw.p, dq.p, heads, dim, n, pos, rot, theta, eps, true, nullptr);
    const auto q = dq.get(); std::vector<double> norm_expected(count);
    for (int t = 0; t < n; ++t) for (int h = 0; h < heads; ++h) {
        const size_t base = (size_t(t) * heads + h) * dim;
        double ss = 0.; for (int d = 0; d < dim; ++d) ss += double(qg[2 * base + d]) * qg[2 * base + d];
        const double inv = 1. / std::sqrt(ss / dim + eps);
        for (int d = 0; d < dim; ++d) {
            double value = qg[2 * base + d] * inv * gamma[d];
            if (d < rot) {
                const double angle = (pos + t) / std::pow(double(theta), double(2 * (d % (rot / 2))) / rot);
                const int pair = d < rot / 2 ? d + rot / 2 : d - rot / 2;
                const double other = qg[2 * base + pair] * inv * gamma[pair];
                value = value * std::cos(angle) + (d < rot / 2 ? -other : other) * std::sin(angle);
            }
            norm_expected[base + d] = value;
        }
    }
    compare(q, norm_expected, 2e-6, 2e-5);
    std::vector<__nv_bfloat16> k(size_t(total) * kvheads * dim), v(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = __float2bfloat16(.5f * std::sin(float(i) * .049f));
        v[i] = __float2bfloat16(.6f * std::cos(float(i) * .073f));
    }
    Device<__nv_bfloat16> dk(k.size()), dv(v.size()); dk.put(k); dv.put(v);
    std::vector<double> expected(count);
    for (int t = 0; t < n; ++t) for (int h = 0; h < heads; ++h) {
        const int kh = h / (heads / kvheads), seq = pos + t + 1;
        const size_t base = (size_t(t) * heads + h) * dim;
        std::vector<double> scores(seq); double maximum = -INFINITY, sum = 0.;
        for (int j = 0; j < seq; ++j) {
            double dot = 0.;
            for (int d = 0; d < dim; ++d) dot += double(q[base + d]) * __bfloat162float(k[(size_t(j) * kvheads + kh) * dim + d]);
            scores[j] = dot / std::sqrt(double(dim)); maximum = std::max(maximum, scores[j]);
        }
        for (auto & score : scores) { score = std::exp(score - maximum); sum += score; }
        for (int d = 0; d < dim; ++d) {
            double value = 0.;
            for (int j = 0; j < seq; ++j) value += scores[j] / sum * __bfloat162float(v[(size_t(j) * kvheads + kh) * dim + d]);
            expected[base + d] = value;
        }
    }
    cudaStream_t stream = cudaStreamPerThread;
    auto run = [&] { tqllm_cuda::fn_dense_attention(dq.p, dk.p, dv.p, out.p, heads, kvheads, dim, n, pos, stream); };
    run(); ck(cudaStreamSynchronize(stream)); const auto got = out.get(); compare(got, expected);
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); run(); ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0)); ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    require(out.get() == got, "attention CUDA graph replay differs");
    tqllm_cuda::fn_attention_gate(out.p, dx.p, heads, dim, n, stream); ck(cudaStreamSynchronize(stream));
    for (int i = 0; i < count; ++i) expected[i] /= 1. + std::exp(-double(qg[(size_t(i) / dim) * 2 * dim + dim + i % dim]));
    compare(out.get(), expected);
    bool rejected = false;
    try { tqllm_cuda::fn_dense_attention(dq.p, dk.p, dv.p, out.p, heads, kvheads, dim, n, 1 << 20, stream); }  // past the 1M-position budget
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "a context past the dense-attention budget was accepted");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); 
    std::printf("Flash-Next text attention heads=%d kv=%d dim=%d pos=%d n=%d BF16/causal/gate/oracle/graph: PASS\n", heads, kvheads, dim, pos, n);
}

static void recurrence_test(int kh, int vh, int dim, int n, float input_scale) {
    const int channels = (2 * kh + vh) * dim, state_count = vh * dim * dim;
    const float eps = 1e-6f;
    std::vector<float> x(size_t(n) * channels), a(n * vh), b(a.size()), neg(vh), bias(vh), initial(state_count);
    for (size_t i = 0; i < x.size(); ++i) x[i] = input_scale * std::sin(float(i) * .017f + .03f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = i % 11 == 0 ? 100.f : i % 13 == 0 ? -100.f : .3f * std::cos(float(i));
        b[i] = i % 7 == 0 ? -100.f : i % 17 == 0 ? 100.f : .7f * std::sin(float(i));
    }
    for (int i = 0; i < vh; ++i) { neg[i] = -.2f - .017f * i; bias[i] = .1f * std::cos(float(i)); }
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = .03f * std::sin(float(i) * .021f);
    std::vector<double> state(initial.begin(), initial.end()), expected(size_t(n) * vh * dim);
    for (int t = 0; t < n; ++t) for (int h = 0; h < vh; ++h) {
        std::vector<double> q(dim), k(dim); double qq = 0., kk = 0.;
        for (int i = 0; i < dim; ++i) {
            q[i] = x[size_t(t) * channels + (h % kh) * dim + i];
            k[i] = x[size_t(t) * channels + kh * dim + (h % kh) * dim + i]; qq += q[i] * q[i]; kk += k[i] * k[i];
        }
        for (int i = 0; i < dim; ++i) {
            q[i] /= std::max(std::sqrt(qq), double(eps)) * std::sqrt(double(dim));
            k[i] /= std::max(std::sqrt(kk), double(eps));
        }
        const double av = double(a[t * vh + h]) + bias[h];
        const double decay = std::exp(neg[h] * (std::max(av, 0.) + std::log1p(std::exp(-std::abs(av)))));
        const double beta = 1. / (1. + std::exp(-double(b[t * vh + h])));
        for (int value = 0; value < dim; ++value) {
            const size_t base = (size_t(h) * dim + value) * dim;
            double memory = 0.; for (int i = 0; i < dim; ++i) { state[base + i] *= decay; memory += state[base + i] * k[i]; }
            const double delta = (x[size_t(t) * channels + 2 * kh * dim + h * dim + value] - memory) * beta;
            double out = 0.;
            for (int i = 0; i < dim; ++i) { state[base + i] += k[i] * delta; out += state[base + i] * q[i]; }
            expected[(size_t(t) * vh + h) * dim + value] = out;
        }
    }
    Device<float> dx(x.size()), da(a.size()), db(b.size()), dn(vh), dw(vh), ds(state_count), next(state_count), dy(expected.size());
    dx.put(x); da.put(a); db.put(b); dn.put(neg); dw.put(bias); ds.put(initial);
    cudaStream_t stream = cudaStreamPerThread;
    auto run = [&] { tqllm_cuda::fn_gdn(dx.p, da.p, db.p, dn.p, dw.p, ds.p, next.p, dy.p, kh, vh, dim, n, eps, stream); };
    run(); ck(cudaStreamSynchronize(stream)); const auto got = dy.get(), final = next.get();
    compare(got, expected); compare(final, state);
    // Stateful chunking uses the same sequence state, without recomputing any prefix.
    if (n > 1) {
        const int first = n / 2;
        tqllm_cuda::fn_gdn(dx.p, da.p, db.p, dn.p, dw.p, ds.p, next.p, dy.p, kh, vh, dim, first, eps, stream);
        tqllm_cuda::fn_gdn(dx.p + size_t(first) * channels, da.p + first * vh, db.p + first * vh, dn.p, dw.p,
                          next.p, next.p, dy.p + size_t(first) * vh * dim, kh, vh, dim, n - first, eps, stream);
        ck(cudaStreamSynchronize(stream));
        require(dy.get() == got && next.get() == final, "chunked recurrence/state differs from one batch");
    }
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)); run(); ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    require(dy.get() == got && next.get() == final, "graph recurrence replay differs");
    // Changing the live state must affect a captured graph, then resetting it must restore output.
    ds.put(std::vector<float>(state_count, 0.f)); ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    if (input_scale > 0.f) require(dy.get() != got, "captured graph ignores updated state");
    ds.put(initial); ck(cudaGraphLaunch(exec, stream)); ck(cudaStreamSynchronize(stream));
    require(dy.get() == got && next.get() == final, "sequence reset does not restore graph output");
    bool rejected = false;
    try { tqllm_cuda::fn_gdn(dx.p, da.p, db.p, dn.p, dw.p, ds.p, next.p, dy.p, kh, vh, 31, n, eps, stream); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid GDN width accepted");
    rejected = false;
    try { tqllm_cuda::fn_gdn(dx.p, da.p, db.p, dn.p, dw.p, ds.p, next.p, dy.p, kh, vh, dim, n, 0.f, stream); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "zero L2 epsilon accepted");
    rejected = false;
    try { tqllm_cuda::fn_gdn(dx.p, da.p, db.p, dn.p, dw.p, ds.p, dy.p, dy.p, kh, vh, dim, n, eps, stream); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "aliased GDN state/output accepted");
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); 
    std::printf("Flash-Next GDN kh=%d vh=%d dim=%d n=%d scale=%g oracle/chunk/reset/graph: PASS\n", kh, vh, dim, n, input_scale);
}
static void shape_overflow_test() {
    Device<float> buffers(4);
    constexpr int huge = std::numeric_limits<int>::max();
    bool rejected = false;
    try { tqllm_cuda::fn_gated_rms(buffers.p, buffers.p + 1, buffers.p + 2, buffers.p + 3,
                                 huge, huge, 3, 1e-6f, nullptr); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "overflowed gated RMS element count accepted");
    rejected = false;
    try { tqllm_cuda::fn_norm_rope(buffers.p, buffers.p + 1, buffers.p + 2, huge, huge, 3,
                                 0, 64, 10000000.f, 1e-6f, false, nullptr); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "overflowed norm/RoPE element count accepted");
    rejected = false;
    try { tqllm_cuda::fn_attention_gate(buffers.p, buffers.p + 1, huge, huge, 3, nullptr); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "overflowed attention gate element count accepted");
    ck(cudaGetLastError());
    std::puts("Flash-Next shape overflow rejection: PASS");
}
static void combine_check_test(bool legacy = false) {
    constexpr int h = 257, k = 3, n = 3;
    std::vector<__nv_bfloat16> routed(size_t(n) * k * h, __float2bfloat16(0.5f));
    std::vector<float> weights(n * k, 0.25f), shared(n * h, -0.5f), sw(n, 0.125f);
    Device<__nv_bfloat16> dr(routed.size()); Device<float> dw(weights.size()), ds(shared.size()), dsw(sw.size());
    Device<float> out(n * h), reference(n * h); Device<int> bad(1);
    auto run = [&](bool expected_bad) {
        dr.put(routed); dw.put(weights); ds.put(shared); dsw.put(sw); bad.put({0});
        tqllm_cuda::fn_moe_combine(dr.p, dw.p, ds.p, dsw.p, reference.p, h, k, n, nullptr);
        if (legacy) tqllm_cuda::fn_moe_combine(dr.p, dw.p, ds.p, dsw.p, out.p, h, k, n, nullptr);
        else tqllm_cuda::fn_moe_combine_checked(dr.p, dw.p, ds.p, dsw.p, out.p, h, k, n, bad.p, nullptr);
        const auto got = out.get(), want = reference.get();
        const bool host_bad = !std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); });
        require(host_bad == expected_bad, "combine test did not exercise the intended finite/nonfinite case");
        require((bad.get()[0] != 0) == host_bad, "device finite check differs from full host scan");
        require(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0,
                "checked combine changed output bits");
    };
    run(false);
    routed.back() = __float2bfloat16(std::numeric_limits<float>::quiet_NaN()); run(true);
    routed.back() = __float2bfloat16(0.5f);
    shared.back() = std::numeric_limits<float>::infinity(); run(true);
    shared.back() = -std::numeric_limits<float>::infinity(); run(true);
    shared.back() = -0.5f;
    weights[0] = std::numeric_limits<float>::quiet_NaN(); run(true);
    weights[0] = std::numeric_limits<float>::max(); routed[0] = __float2bfloat16(2.f); run(true);
    weights[0] = 0.25f; routed[0] = __float2bfloat16(0.5f);
    shared.back() = -0.f; run(false);
    ck(cudaGetLastError());
    std::puts("Flash-Next checked combine: PASS (finite, NaN, +/-Inf, overflow, reset, odd tail; bits unchanged)");
}
int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--combine-check") { combine_check_test(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--legacy-combine-check") { combine_check_test(true); return 0; }
        combine_check_test();
        if (argc == 2 && std::string(argv[1]) == "--shape-overflow") { shape_overflow_test(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "--legacy-normalization") { legacy_normalization_test(); return 0; }
        const bool legacy = argc == 2 && std::string(argv[1]) == "--legacy-gate";
        gate_test(legacy); if (legacy) return 0;
        shape_overflow_test();
        for (int n : {1, 5, 17}) for (float scale : {0.f, 1e-8f, .4f}) recurrence_test(2, 6, 32, n, scale);
        recurrence_test(4, 12, 256, 5, .4f);
        recurrence_test(16, 48, 128, 5, .4f);
        attention_test(4, 2, 128, 0, 5);
        attention_test(24, 2, 256, 37, 5);
        attention_test(4, 2, 256, 2047, 1);
        std::puts("FLASHNEXT_RECURRENT_ATTENTION_DONE cases=16"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

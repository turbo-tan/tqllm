// GLM-5.3-Flash kernels (src/models/glm/glm5next.cu) against CPU code that is the math of tools/glm_ref.cpp, which
// matches llama.cpp. Stateful kernels (KDA conv history and recurrence, MLA cache) run several steps.
#include "glm5next_kernels.cuh"
#include "moe.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
int failures = 0;
std::mt19937 rng(7);
std::vector<float> rnd(size_t n, float lo = -1.f, float hi = 1.f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = u(rng);
    return v;
}
template <class T> T * dev(const std::vector<T> & h) {
    T * p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> T * dev0(size_t n) {
    T * p = nullptr;
    cudaMalloc(&p, n * sizeof(T));
    cudaMemset(p, 0, n * sizeof(T));
    return p;
}
template <class T> std::vector<T> host(const T * p, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
void compare(const char * what, const std::vector<float> & got, const std::vector<float> & want, float tol) {
    float worst = 0;
    for (size_t i = 0; i < want.size(); ++i)
        worst = std::max(worst, std::fabs(got[i] - want[i]) / std::max(1.f, std::fabs(want[i])));
    const bool ok = worst <= tol && got.size() == want.size();
    if (!ok) ++failures;
    std::printf("%s %s (max rel err %.2e)\n", ok ? "PASS" : "FAIL", what, double(worst));
}
float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }
float silu(float x) { return x / (1.f + std::exp(-x)); }
void rms(float * x, int n, const float * w, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float s = float(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) x[i] = x[i] * s * (w ? w[i] : 1.f);
}

void test_hc() {
    const int H = 300, hc = 4, iters = 20;
    const float eps = 1e-6f;
    auto mix = rnd(24, -3, 3), scale = rnd(3, 0.01f, 0.2f), base = rnd(24, -2, 2), S = rnd(size_t(hc) * H), f = rnd(size_t(H));
    // CPU: glm_ref.cpp hc_pre / hc_post
    std::vector<float> pre(4), post(4), C(16), x(size_t(H), 0.f), out(size_t(hc) * H);
    for (int i = 0; i < hc; ++i) { pre[i] = sigm(mix[i] * scale[0] + base[i]) + eps; post[i] = 2 * sigm(mix[hc + i] * scale[1] + base[hc + i]); }
    auto c = [&](int d, int s) -> float & { return C[size_t(d) * hc + s]; };
    for (int s = 0; s < hc; ++s) {
        float m = -INFINITY;
        for (int d = 0; d < hc; ++d) { c(d, s) = mix[size_t(2 * hc + d + hc * s)] * scale[2] + base[size_t(2 * hc + d + hc * s)]; m = std::max(m, c(d, s)); }
        float z = 0;
        for (int d = 0; d < hc; ++d) z += (c(d, s) = std::exp(c(d, s) - m));
        for (int d = 0; d < hc; ++d) c(d, s) = c(d, s) / z + eps;
    }
    auto cols = [&] { for (int d = 0; d < hc; ++d) { float z = 0; for (int s = 0; s < hc; ++s) z += c(d, s); for (int s = 0; s < hc; ++s) c(d, s) /= z + eps; } };
    auto rows = [&] { for (int s = 0; s < hc; ++s) { float z = 0; for (int d = 0; d < hc; ++d) z += c(d, s); for (int d = 0; d < hc; ++d) c(d, s) /= z + eps; } };
    cols();
    for (int i = 1; i < iters; ++i) { rows(); cols(); }
    for (int i = 0; i < hc; ++i) for (int j = 0; j < H; ++j) x[size_t(j)] += pre[i] * S[size_t(i) * H + j];
    for (int d = 0; d < hc; ++d) for (int j = 0; j < H; ++j) {
        float v = f[size_t(j)] * post[d];
        for (int s = 0; s < hc; ++s) v += S[size_t(s) * H + j] * c(d, s);
        out[size_t(d) * H + j] = v;
    }
    float * dmix = dev(mix), * dsc = dev(scale), * dba = dev(base), * dS = dev(S), * df = dev(f);
    float * dx = dev0<float>(size_t(H)), * dpost = dev0<float>(4), * dcomb = dev0<float>(16), * dout = dev0<float>(size_t(hc) * H);
    tqllm_cuda::glm_hc_pre(dmix, dsc, dba, dS, dx, dpost, dcomb, H, iters, eps, nullptr);
    tqllm_cuda::glm_hc_post(df, dS, dpost, dcomb, dout, H, nullptr);
    compare("hc_pre: post", host(dpost, 4), post, 1e-5f);
    compare("hc_pre: Sinkhorn comb", host(dcomb, 16), C, 1e-5f);
    compare("hc_pre: sublayer input", host(dx, size_t(H)), x, 1e-5f);
    compare("hc_post", host(dout, size_t(hc) * H), out, 1e-5f);
    std::vector<float> mean(size_t(H), 0.f);
    for (int i = 0; i < hc; ++i) for (int j = 0; j < H; ++j) mean[size_t(j)] += S[size_t(i) * H + j] / 4.f;
    tqllm_cuda::glm_hc_mean(dS, dx, H, nullptr);
    compare("hc_mean", host(dx, size_t(H)), mean, 1e-6f);
}

void test_rms_mv() {
    const int n = 5000;
    auto x = rnd(size_t(n)), w = rnd(size_t(n));
    auto want = x;
    rms(want.data(), n, w.data(), 1e-5f);
    float * dx = dev(x), * dw = dev(w), * dy = dev0<float>(size_t(n));
    tqllm_cuda::glm_rmsnorm(dx, dw, dy, nullptr, n, 1e-5f, nullptr);
    compare("rmsnorm", host(dy, size_t(n)), want, 1e-5f);
    {  // short rows (one warp per row): 37 rows of 128, FP32 and FP16 outputs
        const int sr = 37, sn = 128;
        auto xs = rnd(size_t(sr) * sn), ws = rnd(size_t(sn));
        auto ref = xs;
        for (int r = 0; r < sr; ++r) rms(ref.data() + size_t(r) * sn, sn, ws.data(), 1e-5f);
        float * dxs = dev(xs), * dws = dev(ws), * dys = dev0<float>(xs.size());
        __half * dyh = dev0<__half>(xs.size());
        tqllm_cuda::glm_rmsnorm(dxs, dws, dys, dyh, sn, 1e-5f, nullptr, sr);
        compare("rmsnorm short rows", host(dys, xs.size()), ref, 1e-5f);
        const auto hh = host(dyh, xs.size());
        std::vector<float> hf(hh.size()); for (size_t i = 0; i < hh.size(); ++i) hf[i] = __half2float(hh[i]);
        compare("rmsnorm short rows fp16", hf, ref, 2e-2f);
    }
    const int rows = 37, cols = 777;
    auto W = rnd(size_t(rows) * cols), v = rnd(size_t(cols));
    std::vector<float> y(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) { double a = 0; for (int c = 0; c < cols; ++c) a += double(W[size_t(r) * cols + c]) * v[size_t(c)]; y[size_t(r)] = float(a); }
    float * dW = dev(W), * dv = dev(v), * dyy = dev0<float>(size_t(rows));
    tqllm_cuda::glm_mv_f32(dW, dv, dyy, rows, cols, 1, nullptr);
    compare("mv_f32", host(dyy, size_t(rows)), y, 1e-5f);
    {  // split-K for few rows: hc_fn shape (24 x 16384), 3 tokens
        const int r2 = 24, c2 = 16384, n2 = 3, sp = 32;
        auto W2 = rnd(size_t(r2) * c2), x2 = rnd(size_t(n2) * c2);
        std::vector<float> y2(size_t(n2) * r2);
        for (int t = 0; t < n2; ++t) for (int r = 0; r < r2; ++r) {
            double a = 0;
            for (int c = 0; c < c2; ++c) a += double(W2[size_t(r) * c2 + c]) * x2[size_t(t) * c2 + c];
            y2[size_t(t) * r2 + r] = float(a);
        }
        float * dW2 = dev(W2), * dx2 = dev(x2), * dy2 = dev0<float>(y2.size()), * part = dev0<float>(size_t(n2) * r2 * sp);
        tqllm_cuda::glm_mv_f32_splitk(dW2, dx2, dy2, part, r2, c2, n2, sp, nullptr);
        compare("mv_f32 split-K (24 x 16384, 3 tokens)", host(dy2, y2.size()), y2, 1e-4f);
        const auto first = host(dy2, y2.size());
        tqllm_cuda::glm_mv_f32_splitk(dW2, dx2, dy2, part, r2, c2, n2, sp, nullptr);
        compare("mv_f32 split-K is deterministic", host(dy2, y2.size()), first, 0.f);
    }
}

void test_kda() {
    const int heads = 4, D = 128, d = heads * D, K = 4, steps = 6;
    const float lb = -5.f, eps = 1e-5f;
    auto w = rnd(size_t(3) * d * K), dtb = rnd(size_t(d)), A = rnd(size_t(heads), -2.f, -0.1f), onw = rnd(size_t(D), 0.5f, 1.5f);
    std::vector<float> hist(size_t(3) * (K - 1) * d, 0.f), S(size_t(heads) * D * D, 0.f);
    float * dw = dev(w), * dhist = dev0<float>(hist.size()), * dS = dev0<float>(S.size()), * ddtb = dev(dtb), * dA = dev(A), * don = dev(onw);
    float * dproj = dev0<float>(size_t(3) * d), * dcv = dev0<float>(size_t(3) * d), * dg = dev0<float>(size_t(d)),
          * dbeta = dev0<float>(size_t(heads)), * dout = dev0<float>(size_t(d)), * dg2 = dev0<float>(size_t(d)), * dy = dev0<float>(size_t(d));
    for (int t = 0; t < steps; ++t) {
        auto proj = rnd(size_t(3) * d, -2, 2), g = rnd(size_t(d), -3, 3), beta = rnd(size_t(heads), -2, 2), g2 = rnd(size_t(d), -3, 3);
        // CPU: glm_ref.cpp kda()
        std::vector<float> cv(size_t(3) * d);
        for (int p = 0; p < 3; ++p) {
            float * h = hist.data() + size_t(p) * (K - 1) * d;
            for (int ch = 0; ch < d; ++ch) {
                const float * ww = w.data() + size_t(p) * d * K + size_t(ch) * K;
                float acc = 0;
                for (int j = 0; j < K - 1; ++j) acc += h[size_t(j) * d + ch] * ww[j];
                acc += proj[size_t(p) * d + ch] * ww[K - 1];
                cv[size_t(p) * d + ch] = silu(acc);
            }
            for (int j = 0; j + 1 < K - 1; ++j) std::copy(h + size_t(j + 1) * d, h + size_t(j + 2) * d, h + size_t(j) * d);
            std::copy(proj.begin() + p * d, proj.begin() + (p + 1) * d, h + size_t(K - 2) * d);
        }
        std::vector<float> gg = g;
        for (int i = 0; i < d; ++i) gg[size_t(i)] = lb * sigm(-((g[size_t(i)] + dtb[size_t(i)]) * A[size_t(i / D)]));
        std::vector<float> bb = beta, o(static_cast<size_t>(d));
        for (auto & b : bb) b = sigm(b);
        for (int h = 0; h < heads; ++h) {
            std::vector<float> q(cv.begin() + h * D, cv.begin() + (h + 1) * D), k(cv.begin() + d + h * D, cv.begin() + d + (h + 1) * D);
            const float * v = cv.data() + 2 * d + h * D;
            for (auto * z : {&q, &k}) { double ss = 0; for (float e : *z) ss += double(e) * e; const float s = float(1 / std::sqrt(ss + 1e-6)); for (auto & e : *z) e *= s; }
            for (auto & e : q) e /= std::sqrt(float(D));
            float * St = S.data() + size_t(h) * D * D;
            for (int a = 0; a < D; ++a) { const float e = std::exp(gg[size_t(h) * D + a]); for (int b = 0; b < D; ++b) St[size_t(a) * D + b] *= e; }
            for (int b = 0; b < D; ++b) {
                float sk = 0;
                for (int a = 0; a < D; ++a) sk += St[size_t(a) * D + b] * k[size_t(a)];
                const float dl = (v[b] - sk) * bb[size_t(h)];
                for (int a = 0; a < D; ++a) St[size_t(a) * D + b] += k[size_t(a)] * dl;
            }
            for (int b = 0; b < D; ++b) { float acc = 0; for (int a = 0; a < D; ++a) acc += St[size_t(a) * D + b] * q[size_t(a)]; o[size_t(h) * D + b] = acc; }
        }
        std::vector<float> y = o;
        for (int h = 0; h < heads; ++h) {
            rms(y.data() + size_t(h) * D, D, onw.data(), eps);
            for (int i = 0; i < D; ++i) y[size_t(h) * D + i] *= sigm(g2[size_t(h) * D + i]);
        }
        cudaMemcpy(dproj, proj.data(), proj.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dg, g.data(), g.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dbeta, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dg2, g2.data(), g2.size() * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::glm_kda_conv(dproj, dhist, dw, dcv, d, K, nullptr);
        tqllm_cuda::glm_kda_gate(dg, ddtb, dA, lb, heads, D, nullptr);
        tqllm_cuda::glm_sigmoid(dbeta, heads, nullptr);
        tqllm_cuda::glm_kda_recur(dcv, dcv + d, dcv + 2 * d, dg, dbeta, dS, nullptr, dout, heads, D, nullptr);
        tqllm_cuda::glm_kda_out(dout, dg2, don, dy, nullptr, heads, D, eps, nullptr);
        char what[64];
        std::snprintf(what, sizeof what, "kda step %d: conv", t); compare(what, host(dcv, cv.size()), cv, 1e-5f);
        std::snprintf(what, sizeof what, "kda step %d: gate", t); compare(what, host(dg, gg.size()), gg, 1e-5f);
        std::snprintf(what, sizeof what, "kda step %d: recurrence", t); compare(what, host(dout, o.size()), o, 1e-4f);
        std::snprintf(what, sizeof what, "kda step %d: gated norm", t); compare(what, host(dy, y.size()), y, 1e-4f);
    }
    compare("kda: state after the last step", host(dS, S.size()), S, 1e-4f);
}

void test_mla() {
    const int heads = 3, R = 512, dk = 256, dv = 256, n = 9;
    const float scale = 1.f / 16.f;
    auto q = rnd(size_t(heads) * dk), kb = rnd(size_t(heads) * R * dk, -0.1f, 0.1f), vb = rnd(size_t(heads) * dv * R, -0.1f, 0.1f),
         lat = rnd(size_t(n) * R);
    std::vector<__half> kbh(kb.size()), vbh(vb.size());
    for (size_t i = 0; i < kb.size(); ++i) { kbh[i] = __float2half(kb[i]); kb[i] = __half2float(kbh[i]); }
    for (size_t i = 0; i < vb.size(); ++i) { vbh[i] = __float2half(vb[i]); vb[i] = __half2float(vbh[i]); }
    std::vector<float> qa(size_t(heads) * R), kv(size_t(heads) * R), out(size_t(heads) * dv);
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < R; ++j) { double a = 0; for (int i = 0; i < dk; ++i) a += double(kb[(size_t(h) * R + j) * dk + i]) * q[size_t(h) * dk + i]; qa[size_t(h) * R + j] = float(a); }
        std::vector<double> p(static_cast<size_t>(n));
        double m = -1e300, z = 0;
        for (int t = 0; t < n; ++t) { double s = 0; for (int j = 0; j < R; ++j) s += double(qa[size_t(h) * R + j]) * lat[size_t(t) * R + j]; p[size_t(t)] = s * scale; m = std::max(m, p[size_t(t)]); }
        for (auto & e : p) z += (e = std::exp(e - m));
        for (int j = 0; j < R; ++j) { double a = 0; for (int t = 0; t < n; ++t) a += p[size_t(t)] / z * lat[size_t(t) * R + j]; kv[size_t(h) * R + j] = float(a); }
        for (int i = 0; i < dv; ++i) { double a = 0; for (int j = 0; j < R; ++j) a += double(vb[(size_t(h) * dv + i) * R + j]) * kv[size_t(h) * R + j]; out[size_t(h) * dv + i] = float(a); }
    }
    float * dq = dev(q), * dlat = dev(lat), * dqa = dev0<float>(qa.size()), * dkv = dev0<float>(kv.size()), * dout = dev0<float>(out.size());
    __half * dkb = dev(kbh), * dvb = dev(vbh);
    tqllm_cuda::glm_mla_qabs(dq, dkb, dqa, heads, R, dk, nullptr);
    tqllm_cuda::glm_mla_attn(dqa, dlat, dkv, heads, R, n, scale, nullptr);
    tqllm_cuda::glm_mla_vb(dkv, dvb, dout, nullptr, heads, R, dv, nullptr);
    compare("mla: absorbed query", host(dqa, qa.size()), qa, 1e-4f);
    compare("mla: attention over the latent cache", host(dkv, kv.size()), kv, 1e-4f);
    compare("mla: value up-projection", host(dout, out.size()), out, 1e-4f);
}

void test_route() {
    const int E = 288, k = 8;
    for (int trial = 0; trial < 20; ++trial) {
        auto logits = rnd(size_t(E), -6, 6), bias = rnd(size_t(E), -0.5f, 0.5f);
        tqllm::MoeConfig mc;
        mc.n_expert = E; mc.n_expert_used = k; mc.gating = tqllm::MoeGating::Sigmoid; mc.norm_weights = true; mc.weight_scale = 2.5f;
        const auto sel = tqllm::moe_route(mc, logits.data(), bias.data());
        float * dl = dev(logits), * db = dev(bias), * dw = dev0<float>(size_t(k));
        int * di = dev0<int>(size_t(k));
        tqllm_cuda::glm_route(dl, db, di, dw, E, k, true, 2.5f, nullptr);
        const auto ids = host(di, size_t(k));
        const auto w = host(dw, size_t(k));
        if (ids != sel.ids) { ++failures; std::printf("FAIL route trial %d: selected experts differ\n", trial); }
        else compare("route weights", w, sel.weights, 1e-6f);
    }
}

void test_ffn_bits() {
    const int n = 1000, k = 8, H = 300;
    const float limit = 1.5f;
    auto g = rnd(size_t(n), -3, 3), u = rnd(size_t(n), -3, 3);
    std::vector<__nv_bfloat16> gb(static_cast<size_t>(n)), ub(static_cast<size_t>(n));
    std::vector<float> want(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        gb[size_t(i)] = __float2bfloat16(g[size_t(i)]); ub[size_t(i)] = __float2bfloat16(u[size_t(i)]);
        const float gg = std::min(__bfloat162float(gb[size_t(i)]), limit), uu = std::clamp(__bfloat162float(ub[size_t(i)]), -limit, limit);
        want[size_t(i)] = __half2float(__float2half(silu(gg) * uu));
    }
    __nv_bfloat16 * dg = dev(gb), * du = dev(ub);
    __half * dout = dev0<__half>(size_t(n));
    tqllm_cuda::glm_swiglu(dg, du, dout, n, limit, nullptr);
    const auto oh = host(dout, size_t(n));
    std::vector<float> of(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) of[size_t(i)] = __half2float(oh[size_t(i)]);
    compare("swiglu with clamp", of, want, 1e-3f);
    auto y = rnd(size_t(k) * H), w = rnd(size_t(k)), sh = rnd(size_t(H));
    std::vector<__nv_bfloat16> yb(y.size());
    std::vector<float> sum(static_cast<size_t>(H));
    for (size_t i = 0; i < y.size(); ++i) yb[i] = __float2bfloat16(y[i]);
    for (int j = 0; j < H; ++j) { float v = sh[size_t(j)]; for (int i = 0; i < k; ++i) v += w[size_t(i)] * __bfloat162float(yb[size_t(i) * H + j]); sum[size_t(j)] = v; }
    float * dw = dev(w), * dsh = dev(sh), * ds = dev0<float>(size_t(H));
    __nv_bfloat16 * dy = dev(yb);
    tqllm_cuda::glm_moe_sum(dy, dw, dsh, ds, k, H, nullptr);
    compare("moe weighted sum", host(ds, size_t(H)), sum, 1e-5f);
    auto x = rnd(100000);
    x[77777] = 5.f;
    int * da = dev0<int>(1);
    tqllm_cuda::glm_argmax(dev(x), int(x.size()), da, nullptr);
    if (host(da, 1)[0] != 77777) { ++failures; std::printf("FAIL argmax\n"); } else std::printf("PASS argmax\n");
}

// chunked kernels (n tokens per call) must equal n single-token calls bit for bit
void test_chunks() {
    const int heads = 4, D = 128, d = heads * D, K = 4, n = 5;
    auto w = rnd(size_t(3) * d * K), dtb = rnd(size_t(d)), A = rnd(size_t(heads), -2.f, -0.1f), onw = rnd(size_t(D), 0.5f, 1.5f);
    auto proj = rnd(size_t(3) * n * d, -2, 2), g = rnd(size_t(n) * d, -3, 3), beta = rnd(size_t(n) * heads, -2, 2),
         g2 = rnd(size_t(n) * d, -3, 3);
    float * dw = dev(w), * ddtb = dev(dtb), * dA = dev(A), * don = dev(onw);
    // chunk: proj [3][n][d]
    float * hc = dev0<float>(size_t(3) * (K - 1) * d), * Sc = dev0<float>(size_t(heads) * D * D), * pc = dev(proj),
          * cvc = dev0<float>(size_t(3) * n * d), * gc = dev(g), * bc = dev(beta), * oc = dev0<float>(size_t(n) * d),
          * g2c = dev(g2), * yc = dev0<float>(size_t(n) * d);
    tqllm_cuda::glm_kda_conv(pc, hc, dw, cvc, d, K, nullptr, n);
    tqllm_cuda::glm_kda_gate(gc, ddtb, dA, -5.f, heads, D, nullptr, n);
    tqllm_cuda::glm_sigmoid(bc, n * heads, nullptr);
    tqllm_cuda::glm_kda_recur(cvc, cvc + size_t(n) * d, cvc + size_t(2) * n * d, gc, bc, Sc, nullptr, oc, heads, D, nullptr, n, d);
    tqllm_cuda::glm_kda_out(oc, g2c, don, yc, nullptr, heads, D, 1e-5f, nullptr, n);
    // one token at a time: proj [3][d] per token
    float * h1 = dev0<float>(size_t(3) * (K - 1) * d), * S1 = dev0<float>(size_t(heads) * D * D), * p1 = dev0<float>(size_t(3) * d),
          * cv1 = dev0<float>(size_t(3) * d), * g1 = dev0<float>(size_t(d)), * b1 = dev0<float>(size_t(heads)),
          * o1 = dev0<float>(size_t(d)), * g21 = dev0<float>(size_t(d)), * y1 = dev0<float>(size_t(d));
    std::vector<float> cv_seq(size_t(3) * n * d), o_seq(size_t(n) * d), y_seq(size_t(n) * d), g_seq(size_t(n) * d);
    for (int t = 0; t < n; ++t) {
        for (int p = 0; p < 3; ++p)
            cudaMemcpy(p1 + size_t(p) * d, proj.data() + (size_t(p) * n + t) * d, size_t(d) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(g1, g.data() + size_t(t) * d, size_t(d) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(b1, beta.data() + size_t(t) * heads, size_t(heads) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(g21, g2.data() + size_t(t) * d, size_t(d) * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::glm_kda_conv(p1, h1, dw, cv1, d, K, nullptr);
        tqllm_cuda::glm_kda_gate(g1, ddtb, dA, -5.f, heads, D, nullptr);
        tqllm_cuda::glm_sigmoid(b1, heads, nullptr);
        tqllm_cuda::glm_kda_recur(cv1, cv1 + d, cv1 + 2 * d, g1, b1, S1, nullptr, o1, heads, D, nullptr);
        tqllm_cuda::glm_kda_out(o1, g21, don, y1, nullptr, heads, D, 1e-5f, nullptr);
        const auto c = host(cv1, size_t(3) * d);
        for (int p = 0; p < 3; ++p) std::copy(c.begin() + p * d, c.begin() + (p + 1) * d, cv_seq.begin() + (size_t(p) * n + t) * d);
        const auto o = host(o1, size_t(d)), y = host(y1, size_t(d)), gg = host(g1, size_t(d));
        std::copy(o.begin(), o.end(), o_seq.begin() + size_t(t) * d);
        std::copy(y.begin(), y.end(), y_seq.begin() + size_t(t) * d);
        std::copy(gg.begin(), gg.end(), g_seq.begin() + size_t(t) * d);
    }
    compare("chunk == steps: kda conv", host(cvc, cv_seq.size()), cv_seq, 0.f);
    compare("chunk == steps: kda gate", host(gc, g_seq.size()), g_seq, 0.f);
    compare("chunk == steps: kda recurrence", host(oc, o_seq.size()), o_seq, 0.f);
    compare("chunk == steps: kda state", host(Sc, size_t(heads) * D * D), host(S1, size_t(heads) * D * D), 0.f);
    compare("chunk == steps: kda gated norm", host(yc, y_seq.size()), y_seq, 0.f);
    // MLA: a chunk of n tokens whose latents sit at positions p0 .. p0 + n - 1
    const int R = 512, dk = 256, dv = 256, p0 = 6;
    auto q = rnd(size_t(n) * heads * dk), lat = rnd(size_t(p0 + n) * R);
    auto kbf = rnd(size_t(heads) * R * dk, -0.1f, 0.1f), vbf = rnd(size_t(heads) * dv * R, -0.1f, 0.1f);
    std::vector<__half> kb(kbf.size()), vb(vbf.size());
    for (size_t i = 0; i < kb.size(); ++i) kb[i] = __float2half(kbf[i]);
    for (size_t i = 0; i < vb.size(); ++i) vb[i] = __float2half(vbf[i]);
    __half * dkb = dev(kb), * dvb = dev(vb);
    float * dq = dev(q), * dlat = dev(lat), * qa = dev0<float>(size_t(n) * heads * R), * kv = dev0<float>(size_t(n) * heads * R),
          * out = dev0<float>(size_t(n) * heads * dv);
    tqllm_cuda::glm_mla_qabs(dq, dkb, qa, heads, R, dk, nullptr, n);
    tqllm_cuda::glm_mla_attn(qa, dlat, kv, heads, R, p0 + 1, 1.f / 16.f, nullptr, n);
    tqllm_cuda::glm_mla_vb(kv, dvb, out, nullptr, heads, R, dv, nullptr, n);
    const auto qa_c = host(qa, size_t(n) * heads * R), out_c = host(out, size_t(n) * heads * dv);
    std::vector<float> qa_s, out_s;
    float * qa1 = dev0<float>(size_t(heads) * R), * kv1 = dev0<float>(size_t(heads) * R), * out1 = dev0<float>(size_t(heads) * dv);
    for (int t = 0; t < n; ++t) {
        tqllm_cuda::glm_mla_qabs(dq + size_t(t) * heads * dk, dkb, qa1, heads, R, dk, nullptr);
        tqllm_cuda::glm_mla_attn(qa1, dlat, kv1, heads, R, p0 + 1 + t, 1.f / 16.f, nullptr);
        tqllm_cuda::glm_mla_vb(kv1, dvb, out1, nullptr, heads, R, dv, nullptr);
        const auto a = host(qa1, size_t(heads) * R), b = host(out1, size_t(heads) * dv);
        qa_s.insert(qa_s.end(), a.begin(), a.end()); out_s.insert(out_s.end(), b.begin(), b.end());
    }
    compare("chunk == steps: mla absorbed query", qa_c, qa_s, 0.f);
    compare("chunk == steps: mla causal attention + value", out_c, out_s, 0.f);
    // FP32 matvec over n columns
    const int rows = 29, cols = 300;
    auto W = rnd(size_t(rows) * cols), x = rnd(size_t(n) * cols);
    float * dW = dev(W), * dx = dev(x), * y = dev0<float>(size_t(n) * rows), * y1v = dev0<float>(size_t(rows));
    tqllm_cuda::glm_mv_f32(dW, dx, y, rows, cols, n, nullptr);
    std::vector<float> ys;
    for (int t = 0; t < n; ++t) {
        tqllm_cuda::glm_mv_f32(dW, dx + size_t(t) * cols, y1v, rows, cols, 1, nullptr);
        const auto r = host(y1v, size_t(rows));
        ys.insert(ys.end(), r.begin(), r.end());
    }
    compare("chunk == steps: mv_f32", host(y, size_t(n) * rows), ys, 0.f);
}
// KDA rollback for speculative verify: a block of n tokens, then commit(m) replaying the first m tokens, must give the
// state, the conv history and the outputs of m single-token steps.
void test_kda_rollback() {
    const int heads = 4, D = 128, d = heads * D, K = 4, n = 5;
    auto w = rnd(size_t(3) * d * K), dtb = rnd(size_t(d)), A = rnd(size_t(heads), -2.f, -0.1f);
    // the engine's layout: token-major [n][3 d] (ts = 3 d, ps = d), which is what the engine keeps for the replay
    auto proj = rnd(size_t(n) * 3 * d, -2, 2), g = rnd(size_t(n) * d, -3, 3), beta = rnd(size_t(n) * heads, -2, 2);
    std::vector<float> hist0 = std::vector<float>(size_t(3) * (K - 1) * d, 0.f);
    for (auto & x : hist0) x = 0.f;  // a fresh sequence: the state and history start at zero
    const size_t sn = size_t(heads) * D * D;
    float * dw = dev(w), * ddtb = dev(dtb), * dA = dev(A);
    // reference: one token at a time, state and history after every token
    float * h1 = dev0<float>(size_t(3) * (K - 1) * d), * S1 = dev0<float>(sn), * p1 = dev0<float>(size_t(3) * d),
          * cv1 = dev0<float>(size_t(3) * d), * g1 = dev0<float>(size_t(d)), * b1 = dev0<float>(size_t(heads)),
          * o1 = dev0<float>(size_t(d));
    std::vector<std::vector<float>> ref_state, ref_hist;
    std::vector<float> ref_out;
    for (int t = 0; t < n; ++t) {
        cudaMemcpy(p1, proj.data() + size_t(t) * 3 * d, size_t(3) * d * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(g1, g.data() + size_t(t) * d, size_t(d) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(b1, beta.data() + size_t(t) * heads, size_t(heads) * 4, cudaMemcpyHostToDevice);
        tqllm_cuda::glm_kda_conv(p1, h1, dw, cv1, d, K, nullptr);
        tqllm_cuda::glm_kda_gate(g1, ddtb, dA, -5.f, heads, D, nullptr);
        tqllm_cuda::glm_sigmoid(b1, heads, nullptr);
        tqllm_cuda::glm_kda_recur(cv1, cv1 + d, cv1 + 2 * d, g1, b1, S1, nullptr, o1, heads, D, nullptr);
        ref_hist.push_back(host(h1, size_t(3) * (K - 1) * d));
        ref_state.push_back(host(S1, sn));
        const auto o = host(o1, size_t(d));
        ref_out.insert(ref_out.end(), o.begin(), o.end());  // [t+1][d] concatenated: the first (m*d) floats are token m's
    }
    // the block in one pass: the conv runs on the live history, and with S_out the recurrence leaves the buffer it read
    // (the pre-block state) untouched
    float * hc = dev(hist0), * Sc = dev0<float>(sn), * S2 = dev0<float>(sn), * pc = dev(proj),
          * cvc = dev0<float>(size_t(n) * 3 * d), * gc = dev(g), * bc = dev(beta), * oc = dev0<float>(size_t(n) * d);
    tqllm_cuda::glm_kda_conv(pc, hc, dw, cvc, d, K, nullptr, n, 3 * d, d);
    tqllm_cuda::glm_kda_gate(gc, ddtb, dA, -5.f, heads, D, nullptr, n);
    tqllm_cuda::glm_sigmoid(bc, n * heads, nullptr);
    tqllm_cuda::glm_kda_recur(cvc, cvc + d, cvc + 2 * d, gc, bc, Sc, S2, oc, heads, D, nullptr, n, 3 * d);
    compare("rollback: the pre-block state is untouched", host(Sc, sn), std::vector<float>(sn, 0.f), 0.f);
    compare("rollback: the block's post-block state", host(S2, sn), ref_state[size_t(n - 1)], 0.f);
    compare("rollback: the block's outputs", host(oc, size_t(n) * d), ref_out, 0.f);
    // commit(m) for every m < n: replay the recurrence from the pre-block state on the saved conv output, and rebuild
    // the history from the pre-block history plus the saved raw projections
    float * hrb = dev0<float>(size_t(3) * (K - 1) * d), * Srb = dev0<float>(sn), * orb = dev0<float>(size_t(n) * d);
    for (int m = 1; m < n; ++m) {
        tqllm_cuda::glm_kda_recur(cvc, cvc + d, cvc + 2 * d, gc, bc, Sc, Srb, orb, heads, D, nullptr, m, 3 * d);
        tqllm_cuda::glm_kda_hist_rollback(dev(hist0), pc, hrb, d, K, m, nullptr, 3 * d, d);
        char what[80];
        std::snprintf(what, sizeof what, "rollback to %d of %d: state", m, n);
        compare(what, host(Srb, sn), ref_state[size_t(m - 1)], 0.f);
        std::snprintf(what, sizeof what, "rollback to %d of %d: conv history", m, n);
        compare(what, host(hrb, size_t(3) * (K - 1) * d), ref_hist[size_t(m - 1)], 0.f);
        std::snprintf(what, sizeof what, "rollback to %d of %d: outputs", m, n);
        const auto got = host(orb, size_t(m) * d);
        compare(what, std::vector<float>(got.begin(), got.begin() + size_t(m) * d),
                std::vector<float>(ref_out.begin(), ref_out.begin() + size_t(m) * d), 0.f);
    }
    // a non-zero starting history exercises the "row < K-1 comes from the old history" side of the rebuild
    std::vector<float> hz(size_t(3) * (K - 1) * d);
    for (size_t i = 0; i < hz.size(); ++i) hz[i] = float(int(i % 17) - 8) * 0.25f;
    float * hz0 = dev(hz), * hz1 = dev(hz), * hz2 = dev0<float>(size_t(3) * (K - 1) * d), * cvz = dev0<float>(size_t(n) * 3 * d);
    tqllm_cuda::glm_kda_conv(pc, hz1, dw, cvz, d, K, nullptr, n, 3 * d, d);  // hz1 now holds the post-block history
    tqllm_cuda::glm_kda_hist_rollback(hz0, pc, hz2, d, K, 1, nullptr, 3 * d, d);
    std::vector<float> want(size_t(3) * (K - 1) * d);
    for (int p = 0; p < 3; ++p)
        for (int j = 0; j < K - 1; ++j)
            for (int ch = 0; ch < d; ++ch) {
                const int row = 1 + j;
                want[size_t(p) * (K - 1) * d + size_t(j) * d + ch] =
                    row < K - 1 ? hz[size_t(p) * (K - 1) * d + size_t(row) * d + ch] : proj[size_t(row - (K - 1)) * 3 * d + size_t(p) * d + ch];
            }
    compare("rollback: history from a non-zero start", host(hz2, want.size()), want, 0.f);
    // the engine's shape: a block longer than the chunk size runs chunk by chunk, with ONE snapshot before the first
    // chunk and every chunk saving its rows below the previous chunk's. The state updates in place per chunk, so the
    // snapshot - not a swap - is what commit(m) replays from, and it still has to give m single steps.
    float * hlive = dev0<float>(size_t(3) * (K - 1) * d), * Slive = dev0<float>(sn),
          * hsnap = dev0<float>(size_t(3) * (K - 1) * d), * Ssnap = dev0<float>(sn),
          * graw = dev(g), * braw = dev(beta), * skv = dev0<float>(size_t(n) * 3 * d), * scv = dev0<float>(size_t(n) * 3 * d),
          * sg = dev0<float>(size_t(n) * d), * sb = dev0<float>(size_t(n) * heads), * scvx = dev0<float>(size_t(n) * 3 * d),
          * soc = dev0<float>(size_t(n) * d);
    cudaMemcpy(hsnap, hlive, size_t(3) * (K - 1) * d * 4, cudaMemcpyDeviceToDevice);  // begin_block(): snapshot, then chunks
    cudaMemcpy(Ssnap, Slive, sn * 4, cudaMemcpyDeviceToDevice);
    for (int off = 0; off < n;) {
        const int ci = std::min(n - off, off == 0 ? 3 : 2);  // chunks of 3 and 2, like a verify longer than the chunk size
        const float * pcx = pc + size_t(off) * 3 * d;
        float * cvx = scvx + size_t(off) * 3 * d;
        tqllm_cuda::glm_kda_conv(pcx, hlive, dw, cvx, d, K, nullptr, ci, 3 * d, d);
        tqllm_cuda::glm_kda_gate(graw + size_t(off) * d, ddtb, dA, -5.f, heads, D, nullptr, ci);
        tqllm_cuda::glm_sigmoid(braw + size_t(off) * heads, ci * heads, nullptr);
        cudaMemcpy(skv + size_t(off) * 3 * d, pcx, size_t(ci) * 3 * d * 4, cudaMemcpyDeviceToDevice);  // save_kda()
        cudaMemcpy(scv + size_t(off) * 3 * d, cvx, size_t(ci) * 3 * d * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sg + size_t(off) * d, graw + size_t(off) * d, size_t(ci) * d * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(sb + size_t(off) * heads, braw + size_t(off) * heads, size_t(ci) * heads * 4, cudaMemcpyDeviceToDevice);
        tqllm_cuda::glm_kda_recur(cvx, cvx + d, cvx + 2 * d, sg + size_t(off) * d, sb + size_t(off) * heads, Slive, nullptr,
                                  soc + size_t(off) * d, heads, D, nullptr, ci, 3 * d);
        off += ci;
    }
    compare("chunked block: state after the whole block", host(Slive, sn), ref_state[size_t(n - 1)], 0.f);
    compare("chunked block: history after the whole block", host(hlive, size_t(3) * (K - 1) * d), ref_hist[size_t(n - 1)], 0.f);
    float * hrb2 = dev0<float>(size_t(3) * (K - 1) * d), * Srb2 = dev0<float>(sn), * orb2 = dev0<float>(size_t(n) * d);
    for (int m = 1; m < n; ++m) {
        tqllm_cuda::glm_kda_recur(scv, scv + d, scv + 2 * d, sg, sb, Ssnap, Srb2, orb2, heads, D, nullptr, m, 3 * d);
        tqllm_cuda::glm_kda_hist_rollback(hsnap, skv, hrb2, d, K, m, nullptr, 3 * d, d);
        char what[80];
        std::snprintf(what, sizeof what, "chunked rollback to %d of %d: state", m, n);
        compare(what, host(Srb2, sn), ref_state[size_t(m - 1)], 0.f);
        std::snprintf(what, sizeof what, "chunked rollback to %d of %d: conv history", m, n);
        compare(what, host(hrb2, size_t(3) * (K - 1) * d), ref_hist[size_t(m - 1)], 0.f);
        std::snprintf(what, sizeof what, "chunked rollback to %d of %d: outputs", m, n);
        const auto got = host(orb2, size_t(m) * d);
        compare(what, std::vector<float>(got.begin(), got.begin() + size_t(m) * d),
                std::vector<float>(ref_out.begin(), ref_out.begin() + size_t(m) * d), 0.f);
    }
}

}  // namespace

int main() {
    test_hc();
    test_rms_mv();
    test_kda();
    test_mla();
    test_route();
    test_ffn_bits();
    test_chunks();
    test_kda_rollback();
    if (cudaGetLastError() != cudaSuccess) { std::printf("FAIL: CUDA error\n"); ++failures; }
    std::printf("%s\n", failures ? "glm5next kernels: FAILED" : "glm5next kernels: PASS");
    return failures ? 1 : 0;
}

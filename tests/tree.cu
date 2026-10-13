// Tree verification primitives: every node of a random tree must give exactly (bit for bit) what its root-to-node
// path gives as a plain chain, for the Gated DeltaNet recurrence, the causal conv, RoPE and two-part attention (BF16 and
// the compressed KV codecs).
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <random>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;
int failures = 0;
std::mt19937 rng(7);

template <class T> T * dev(const std::vector<T> & h) {
    T * p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T * p, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
std::vector<bf16> randb(size_t n, float s = 1.f) {
    std::normal_distribution<float> nd(0.f, s);
    std::vector<bf16> v(n);
    for (auto & x : v) x = __float2bfloat16_rn(nd(rng));
    return v;
}
std::vector<float> randf(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = u(rng);
    return v;
}
std::vector<int> random_tree(int n) {  // parent < child, node 0 is the root
    std::vector<int> par(n, -1);
    for (int i = 1; i < n; ++i) par[i] = std::uniform_int_distribution<int>(0, i - 1)(rng);
    return par;
}
std::vector<int> path_of(const std::vector<int> & par, int z) {  // root .. z
    std::vector<int> p;
    for (int a = z; a >= 0; a = par[a]) p.insert(p.begin(), a);
    return p;
}
std::vector<int> chain(int n) {
    std::vector<int> c(n);
    for (int i = 0; i < n; ++i) c[i] = i - 1;
    return c;
}
bool same(const bf16 * a, const bf16 * b, size_t n) { return std::memcmp(a, b, n * sizeof(bf16)) == 0; }
void report(const char * name, bool ok) {
    std::printf("%-28s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

void test_gdn() {
    const int n = 8, kh = 16, vh = 48, dk = 128, dv = 128, qkv = 2 * kh * dk + vh * dv, ab = 2 * vh;
    bool ok = true;
    for (int trial = 0; trial < 4; ++trial) {
        const auto par = random_tree(n);
        const auto x = randb(size_t(n) * qkv, 0.5f), ba = randb(size_t(n) * ab, 1.f);
        const auto neg_a = randf(vh, -1.f, -0.1f), dt = randf(vh, -0.5f, 0.5f), st = randf(size_t(vh) * dk * dv, -0.1f, 0.1f);
        auto *dx = dev(x), *dba = dev(ba);
        auto *dna = dev(neg_a), *ddt = dev(dt), *dst = dev(st);
        float * dso = nullptr;
        cudaMalloc(&dso, st.size() * 4);
        bf16 * dy = nullptr;
        cudaMalloc(&dy, size_t(n) * vh * dv * 2);
        auto * dpar = dev(par);
        tqllm_cuda::gdn_multi(dx, n, dba + vh, dba, ab, dna, ddt, dst, dso, dy, kh, vh, dk, dv, true, nullptr, dpar);
        const auto yt = host(dy, size_t(n) * vh * dv);
        for (int z = 0; z < n; ++z) {  // node z vs its path as a chain
            const auto p = path_of(par, z);
            const int m = int(p.size());
            std::vector<bf16> xc(size_t(m) * qkv), bc(size_t(m) * ab);
            for (int i = 0; i < m; ++i) {
                std::memcpy(&xc[size_t(i) * qkv], &x[size_t(p[i]) * qkv], size_t(qkv) * 2);
                std::memcpy(&bc[size_t(i) * ab], &ba[size_t(p[i]) * ab], size_t(ab) * 2);
            }
            auto *dxc = dev(xc), *dbc = dev(bc);
            auto * dch = dev(chain(m));
            tqllm_cuda::gdn_multi(dxc, m, dbc + vh, dbc, ab, dna, ddt, dst, dso, dy, kh, vh, dk, dv, true, nullptr, dch);
            const auto yc = host(dy, size_t(m) * vh * dv);
            ok &= same(&yt[size_t(z) * vh * dv], &yc[size_t(m - 1) * vh * dv], size_t(vh) * dv);
            cudaFree(dxc), cudaFree(dbc), cudaFree(dch);
        }
        cudaFree(dx), cudaFree(dba), cudaFree(dna), cudaFree(ddt), cudaFree(dst), cudaFree(dso), cudaFree(dy), cudaFree(dpar);
    }
    report("gdn tree == path chain", ok);
}

void test_conv() {
    const int n = 8, C = 10240;
    bool ok = true;
    for (int trial = 0; trial < 4; ++trial) {
        const auto par = random_tree(n);
        const auto x = randb(size_t(n) * C), st = randb(size_t(3) * C);
        const auto w = randf(size_t(C) * 4, -0.5f, 0.5f);
        auto *dx = dev(x), *dst = dev(st);
        auto * dw = dev(w);
        bf16 *dso = nullptr, *dy = nullptr;
        cudaMalloc(&dso, st.size() * 2);
        cudaMalloc(&dy, size_t(n) * C * 2);
        auto * dpar = dev(par);
        tqllm_cuda::conv_multi(dx, C, n, dst, dso, dw, dy, C, nullptr, dpar);
        const auto yt = host(dy, size_t(n) * C);
        for (int z = 0; z < n; ++z) {
            const auto p = path_of(par, z);
            const int m = int(p.size());
            std::vector<bf16> xc(size_t(m) * C);
            for (int i = 0; i < m; ++i) std::memcpy(&xc[size_t(i) * C], &x[size_t(p[i]) * C], size_t(C) * 2);
            auto * dxc = dev(xc);
            auto * dch = dev(chain(m));
            tqllm_cuda::conv_multi(dxc, C, m, dst, dso, dw, dy, C, nullptr, dch);
            const auto yc = host(dy, size_t(m) * C);
            ok &= same(&yt[size_t(z) * C], &yc[size_t(m - 1) * C], C);
            if (z == n - 1 && p.size() == size_t(n)) {  // chain state after the last node vs the default kernel's
                bf16 * dso2 = nullptr;
                cudaMalloc(&dso2, st.size() * 2);
                tqllm_cuda::conv_multi(dxc, C, m, dst, dso2, dw, dy, C, nullptr);
                const auto a = host(dso, st.size()), b = host(dso2, st.size());
                ok &= same(a.data(), b.data(), a.size());
                cudaFree(dso2);
            }
            cudaFree(dxc), cudaFree(dch);
        }
        // chain parent array == default kernel (outputs)
        auto * dch = dev(chain(n));
        tqllm_cuda::conv_multi(dx, C, n, dst, dso, dw, dy, C, nullptr, dch);
        const auto a = host(dy, size_t(n) * C);
        tqllm_cuda::conv_multi(dx, C, n, dst, dso, dw, dy, C, nullptr);
        const auto b = host(dy, size_t(n) * C);
        ok &= same(a.data(), b.data(), a.size());
        cudaFree(dx), cudaFree(dst), cudaFree(dw), cudaFree(dso), cudaFree(dy), cudaFree(dpar), cudaFree(dch);
    }
    report("conv tree == path chain", ok);
}

void test_conv_long() {  // long blocks (token-parallel kernel) == the same tokens in short chained blocks (serial kernel)
    const int n = 300, C = 10240, step = 50;
    const auto x = randb(size_t(n) * C), st = randb(size_t(3) * C);
    const auto w = randf(size_t(C) * 4, -0.5f, 0.5f);
    auto *dx = dev(x), *dst = dev(st);
    auto * dw = dev(w);
    bf16 *dso = nullptr, *dy = nullptr, *dsa = nullptr, *dsb = nullptr, *dy2 = nullptr;
    cudaMalloc(&dso, st.size() * 2), cudaMalloc(&dsa, st.size() * 2), cudaMalloc(&dsb, st.size() * 2);
    cudaMalloc(&dy, size_t(n) * C * 2), cudaMalloc(&dy2, size_t(n) * C * 2);
    tqllm_cuda::conv_multi(dx, C, n, dst, dso, dw, dy, C, nullptr);
    cudaMemcpy(dsa, dst, st.size() * 2, cudaMemcpyDeviceToDevice);
    for (int i0 = 0; i0 < n; i0 += step) {
        tqllm_cuda::conv_multi(dx + size_t(i0) * C, C, std::min(step, n - i0), dsa, dsb, dw, dy2 + size_t(i0) * C, C, nullptr);
        std::swap(dsa, dsb);
    }
    const auto a = host(dy, size_t(n) * C), b = host(dy2, size_t(n) * C);
    const auto sa = host(dso, st.size()), sb = host(dsa, st.size());
    report("conv long block == short chained blocks", same(a.data(), b.data(), a.size()) && same(sa.data(), sb.data(), sa.size()));
    cudaFree(dx), cudaFree(dst), cudaFree(dw), cudaFree(dso), cudaFree(dsa), cudaFree(dsb), cudaFree(dy), cudaFree(dy2);
}

void test_replay() {  // deferred commit: replay + tree in one kernel == replay (chain) then tree, outputs and mid state
    const int n = 8, k = 5, kh = 16, vh = 48, dk = 128, dv = 128, qkv = 2 * kh * dk + vh * dv, ab = 2 * vh, C = 10240;
    bool ok = true;
    for (int trial = 0; trial < 3; ++trial) {
        const auto par = random_tree(n);
        // GDN
        const auto x = randb(size_t(n) * qkv, 0.5f), ba = randb(size_t(n) * ab), xr = randb(size_t(k) * qkv, 0.5f),
                   bar = randb(size_t(k) * ab);
        const auto neg_a = randf(vh, -1.f, -0.1f), dt = randf(vh, -0.5f, 0.5f), st = randf(size_t(vh) * dk * dv, -0.1f, 0.1f);
        auto *dx = dev(x), *dba = dev(ba), *dxr = dev(xr), *dbar = dev(bar);
        auto *dna = dev(neg_a), *ddt = dev(dt), *dst = dev(st);
        float *dmid = nullptr, *dmid2 = nullptr, *dfin = nullptr;
        cudaMalloc(&dmid, st.size() * 4), cudaMalloc(&dmid2, st.size() * 4), cudaMalloc(&dfin, st.size() * 4);
        bf16 *dy1 = nullptr, *dy2 = nullptr;
        cudaMalloc(&dy1, size_t(n) * vh * dv * 2), cudaMalloc(&dy2, size_t(n) * vh * dv * 2);
        auto *dpar = dev(par), *dch = dev(chain(k));
        const std::vector<int> pk{k};
        auto * dpk = dev(pk);
        tqllm_cuda::gdn_multi(dxr, k, dbar + vh, dbar, ab, dna, ddt, dst, dmid, dy2, kh, vh, dk, dv, true, nullptr, dch);
        tqllm_cuda::gdn_multi(dx, n, dba + vh, dba, ab, dna, ddt, dmid, dfin, dy1, kh, vh, dk, dv, true, nullptr, dpar);
        tqllm_cuda::GdnReplay rp;
        rp.qkv = dxr, rp.a0 = dbar + vh, rp.b0 = dbar, rp.abs_ = ab, rp.pend = dpk, rp.smid = dmid2;
        tqllm_cuda::gdn_multi(dx, n, dba + vh, dba, ab, dna, ddt, dst, dfin, dy2, kh, vh, dk, dv, true, nullptr, dpar, rp);
        const auto a1 = host(dy1, size_t(n) * vh * dv), a2 = host(dy2, size_t(n) * vh * dv);
        const auto m1 = host(dmid, st.size()), m2 = host(dmid2, st.size());
        ok &= same(a1.data(), a2.data(), a1.size()) && std::memcmp(m1.data(), m2.data(), m1.size() * 4) == 0;
        // conv
        const auto cx = randb(size_t(n) * C), cxr = randb(size_t(k) * C), cst = randb(size_t(3) * C);
        const auto cw = randf(size_t(C) * 4, -0.5f, 0.5f);
        auto *dcx = dev(cx), *dcxr = dev(cxr), *dcst = dev(cst);
        auto * dcw = dev(cw);
        bf16 *cmid = nullptr, *cmid2 = nullptr, *cfin = nullptr, *cy1 = nullptr, *cy2 = nullptr, *cry = nullptr, *cry2 = nullptr;
        cudaMalloc(&cmid, cst.size() * 2), cudaMalloc(&cmid2, cst.size() * 2), cudaMalloc(&cfin, cst.size() * 2);
        cudaMalloc(&cy1, size_t(n) * C * 2), cudaMalloc(&cy2, size_t(n) * C * 2);
        cudaMalloc(&cry, size_t(k) * C * 2), cudaMalloc(&cry2, size_t(k) * C * 2);
        tqllm_cuda::conv_multi(dcxr, C, k, dcst, cmid, dcw, cry, C, nullptr, dch);
        tqllm_cuda::conv_multi(dcx, C, n, cmid, cfin, dcw, cy1, C, nullptr, dpar);
        tqllm_cuda::ConvReplay cr;
        cr.x = dcxr, cr.pend = dpk, cr.smid = cmid2, cr.y = cry2;
        tqllm_cuda::conv_multi(dcx, C, n, dcst, cfin, dcw, cy2, C, nullptr, dpar, cr);
        const auto b1 = host(cy1, size_t(n) * C), b2 = host(cy2, size_t(n) * C);
        const auto r1 = host(cry, size_t(k) * C), r2 = host(cry2, size_t(k) * C);
        const auto c1 = host(cmid, cst.size()), c2 = host(cmid2, cst.size());
        ok &= same(b1.data(), b2.data(), b1.size()) && same(r1.data(), r2.data(), r1.size()) && same(c1.data(), c2.data(), c1.size());
        for (void * p2 : std::initializer_list<void *>{dx, dba, dxr, dbar, dna, ddt, dst, dmid, dmid2, dfin, dy1, dy2, dpar, dch, dpk,
                                                       dcx, dcxr, dcst, dcw, cmid, cmid2, cfin, cy1, cy2, cry, cry2})
            cudaFree(p2);
    }
    report("deferred replay + tree == replay then tree", ok);
}

void test_rope() {
    const int n = 8, heads = 24, dim = 256, rot = 64, ts = heads * dim;
    const auto par = random_tree(n);
    std::vector<int> dep(n, 0);
    for (int i = 1; i < n; ++i) dep[i] = dep[par[i]] + 1;
    const auto x = randb(size_t(n) * ts);
    auto * dx = dev(x);
    auto * ddep = dev(dep);
    const std::vector<int> pos{300, 301};
    auto * dpos = dev(pos);
    tqllm_cuda::rope_partial_multi(dx, ts, n, heads, dim, rot, dpos, 1e7f, nullptr, ddep);
    const auto yt = host(dx, x.size());
    bool ok = true;
    for (int z = 0; z < n; ++z) {  // node z rotated by its depth == a chain token at that depth
        std::vector<bf16> xc(size_t(dep[z] + 1) * ts);
        std::memcpy(&xc[size_t(dep[z]) * ts], &x[size_t(z) * ts], size_t(ts) * 2);
        auto * dxc = dev(xc);
        tqllm_cuda::rope_partial_multi(dxc, ts, dep[z] + 1, heads, dim, rot, dpos, 1e7f, nullptr);
        const auto yc = host(dxc, xc.size());
        ok &= same(&yt[size_t(z) * ts], &yc[size_t(dep[z]) * ts], ts);
        cudaFree(dxc);
    }
    cudaFree(dx), cudaFree(ddep), cudaFree(dpos);
    report("rope depth == chain position", ok);
}

void test_attention(int n, int pos) {
    const int heads = 24, kvh = 4, dim = 256, ctx = 512, nsplit = 20;
    const size_t kvw = size_t(kvh) * dim;
    bool ok = true;
    float maxerr = 0.f;
    for (int trial = 0; trial < 4; ++trial) {
        const auto par = random_tree(n);
        const auto kc = randb(size_t(ctx) * kvw), vc = randb(size_t(ctx) * kvw), q = randb(size_t(n) * heads * dim);
        auto *dk = dev(kc), *dv = dev(vc), *dq = dev(q);
        bf16 * dy = nullptr;
        float * part = nullptr;
        cudaMalloc(&dy, size_t(n) * heads * dim * 2);
        cudaMalloc(&part, size_t(n) * heads * (nsplit + 1) * (dim + 2) * 4);
        const std::vector<int> dpos{pos, pos + 1};
        auto * dp = dev(dpos);
        auto * dpar = dev(par);
        const float scale = 1.f / 16.f;
        std::vector<int> dep(n, 0);
        for (int i = 1; i < n; ++i) dep[i] = dep[par[i]] + 1;
        auto * ddep = dev(dep);
        tqllm_cuda::attention_mq256(dq, dk, dv, dy, part, dp + 1, n, heads, kvh, scale, nsplit, nullptr, 0, dpar, ddep);
        const auto yt = host(dy, size_t(n) * heads * dim);
        for (int z = 0; z < n; ++z) {  // node z vs its path with the first c tokens already committed to the cache
            const auto p = path_of(par, z);
            const int m = int(p.size());
            for (int c : {m - 1, m / 2}) {
                auto kc2 = kc, vc2 = vc;
                for (int i = 0; i < m; ++i) {
                    std::memcpy(&kc2[size_t(pos + i) * kvw], &kc[size_t(pos + p[i]) * kvw], kvw * 2);
                    std::memcpy(&vc2[size_t(pos + i) * kvw], &vc[size_t(pos + p[i]) * kvw], kvw * 2);
                }
                const int mb = m - c;  // chain block of the remaining path at slots pos + c ..
                std::vector<bf16> q2(size_t(mb) * heads * dim);
                for (int i = 0; i < mb; ++i)
                    std::memcpy(&q2[size_t(i) * heads * dim], &q[size_t(p[c + i]) * heads * dim], size_t(heads) * dim * 2);
                auto *dk2 = dev(kc2), *dv2 = dev(vc2), *dq2 = dev(q2);
                auto * dch = dev(chain(mb));
                const std::vector<int> dpos2{pos + c, pos + c + 1};
                auto * dp2 = dev(dpos2);
                tqllm_cuda::attention_mq256(dq2, dk2, dv2, dy, part, dp2 + 1, mb, heads, kvh, scale, nsplit, nullptr, 0, dch);
                const auto yc = host(dy, size_t(mb) * heads * dim);
                ok &= same(&yt[size_t(z) * heads * dim], &yc[size_t(mb - 1) * heads * dim], size_t(heads) * dim);
                cudaFree(dk2), cudaFree(dv2), cudaFree(dq2), cudaFree(dch), cudaFree(dp2);
            }
        }
        cudaFree(ddep);
        for (int z = 0; z < n; ++z) {  // node z vs its path laid out as a chain at slots pos ..
            const auto p = path_of(par, z);
            const int m = int(p.size());
            auto kc2 = kc, vc2 = vc;
            std::vector<bf16> q2(size_t(m) * heads * dim);
            for (int i = 0; i < m; ++i) {
                std::memcpy(&kc2[size_t(pos + i) * kvw], &kc[size_t(pos + p[i]) * kvw], kvw * 2);
                std::memcpy(&vc2[size_t(pos + i) * kvw], &vc[size_t(pos + p[i]) * kvw], kvw * 2);
                std::memcpy(&q2[size_t(i) * heads * dim], &q[size_t(p[i]) * heads * dim], size_t(heads) * dim * 2);
            }
            auto *dk2 = dev(kc2), *dv2 = dev(vc2), *dq2 = dev(q2);
            auto * dch = dev(chain(m));
            tqllm_cuda::attention_mq256(dq2, dk2, dv2, dy, part, dp + 1, m, heads, kvh, scale, nsplit, nullptr, 0, dch);
            const auto yc = host(dy, size_t(m) * heads * dim);
            ok &= same(&yt[size_t(z) * heads * dim], &yc[size_t(m - 1) * heads * dim], size_t(heads) * dim);
            // FP64 reference for node z (keys 0 .. pos - 1, then the path)
            for (int h = 0; h < heads; h += 7) {
                const int g = h / (heads / kvh);
                std::vector<int> slots;
                for (int s = 0; s < pos; ++s) slots.push_back(s);
                for (int i = 0; i < m; ++i) slots.push_back(pos + p[i]);
                double mx = -1e300;
                std::vector<double> sc(slots.size());
                for (size_t t = 0; t < slots.size(); ++t) {
                    double a = 0;
                    for (int d = 0; d < dim; ++d)
                        a += double(__bfloat162float(q[(size_t(z) * heads + h) * dim + d])) *
                             __bfloat162float(kc[size_t(slots[t]) * kvw + size_t(g) * dim + d]);
                    sc[t] = a * scale;
                    mx = std::max(mx, sc[t]);
                }
                double den = 0;
                for (auto & v : sc) den += (v = std::exp(v - mx));
                for (int d = 0; d < dim; d += 37) {
                    double o = 0;
                    for (size_t t = 0; t < slots.size(); ++t) o += sc[t] * __bfloat162float(vc[size_t(slots[t]) * kvw + size_t(g) * dim + d]);
                    o /= den;
                    maxerr = std::max(maxerr, float(std::fabs(o - __bfloat162float(yt[(size_t(z) * heads + h) * dim + d]))));
                }
            }
            cudaFree(dk2), cudaFree(dv2), cudaFree(dq2), cudaFree(dch);
        }
        cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(dy), cudaFree(part), cudaFree(dp), cudaFree(dpar);
    }
    report("attention tree == path, any block start", ok);
    std::printf("%-28s max_abs %.3e vs FP64\n", "attention tree accuracy", maxerr);
    if (!(maxerr < 2e-2f)) ++failures;
}


// The tree kernels on a plain chain (par = parent i - 1) must equal the causal kernels (par = null) bit for bit: in tree
// mode every verify goes through the tree path, plain decoding through the causal one, and greedy output must not change
// (.77 10-13: DFlash2 tree changed a 3.7K-token answer at token 368; --no-tree did not).
void test_chain_vs_causal(int n, int pos) {
    const int heads = 24, kvh = 4, dim = 256, ctx = 512, nsplit = 20;
    const size_t kvw = size_t(kvh) * dim;
    bool ok = true;
    for (int trial = 0; trial < 4; ++trial) {
        const auto kc = randb(size_t(ctx) * kvw), vc = randb(size_t(ctx) * kvw), q = randb(size_t(n) * heads * dim);
        auto *dk = dev(kc), *dv = dev(vc), *dq = dev(q);
        bf16 * dy = nullptr;
        float * part = nullptr;
        cudaMalloc(&dy, size_t(n) * heads * dim * 2);
        cudaMalloc(&part, size_t(n) * heads * (nsplit + 1) * (dim + 2) * 4);
        const std::vector<int> dpos{pos, pos + 1};
        auto * dp = dev(dpos);
        auto * dch = dev(chain(n));
        std::vector<int> depth(n);
        for (int i = 0; i < n; ++i) depth[i] = i;
        auto * ddep = dev(depth);
        tqllm_cuda::attention_mq256(dq, dk, dv, dy, part, dp + 1, n, heads, kvh, 1.f / 16.f, nsplit, nullptr, 0, dch, ddep);
        auto yt = host(dy, size_t(n) * heads * dim);
        if (std::getenv("TREE_CAUSAL_REF") && n <= 8) {  // control: causal block of n queries in one call
            tqllm_cuda::attention_mq256(dq, dk, dv, dy, part, dp + 1, n, heads, kvh, 1.f / 16.f, nsplit, nullptr);
            yt = host(dy, size_t(n) * heads * dim);
        }
        // plain decoding: one query per step, query z seeing keys [0, pos + 1 + z)
        for (int z = 0; z < n; ++z) {
            const std::vector<int> d1{pos + 1 + z};
            auto * dd = dev(d1);
            tqllm_cuda::attention_mq256(dq + size_t(z) * heads * dim, dk, dv, dy, part, dd, 1, heads, kvh, 1.f / 16.f, nsplit, nullptr);
            const auto y1 = host(dy, size_t(heads) * dim);
            if (!same(&yt[size_t(z) * heads * dim], y1.data(), y1.size())) {
                ok = false;
                if (trial == 0) std::printf("  query %d differs from one-query decode\n", z);
            }
            cudaFree(dd);
        }
        cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(dy), cudaFree(part), cudaFree(dp), cudaFree(dch), cudaFree(ddep);
    }
    char name[64];
    std::snprintf(name, sizeof name, "attention chain == causal (n %d)", n);
    report(name, ok);
}
void test_gdn_chain_vs_causal() {
    const int n = 12, kh = 16, vh = 48, dk = 128, dv = 128, qkv = 2 * kh * dk + vh * dv, ab = 2 * vh;
    const auto x = randb(size_t(n) * qkv, 0.5f), ba = randb(size_t(n) * ab, 1.f);
    const auto neg_a = randf(vh, -1.f, -0.1f), dt = randf(vh, -0.5f, 0.5f), st = randf(size_t(vh) * dk * dv, -0.1f, 0.1f);
    auto *dx = dev(x), *dba = dev(ba);
    auto *dna = dev(neg_a), *ddt = dev(dt), *dst = dev(st);
    float * dso = nullptr;
    cudaMalloc(&dso, st.size() * 4);
    bf16 * dy = nullptr;
    cudaMalloc(&dy, size_t(n) * vh * dv * 2);
    auto * dch = dev(chain(n));
    tqllm_cuda::gdn_multi(dx, n, dba + vh, dba, ab, dna, ddt, dst, dso, dy, kh, vh, dk, dv, true, nullptr, dch);
    const auto yt = host(dy, size_t(n) * vh * dv);
    tqllm_cuda::gdn_multi(dx, n, dba + vh, dba, ab, dna, ddt, dst, dso, dy, kh, vh, dk, dv, true, nullptr, nullptr);
    const auto yc = host(dy, size_t(n) * vh * dv);
    report("gdn chain == causal", same(yt.data(), yc.data(), yt.size()));
    cudaFree(dx), cudaFree(dba), cudaFree(dna), cudaFree(ddt), cudaFree(dst), cudaFree(dso), cudaFree(dy), cudaFree(dch);
}

using tqllm_cuda::KvCodec;
size_t codec_bytes(KvCodec c) {
    switch (c) {
        case KvCodec::Tq3: return sizeof(tqllm_cuda::Tq3KV256);
        case KvCodec::Tq2s: return sizeof(tqllm_cuda::Tq2V256);
        case KvCodec::Q8: return sizeof(tqllm_cuda::Q8KV256);
        case KvCodec::Q4: return sizeof(tqllm_cuda::Q4KV256);
        case KvCodec::Tcq2: return sizeof(tqllm_cuda::Tcq2V256);
        case KvCodec::Tcq3: return sizeof(tqllm_cuda::Tcq3K256);
    }
    return 0;
}
// Split-cache two-part attention: node z == its path verified as a chain (rows moved like the commit's gather_path),
// with 0 or c path tokens already committed; and close to the plain (non-tree) attention of that chain.
void test_attention_codec(const char * name, KvCodec kcod, KvCodec vcod, int n, int pos) {
    const int heads = 24, kvh = 4, dim = 256, ctx = 512, nsplit = 18;
    const size_t kr = kvh * codec_bytes(kcod), vr = kvh * codec_bytes(vcod), qw = size_t(heads) * dim;
    bool ok = true;
    float maxdiff = 0.f;
    for (int trial = 0; trial < 2; ++trial) {
        const auto par = random_tree(n);
        std::vector<int> dep(n, 0);
        for (int i = 1; i < n; ++i) dep[i] = dep[par[i]] + 1;
        const auto kv = randb(size_t(ctx) * 2 * kvh * dim), q = randb(size_t(n) * qw);
        auto *dkv = dev(kv), *dq = dev(q);
        auto * d0 = dev(std::vector<int>{0, 1});
        uint8_t *dk = nullptr, *dv = nullptr;
        cudaMalloc(&dk, ctx * kr), cudaMalloc(&dv, ctx * vr);
        tqllm_cuda::cache_write_kv_side(kcod, dkv, 2 * kvh * dim, ctx, dk, d0, kvh, dim, nullptr);
        tqllm_cuda::cache_write_kv_side(vcod, dkv + kvh * dim, 2 * kvh * dim, ctx, dv, d0, kvh, dim, nullptr);
        const auto kc = host(dk, ctx * kr), vc = host(dv, ctx * vr);
        bf16 * dy = nullptr;
        float * part = nullptr;
        cudaMalloc(&dy, size_t(n) * qw * 2);
        cudaMalloc(&part, size_t(n) * heads * (nsplit + 1) * (dim + 2) * 4);
        auto * dp = dev(std::vector<int>{pos, pos + 1});
        auto *dpar = dev(par), *ddep = dev(dep);
        const float scale = 1.f / 16.f;
        tqllm_cuda::attention_kv_split(kcod, vcod, dq, dk, dv, dy, part, dp + 1, n, heads, kvh, scale, nsplit, nullptr, dpar, ddep);
        const auto yt = host(dy, size_t(n) * qw);
        for (int z = 0; z < n; ++z) {
            const auto p = path_of(par, z);
            const int m = int(p.size());
            auto kc2 = kc, vc2 = vc;
            for (int i = 0; i < m; ++i) {
                std::memcpy(&kc2[size_t(pos + i) * kr], &kc[size_t(pos + p[i]) * kr], kr);
                std::memcpy(&vc2[size_t(pos + i) * vr], &vc[size_t(pos + p[i]) * vr], vr);
            }
            auto *dk2 = dev(kc2), *dv2 = dev(vc2);
            for (int c : {0, m / 2}) {
                const int mb = m - c;
                std::vector<bf16> q2(size_t(mb) * qw);
                for (int i = 0; i < mb; ++i) std::memcpy(&q2[size_t(i) * qw], &q[size_t(p[c + i]) * qw], qw * 2);
                auto * dq2 = dev(q2);
                auto * dch = dev(chain(mb));
                auto * dp2 = dev(std::vector<int>{pos + c, pos + c + 1});
                tqllm_cuda::attention_kv_split(kcod, vcod, dq2, dk2, dv2, dy, part, dp2 + 1, mb, heads, kvh, scale, nsplit,
                                               nullptr, dch);
                const auto yc = host(dy, size_t(mb) * qw);
                ok &= same(&yt[size_t(z) * qw], &yc[size_t(mb - 1) * qw], qw);
                if (c == 0) {  // plain attention of the same chain (int8 Q, different summation): close, not equal
                    tqllm_cuda::attention_kv_split(kcod, vcod, dq2, dk2, dv2, dy, part, dp2 + 1, mb, heads, kvh, scale,
                                                   nsplit, nullptr);
                    const auto yp = host(dy, size_t(mb) * qw);
                    for (size_t e = 0; e < qw; ++e)
                        maxdiff = std::max(maxdiff, std::fabs(__bfloat162float(yt[size_t(z) * qw + e]) -
                                                              __bfloat162float(yp[size_t(mb - 1) * qw + e])));
                }
                cudaFree(dq2), cudaFree(dch), cudaFree(dp2);
            }
            cudaFree(dk2), cudaFree(dv2);
        }
        cudaFree(dkv), cudaFree(dq), cudaFree(d0), cudaFree(dk), cudaFree(dv), cudaFree(dy), cudaFree(part), cudaFree(dp),
            cudaFree(dpar), cudaFree(ddep);
    }
    char label[64];
    std::snprintf(label, sizeof label, "attn %s n%d pos%d", name, n, pos);
    report(label, ok);
    std::printf("%-28s max_abs %.3e vs plain\n", label, maxdiff);
    if (!(maxdiff < 3e-2f)) ++failures;
}

}  // namespace

// Codec attention (attention_kv_split): a chain through the tree path == one-query causal decode, bit for bit
void test_codec_chain_vs_causal(const char * name, KvCodec kcod, KvCodec vcod, int n, int pos) {
    const int heads = 24, kvh = 4, dim = 256, ctx = 512, nsplit = 18;
    const size_t kr = kvh * codec_bytes(kcod), vr = kvh * codec_bytes(vcod), qw = size_t(heads) * dim;
    const auto kv = randb(size_t(ctx) * 2 * kvh * dim), q = randb(size_t(n) * qw);
    auto *dkv = dev(kv), *dq = dev(q);
    auto * d0 = dev(std::vector<int>{0, 1});
    uint8_t *dk = nullptr, *dv = nullptr;
    cudaMalloc(&dk, ctx * kr), cudaMalloc(&dv, ctx * vr);
    tqllm_cuda::cache_write_kv_side(kcod, dkv, 2 * kvh * dim, ctx, dk, d0, kvh, dim, nullptr);
    tqllm_cuda::cache_write_kv_side(vcod, dkv + kvh * dim, 2 * kvh * dim, ctx, dv, d0, kvh, dim, nullptr);
    bf16 * dy = nullptr;
    float * part = nullptr;
    cudaMalloc(&dy, size_t(n) * qw * 2);
    cudaMalloc(&part, size_t(n) * heads * (nsplit + 1) * (dim + 2) * 4);
    auto * dp = dev(std::vector<int>{pos, pos + 1});
    std::vector<int> depth(n);
    for (int i = 0; i < n; ++i) depth[i] = i;
    auto *dch = dev(chain(n)), *ddep = dev(depth);
    tqllm_cuda::attention_kv_split(kcod, vcod, dq, dk, dv, dy, part, dp + 1, n, heads, kvh, 1.f / 16.f, nsplit, nullptr, dch, ddep);
    const auto yt = host(dy, size_t(n) * qw);
    bool ok = true;
    for (int z = 0; z < n; ++z) {
        auto * dd = dev(std::vector<int>{pos + 1 + z});
        tqllm_cuda::attention_kv_split(kcod, vcod, dq + size_t(z) * qw, dk, dv, dy, part, dd, 1, heads, kvh, 1.f / 16.f, nsplit, nullptr);
        ok &= same(&yt[size_t(z) * qw], host(dy, qw).data(), qw);
        cudaFree(dd);
    }
    char label[64];
    std::snprintf(label, sizeof label, "attn %s chain == causal n%d", name, n);
    report(label, ok);
    cudaFree(dkv), cudaFree(dq), cudaFree(d0), cudaFree(dk), cudaFree(dv), cudaFree(dy), cudaFree(part), cudaFree(dp), cudaFree(dch), cudaFree(ddep);
}

int main() {
    test_gdn();
    test_conv();
    test_conv_long();
    test_replay();
    test_rope();
    test_attention(8, 333);
    test_attention(12, 333);  // > 8 queries: tiled part in chunks
    test_attention(12, 350);  // deep nodes with the block start near a 32-key boundary
    test_chain_vs_causal(8, 333);
    test_chain_vs_causal(12, 333);
    test_chain_vs_causal(12, 350);
    test_gdn_chain_vs_causal();
    const struct { const char * name; KvCodec k, v; } codecs[] = {
        {"tq3/tq3", KvCodec::Tq3, KvCodec::Tq3}, {"tq3/tq2s", KvCodec::Tq3, KvCodec::Tq2s},
        {"tcq3/tcq2", KvCodec::Tcq3, KvCodec::Tcq2}, {"q8/q8", KvCodec::Q8, KvCodec::Q8}, {"q4/tcq2", KvCodec::Q4, KvCodec::Tcq2}};
    for (const auto & c : codecs) test_attention_codec(c.name, c.k, c.v, 8, 333);
    test_attention_codec("tq3/tq3", KvCodec::Tq3, KvCodec::Tq3, 12, 350);
    test_attention_codec("tcq3/tcq2", KvCodec::Tcq3, KvCodec::Tcq2, 12, 350);
    test_codec_chain_vs_causal("q4/tcq2", KvCodec::Q4, KvCodec::Tcq2, 8, 333);
    test_codec_chain_vs_causal("tq3/tq3", KvCodec::Tq3, KvCodec::Tq3, 8, 333);
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::printf("%s\n", failures ? "TREE TESTS FAILED" : "all tree tests passed");
    return failures ? 1 : 0;
}

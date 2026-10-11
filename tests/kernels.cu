// Per-primitive numerical tests for kernels/qwen2.cu against FP32 CPU references computed from the same inputs.
// Gates: FP32 outputs max abs err <= 1e-5 (relative for dot products); BF16 outputs max abs <= 2e-2 and
// rel-RMS <= 5e-3; argmax exact.
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;
int failures = 0;

std::vector<bf16> rand_bf16(size_t n, std::mt19937 & g, float scale = 1.f) {
    std::normal_distribution<float> d(0.f, scale);
    std::vector<bf16> v(n);
    for (auto & x : v) x = __float2bfloat16_rn(d(g));
    return v;
}

std::vector<float> f32(const std::vector<bf16> & v) {
    std::vector<float> r(v.size());
    for (size_t i = 0; i < v.size(); ++i) r[i] = __bfloat162float(v[i]);
    return r;
}

template <class T> T * dev(const std::vector<T> & h) {
    T * p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T));
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <class T> std::vector<T> host(const T * p, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

void check_bf16(const char * name, const std::vector<bf16> & got, const std::vector<float> & ref) {
    double max_abs = 0, se = 0, sr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = double(__bfloat162float(got[i])) - ref[i];
        max_abs = std::fmax(max_abs, std::fabs(e));
        se += e * e;
        sr += double(ref[i]) * ref[i];
    }
    const double rel = std::sqrt(se / (sr + 1e-30));
    const bool ok = max_abs <= 2e-2 && rel <= 5e-3;
    std::printf("%-12s bf16  max_abs %.3e  rel_rms %.3e  %s\n", name, max_abs, rel, ok ? "OK" : "FAIL");
    failures += !ok;
}

void check_f32(const char * name, const std::vector<float> & got, const std::vector<float> & ref, bool relative) {
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double e = std::fabs(double(got[i]) - ref[i]);
        if (relative) e /= std::fmax(1.0, std::fabs(ref[i]));
        worst = std::fmax(worst, e);
    }
    const bool ok = worst <= 1e-5;
    std::printf("%-12s fp32  max_%s %.3e  %s\n", name, relative ? "rel" : "abs", worst, ok ? "OK" : "FAIL");
    failures += !ok;
}

} // namespace

int main() {
    std::mt19937 g(1234);
    cudaStream_t s = nullptr;
    const int H = 896, heads = 14, kvh = 2, dim = 64, seq = 37, ctx = 64, F = 4864;

    { // rms norm
        auto x = rand_bf16(H, g), w = rand_bf16(H, g);
        auto xf = f32(x), wf = f32(w);
        double ss = 0;
        for (float v : xf) ss += double(v) * v;
        const float inv = 1.f / std::sqrt(float(ss / H) + 1e-6f);
        std::vector<float> ref(H);
        for (int i = 0; i < H; ++i) ref[i] = xf[i] * inv * wf[i];
        auto *dx = dev(x), *dw = dev(w), *dy = dev(std::vector<bf16>(H));
        tqllm_cuda::rms(dx, dw, dy, H, 1e-6f, s);
        check_bf16("rms", host(dy, H), ref);
    }
    { // bias, residual, swiglu
        auto a = rand_bf16(F, g), b = rand_bf16(F, g);
        auto af = f32(a), bf = f32(b);
        std::vector<float> add(F), sw(F);
        for (int i = 0; i < F; ++i) {
            add[i] = af[i] + bf[i];
            sw[i] = af[i] / (1.f + std::exp(-af[i])) * bf[i];
        }
        auto *da = dev(a), *db = dev(b), *dy = dev(std::vector<bf16>(F));
        tqllm_cuda::bias(da, db, dy, F, s);
        check_bf16("bias", host(dy, F), add);
        tqllm_cuda::swiglu(da, db, dy, F, s);
        check_bf16("swiglu", host(dy, F), sw);
        tqllm_cuda::residual(da, db, F, s);
        check_bf16("residual", host(da, F), add);
    }
    { // rope (rotate-half, theta 1e6)
        const int pos = 1234;
        auto x = rand_bf16(size_t(heads) * dim, g);
        auto xf = f32(x);
        std::vector<float> ref = xf;
        for (int h = 0; h < heads; ++h)
            for (int d = 0; d < dim / 2; ++d) {
                const float fr = std::pow(1e6f, -float(d) / (dim / 2)), ang = pos * fr;
                const float a = xf[h * dim + d], b = xf[h * dim + d + dim / 2];
                ref[h * dim + d] = a * std::cos(ang) - b * std::sin(ang);
                ref[h * dim + d + dim / 2] = a * std::sin(ang) + b * std::cos(ang);
            }
        auto * dx = dev(x);
        tqllm_cuda::rope(dx, heads, dim, pos, 1e6f, s);
        check_bf16("rope", host(dx, x.size()), ref);
    }
    { // attention: scores (fp32) -> softmax (fp32) -> attend (bf16), GQA 14:2
        auto q = rand_bf16(size_t(heads) * dim, g), kc = rand_bf16(size_t(ctx) * kvh * dim, g),
             vc = rand_bf16(size_t(ctx) * kvh * dim, g);
        auto qf = f32(q), kf = f32(kc), vf = f32(vc);
        const float scale = 1.f / std::sqrt(float(dim));
        std::vector<float> sc(size_t(heads) * ctx, 0.f), pr(size_t(heads) * ctx, 0.f), out(size_t(heads) * dim);
        for (int h = 0; h < heads; ++h) {
            const int kh = h / (heads / kvh);
            float m = -INFINITY, z = 0;
            for (int p = 0; p < seq; ++p) {
                double acc = 0;
                for (int d = 0; d < dim; ++d) acc += double(qf[h * dim + d]) * kf[(size_t(p) * kvh + kh) * dim + d];
                sc[h * ctx + p] = float(acc) * scale;
                m = std::fmax(m, sc[h * ctx + p]);
            }
            for (int p = 0; p < seq; ++p) z += (pr[h * ctx + p] = std::exp(sc[h * ctx + p] - m));
            for (int p = 0; p < seq; ++p) pr[h * ctx + p] /= z;
            for (int d = 0; d < dim; ++d) {
                double acc = 0;
                for (int p = 0; p < seq; ++p) acc += double(pr[h * ctx + p]) * vf[(size_t(p) * kvh + kh) * dim + d];
                out[h * dim + d] = float(acc);
            }
        }
        auto *dq = dev(q), *dk = dev(kc), *dv = dev(vc), *dy = dev(std::vector<bf16>(size_t(heads) * dim));
        auto * dsc = dev(std::vector<float>(size_t(heads) * ctx, 0.f));
        tqllm_cuda::scores(dq, dk, dsc, seq, ctx, heads, kvh, dim, scale, s);
        auto got_sc = host(dsc, sc.size());
        std::vector<float> g_sc, r_sc;
        for (int h = 0; h < heads; ++h)
            for (int p = 0; p < seq; ++p) {
                g_sc.push_back(got_sc[h * ctx + p]);
                r_sc.push_back(sc[h * ctx + p]);
            }
        check_f32("scores", g_sc, r_sc, true);
        tqllm_cuda::softmax(dsc, seq, ctx, heads, s);
        auto got_pr = host(dsc, pr.size());
        std::vector<float> g_pr, r_pr;
        for (int h = 0; h < heads; ++h)
            for (int p = 0; p < seq; ++p) {
                g_pr.push_back(got_pr[h * ctx + p]);
                r_pr.push_back(pr[h * ctx + p]);
            }
        check_f32("softmax", g_pr, r_pr, false);
        tqllm_cuda::attend(dsc, dv, dy, seq, ctx, heads, kvh, dim, s);
        check_bf16("attend", host(dy, out.size()), out);
    }
    for (int sq : {1, 37, 128, 129, 1000, 5000}) { // fused split-KV decode attention vs FP64-accumulated reference
        const int cx = 5120;
        auto q = rand_bf16(size_t(heads) * dim, g), kc = rand_bf16(size_t(cx) * kvh * dim, g),
             vc = rand_bf16(size_t(cx) * kvh * dim, g);
        auto qf = f32(q), kf = f32(kc), vf = f32(vc);
        const float scale = 1.f / std::sqrt(float(dim));
        std::vector<float> out(size_t(heads) * dim);
        for (int h = 0; h < heads; ++h) {
            const int kh = h / (heads / kvh);
            std::vector<double> sc(sq);
            double m = -1e300, z = 0;
            for (int p = 0; p < sq; ++p) {
                double acc = 0;
                for (int d = 0; d < dim; ++d) acc += double(qf[h * dim + d]) * kf[(size_t(p) * kvh + kh) * dim + d];
                sc[p] = acc * scale;
                m = std::fmax(m, sc[p]);
            }
            for (auto & v : sc) z += (v = std::exp(v - m));
            for (int d = 0; d < dim; ++d) {
                double acc = 0;
                for (int p = 0; p < sq; ++p) acc += sc[p] / z * vf[(size_t(p) * kvh + kh) * dim + d];
                out[h * dim + d] = float(acc);
            }
        }
        auto *dq = dev(q), *dk = dev(kc), *dv = dev(vc), *dy = dev(std::vector<bf16>(size_t(heads) * dim));
        auto * dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_partial_floats(heads, cx, dim))));
        tqllm_cuda::attention_decode(dq, dk, dv, dy, dp, sq, heads, kvh, dim, scale, s);
        char name[32];
        std::snprintf(name, sizeof(name), "attn_dec%d", sq);
        check_bf16(name, host(dy, out.size()), out);
    }

    { // Qwen3.5: zero-centred RMSNorm on interleaved rows (q_norm peels q out of [q | gate])
        const int nh = 16, hd = 256;
        auto qg = rand_bf16(size_t(nh) * 2 * hd, g), w = rand_bf16(hd, g, 0.1f);
        auto qf = f32(qg), wf = f32(w);
        std::vector<float> w1(hd);  // zero-centred weight baked to 1 + w at load
        for (int d = 0; d < hd; ++d) w1[d] = 1.f + wf[d];
        std::vector<float> ref(size_t(nh) * hd);
        for (int h = 0; h < nh; ++h) {
            double ss = 0;
            for (int d = 0; d < hd; ++d) ss += double(qf[h * 2 * hd + d]) * qf[h * 2 * hd + d];
            const float inv = 1.f / std::sqrt(float(ss / hd) + 1e-6f);
            for (int d = 0; d < hd; ++d) ref[h * hd + d] = qf[h * 2 * hd + d] * inv * (1.f + wf[d]);
        }
        auto *dq = dev(qg), *dy = dev(std::vector<bf16>(size_t(nh) * hd));
        auto * dw = dev(w1);
        tqllm_cuda::rms_f(dq, 2 * hd, dw, dy, hd, hd, nh, 1e-6f, s);
        check_bf16("rms_f", host(dy, ref.size()), ref);

        std::vector<float> sg(size_t(nh) * hd);  // sigmoid gate on the same layout
        auto o = rand_bf16(size_t(nh) * hd, g);
        auto of = f32(o);
        for (int h = 0; h < nh; ++h)
            for (int d = 0; d < hd; ++d)
                sg[h * hd + d] = of[h * hd + d] / (1.f + std::exp(-qf[h * 2 * hd + hd + d]));
        auto * dO = dev(o);
        tqllm_cuda::sigmoid_gate(dO, dq, nh, hd, s);
        check_bf16("sigm_gate", host(dO, sg.size()), sg);
    }
    { // partial RoPE: first 64 of 256 dims, theta 1e7
        const int nh = 4, hd = 256, rot = 64, pos = 20000;
        auto x = rand_bf16(size_t(nh) * hd, g);
        auto xf = f32(x);
        std::vector<float> ref = xf;
        for (int h = 0; h < nh; ++h)
            for (int d = 0; d < rot / 2; ++d) {
                const double ang = double(pos) / std::pow(1e7, 2.0 * d / rot);
                const float a = xf[h * hd + d], b = xf[h * hd + d + rot / 2];
                ref[h * hd + d] = float(a * std::cos(ang) - b * std::sin(ang));
                ref[h * hd + d + rot / 2] = float(b * std::cos(ang) + a * std::sin(ang));
            }
        auto * dx = dev(x);
        tqllm_cuda::rope_partial(dx, nh, hd, rot, pos, 1e7f, s);
        check_bf16("rope_part", host(dx, ref.size()), ref);
    }
    { // conv step (k=4) + SiLU, two consecutive steps to exercise the state shift
        const int C = 8192;
        auto w = rand_bf16(size_t(C) * 4, g, 0.5f), x1 = rand_bf16(C, g), x2 = rand_bf16(C, g), st = rand_bf16(size_t(3) * C, g);
        auto wf = f32(w), a = f32(x1), b = f32(x2), sf = f32(st);
        auto * dwf = dev(wf);
        std::vector<float> ref(C);
        for (int c = 0; c < C; ++c) {  // second step sees state [s1, s2, x1] and input x2
            const float acc = wf[c * 4] * sf[C + c] + wf[c * 4 + 1] * sf[2 * C + c] + wf[c * 4 + 2] * a[c] + wf[c * 4 + 3] * b[c];
            ref[c] = acc / (1.f + std::exp(-acc));
        }
        auto *d1 = dev(x1), *d2 = dev(x2), *ds = dev(st), *dy = dev(std::vector<bf16>(C));
        tqllm_cuda::conv_step(d1, ds, dwf, dy, C, s);
        tqllm_cuda::conv_step(d2, ds, dwf, dy, C, s);
        check_bf16("conv_step", host(dy, C), ref);
    }
    for (bool tiled : {false, true}) { // gated delta rule: 3 steps from a random state, FP64 reference
        const int kh = 16, vh = 32, dk = 128, dv = 128, qkvd = 2 * kh * dk + vh * dv;
        std::vector<float> S0(size_t(vh) * dk * dv);
        std::normal_distribution<float> nd(0.f, 0.05f);
        for (auto & v : S0) v = nd(g);
        auto alog = std::vector<float>(vh), nega = std::vector<float>(vh);
        for (int h = 0; h < vh; ++h) nega[h] = -std::exp(alog[h] = std::log(1.f + h * 0.5f));
        auto dtf = f32(rand_bf16(vh, g, 0.5f));
        std::vector<double> S(S0.begin(), S0.end());
        auto * dS = dev(S0);
        auto * dal = dev(nega);
        auto * ddt = dev(dtf);
        auto * dout = dev(std::vector<bf16>(size_t(vh) * dv));
        std::vector<float> ref(size_t(vh) * dv);
        for (int step = 0; step < 3; ++step) {
            auto qkv = rand_bf16(qkvd, g), a = rand_bf16(vh, g), b = rand_bf16(vh, g);
            auto qf = f32(qkv), af = f32(a), bf = f32(b);
            for (int h = 0; h < vh; ++h) {
                const int k0 = tiled ? h % kh : h / (vh / kh);
                std::vector<double> q(dk), k(dk);
                double nq = 0, nk = 0;
                for (int i = 0; i < dk; ++i) {
                    q[i] = qf[k0 * dk + i];
                    k[i] = qf[kh * dk + k0 * dk + i];
                    nq += q[i] * q[i];
                    nk += k[i] * k[i];
                }
                for (int i = 0; i < dk; ++i) {
                    q[i] = q[i] / std::sqrt(nq + 1e-6) / std::sqrt(double(dk));
                    k[i] = k[i] / std::sqrt(nk + 1e-6);
                }
                const double x = double(af[h]) + dtf[h];
                const double decay = std::exp(-std::exp(double(alog[h])) * std::log1p(std::exp(x)));
                const double beta = 1.0 / (1.0 + std::exp(-double(bf[h])));
                double * Sh = S.data() + size_t(h) * dk * dv;
                for (int j = 0; j < dv; ++j) {
                    double mem = 0;
                    for (int i = 0; i < dk; ++i) mem += (Sh[i * dv + j] *= decay) * k[i];
                    const double delta = (qf[2 * kh * dk + h * dv + j] - mem) * beta;
                    double o = 0;
                    for (int i = 0; i < dk; ++i) o += (Sh[i * dv + j] += k[i] * delta) * q[i];
                    ref[h * dv + j] = float(o);
                }
            }
            auto *dq = dev(qkv), *da = dev(a), *db = dev(b);
            tqllm_cuda::gdn_step(dq, da, db, dal, ddt, dS, dout, kh, vh, dk, dv, tiled, s);
        }
        check_bf16(tiled ? "gdn_tiled" : "gdn_grouped", host(dout, ref.size()), ref);
        auto got = host(dS, S0.size());
        double worst = 0;
        for (size_t i = 0; i < S.size(); ++i) worst = std::fmax(worst, std::fabs(got[i] - S[i]));
        const bool ok = worst <= 1e-5;
        std::printf("%-12s fp32  max_abs %.3e  %s\n", "gdn_state", worst, ok ? "OK" : "FAIL");
        failures += !ok;
    }
    { // gated RMSNorm per head (FP32 weight)
        const int nh = 32, dv = 128;
        auto x = rand_bf16(size_t(nh) * dv, g), z = rand_bf16(size_t(nh) * dv, g);
        std::vector<float> w(dv);
        for (int i = 0; i < dv; ++i) w[i] = 0.5f + 0.01f * i;
        auto xf = f32(x), zf = f32(z);
        std::vector<float> ref(size_t(nh) * dv);
        for (int h = 0; h < nh; ++h) {
            double ss = 0;
            for (int d = 0; d < dv; ++d) ss += double(xf[h * dv + d]) * xf[h * dv + d];
            const float inv = 1.f / std::sqrt(float(ss / dv) + 1e-6f);
            for (int d = 0; d < dv; ++d) {
                const float zz = zf[h * dv + d];
                // HF casts the normalised value back to BF16 before the FP32 weight and gate multiply
                const float n = __bfloat162float(__float2bfloat16_rn(xf[h * dv + d] * inv));
                ref[h * dv + d] = w[d] * n * zz / (1.f + std::exp(-zz));
            }
        }
        auto *dx = dev(x), *dz = dev(z), *dy = dev(std::vector<bf16>(ref.size()));
        auto * dw = dev(w);
        tqllm_cuda::gated_rms(dx, dz, dw, dy, nh, dv, 1e-6f, s);
        check_bf16("gated_rms", host(dy, ref.size()), ref);
    }
    for (int sq : {1, 63, 64, 65, 3000}) { // head_dim 256 attention, GQA 16:4
        const int nh = 16, kv = 4, hd = 256, cx = 3072;
        auto q = rand_bf16(size_t(nh) * hd, g), kc = rand_bf16(size_t(cx) * kv * hd, g), vc = rand_bf16(size_t(cx) * kv * hd, g);
        auto qf = f32(q), kf = f32(kc), vf = f32(vc);
        const float scale = 1.f / 16.f;
        std::vector<float> out(size_t(nh) * hd);
        for (int h = 0; h < nh; ++h) {
            const int k0 = h / (nh / kv);
            std::vector<double> sc(sq);
            double m = -1e300, z = 0;
            for (int p = 0; p < sq; ++p) {
                double acc = 0;
                for (int d = 0; d < hd; ++d) acc += double(qf[h * hd + d]) * kf[(size_t(p) * kv + k0) * hd + d];
                m = std::fmax(m, sc[p] = acc * scale);
            }
            for (auto & v : sc) z += (v = std::exp(v - m));
            for (int d = 0; d < hd; ++d) {
                double acc = 0;
                for (int p = 0; p < sq; ++p) acc += sc[p] / z * vf[(size_t(p) * kv + k0) * hd + d];
                out[h * hd + d] = float(acc);
            }
        }
        auto *dq = dev(q), *dk = dev(kc), *dv = dev(vc), *dy = dev(std::vector<bf16>(out.size()));
        auto * dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_partial_floats(nh, cx, hd))));
        tqllm_cuda::attention_decode(dq, dk, dv, dy, dp, sq, nh, kv, hd, scale, s);
        char name[32];
        std::snprintf(name, sizeof(name), "attn256_%d", sq);
        check_bf16(name, host(dy, out.size()), out);
    }
    { // argmax with a tie: lowest index wins
        std::vector<float> x(151936);
        std::mt19937 gg(7);
        std::normal_distribution<float> d;
        for (auto & v : x) v = d(gg);
        x[1000] = x[90000] = 50.f;
        auto * dx = dev(x);
        uint32_t * di = nullptr;
        cudaMalloc(&di, sizeof(uint32_t));
        tqllm_cuda::argmax(dx, int(x.size()), di, s);
        const auto got = host(di, 1)[0];
        const bool ok = got == 1000;
        std::printf("%-12s exact got %u want 1000  %s\n", "argmax", got, ok ? "OK" : "FAIL");
        failures += !ok;
    }
    for (const int P : {0, 333, 2500}) {  // prefill attention (head_dim 256, 6 heads per KV head) vs FP32 CPU, causal
        const int n = P > 1000 ? 640 : 200, ah = 24, akv = 4, D = 256, keys = P + n, grp = ah / akv;
        auto q = rand_bf16(size_t(n) * ah * D, g), k = rand_bf16(size_t(keys) * akv * D, g), v = rand_bf16(size_t(keys) * akv * D, g);
        const auto qf = f32(q), kf = f32(k), vf = f32(v);
        const float scale = 1.f / 16.f;
        std::vector<float> ref(size_t(n) * ah * D);
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < ah; ++h) {
                const int kvh_ = h / grp, nk = P + i + 1;
                std::vector<double> sc(size_t(nk) + 0);
                double mx = -1e30, z = 0;
                for (int j = 0; j < nk; ++j) {
                    double d = 0;
                    for (int e = 0; e < D; ++e) d += double(qf[(size_t(i) * ah + h) * D + e]) * kf[(size_t(j) * akv + kvh_) * D + e];
                    sc[size_t(j)] = d * scale;
                    mx = std::fmax(mx, sc[size_t(j)]);
                }
                for (auto & x : sc) z += (x = std::exp(x - mx));
                for (int e = 0; e < D; ++e) {
                    double acc = 0;
                    for (int j = 0; j < nk; ++j) acc += sc[size_t(j)] * vf[(size_t(j) * akv + kvh_) * D + e];
                    ref[(size_t(i) * ah + h) * D + e] = float(acc / z);
                }
            }
        const int pos[2] = {P, P + 1};
        auto *dq = dev(q), *dk = dev(k), *dv = dev(v), *dy = dev(std::vector<bf16>(ref.size()));
        int * dp = dev(std::vector<int>(pos, pos + 2));
        const int nsplit = 8;
        const size_t pf = size_t(tqllm_cuda::attention_mq_partial_floats(ah, n, 1));
        float * part = dev(std::vector<float>(std::max(pf, size_t(tqllm_cuda::attention_mq_partial_floats(ah, 8, nsplit)))));
        tqllm_cuda::attention_mq256_prefill(dq, dk, dv, dy, part, pf, dp + 1, n, ah, akv, scale, nsplit, s);
        check_bf16(P > 1000 ? "attn_pf_2500" : P ? "attn_pf_333" : "attn_pf_0", host(dy, ref.size()), ref);
        for (int off = 0; off < n; off += 8)  // the verify kernel per 8-query group, same reference
            tqllm_cuda::attention_mq256(dq + size_t(off) * ah * D, dk, dv, dy + size_t(off) * ah * D, part, dp + 1,
                                        std::min(8, n - off), ah, akv, scale, nsplit, s, off);
        check_bf16(P > 1000 ? "attn_mq_2500" : P ? "attn_mq_333" : "attn_mq_0", host(dy, ref.size()), ref);
        cudaFree(dq), cudaFree(dk), cudaFree(dv), cudaFree(dy), cudaFree(dp), cudaFree(part);
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::printf("%s\n", failures ? "KERNEL TESTS FAILED" : "all kernel tests passed");
    return failures ? 1 : 0;
}

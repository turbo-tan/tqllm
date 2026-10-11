// DFlash2 kernels vs CPU references.
#include "dflash_kernels.cuh"
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"
#include "quant_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {
using bf16 = __nv_bfloat16;
int failures = 0;
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
void verdict(const char * name, bool ok, const char * detail) {
    std::printf("%-18s %s  %s\n", name, detail, ok ? "OK" : "FAIL");
    failures += !ok;
}

const float T3_S1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
const float T3_S2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};
const float T3_C[8] = {-0.190207f,-0.118786f,-0.066822f,-0.021663f,0.021663f,0.066822f,0.118786f,0.190207f};

void fwht128(float * x, const float * sign, float norm) {
    for (int i = 0; i < 128; ++i) x[i] *= sign[i];
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
    for (int i = 0; i < 128; ++i) x[i] *= norm;
}

// ---- CPU port of llama.cpp-tq3 quantize_row_tq3_4s_ref ----
const float SG[32] = {+1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1,
                      -1, -1, +1, -1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, -1, +1};
const float CB[8] = {-1.996684f, -1.291398f, -0.740341f, -0.247508f, 0.230106f, 0.725222f, 1.277503f, 1.988943f};
const float BD[7] = {-1.644041f, -1.015870f, -0.493924f, -0.008701f, 0.477664f, 1.001362f, 1.633223f};
uint8_t enc(float val) {
    if (val <= 0.f) return 0;
    int e = int(std::floor(std::log2(val))) + 9;
    e = std::clamp(e, 0, 7);
    int m = int(std::lrint((val / std::ldexp(1.f, e - 9) - 1.f) * 32.f));
    m = std::clamp(m, 0, 31);
    return uint8_t((e << 5) | m);
}
void quant_ref(const float * x, uint8_t * y) {
    float r[32];
    for (int i = 0; i < 32; ++i) r[i] = x[i] * SG[i];
    for (int step = 1; step < 32; step <<= 1)
        for (int i = 0; i < 32; i += step << 1)
            for (int j = i; j < i + step; ++j) {
                const float a = r[j], b = r[j + step];
                r[j] = a + b;
                r[j + step] = a - b;
            }
    for (float & v : r) v /= std::sqrt(32.f);
    uint8_t idx[32];
    for (int g = 0; g < 4; ++g) {
        float ss = 0;
        for (int j = 0; j < 8; ++j) ss += r[g * 8 + j] * r[g * 8 + j];
        float scale = std::fmax(std::sqrt(ss / 8.f), 1e-10f);
        for (int it = 0; it < 4; ++it) {
            const float inv = 1.f / std::fmax(scale, 1e-10f);
            float nu = 0, de = 0;
            for (int j = 0; j < 8; ++j) {
                uint8_t id = 0;
                for (int b = 0; b < 7; ++b) if (r[g * 8 + j] * inv > BD[b]) id = uint8_t(b + 1);
                idx[g * 8 + j] = id;
                nu += r[g * 8 + j] * CB[id];
                de += CB[id] * CB[id];
            }
            if (de > 1e-12f) scale = std::fmax(nu / de, 1e-10f);
        }
        y[g] = enc(scale);
        const uint8_t * d = idx + g * 8;
        uint8_t * qp = y + 4 + g * 3;
        qp[0] = uint8_t(d[0] | (d[1] << 3) | (d[2] << 6));
        qp[1] = uint8_t((d[2] >> 2) | (d[3] << 1) | (d[4] << 4) | (d[5] << 7));
        qp[2] = uint8_t((d[5] >> 1) | (d[6] << 2) | (d[7] << 5));
    }
}
} // namespace

int main() {
    std::mt19937 g(5);
    std::normal_distribution<float> nd;
    { // quantizer: fraction of blocks bit-identical to the reference (float op order may flip rare ties)
        const int nb = 4096;
        std::vector<float> x(size_t(nb) * 32);
        for (auto & v : x) v = nd(g) * 0.05f;
        std::vector<uint8_t> ref(size_t(nb) * 16);
        for (int b = 0; b < nb; ++b) quant_ref(&x[size_t(b) * 32], &ref[size_t(b) * 16]);
        auto * dx = dev(x);
        uint8_t * dq = nullptr;
        cudaMalloc(&dq, ref.size());
        tqllm_cuda::tq3_quantize(dx, dq, nb, nullptr);
        const auto got = host(dq, ref.size());
        int same = 0;
        for (int b = 0; b < nb; ++b) same += std::memcmp(&got[size_t(b) * 16], &ref[size_t(b) * 16], 16) == 0;
        char d[64];
        std::snprintf(d, sizeof(d), "%d/%d blocks bit-identical", same, nb);
        verdict("tq3_quantize", same >= nb * 99 / 100, d);
    }
    { // dynamic conv, kernel 2, group 16
        const int n = 8, C = 5120, grp = 16, G = C / grp;
        std::vector<bf16> h(size_t(n) * C), dyn(size_t(n) * 4 * G);
        std::vector<float> base(size_t(4) * C);
        for (auto & v : h) v = __float2bfloat16_rn(nd(g));
        for (auto & v : dyn) v = __float2bfloat16_rn(nd(g) * 0.1f);
        for (auto & v : base) v = nd(g) * 0.1f;
        auto *dh = dev(h), *dd = dev(dyn), *dy = dev(std::vector<bf16>(h.size()));
        auto * db = dev(base);
        double worst = 0;
        for (int side = 0; side < 2; ++side) {
            tqllm_cuda::dflash_conv(dh, dd, 4 * G, db, side, dy, n, C, grp, nullptr);
            const auto y = host(dy, h.size());
            for (int t = 0; t < n; ++t)
                for (int c = 0; c < C; ++c) {
                    double acc = 0;
                    for (int tap = 0; tap < 2; ++tap) {
                        if (t - tap < 0) continue;
                        const int ci = tap + 2 * side;
                        acc += (double(__bfloat162float(dyn[size_t(t) * 4 * G + c / grp + G * ci])) + base[c + size_t(C) * ci]) *
                               __bfloat162float(h[size_t(t - tap) * C + c]);
                    }
                    worst = std::fmax(worst, std::fabs(acc - __bfloat162float(y[size_t(t) * C + c])) / (1 + std::fabs(acc)));
                }
        }
        char d[64];
        std::snprintf(d, sizeof(d), "max rel %.2e", worst);
        verdict("dflash_conv", worst < 1e-2, d);
    }
    { // top-16
        const int rows = 7, len = 40960;
        std::vector<float> x(size_t(rows) * len);
        for (auto & v : x) v = nd(g);
        x[5] = x[len - 1] = 99.f;  // tie on the maximum
        auto * dx = dev(x);
        float * dv = nullptr;
        uint32_t * di = nullptr;
        cudaMalloc(&dv, rows * 16 * 4);
        cudaMalloc(&di, rows * 16 * 4);
        tqllm_cuda::topk16(dx, len, rows, dv, di, nullptr);
        const auto gi = host(di, rows * 16);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            std::vector<uint32_t> o(len);
            std::iota(o.begin(), o.end(), 0u);
            const float * xr = &x[size_t(r) * len];
            std::stable_sort(o.begin(), o.end(), [&](uint32_t a, uint32_t b) { return xr[a] > xr[b]; });
            for (int k = 0; k < 16; ++k) ok &= gi[r * 16 + k] == o[k];
        }
        verdict("topk16", ok, "exact order incl. ties");
    }
    { // attention_gen128: causal, non-causal block, sliding window
        const int heads = 32, kvh = 8, D = 128, ctx = 3000, nq = 8;
        std::vector<bf16> q(size_t(nq) * heads * D), kc(size_t(ctx) * kvh * D), vc(kc.size());
        for (auto & v : q) v = __float2bfloat16_rn(nd(g));
        for (auto & v : kc) v = __float2bfloat16_rn(nd(g));
        for (auto & v : vc) v = __float2bfloat16_rn(nd(g));
        auto *dq = dev(q), *dk = dev(kc), *dv = dev(vc), *dy = dev(std::vector<bf16>(q.size()));
        auto * dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_gen_partial_floats(heads, ctx, nq))));
        const int ring = 2176;  // window 2048 + nq + 64, multiple of 64
        std::vector<bf16> kr(size_t(ring) * kvh * D), vr(kr.size());
        for (int p = 0; p < ctx; ++p)  // later positions overwrite earlier ones, as sequential cache writes do
            for (int i = 0; i < kvh * D; ++i) {
                kr[size_t(p % ring) * kvh * D + i] = kc[size_t(p) * kvh * D + i];
                vr[size_t(p % ring) * kvh * D + i] = vc[size_t(p) * kvh * D + i];
            }
        auto *dkr = dev(kr), *dvr = dev(vr);
        for (int mode = 0; mode < 5; ++mode) {  // mode 4: ring early in the sequence (slots == positions)
            const int p0 = mode == 4 ? 3 : mode >= 2 ? 2900 : 1234, window = mode >= 2 ? 2048 : 0;
            const bool causal = mode == 0;
            auto * dpos = dev(std::vector<int>{p0});
            if (mode == 4)
                tqllm_cuda::attention_gen128(dq, dk, dv, dy, dp, dpos, nq, ctx, heads, kvh, 1.f / std::sqrt(128.f),
                                             causal, window, nullptr, ring);
            else if (mode == 3)
                tqllm_cuda::attention_gen128(dq, dkr, dvr, dy, dp, dpos, nq, ctx, heads, kvh, 1.f / std::sqrt(128.f),
                                             causal, window, nullptr, ring);
            else
                tqllm_cuda::attention_gen128(dq, dk, dv, dy, dp, dpos, nq, ctx, heads, kvh, 1.f / std::sqrt(128.f),
                                             causal, window, nullptr);
            const auto y = host(dy, q.size());
            double se = 0, sr = 0;
            for (int z = 0; z < nq; ++z) {
                const int pz = p0 + z, hi = causal ? pz + 1 : p0 + nq, lo = window ? std::max(0, pz - window + 1) : 0;
                for (int h = 0; h < heads; ++h) {
                    const int k0 = h / (heads / kvh);
                    std::vector<double> sc(hi - lo);
                    double m = -1e300, zs = 0;
                    for (int p = lo; p < hi; ++p) {
                        double a = 0;
                        for (int d = 0; d < D; ++d)
                            a += double(__bfloat162float(q[(size_t(z) * heads + h) * D + d])) *
                                 __bfloat162float(kc[(size_t(p) * kvh + k0) * D + d]);
                        m = std::fmax(m, sc[p - lo] = a / std::sqrt(128.0));
                    }
                    for (auto & v : sc) zs += (v = std::exp(v - m));
                    for (int d = 0; d < D; ++d) {
                        double a = 0;
                        for (int p = lo; p < hi; ++p) a += sc[p - lo] / zs * __bfloat162float(vc[(size_t(p) * kvh + k0) * D + d]);
                        const double e = a - __bfloat162float(y[(size_t(z) * heads + h) * D + d]);
                        se += e * e;
                        sr += a * a;
                    }
                }
            }
            char name[32], d[64];
            std::snprintf(name, sizeof(name), "attn128_%s", mode == 0 ? "causal" : mode == 1 ? "block" : mode == 4 ? "ring_early" : "window");
            std::snprintf(d, sizeof(d), "rel_rms %.2e", std::sqrt(se / sr));
            verdict(name, std::sqrt(se / sr) < 5e-3, d);
        }
    }
    { // multi-query verify attention (head_dim 256) vs CPU, causal, several depths
        const int heads = 24, kvh = 4, D = 256;
        struct Case { int seq0, nq; bool q8; };
        for (auto [seq0, nq, q8] : {Case{5, 8, false}, {100, 8, false}, {3000, 8, false}, {3000, 3, false}, {37, 1, false},
                                    {3000, 8, true}, {77, 5, true}}) {
            const int ctx = seq0 + nq + 64;
            std::vector<bf16> q(size_t(nq) * heads * D), kc(size_t(ctx) * kvh * D), vc(kc.size());
            for (auto & v : q) v = __float2bfloat16_rn(nd(g));
            for (auto & v : kc) v = __float2bfloat16_rn(nd(g));
            for (auto & v : vc) v = __float2bfloat16_rn(nd(g));
            std::vector<int8_t> k8(kc.size()), v8(vc.size());
            std::vector<__half> ks(kc.size() / 32), vs(vc.size() / 32);
            if (q8)  // host q8_0-style quantisation; the reference below then uses the dequantised values
                for (size_t b = 0; b < kc.size() / 32; ++b)
                    for (int which = 0; which < 2; ++which) {
                        auto & x = which ? vc : kc;
                        float m = 0.f;
                        for (int i = 0; i < 32; ++i) m = std::fmax(m, std::fabs(__bfloat162float(x[b * 32 + i])));
                        const __half hs = __float2half_rn(m / 127.f);
                        const float sf = __half2float(hs);
                        (which ? vs : ks)[b] = hs;
                        for (int i = 0; i < 32; ++i) {
                            const int qv = sf > 0.f ? int(std::lrint(__bfloat162float(x[b * 32 + i]) / sf)) : 0;
                            (which ? v8 : k8)[b * 32 + i] = int8_t(std::max(-127, std::min(127, qv)));
                            x[b * 32 + i] = __float2bfloat16_rn(float((which ? v8 : k8)[b * 32 + i]) * sf);
                        }
                    }
            auto *dq = dev(q), *dk = dev(kc), *dv = dev(vc), *dy = dev(std::vector<bf16>(q.size()));
            auto *dk8 = dev(k8), *dv8 = dev(v8);
            auto *dks = dev(ks), *dvs = dev(vs);
            const int nsplit = 32;
            auto * dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_mq_partial_floats(heads, nq, nsplit))));
            auto * ds = dev(std::vector<int>{seq0});
            if (q8)
                tqllm_cuda::attention_mq256_q8(dq, dk8, dv8, dks, dvs, dy, dp, ds, nq, heads, kvh, 1.f / 16.f, nsplit, nullptr);
            else
                tqllm_cuda::attention_mq256(dq, dk, dv, dy, dp, ds, nq, heads, kvh, 1.f / 16.f, nsplit, nullptr);
            const auto y = host(dy, q.size());
            double se = 0, sr = 0;
            for (int z = 0; z < nq; ++z)
                for (int h = 0; h < heads; ++h) {
                    const int k0 = h / (heads / kvh), hi = seq0 + z;
                    std::vector<double> sc(hi);
                    double m = -1e300, zs = 0;
                    for (int p = 0; p < hi; ++p) {
                        double a = 0;
                        for (int d = 0; d < D; ++d)
                            a += double(__bfloat162float(q[(size_t(z) * heads + h) * D + d])) * __bfloat162float(kc[(size_t(p) * kvh + k0) * D + d]);
                        m = std::fmax(m, sc[p] = a / 16.0);
                    }
                    for (auto & v : sc) zs += (v = std::exp(v - m));
                    for (int d = 0; d < D; ++d) {
                        double a = 0;
                        for (int p = 0; p < hi; ++p) a += sc[p] / zs * __bfloat162float(vc[(size_t(p) * kvh + k0) * D + d]);
                        const double e = a - __bfloat162float(y[(size_t(z) * heads + h) * D + d]);
                        se += e * e;
                        sr += a * a;
                    }
                }
            char name[32], d[64];
            std::snprintf(name, sizeof(name), "attn_mq%s_%d_n%d", q8 ? "8" : "", seq0, nq);
            std::snprintf(d, sizeof(d), "rel_rms %.2e", std::sqrt(se / sr));
            verdict(name, std::sqrt(se / sr) < 5e-3, d);
        }
    }
    { // asymmetric Q8-K / Turbo3-V cache writer and attention against CPU reference
        const int heads = 24, kvh = 4, D = 256, seq0 = 37, nq = 8, n = seq0 + nq, ctx = n;
        std::vector<bf16> q(size_t(nq) * heads * D), k(size_t(n) * kvh * D), v(k.size());
        for (auto & x : q) x = __float2bfloat16_rn(nd(g));
        for (auto & x : k) x = __float2bfloat16_rn(nd(g));
        for (auto & x : v) x = __float2bfloat16_rn(nd(g));
        std::vector<int8_t> k8(size_t(ctx) * kvh * D);
        std::vector<__half> ks(k8.size() / 32);
        std::vector<tqllm_cuda::Turbo3V256> v3(size_t(ctx) * kvh);
        auto idx_of = [](const tqllm_cuda::Turbo3V256 & r, int d) {  // GGML codebook index of dim d
            const int c = d / 32, l = d % 32 % 8, j = d % 32 / 8, bit = 12 * l + 3 * j;
            int code = 0;
            for (int b = 0; b < 3; ++b) code |= int((r.w[3 * c + (bit + b) / 32] >> ((bit + b) % 32)) & 1u) << b;
            return code >> 2 ? 4 + (code & 3) : 3 - (code & 3);
        };
        auto *dq = dev(q), *dk0 = dev(k), *dv0 = dev(v);
        auto *dk8 = dev(k8);
        auto *dks = dev(ks);
        auto *dv3 = dev(v3);
        auto *dy = dev(std::vector<bf16>(q.size()));
        auto *dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_mq_partial_floats(heads, nq, 8))));
        auto *dpos = dev(std::vector<int>{0}), *ds = dev(std::vector<int>{seq0});
        tqllm_cuda::cache_write_multi_q8_turbo3v(dk0, dv0, kvh * D, n, dk8, dks, dv3, dpos, kvh, D, nullptr);
        tqllm_cuda::attention_mq256_q8_turbo3v(dq, dk8, dv3, dks, dy, dp, ds, nq, heads, kvh,
                                               1.f / 16.f, 8, nullptr);
        const auto got_k8 = host(dk8, k8.size());
        const auto got_ks = host(dks, ks.size());
        const auto got_v3 = host(dv3, v3.size());
        const auto got_y = host(dy, q.size());
        std::vector<float> kr(k.size()), vr(v.size());
        for (size_t i = 0; i < k.size(); ++i) {
            const float sk = __half2float(got_ks[i / 32]);
            kr[i] = float(got_k8[i]) * sk;
        }
        for (int t = 0; t < n; ++t)
            for (int kh = 0; kh < kvh; ++kh)
                for (int sub = 0; sub < D / 128; ++sub) {
                    const auto & b = got_v3[size_t(t) * kvh + kh];
                    float rotated[128];
                    const float norm = __half2float(b.norm[sub]);
                    for (int j = 0; j < 128; ++j) rotated[j] = T3_C[idx_of(b, sub * 128 + j)] * norm;
                    fwht128(rotated, T3_S2, 0.08838834764831845f);
                    for (int j = 0; j < 128; ++j)
                        vr[(size_t(t) * kvh + kh) * D + sub * 128 + j] = rotated[j] * T3_S1[j];
                }
        int vbytes_bad = 0, vnorm_bad = 0;
        for (int t = 0; t < n; ++t)
            for (int kh = 0; kh < kvh; ++kh)
                for (int sub = 0; sub < D / 128; ++sub) {
                    float x[128], n2 = 0.f;
                    const size_t in = (size_t(t) * kvh + kh) * D + sub * 128;
                    const auto & actual = got_v3[size_t(t) * kvh + kh];
                    for (int j = 0; j < 128; ++j) {
                        x[j] = __bfloat162float(v[in + j]);
                        n2 += x[j] * x[j];
                    }
                    const float norm = std::sqrt(n2);
                    for (int j = 0; j < 128; ++j) x[j] = norm > 1e-10f ? x[j] / norm : 0.f;
                    fwht128(x, T3_S1, 1.f);
                    for (int j = 0; j < 128; ++j) x[j] *= 0.08838834764831845f * T3_S2[j];
                    int code_bad = 0;
                    double recon_sq = 0;
                    for (int j = 0; j < 128; ++j) {
                        uint8_t ix = 0;
                        if (x[j] >= -0.154496f) ix = 1;
                        if (x[j] >= -0.092804f) ix = 2;
                        if (x[j] >= -0.044243f) ix = 3;
                        if (x[j] >= 0.f) ix = 4;
                        if (x[j] >= 0.044243f) ix = 5;
                        if (x[j] >= 0.092804f) ix = 6;
                        if (x[j] >= 0.154496f) ix = 7;
                        code_bad += idx_of(actual, sub * 128 + j) != ix;
                        recon_sq += T3_C[ix] * T3_C[ix];
                    }
                    vbytes_bad += code_bad != 0;
                    const float expected_norm = __half2float(__float2half_rn(norm / std::sqrt(float(recon_sq))));
                    vnorm_bad += std::fabs(expected_norm - __half2float(actual.norm[sub])) > 1e-3f;
                }
        double se = 0, sr = 0, vse = 0, vsr = 0;
        for (int z = 0; z < nq; ++z)
            for (int h = 0; h < heads; ++h) {
                const int kh = h / (heads / kvh), hi = seq0 + z;
                std::vector<double> scores(hi);
                double mx = -1e300, sum = 0;
                for (int p = 0; p < hi; ++p) {
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += double(__bfloat162float(q[(size_t(z) * heads + h) * D + d])) * kr[(size_t(p) * kvh + kh) * D + d];
                    mx = std::max(mx, scores[p] = a / 16.0);
                }
                for (auto & a : scores) sum += (a = std::exp(a - mx));
                for (int d = 0; d < D; ++d) {
                    double ref = 0;
                    for (int p = 0; p < hi; ++p) ref += scores[p] / sum * vr[(size_t(p) * kvh + kh) * D + d];
                    const double actual = __bfloat162float(got_y[(size_t(z) * heads + h) * D + d]);
                    const double e = ref - actual;
                    se += e * e;
                    sr += ref * ref;
                    const double raw = __bfloat162float(v[(size_t(0) * kvh + kh) * D + d]);
                    vse += (vr[(size_t(0) * kvh + kh) * D + d] - raw) * (vr[(size_t(0) * kvh + kh) * D + d] - raw);
                    vsr += raw * raw;
                }
            }
        char d[96];
        std::snprintf(d, sizeof(d), "attention %.2e; V %.3f; ref mismatches %d/%d", std::sqrt(se / sr), std::sqrt(vse / vsr), vbytes_bad, vnorm_bad);
        verdict("attn_mq_turbo3v", std::sqrt(se / sr) < 8e-3 && std::sqrt(vse / vsr) < 0.4 && vbytes_bad == 0 && vnorm_bad == 0, d);
    }
    for (int v4 = 0; v4 < 2; ++v4) { // Turbo4-K / Turbo3-V (v4 = 0) or Turbo4-V cache writer + attention vs CPU
        const int heads = 24, kvh = 4, D = 256, seq0 = 45, nq = 8, n = seq0 + nq;
        const int MAG[8] = {6, 18, 31, 44, 58, 75, 96, 127};
        const float B16[7] = {0.022824f, 0.046174f, 0.070671f, 0.097164f, 0.127026f, 0.162947f, 0.212203f};
        std::vector<bf16> q(size_t(nq) * heads * D), k(size_t(n) * kvh * D), v(k.size());
        for (auto & x : q) x = __float2bfloat16_rn(nd(g));
        for (auto & x : k) x = __float2bfloat16_rn(nd(g) * 2.f);
        for (auto & x : v) x = __float2bfloat16_rn(nd(g));
        auto *dq = dev(q), *dk0 = dev(k), *dv0 = dev(v);
        auto *dk4 = dev(std::vector<tqllm_cuda::Turbo4K256>(size_t(n) * kvh));
        auto *dv3 = dev(std::vector<tqllm_cuda::Turbo3V256>(size_t(n) * kvh));
        auto *dv4 = dev(std::vector<tqllm_cuda::Turbo4K256>(size_t(n) * kvh));
        auto *dy = dev(std::vector<bf16>(q.size()));
        auto *dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_mq_partial_floats(heads, nq, 8))));
        auto *dpos = dev(std::vector<int>{0}), *ds = dev(std::vector<int>{seq0});
        if (v4) {
            tqllm_cuda::cache_write_multi_t4kv(dk0, dv0, kvh * D, n, dk4, dv4, dpos, kvh, D, nullptr);
            tqllm_cuda::attention_mq256_t4kv(dq, dk4, dv4, dy, dp, ds, nq, heads, kvh, 1.f / 16.f, 8, nullptr);
        } else {
            tqllm_cuda::cache_write_multi_t4k_turbo3v(dk0, dv0, kvh * D, n, dk4, dv3, dpos, kvh, D, nullptr);
            tqllm_cuda::attention_mq256_t4k_turbo3v(dq, dk4, dv3, dy, dp, ds, nq, heads, kvh, 1.f / 16.f, 8, nullptr);
        }
        const auto k4 = host(dk4, size_t(n) * kvh);
        const auto v3 = host(dv3, size_t(n) * kvh);
        const auto v4r = host(dv4, size_t(n) * kvh);
        auto vnib = [](const tqllm_cuda::Turbo4K256 & r, int d) {  // V layout
            const int c = d / 32, o = d % 32, j = o / 8, l = o % 8;
            return int((r.w[4 * c + l / 2] >> (16 * (l % 2) + 4 * j)) & 15u);
        };
        const auto y = host(dy, q.size());
        auto nib = [](const tqllm_cuda::Turbo4K256 & r, int d) {
            const int kb = d / 32, o = d % 32, tg = o < 16 ? o / 4 : (o - 16) / 4, p = o < 16 ? o % 4 : 4 + o % 4;
            return int((r.w[kb * 4 + tg] >> (4 * p)) & 15u);
        };
        auto v3idx = [](const tqllm_cuda::Turbo3V256 & r, int d) {
            const int c = d / 32, l = d % 32 % 8, j = d % 32 / 8, bit = 12 * l + 3 * j;
            int code = 0;
            for (int b = 0; b < 3; ++b) code |= int((r.w[3 * c + (bit + b) / 32] >> ((bit + b) % 32)) & 1u) << b;
            return code >> 2 ? 4 + (code & 3) : 3 - (code & 3);
        };
        std::vector<float> kr(k.size()), vr(v.size());
        int code_bad = 0, scale_bad = 0;
        double kse = 0, ksr = 0;
        for (int t = 0; t < n; ++t)
            for (int kh = 0; kh < kvh; ++kh)
                for (int sub = 0; sub < 2; ++sub) {
                    const size_t in = (size_t(t) * kvh + kh) * D + sub * 128;
                    const auto & rk = k4[size_t(t) * kvh + kh];
                    const auto & rv = v3[size_t(t) * kvh + kh];
                    // CPU reference quantiser for K
                    float x[128], n2 = 0.f;
                    for (int j = 0; j < 128; ++j) n2 += (x[j] = __bfloat162float(k[in + j])) * x[j];
                    const float norm = std::sqrt(n2);
                    for (int j = 0; j < 128; ++j) x[j] = norm > 1e-10f ? x[j] / norm : 0.f;
                    fwht128(x, T3_S1, 1.f);
                    double rs = 0;
                    for (int j = 0; j < 128; ++j) {
                        const float r = x[j] * 0.08838834764831845f * T3_S2[j], a = std::fabs(r);
                        int rank = 0;
                        for (int b = 0; b < 7; ++b) rank += a >= B16[b];
                        code_bad += nib(rk, sub * 128 + j) != (((r < 0.f) << 3) | rank);
                        const double lv = MAG[rank] * (0.241529 / 127.0);
                        rs += lv * lv;
                    }
                    const float want = norm / float(std::sqrt(rs)) * (0.241529f / 127.f);
                    scale_bad += std::fabs(__half2float(rk.scale[sub]) - want) > 2e-3f * std::fabs(want) + 1e-7f;
                    // reconstruct K and V from the GPU records (inverse transform)
                    float kk[128], vv[128];
                    for (int j = 0; j < 128; ++j) {
                        const int c = nib(rk, sub * 128 + j);
                        kk[j] = float((c & 8) ? -MAG[c & 7] : MAG[c & 7]) * __half2float(rk.scale[sub]);
                        if (v4) {
                            const auto & r4 = v4r[size_t(t) * kvh + kh];
                            const int cv = vnib(r4, sub * 128 + j);
                            vv[j] = float((cv & 8) ? -MAG[cv & 7] : MAG[cv & 7]) * __half2float(r4.scale[sub]);
                        } else {
                            vv[j] = T3_C[v3idx(rv, sub * 128 + j)] * __half2float(rv.norm[sub]);
                        }
                    }
                    fwht128(kk, T3_S2, 0.08838834764831845f);
                    fwht128(vv, T3_S2, 0.08838834764831845f);
                    for (int j = 0; j < 128; ++j) {
                        kr[in + j] = kk[j] * T3_S1[j];
                        vr[in + j] = vv[j] * T3_S1[j];
                        const double e = kr[in + j] - __bfloat162float(k[in + j]);
                        kse += e * e;
                        ksr += double(__bfloat162float(k[in + j])) * __bfloat162float(k[in + j]);
                    }
                }
        double se = 0, sr = 0;
        for (int z = 0; z < nq; ++z)
            for (int h = 0; h < heads; ++h) {
                const int kh = h / (heads / kvh), hi = seq0 + z;
                std::vector<double> sc(hi);
                double mx = -1e300, sum = 0;
                for (int p = 0; p < hi; ++p) {
                    double a = 0;
                    for (int d = 0; d < D; ++d)
                        a += double(__bfloat162float(q[(size_t(z) * heads + h) * D + d])) * kr[(size_t(p) * kvh + kh) * D + d];
                    mx = std::max(mx, sc[p] = a / 16.0);
                }
                for (auto & a : sc) sum += (a = std::exp(a - mx));
                for (int d = 0; d < D; ++d) {
                    double ref = 0;
                    for (int p = 0; p < hi; ++p) ref += sc[p] / sum * vr[(size_t(p) * kvh + kh) * D + d];
                    const double e = ref - __bfloat162float(y[(size_t(z) * heads + h) * D + d]);
                    se += e * e;
                    sr += ref * ref;
                }
            }
        char d[128];
        std::snprintf(d, sizeof(d), "attention %.2e; K %.3f; code mismatches %d, scale %d", std::sqrt(se / sr),
                      std::sqrt(kse / ksr), code_bad, scale_bad);
        verdict(v4 ? "attn_t4kv" : "attn_t4k_turbo3v", std::sqrt(se / sr) < 1.2e-2 && std::sqrt(kse / ksr) < 0.12 && code_bad <= 4 && scale_bad == 0, d);
    }
    { // Turbo4 K+V prefill attention (one launch, FA-style) vs the verify kernel in 8-query groups
        const int heads = 24, kvh = 4, D = 256, nq = 300, n = 1000, s0 = n - nq + 1;
        std::vector<bf16> q(size_t(nq) * heads * D), k(size_t(n) * kvh * D), v(k.size());
        for (auto & x : q) x = __float2bfloat16_rn(nd(g));
        for (auto & x : k) x = __float2bfloat16_rn(nd(g) * 2.f);
        for (auto & x : v) x = __float2bfloat16_rn(nd(g));
        auto *dq = dev(q), *dk0 = dev(k), *dv0 = dev(v);
        auto *dk4 = dev(std::vector<tqllm_cuda::Turbo4K256>(size_t(n) * kvh));
        auto *dv4 = dev(std::vector<tqllm_cuda::Turbo4K256>(size_t(n) * kvh));
        auto *dy = dev(std::vector<bf16>(q.size())), *dr = dev(std::vector<bf16>(q.size()));
        auto *dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_mq_partial_floats(heads, nq, 8))));
        auto *dpos = dev(std::vector<int>{0}), *ds = dev(std::vector<int>{s0});
        tqllm_cuda::cache_write_multi_t4kv(dk0, dv0, kvh * D, n, dk4, dv4, dpos, kvh, D, nullptr);
        for (int off = 0; off < nq; off += 8)
            tqllm_cuda::attention_mq256_t4kv(dq + size_t(off) * heads * D, dk4, dv4, dr + size_t(off) * heads * D, dp, ds,
                                            std::min(8, nq - off), heads, kvh, 1.f / 16.f, 8, nullptr, off);
        tqllm_cuda::attention_mq256_t4kv_prefill(dq, dk4, dv4, dy, dp, ds, nq, heads, kvh, 1.f / 16.f, 1, nullptr);
        const auto y = host(dy, q.size()), r = host(dr, q.size());
        double se = 0, sr = 0, worst = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            const double a = __bfloat162float(y[i]), b = __bfloat162float(r[i]);
            se += (a - b) * (a - b), sr += b * b, worst = std::max(worst, std::fabs(a - b));
        }
        char d[96];
        std::snprintf(d, sizeof(d), "vs verify kernel %.2e (max abs %.3f)", std::sqrt(se / sr), worst);
        verdict("attn_t4kv_prefill", std::isfinite(se) && std::sqrt(se / sr) < 1e-2, d);
    }
    { // TQ3_0 K + V: writer + attention vs BF16 attention on fake-quantised (TQ3_0) K / V
        const int heads = 24, kvh = 4, D = 256, n = 1000;
        std::vector<bf16> k(size_t(n) * kvh * D), v(k.size());
        for (auto & x : k) x = __float2bfloat16_rn(nd(g) * 2.f);
        for (auto & x : v) x = __float2bfloat16_rn(nd(g));
        auto *dk = dev(k), *dv = dev(v);
        auto *d3k = dev(std::vector<tqllm_cuda::Tq3KV256>(size_t(n) * kvh)), *d3v = dev(std::vector<tqllm_cuda::Tq3KV256>(size_t(n) * kvh));
        auto *dpos = dev(std::vector<int>{0});
        tqllm_cuda::cache_write_multi_tq3kv(dk, dv, kvh * D, n, d3k, d3v, dpos, kvh, D, nullptr);
        tqllm_cuda::kv_fakequant(dk, kvh * D, n, kvh, D, 3, nullptr);  // reference K / V (in place)
        tqllm_cuda::kv_fakequant(dv, kvh * D, n, kvh, D, 3, nullptr);
        for (int shape = 0; shape < 2; ++shape) {
            const int nq = shape ? 300 : 8, s0 = n - nq + 1, ns = shape ? 1 : 8;
            std::vector<bf16> q(size_t(nq) * heads * D);
            for (auto & x : q) x = __float2bfloat16_rn(nd(g));
            auto *dq = dev(q), *dy = dev(std::vector<bf16>(q.size())), *dr = dev(std::vector<bf16>(q.size()));
            auto *dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_mq_partial_floats(heads, nq, 8))));
            auto *ds = dev(std::vector<int>{s0});
            for (int off = 0; off < nq; off += 8)
                tqllm_cuda::attention_mq256(dq + size_t(off) * heads * D, dk, dv, dr + size_t(off) * heads * D, dp, ds,
                                            std::min(8, nq - off), heads, kvh, 1.f / 16.f, 8, nullptr, off);
            tqllm_cuda::attention_tq3kv(dq, d3k, d3v, dy, dp, ds, nq, heads, kvh, 1.f / 16.f, ns, nullptr);
            const auto y = host(dy, q.size()), r = host(dr, q.size());
            double se = 0, sr = 0, worst = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double a2 = __bfloat162float(y[i]), b2 = __bfloat162float(r[i]);
                se += (a2 - b2) * (a2 - b2), sr += b2 * b2, worst = std::max(worst, std::fabs(a2 - b2));
            }
            char d[96];
            std::snprintf(d, sizeof(d), "vs fake-quant BF16 attention %.2e (max abs %.3f)", std::sqrt(se / sr), worst);
            verdict(shape ? "attn_tq3kv_prefill" : "attn_tq3kv_verify", std::isfinite(se) && std::sqrt(se / sr) < 1.5e-2, d);
        }
    }
    { // TQ3 K + 2-bit V: the K path is shared; compare attention against the established TQ3 K/V path.
        const int heads = 24, kvh = 4, D = 256, n = 1000;
        std::vector<bf16> k(size_t(n) * kvh * D), v(k.size());
        for (auto & x : k) x = __float2bfloat16_rn(nd(g) * 2.f);
        for (auto & x : v) x = __float2bfloat16_rn(nd(g));
        auto *dk = dev(k), *dv = dev(v);
        auto *d3k = dev(std::vector<tqllm_cuda::Tq3KV256>(size_t(n) * kvh));
        auto *d3v = dev(std::vector<tqllm_cuda::Tq3KV256>(size_t(n) * kvh));
        auto *d2v = dev(std::vector<tqllm_cuda::Tq2V256>(size_t(n) * kvh));
        auto *dpos = dev(std::vector<int>{0});
        tqllm_cuda::cache_write_multi_tq3kv(dk, dv, kvh * D, n, d3k, d3v, dpos, kvh, D, nullptr);
        tqllm_cuda::cache_write_multi_tq3k_tq2v(dk, dv, kvh * D, n, d3k, d2v, dpos, kvh, D, nullptr);
        for (int shape = 0; shape < 2; ++shape) {
            const int nq = shape ? 300 : 8, s0 = n - nq + 1, ns = shape ? 1 : 8;
            std::vector<bf16> q(size_t(nq) * heads * D);
            for (auto & x : q) x = __float2bfloat16_rn(nd(g));
            auto *dq = dev(q), *dy3 = dev(std::vector<bf16>(q.size())), *dy2 = dev(std::vector<bf16>(q.size()));
            auto *dp = dev(std::vector<float>(size_t(tqllm_cuda::attention_mq_partial_floats(heads, nq, 8))));
            auto *ds = dev(std::vector<int>{s0});
            tqllm_cuda::attention_tq3kv(dq, d3k, d3v, dy3, dp, ds, nq, heads, kvh, 1.f / 16.f, ns, nullptr);
            tqllm_cuda::attention_tq3k_tq2v(dq, d3k, d2v, dy2, dp, ds, nq, heads, kvh, 1.f / 16.f, ns, nullptr);
            const auto y3 = host(dy3, q.size()), y2 = host(dy2, q.size());
            double se = 0, sr = 0;
            for (size_t i = 0; i < y3.size(); ++i) {
                const double a = __bfloat162float(y2[i]), b = __bfloat162float(y3[i]);
                se += (a - b) * (a - b), sr += b * b;
            }
            const double rel = std::sqrt(se / sr);
            char d[96];
            std::snprintf(d, sizeof(d), "vs TQ3 K/V %.2e relative RMS", rel);
            verdict(shape ? "attn_tq3k_tq2v_prefill" : "attn_tq3k_tq2v_verify", std::isfinite(rel) && rel < 0.4, d);
        }
    }
    { // TQ4: matvec (1..8 columns) + dequant vs CPU, rotation vs explicit Hadamard
        const int rows = 200, k = 512, ng = k / 128;
        const int LV[16] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
        std::vector<uint8_t> codes(size_t(rows) * k / 2);
        for (auto & c : codes) c = uint8_t(g() & 0xFF);
        std::vector<__half> sc(size_t(rows) * ng);
        for (auto & x : sc) x = __float2half(0.001f + 0.01f * float(g() % 1000) / 1000.f);
        std::vector<__half> xh(size_t(8) * k);
        for (auto & x : xh) x = __float2half(nd(g));
        auto *dc = dev(codes);
        auto *dxh = dev(xh);
        auto *ds = dev(sc);
        auto *dy = dev(std::vector<float>(size_t(8) * rows));
        auto *dw = dev(std::vector<__half>(size_t(rows) * k));
        double worst = 0;
        for (int nc = 1; nc <= 8; ++nc) {
            tqllm_cuda::tq4_matvec(dc, ds, dxh, dy, rows, k, nc, nullptr);
            const auto y = host(dy, size_t(nc) * rows);
            for (int n = 0; n < nc; ++n)
                for (int r = 0; r < rows; ++r) {
                    double ref = 0, mag = 0;
                    for (int i = 0; i < k; ++i) {
                        const int code = (codes[(size_t(r) * k + i) / 2] >> ((i & 1) * 4)) & 15;
                        const double w = double(__half2float(__float2half(float(LV[code]) * __half2float(sc[size_t(r) * ng + i / 128]))));
                        ref += w * __half2float(xh[size_t(n) * k + i]);
                        mag += std::fabs(w * __half2float(xh[size_t(n) * k + i]));
                    }
                    worst = std::max(worst, std::fabs(y[size_t(n) * rows + r] - ref) / (mag + 1e-9));
                }
        }
        tqllm_cuda::tq4_dequant_h(dc, ds, dw, rows, k, nullptr);
        const auto wd = host(dw, size_t(rows) * k);
        int dq_bad = 0;
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < k; ++i) {
                const int code = (codes[(size_t(r) * k + i) / 2] >> ((i & 1) * 4)) & 15;
                const float w = __half2float(__float2half(float(LV[code]) * __half2float(sc[size_t(r) * ng + i / 128])));
                dq_bad += std::fabs(__half2float(wd[size_t(r) * k + i]) - w) > 1e-3f * std::fabs(w) + 1e-6f;
            }
        // rotation: x' = H (s1 * x) / sqrt(128) per 128
        std::vector<bf16> xb(size_t(2) * k);
        for (auto & x : xb) x = __float2bfloat16_rn(nd(g));
        auto *dxb = dev(xb);
        tqllm_cuda::tq4_rotate_h(dxb, dxh, k, 2, nullptr);
        const auto xr = host(dxh, size_t(2) * k);
        const uint32_t S1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
        double rot_err = 0;
        for (int n = 0; n < 2; ++n)
            for (int gq = 0; gq < k / 128; ++gq)
                for (int i = 0; i < 128; ++i) {
                    double acc = 0;
                    for (int j = 0; j < 128; ++j) {
                        const double sj = (S1[j / 32] >> (j % 32)) & 1u ? -1.0 : 1.0;
                        acc += (__builtin_popcount(unsigned(i & j)) & 1 ? -1.0 : 1.0) * sj * __bfloat162float(xb[size_t(n) * k + gq * 128 + j]);
                    }
                    rot_err = std::max(rot_err, std::fabs(acc / std::sqrt(128.0) - __half2float(xr[size_t(n) * k + gq * 128 + i])));
                }
        char d[128];
        std::snprintf(d, sizeof(d), "matvec max rel %.2e, dequant mismatches %d, rotation max abs %.2e", worst, dq_bad, rot_err);
        verdict("tq4_kernels", worst < 5e-3 && dq_bad == 0 && rot_err < 1e-2, d);
    }
    { // lattice
        const int n = 8, rank = 256, vocab = 1000;
        std::vector<uint32_t> cand(size_t(n) * 16);
        for (auto & c : cand) c = g() % vocab;
        std::vector<float> unary(size_t(n) * 16);
        for (auto & u : unary) u = nd(g);
        std::vector<bf16> gate(size_t(n) * rank), nx(size_t(vocab) * rank), pv(nx.size());
        for (auto & v : gate) v = __float2bfloat16_rn(nd(g));
        for (auto & v : nx) v = __float2bfloat16_rn(nd(g) * 0.1f);
        for (auto & v : pv) v = __float2bfloat16_rn(nd(g) * 0.1f);
        const uint32_t anchor = 7;
        auto * dc = dev(cand);
        auto * du = dev(unary);
        auto *dg = dev(gate), *dn = dev(nx), *dpv = dev(pv);
        auto * ds = dev(std::vector<float>(size_t(n) * 256, 0.f));
        tqllm_cuda::dflash_lattice(dc, du, dg, rank, dn, dpv, anchor, n, rank, ds, nullptr);
        const auto got = host(ds, size_t(n) * 256);
        double worst = 0;
        for (int i = 1; i < n; ++i)
            for (int j = 0; j < 16; ++j)
                for (int k = 0; k < 16; ++k) {
                    const uint32_t p = i == 1 ? anchor : cand[(i - 1) * 16 + j];
                    double a = unary[i * 16 + k];
                    for (int r = 0; r < rank; ++r)
                        a += double(__bfloat162float(nx[size_t(cand[i * 16 + k]) * rank + r])) *
                             __bfloat162float(pv[size_t(p) * rank + r]) * __bfloat162float(gate[size_t(i) * rank + r]);
                    worst = std::fmax(worst, std::fabs(a - got[(size_t(i) * 16 + j) * 16 + k]));
                }
        char d[64];
        std::snprintf(d, sizeof(d), "max abs %.2e", worst);
        verdict("dflash_lattice", worst < 1e-3, d);
    }
    { // device greedy chain == the host chain in dflash_draft_batch (first maximum on ties)
        const int B = 16, want = 15;
        std::vector<uint32_t> idx(size_t(B) * 16);
        for (auto & c : idx) c = g() % 248320;
        std::vector<float> score(size_t(B) * 256);
        for (auto & v : score) v = float(int(nd(g) * 2));  // coarse values: ties occur
        auto * di = dev(idx);
        auto * ds = dev(score);
        auto * dt = dev(std::vector<uint32_t>(size_t(want), 0u));
        tqllm_cuda::dflash_chain(di, ds, want, dt, nullptr);
        const auto got = host(dt, size_t(want));
        std::vector<uint32_t> ref;
        for (int i = 1, pred = 0; i <= want; ++i) {
            const float * row = score.data() + (size_t(i) * 16 + pred) * 16;
            pred = int(std::max_element(row, row + 16) - row);
            ref.push_back(idx[size_t(i) * 16 + pred]);
        }
        verdict("dflash_chain", got == ref, "vs host chain");
    }
    std::printf("%s\n", failures ? "DFLASH TESTS FAILED" : "all dflash tests passed");
    return failures ? 1 : 0;
}

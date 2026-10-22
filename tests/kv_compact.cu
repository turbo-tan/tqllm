// GPU test: TqCache stand-in segment in the tq3 attention kernels (KvCompactSeg). The main kernels skip cache positions
// below `start` and one extra partial covers the stand-in rows (keys in the original basis, values in the TQ3 rotated
// basis, per-row log-weight bias). Checked for decode (1 query), verify (12), a prefill chunk (300) and a tree verify:
//   identity  stand-in = the cache's own decoded rows [0, C), bias 0  -> same output as plain attention
//   bias      every stand-in row twice with bias -ln 2                 -> same output again (mass kept)
//   general   random stand-in rows and biases                          -> matches an FP32 host reference
#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

using namespace tqllm_cuda;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

constexpr int H = 24, KVH = 4, D = 256, G = H / KVH;
static const int SIGNS[32] = {1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1};
static const float LEV[8] = {-127, -82, -47, -16, 15, 46, 81, 127};
static const float F = 127.f / 1.996684f;

static void wht32(float * x) {  // unnormalised Walsh-Hadamard over 32 values, as the cache writer
    for (int m = 1; m < 32; m <<= 1)
        for (int i = 0; i < 32; ++i)
            if (!(i & m)) { const float a = x[i], b = x[i + m]; x[i] = a + b, x[i + m] = a - b; }
}
// decode one Tq3KV256 record (one KV head of one position) to its 256 values in the rotated basis
static void decode_rot(const Tq3KV256 & r, float * out) {
    for (int b = 0; b < 8; ++b) {
        const float rms = __half2float(r.rms[b]);
        for (int lane = 0; lane < 32; ++lane) {
            const int bit = 3 * lane;
            uint64_t lo = uint64_t(r.w[3 * b]) | (uint64_t(r.w[3 * b + 1]) << 32);
            uint32_t idx;
            if (bit + 3 <= 64) idx = uint32_t(lo >> bit) & 7u;
            else if (bit >= 64) idx = (r.w[3 * b + 2] >> (bit - 64)) & 7u;
            else idx = uint32_t((lo >> bit) | (uint64_t(r.w[3 * b + 2]) << (64 - bit))) & 7u;
            out[b * 32 + lane] = LEV[idx] / F * rms;
        }
    }
}
static void rot_to_orig(const float * r, float * o) {  // R^T: signs * WHT / sqrt(32), per 32
    for (int b = 0; b < 8; ++b) {
        float x[32];
        for (int i = 0; i < 32; ++i) x[i] = r[b * 32 + i];
        wht32(x);
        for (int i = 0; i < 32; ++i) o[b * 32 + i] = x[i] / std::sqrt(32.f) * SIGNS[i];
    }
}
static void orig_to_rot(const float * o, float * r) {  // R: WHT(signs * x) / sqrt(32)
    for (int b = 0; b < 8; ++b) {
        float x[32];
        for (int i = 0; i < 32; ++i) x[i] = o[b * 32 + i] * SIGNS[i];
        wht32(x);
        for (int i = 0; i < 32; ++i) r[b * 32 + i] = x[i] / std::sqrt(32.f);
    }
}

static void q8_host(const float * x, KvStandIn256 & r) {  // as the compactor stores a stand-in row
    for (int b = 0; b < 8; ++b) {
        float a = 0.f;
        for (int i = 0; i < 32; ++i) a = std::max(a, std::fabs(x[b * 32 + i]));
        const float s = a > 0.f ? a / 127.f : 1.f;
        r.s[b] = __float2half_rn(s);
        const float sh = __half2float(r.s[b]);
        for (int i = 0; i < 32; ++i) r.q[b * 32 + i] = int8_t(std::lrint(std::max(-127.f, std::min(127.f, x[b * 32 + i] / s))));
        (void)sh;
    }
}
static void q8_dequant(const KvStandIn256 & r, float * x) {
    for (int i = 0; i < 256; ++i) x[i] = float(r.q[i]) * __half2float(r.s[i / 32]);
}

static double rel_rms(const std::vector<float> & a, const std::vector<float> & b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) e += (a[i] - b[i]) * (a[i] - b[i]), n += b[i] * b[i];
    return std::sqrt(e / std::max(n, 1e-30));
}

int main() {
    const int N = 4096, C = 1024;  // cache positions, stand-in boundary (multiple of 32)
    // the thread's default stream: the synchronous cudaMemcpy calls below are ordered with these kernels (a created
    // blocking stream no longer synchronizes with stream 0 now that it is per thread)
    cudaStream_t st = cudaStreamPerThread;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    std::mt19937 g(11);
    std::normal_distribution<float> nd;
    // K / V for every position, written through the real cache writer
    const int ts = 2 * KVH * D;
    std::vector<__nv_bfloat16> hkv(size_t(N) * ts);
    for (auto & x : hkv) x = __float2bfloat16(nd(g));
    __nv_bfloat16 * dkv;
    Tq3KV256 *kc, *vc;
    int * dpos;
    CK(cudaMalloc(&dkv, hkv.size() * 2));
    CK(cudaMalloc(&kc, size_t(N) * KVH * sizeof(Tq3KV256)));
    CK(cudaMalloc(&vc, size_t(N) * KVH * sizeof(Tq3KV256)));
    CK(cudaMalloc(&dpos, 8));
    CK(cudaMemcpy(dkv, hkv.data(), hkv.size() * 2, cudaMemcpyHostToDevice));
    for (int p = 0; p < N; p += 512) {
        const int hp[2] = {p, p + 1};
        CK(cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice));
        cache_write_multi_tq3kv(dkv + size_t(p) * ts, dkv + size_t(p) * ts + KVH * D, ts, 512, kc, vc, dpos, KVH, D, st);
    }
    CK(cudaStreamSynchronize(st));
    std::vector<Tq3KV256> hk(size_t(N) * KVH), hv(hk.size());
    CK(cudaMemcpy(hk.data(), kc, hk.size() * sizeof(Tq3KV256), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hv.data(), vc, hv.size() * sizeof(Tq3KV256), cudaMemcpyDeviceToHost));
    // decoded cache: keys in the original basis, values in the rotated basis (what a stand-in stores)
    std::vector<float> kd(size_t(N) * KVH * D), vd(kd.size());
    for (size_t i = 0; i < hk.size(); ++i) {
        float r[D];
        decode_rot(hk[i], r);
        rot_to_orig(r, &kd[i * D]);
        decode_rot(hv[i], &vd[i * D]);
    }
    const int maxq = 300, maxsplit = 2 * sms + 2;
    __nv_bfloat16 *q, *y;
    float * part;
    CK(cudaMalloc(&q, size_t(maxq) * H * D * 2));
    CK(cudaMalloc(&y, size_t(maxq) * H * D * 2));
    CK(cudaMalloc(&part, size_t(attention_mq_partial_floats(H, maxq, maxsplit)) * 4));
    std::vector<__nv_bfloat16> hq(size_t(maxq) * H * D);
    for (auto & x : hq) x = __float2bfloat16(nd(g) * 0.5f);
    CK(cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    // stand-in buffers
    const int RMAX = 2 * C;
    KvStandIn256 *ck, *cv;
    float * cb;
    int *cn, *cstart;
    CK(cudaMalloc(&ck, size_t(RMAX) * KVH * sizeof(KvStandIn256)));
    CK(cudaMalloc(&cv, size_t(RMAX) * KVH * sizeof(KvStandIn256)));
    CK(cudaMalloc(&cb, size_t(RMAX) * KVH * 4));
    CK(cudaMalloc(&cn, 4));
    CK(cudaMalloc(&cstart, 4));
    CK(cudaMemcpy(cstart, &C, 4, cudaMemcpyHostToDevice));
    // quantises stand-in rows to Q8 (as stored) and returns the dequantised values the kernel sees, for the reference
    // as the compactor stores them: keys rotated to the TQ3_0 basis, then Q8; the reference keys are the stored bytes
    // decoded back to the original basis. Rows from `rows` up to RMAX hold NaN scales and biases: reading past n shows.
    auto upload_seg = [&](std::vector<float> & k, std::vector<float> & v, const std::vector<float> & b, int rows) {
        std::vector<KvStandIn256> hk2(size_t(RMAX) * KVH), hv2(hk2.size());
        std::vector<float> hb(size_t(RMAX) * KVH, std::numeric_limits<float>::quiet_NaN());
        for (size_t i = 0; i < hk2.size(); ++i) {
            if (i < size_t(rows) * KVH) {
                float r[D], o[D];
                orig_to_rot(&k[i * D], r);
                q8_host(r, hk2[i]), q8_host(&v[i * D], hv2[i]);
                q8_dequant(hk2[i], o), rot_to_orig(o, &k[i * D]);
                q8_dequant(hv2[i], &v[i * D]);
                hb[i] = b[i];
            } else {
                for (int j = 0; j < 8; ++j) hk2[i].s[j] = hv2[i].s[j] = __float2half(std::numeric_limits<float>::quiet_NaN());
                std::fill(hk2[i].q, hk2[i].q + D, int8_t(1)), std::fill(hv2[i].q, hv2[i].q + D, int8_t(1));
            }
        }
        cudaMemcpy(ck, hk2.data(), hk2.size() * sizeof(KvStandIn256), cudaMemcpyHostToDevice);
        cudaMemcpy(cv, hv2.data(), hv2.size() * sizeof(KvStandIn256), cudaMemcpyHostToDevice);
        cudaMemcpy(cb, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(cn, &rows, 4, cudaMemcpyHostToDevice);
    };
    float * splitbuf;
    CK(cudaMalloc(&splitbuf, kv_compact_split_floats(maxq, H) * sizeof(float)));
    KvCompactSeg seg{cstart, ck, cv, cb, cn, splitbuf, maxq * H * 32};
    const float scale = 1.f / 16.f;
    auto run = [&](int nq, const KvCompactSeg * cs, const int * par = nullptr) {
        const int hp[2] = {N - nq, N - nq + 1};  // queries at the last nq positions
        cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
        const int split = std::clamp(2 * sms / KVH, 1, 256);
        attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, q, kc, vc, y, part, dpos + 1, nq, H, KVH, scale, nq > 12 ? 1 : split, st,
                           par, nullptr, cs);
        std::vector<__nv_bfloat16> o(size_t(nq) * H * D);
        cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<float> f(o.size());
        for (size_t i = 0; i < o.size(); ++i) f[i] = __bfloat162float(o[i]);
        return f;
    };
    // FP32 host reference: stand-in rows (k, v, bias) + decoded cache rows [C, pos]
    auto reference = [&](int nq, const std::vector<float> & sk, const std::vector<float> & sv, const std::vector<float> & sb, int rows) {
        std::vector<float> out(size_t(nq) * H * D);
        for (int z = 0; z < nq; ++z) {
            const int last = N - nq + z;
            for (int h = 0; h < H; ++h) {
                const int kh = h / G;
                const __nv_bfloat16 * qr = &hq[(size_t(z) * H + h) * D];
                std::vector<double> lg;
                std::vector<const float *> vr;
                for (int j = 0; j < rows; ++j) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += __bfloat162float(qr[i]) * sk[(size_t(j) * KVH + kh) * D + i];
                    lg.push_back(d * scale + sb[size_t(j) * KVH + kh]);
                    vr.push_back(&sv[(size_t(j) * KVH + kh) * D]);
                }
                for (int p = C; p <= last; ++p) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += __bfloat162float(qr[i]) * kd[(size_t(p) * KVH + kh) * D + i];
                    lg.push_back(d * scale);
                    vr.push_back(&vd[(size_t(p) * KVH + kh) * D]);
                }
                const double m = *std::max_element(lg.begin(), lg.end());
                double l = 0;
                std::vector<double> acc(D, 0.0);
                for (size_t j = 0; j < lg.size(); ++j) {
                    const double w = std::exp(lg[j] - m);
                    l += w;
                    for (int i = 0; i < D; ++i) acc[i] += w * vr[j][i];
                }
                float rot[D], orig[D];
                for (int i = 0; i < D; ++i) rot[i] = float(acc[i] / l);
                rot_to_orig(rot, orig);  // the kernels' inverse rotation of the output
                for (int i = 0; i < D; ++i) out[(size_t(z) * H + h) * D + i] = orig[i];
            }
        }
        return out;
    };
    // identity stand-in: the cache's own decoded rows [0, C)
    std::vector<float> ik(kd.begin(), kd.begin() + size_t(C) * KVH * D), iv(vd.begin(), vd.begin() + size_t(C) * KVH * D);
    std::vector<float> ib(size_t(C) * KVH, 0.f);
    // the same rows twice, each with log weight -ln 2
    std::vector<float> dk(size_t(2 * C) * KVH * D), dv(dk.size()), db(size_t(2 * C) * KVH, -std::log(2.f));
    for (int j = 0; j < 2 * C; ++j)
        for (int i = 0; i < KVH * D; ++i)
            dk[size_t(j) * KVH * D + i] = ik[size_t(j % C) * KVH * D + i], dv[size_t(j) * KVH * D + i] = iv[size_t(j % C) * KVH * D + i];
    // random stand-in: 300 rows, keys at the cache's scale, values in the rotated basis, biases in [-2, 1]
    const int RR = 300;
    std::vector<float> rk(size_t(RR) * KVH * D), rv(rk.size()), rb(size_t(RR) * KVH);
    for (auto & x : rk) x = nd(g);
    for (size_t j = 0; j < size_t(RR) * KVH; ++j) {
        float o[D], r[D];
        for (int i = 0; i < D; ++i) o[i] = nd(g);
        orig_to_rot(o, r);
        std::copy(r, r + D, &rv[j * D]);
        rb[j] = -2.f + 3.f * float(g() % 1000) / 1000.f;
    }
    for (int nq : {1, 12, 300}) {
        const auto plain = run(nq, nullptr);
        upload_seg(ik, iv, ib, C);
        const auto ident = run(nq, &seg);
        upload_seg(dk, dv, db, 2 * C);
        const auto twice = run(nq, &seg);
        upload_seg(rk, rv, rb, RR);
        const auto gen = run(nq, &seg);
        const auto ref = reference(nq, rk, rv, rb, RR);
        const double e1 = rel_rms(ident, plain), e2 = rel_rms(twice, plain), e3 = rel_rms(gen, ref);
        std::printf("kv_compact nq %3d: identity %.2e, bias %.2e, general vs FP32 reference %.2e\n", nq, e1, e2, e3);
        CHECK(e1 < 2e-2, "nq %d: identity stand-in differs from plain attention (%.3e)", nq, e1);
        CHECK(e2 < 2e-2, "nq %d: duplicated rows with -ln 2 bias differ (%.3e)", nq, e2);
        CHECK(e3 < 2e-2, "nq %d: general stand-in differs from the FP32 reference (%.3e)", nq, e3);
    }
    {  // tree verify (a chain of 8 as a tree) with a stand-in: same as the non-tree verify of the same queries
        std::vector<int> hpar(8);
        for (int i = 0; i < 8; ++i) hpar[i] = i - 1;
        int * par;
        CK(cudaMalloc(&par, 32));
        CK(cudaMemcpy(par, hpar.data(), 32, cudaMemcpyHostToDevice));
        upload_seg(rk, rv, rb, RR);
        const auto tree = run(8, &seg, par), flat = run(8, &seg);
        const double e = rel_rms(tree, flat);
        std::printf("kv_compact tree chain 8 vs verify: %.2e\n", e);
        CHECK(e < 2e-2, "tree verify with a stand-in differs from verify (%.3e)", e);
        cudaFree(par);
    }
    {  // speed: a stand-in segment of `rows` rows vs plain attention over as many cache positions (12-query verify)
        const int rows = 4000, nq = 12;  // 4000 x 8 = 32K old positions per stand-in
        std::vector<float> sk(size_t(rows) * KVH * D), sv(sk.size()), sb(size_t(rows) * KVH, 0.f);
        for (auto & x : sk) x = nd(g) * 0.5f;
        for (auto & x : sv) x = nd(g) * 0.5f;
        KvStandIn256 *bk, *bv;
        float * bb;
        int * bn;
        CK(cudaMalloc(&bk, size_t(rows) * KVH * sizeof(KvStandIn256)));
        CK(cudaMalloc(&bv, size_t(rows) * KVH * sizeof(KvStandIn256)));
        CK(cudaMalloc(&bb, sb.size() * 4));
        CK(cudaMalloc(&bn, 4));
        std::vector<KvStandIn256> t1(size_t(rows) * KVH), t2(t1.size());
        for (size_t i = 0; i < t1.size(); ++i) {
            float r[D];
            orig_to_rot(&sk[i * D], r);
            q8_host(r, t1[i]), q8_host(&sv[i * D], t2[i]);
        }
        CK(cudaMemcpy(bk, t1.data(), t1.size() * sizeof(KvStandIn256), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bv, t2.data(), t2.size() * sizeof(KvStandIn256), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bb, sb.data(), sb.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bn, &rows, 4, cudaMemcpyHostToDevice));
        const int start = N - 32;  // the cache part is one tile: the time is the stand-in's
        CK(cudaMemcpy(cstart, &start, 4, cudaMemcpyHostToDevice));
        KvCompactSeg sseg{cstart, bk, bv, bb, bn, splitbuf, maxq * H * 32};
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0), cudaEventCreate(&e1);
        auto timeit = [&](const KvCompactSeg * cs) {
            const int hp[2] = {N - nq, N - nq + 1};
            cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
            const int split = std::clamp(2 * sms / KVH, 1, 256);
            for (int w = 0; w < 3; ++w) attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, q, kc, vc, y, part, dpos + 1, nq, H, KVH, scale, split, st, nullptr, nullptr, cs);
            cudaEventRecord(e0, st);
            for (int it = 0; it < 20; ++it) attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, q, kc, vc, y, part, dpos + 1, nq, H, KVH, scale, split, st, nullptr, nullptr, cs);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            return ms / 20;
        };
        const float t_seg = timeit(&sseg), t_plain = timeit(nullptr);  // plain: all N = 4096 cache positions
        std::printf("kv_compact speed (12 queries, one layer): %d stand-in rows %.3f ms | %d tq3 cache positions %.3f ms "
                    "(%.1f us per 1K rows vs %.1f us per 1K positions)\n", rows, t_seg, N, t_plain, 1000 * t_seg / rows * 1000 / 1000,
                    1000 * t_plain / N * 1000 / 1000);
        cudaFree(bk), cudaFree(bv), cudaFree(bb), cudaFree(bn);
        CK(cudaMemcpy(cstart, &C, 4, cudaMemcpyHostToDevice));
    }
    {  // the GPU compactor: the block [0, C) of the cache -> stand-in rows; queries from a ring; then attention with the
       // stand-in vs plain attention for held-out queries. Ratio 1 (every key kept) must reproduce attention closely.
        const int R = 256, nb = C;
        __nv_bfloat16 * ring;
        CK(cudaMalloc(&ring, size_t(R) * H * D * 2));
        std::vector<__nv_bfloat16> hr(size_t(R) * H * D);
        for (auto & x : hr) x = __float2bfloat16(nd(g) * 0.5f);
        CK(cudaMemcpy(ring, hr.data(), hr.size() * 2, cudaMemcpyHostToDevice));
        cublasHandle_t blas;
        cusolverDnHandle_t sol;
        cublasCreate(&blas);
        cusolverDnCreate(&sol);
        void * scratch;
        CK(cudaMalloc(&scratch, kv_compact_scratch_bytes(nb, R * G, nb, KVH)));
        for (int ratio : {1, 8}) {
            const int m = nb / ratio;
            kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, nb, KVH, H, ring, R, scale, m, 0, ck, cv, cb, scratch, blas, sol, st);
            CK(cudaMemcpy(cn, &m, 4, cudaMemcpyHostToDevice));
            CK(cudaStreamSynchronize(st));
            const auto plain = run(12, nullptr), comp = run(12, &seg);
            const double e = rel_rms(comp, plain);
            std::printf("kv_compact compactor %dx (%d of %d rows): attention vs exact %.3e (random queries)\n", ratio, m, nb, e);
            if (ratio == 1) CHECK(e < 3e-2, "compactor at ratio 1 does not reproduce attention (%.3e)", e);
        }
        {  // a short turn: fewer query rows than stand-in rows (16 ring rows x 6 heads = 96 < m). Scored on the fit's own
           // queries, ratio 1 must reproduce attention; values written over their own inputs (C sharing O's space) do not
            const int Rs = 16;
            __nv_bfloat16 * qr;
            CK(cudaMalloc(&qr, size_t(Rs) * H * D * 2));
            CK(cudaMemcpy(qr, ring, size_t(Rs) * H * D * 2, cudaMemcpyDeviceToDevice));
            auto run_r = [&](const KvCompactSeg * cs) {
                const int hp[2] = {N - Rs, N - Rs + 1};
                cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
                attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, qr, kc, vc, y, part, dpos + 1, Rs, H, KVH, scale, 1, st, nullptr, nullptr, cs);
                std::vector<__nv_bfloat16> o(size_t(Rs) * H * D);
                cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
                std::vector<float> f(o.size());
                for (size_t k = 0; k < o.size(); ++k) f[k] = __bfloat162float(o[k]);
                return f;
            };
            for (int ratio : {1, 8}) {
                const int m = nb / ratio;
                kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, nb, KVH, H, ring, Rs, scale, m, 0, ck, cv, cb, scratch, blas, sol, st);
                CK(cudaMemcpy(cn, &m, 4, cudaMemcpyHostToDevice));
                CK(cudaStreamSynchronize(st));
                const double e = rel_rms(run_r(&seg), run_r(nullptr));
                std::printf("kv_compact compactor %dx, %d ring rows (nq %d < m %d): fit queries vs exact %.3e\n", ratio, Rs, Rs * G, m, e);
                if (ratio == 1) CHECK(std::isfinite(e) && e < 1e-2, "short-turn compaction corrupts the stand-ins (%.3e)", e);
            }
            cudaFree(qr);
        }
        cudaFree(scratch), cudaFree(ring);
        cublasDestroy(blas);
        cusolverDnDestroy(sol);
    }
    {  // trellis KV (tcq3 K with tq3 or tcq2 V): the same K / V through the trellis writers, compactor at ratio 1 and 8,
       // attention with the stand-in segment vs plain attention over the same records
        const int R = 256, nb = C;
        __nv_bfloat16 * ring;
        CK(cudaMalloc(&ring, size_t(R) * H * D * 2));
        std::vector<__nv_bfloat16> hr(size_t(R) * H * D);
        for (auto & x : hr) x = __float2bfloat16(nd(g) * 0.5f);
        CK(cudaMemcpy(ring, hr.data(), hr.size() * 2, cudaMemcpyHostToDevice));
        cublasHandle_t blas;
        cusolverDnHandle_t sol;
        cublasCreate(&blas);
        cusolverDnCreate(&sol);
        void * scratch;
        CK(cudaMalloc(&scratch, kv_compact_scratch_bytes(nb, R * G, nb, KVH)));
        void *tk, *tv;
        CK(cudaMalloc(&tk, size_t(N) * KVH * sizeof(Tcq3K256)));
        CK(cudaMalloc(&tv, size_t(N) * KVH * sizeof(Tcq2V256)));
        for (int p = 0; p < N; p += 512) {
            const int hp[2] = {p, p + 1};
            CK(cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice));
            cache_write_kv_side(KvCodec::Tcq3, dkv + size_t(p) * ts, ts, 512, tk, dpos, KVH, D, st);
            cache_write_kv_side(KvCodec::Tcq2, dkv + size_t(p) * ts + KVH * D, ts, 512, tv, dpos, KVH, D, st);
        }
        CK(cudaStreamSynchronize(st));
        for (const auto & [vcod, vbuf, name] : {std::tuple{KvCodec::Tq3, static_cast<void *>(vc), "tcq3/tq3"},
                                                std::tuple{KvCodec::Tcq2, tv, "tcq3/tcq2"}}) {
            auto run_t = [&](int nq, const KvCompactSeg * cs) {
                const int hp[2] = {N - nq, N - nq + 1};
                cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
                const int split = std::clamp(2 * sms / KVH, 1, 256);
                attention_kv_split(KvCodec::Tcq3, vcod, q, tk, vbuf, y, part, dpos + 1, nq, H, KVH, scale, split, st, nullptr, nullptr, cs);
                std::vector<__nv_bfloat16> o(size_t(nq) * H * D);
                cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
                std::vector<float> f(o.size());
                for (size_t i = 0; i < o.size(); ++i) f[i] = __bfloat162float(o[i]);
                return f;
            };
            for (int ratio : {1, 8}) {
                const int m = nb / ratio;
                kv_compact_block(tk, vbuf, KvCodec::Tcq3, vcod, nb, KVH, H, ring, R, scale, m, 0, ck, cv, cb, scratch, blas, sol, st);
                CK(cudaMemcpy(cn, &m, 4, cudaMemcpyHostToDevice));
                CK(cudaStreamSynchronize(st));
                const auto plain = run_t(12, nullptr), comp = run_t(12, &seg);
                const double e = rel_rms(comp, plain);
                std::printf("kv_compact %s compactor %dx (%d of %d rows): attention vs exact %.3e (random queries)\n", name, ratio, m, nb, e);
                if (ratio == 1) CHECK(e < 3e-2, "%s compactor at ratio 1 does not reproduce attention (%.3e)", name, e);
            }
        }
        cudaFree(tk), cudaFree(tv), cudaFree(scratch), cudaFree(ring);
        cublasDestroy(blas);
        cusolverDnDestroy(sol);
    }
    {  // keys the fit queries never attend to must stay visible: fit queries are copies of keys from the first half of
       // the block, later queries copies of keys from the second half (an answer copies the digits after what the
       // question matched). At ratio 1 every key is kept, so attention for the later queries must match the exact cache.
        const int nb = C, Rq = 64, Rt = 12;
        auto key_query = [&](int pos, int kh, float alpha, __nv_bfloat16 * dst) {  // one query row = alpha * key (pos, kh)
            for (int d = 0; d < D; ++d) dst[d] = __float2bfloat16(alpha * kd[(size_t(pos) * KVH + kh) * D + d]);
        };
        std::vector<__nv_bfloat16> hfit(size_t(Rq) * H * D), htest(size_t(Rt) * H * D);
        for (int r = 0; r < Rq; ++r)
            for (int h = 0; h < H; ++h) key_query((r * 7) % (nb / 2), h / G, 0.5f, &hfit[(size_t(r) * H + h) * D]);
        for (int r = 0; r < Rt; ++r)
            for (int h = 0; h < H; ++h) key_query(nb / 2 + (r * 37) % (nb / 2), h / G, 0.5f, &htest[(size_t(r) * H + h) * D]);
        __nv_bfloat16 *fitq, *testq;
        CK(cudaMalloc(&fitq, hfit.size() * 2));
        CK(cudaMalloc(&testq, htest.size() * 2));
        CK(cudaMemcpy(fitq, hfit.data(), hfit.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(testq, htest.data(), htest.size() * 2, cudaMemcpyHostToDevice));
        cublasHandle_t blas;
        cusolverDnHandle_t sol;
        cublasCreate(&blas);
        cusolverDnCreate(&sol);
        void * scratch;
        CK(cudaMalloc(&scratch, kv_compact_scratch_bytes(nb, Rq * G, nb, KVH)));
        auto run_q = [&](const __nv_bfloat16 * qq, int nq, const KvCompactSeg * cs) {
            const int hp[2] = {N - nq, N - nq + 1};
            cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
            attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, qq, kc, vc, y, part, dpos + 1, nq, H, KVH, scale, 1, st, nullptr, nullptr, cs);
            std::vector<__nv_bfloat16> o(size_t(nq) * H * D);
            cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> f(o.size());
            for (size_t k = 0; k < o.size(); ++k) f[k] = __bfloat162float(o[k]);
            return f;
        };
        kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, nb, KVH, H, fitq, Rq, scale, nb, 0, ck, cv, cb, scratch, blas, sol, st);
        CK(cudaMemcpy(cn, &nb, 4, cudaMemcpyHostToDevice));
        CK(cudaStreamSynchronize(st));
        const double e_fit = rel_rms(run_q(fitq, Rt, &seg), run_q(fitq, Rt, nullptr));
        const double e_unseen = rel_rms(run_q(testq, Rt, &seg), run_q(testq, Rt, nullptr));
        std::vector<float> hb(size_t(nb) * KVH);
        CK(cudaMemcpy(hb.data(), cb, hb.size() * 4, cudaMemcpyDeviceToHost));
        double bmin_unseen = 1e30;
        for (int j = nb / 2; j < nb; ++j) for (int kh = 0; kh < KVH; ++kh) bmin_unseen = std::min(bmin_unseen, double(hb[size_t(j) * KVH + kh]));
        std::printf("kv_compact unattended keys (ratio 1): fit queries %.3e, unseen queries %.3e, min bias of unattended keys %.1f\n",
                    e_fit, e_unseen, bmin_unseen);
        CHECK(e_unseen < 3e-2, "keys the fit queries ignore become invisible (unseen queries %.3e, min bias %.1f)", e_unseen, bmin_unseen);
        cudaFree(fitq), cudaFree(testq), cudaFree(scratch);
        cublasDestroy(blas);
        cusolverDnDestroy(sol);
    }
    {  // a partly filled query ring (a short turn): the 16 newest rows sit at ring rows 100..115 and every other row holds a
       // stale query of an earlier turn. The fit must use the newest rows; reading rows 0..15 fits the stale queries.
        const int nb = C, RR = 256, Rn = 16, end = 116, m = nb / 8;
        auto key_query = [&](int pos, int kh, __nv_bfloat16 * dst) {
            for (int d = 0; d < D; ++d) dst[d] = __float2bfloat16(0.5f * kd[(size_t(pos) * KVH + kh) * D + d]);
        };
        std::vector<__nv_bfloat16> hring(size_t(RR) * H * D), hnew(size_t(Rn) * H * D);
        for (int r = 0; r < RR; ++r) {
            const bool fresh = r >= end - Rn && r < end;  // fresh rows attend the first half of the block, stale the second
            const int pos = fresh ? (r * 29) % (nb / 2) : nb / 2 + (r * 37) % (nb / 2);
            for (int h = 0; h < H; ++h) key_query(pos, h / G, &hring[(size_t(r) * H + h) * D]);
        }
        std::copy(hring.begin() + size_t(end - Rn) * H * D, hring.begin() + size_t(end) * H * D, hnew.begin());
        __nv_bfloat16 *dring, *dnew;
        CK(cudaMalloc(&dring, hring.size() * 2));
        CK(cudaMalloc(&dnew, hnew.size() * 2));
        CK(cudaMemcpy(dring, hring.data(), hring.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnew, hnew.data(), hnew.size() * 2, cudaMemcpyHostToDevice));
        cublasHandle_t blas;
        cusolverDnHandle_t sol;
        cublasCreate(&blas);
        cusolverDnCreate(&sol);
        void * scratch;
        CK(cudaMalloc(&scratch, kv_compact_scratch_bytes(nb, RR * G, nb, KVH)));
        auto err_new = [&](const __nv_bfloat16 * ring, int ring_rows, int ring_end) {  // attention error of the newest queries
            kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, nb, KVH, H, ring, Rn, scale, m, 0, ck, cv, cb, scratch, blas, sol, st,
                             100, 1e-3f, 0, ring_rows, ring_end);
            cudaMemcpy(cn, &m, 4, cudaMemcpyHostToDevice);
            cudaStreamSynchronize(st);
            const int hp[2] = {N - Rn, N - Rn + 1};
            auto att = [&](const KvCompactSeg * cs) {
                cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
                attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, dnew, kc, vc, y, part, dpos + 1, Rn, H, KVH, scale, 1, st, nullptr, nullptr, cs);
                std::vector<__nv_bfloat16> o(size_t(Rn) * H * D);
                cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
                std::vector<float> f(o.size());
                for (size_t k = 0; k < o.size(); ++k) f[k] = __bfloat162float(o[k]);
                return f;
            };
            return rel_rms(att(&seg), att(nullptr));
        };
        const double e_clean = err_new(dnew, 0, 0);       // only the newest rows, contiguous: the target
        const double e_old = err_new(dring, 0, 0);        // rows 0..15 of the ring (the old read)
        const double e_ring = err_new(dring, RR, end);    // the newest rows of the ring
        std::printf("kv_compact partly filled ring (16 new of 256 rows, 8x): newest-only fit %.3e | rows 0..15 %.3e | ring-aware %.3e\n",
                    e_clean, e_old, e_ring);
        CHECK(e_ring < 1.05 * e_clean + 1e-3, "the fit does not use the newest ring rows (%.3e vs %.3e)", e_ring, e_clean);
        CHECK(e_old > 1.5 * e_clean, "fixture does not reproduce the stale-row fit (%.3e vs %.3e)", e_old, e_clean);
        cudaFree(dring), cudaFree(dnew), cudaFree(scratch);
        cublasDestroy(blas);
        cusolverDnDestroy(sol);
    }
    {  // the nq x nq path (value solve and NNLS on the thin P when the turn has fewer query rows than stand-in rows) must
       // fit at least as well as the m x m path: attention vs the exact cache for the queries it was fitted to
        const int nb = C, Rs = 16;
        cublasHandle_t blas;
        cusolverDnHandle_t sol;
        cublasCreate(&blas);
        cusolverDnCreate(&sol);
        void * scratch;
        CK(cudaMalloc(&scratch, kv_compact_scratch_bytes(nb, 256 * G, nb, KVH)));
        __nv_bfloat16 * qr;
        CK(cudaMalloc(&qr, size_t(Rs) * H * D * 2));
        std::vector<__nv_bfloat16> hq2(size_t(Rs) * H * D);
        for (auto & x : hq2) x = __float2bfloat16(nd(g) * 0.5f);
        CK(cudaMemcpy(qr, hq2.data(), hq2.size() * 2, cudaMemcpyHostToDevice));
        auto att = [&](const KvCompactSeg * cs) {
            const int hp[2] = {N - Rs, N - Rs + 1};
            cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
            attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, qr, kc, vc, y, part, dpos + 1, Rs, H, KVH, scale, 1, st, nullptr, nullptr, cs);
            std::vector<__nv_bfloat16> o(size_t(Rs) * H * D);
            cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> f(o.size());
            for (size_t k = 0; k < o.size(); ++k) f[k] = __bfloat162float(o[k]);
            return f;
        };
        const auto exact = att(nullptr);
        for (int ratio : {1, 8}) {
            const int m = nb / ratio;
            double e[2];
            for (int solve : {1, 2}) {
                kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, nb, KVH, H, qr, Rs, scale, m, 0, ck, cv, cb, scratch, blas, sol, st,
                                 100, 1e-3f, 0, 0, 0, solve);
                cudaMemcpy(cn, &m, 4, cudaMemcpyHostToDevice);
                cudaStreamSynchronize(st);
                e[solve - 1] = rel_rms(att(&seg), exact);
            }
            std::printf("kv_compact nq x nq path %dx (nq %d < m %d): fit queries vs exact %.4e (m x m path %.4e)\n", ratio, Rs * G, m, e[1], e[0]);
            CHECK(e[1] <= 1.02 * e[0] + 1e-4, "the nq x nq path fits worse than the m x m path (%.4e vs %.4e)", e[1], e[0]);
        }
        cudaFree(qr), cudaFree(scratch);
        cublasDestroy(blas);
        cusolverDnDestroy(sol);
    }
    if (std::getenv("KV_COMPACT_ITERS")) {  // diagnostic, not a gate: fit quality vs NNLS iterations (8x, 64 ring rows)
        const int nb = C, Rs = 64, m = nb / 8;
        cublasHandle_t blas;
        cusolverDnHandle_t sol;
        cublasCreate(&blas);
        cusolverDnCreate(&sol);
        void * scratch;
        CK(cudaMalloc(&scratch, kv_compact_scratch_bytes(nb, 256 * G, nb, KVH)));
        __nv_bfloat16 * qr;
        CK(cudaMalloc(&qr, size_t(Rs) * H * D * 2));
        std::vector<__nv_bfloat16> hq2(size_t(Rs) * H * D);
        for (auto & x : hq2) x = __float2bfloat16(nd(g) * 0.5f);
        CK(cudaMemcpy(qr, hq2.data(), hq2.size() * 2, cudaMemcpyHostToDevice));
        auto att = [&](const __nv_bfloat16 * qq, const KvCompactSeg * cs) {
            const int hp[2] = {N - 12, N - 11};
            cudaMemcpy(dpos, hp, 8, cudaMemcpyHostToDevice);
            attention_kv_split(KvCodec::Tq3, KvCodec::Tq3, qq, kc, vc, y, part, dpos + 1, 12, H, KVH, scale, 1, st, nullptr, nullptr, cs);
            std::vector<__nv_bfloat16> o(size_t(12) * H * D);
            cudaMemcpy(o.data(), y, o.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> f(o.size());
            for (size_t k = 0; k < o.size(); ++k) f[k] = __bfloat162float(o[k]);
            return f;
        };
        const auto ex_fit = att(qr, nullptr), ex_new = att(q, nullptr);
        for (int iters : {300, 100, 30, 10, 3, 0}) {
            kv_compact_block(kc, vc, KvCodec::Tq3, KvCodec::Tq3, nb, KVH, H, qr, Rs, scale, m, 0, ck, cv, cb, scratch, blas, sol, st, iters);
            cudaMemcpy(cn, &m, 4, cudaMemcpyHostToDevice);
            cudaStreamSynchronize(st);
            std::printf("kv_compact iters %3d (8x, nq %d, m %d): fit queries %.4e, unseen queries %.4e\n", iters, Rs * G, m,
                        rel_rms(att(qr, &seg), ex_fit), rel_rms(att(q, &seg), ex_new));
        }
        cudaFree(qr), cudaFree(scratch);
        cublasDestroy(blas);
        cusolverDnDestroy(sol);
    }
    {  // GPU top-m selection == the host rule (score desc, index asc), with heavy ties and the successor spread
        std::mt19937 rg(7);
        float * dm;
        int * ds;
        CK(cudaMalloc(&dm, 8192 * 4));
        CK(cudaMalloc(&ds, 8192 * 4));
        int bad = 0, cases = 0;
        for (int nb : {1024, 8192, 1000})
            for (int m : {nb / 8, nb / 4, 333})
                for (int succ : {0, 6}) {
                    if (m > nb) continue;
                    std::vector<float> mass(nb);
                    for (auto & x : mass) x = float(rg() % 37) * 0.25f;  // few distinct values: many ties
                    CK(cudaMemcpy(dm, mass.data(), nb * 4, cudaMemcpyHostToDevice));
                    kv_topm_select(dm, nb, m, succ, ds, st);
                    std::vector<int> got(m);
                    CK(cudaMemcpy(got.data(), ds, m * 4, cudaMemcpyDeviceToHost));
                    std::vector<float> sc(mass);
                    for (int j = 0; j < nb; ++j)
                        for (int d = 1; d <= succ && j - d >= 0; ++d) sc[j] = std::max(sc[j], mass[j - d]);
                    std::vector<int> idx(nb);
                    std::iota(idx.begin(), idx.end(), 0);
                    std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) { return sc[x] > sc[y]; });
                    std::vector<int> want(idx.begin(), idx.begin() + m);
                    std::sort(want.begin(), want.end());
                    ++cases;
                    if (got != want) ++bad;
                }
        std::printf("kv_compact GPU top-m selection: %d of %d cases match the host rule\n", cases - bad, cases);
        CHECK(bad == 0, "GPU top-m selection differs from the host rule in %d case(s)", bad);
        cudaFree(dm), cudaFree(ds);
    }
    if (fails) std::printf("kv_compact: %d failure(s)\n", fails);
    else std::printf("kv_compact: ok\n");
    return fails ? 1 : 0;
}

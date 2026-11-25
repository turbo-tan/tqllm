// Inkling-2-Mini few-token kernels (src/models/inkling2/loop/inkling2_kernels.cu) against double CPU references: multi-segment
// BF16 gemv (BF16 / F32 outputs, K = 64 .. 4608, odd row counts), fused routed experts (gate + up + softcapped SwiGLU,
// down + combine) on NVF4 planes and MXFP8 rows, router top-k from logits, multi-block depth attention. Every few-token kernel
// must give token t of an n-token call bitwise the same result as a 1-token call on that token (verify == plain decode).
// Prompt-chunk kernels: grouped expert GEMM (tensor cores) vs the CPU reference, parallel convolutions vs the per-step ones;
// the spec-decode conv rollback vs running only the kept tokens; the exact greedy LM head; one depth-attention scratch for every n.
#include "inkling2_dflash2.cuh"
#include "inkling2_grouped.cuh"
#include "inkling2_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = tqllm_cuda::inkling2;
using bf16 = __nv_bfloat16;

namespace {
bool ok = true;
void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); std::exit(e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver ? 77 : 1); }
}
float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
template <class T> T * up(const std::vector<T> & v) {
    T * p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class T> std::vector<T> down(const T * p, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
std::vector<bf16> to_bf(const std::vector<float> & v) {
    std::vector<bf16> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = __float2bfloat16(v[i]);
    return o;
}
std::vector<float> from_bf(const std::vector<bf16> & v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = __bfloat162float(v[i]);
    return o;
}
void report(const char * what, double err, double tol, bool bitwise) {
    const bool pass = err <= tol && bitwise;
    std::printf("%-46s worst rel err %.2e (tol %.0e), n-token == 1-token bitwise: %s %s\n", what, err, tol, bitwise ? "yes" : "NO", pass ? "ok" : "FAIL");
    ok = ok && pass;
}
std::mt19937 rng(7);
std::vector<float> randv(size_t n, float sd) {
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (auto & x : v) x = bf(d(rng));
    return v;
}

void test_gemv() {
    struct S { int rows, K; bool f32; };
    const S shapes[] = {{37, 2304, false}, {3072, 2304, false}, {128, 2304, true}, {64, 4608, false}, {2304, 64, false}, {25, 3072, true}};
    const int ns = 6, NMAX = 8;
    std::vector<std::vector<float>> W(ns), X(ns);
    std::vector<bf16 *> dW(ns), dX(ns);
    std::vector<void *> dY(ns);
    for (int i = 0; i < ns; ++i) {
        W[i] = randv(size_t(shapes[i].rows) * shapes[i].K, 0.05f);
        X[i] = randv(size_t(NMAX) * shapes[i].K, 1.0f);
        dW[i] = up(to_bf(W[i])); dX[i] = up(to_bf(X[i]));
        ck(cudaMalloc(&dY[i], size_t(NMAX) * shapes[i].rows * 4), "y");
    }
    auto segs = [&](int off) {
        std::vector<K::GemvSeg> g(ns);
        for (int i = 0; i < ns; ++i)
            g[i] = K::GemvSeg{dW[i], dX[i] + size_t(off) * shapes[i].K, dY[i], shapes[i].rows, shapes[i].K, shapes[i].K, shapes[i].rows, shapes[i].f32 ? 1 : 0};
        return g;
    };
    auto get = [&](int i, int n) {
        std::vector<float> o(size_t(n) * shapes[i].rows);
        if (shapes[i].f32) o = down(static_cast<float *>(dY[i]), o.size());
        else o = from_bf(down(static_cast<bf16 *>(dY[i]), o.size()));
        return o;
    };
    double err = 0;
    bool bit = true;
    auto g = segs(0);
    K::gemv(g.data(), ns, NMAX, 0);
    ck(cudaDeviceSynchronize(), "gemv");
    std::vector<std::vector<float>> full(ns);
    for (int i = 0; i < ns; ++i) {
        full[i] = get(i, NMAX);
        const int R = shapes[i].rows, Kc = shapes[i].K;
        for (int t = 0; t < NMAX; ++t)
            for (int r = 0; r < R; ++r) {
                double ref = 0, mag = 0;
                for (int c = 0; c < Kc; ++c) { ref += double(W[i][size_t(r) * Kc + c]) * X[i][size_t(t) * Kc + c]; mag += std::fabs(double(W[i][size_t(r) * Kc + c]) * X[i][size_t(t) * Kc + c]); }
                // BF16 outputs round to 2^-9 of |ref|; the accumulation error scales with the typical term size
                err = std::max(err, std::fabs(full[i][size_t(t) * R + r] - ref) / (std::max(std::fabs(ref), mag / std::sqrt(double(Kc))) + 1e-6));
            }
    }
    for (int n = 1; n < NMAX; ++n)  // tokens off .. off + n - 1 of a smaller call == the same rows of the 8-token call
        for (int off = 0; off + n <= NMAX; off += n) {
            auto h = segs(off);
            K::gemv(h.data(), ns, n, 0);
            ck(cudaDeviceSynchronize(), "gemv n");
            for (int i = 0; i < ns; ++i) {
                const auto o = get(i, n);
                bit = bit && std::memcmp(o.data(), full[i].data() + size_t(off) * shapes[i].rows, o.size() * 4) == 0;
            }
        }
    report("gemv (6 segments, K 64..4608, n 1..8)", err, 1e-2, bit);
    for (int i = 0; i < ns; ++i) { cudaFree(dW[i]); cudaFree(dX[i]); cudaFree(dY[i]); }
}

// INT8 W8A16 rows (fast mode): per-row max-abs quantisation, the gemv vs a CPU dot of the dequantised rows, n-invariance
void test_gemv_i8() {
    const int R = 75, Kc = 2304, NMAX = 8;
    const auto W = randv(size_t(R) * Kc, 0.05f), X = randv(size_t(NMAX) * Kc, 1.0f);
    bf16 * dW = up(to_bf(W)), * dX = up(to_bf(X)), * dY = nullptr;
    int8_t * dq = nullptr; float * ds = nullptr;
    ck(cudaMalloc(&dq, size_t(R) * Kc), "q"); ck(cudaMalloc(&ds, R * 4), "s"); ck(cudaMalloc(&dY, size_t(NMAX) * R * 2), "y");
    K::quantize_rows_i8(dW, dq, ds, R, Kc, 0);
    ck(cudaDeviceSynchronize(), "quantize");
    std::vector<int8_t> q(size_t(R) * Kc);
    ck(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost), "q");
    const auto sc = down(ds, size_t(R));
    const auto Wb = from_bf(to_bf(W));  // the kernel quantises the BF16 weights, not the F32 draw
    double qerr = 0;  // the quantisation itself: within half a step of each row's scale
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < Kc; ++c) qerr = std::max(qerr, std::fabs(Wb[size_t(r) * Kc + c] - q[size_t(r) * Kc + c] * double(sc[r])) / sc[r]);
    auto run = [&](int off, int n) {
        K::GemvSeg g{reinterpret_cast<const bf16 *>(dq), dX + size_t(off) * Kc, dY, R, Kc, Kc, R, 0, ds};
        K::gemv(&g, 1, n, 0);
        ck(cudaDeviceSynchronize(), "gemv i8");
        return from_bf(down(dY, size_t(n) * R));
    };
    const auto full = run(0, NMAX);
    double err = 0;
    for (int t = 0; t < NMAX; ++t)
        for (int r = 0; r < R; ++r) {
            double ref = 0, mag = 0;
            for (int c = 0; c < Kc; ++c) { const double w = q[size_t(r) * Kc + c] * double(sc[r]); ref += w * X[size_t(t) * Kc + c]; mag += std::fabs(w * X[size_t(t) * Kc + c]); }
            err = std::max(err, std::fabs(full[size_t(t) * R + r] - ref) / (std::max(std::fabs(ref), mag / std::sqrt(double(Kc))) + 1e-6));
        }
    bool bit = true;
    for (int t = 0; t < NMAX; ++t) { const auto one = run(t, 1); bit = bit && std::memcmp(one.data(), full.data() + size_t(t) * R, size_t(R) * 4) == 0; }
    report("gemv INT8 rows (quantisation <= 0.5 step)", qerr <= 0.5 + 1e-5 ? err : 1.0, 1e-2, bit);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dq); cudaFree(ds);
}

// expert stacks: fmt 2 NVF4 planes (codes [rows][cols / 2], low nibble first; aux [rows][cols / 16 E4M3, F32 global]),
// fmt 1 MXFP8 rows [cols E4M3][cols / 32 E8M0]; w: the reference's decoded weights [E][rows][cols]
const float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
float e4m3(uint8_t b) { __nv_fp8_e4m3 v; v.__x = b; return float(v); }
uint8_t rand_e4m3(int lo, int hi) {  // a byte in [lo, hi] that is not NaN (0x7F / 0xFF)
    for (;;) { const int b = lo + int(rng() % unsigned(hi - lo + 1)); if ((b & 0x7f) != 0x7f) return uint8_t(b); }
}
struct ExpertBank {
    std::vector<uint8_t> bytes, aux; std::vector<float> w; K::ExpertMat m;
    uint8_t * dw = nullptr, * da = nullptr;
    void upload() { dw = up(bytes); da = aux.empty() ? nullptr : up(aux); m.w = dw; m.aux = da; }
    ~ExpertBank() { if (dw) cudaFree(dw); if (da) cudaFree(da); }
};
void make_bank(ExpertBank & b, int fmt, int E, int rows, int cols) {
    b.m.fmt = fmt;
    const size_t rb = fmt == 2 ? size_t(cols) / 2 : size_t(cols + cols / 32), ab = size_t(cols) / 16 + 4;
    b.m.stride = rb * rows; b.m.aux_stride = fmt == 2 ? ab * rows : 0;
    b.bytes.resize(b.m.stride * E);
    if (fmt == 2) b.aux.resize(b.m.aux_stride * E);
    b.w.resize(size_t(E) * rows * cols);
    for (int e = 0; e < E; ++e) {
        const float g = 0.002f * (1 + int(rng() % 8));  // one global per expert, as the checkpoint
        for (int r = 0; r < rows; ++r) {
            uint8_t * row = b.bytes.data() + e * b.m.stride + size_t(r) * rb;
            float * wr = b.w.data() + (size_t(e) * rows + r) * cols;
            if (fmt == 2) {
                uint8_t * ar = b.aux.data() + e * b.m.aux_stride + size_t(r) * ab;
                for (int j = 0; j < cols / 16; ++j) ar[j] = rand_e4m3(0x30, 0x48);
                std::memcpy(ar + cols / 16, &g, 4);
                for (int j = 0; j < cols / 2; ++j) row[j] = uint8_t(rng());
                for (int c = 0; c < cols; ++c)
                    wr[c] = kE2M1[c % 2 ? row[c / 2] >> 4 : row[c / 2] & 15] * (e4m3(ar[c / 16]) * g);  // e2m1 x (scale x global)
            } else {
                for (int c = 0; c < cols; ++c) row[c] = rand_e4m3(0x00, 0xfe);
                for (int j = 0; j < cols / 32; ++j) row[cols + j] = uint8_t(118 + rng() % 5);
                for (int c = 0; c < cols; ++c) wr[c] = e4m3(row[c]) * std::ldexp(1.0f, int(row[cols + c / 32]) - 127);
            }
        }
    }
    b.upload();
}

void test_experts(int fmt, int NMAX = 4) {  // NMAX 8: dedup groups of up to 8 members (16 experts)
    const int E = 16, topk = 8, D = 2304, I = 1152;
    const float gc = 4.0f, uc = 25.0f;
    ExpertBank G, U, Dn;
    make_bank(G, fmt, E, I, D); make_bank(U, fmt, E, I, D); make_bank(Dn, fmt, E, D, I);
    const auto X = randv(size_t(NMAX) * D, 1.0f);
    std::vector<int> ids(size_t(NMAX) * topk);
    std::vector<float> wts(ids.size());
    for (int t = 0; t < NMAX; ++t) {
        std::vector<int> p(E);
        for (int e = 0; e < E; ++e) p[e] = e;
        std::shuffle(p.begin(), p.end(), rng);
        if (t % 2) std::swap(p[0], *std::find(p.begin(), p.end(), ids[(t - 1) * topk]));  // an expert shared with the previous token
        for (int k = 0; k < topk; ++k) { ids[t * topk + k] = p[k]; wts[t * topk + k] = 0.5f + float(rng() % 100) / 100.0f; }
    }
    bf16 * dx = up(to_bf(X)), * dact = nullptr, * dout = nullptr;
    int * dids = up(ids);
    float * dw = up(wts);
    ck(cudaMalloc(&dact, size_t(NMAX) * topk * I * 2), "act"); ck(cudaMalloc(&dout, size_t(NMAX) * D * 2), "out");
    K::expert_gate_up(G.m, U.m, I, D, dx, dids, dact, NMAX, topk, gc, uc, 0);
    K::expert_down(Dn.m, D, I, dact, dids, dw, dout, NMAX, topk, 0);
    ck(cudaDeviceSynchronize(), "experts");
    const auto act = from_bf(down(dact, size_t(NMAX) * topk * I)), out = from_bf(down(dout, size_t(NMAX) * D));
    double ea = 0, eo = 0;
    for (int t = 0; t < NMAX; ++t) {
        std::vector<double> y(D, 0.0);
        for (int k = 0; k < topk; ++k) {
            const int a = t * topk + k, e = ids[a];
            for (int i = 0; i < I; ++i) {
                double g = 0, u = 0;
                for (int c = 0; c < D; ++c) {
                    g += double(G.w[(size_t(e) * I + i) * D + c]) * X[size_t(t) * D + c];
                    u += double(U.w[(size_t(e) * I + i) * D + c]) * X[size_t(t) * D + c];
                }
                g = bf(float(g)); u = bf(float(u));
                const double ref = gc * std::tanh(g / gc) / (1.0 + std::exp(-g)) * uc * std::tanh(u / uc);
                ea = std::max(ea, std::fabs(act[size_t(a) * I + i] - ref) / (std::fabs(ref) + 0.05));
            }
            for (int r = 0; r < D; ++r) {  // down from the GPU's own act, so the two checks are independent
                double d = 0;
                for (int i = 0; i < I; ++i) d += double(Dn.w[(size_t(e) * D + r) * I + i]) * act[size_t(a) * I + i];
                y[r] += bf(float(d)) * wts[a];
            }
        }
        for (int r = 0; r < D; ++r) eo = std::max(eo, std::fabs(out[size_t(t) * D + r] - y[r]) / (std::fabs(y[r]) + 0.05));
    }
    bool bit = true;
    for (int t = 0; t < NMAX; ++t) {
        K::expert_gate_up(G.m, U.m, I, D, dx + size_t(t) * D, dids + t * topk, dact, 1, topk, gc, uc, 0);
        K::expert_down(Dn.m, D, I, dact, dids + t * topk, dw + t * topk, dout, 1, topk, 0);
        ck(cudaDeviceSynchronize(), "experts one");
        const auto a1 = from_bf(down(dact, size_t(topk) * I)), o1 = from_bf(down(dout, size_t(D)));
        bit = bit && std::memcmp(a1.data(), act.data() + size_t(t) * topk * I, a1.size() * 4) == 0;
        bit = bit && std::memcmp(o1.data(), out.data() + size_t(t) * D, o1.size() * 4) == 0;
    }
    {   // verify-block path: each distinct expert once for all its assignments, outputs == the per-token kernels bitwise
        int * groups = nullptr; __nv_bfloat16 * yd = nullptr, * act2 = nullptr, * out2 = nullptr;
        ck(cudaMalloc(&groups, size_t(1 + 8 * 8 * 9) * 4), "groups"); ck(cudaMalloc(&yd, size_t(NMAX) * topk * D * 2), "yd");
        ck(cudaMalloc(&act2, size_t(NMAX) * topk * I * 2), "act2"); ck(cudaMalloc(&out2, size_t(NMAX) * D * 2), "out2");
        K::expert_gate_up(G.m, U.m, I, D, dx, dids, dact, NMAX, topk, gc, uc, 0);
        K::expert_down(Dn.m, D, I, dact, dids, dw, dout, NMAX, topk, 0);
        K::expert_dedup(G.m, U.m, Dn.m, I, D, dx, dids, dw, groups, act2, yd, out2, NMAX, topk, gc, uc, 0);
        ck(cudaDeviceSynchronize(), "dedup");
        const auto o1 = from_bf(down(dout, size_t(NMAX) * D)), o2 = from_bf(down(out2, size_t(NMAX) * D));
        const auto a1 = from_bf(down(dact, size_t(NMAX) * topk * I)), a2 = from_bf(down(act2, size_t(NMAX) * topk * I));
        bit = bit && std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0 && std::memcmp(a1.data(), a2.data(), a1.size() * 4) == 0;
        cudaFree(groups); cudaFree(yd); cudaFree(act2); cudaFree(out2);
    }
    if (NMAX > 4) {  // the numbers are checked at 4 tokens; at 8 the dedup groups (up to 8 members) == the per-token kernels bitwise
        report(fmt == 2 ? "experts NVF4 dedup, 8 tokens" : "experts MXFP8 dedup, 8 tokens", 0.0, 0.0, bit);
    } else {
        report(fmt == 2 ? "experts NVF4 gate/up + SwiGLU" : "experts MXFP8 gate/up + SwiGLU", ea, 2e-2, bit);
        report(fmt == 2 ? "experts NVF4 down + combine" : "experts MXFP8 down + combine", eo, 2e-2, bit);
    }
    cudaFree(dx); cudaFree(dact); cudaFree(dout); cudaFree(dids); cudaFree(dw);
}

// TQT (fmt 3): random stream bytes and FP16 scales; the reference weights are the code's values x scale in TQ4's rotated basis
// (written from the format description: tools/convert/qwen35/tqt.py code_values) un-rotated per 128 (R^-1 = D H / sqrt(128))
const uint32_t kS1[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
float tqt_sign(int c) { return (kS1[c / 32] >> (c % 32)) & 1u ? -1.0f : 1.0f; }
void tqt_pair(uint32_t w, float & lo, float & hi, int L = 18) {
    const uint32_t x = (w * 0x9E3779B1u + 0x7F4A7C15u) << (32 - L), y = (x << 16) | (x >> 16);
    const uint32_t z = y * 0x2C1B3C6Du + 0x9E3779B9u, z2 = z * 0x297A2D39u + 0x7F4A7C15u;
    const uint32_t h1 = (z & 0x8FFF8FFFu) ^ 0x3B603B60u, h2 = (z2 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    auto f16 = [](uint32_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); };
    lo = __half2float(__float2half_rn(f16(h1 & 0xFFFF) + f16(h2 & 0xFFFF)));
    hi = __half2float(__float2half_rn(f16(h1 >> 16) + f16(h2 >> 16)));
}
void fwht128(double * v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) { const double a = v[j], b = v[j + h]; v[j] = a + b; v[j + h] = a - b; }
}
void rot128(const float * x, float * y) {  // R x
    double v[128];
    for (int c = 0; c < 128; ++c) v[c] = double(x[c]) * tqt_sign(c);
    fwht128(v);
    for (int c = 0; c < 128; ++c) y[c] = float(v[c] / std::sqrt(128.0));
}
void make_bank_tqt(ExpertBank & b, int E, int rows, int cols, int fmt = 3) {
    b.m.fmt = fmt;
    const size_t rb = fmt == 5 ? size_t(cols) * 3 / 8 : size_t(cols) / (fmt == 4 ? 4 : 2), nb = size_t(cols) / 128;
    b.m.stride = rb * rows; b.m.aux_stride = nb * 2 * rows;
    b.bytes.resize(b.m.stride * E); b.aux.resize(b.m.aux_stride * E);
    b.w.resize(size_t(E) * rows * cols);
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < rows; ++r) {
            uint8_t * row = b.bytes.data() + e * b.m.stride + size_t(r) * rb;
            for (size_t j = 0; j < rb; ++j) row[j] = uint8_t(rng());
            for (size_t k = 0; k < nb; ++k) {
                const __half sc = __float2half_rn(0.004f + 0.0001f * float(rng() % 100));
                std::memcpy(b.aux.data() + e * b.m.aux_stride + (size_t(r) * nb + k) * 2, &sc, 2);
                double q[128];
                for (int p = 0; p < 64; ++p) {
                    float lo, hi;
                    if (fmt == 5) {
                        const uint8_t * s = row + k * 48;
                        uint32_t w = 0;
                        for (int j = 0; j < 18; ++j) { const int i = (6 * p + j) % 384; w |= uint32_t((s[i >> 3] >> (i & 7)) & 1) << j; }
                        tqt_pair(w, lo, hi, 18);
                    } else if (fmt == 4) {
                        const uint8_t * s = row + k * 32;
                        uint32_t w = 0;
                        for (int j = 0; j < 4; ++j) { const int i = (p + j) & 63; w |= uint32_t((s[i >> 1] >> (4 * (i & 1))) & 15) << (4 * j); }
                        tqt_pair(w, lo, hi, 16);
                    } else {
                        const uint8_t * s = row + k * 64;
                        tqt_pair((uint32_t(s[p]) | uint32_t(s[(p + 1) & 63]) << 8 | uint32_t(s[(p + 2) & 63]) << 16) & 0x3FFFF, lo, hi);
                    }
                    q[2 * p] = double(lo) * __half2float(sc); q[2 * p + 1] = double(hi) * __half2float(sc);
                }
                fwht128(q);
                float * wr = b.w.data() + (size_t(e) * rows + r) * cols + k * 128;
                for (int c = 0; c < 128; ++c) wr[c] = float(q[c] / std::sqrt(128.0)) * tqt_sign(c);
            }
        }
    b.upload();
}
std::vector<__half> rot_half(const std::vector<float> & x, size_t rows, size_t cols) {
    std::vector<__half> o(x.size());
    std::vector<float> y(128);
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < cols / 128; ++b) {
            rot128(x.data() + r * cols + b * 128, y.data());
            for (int c = 0; c < 128; ++c) o[r * cols + b * 128 + c] = __float2half_rn(y[c]);
        }
    return o;
}

void test_experts_tqt(int fmt) {
    const int E = 16, topk = 8, D = 2304, I = 1152, NMAX = 8;  // 8 tokens over 16 experts: dedup groups of 1 .. 8 members
    const float gc = 4.0f, uc = 25.0f;
    ExpertBank G, U, Dn;
    make_bank_tqt(G, E, I, D, fmt); make_bank_tqt(U, E, I, D, fmt); make_bank_tqt(Dn, E, D, I, fmt);
    {   // the BF16 decode (original basis) against the reference weights, one expert of each stack
        bf16 * dd = nullptr;
        ck(cudaMalloc(&dd, size_t(E) * I * D * 2), "dec");
        K::experts_decode_bf16(G.m, E, I, D, I, dd, 0);
        ck(cudaDeviceSynchronize(), "decode");
        const auto h = from_bf(down(dd, size_t(E) * I * D));
        double worst = 0;
        for (size_t i = 0; i < h.size(); ++i) worst = std::max(worst, std::fabs(h[i] - bf(G.w[i])) / (std::fabs(G.w[i]) + 1e-3));
        report(fmt == 5 ? "experts TQT3 decode to BF16 (un-rotated)" : fmt == 4 ? "experts TQT2 decode to BF16 (un-rotated)" : "experts TQT decode to BF16 (un-rotated)", worst, 1e-2, true);
        cudaFree(dd);
    }
    const auto X = randv(size_t(NMAX) * D, 1.0f);
    std::vector<int> ids(size_t(NMAX) * topk);
    std::vector<float> wts(ids.size());
    for (int t = 0; t < NMAX; ++t) {
        std::vector<int> p(E);
        for (int e = 0; e < E; ++e) p[e] = e;
        std::shuffle(p.begin(), p.end(), rng);
        for (int k = 0; k < topk; ++k) { ids[t * topk + k] = p[k]; wts[t * topk + k] = 0.5f + float(rng() % 100) / 100.0f; }
    }
    const auto xr = rot_half(X, NMAX, D);
    __half * dxh = up(xr), * dah = nullptr;
    {   // the device rotation against the host one (FP16 outputs, FP32 butterflies in another order: <= 1 FP16 step)
        bf16 * dxb = up(to_bf(X));
        __half * drot = nullptr;
        ck(cudaMalloc(&drot, xr.size() * 2), "rot");
        K::rotate_h(dxb, drot, D, NMAX, 0);
        ck(cudaDeviceSynchronize(), "rotate");
        const auto r = down(drot, xr.size());
        double worst = 0;
        for (size_t i = 0; i < r.size(); ++i) worst = std::max(worst, std::fabs(__half2float(r[i]) - __half2float(xr[i])) / (std::fabs(__half2float(xr[i])) + 0.05));
        report("rotate_h (TQ4 rotation, FP16)", worst, 2e-3, true);
        cudaFree(dxb); cudaFree(drot);
    }
    bf16 * dact = nullptr, * dout = nullptr;
    int * dids = up(ids);
    float * dw = up(wts);
    ck(cudaMalloc(&dact, size_t(NMAX) * topk * I * 2), "act"); ck(cudaMalloc(&dout, size_t(NMAX) * D * 2), "out");
    ck(cudaMalloc(&dah, size_t(NMAX) * topk * I * 2), "acth");
    K::expert_gate_up(G.m, U.m, I, D, nullptr, dids, dact, NMAX, topk, gc, uc, 0, dxh);
    ck(cudaDeviceSynchronize(), "gate up");
    const auto act = from_bf(down(dact, size_t(NMAX) * topk * I));
    const auto ar = rot_half(act, size_t(NMAX) * topk, I);
    ck(cudaMemcpy(dah, ar.data(), ar.size() * 2, cudaMemcpyHostToDevice), "acth");
    K::expert_down(Dn.m, D, I, nullptr, dids, dw, dout, NMAX, topk, 0, dah);
    ck(cudaDeviceSynchronize(), "down");
    const auto out = from_bf(down(dout, size_t(NMAX) * D));
    double ea = 0, eo = 0;
    for (int t = 0; t < NMAX; ++t) {
        std::vector<double> y(D, 0.0);
        for (int k = 0; k < topk; ++k) {
            const int a = t * topk + k, e = ids[a];
            for (int i = 0; i < I; ++i) {
                double g = 0, u = 0;
                for (int c = 0; c < D; ++c) {
                    g += double(G.w[(size_t(e) * I + i) * D + c]) * X[size_t(t) * D + c];
                    u += double(U.w[(size_t(e) * I + i) * D + c]) * X[size_t(t) * D + c];
                }
                g = bf(float(g)); u = bf(float(u));
                const double ref = gc * std::tanh(g / gc) / (1.0 + std::exp(-g)) * uc * std::tanh(u / uc);
                ea = std::max(ea, std::fabs(act[size_t(a) * I + i] - ref) / (std::fabs(ref) + 0.05));
            }
            for (int r = 0; r < D; ++r) {
                double d = 0;
                for (int i = 0; i < I; ++i) d += double(Dn.w[(size_t(e) * D + r) * I + i]) * act[size_t(a) * I + i];
                y[r] += bf(float(d)) * wts[a];
            }
        }
        for (int r = 0; r < D; ++r) eo = std::max(eo, std::fabs(out[size_t(t) * D + r] - y[r]) / (std::fabs(y[r]) + 0.05));
    }
    bool bit = true;
    for (int t = 0; t < NMAX; ++t) {
        K::expert_gate_up(G.m, U.m, I, D, nullptr, dids + t * topk, dact, 1, topk, gc, uc, 0, dxh + size_t(t) * D);
        K::expert_down(Dn.m, D, I, nullptr, dids + t * topk, dw + t * topk, dout, 1, topk, 0, dah + size_t(t) * topk * I);
        ck(cudaDeviceSynchronize(), "experts one");
        const auto a1 = from_bf(down(dact, size_t(topk) * I)), o1 = from_bf(down(dout, size_t(D)));
        bit = bit && std::memcmp(a1.data(), act.data() + size_t(t) * topk * I, a1.size() * 4) == 0;
        bit = bit && std::memcmp(o1.data(), out.data() + size_t(t) * D, o1.size() * 4) == 0;
    }
    report(fmt == 5 ? "experts TQT3 gate/up + SwiGLU" : fmt == 4 ? "experts TQT2 gate/up + SwiGLU" : "experts TQT gate/up + SwiGLU", ea, 2e-2, bit);
    report(fmt == 5 ? "experts TQT3 down + combine" : fmt == 4 ? "experts TQT2 down + combine" : "experts TQT down + combine", eo, 3e-2, bit);
    {   // verify blocks: expert_dedup_tqt (groups of 1 .. 8 members, rows read once) == the per-assignment kernels with the device rotation, bitwise
        bf16 * act2 = nullptr, * out2 = nullptr, * yd = nullptr, * out1 = nullptr;
        __half * ah1 = nullptr, * ah2 = nullptr;
        int * groups = nullptr;
        ck(cudaMalloc(&act2, size_t(NMAX) * topk * I * 2), "act2"); ck(cudaMalloc(&out2, size_t(NMAX) * D * 2), "out2");
        ck(cudaMalloc(&out1, size_t(NMAX) * D * 2), "out1"); ck(cudaMalloc(&yd, size_t(NMAX) * topk * D * 2), "yd");
        ck(cudaMalloc(&ah1, size_t(NMAX) * topk * I * 2), "ah1"); ck(cudaMalloc(&ah2, size_t(NMAX) * topk * I * 2), "ah2");
        ck(cudaMalloc(&groups, size_t(1 + NMAX * topk * (1 + NMAX)) * 4), "groups");
        K::expert_gate_up(G.m, U.m, I, D, nullptr, dids, dact, NMAX, topk, gc, uc, 0, dxh);
        K::rotate_h(dact, ah1, I, NMAX * topk, 0);
        K::expert_down(Dn.m, D, I, nullptr, dids, dw, out1, NMAX, topk, 0, ah1);
        K::expert_dedup_tqt(G.m, U.m, Dn.m, I, D, dxh, dids, dw, groups, act2, ah2, yd, out2, NMAX, topk, gc, uc, 0);
        ck(cudaDeviceSynchronize(), "dedup tqt");
        const auto a1 = down(dact, size_t(NMAX) * topk * I), a2 = down(act2, size_t(NMAX) * topk * I);
        const auto o1 = down(out1, size_t(NMAX) * D), o2 = down(out2, size_t(NMAX) * D);
        const bool same = std::memcmp(a1.data(), a2.data(), a1.size() * 2) == 0 && std::memcmp(o1.data(), o2.data(), o1.size() * 2) == 0;
        report(fmt == 5 ? "experts TQT3 dedup, 8 tokens == per assignment" : fmt == 4 ? "experts TQT2 dedup, 8 tokens == per assignment"
                                                                                   : "experts TQT dedup, 8 tokens == per assignment", 0.0, 0.0, same);
        cudaFree(act2); cudaFree(out2); cudaFree(out1); cudaFree(yd); cudaFree(ah1); cudaFree(ah2); cudaFree(groups);
    }
    cudaFree(dxh); cudaFree(dah); cudaFree(dact); cudaFree(dout); cudaFree(dids); cudaFree(dw);
}

void test_route() {
    const int n = 5, E = 128, topk = 8;
    const auto lg = randv(size_t(n) * E, 2.0f), bias = randv(E, 0.3f);
    int * dids = nullptr;
    float * dw = nullptr, * dl = up(lg), * db = up(bias);
    ck(cudaMalloc(&dids, n * topk * 4), "ids"); ck(cudaMalloc(&dw, n * topk * 4), "w");
    K::route_topk(dl, db, dids, dw, n, E, topk, 8.0f, 0);
    ck(cudaDeviceSynchronize(), "route");
    const auto ids = down(dids, size_t(n) * topk);
    const auto w = down(dw, size_t(n) * topk);
    double err = 0;
    bool same = true;
    for (int t = 0; t < n; ++t) {
        std::vector<int> p(E);
        for (int e = 0; e < E; ++e) p[e] = e;
        auto sg = [&](int e) { return 1.0 / (1.0 + std::exp(-double(lg[t * E + e]))); };
        std::stable_sort(p.begin(), p.end(), [&](int a, int b) { return sg(a) + bias[a] > sg(b) + bias[b]; });
        double sum = 0;
        for (int k = 0; k < topk; ++k) sum += sg(p[k]);
        for (int k = 0; k < topk; ++k) {
            same = same && ids[t * topk + k] == p[k];
            err = std::max(err, std::fabs(w[t * topk + k] - sg(p[k]) / sum * 8.0) / 8.0);
        }
    }
    report("router top-k from logits (ids equal)", same ? err : 1.0, 1e-5, true);
    cudaFree(dids); cudaFree(dw); cudaFree(dl); cudaFree(db);
}

void test_attn_res() {
    const int D = 2304, n = 3, cnt = 5, nq = 3;
    std::vector<std::vector<float>> S(cnt);
    std::vector<float *> dS(cnt);
    K::States st{};
    st.count = cnt;
    for (int j = 0; j < cnt; ++j) { S[j] = randv(size_t(n) * D, 1.0f + j); dS[j] = up(S[j]); st.s[j] = dS[j]; }
    const auto q = randv(size_t(nq) * D, 0.1f);
    float * dq = up(q), * dout = nullptr, * dscr = nullptr;
    ck(cudaMalloc(&dout, size_t(nq) * n * D * 4), "out");
    ck(cudaMalloc(&dscr, size_t(K::attn_res_norm_scratch_words(K::kSmallN, D)) * 4), "scratch");
    ck(cudaMemset(dscr, 0, size_t(K::attn_res_norm_scratch_words(K::kSmallN, D)) * 4), "scratch zero");
    K::attn_res(st, dq, nq, dout, n, D, dscr, 0);
    ck(cudaDeviceSynchronize(), "attn_res");
    const auto out = down(dout, size_t(nq) * n * D);
    double err = 0;
    for (int t = 0; t < n; ++t) {
        std::vector<double> inv(cnt);
        for (int j = 0; j < cnt; ++j) { double ss = 0; for (int c = 0; c < D; ++c) ss += double(S[j][size_t(t) * D + c]) * S[j][size_t(t) * D + c]; inv[j] = 1.0 / std::sqrt(ss / D + 1e-6); }
        for (int i = 0; i < nq; ++i) {
            std::vector<double> lgt(cnt);
            double m = -1e300, l = 0;
            for (int j = 0; j < cnt; ++j) { double d = 0; for (int c = 0; c < D; ++c) d += double(q[size_t(i) * D + c]) * S[j][size_t(t) * D + c]; lgt[j] = d * inv[j] / std::sqrt(double(D)); m = std::max(m, lgt[j]); }
            for (int j = 0; j < cnt; ++j) l += std::exp(lgt[j] - m);
            for (int c = 0; c < D; ++c) {
                double y = 0;
                for (int j = 0; j < cnt; ++j) y += std::exp(lgt[j] - m) / l * S[j][size_t(t) * D + c] * inv[j];
                err = std::max(err, std::fabs(out[(size_t(i) * n + t) * D + c] - y) / (std::fabs(y) + 0.01));
            }
        }
    }
    report("depth attention (5 states, 3 queries, 3 tokens)", err, 1e-4, true);
    {   // fused with the three input norms: F32 out vs attn_res, BF16 norms vs a CPU rms of attn_res's output
        const auto wv = randv(size_t(nq) * D, 0.3f);
        float * dw = up(wv), * dout2 = nullptr;
        ck(cudaMalloc(&dout2, size_t(nq) * n * D * 4), "out2");
        bf16 * dy = nullptr;
        ck(cudaMalloc(&dy, size_t(nq) * n * D * 2), "y");
        const float * w[3] = {dw, dw + D, dw + 2 * D};
        bf16 * y[3] = {dy, dy + size_t(n) * D, dy + size_t(2) * n * D};
        K::attn_res_norm(st, dq, nq, dout2, w, y, n, D, 1e-6f, dscr, 0);
        ck(cudaDeviceSynchronize(), "attn_res_norm");
        const auto o2 = down(dout2, out.size());
        const auto yv = from_bf(down(dy, size_t(nq) * n * D));
        double e1 = 0, e2 = 0;
        for (size_t i = 0; i < o2.size(); ++i) e1 = std::max(e1, double(std::fabs(o2[i] - out[i])) / (std::fabs(out[i]) + 0.01));
        for (int i = 0; i < nq; ++i)
            for (int t = 0; t < n; ++t) {
                const float * x = out.data() + (size_t(i) * n + t) * D;
                double ss = 0;
                for (int c = 0; c < D; ++c) ss += double(x[c]) * x[c];
                const double r = 1.0 / std::sqrt(ss / D + 1e-6);
                for (int c = 0; c < D; ++c) {
                    const double ref = wv[size_t(i) * D + c] * x[c] * r;
                    e2 = std::max(e2, std::fabs(yv[(size_t(i) * n + t) * D + c] - ref) / (std::fabs(ref) + 0.01));
                }
            }
        report("depth attention + input norms (fused)", std::max(e1 * 100, e2), 1e-2, true);
        cudaFree(dw); cudaFree(dout2); cudaFree(dy);
    }
    for (auto p : dS) cudaFree(p);
    cudaFree(dq); cudaFree(dout); cudaFree(dscr);
}

// message gates: z = bf16(cat(rms(h), rms(m))) and y = (1 - tanh(F g)) h + tanh(G g) m, against double references;
// the token-tiled prompt kernel (n >= 8) must match the per-token one bitwise (verify == plain decode)
void test_msg_gates() {
    const int D = 2304, rank = 64, n = 12;
    const auto h = randv(size_t(n) * D, 1.0f), m = randv(size_t(n) * D, 1.0f), gv = randv(size_t(n) * rank, 0.3f);
    const auto ft = randv(size_t(rank) * D, 0.05f), gt = randv(size_t(rank) * D, 0.05f);
    float * dhf = up(h), * dmf = up(m), * dy = nullptr, * dy1 = nullptr;
    ck(cudaMalloc(&dy, size_t(n) * D * 4), "y");
    ck(cudaMalloc(&dy1, size_t(n) * D * 4), "y1");
    bf16 * dz = nullptr, * dg = up(to_bf(gv)), * dft = up(to_bf(ft)), * dgt = up(to_bf(gt));
    ck(cudaMalloc(&dz, size_t(n) * 2 * D * 2), "z");
    K::msg_gate_in(dhf, dmf, dz, n, D, 1e-6f, 0);
    K::msg_gate_out(dg, dft, dgt, dhf, dmf, dy, n, D, rank, 0);
    ck(cudaDeviceSynchronize(), "msg gates");
    const auto z = from_bf(down(dz, size_t(n) * 2 * D)), y = down(dy, size_t(n) * D);
    double e1 = 0, e2 = 0;
    for (int t = 0; t < n; ++t) {
        double sa = 0, sb = 0;
        for (int c = 0; c < D; ++c) { sa += double(h[size_t(t) * D + c]) * h[size_t(t) * D + c]; sb += double(m[size_t(t) * D + c]) * m[size_t(t) * D + c]; }
        const double ra = 1.0 / std::sqrt(sa / D + 1e-6), rb = 1.0 / std::sqrt(sb / D + 1e-6);
        for (int c = 0; c < D; ++c) {
            e1 = std::max(e1, std::fabs(z[size_t(2 * t) * D + c] - bf(float(h[size_t(t) * D + c] * ra))) / 0.02);
            e1 = std::max(e1, std::fabs(z[size_t(2 * t + 1) * D + c] - bf(float(m[size_t(t) * D + c] * rb))) / 0.02);
        }
        for (int c = 0; c < D; ++c) {
            double f = 0, a = 0;
            for (int r = 0; r < rank; ++r) { f += double(ft[size_t(r) * D + c]) * gv[size_t(t) * rank + r]; a += double(gt[size_t(r) * D + c]) * gv[size_t(t) * rank + r]; }
            const double ref = (1.0 - std::tanh(f)) * h[size_t(t) * D + c] + std::tanh(a) * m[size_t(t) * D + c];
            e2 = std::max(e2, std::fabs(y[size_t(t) * D + c] - ref) / (std::fabs(ref) + 0.01));
        }
    }
    report("msg gate in / out vs CPU reference", std::max(e1, e2), 1e-3, true);
    bool same = true;  // token t of the tiled (n = 12) call == a 1-token call
    for (int t = 0; t < n && same; ++t) {
        K::msg_gate_out(dg + size_t(t) * rank, dft, dgt, dhf + size_t(t) * D, dmf + size_t(t) * D, dy1, 1, D, rank, 0);
        ck(cudaDeviceSynchronize(), "msg gate out 1");
        const auto y1 = down(dy1, size_t(D));
        for (int c = 0; c < D; ++c) if (std::memcmp(&y1[c], &y[size_t(t) * D + c], 4) != 0) { same = false; break; }
    }
    report("msg gate out: tiled n-token == 1-token", 0.0, 0.0, same);
    cudaFree(dhf); cudaFree(dmf); cudaFree(dy); cudaFree(dy1); cudaFree(dz); cudaFree(dg); cudaFree(dft); cudaFree(dgt);
}

// grouped prompt GEMM: 120 tokens x top-8 of 16 experts with a skewed router (experts with 0, 1 and > 64 assignments)
void test_grouped(int fmt) {
    const int E = 16, topk = 8, D = 384, I = 256, n = 120, A = n * topk;
    const float gc = 4.0f, uc = 25.0f;
    ExpertBank G, U, Dn;
    make_bank(G, fmt, E, I, D); make_bank(U, fmt, E, I, D); make_bank(Dn, fmt, E, D, I);
    const auto X = randv(size_t(n) * D, 1.0f);
    std::vector<int> ids(A);
    for (int t = 0; t < n; ++t) {
        std::vector<int> p;
        for (int e = 0; e < E; ++e) if (e != 5) p.push_back(e);  // expert 5 never routed
        std::shuffle(p.begin() + 1, p.end(), rng);                // expert 0 in every token's top-8 (120 > 64)
        for (int k = 0; k < topk; ++k) ids[t * topk + k] = p[k];
    }
    bf16 * dx = up(to_bf(X)), * dact = nullptr, * dy = nullptr;
    int * dids = up(ids), * order = nullptr, * counts = nullptr, * offsets = nullptr, * tiles = nullptr;
    ck(cudaMalloc(&dact, size_t(A) * I * 2), "act"); ck(cudaMalloc(&dy, size_t(A) * D * 2), "y");
    ck(cudaMalloc(&order, A * 4), "order"); ck(cudaMalloc(&counts, E * 4), "counts"); ck(cudaMalloc(&offsets, E * 4), "offsets");
    ck(cudaMalloc(&tiles, (E + 1) * 4), "tiles");
    K::group_assignments(dids, A, E, order, counts, offsets, tiles, 0);
    K::grouped_gate_up(G.m, U.m, I, D, dx, A, E, topk, order, counts, offsets, tiles, dact, gc, uc, 0);
    K::grouped_down(Dn.m, D, I, dact, A, E, order, counts, offsets, tiles, dy, 0);
    ck(cudaDeviceSynchronize(), "grouped");
    const auto act = from_bf(down(dact, size_t(A) * I)), y = from_bf(down(dy, size_t(A) * D));
    const auto ord = down(order, size_t(A));
    bool grouping = true;  // order: grouped by expert, ascending inside
    for (int a = 1; a < A; ++a) grouping = grouping && (ids[ord[a - 1]] < ids[ord[a]] || (ids[ord[a - 1]] == ids[ord[a]] && ord[a - 1] < ord[a]));
    double ea = 0, ey = 0;
    for (int a = 0; a < A; ++a) {
        const int e = ids[a], t = a / topk;
        for (int i = 0; i < I; ++i) {
            double g = 0, u = 0;
            for (int c = 0; c < D; ++c) {
                g += double(bf(G.w[(size_t(e) * I + i) * D + c])) * X[size_t(t) * D + c];
                u += double(bf(U.w[(size_t(e) * I + i) * D + c])) * X[size_t(t) * D + c];
            }
            g = bf(float(g)); u = bf(float(u));
            const double ref = gc * std::tanh(g / gc) / (1.0 + std::exp(-g)) * uc * std::tanh(u / uc);
            ea = std::max(ea, std::fabs(act[size_t(a) * I + i] - ref) / (std::fabs(ref) + 0.05));
        }
        for (int r = 0; r < D; ++r) {
            double d = 0;
            for (int i = 0; i < I; ++i) d += double(bf(Dn.w[(size_t(e) * D + r) * I + i])) * act[size_t(a) * I + i];
            ey = std::max(ey, std::fabs(y[size_t(a) * D + r] - d) / (std::fabs(d) + 0.05));
        }
    }
    // NVF4 weights (e2m1 x e4m3 x F32 global) are decoded to BF16 for the MMA, so the reference rounds them the same way
    // gate / up are rounded to BF16 before the SwiGLU: an F32 sum on the other side of a rounding boundary than the double
    // reference moves one value by a BF16 ulp (0.4 %), which the softcapped SwiGLU can amplify to ~2-3 % of its output
    report(fmt == 2 ? "grouped GEMM NVF4 gate/up (+ order)" : "grouped GEMM MXFP8 gate/up (+ order)", grouping ? ea : 1.0, 4e-2, true);
    report(fmt == 2 ? "grouped GEMM NVF4 down" : "grouped GEMM MXFP8 down", ey, 2e-2, true);
    cudaFree(dx); cudaFree(dact); cudaFree(dy); cudaFree(dids);
    cudaFree(order); cudaFree(counts); cudaFree(offsets); cudaFree(tiles);
}

// BF16 prompt copies (big-card path): the decode equals the kernel's own rounding (bitwise), and gather + per-expert
// cuBLAS + SwiGLU + scatter matches the same CPU reference as test_grouped within its tolerances
// dispatch threshold: chunks under ~4K tokens keep the WMMA grouped path (1K prefill regressed 1272 -> 1102 on cuBLAS)
void test_grouped_bf16_threshold() {
    const bool wmma = !K::use_grouped_bf16(1) && !K::use_grouped_bf16(1024) && !K::use_grouped_bf16(3840);
    const bool cublas = K::use_grouped_bf16(4096) && K::use_grouped_bf16(8192);
    report("grouped bf16 dispatch: WMMA < 4096 <= cuBLAS", (wmma && cublas) ? 0.0 : 1.0, 0.0, true);
}

void test_grouped_bf16(int fmt) {
    const int E = 16, topk = 8, D = 384, I = 256, n = 120, A = n * topk;
    const float gc = 4.0f, uc = 25.0f;
    ExpertBank G, U, Dn;
    make_bank(G, fmt, E, I, D); make_bank(U, fmt, E, I, D); make_bank(Dn, fmt, E, D, I);
    // decode to BF16 and check against the reference decoded in the kernel's order: (e2m1 x e4m3) x global, F32, then BF16
    bf16 * dbgu = nullptr, * dbdn = nullptr;
    ck(cudaMalloc(&dbgu, size_t(E) * 2 * I * D * 2), "bgu"); ck(cudaMalloc(&dbdn, size_t(E) * D * I * 2), "bdn");
    K::experts_decode_bf16(G.m, E, I, D, 2 * I, dbgu, 0);
    K::experts_decode_bf16(U.m, E, I, D, 2 * I, dbgu + size_t(I) * D, 0);
    K::experts_decode_bf16(Dn.m, E, D, I, D, dbdn, 0);
    ck(cudaDeviceSynchronize(), "decode bf16");
    const auto bgu_h = down(dbgu, size_t(E) * 2 * I * D), bdn_h = down(dbdn, size_t(E) * D * I);
    bool dec = true;
    for (int which = 0; which < 3 && dec; ++which) {
        const ExpertBank & b = which == 0 ? G : (which == 1 ? U : Dn);
        const int rows = which == 2 ? D : I, cols = which == 2 ? I : D;
        const size_t rb = fmt == 2 ? size_t(cols) / 2 : size_t(cols + cols / 32), ab = size_t(cols) / 16 + 4;
        const bf16 * ld = which == 2 ? bdn_h.data() : bgu_h.data() + (which == 1 ? size_t(I) * D : 0);
        const int ldrows = which == 2 ? rows : 2 * rows;
        for (int e = 0; e < E && dec; ++e) {
            const uint8_t * w = b.bytes.data() + e * b.m.stride, * a = b.aux.data() + e * b.m.aux_stride;
            float g = 0.f;
            for (int r = 0; r < rows && dec; ++r) {
                if (fmt == 2) std::memcpy(&g, a + size_t(r) * ab + cols / 16, 4);
                for (int c = 0; c < cols; ++c) {
                    float v;
                    if (fmt == 2) v = (kE2M1[c % 2 ? w[size_t(r) * rb + c / 2] >> 4 : w[size_t(r) * rb + c / 2] & 15] * e4m3(a[size_t(r) * ab + c / 16])) * g;
                    else v = e4m3(w[size_t(r) * rb + c]) * std::ldexp(1.0f, int(w[size_t(r) * rb + cols + c / 32]) - 127);
                    dec = dec && ld[size_t((size_t(e) * ldrows + r) * cols + c)] == __float2bfloat16(v);
                }
            }
        }
    }
    report(fmt == 2 ? "experts BF16 copy NVF4 (bitwise decode)" : "experts BF16 copy MXFP8 (bitwise decode)", 0.0, 1.0, dec);
    // the cuBLAS prompt path vs the CPU reference (same inputs and tolerances as test_grouped)
    const auto X = randv(size_t(n) * D, 1.0f);
    std::vector<int> ids(A);
    for (int t = 0; t < n; ++t) {
        std::vector<int> p;
        for (int e = 0; e < E; ++e) if (e != 5) p.push_back(e);
        std::shuffle(p.begin() + 1, p.end(), rng);
        for (int k = 0; k < topk; ++k) ids[t * topk + k] = p[k];
    }
    std::vector<float> wts(A);
    for (int a = 0; a < A; ++a) wts[a] = 0.5f + float(rng() % 100) / 100.0f;
    bf16 * dx = up(to_bf(X)), * dact = nullptr, * dout = nullptr, * dxg = nullptr, * dgu = nullptr, * dyg = nullptr;
    int * dids = up(ids), * order = nullptr, * counts = nullptr, * offsets = nullptr, * tiles = nullptr, * dinv = nullptr;
    float * dw = up(wts);
    ck(cudaMalloc(&dact, size_t(A) * I * 2), "act"); ck(cudaMalloc(&dout, size_t(n) * D * 2), "out");
    ck(cudaMalloc(&dxg, size_t(A) * D * 2), "xg"); ck(cudaMalloc(&dgu, size_t(A) * 2 * I * 2), "gu"); ck(cudaMalloc(&dyg, size_t(A) * D * 2), "yg");
    ck(cudaMalloc(&dinv, A * 4), "inv");
    ck(cudaMalloc(&order, A * 4), "order"); ck(cudaMalloc(&counts, E * 4), "counts"); ck(cudaMalloc(&offsets, E * 4), "offsets");
    ck(cudaMalloc(&tiles, (E + 1) * 4), "tiles");
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasCreate failed\n"); ok = false; }
    K::group_assignments(dids, A, E, order, counts, offsets, tiles, 0);
    const auto hc = down(counts, size_t(E)), ho = down(offsets, size_t(E));
    K::grouped_gate_up_bf16(dbgu, I, D, dx, A, E, topk, order, hc.data(), ho.data(), dxg, dgu, dact, gc, uc, blas, 0);
    K::grouped_down_bf16(dbdn, D, I, dact, A, E, hc.data(), ho.data(), dyg, blas, 0);
    K::moe_combine_grouped(dyg, order, A, dw, dout, n, D, topk, dinv, 0);
    ck(cudaDeviceSynchronize(), "grouped bf16");
    const auto act = from_bf(down(dact, size_t(A) * I)), yg = from_bf(down(dyg, size_t(A) * D)), out = from_bf(down(dout, size_t(n) * D));
    const auto ord = down(order, size_t(A));
    std::vector<int> inv(A);  // act / yg are in grouped order: inv[a] = the grouped row of assignment a
    for (int j = 0; j < A; ++j) inv[ord[j]] = j;
    double ea = 0, ey = 0, ec = 0;
    std::vector<double> y(size_t(n) * D, 0.0);
    for (int a = 0; a < A; ++a) {
        const int e = ids[a], t = a / topk;
        for (int i = 0; i < I; ++i) {
            double g = 0, u = 0;
            for (int c = 0; c < D; ++c) {
                g += double(bf(G.w[(size_t(e) * I + i) * D + c])) * X[size_t(t) * D + c];
                u += double(bf(U.w[(size_t(e) * I + i) * D + c])) * X[size_t(t) * D + c];
            }
            g = bf(float(g)); u = bf(float(u));
            const double ref = gc * std::tanh(g / gc) / (1.0 + std::exp(-g)) * uc * std::tanh(u / uc);
            ea = std::max(ea, std::fabs(act[size_t(inv[a]) * I + i] - ref) / (std::fabs(ref) + 0.05));
        }
        for (int r = 0; r < D; ++r) {
            double d = 0;
            for (int i = 0; i < I; ++i) d += double(bf(Dn.w[(size_t(e) * D + r) * I + i])) * act[size_t(inv[a]) * I + i];
            ey = std::max(ey, std::fabs(yg[size_t(inv[a]) * D + r] - d) / (std::fabs(d) + 0.05));
            y[size_t(t) * D + r] += yg[size_t(inv[a]) * D + r] * wts[a];  // combine on the GPU's own yg: wiring check
        }
    }
    for (int t = 0; t < n; ++t)
        for (int r = 0; r < D; ++r) ec = std::max(ec, std::fabs(out[size_t(t) * D + r] - y[size_t(t) * D + r]) / (std::fabs(y[size_t(t) * D + r]) + 0.05));
    report(fmt == 2 ? "grouped cuBLAS NVF4 gate/up + SwiGLU" : "grouped cuBLAS MXFP8 gate/up + SwiGLU", ea, 4e-2, true);
    report(fmt == 2 ? "grouped cuBLAS NVF4 down" : "grouped cuBLAS MXFP8 down", ey, 2e-2, true);
    report(fmt == 2 ? "grouped cuBLAS NVF4 combine (grouped order)" : "grouped cuBLAS MXFP8 combine (grouped order)", ec, 1e-2, true);
    cublasDestroy(blas);
    cudaFree(dx); cudaFree(dact); cudaFree(dout); cudaFree(dxg); cudaFree(dgu); cudaFree(dyg); cudaFree(dids); cudaFree(dw); cudaFree(dinv);
    cudaFree(order); cudaFree(counts); cudaFree(offsets); cudaFree(tiles); cudaFree(dbgu); cudaFree(dbdn);
}

// prompt-chunk convolutions (parallel over tokens) == the same tokens one step at a time, outputs and final state bitwise
void test_conv() {
    const int C = 1024, n = 20;
    const auto x = randv(size_t(n) * C, 1.0f), w = randv(size_t(C) * 4, 0.3f), st = randv(size_t(C) * 3, 1.0f), stat = randv(size_t(4) * C, 0.3f),
               dyn = randv(size_t(n) * (C / 128) * 4, 0.3f), res = randv(size_t(n) * C, 1.0f);
    bf16 * dx = up(to_bf(x)), * ddyn = up(to_bf(dyn)), * dy16 = nullptr;
    float * dw = up(w), * dstat = up(stat), * dres = up(res), * dy = nullptr;
    ck(cudaMalloc(&dy, size_t(n) * C * 4), "y"); ck(cudaMalloc(&dy16, size_t(n) * C * 2), "y16");
    bool bit = true;
    for (int kind = 0; kind < 2; ++kind) {
        float * s1 = up(st), * s2 = up(st);
        if (kind == 0) K::conv_static(dx, dw, s1, dy, dy16, n, C, 0, dres, true);
        else K::conv_dynamic(dx, dstat, ddyn, s1, dy16, n, C, 0);
        ck(cudaDeviceSynchronize(), "conv");
        const auto a32 = down(dy, size_t(n) * C);
        const auto a16 = from_bf(down(dy16, size_t(n) * C));
        for (int t = 0; t < n; ++t) {
            if (kind == 0) K::conv_static(dx + size_t(t) * C, dw, s2, dy + size_t(t) * C, dy16 + size_t(t) * C, 1, C, 0, dres + size_t(t) * C, true);
            else K::conv_dynamic(dx + size_t(t) * C, dstat, ddyn + size_t(t) * (C / 128) * 4, s2, dy16 + size_t(t) * C, 1, C, 0);
        }
        ck(cudaDeviceSynchronize(), "conv steps");
        const auto b16 = from_bf(down(dy16, size_t(n) * C));
        bit = bit && std::memcmp(a16.data(), b16.data(), a16.size() * 4) == 0 && down(s1, size_t(C) * 3) == down(s2, size_t(C) * 3);
        if (kind == 0) bit = bit && down(dy, size_t(n) * C) == a32;
        cudaFree(s1); cudaFree(s2);
    }
    report("prompt convolutions == per-step (static, dynamic)", 0.0, 1.0, bit);
    {   // spec-decode rollback: 8 tokens with the input history, rolled back to keep k == the first k tokens alone, bitwise
        bool rb = true;
        __nv_bfloat16 * hist = nullptr;
        ck(cudaMalloc(&hist, size_t(8) * C * 2), "hist");
        for (int keep = 1; keep <= 8; ++keep) {
            float * sa = up(st), * sb = up(st), * snap = up(st);
            K::conv_static(dx, dw, sa, dy, dy16, 8, C, 0, nullptr, false, hist);
            K::conv_rollback(snap, hist, sa, keep, C, 0);
            K::conv_static(dx, dw, sb, dy, dy16, keep, C, 0);
            ck(cudaDeviceSynchronize(), "rollback");
            rb = rb && down(sa, size_t(C) * 3) == down(sb, size_t(C) * 3);
            cudaFree(sa); cudaFree(sb); cudaFree(snap);
        }
        cudaFree(hist);
        report("conv rollback to keep 1..8 == keep tokens alone", 0.0, 1.0, rb);
    }
    cudaFree(dx); cudaFree(ddyn); cudaFree(dy16); cudaFree(dw); cudaFree(dstat); cudaFree(dres); cudaFree(dy);
}

// first use of a token count inside graph capture: the decode path captures attn_res_norm on the first n it sees, so a
// scratch allocation there runs while the stream is capturing. Compare a captured call (and its replays) with the CPU
// reference; the scratch counters must be back at zero for every replay.
void test_attn_res_capture(int n) {
    const int D = 2304, nq = 3, cnt = 5;
    std::vector<std::vector<float>> S(cnt);
    std::vector<float *> dS(cnt);
    K::States st{};
    st.count = cnt;
    for (int j = 0; j < cnt; ++j) { S[j] = randv(size_t(n) * D, 1.0f + j); dS[j] = up(S[j]); st.s[j] = dS[j]; }
    const auto q = randv(size_t(nq) * D, 0.1f);
    const auto wv = randv(size_t(nq) * D, 0.3f);
    float * dq = up(q), * dw = up(wv), * dout = nullptr, * dscr = nullptr;
    bf16 * dy = nullptr;
    ck(cudaMalloc(&dout, size_t(nq) * n * D * 4), "out");
    ck(cudaMalloc(&dy, size_t(nq) * n * D * 2), "y");
    ck(cudaMalloc(&dscr, size_t(K::attn_res_norm_scratch_words(K::kSmallN, D)) * 4), "scratch");
    ck(cudaMemset(dscr, 0, size_t(K::attn_res_norm_scratch_words(K::kSmallN, D)) * 4), "scratch zero");
    const float * w[3] = {dw, dw + D, dw + 2 * D};
    bf16 * y[3] = {dy, dy + size_t(n) * D, dy + size_t(2) * n * D};

    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    bool captured = false;
    std::string thrown;
    cudaGraph_t g = nullptr;
    const cudaError_t begin = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    if (begin == cudaSuccess) {
        try {
            K::attn_res_norm(st, dq, nq, dout, w, y, n, D, 1e-6f, dscr, s);
            captured = true;
        } catch (const std::exception & e) { thrown = e.what(); }
        const cudaError_t end = cudaStreamEndCapture(s, &g);
        if (!captured || end != cudaSuccess) {
            const cudaError_t last = cudaGetLastError();
            std::printf("attn_res_norm first-use capture (n=%d) FAIL: %s; end capture: %s; CUDA: %s\n", n,
                        thrown.empty() ? "stream still capturing" : thrown.c_str(), cudaGetErrorString(end), cudaGetErrorString(last));
            ok = false;
        }
    } else {
        std::printf("attn_res_norm first-use capture (n=%d) FAIL: begin capture: %s\n", n, cudaGetErrorString(begin));
        ok = false;
    }

    if (captured && g) {
        cudaGraphExec_t ex = nullptr;
        if (cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) {
            std::printf("attn_res_norm first-use capture (n=%d) FAIL: instantiate: %s\n", n, cudaGetErrorString(cudaGetLastError()));
            ok = false;
        } else {
            std::vector<float> first;
            bool replay_equal = true;
            double err = 0;
            for (int r = 0; r < 3; ++r) {
                ck(cudaGraphLaunch(ex, s), "graph launch");
                ck(cudaStreamSynchronize(s), "graph sync");
                const auto out = down(dout, size_t(nq) * n * D);
                if (r == 0) {
                    first = out;
                    for (int t = 0; t < n; ++t) {
                        std::vector<double> inv(cnt);
                        for (int j = 0; j < cnt; ++j) { double ss = 0; for (int c = 0; c < D; ++c) ss += double(S[j][size_t(t) * D + c]) * S[j][size_t(t) * D + c]; inv[j] = 1.0 / std::sqrt(ss / D + 1e-6); }
                        for (int i = 0; i < nq; ++i) {
                            std::vector<double> lgt(cnt);
                            double mx = -1e300, l = 0;
                            for (int j = 0; j < cnt; ++j) { double d = 0; for (int c = 0; c < D; ++c) d += double(q[size_t(i) * D + c]) * S[j][size_t(t) * D + c]; lgt[j] = d * inv[j] / std::sqrt(double(D)); mx = std::max(mx, lgt[j]); }
                            for (int j = 0; j < cnt; ++j) l += std::exp(lgt[j] - mx);
                            for (int c = 0; c < D; ++c) {
                                double ref = 0;
                                for (int j = 0; j < cnt; ++j) ref += std::exp(lgt[j] - mx) / l * S[j][size_t(t) * D + c] * inv[j];
                                err = std::max(err, std::fabs(out[(size_t(i) * n + t) * D + c] - ref) / (std::fabs(ref) + 0.01));
                            }
                        }
                    }
                } else {
                    replay_equal = replay_equal && std::memcmp(first.data(), out.data(), first.size() * 4) == 0;
                }
            }
            char what[96];
            std::snprintf(what, sizeof(what), "attn_res_norm first-use graph capture (n=%d)", n);
            std::printf("%-46s worst rel err %.2e (tol 1e-4), replays bitwise: %s %s\n", what, err, replay_equal ? "yes" : "NO",
                        (err <= 1e-4 && replay_equal) ? "ok" : "FAIL");
            ok = ok && err <= 1e-4 && replay_equal;
            cudaGraphExecDestroy(ex);
        }
        cudaGraphDestroy(g);
    }
    cudaStreamDestroy(s);
    cudaFree(dq); cudaFree(dw); cudaFree(dout); cudaFree(dy); cudaFree(dscr);
    for (auto p : dS) cudaFree(p);
}
}  // namespace

// small cards: the TQT prompt path decodes and multiplies G experts at a time (expert_slice + grouped_gather +
// grouped_gate_up_part_bf16 + grouped_down_bf16 on the group's hc / ho slices) == the whole layer at once, bitwise;
// a group with no routed rows is skipped. choose_tqt_group keeps every expert resident when a smaller group allows it.
// dense TQT (q / k / v / o of a dense-TQT overlay): the model runs a dense matrix as one TQT expert through the down
// kernel (id 0, weight 1) on the input rotated per 128, and prompt chunks decode it to BF16 for the GEMM; both must equal
// W x with W the record's original-basis weights
void test_tqt_dense(int fmt) {
    const int rows = 3072, cols = 2304, n = 3;
    ExpertBank W;
    make_bank_tqt(W, 1, rows, cols, fmt);
    const auto X = randv(size_t(n) * cols, 1.0f);
    const auto Xb = to_bf(X);
    bf16 * dx = up(Xb), * dy = nullptr, * dw16 = nullptr;
    __half * dxh = nullptr;
    std::vector<int> ids(n, 0); std::vector<float> wts(n, 1.0f);
    int * dids = up(ids); float * dwts = up(wts);
    ck(cudaMalloc(&dxh, size_t(n) * cols * 2), "xh"); ck(cudaMalloc(&dy, size_t(n) * rows * 2), "y");
    ck(cudaMalloc(&dw16, size_t(rows) * cols * 2), "w16");
    K::rotate_h(dx, dxh, cols, n, 0);
    K::expert_down(W.m, rows, cols, dx, dids, dwts, dy, n, 1, 0, dxh);
    K::experts_decode_bf16(W.m, 1, rows, cols, rows, dw16, 0);
    ck(cudaDeviceSynchronize(), "dense tqt");
    const auto y = from_bf(down(dy, size_t(n) * rows));
    const auto w16 = from_bf(down(dw16, size_t(rows) * cols));
    double err = 0, derr = 0;
    for (int t = 0; t < n; ++t)
        for (int r = 0; r < rows; ++r) {
            double ref = 0, mag = 0;
            for (int c = 0; c < cols; ++c) {
                const double xv = from_bf(std::vector<bf16>{Xb[size_t(t) * cols + c]})[0];
                ref += double(W.w[size_t(r) * cols + c]) * xv; mag += std::fabs(double(W.w[size_t(r) * cols + c]) * xv);
            }
            err = std::max(err, std::fabs(y[size_t(t) * rows + r] - ref) / (std::max(std::fabs(ref), mag / std::sqrt(double(cols))) + 1e-6));
        }
    for (size_t i = 0; i < w16.size(); ++i) derr = std::max(derr, double(std::fabs(w16[i] - W.w[i])) / (std::fabs(W.w[i]) + 1e-3));
    report(fmt == 3 ? "dense TQT 3072x2304: one-expert down == W x" : fmt == 5 ? "dense TQT3 3072x2304: one-expert down == W x"
                                                                              : "dense TQT2 3072x2304: one-expert down == W x", err, 2e-2, true);
    report("dense TQT decode to BF16 (prompt path) == record weights", derr, 1e-2, true);
    cudaFree(dx); cudaFree(dy); cudaFree(dw16); cudaFree(dxh); cudaFree(dids); cudaFree(dwts);
}
void test_tqt_groups(int fmt) {
    const int E = 16, G = 4, topk = 4, D = 256, I = 128, n = 96, A = n * topk;
    const float gc = 4.0f, uc = 25.0f;
    ExpertBank Gt, Ut, Dt;
    make_bank_tqt(Gt, E, I, D, fmt); make_bank_tqt(Ut, E, I, D, fmt); make_bank_tqt(Dt, E, D, I, fmt);
    const auto X = randv(size_t(n) * D, 1.0f);
    std::vector<int> ids(A);
    for (int t = 0; t < n; ++t) {  // experts 8-11 (one whole group) never routed, expert 2 never routed
        std::vector<int> p;
        for (int e = 0; e < E; ++e) if (e != 2 && (e < 8 || e > 11)) p.push_back(e);
        std::shuffle(p.begin(), p.end(), rng);
        for (int k = 0; k < topk; ++k) ids[t * topk + k] = p[k];
    }
    bf16 * dx = up(to_bf(X)), * dbgu = nullptr, * dbdn = nullptr, * sgu = nullptr, * sdn = nullptr;
    bf16 * dxg = nullptr, * dgu = nullptr, * dact = nullptr, * dyg = nullptr, * dact2 = nullptr, * dyg2 = nullptr;
    int * dids = up(ids), * order = nullptr, * counts = nullptr, * offsets = nullptr, * tiles = nullptr;
    ck(cudaMalloc(&dbgu, size_t(E) * 2 * I * D * 2), "bgu"); ck(cudaMalloc(&dbdn, size_t(E) * D * I * 2), "bdn");
    ck(cudaMalloc(&sgu, size_t(G) * 2 * I * D * 2), "sgu"); ck(cudaMalloc(&sdn, size_t(G) * D * I * 2), "sdn");
    ck(cudaMalloc(&dxg, size_t(A) * D * 2), "xg"); ck(cudaMalloc(&dgu, size_t(A) * 2 * I * 2), "gu");
    ck(cudaMalloc(&dact, size_t(A) * I * 2), "act"); ck(cudaMalloc(&dyg, size_t(A) * D * 2), "yg");
    ck(cudaMalloc(&dact2, size_t(A) * I * 2), "act2"); ck(cudaMalloc(&dyg2, size_t(A) * D * 2), "yg2");
    ck(cudaMalloc(&order, A * 4), "order"); ck(cudaMalloc(&counts, E * 4), "counts"); ck(cudaMalloc(&offsets, E * 4), "offsets");
    ck(cudaMalloc(&tiles, (E + 1) * 4), "tiles");
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) { std::printf("cublasCreate failed\n"); ok = false; return; }
    K::group_assignments(dids, A, E, order, counts, offsets, tiles, 0);
    const auto hc = down(counts, size_t(E)), ho = down(offsets, size_t(E));
    // whole layer
    K::experts_decode_bf16(Gt.m, E, I, D, 2 * I, dbgu, 0);
    K::experts_decode_bf16(Ut.m, E, I, D, 2 * I, dbgu + size_t(I) * D, 0);
    K::experts_decode_bf16(Dt.m, E, D, I, D, dbdn, 0);
    K::grouped_gate_up_bf16(dbgu, I, D, dx, A, E, topk, order, hc.data(), ho.data(), dxg, dgu, dact, gc, uc, blas, 0);
    K::grouped_down_bf16(dbdn, D, I, dact, A, E, hc.data(), ho.data(), dyg, blas, 0);
    ck(cudaDeviceSynchronize(), "whole");
    // G experts at a time (the model's loop)
    ck(cudaMemset(dact2, 0, size_t(A) * I * 2), "act2"); ck(cudaMemset(dyg2, 0, size_t(A) * D * 2), "yg2");
    K::grouped_gather(dx, A, order, topk, dxg, D, 0);
    int groups_run = 0;
    for (int e0 = 0; e0 < E; e0 += G) {
        int rows = 0;
        for (int e = e0; e < e0 + G; ++e) rows += hc[e];
        if (rows == 0) continue;
        ++groups_run;
        K::experts_decode_bf16(K::expert_slice(Gt.m, e0), G, I, D, 2 * I, sgu, 0);
        K::experts_decode_bf16(K::expert_slice(Ut.m, e0), G, I, D, 2 * I, sgu + size_t(I) * D, 0);
        K::experts_decode_bf16(K::expert_slice(Dt.m, e0), G, D, I, D, sdn, 0);
        K::grouped_gate_up_part_bf16(sgu, I, D, dxg, G, hc.data() + e0, ho.data() + e0, dgu, dact2, gc, uc, blas, 0);
        K::grouped_down_bf16(sdn, D, I, dact2, A, G, hc.data() + e0, ho.data() + e0, dyg2, blas, 0);
        ck(cudaDeviceSynchronize(), "group");
    }
    const auto a1 = down(dact, size_t(A) * I), a2 = down(dact2, size_t(A) * I), y1 = down(dyg, size_t(A) * D), y2 = down(dyg2, size_t(A) * D);
    const bool same = std::memcmp(a1.data(), a2.data(), a1.size() * 2) == 0 && std::memcmp(y1.data(), y2.data(), y1.size() * 2) == 0;
    report(fmt == 3 ? "TQT prompt experts in groups == whole layer (bitwise)" : "TQT3 prompt experts in groups == whole layer (bitwise)", 0.0, 1.0,
           same && groups_run == E / G - 1);
    cublasDestroy(blas);
    cudaFree(dx); cudaFree(dbgu); cudaFree(dbdn); cudaFree(sgu); cudaFree(sdn); cudaFree(dxg); cudaFree(dgu); cudaFree(dact); cudaFree(dyg);
    cudaFree(dact2); cudaFree(dyg2); cudaFree(dids); cudaFree(order); cudaFree(counts); cudaFree(offsets); cudaFree(tiles);
}

void test_tqt_group_choice() {
    const size_t GB = size_t(1) << 30, per = size_t(3) * 1152 * 2304 * 2;  // Inkling: gate + up + down of one expert in BF16 (15.9 MB)
    // 16 GB card (5060 Ti): 15.2 GB free, 12.61 GB of TQT experts, 1.5 GB reserve + 0.23 GB chunk buffers:
    // the whole layer (128 x 15.9 MB = 2.04 GB) would push experts out; 64 does not
    const int g16 = K::choose_tqt_group(128, size_t(15.2 * GB), size_t(12.61 * GB), size_t(1.73 * GB), per);
    const int g24 = K::choose_tqt_group(128, size_t(22.5 * GB), size_t(12.61 * GB), size_t(1.73 * GB), per);  // 24 GB card: whole layer
    const int gmin = K::choose_tqt_group(128, size_t(13.0 * GB), size_t(12.61 * GB), size_t(1.73 * GB), per);  // nothing fits: 8
    const bool fits16 = 12.61 * GB + 1.73 * GB + double(g16) * per <= 15.2 * GB;
    report("TQT group choice (16 GB: smaller group, 24 GB: whole layer, too small: 8)", 0.0, 1.0,
           g16 < 128 && g16 >= 8 && fits16 && g24 == 128 && gmin == 8);
    std::printf("  groups: 16 GB -> %d, 24 GB -> %d, too small -> %d\n", g16, g24, gmin);
}

// Exact greedy LM head (K::head_greedy: INT8 row copy + bounds, BF16 rescoring of the rows that can still win) and the
// device argmax against the full head (BF16 gemv, bf16, softcap, host std::max_element): equal tokens over random
// inputs, rows that duplicate the best row up to one BF16 ulp, exact duplicates (ties: lowest index) and large logits.
void test_head_greedy() {
    const int V = 6000, D = 2304, trials = 64;
    const float cap = 30.0f;
    auto W = randv(size_t(V) * D, 0.05f);
    std::vector<bf16> wb = to_bf(W);
    bf16 * dw = up(wb);
    int8_t * dq = nullptr; float * dsc = nullptr, * dbound = nullptr, * dapprox = nullptr, * dlog = nullptr;
    ck(cudaMalloc(&dq, size_t(V) * D), "q"); ck(cudaMalloc(&dsc, V * 4), "sc"); ck(cudaMalloc(&dbound, V * 4), "b");
    ck(cudaMalloc(&dapprox, V * 4), "ap"); ck(cudaMalloc(&dlog, V * 4), "lg");
    bf16 * dl16 = nullptr, * dx = nullptr;
    ck(cudaMalloc(&dl16, V * 2), "l16"); ck(cudaMalloc(&dx, D * 2), "x");
    unsigned long long * cells = nullptr;
    int * dtok = nullptr;
    ck(cudaMalloc(&cells, 16), "cells"); ck(cudaMemset(cells, 0, 16), "cells"); ck(cudaMalloc(&dtok, 8), "tok");
    bool same = true;
    int n_eq = 0;
    for (int tr = 0; tr < trials; ++tr) {
        auto x = randv(D, tr % 8 == 7 ? 40.0f : 1.0f);  // some inputs with logits far past the cap
        const auto xb = to_bf(x);
        ck(cudaMemcpy(dx, xb.data(), D * 2, cudaMemcpyHostToDevice), "x");
        // full head on the device: the reference logits
        K::GemvSeg sg{dw, dx, dl16, V, D, D, V, 0};
        K::gemv_rows(&sg, 1, 1, 0);
        K::to_f32(dl16, dlog, V, 0);
        K::softcap(dlog, V, cap, 0);
        auto lg = down(dlog, V);
        int best = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
        if (tr % 2 == 1) {  // near ties: rows near the best one (one BF16 ulp apart, and an exact copy at a lower index)
            const int c1 = (best + 17) % V, c2 = best > 0 ? best / 2 : 1;
            for (int k = 0; k < D; ++k) {
                wb[size_t(c1) * D + k] = __float2bfloat16(__bfloat162float(wb[size_t(best) * D + k]) * (k == 0 ? 1.0078125f : 1.0f));
                wb[size_t(c2) * D + k] = wb[size_t(best) * D + k];
            }
            ck(cudaMemcpy(dw, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice), "w");
            K::gemv_rows(&sg, 1, 1, 0);
            K::to_f32(dl16, dlog, V, 0);
            K::softcap(dlog, V, cap, 0);
            lg = down(dlog, V);
            best = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
        }
        K::quantize_rows_i8(dw, dq, dsc, V, D, 0);
        K::head_bound_build(dw, dq, dsc, dbound, V, D, 0);
        const K::HeadBound hb{dw, dq, dsc, dbound, V, D, V};
        K::head_greedy(hb, dx, 1, cap, dapprox, cells, dtok, dtok + 1, 0);
        K::argmax_greedy(dlog, size_t(V), cells, dtok + 1, nullptr, 0);
        const auto t = down(dtok, 2);
        if (t[0] != best || t[1] != best)
            std::printf("  head greedy trial %d (x sd %s): full head %d (%.6f), bounded %d, device argmax %d\n", tr, tr % 8 == 7 ? "40" : "1", best, lg[best], t[0], t[1]);
        same = same && t[0] == best && t[1] == best;
        n_eq += t[0] == best;
    }
    {   // three rows in one call (a verify block): each row's token == its full-head argmax
        const auto x = randv(size_t(3) * D, 1.0f);
        bf16 * dx3 = up(to_bf(x));
        float * dap3 = nullptr; ck(cudaMalloc(&dap3, size_t(3) * V * 4), "ap3");
        K::quantize_rows_i8(dw, dq, dsc, V, D, 0);
        K::head_bound_build(dw, dq, dsc, dbound, V, D, 0);
        const K::HeadBound hb{dw, dq, dsc, dbound, V, D, V};
        int * dt3 = nullptr; ck(cudaMalloc(&dt3, 12), "t3");
        K::head_greedy(hb, dx3, 3, cap, dap3, cells, dt3, nullptr, 0);
        const auto t3 = down(dt3, 3);
        for (int r = 0; r < 3; ++r) {
            K::GemvSeg sg{dw, dx3 + size_t(r) * D, dl16, V, D, D, V, 0};
            K::gemv_rows(&sg, 1, 1, 0);
            K::to_f32(dl16, dlog, V, 0);
            K::softcap(dlog, V, cap, 0);
            const auto lg = down(dlog, V);
            const int best = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
            if (t3[r] != best) std::printf("  head greedy 3-row call: row %d full head %d, bounded %d\n", r, best, t3[r]);
            same = same && t3[r] == best;
        }
        cudaFree(dx3); cudaFree(dap3); cudaFree(dt3);
    }
    char what[96];
    std::snprintf(what, sizeof what, "exact greedy head (%d / %d tokens equal, + a 3-row call)", n_eq, trials);
    report(what, 0.0, 0.0, same);
    cudaFree(dw); cudaFree(dq); cudaFree(dsc); cudaFree(dbound); cudaFree(dapprox); cudaFree(dlog); cudaFree(dl16); cudaFree(dx); cudaFree(cells); cudaFree(dtok);
}

// one scratch shared by calls of different n (a decode step, then a verify block, as the model makes them): each call ==
// the same call on fresh scratch, bitwise, and the counters at the head of the scratch are zero after it. The fused depth-attention kernel's counters once sat behind the n-sized
// partials, inside an n = 1 call's result area, so an n = 5 call after an n = 1 call started on stale counters.
void test_attn_res_shared_scratch() {
    const int D = 2304, nq = 3, cnt = 5, N = 5;
    K::States st{};
    st.count = cnt;
    std::vector<float *> dS(cnt);
    for (int j = 0; j < cnt; ++j) { dS[j] = up(randv(size_t(N) * D, 1.0f + j)); st.s[j] = dS[j]; }
    float * dq = up(randv(size_t(nq) * D, 0.1f)), * dw = up(randv(size_t(nq) * D, 0.3f));
    const size_t sw = size_t(K::attn_res_norm_scratch_words(K::kSmallN, D));
    float * shared = nullptr, * fresh = nullptr;
    ck(cudaMalloc(&shared, sw * 4), "s"); ck(cudaMemset(shared, 0, sw * 4), "s");
    ck(cudaMalloc(&fresh, sw * 4), "f");
    bf16 * y = nullptr;
    ck(cudaMalloc(&y, size_t(nq) * N * D * 2), "y");
    const float * w[3] = {dw, dw + D, dw + 2 * D};
    bool same = true;
    auto call = [&](int n, float * scr) {
        bf16 * yy[3] = {y, y + size_t(n) * D, y + size_t(2) * n * D};
        K::attn_res_norm(st, dq, nq, nullptr, w, yy, n, D, 1e-6f, scr, 0);
        ck(cudaDeviceSynchronize(), "attn_res");
        return down(y, size_t(nq) * n * D);
    };
    for (int rep = 0; rep < 4; ++rep)
        for (int n : {1, N, 1, 3}) {
            const auto a = call(n, shared);
            // the fused kernel's counters (the first 3 kSmallN words) are zero again after every call, whatever n
            const auto c = down(reinterpret_cast<unsigned *>(shared), size_t(3) * K::kSmallN);
            for (const unsigned v : c) same = same && v == 0u;
            ck(cudaMemset(fresh, 0, sw * 4), "f");
            const auto b = call(n, fresh);
            same = same && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
        }
    report("depth attention: n = 1 / 5 / 3 calls on one scratch", 0.0, 0.0, same);
    for (auto p : dS) cudaFree(p);
    cudaFree(dq); cudaFree(dw); cudaFree(shared); cudaFree(fresh); cudaFree(y);
}

// DFlash2 drafter kernels vs CPU references: the two-tap dynamic convolution and double-rounded SiLU-mul bitwise, top-16
// (ties broken by the lower id, across chunk borders) exactly, and the selector walk (path and the stop at the vocabulary)
void test_dflash2() {
    const int m = 8, H = 2304, group = 16, G = H / group, I = 6912;
    bool same = true;
    {
        const auto x = randv(size_t(m) * H, 1.0f), res = randv(size_t(m) * H, 2.0f), base = randv(size_t(4) * H, 0.5f);
        const auto dyn = randv(size_t(m) * 4 * G, 0.5f);
        float * dx = up(x), * dr = up(res), * db = up(base), * dy = nullptr;
        bf16 * dd = up(to_bf(dyn));
        ck(cudaMalloc(&dy, size_t(m) * H * 4), "y");
        for (int side = 0; side < 2; ++side)
            for (int withres = 0; withres < 2; ++withres) {
                K::dflash_conv(dx, dd, db, m, H, group, side, withres ? dr : nullptr, dy, 0);
                const auto y = down(dy, size_t(m) * H);
                for (int r = 0; r < m; ++r)
                    for (int c = 0; c < H; ++c) {
                        const int gi = c / group, c0 = 2 * side;
                        const float w0 = dyn[size_t(r) * 4 * G + c0 * G + gi] + base[size_t(c0) * H + c];
                        const float w1 = dyn[size_t(r) * 4 * G + (c0 + 1) * G + gi] + base[size_t(c0 + 1) * H + c];
                        float acc = 0.0f + w0 * x[size_t(r) * H + c];
                        acc = acc + w1 * (r ? x[size_t(r - 1) * H + c] : 0.0f);
                        float o = bf(acc);
                        if (withres) o = bf(res[size_t(r) * H + c] + o);
                        same = same && o == y[size_t(r) * H + c];
                    }
            }
        cudaFree(dx); cudaFree(dr); cudaFree(db); cudaFree(dy); cudaFree(dd);
    }
    report("dflash2 conv (both sides, with / without residual)", 0.0, 0.0, same);
    same = true;
    {
        const auto gu = randv(size_t(m) * 2 * I, 2.0f);
        bf16 * dg = up(to_bf(gu)), * da = nullptr;
        ck(cudaMalloc(&da, size_t(m) * I * 2), "a");
        K::dflash_silu_mul(dg, da, m, I, 0);
        const auto a = from_bf(down(da, size_t(m) * I));
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < I; ++c) {
                const float g = gu[size_t(r) * 2 * I + c], u = gu[size_t(r) * 2 * I + I + c];
                const float want = bf(bf(g * (1.0f / (1.0f + std::exp(-g)))) * u);
                same = same && std::fabs(want - a[size_t(r) * I + c]) <= std::fabs(want) * 0.0079f;  // expf vs exp: one BF16 ulp
            }
        cudaFree(dg); cudaFree(da);
    }
    report("dflash2 silu-mul (double rounded)", 0.0, 0.0, same);
    same = true;
    {
        const int n = 200058, ld = 201024;
        std::vector<float> x(size_t(m) * ld, 1e30f);  // the padding columns must not count
        std::uniform_int_distribution<int> q(-400, 400);
        for (int r = 0; r < m; ++r) for (int c = 0; c < n; ++c) x[size_t(r) * ld + c] = q(rng) * 0.25f;  // many ties
        for (int c : {4095, 4096, 8191, 199999}) x[size_t(3) * ld + c] = 1000.0f;  // equal maxima on chunk borders
        float * dx = up(x), * dv = nullptr, * ds = nullptr;
        int * di = nullptr;
        ck(cudaMalloc(&dv, size_t(m) * 16 * 4), "v"); ck(cudaMalloc(&di, size_t(m) * 16 * 4), "i");
        ck(cudaMalloc(&ds, size_t(K::dflash_top16_floats(m, n)) * 4), "s");
        K::dflash_top16(dx, m, n, ld, di, dv, ds, 0);
        const auto ids = down(di, size_t(m) * 16);
        const auto vals = down(dv, size_t(m) * 16);
        for (int r = 0; r < m; ++r) {
            std::vector<int> o(static_cast<size_t>(n));
            for (int c = 0; c < n; ++c) o[size_t(c)] = c;
            const float * xr = x.data() + size_t(r) * ld;
            std::partial_sort(o.begin(), o.begin() + 16, o.end(), [&](int a, int b) { return xr[a] > xr[b] || (xr[a] == xr[b] && a < b); });
            for (int j = 0; j < 16; ++j) same = same && ids[size_t(r) * 16 + j] == o[size_t(j)] && vals[size_t(r) * 16 + j] == xr[o[size_t(j)]];
        }
        // the selector walk over these candidates
        const int V = n, Rk = 256, count = 7;
        const auto gate = randv(size_t(m) * Rk, 1.0f), pred = randv(size_t(V) * Rk, 0.3f), succ = randv(size_t(V) * Rk, 0.3f);
        bf16 * dgt = up(to_bf(gate)), * dp = up(to_bf(pred)), * dsu = up(to_bf(succ));
        int * dt = nullptr, * da = nullptr;
        const int anchor = 46389;
        da = up(std::vector<int>{anchor});
        ck(cudaMalloc(&dt, count * 4), "t");
        auto walk = [&](const std::vector<int> & cid, int vocab) {
            std::vector<int> out;
            int prev = anchor;
            for (int e = 1; e <= count; ++e) {
                double best = -1e300; int w = 0;
                for (int c = 0; c < 16; ++c) {
                    double sum = 0;
                    for (int k = 0; k < Rk; ++k) sum += double(succ[size_t(cid[size_t(e * 16 + c)]) * Rk + k]) * (pred[size_t(prev) * Rk + k] * gate[size_t(e) * Rk + k]);
                    const double sc = sum + vals[size_t(e * 16 + c)];
                    if (sc > best) { best = sc; w = c; }
                }
                prev = cid[size_t(e * 16 + w)];
                if (prev >= vocab) { while (int(out.size()) < count) out.push_back(vocab); break; }
                out.push_back(prev);
            }
            return out;
        };
        K::dflash_select(di, dv, dgt, dsu, dp, da, count, V, dt, 0);
        const auto path = down(dt, count);
        const auto want = walk(ids, V);
        bool p1 = std::equal(want.begin(), want.end(), path.begin());
        // a vocabulary below every candidate of row 1: the walk stops at once
        K::dflash_select(di, dv, dgt, dsu, dp, da, count, 0, dt, 0);
        const auto stop = down(dt, count);
        for (int v : stop) p1 = p1 && v == 0;
        if (!p1) { std::printf("  select path:"); for (int v : path) std::printf(" %d", v); std::printf(" want"); for (int v : want) std::printf(" %d", v); std::printf("\n"); }
        same = same && p1;
        cudaFree(dx); cudaFree(dv); cudaFree(ds); cudaFree(di); cudaFree(dgt); cudaFree(dp); cudaFree(dsu); cudaFree(dt); cudaFree(da);
    }
    report("dflash2 top-16 (ties, chunk borders) + selector walk", 0.0, 0.0, same);
}

// tensor-core experts (K::expert_mma) vs the CPU reference; every token of an 8-token block == the same token alone bitwise,
// and the same token in each of the 8 MMA columns (slot invariance) == alone bitwise
void test_experts_mma(int fmt) {
    const int E = 16, topk = 8, D = 2304, I = 1152, N = 8;
    const float gc = 4.0f, uc = 25.0f;
    ExpertBank G, U, Dn;
    if (fmt >= 3) { make_bank_tqt(G, E, I, D, fmt); make_bank_tqt(U, E, I, D, fmt); make_bank_tqt(Dn, E, D, I, fmt); }
    else { make_bank(G, fmt, E, I, D); make_bank(U, fmt, E, I, D); make_bank(Dn, fmt, E, D, I); }
    const char * nm = fmt == 2 ? "NVF4" : fmt == 1 ? "MXFP8" : fmt == 3 ? "TQT" : fmt == 4 ? "TQT2" : "TQT3";
    const auto X = randv(size_t(N) * D, 1.0f);
    std::vector<int> ids(size_t(N) * topk);
    std::vector<float> wts(ids.size());
    for (int t = 0; t < N; ++t) {
        std::vector<int> p(E);
        for (int e = 0; e < E; ++e) p[e] = e;
        std::shuffle(p.begin(), p.end(), rng);
        for (int k = 0; k < topk; ++k) { ids[t * topk + k] = p[k]; wts[t * topk + k] = 0.5f + float(rng() % 100) / 100.0f; }
    }
    bf16 * dx = up(to_bf(X)), * act = nullptr, * yd = nullptr, * out = nullptr;
    int * dids = up(ids), * groups = nullptr;
    float * dw = up(wts);
    ck(cudaMalloc(&act, size_t(N) * topk * I * 2), "act"); ck(cudaMalloc(&yd, size_t(N) * topk * D * 2), "yd");
    ck(cudaMalloc(&out, size_t(N) * D * 2), "out"); ck(cudaMalloc(&groups, size_t(1 + 8 * 8 * 9) * 4), "groups");
    __half * xh = nullptr, * acth = nullptr;
    ck(cudaMalloc(&xh, size_t(N) * D * 2), "xh"); ck(cudaMalloc(&acth, size_t(N) * topk * I * 2), "acth");
    auto run = [&](const bf16 * x, const int * id, const float * w, int n) {
        K::expert_mma(G.m, U.m, Dn.m, I, D, x, id, w, groups, act, yd, out, n, topk, gc, uc, 0, xh, acth);
        ck(cudaDeviceSynchronize(), "expert_mma");
        return std::make_pair(from_bf(down(act, size_t(n) * topk * I)), from_bf(down(out, size_t(n) * D)));
    };
    const auto [a8, o8] = run(dx, dids, dw, N);
    double ea = 0, eo = 0;
    for (int t = 0; t < N; ++t) {
        std::vector<double> y(D, 0.0);
        for (int k = 0; k < topk; ++k) {
            const int a = t * topk + k, e = ids[a];
            for (int i = 0; i < I; ++i) {
                double g = 0, u = 0;
                for (int c = 0; c < D; ++c) {
                    g += double(G.w[(size_t(e) * I + i) * D + c]) * X[size_t(t) * D + c];
                    u += double(U.w[(size_t(e) * I + i) * D + c]) * X[size_t(t) * D + c];
                }
                g = bf(float(g)); u = bf(float(u));
                const double ref = gc * std::tanh(g / gc) / (1.0 + std::exp(-g)) * uc * std::tanh(u / uc);
                ea = std::max(ea, std::fabs(a8[size_t(a) * I + i] - ref) / (std::fabs(ref) + 0.05));
            }
            for (int r = 0; r < D; ++r) {
                double d = 0;
                for (int i = 0; i < I; ++i) d += double(Dn.w[(size_t(e) * D + r) * I + i]) * a8[size_t(a) * I + i];
                y[r] += bf(float(d)) * wts[a];
            }
        }
        double ms = 0;  // error relative to |y|, floored at a tenth of the token's RMS (outputs near zero carry the absolute error)
        for (int r = 0; r < D; ++r) ms += y[r] * y[r];
        const double floor_ = 0.1 * std::sqrt(ms / D);
        for (int r = 0; r < D; ++r) eo = std::max(eo, std::fabs(o8[size_t(t) * D + r] - y[r]) / (std::fabs(y[r]) + floor_));
    }
    bool bit = true;
    for (int t = 0; t < N; ++t) {
        const auto [a1, o1] = run(dx + size_t(t) * D, dids + t * topk, dw + t * topk, 1);
        bit = bit && std::memcmp(a1.data(), a8.data() + size_t(t) * topk * I, a1.size() * 4) == 0;
        bit = bit && std::memcmp(o1.data(), o8.data() + size_t(t) * D, o1.size() * 4) == 0;
    }
    char name[96];
    std::snprintf(name, sizeof name, "mma experts %s gate/up, 8 tokens == 1", nm);
    report(name, ea, fmt >= 3 ? 4e-2 : 2e-2, bit);  // TQT: FP16 activations after the rotation
    // tensor-core F32 accumulation (truncating alignment); MXFP8's wider E4M3 values spread the error further
    std::snprintf(name, sizeof name, "mma experts %s down + combine", nm);
    report(name, eo, fmt >= 3 ? 6e-2 : 3e-2, bit);  // TQT: the activations enter as FP16 after the rotation
    {   // token 0 in all 8 columns: every column == token 0 alone
        const auto [a1, o1] = run(dx, dids, dw, 1);
        std::vector<bf16> x8(size_t(N) * D);
        const auto x0 = to_bf(std::vector<float>(X.begin(), X.begin() + D));
        for (int t = 0; t < N; ++t) std::copy(x0.begin(), x0.end(), x8.begin() + size_t(t) * D);
        std::vector<int> id8(size_t(N) * topk);
        std::vector<float> w8(id8.size());
        for (int t = 0; t < N; ++t) for (int k = 0; k < topk; ++k) { id8[t * topk + k] = ids[k]; w8[t * topk + k] = wts[k]; }
        bf16 * dx8 = up(x8); int * di8 = up(id8); float * dw8 = up(w8);
        const auto [as, os] = run(dx8, di8, dw8, N);
        bool slot = true;
        for (int t = 0; t < N; ++t) {
            slot = slot && std::memcmp(a1.data(), as.data() + size_t(t) * topk * I, a1.size() * 4) == 0;
            slot = slot && std::memcmp(o1.data(), os.data() + size_t(t) * D, o1.size() * 4) == 0;
        }
        std::snprintf(name, sizeof name, "mma experts %s: one token in each of 8 columns", nm);
        report(name, 0.0, 0.0, slot);
        cudaFree(dx8); cudaFree(di8); cudaFree(dw8);
    }
    cudaFree(dx); cudaFree(act); cudaFree(yd); cudaFree(out); cudaFree(dids); cudaFree(groups); cudaFree(dw); cudaFree(xh); cudaFree(acth);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
    test_gemv();
    test_gemv_i8();
    test_experts(2);
    test_experts(1);
    test_experts_tqt(3);
    test_experts_tqt(4);
    test_experts_tqt(5);
    test_route();
    test_attn_res_capture(1);
    test_attn_res_capture(5);
    test_attn_res_shared_scratch();
    test_attn_res();
    test_msg_gates();
    test_grouped(2);
    test_grouped(1);
    test_grouped_bf16_threshold();
    test_grouped_bf16(2);
    test_grouped_bf16(1);
    test_tqt_dense(3);
    test_tqt_dense(4);
    test_tqt_dense(5);
    test_tqt_groups(3);
    test_tqt_groups(5);
    test_tqt_group_choice();
    test_conv();
    test_head_greedy();
    test_dflash2();
    test_experts(2, 8);  // last: its random draws would move every later test's data
    test_experts(1, 8);
    {   // per-device family choice: plain-only picks the faster 1-token step; with a drafter the better of plain and spec
        const bool a = K::prefer_mma(70, 550, 68, 340, false), b = K::prefer_mma(60, 550, 75, 340, false), c = K::prefer_mma(60, 550, 75, 200, true),
                   d = K::prefer_mma(60, 200, 75, 340, true);
        // a near tie (warp 5 % faster) keeps the tensor cores, so a restart does not change the rounding
        const bool e = K::prefer_mma(140, 1140, 147, 1188, true), f = K::prefer_mma(140, 1140, 147, 1188, false);
        report("kernel family choice (1-token / verify times)", 0.0, 0.0, a && !b && c && !d && e && f);
    }
    test_experts_mma(2);
    test_experts_mma(1);
    test_experts_mma(3);
    test_experts_mma(4);
    test_experts_mma(5);
    std::printf(ok ? "inkling2-kernels: PASS\n" : "inkling2-kernels: FAIL\n");
    return ok ? 0 : 1;
}

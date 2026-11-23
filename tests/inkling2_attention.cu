// Inkling-2-Mini attention (src/models/inkling2/loop/inkling2_attention.cu) against a double-precision CPU reference of the
// transformers PR #51050 formulas: q / relative-state scale + L2 clip (+ log scale), relative bias over the last
// rel_extent distances, sliding window, causal queries inside a block, per-token sink column, sigmoid output gate; and
// inkling2_kv_finish (k L2 clip, v RMS without weight). Cases: a full-attention layer past the log-scale floor (tau > 1),
// a sliding layer deep in the context, a prefill block from position 0, a sink that takes most of the mass.
#include "inkling2_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tqllm_cuda;

namespace {
bool ok = true;
void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); std::exit(e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver ? 77 : 1); }
}
float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }
std::vector<__nv_bfloat16> to_bf(const std::vector<float> & v) {
    std::vector<__nv_bfloat16> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = __float2bfloat16(v[i]);
    return o;
}
template <class T> T * up(const std::vector<T> & v) {
    T * p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
std::vector<float> down(const __nv_bfloat16 * p, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    ck(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost), "download");
    std::vector<float> o(n);
    for (size_t i = 0; i < n; ++i) o[i] = __bfloat162float(h[i]);
    return o;
}

struct Case { const char * name; int pos0, nq, window; bool log_scale; float sink_mean; };

void run(const Case & c, std::mt19937 & rng) {
    const int H = 24, G = 8, HD = 128, DR = 16, E = 1024, ctx = c.pos0 + c.nq;
    std::normal_distribution<float> n01(0.0f, 1.0f);
    auto randv = [&](size_t n, float sd, float mean = 0.0f) { std::vector<float> v(n); for (auto & x : v) x = bf(mean + sd * n01(rng)); return v; };
    const auto q = randv(size_t(c.nq) * H * HD, 3.0f), rs = randv(size_t(c.nq) * H * DR, 3.0f), sink = randv(size_t(c.nq) * H, 2.0f, c.sink_mean),
               gate = randv(size_t(c.nq) * H * HD, 1.5f), qsc = randv(size_t(H) * HD, 0.2f, 1.0f), rsc = randv(size_t(H) * DR, 0.2f, 1.0f),
               proj = randv(size_t(DR) * E, 1.0f), kraw = randv(size_t(ctx) * G * HD, 3.0f), vraw = randv(size_t(ctx) * G * HD, 2.0f);
    {   // the inputs' fingerprint: hosts must draw the same inputs (compiled with -ffp-contract=off) for cross-GPU comparisons
        uint64_t hsh = 1469598103934665603ull;
        for (const auto * v : {&q, &rs, &sink, &gate, &qsc, &rsc, &proj, &kraw, &vraw})
            for (float x : *v) { uint32_t u; std::memcpy(&u, &x, 4); hsh = (hsh ^ u) * 1099511628211ull; }
        std::printf("  inputs %016llx\n", (unsigned long long) hsh);
    }
    __nv_bfloat16 * dk = up(to_bf(kraw)), * dv = up(to_bf(vraw));
    const float clip = 22.627417f, rclip = 8.0f, eps = 1e-6f, alpha = 0.1f, floor_ = 8192.0f;
    inkling2_kv_finish(dk, dv, dk, dv, ctx, G, clip, eps, nullptr, 0, 0);
    ck(cudaDeviceSynchronize(), "kv_finish");
    const auto K = down(dk, kraw.size()), V = down(dv, vraw.size());
    double kv_err = 0;
    for (int r = 0; r < ctx * G; ++r) {  // reference kv_finish
        double sk = 0, sv = 0;
        for (int d = 0; d < HD; ++d) { sk += double(kraw[r * HD + d]) * kraw[r * HD + d]; sv += double(vraw[r * HD + d]) * vraw[r * HD + d]; }
        const double fk = std::min(1.0, clip / (std::sqrt(sk) + 1e-6)), fv = 1.0 / std::sqrt(sv / HD + eps);
        for (int d = 0; d < HD; ++d) {
            kv_err = std::max(kv_err, std::fabs(K[r * HD + d] - bf(float(kraw[r * HD + d] * fk))) / (std::fabs(K[r * HD + d]) + 1e-3));
            kv_err = std::max(kv_err, std::fabs(V[r * HD + d] - bf(float(vraw[r * HD + d] * fv))) / (std::fabs(V[r * HD + d]) + 1e-3));
        }
    }
    Inkling2AttnArgs a;
    a.q = up(to_bf(q)); a.rs = up(to_bf(rs)); a.sink = up(to_bf(sink)); a.gate = up(to_bf(gate));
    a.q_scale = up(to_bf(qsc)); a.rel_scale = up(to_bf(rsc)); a.rel_proj = up(to_bf(proj)); a.kc = dk; a.vc = dv;
    std::vector<__nv_bfloat16> zero(size_t(c.nq) * H * HD);
    __nv_bfloat16 * dout = up(zero);
    a.out = dout;
    ck(cudaMalloc(&a.part, size_t(inkling2_attention_partial_floats(H, c.nq, ctx)) * 4), "part");
    a.nq = c.nq; a.pos0 = c.pos0; a.heads = H; a.kvheads = G; a.window = c.window; a.rel_extent = E; a.log_scale = c.log_scale;
    std::vector<double> refv(zero.size());
    double sink_mass = 0;
    for (int i = 0; i < c.nq; ++i) {
        const int pos = c.pos0 + i, lo = c.window > 0 ? std::max(0, pos - c.window + 1) : 0;
        const double tau = 1.0 + alpha * std::log(std::max((pos + 1) / double(floor_), 1.0));
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / G);
            std::vector<double> qv(HD), rv(DR);
            double qn = 0, rn = 0;
            for (int d = 0; d < HD; ++d) { qv[d] = double(q[(i * H + h) * HD + d]) * qsc[h * HD + d]; qn += qv[d] * qv[d]; }
            for (int d = 0; d < HD; ++d) { qv[d] = bf(float(qv[d] * std::min(1.0, clip / (std::sqrt(qn) + 1e-6)))); if (c.log_scale) qv[d] = bf(float(qv[d] * tau)); }
            for (int d = 0; d < DR; ++d) { rv[d] = double(rs[(i * H + h) * DR + d]) * rsc[h * DR + d]; rn += rv[d] * rv[d]; }
            for (int d = 0; d < DR; ++d) { rv[d] = bf(float(rv[d] * std::min(1.0, rclip / (std::sqrt(rn) + 1e-6)))); if (c.log_scale) rv[d] = bf(float(rv[d] * tau)); }
            std::vector<double> sc;
            double M = sink[i * H + h];
            for (int t = lo; t <= pos; ++t) {
                double s = 0;
                for (int d = 0; d < HD; ++d) s += qv[d] * K[(size_t(t) * G + g) * HD + d];
                s /= std::sqrt(double(HD));
                if (pos - t < E) for (int r = 0; r < DR; ++r) s += rv[r] * proj[size_t(r) * E + (pos - t)];
                sc.push_back(s);
                M = std::max(M, s);
            }
            double L = std::exp(sink[i * H + h] - M);
            std::vector<double> acc(HD, 0.0);
            for (int t = lo; t <= pos; ++t) {
                const double p = std::exp(sc[t - lo] - M);
                L += p;
                for (int d = 0; d < HD; ++d) acc[d] += p * V[(size_t(t) * G + g) * HD + d];
            }
            sink_mass += std::exp(sink[i * H + h] - M) / L;
            for (int d = 0; d < HD; ++d) refv[(i * H + h) * HD + d] = acc[d] / L / (1.0 + std::exp(-gate[(i * H + h) * HD + d]));
        }
    }
    auto err = [&](const std::vector<float> & out) {
        double w = 0;
        for (size_t j = 0; j < out.size(); ++j) w = std::max(w, std::fabs(out[j] - refv[j]) / (std::fabs(refv[j]) + 0.02));
        return w;
    };
    auto same = [](const std::vector<float> & x, const std::vector<float> & y) { return std::memcmp(x.data(), y.data(), x.size() * 4) == 0; };
    inkling2_attention(a, 0);
    ck(cudaDeviceSynchronize(), "attention");
    const auto out = down(dout, zero.size());
    double worst = err(out), worst_dec = 0;
    bool bitwise = true;
    // ring cache (windowed layers): rows key % ring for the last ring keys; outputs must equal the absolute layout bitwise
    __nv_bfloat16 * rk = nullptr, * rv_ = nullptr;
    const int ring = c.window > 0 ? c.window + std::max(c.nq, 8) : 0;
    if (ring) {
        ck(cudaMalloc(&rk, size_t(ring) * G * HD * 2), "ring"); ck(cudaMalloc(&rv_, size_t(ring) * G * HD * 2), "ring");
        for (int key = std::max(0, ctx - ring); key < ctx; ++key) {
            ck(cudaMemcpy(rk + size_t(key % ring) * G * HD, dk + size_t(key) * G * HD, G * HD * 2, cudaMemcpyDeviceToDevice), "ring k");
            ck(cudaMemcpy(rv_ + size_t(key % ring) * G * HD, dv + size_t(key) * G * HD, G * HD * 2, cudaMemcpyDeviceToDevice), "ring v");
        }
        Inkling2AttnArgs b = a; b.kc = rk; b.vc = rv_; b.ring = ring;
        inkling2_attention(b, 0);
        ck(cudaDeviceSynchronize(), "attention ring");
        bitwise = bitwise && same(out, down(dout, zero.size()));
    }
    if (c.nq <= 8) {  // decode kernel: accuracy, ring layout, and every query alone == the batch (verify == plain steps)
        int * dpos = nullptr;
        ck(cudaMalloc(&dpos, 4), "dpos");
        float * dpart = nullptr;
        const int splits = c.window > 0 ? 16 : 64;
        ck(cudaMalloc(&dpart, size_t(inkling2_attention_decode_floats(H, c.nq, splits)) * 4), "dpart");
        Inkling2AttnArgs d = a; d.dpos = dpos; d.splits = splits; d.part = dpart;
        ck(cudaMemcpy(dpos, &c.pos0, 4, cudaMemcpyHostToDevice), "dpos");
        inkling2_attention_decode(d, 0);
        ck(cudaDeviceSynchronize(), "decode");
        const auto dec = down(dout, zero.size());
        worst_dec = err(dec);
        if (ring) {
            Inkling2AttnArgs b = d; b.kc = rk; b.vc = rv_; b.ring = ring;
            inkling2_attention_decode(b, 0);
            ck(cudaDeviceSynchronize(), "decode ring");
            bitwise = bitwise && same(dec, down(dout, zero.size()));
        }
        for (int i = 0; i < c.nq; ++i) {
            Inkling2AttnArgs o = d; o.nq = 1;
            o.q += size_t(i) * H * HD; o.rs += size_t(i) * H * DR; o.sink += size_t(i) * H; o.gate += size_t(i) * H * HD;
            const int p = c.pos0 + i;
            ck(cudaMemcpy(dpos, &p, 4, cudaMemcpyHostToDevice), "dpos");
            inkling2_attention_decode(o, 0);
            ck(cudaDeviceSynchronize(), "decode one");
            const auto one = down(dout, size_t(H) * HD);
            bitwise = bitwise && std::memcmp(one.data(), dec.data() + size_t(i) * H * HD, one.size() * 4) == 0;
        }
        cudaFree(dpos); cudaFree(dpart);
    }
    double worst_pr = 0, worst_own = 0;
    int prep_ulp = 0;
    {   // prompt kernel (tensor cores, BF16 P): accuracy, and the ring layout bitwise
        int * dpos = nullptr;
        ck(cudaMalloc(&dpos, 4), "dpos");
        ck(cudaMemcpy(dpos, &c.pos0, 4, cudaMemcpyHostToDevice), "dpos");
        __nv_bfloat16 * qb = nullptr; float * rb = nullptr;
        ck(cudaMalloc(&qb, size_t(c.nq) * H * HD * 2), "qb"); ck(cudaMalloc(&rb, size_t(c.nq) * H * DR * 4), "rb");
        Inkling2AttnArgs p = a; p.dpos = dpos;
        inkling2_attention_prompt(p, qb, rb, 0);
        ck(cudaDeviceSynchronize(), "prompt");
        const auto pr = down(dout, zero.size());
        worst_pr = err(pr);
        {   // the same attention in double on the kernel's own processed q / rs (BF16-rounded on the GPU): separates the
            // kernel's arithmetic from inputs that a different rounding of q (F32 vs double) moves across a near tie
            std::vector<__nv_bfloat16> qh(size_t(c.nq) * H * HD);
            std::vector<float> rh(size_t(c.nq) * H * DR);
            ck(cudaMemcpy(qh.data(), qb, qh.size() * 2, cudaMemcpyDeviceToHost), "qb");
            ck(cudaMemcpy(rh.data(), rb, rh.size() * 4, cudaMemcpyDeviceToHost), "rb");
            double w2 = 0;
            long qdiff = 0, rdiff = 0;
            int qulp = 0, rulp = 0;
            auto ulps = [](float a, float b) {  // BF16 values a, b: their distance in BF16 steps
                uint32_t ua, ub; std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4);
                const int ia = (ua & 0x80000000u) ? -int((ua >> 16) & 0x7fff) : int(ua >> 16), ib = (ub & 0x80000000u) ? -int((ub >> 16) & 0x7fff) : int(ub >> 16);
                return std::abs(ia - ib);
            };
            for (int i = 0; i < c.nq; ++i) {
                const int pos = c.pos0 + i, lo = c.window > 0 ? std::max(0, pos - c.window + 1) : 0;
                const double tau = 1.0 + alpha * std::log(std::max((pos + 1) / double(floor_), 1.0));
                for (int h = 0; h < H; ++h) {
                    const int g = h / (H / G);
                    std::vector<double> qv(HD), qc(HD);
                    double qn = 0;
                    for (int d = 0; d < HD; ++d) { qc[d] = double(q[(i * H + h) * HD + d]) * qsc[h * HD + d]; qn += qc[d] * qc[d]; }
                    for (int d = 0; d < HD; ++d) {
                        double x = bf(float(qc[d] * std::min(1.0, clip / (std::sqrt(qn) + 1e-6))));
                        if (c.log_scale) x = bf(float(x * tau));
                        qv[d] = __bfloat162float(qh[(size_t(i) * H + h) * HD + d]);
                        qdiff += qv[d] != x;
                        qulp = std::max(qulp, ulps(float(qv[d]), float(x)));
                    }
                    {
                        std::vector<double> rc(DR);
                        double rn = 0;
                        for (int d = 0; d < DR; ++d) { rc[d] = double(rs[(i * H + h) * DR + d]) * rsc[h * DR + d]; rn += rc[d] * rc[d]; }
                        for (int d = 0; d < DR; ++d) {
                            double x = bf(float(rc[d] * std::min(1.0, rclip / (std::sqrt(rn) + 1e-6))));
                            if (c.log_scale) x = bf(float(x * tau));
                            const float gpu = rh[(size_t(i) * H + h) * DR + d];
                            rdiff += gpu != float(x);
                            rulp = std::max(rulp, ulps(gpu, float(x)));
                        }
                    }
                    std::vector<double> sc;
                    double M = sink[i * H + h];
                    for (int t = lo; t <= pos; ++t) {
                        double sv = 0;
                        for (int d = 0; d < HD; ++d) sv += qv[d] * K[(size_t(t) * G + g) * HD + d];
                        sv /= std::sqrt(double(HD));
                        if (pos - t < E) for (int r = 0; r < DR; ++r) sv += double(rh[(size_t(i) * H + h) * DR + r]) * proj[size_t(r) * E + (pos - t)];
                        sc.push_back(sv);
                        M = std::max(M, sv);
                    }
                    double L = std::exp(sink[i * H + h] - M);
                    std::vector<double> acc(HD, 0.0);
                    for (int t = lo; t <= pos; ++t) {
                        const double pp = std::exp(sc[t - lo] - M);
                        L += pp;
                        for (int d = 0; d < HD; ++d) acc[d] += pp * V[(size_t(t) * G + g) * HD + d];
                    }
                    for (int d = 0; d < HD; ++d) {
                        const double ref = acc[d] / L / (1.0 + std::exp(-gate[(i * H + h) * HD + d]));
                        w2 = std::max(w2, std::fabs(pr[(size_t(i) * H + h) * HD + d] - ref) / (std::fabs(ref) + 0.02));
                    }
                }
            }
            std::printf("  prompt kernel vs a double reference on its own processed q / rs: worst rel err %.2e; vs the double path's rounding "
                        "%ld of %d q values (at most %d BF16 steps) and %ld of %d rs values (at most %d) differ\n", w2, qdiff, c.nq * H * HD, qulp, rdiff,
                        c.nq * H * DR, rulp);
            worst_own = w2;
            prep_ulp = std::max(qulp, rulp);
        }
        if (ring) {
            Inkling2AttnArgs b = p; b.kc = rk; b.vc = rv_; b.ring = ring;
            inkling2_attention_prompt(b, qb, rb, 0);
            ck(cudaDeviceSynchronize(), "prompt ring");
            bitwise = bitwise && same(pr, down(dout, zero.size()));
        }
        cudaFree(dpos); cudaFree(qb); cudaFree(rb);
    }
    if (rk) { cudaFree(rk); cudaFree(rv_); }
    // Held strictly against the reference on the kernels' own processed q / rs (and the prep within 1 BF16 step of the
    // double path). Against the double path's own rounding a near tie can move further: "sliding, prompt chunk ring" on
    // sm_86 is 4.3 / 4.6 % that way but 0.73 % on its own inputs, with 4 of 691,200 q / rs values one BF16 step apart.
    const bool pass = worst < 0.06 && worst_dec < 0.06 && worst_pr < 0.06 && worst_own < 0.02 && prep_ulp <= 1 && kv_err < 0.01 && bitwise;
    std::printf("%-26s pos0 %5d nq %3d window %4d log %d: worst rel err %.2e, decode kernel %.2e, prompt kernel %.2e (kv_finish %.2e), ring / "
                "per-query bitwise %s, mean sink mass %.2f %s\n", c.name, c.pos0, c.nq, c.window, int(c.log_scale), worst, worst_dec, worst_pr, kv_err,
                bitwise ? "equal" : "DIFFER",
                sink_mass / (c.nq * H), pass ? "ok" : "FAIL");
    if (!pass) ok = false;
    cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(a.part);
    cudaFree(const_cast<__nv_bfloat16 *>(a.q)); cudaFree(const_cast<__nv_bfloat16 *>(a.rs)); cudaFree(const_cast<__nv_bfloat16 *>(a.sink));
    cudaFree(const_cast<__nv_bfloat16 *>(a.gate)); cudaFree(const_cast<__nv_bfloat16 *>(a.q_scale));
    cudaFree(const_cast<__nv_bfloat16 *>(a.rel_scale)); cudaFree(const_cast<__nv_bfloat16 *>(a.rel_proj));
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
    std::mt19937 rng(11);
    const Case cases[] = {
        {"full, past the log floor", 9000, 3, 0, true, 0.0f},
        {"full, short context", 40, 2, 0, true, 0.0f},
        {"sliding, deep context", 3000, 4, 1024, false, 0.0f},
        {"sliding prefill from 0", 0, 37, 1024, false, 0.0f},
        {"sliding, window edge", 1023, 3, 1024, false, 0.0f},
        {"sink-heavy", 600, 2, 1024, false, 70.0f},
        {"sink comparable", 600, 2, 1024, false, 32.0f},
        {"full, verify block of 8", 5000, 8, 0, true, 0.0f},
        {"full, fewer keys than splits", 5, 2, 0, true, 0.0f},
        {"sliding, one decode step", 4100, 1, 1024, false, 0.0f},
        {"full, prompt chunk deep", 1500, 300, 0, true, 0.0f},
        {"full, prompt past log floor", 8300, 70, 0, true, 0.0f},
        {"sliding, prompt chunk ring", 3000, 200, 1024, false, 0.0f},
    };
    for (const Case & c : cases) run(c, rng);
    std::printf(ok ? "inkling2_attention: OK\n" : "inkling2_attention: FAILED\n");
    return ok ? 0 : 1;
}

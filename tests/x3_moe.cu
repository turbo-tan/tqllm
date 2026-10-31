// X3 routed-expert matvec (moe_window.cuh) on a bank that mixes K2 and K3 experts (an EXL3 pack does): every
// (token, slot) output must match a host GEMV of that expert's dequant_rows weights, and a token's output must be
// bit-identical whether it runs alone (n = 1) or in a batch (n = 6, the verify chunk), for both input modes.
#include "weight_format.hpp"
#include "x3_record.hpp"
#include "gguf.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tqllm;

namespace {
uint32_t g = 777u;
uint32_t lcg() { g = g * 1664525u + 1013904223u; return g; }
float rnd() { return (float(int32_t(lcg() >> 8)) / float(1 << 23)) * 0.5f; }
bool ok = true;
void check(bool c, const std::string & what) { if (!c) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ok = false; } }
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); std::exit(2); } }
template <class T> T * up(const std::vector<T> & h) { T * d; ck(cudaMalloc(&d, h.size() * sizeof(T))); ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> std::vector<T> down(const T * d, size_t n) { std::vector<T> h(n); ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
}  // namespace

int main() try {
    const int E = 6, rows = 512, cols = 1024, topk = 2;  // rows: one 512-column item of the prompt lists
    const int K[E] = {2, 3, 2, 2, 3, 2};
    std::vector<char> raw;
    for (int e = 0; e < E; ++e) {
        const uint32_t words = 16 * uint32_t(K[e]);
        const X3Record rec = x3_record_make(rows, cols, words, X3_CODEC_MUL1);
        raw.insert(raw.end(), reinterpret_cast<const char *>(&rec), reinterpret_cast<const char *>(&rec) + sizeof rec);
        for (size_t i = 0; i < size_t(rows / 16) * (cols / 16) * words; ++i) { const uint16_t w = uint16_t(lcg() >> 16); raw.insert(raw.end(), reinterpret_cast<const char *>(&w), reinterpret_cast<const char *>(&w) + 2); }
        for (int i = 0; i < cols + rows; ++i) { const __half h = __float2half(0.5f + std::fabs(rnd())); raw.insert(raw.end(), reinterpret_cast<const char *>(&h), reinterpret_cast<const char *>(&h) + 2); }
    }
    cudaStream_t s; ck(cudaStreamCreate(&s));
    const WeightFormat * wf = weight_format(GGML_X3);
    check(wf && wf->has_expert_matvec(), "x3 format with expert matvec");
    const Mat m = wf->load(raw, rows * E, cols, [](const void * h, size_t b) -> const void * { void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice)); return d; }, s);
    // host reference weights: each expert's [rows][cols] from dequant_rows (dequant_h writes all parts row-stacked)
    __half * wd; ck(cudaMalloc(&wd, size_t(E) * rows * cols * 2));
    wf->dequant_h(m, wd, s); ck(cudaStreamSynchronize(s));
    std::vector<__half> W = down(wd, size_t(E) * rows * cols);

    const int N = 6;
    std::vector<int> ids(N * topk);
    for (int t = 0; t < N; ++t) { ids[t * topk] = (t * 5 + 1) % E; ids[t * topk + 1] = (t * 3 + 4) % E; }  // K2 and K3 mixed
    for (int per_slot = 0; per_slot < 2; ++per_slot) {
        const int xr = per_slot ? N * topk : N;
        std::vector<__half> x(size_t(xr) * cols);
        for (auto & v : x) v = __float2half(rnd());
        __half * dx = up(x); int * di = up(ids);
        __nv_bfloat16 * dy; ck(cudaMalloc(&dy, size_t(N) * topk * rows * 2));
        wf->expert_matvec(m, dx, dy, rows, di, N, topk, per_slot != 0, s); ck(cudaStreamSynchronize(s));
        const auto y = down(dy, size_t(N) * topk * rows);
        double err = 0, ref2 = 0;
        for (int i = 0; i < N * topk; ++i) {
            const int e = ids[i];
            const __half * xi = x.data() + size_t(per_slot ? i : i / topk) * cols;
            for (int r = 0; r < rows; ++r) {
                double a = 0;
                for (int c = 0; c < cols; ++c) a += double(__half2float(W[(size_t(e) * rows + r) * cols + c])) * __half2float(xi[c]);
                const double d = double(__bfloat162float(y[size_t(i) * rows + r])) - a;
                err += d * d; ref2 += a * a;
            }
        }
        const double rel = std::sqrt(err / ref2);
        std::printf("x3_moe per_slot=%d rel_rms %.2e\n", per_slot, rel);
        check(rel < 1e-2, "expert matvec vs host GEMV of dequant_rows");
        // token 3 alone == token 3 inside the batch, bit for bit
        __nv_bfloat16 * dy1; ck(cudaMalloc(&dy1, size_t(topk) * rows * 2));
        const __half * x3 = dx + size_t(per_slot ? 3 * topk : 3) * cols;
        wf->expert_matvec(m, x3, dy1, rows, di + 3 * topk, 1, topk, per_slot != 0, s); ck(cudaStreamSynchronize(s));
        const auto y1 = down(dy1, size_t(topk) * rows);
        check(std::memcmp(y1.data(), y.data() + size_t(3) * topk * rows, y1.size() * 2) == 0, "n = 1 bit-identical to the batch");
        // prompt path: grouped by expert (moe_group's layout: groups [experts][N] of assignments) -> same outputs
        std::vector<int> counts(E, 0), groups(size_t(E) * N, -1);
        for (int a = 0; a < N * topk; ++a) groups[size_t(ids[a]) * N + counts[size_t(ids[a])]++] = a;
        int * dc = up(counts); int * dg = up(groups);
        __nv_bfloat16 * dyg; ck(cudaMalloc(&dyg, size_t(N) * topk * rows * 2));
        check(wf->expert_gemm(m, dx, dyg, rows, dc, dg, N, E, topk, per_slot != 0, s), "x3 expert_gemm available");
        ck(cudaStreamSynchronize(s));
        const auto yg = down(dyg, size_t(N) * topk * rows);
        double ge = 0, g2 = 0;
        for (size_t i = 0; i < yg.size(); ++i) { const double a = __bfloat162float(y[i]), b = __bfloat162float(yg[i]); ge += (a - b) * (a - b); g2 += a * a; }
        std::printf("x3_moe per_slot=%d expert_gemm vs matvec rel_rms %.2e\n", per_slot, std::sqrt(ge / g2));
        check(std::sqrt(ge / g2) < 1e-2, "expert_gemm vs expert_matvec");
        // layer-major prompt (MiniMax forward_lm): one expert_gemm over all tokens == one per chunk of them
        double ce = 0;
        for (int t0 = 0; t0 < N; t0 += 4) {
            const int n = std::min(4, N - t0);
            std::vector<int> cc(E, 0), cg(size_t(E) * n, -1);
            for (int a = 0; a < n * topk; ++a) { const int e = ids[size_t(t0) * topk + a]; cg[size_t(e) * n + cc[size_t(e)]++] = a; }
            int * dcc = up(cc); int * dcg = up(cg);
            __nv_bfloat16 * dyc; ck(cudaMalloc(&dyc, size_t(n) * topk * rows * 2));
            const __half * xc = dx + size_t(per_slot ? t0 * topk : t0) * cols;
            check(wf->expert_gemm(m, xc, dyc, rows, dcc, dcg, n, E, topk, per_slot != 0, s), "x3 expert_gemm on a chunk");
            ck(cudaStreamSynchronize(s));
            const auto yc = down(dyc, size_t(n) * topk * rows);
            for (size_t i = 0; i < yc.size(); ++i) {
                const double a = __bfloat162float(yg[size_t(t0) * topk * rows + i]), b = __bfloat162float(yc[i]);
                ce += (a - b) * (a - b);
            }
            cudaFree(dcc); cudaFree(dcg); cudaFree(dyc);
        }
        std::printf("x3_moe per_slot=%d expert_gemm all tokens vs per chunk rel_rms %.2e\n", per_slot, std::sqrt(ce / g2));
        check(std::sqrt(ce / g2) < 1e-3, "expert_gemm over all tokens == per chunk");
        cudaFree(dc); cudaFree(dg); cudaFree(dyg);
        cudaFree(dx); cudaFree(di); cudaFree(dy); cudaFree(dy1);
    }
    {  // a prompt chunk wide enough for several passes per expert and several lists per codec (> XW_MOE_MAX_PARTS
       // parts): expert_gemm against expert_matvec of each token alone
        const int NT = 1200;
        std::vector<int> bid(size_t(NT) * topk);
        for (int t = 0; t < NT; ++t) { bid[size_t(t) * topk] = int(lcg() % E); bid[size_t(t) * topk + 1] = (bid[size_t(t) * topk] + 1 + int(lcg() % (E - 1))) % E; }
        for (int per_slot = 0; per_slot < 2; ++per_slot) {
            const int xr = per_slot ? NT * topk : NT;
            std::vector<__half> x(size_t(xr) * cols);
            for (auto & v : x) v = __float2half(rnd());
            __half * dx = up(x); int * di = up(bid);
            __nv_bfloat16 * dy; ck(cudaMalloc(&dy, size_t(NT) * topk * rows * 2));
            for (int t = 0; t < NT; ++t)
                wf->expert_matvec(m, dx + size_t(per_slot ? t * topk : t) * cols, dy + size_t(t) * topk * rows, rows,
                                  di + size_t(t) * topk, 1, topk, per_slot != 0, s);
            std::vector<int> counts(E, 0), groups(size_t(E) * NT * topk, -1);
            for (int a = 0; a < NT * topk; ++a) groups[size_t(bid[size_t(a)]) * NT * topk + counts[size_t(bid[size_t(a)])]++] = a;
            int * dc = up(counts); int * dg = up(groups);
            __nv_bfloat16 * dyg; ck(cudaMalloc(&dyg, size_t(NT) * topk * rows * 2));
            check(wf->expert_gemm(m, dx, dyg, rows, dc, dg, NT * topk, E, topk, per_slot != 0, s), "x3 expert_gemm, wide chunk");
            ck(cudaStreamSynchronize(s));
            const auto a = down(dy, size_t(NT) * topk * rows), b = down(dyg, size_t(NT) * topk * rows);
            double e2 = 0, r2 = 0;
            for (size_t i = 0; i < a.size(); ++i) { const double u = __bfloat162float(a[i]), v = __bfloat162float(b[i]); e2 += (u - v) * (u - v); r2 += u * u; }
            std::printf("x3_moe per_slot=%d wide chunk (%d tokens, most %d per expert): expert_gemm vs expert_matvec rel_rms %.2e\n", per_slot,
                        NT, *std::max_element(counts.begin(), counts.end()), std::sqrt(e2 / r2));
            // the prompt lists run each column's whole k range in one warp: the same products, another fp32 summation order
            check(std::sqrt(e2 / r2) < 2e-3, "wide chunk: expert_gemm vs per-token expert_matvec");
            cudaFree(dx); cudaFree(di); cudaFree(dy); cudaFree(dc); cudaFree(dg); cudaFree(dyg);
        }
    }
    std::printf(ok ? "x3_moe: OK\n" : "x3_moe: FAILED\n");
    return ok ? 0 : 1;
} catch (const std::exception & e) { std::fprintf(stderr, "x3_moe: %s\n", e.what()); return 1; }

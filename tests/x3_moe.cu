// X3 routed-expert matvec (moe_window.cuh) on a bank that mixes K2 and K3 experts (an EXL3 pack does): every
// (token, slot) output must match a host GEMV of that expert's dequant_rows weights, and a token's output must be
// bit-identical whether it runs alone (n = 1) or in a batch (n = 6, the verify chunk), for both input modes.
// Two column widths: 1024 (64 k slices, one k part per part) and 3072 (192 slices, so the routed path splits k into
// parts and the partial sums of every column go through P in another order).
#include "weight_format.hpp"
#include "x3_record.hpp"
#include "window.hpp"
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
    const int E = 6, topk = 2;
    const int K[E] = {2, 3, 2, 2, 3, 2};
    const WeightFormat * wf = weight_format(GGML_X3);
    check(wf && wf->has_expert_matvec(), "x3 format with expert matvec");
    cudaStream_t s; ck(cudaStreamCreate(&s));

    auto upload = [](const void * h, size_t b) -> const void * { void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, h, b, cudaMemcpyHostToDevice)); return d; };

    auto run_bank = [&](int rows, int cols) {
        auto build_raw = [&]() {
            std::vector<char> r;
            for (int e = 0; e < E; ++e) {
                const uint32_t words = 16 * uint32_t(K[e]);
                const X3Record rec = x3_record_make(rows, cols, words, X3_CODEC_MUL1);
                r.insert(r.end(), reinterpret_cast<const char *>(&rec), reinterpret_cast<const char *>(&rec) + sizeof rec);
                for (size_t i = 0; i < size_t(rows / 16) * (cols / 16) * words; ++i) { const uint16_t w = uint16_t(lcg() >> 16); r.insert(r.end(), reinterpret_cast<const char *>(&w), reinterpret_cast<const char *>(&w) + 2); }
                for (int i = 0; i < cols + rows; ++i) { const __half h = __float2half(0.5f + std::fabs(rnd())); r.insert(r.end(), reinterpret_cast<const char *>(&h), reinterpret_cast<const char *>(&h) + 2); }
            }
            return r;
        };
        const std::vector<char> raw = build_raw();
        const Mat m = wf->load(raw, rows * E, cols, upload, s);
        // host reference weights: each expert's [rows][cols] from dequant_rows (dequant_h writes all parts row-stacked)
        __half * wd; ck(cudaMalloc(&wd, size_t(E) * rows * cols * 2));
        wf->dequant_h(m, wd, s); ck(cudaStreamSynchronize(s));
        std::vector<__half> W = down(wd, size_t(E) * rows * cols);
        std::printf("x3_moe bank %dx%d: routed kparts %d (%d slices/item), prompt kparts %d\n", rows, cols,
                    x3::xw_moe_kparts(cols), cols / 16 / x3::xw_moe_kparts(cols), x3::xw_pmoe_kparts(cols));
        check(cols / 16 / x3::xw_moe_kparts(cols) <= x3::XW_MOE_MAX_ITEM_SLICES, "routed item within the slice bound");

        const int N = 6;
        std::vector<int> ids(N * topk);
        for (int t = 0; t < N; ++t) { ids[t * topk] = (t * 5 + 1) % E; ids[t * topk + 1] = (t * 3 + 4) % E; }  // K2 and K3 mixed
        for (int per_slot = 0; per_slot < 2; ++per_slot) {
            const int xr = per_slot ? N * topk : N;
            std::vector<__half> x(size_t(xr) * cols);
            for (auto & v : x) v = __float2half(rnd());
            __half * dx = up(x); int * di = up(ids);
            __nv_bfloat16 * dy; ck(cudaMalloc(&dy, size_t(N) * topk * rows * 2));
            // poison: the verify path writes y [assignment][rows] from the out launch itself, so every row must be set
            ck(cudaMemset(dy, 0xA5, size_t(N) * topk * rows * 2));
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
            // every token alone == that token inside the batch, bit for bit (the verify path's out launch writes the
            // assignment rows itself through the build's inverse slot map, so a wrong map shows up here)
            __nv_bfloat16 * dy1; ck(cudaMalloc(&dy1, size_t(topk) * rows * 2));
            for (int t = 0; t < N; ++t) {
                const __half * xt = dx + size_t(per_slot ? t * topk : t) * cols;
                wf->expert_matvec(m, xt, dy1, rows, di + size_t(t) * topk, 1, topk, per_slot != 0, s);
                ck(cudaStreamSynchronize(s));
                const auto y1 = down(dy1, size_t(topk) * rows);
                check(std::memcmp(y1.data(), y.data() + size_t(t) * topk * rows, y1.size() * 2) == 0, "n = 1 bit-identical to the batch");
            }
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
        {   // a verify block wider than the engine's default 8 (a DSpark block past 7 is verified in one forward, 4c):
            // every column of a 12-column block is still that token's own bits, both input modes.
            const int NW = 12;
            std::vector<int> wid(size_t(NW) * topk);
            for (int t = 0; t < NW; ++t) { wid[size_t(t) * topk] = (t * 5 + 1) % E; wid[size_t(t) * topk + 1] = (t * 3 + 4) % E; }
            for (int per_slot = 0; per_slot < 2; ++per_slot) {
                const int xr = NW * topk;
                std::vector<__half> x(size_t(per_slot ? xr : NW) * cols);
                for (auto & v : x) v = __float2half(rnd());
                __half * dx = up(x); int * di = up(wid);
                __nv_bfloat16 * dy, * dy1;
                ck(cudaMalloc(&dy, size_t(NW) * topk * rows * 2));
                ck(cudaMemset(dy, 0xA5, size_t(NW) * topk * rows * 2));   // every assignment row must be written
                ck(cudaMalloc(&dy1, size_t(topk) * rows * 2));
                wf->expert_matvec(m, dx, dy, rows, di, NW, topk, per_slot != 0, s);
                int same = 0;
                for (int t = 0; t < NW; ++t) {
                    const __half * xt = dx + size_t(per_slot ? t * topk : t) * cols;
                    wf->expert_matvec(m, xt, dy1, rows, di + size_t(t) * topk, 1, topk, per_slot != 0, s);
                    ck(cudaStreamSynchronize(s));
                    const auto y1 = down(dy1, size_t(topk) * rows);
                    const auto yb = down(dy, size_t(NW) * topk * rows);
                    same += std::memcmp(y1.data(), yb.data() + size_t(t) * topk * rows, y1.size() * 2) == 0;
                }
                check(same == NW, "n = 12 verify block: every token alone bit-identical to the block");
                std::printf("x3_moe per_slot=%d wide verify block: %d / %d tokens bit-identical\n", per_slot, same, NW);
                cudaFree(dx); cudaFree(di); cudaFree(dy); cudaFree(dy1);
            }
        }
        {   // gate + up in ONE pass (expert_matvec2) == the two calls, bit for bit, plain decode and a verify block. The
            // second matrix is a different bank, so a wrong bank index or up-half offset cannot pass; the output is one
            // allocation with the up half `n * topk * rows` elements behind the gate half, poisoned first. A pair whose
            // banks cannot share one pass throws (the caller falls back to two calls), so no throw means the one-pass
            // path is the one that ran.
            const Mat m2 = wf->load(build_raw(), rows * E, cols, upload, s);
            for (int n : {1, 4, 12}) {
                std::vector<int> pid(size_t(n) * topk);
                for (size_t a = 0; a < pid.size(); ++a) pid[a] = int((a * 5 + 1) % E);
                std::vector<__half> x(size_t(n) * cols);
                for (auto & v : x) v = __float2half(rnd());
                __half * dx = up(x); int * di = up(pid);
                const size_t el = size_t(n) * topk * rows;
                __nv_bfloat16 * p1, * p2;
                ck(cudaMalloc(&p1, 2 * el * 2)); ck(cudaMemset(p1, 0xA5, 2 * el * 2));
                ck(cudaMalloc(&p2, 2 * el * 2)); ck(cudaMemset(p2, 0xA5, 2 * el * 2));
                bool paired = true;
                try { wf->expert_matvec2(m, m2, dx, p1, p1 + el, rows, di, n, topk, s); }
                catch (const std::exception & e) { paired = false; std::printf("x3_moe paired n=%d refused: %s\n", n, e.what()); }
                check(paired, "expert_matvec2 takes the one-pass path for a pairable bank");
                wf->expert_matvec(m, dx, p2, rows, di, n, topk, false, s);
                wf->expert_matvec(m2, dx, p2 + el, rows, di, n, topk, false, s);
                ck(cudaStreamSynchronize(s));
                const auto one = down(p1, 2 * el), two = down(p2, 2 * el);
                check(std::memcmp(one.data(), two.data(), one.size() * 2) == 0, "expert_matvec2 == two expert_matvec calls");
                std::printf("x3_moe paired n=%d: one pass == two calls, bit for bit\n", n);
                cudaFree(dx); cudaFree(di); cudaFree(p1); cudaFree(p2);
            }
        }
        cudaFree(wd);
    };

    run_bank(512, 1024);   // rows: one 512-column item of the prompt lists
    run_bank(512, 3072);   // 192 k slices: the routed part splits its k range (kparts 2), P sums in another order

    std::printf(ok ? "x3_moe: OK\n" : "x3_moe: FAILED\n");
    return ok ? 0 : 1;
} catch (const std::exception & e) { std::fprintf(stderr, "x3_moe: %s\n", e.what()); return 1; }

// x3_window_bench: gemv_window.cu vs the serving x3v path on real PAW-27B-X3.1 matrices (diagnostic,
// not in ctest: needs the GGUF). Per matrix: max error of the window output vs the x3v output (both
// bf16-rounded, relative to max |y|), column bit-identity across n = 1..16, and graph-timed us / GB/s
// for both paths at n = 1 and n = 8, plus a 12-column verify: one call vs the old 8 + 4 split. A "+"
// joins tensors into one fused matvec (e.g. gate+up).
//
// build: nvcc -std=c++17 -O3 -arch=sm_86 -Isrc -Isrc/formats -Ikernels/formats/x3 tests/x3_window_bench.cu \
//          build/libtqllm.a -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lcusolver -lcuda -ldl
// run:   x3_window_bench <X3.1.gguf> [blk.0.ffn_gate+blk.0.ffn_up blk.0.ffn_down ...]
#include <cuda_bf16.h>
#include "gguf.hpp"
#include "x3_record.hpp"
#include "x3.cuh"
#include "x3_codecs.hpp"
#include "codec_mul1.cuh"
#include "gemv_window.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tqllm;
using namespace tqllm::x3;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct Mtx {
    std::string name;
    const X3Codec* codec;
    int words, rows, cols;
    uint16_t* T; half* suh; half* svh;
    size_t tbytes;
};

static Mtx load(GgufFile& f, const std::string& name) {
    const GgufTensor& t = f.tensors.at(name + ".weight");
    std::vector<char> raw = f.read(t);
    X3Record r; std::memcpy(&r, raw.data(), sizeof r);
    Mtx m{name, x3_codec_lookup(r.codec, int(r.words)), int(r.words), int(r.rows), int(r.cols)};
    m.tbytes = size_t(m.rows / 16) * (m.cols / 16) * m.words * 2;
    const char* body = raw.data() + sizeof(X3Record);
    CK(cudaMalloc(&m.T, m.tbytes)); CK(cudaMemcpy(m.T, body, m.tbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&m.suh, m.cols * 2)); CK(cudaMemcpy(m.suh, body + m.tbytes, m.cols * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&m.svh, m.rows * 2)); CK(cudaMemcpy(m.svh, body + m.tbytes + m.cols * 2, m.rows * 2, cudaMemcpyHostToDevice));
    return m;
}

template <class F>
static float graph_us(F body, cudaStream_t s, int iters = 50) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < iters; ++i) body();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e30f;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(a, s); CK(cudaGraphLaunch(ge, s)); cudaEventRecord(b, s); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms * 1000.f / iters);
    }
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    return best;
}

template <class Codec>
static void run_window(const std::vector<Mtx>& ms, const half* x, __nv_bfloat16* y, int ldy, int n, void* ws,
                       int sms, cudaStream_t s) {
    XwList L; xw_begin(L, ms[0].cols, n);
    int off = 0;
    for (const Mtx& m : ms) {
        xw_add(L, m.T, m.suh, m.svh, m.rows, off, sms);
        off += m.rows;
    }
    xw_bind_scratch(L, ws);
    xw_matvec<Codec>(L, x, y, ldy, s);
}

static void window(const std::vector<Mtx>& ms, const half* x, __nv_bfloat16* y, int ldy, int n, void* ws, int sms,
                   cudaStream_t s) {
    switch (ms[0].words) {
        case 56: run_window<Mul1Codec<7>>(ms, x, y, ldy, n, ws, sms, s); break;
        case 48: run_window<Mul1Codec<6>>(ms, x, y, ldy, n, ws, sms, s); break;
        default: printf("words %d not in bench\n", ms[0].words); exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <gguf> [names]\n", argv[0]); return 1; }
    std::vector<std::string> groups;
    for (int i = 2; i < argc; ++i) groups.push_back(argv[i]);
    if (groups.empty())
        groups = {"blk.0.ffn_gate+blk.0.ffn_up", "blk.0.ffn_gate", "blk.0.ffn_down", "blk.0.attn_qkv",
                  "blk.0.attn_gate", "blk.0.ssm_out", "blk.3.attn_q", "blk.3.attn_k+blk.3.attn_v", "blk.3.attn_output"};
    GgufFile f = gguf_open(argv[1]);
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    cudaStream_t s; cudaStreamCreate(&s);

    // shared buffers sized for the largest case
    const int MAXC = 17408, MAXR = 2 * 17408;
    std::vector<half> hx((size_t) XW_MAX_N * MAXC);
    unsigned st = 7;
    for (auto& v : hx) { st = st * 1664525u + 1013904223u; v = __float2half(((int(st >> 10) % 2001) - 1000) / 400.0f); }
    half* dx; float* dA; float* dC; half* dHad; __nv_bfloat16* dy; __nv_bfloat16* dyref; void* ws; int* locks;
    CK(cudaMalloc(&dx, hx.size() * 2)); CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dA, (size_t) XW_MAX_N * MAXC * 4)); CK(cudaMalloc(&dHad, (size_t) XW_MAX_N * MAXC * 2));
    CK(cudaMalloc(&dC, (size_t) XW_MAX_N * MAXR * 4));
    CK(cudaMalloc(&dy, (size_t) XW_MAX_N * MAXR * 2)); CK(cudaMalloc(&dyref, (size_t) XW_MAX_N * MAXR * 2));
    CK(cudaMalloc(&ws, size_t(64) << 20)); CK(cudaMemset(ws, 0, size_t(64) << 20)); CK(cudaMalloc(&locks, size_t(16) << 20)); CK(cudaMemset(locks, 0, size_t(16) << 20));

    {   // clocks ramp for ~1 s after idle: warm up so the first matrix is not timed on a cold GPU
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float ms = 0;
        cudaEventRecord(e0, s);
        while (ms < 1500.f) {
            for (int i = 0; i < 50; ++i) CK(cudaMemsetAsync(dC, i, (size_t) XW_MAX_N * MAXR * 4, s));
            cudaEventRecord(e1, s); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        }
    }
    printf("%-30s %5s %6s %4s | %9s %9s | %8s %8s %7s | %8s %8s %7s | %8s %8s\n", "matrix", "rows", "cols", "K",
           "rel.err", "n-ident", "x3v n1", "win n1", "GB/s", "x3v n8", "win n8", "GB/s", "win 8+4", "win n12");
    double tot_old = 0, tot_new = 0, tot_84 = 0, tot_12 = 0;
    for (const std::string& gname : groups) {
        std::vector<Mtx> ms;
        size_t p = 0;
        while (true) {
            size_t q = gname.find('+', p);
            ms.push_back(load(f, gname.substr(p, q == std::string::npos ? std::string::npos : q - p)));
            if (q == std::string::npos) break;
            p = q + 1;
        }
        const int cols = ms[0].cols;
        int rows = 0; size_t bytes = 0;
        for (auto& m : ms) { rows += m.rows; bytes += m.tbytes; }

        // reference: serving x3v path, f32 A/C, cast to bf16 per part (as x3.cpp does)
        auto old_path = [&](int n) {
            x3_cast_h_f32(dx, dA, n * cols, s);
            int off = 0;
            for (auto& m : ms) {
                m.codec->gemv(dA, m.T, dC, n, cols, m.rows, m.suh, dHad, m.svh, locks, s);
                x3_copy_f32_bf16_strided(dC, dyref, m.rows, rows, off, n, s);
                off += m.rows;
            }
        };
        old_path(8);
        window(ms, dx, dy, rows, 8, ws, sms, s);
        CK(cudaStreamSynchronize(s));
        std::vector<__nv_bfloat16> yr((size_t) 8 * rows), yw((size_t) 8 * rows), yn((size_t) 8 * rows);
        CK(cudaMemcpy(yr.data(), dyref, yr.size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yw.data(), dy, yw.size() * 2, cudaMemcpyDeviceToHost));
        double mx = 0, md = 0;
        for (size_t i = 0; i < yr.size(); ++i) {
            const double a = __bfloat162float(yr[i]), b = __bfloat162float(yw[i]);
            mx = std::max(mx, std::fabs(a)); md = std::max(md, std::fabs(a - b));
        }
        // every n = 1..16 against the n = 16 call (columns 0..7 also against n = 8 above)
        std::vector<__nv_bfloat16> y16((size_t) XW_MAX_N * rows);
        window(ms, dx, dy, rows, XW_MAX_N, ws, sms, s);
        CK(cudaMemcpy(y16.data(), dy, y16.size() * 2, cudaMemcpyDeviceToHost));
        int ident_bad = std::memcmp(y16.data(), yw.data(), (size_t) 8 * rows * 2) != 0;
        yn.resize(y16.size());
        for (int n = 1; n < XW_MAX_N; ++n) {
            CK(cudaMemset(dy, 0, (size_t) XW_MAX_N * rows * 2));
            window(ms, dx, dy, rows, n, ws, sms, s);
            CK(cudaMemcpy(yn.data(), dy, (size_t) n * rows * 2, cudaMemcpyDeviceToHost));
            ident_bad += std::memcmp(yn.data(), y16.data(), (size_t) n * rows * 2) != 0;
        }
        float t_old1 = graph_us([&] { old_path(1); }, s), t_new1 = graph_us([&] { window(ms, dx, dy, rows, 1, ws, sms, s); }, s);
        float t_old8 = graph_us([&] { old_path(8); }, s), t_new8 = graph_us([&] { window(ms, dx, dy, rows, 8, ws, sms, s); }, s);
        float t_84 = graph_us([&] {
            window(ms, dx, dy, rows, 8, ws, sms, s);
            window(ms, dx + (size_t) 8 * cols, dy + (size_t) 8 * rows, rows, 4, ws, sms, s);
        }, s);
        float t_12 = graph_us([&] { window(ms, dx, dy, rows, 12, ws, sms, s); }, s);
        tot_old += t_old1; tot_new += t_new1; tot_84 += t_84; tot_12 += t_12;
        printf("%-30s %5d %6d %4.1f | %9.2e %9s | %8.1f %8.1f %7.0f | %8.1f %8.1f %7.0f | %8.1f %8.1f\n", gname.c_str(), rows,
               cols, ms[0].words / 16.0, md / mx, ident_bad ? "FAIL" : "ok", t_old1, t_new1, bytes / t_new1 / 1e3, t_old8,
               t_new8, bytes / t_new8 / 1e3, t_84, t_12);
        for (auto& m : ms) { cudaFree(m.T); cudaFree(m.suh); cudaFree(m.svh); }
    }
    printf("sum n1: x3v %.1f us, window %.1f us (%.2fx)\n", tot_old, tot_new, tot_old / tot_new);
    printf("sum 12-column verify: 8+4 %.1f us, one call %.1f us (%.2fx)\n", tot_84, tot_12, tot_84 / tot_12);
    return 0;
}

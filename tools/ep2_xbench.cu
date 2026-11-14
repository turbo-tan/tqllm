// Expert-parallel exchange cost over two linked boxes, in situ. Per step and layer: a bandwidth-bound stand-in for the
// rank's own dense / GDN / attention work (pre MB), the input-row exchange (both directions at once), a stand-in for the
// owned-expert compute (exp MB), the result-row exchange. Each rank varies its per-layer work by +-var % on its own
// (seeded per rank), so the exchanges also absorb lockstep waits. The cost of EP's exchange = step time with exchanges
// - step time without, same kernels. Three modes: eager; the whole step as one graph; hybrid = the engine's shape under EP
// (per layer a graph before the input exchange and one between the exchanges, each exec-updated every step as the engine
// re-captures its layer graphs, the exchanges eager between them).
//   ep2-xbench RANK PEER PORT [steps] [case substring]
#include "core/tp_comm.hpp"
#include <nccl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void spin_k(long long) {}  // keeps a captured graph non-empty when its work is 0 MB
__global__ void stream_k(const int4 * __restrict__ x, size_t n, int * sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) { const int4 v = x[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x7fffffff) *sink = acc;
}
constexpr int kLayers = 48;
struct Case { const char * name; double pre_mb, exp_mb; size_t in, out; bool follower = false; };  // follower: rank 1 has no users (no pre work), inputs 0 -> 1, results 1 -> 0
struct Ctx { tqllm::TpComm * tp; char * si, * ri, * so, * ro; const int4 * big; int * sink; cudaStream_t s; };
void read_mb(const Ctx & c, double mb) { const size_t n = size_t(mb * 1e6) / 16; if (n) stream_k<<<48 * 8, 256, 0, c.s>>>(c.big, n, c.sink); }
// one exchange: both directions, or (follower) only `from` -> the other rank
void exchange(const Ctx & c, const Case & k, const char * sb, char * rb, size_t bytes, int from) {
    const int peer = 1 - c.tp->rank();
    if (!k.follower) { ncclGroupStart(); c.tp->send(sb, bytes, peer, c.s); c.tp->recv(rb, bytes, peer, c.s); ncclGroupEnd(); }
    else if (c.tp->rank() == from) c.tp->send(sb, bytes, peer, c.s);
    else c.tp->recv(rb, bytes, peer, c.s);
}
double pre_of(const Ctx & c, const Case & k) { return k.follower && c.tp->rank() == 1 ? 0 : k.pre_mb; }
void step(const Ctx & c, const Case & k, bool exch, std::mt19937 & rng, double var) {
    std::uniform_real_distribution<double> u(1 - var, 1 + var);
    for (int l = 0; l < kLayers; ++l) {
        read_mb(c, pre_of(c, k) * u(rng));
        if (exch) exchange(c, k, c.si, c.ri, k.in, 0);
        read_mb(c, k.exp_mb * u(rng));
        if (exch) exchange(c, k, c.so, c.ro, k.out, 1);
    }
}
// hybrid: per layer and variant v, graphs a (pre work) and b (expert work) captured once; per step each is exec-updated
// from its captured graph (the engine's per-step update) and launched, with the exchanges eager in between
struct Hybrid { std::vector<cudaGraph_t> ga, gb; std::vector<cudaGraphExec_t> ea, eb; };
Hybrid build_hybrid(const Ctx & c, const Case & k, std::mt19937 & rng, double var, int variants) {
    std::uniform_real_distribution<double> u(1 - var, 1 + var);
    Hybrid h; const int n = variants * kLayers;
    h.ga.resize(n); h.gb.resize(n); h.ea.resize(n); h.eb.resize(n);
    for (int i = 0; i < n; ++i) {
        ck(cudaStreamBeginCapture(c.s, cudaStreamCaptureModeThreadLocal)); read_mb(c, pre_of(c, k) * u(rng)); spin_k<<<1, 1, 0, c.s>>>(0); ck(cudaStreamEndCapture(c.s, &h.ga[i]));
        ck(cudaStreamBeginCapture(c.s, cudaStreamCaptureModeThreadLocal)); read_mb(c, k.exp_mb * u(rng)); ck(cudaStreamEndCapture(c.s, &h.gb[i]));
        ck(cudaGraphInstantiate(&h.ea[i], h.ga[i], 0)); ck(cudaGraphInstantiate(&h.eb[i], h.gb[i], 0));
    }
    return h;
}
void hybrid_step(const Ctx & c, const Case & k, bool exch, Hybrid & h, int v) {
    for (int l = 0; l < kLayers; ++l) {
        const int i = v * kLayers + l; cudaGraphExecUpdateResultInfo info;
        ck(cudaGraphExecUpdate(h.ea[i], h.ga[i], &info)); ck(cudaGraphLaunch(h.ea[i], c.s));
        if (exch) exchange(c, k, c.si, c.ri, k.in, 0);
        ck(cudaGraphExecUpdate(h.eb[i], h.gb[i], &info)); ck(cudaGraphLaunch(h.eb[i], c.s));
        if (exch) exchange(c, k, c.so, c.ro, k.out, 1);
    }
}
}  // namespace

int main(int argc, char ** argv) try {
    if (argc < 4) { std::fprintf(stderr, "usage: ep2-xbench RANK PEER PORT [steps] [case substring]\n"); return 2; }
    const int rank = std::atoi(argv[1]), steps = argc > 4 ? std::atoi(argv[4]) : 20;
    tqllm::TpComm tp(rank, 2, argv[2], std::atoi(argv[3]));
    Ctx c{&tp};
    const size_t cap = size_t(2) << 20, big = size_t(1) << 30;  // 1 GB source: no L2 reuse across launches
    for (char ** p : {&c.si, &c.ri, &c.so, &c.ro}) { ck(cudaMalloc(p, cap)); ck(cudaMemset(*p, rank + 1, cap)); }
    ck(cudaMalloc(const_cast<int4 **>(&c.big), big)); ck(cudaMemset(const_cast<int4 *>(c.big), 1, big)); ck(cudaMalloc(&c.sink, 4));
    ck(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
    // stage 0 shapes (per layer): 16 users = 8 per box, step ~145 ms (experts 72 -> ~58 after EP); 1 user, step ~48 ms (experts 17.6 -> ~9).
    // pre / exp MB at ~240 GB/s; in = all n input rows FP16 x 2560 + ids; out = the result-row cap (per assignment BF16 x 2560).
    const std::vector<Case> cases = {
        {"16u cap p99 150", 360, 290, 24 * 5120 + 1024, 150 * 5120},
        {"16u cap max 176", 360, 290, 24 * 5120 + 1024, 176 * 5120},
        {"1u cap 40", 150, 45, 8 * 5120 + 512, 40 * 5120},
        {"1u cap 80", 150, 45, 8 * 5120 + 512, 80 * 5120},
        {"empty 1 B", 150, 45, 1, 1},
        {"1u follower cap 40", 150, 45, 8 * 5120 + 512, 40 * 5120, true},
        {"1u follower cap 80", 150, 45, 8 * 5120 + 512, 80 * 5120, true},
        {"follower empty", 150, 45, 1, 1, true},
        // exact TP2 (11-13): per layer, in = all-gather of the head outputs (rows x 3072 FP32 per rank) before the replicated
        // output projection, out = expert results per assignment (BF16 x 2560); work = per-rank stand-ins at ~240 GB/s
        {"tp2 16u p50", 70, 297, 48 * 3072 * 4, 240 * 5120},
        {"tp2 16u max", 70, 297, 48 * 3072 * 4, 300 * 5120},
        {"tp2 8u", 45, 195, 24 * 3072 * 4, 120 * 5120},
        {"tp2 1u", 24, 56, 3 * 3072 * 4, 15 * 5120},
    };
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    for (const auto & k : cases) for (double var : {0.0, 0.10}) {
        if (argc > 5 && !std::strstr(k.name, argv[5])) continue;
        double t[2][3];  // [exch][eager, graph, hybrid]
        for (int exch = 0; exch < 2; ++exch) {
            std::mt19937 rng(1234 + 7919 * rank);
            for (int w = 0; w < 2; ++w) step(c, k, exch, rng, var);
            ck(cudaStreamSynchronize(c.s));
            auto t0 = now();
            for (int i = 0; i < steps; ++i) step(c, k, exch, rng, var);
            ck(cudaStreamSynchronize(c.s)); t[exch][0] = ms(t0, now()) / steps;
            // graph: a few captured variants (different random work) replayed round robin
            std::vector<cudaGraphExec_t> ex(4);
            for (auto & e : ex) { cudaGraph_t g; ck(cudaStreamBeginCapture(c.s, cudaStreamCaptureModeThreadLocal)); step(c, k, exch, rng, var);
                                  ck(cudaStreamEndCapture(c.s, &g)); ck(cudaGraphInstantiate(&e, g, 0)); cudaGraphDestroy(g); }
            for (int w = 0; w < 4; ++w) ck(cudaGraphLaunch(ex[w], c.s));
            ck(cudaStreamSynchronize(c.s));
            t0 = now();
            for (int i = 0; i < steps; ++i) ck(cudaGraphLaunch(ex[i % 4], c.s));
            ck(cudaStreamSynchronize(c.s)); t[exch][1] = ms(t0, now()) / steps;
            for (auto e : ex) cudaGraphExecDestroy(e);
            Hybrid h = build_hybrid(c, k, rng, var, 4);
            for (int w = 0; w < 4; ++w) hybrid_step(c, k, exch, h, w);
            ck(cudaStreamSynchronize(c.s));
            t0 = now();
            for (int i = 0; i < steps; ++i) hybrid_step(c, k, exch, h, i % 4);
            ck(cudaStreamSynchronize(c.s)); t[exch][2] = ms(t0, now()) / steps;
            for (size_t i = 0; i < h.ea.size(); ++i) { cudaGraphExecDestroy(h.ea[i]); cudaGraphExecDestroy(h.eb[i]); cudaGraphDestroy(h.ga[i]); cudaGraphDestroy(h.gb[i]); }
        }
        std::printf("rank %d %-16s var %2.0f%% in %6zu out %6zu B: no-exch eager %7.2f graph %7.2f ms/step; exch eager %7.2f graph %7.2f;"
                    " hybrid no-exch %7.2f exch %7.2f; EXCHANGE COST eager %5.2f graph %5.2f HYBRID %5.2f ms/step\n", rank, k.name, var * 100, k.in, k.out,
                    t[0][0], t[0][1], t[1][0], t[1][1], t[0][2], t[1][2], t[1][0] - t[0][0], t[1][1] - t[0][1], t[1][2] - t[0][2]);
        std::fflush(stdout);
    }
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "ep2-xbench: %s\n", e.what()); return 1; }

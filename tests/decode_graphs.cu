// Flash-Next decode graphs (fn_graphs.hpp) under request churn: many owners (DecodeModels) each warm up, capture and
// launch a step graph, then end. Released owners leave no execs, no warm-up entries and no device memory behind; a new
// owner's first calls run plain; an owner that is never released keeps its execs (what every finished request did when
// graphs were keyed by the model's address: G2). The launched graphs ran (a counter).
#include "fn_graphs.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

using tqllm::flashnext::DecodeGraphs;
using tqllm::flashnext::next_graph_owner;
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static int failures = 0;
static void expect(bool ok, const std::string & what) { std::printf("%s: %s\n", what.c_str(), ok ? "ok" : "FAIL"); failures += !ok; }
__global__ void bump(int * c, int k) { atomicAdd(c, k); }

// one decode step of a layer for `owner` (rows n): plain unless the warm-up says capture, as layer_forward_step does
static bool step(DecodeGraphs & g, uint64_t owner, int layer, int n, int * counter) {
    const bool cap = g.capture(owner, layer, n, false);
    if (cap) check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < 4; ++k) bump<<<1, 1, 0, cudaStreamPerThread>>>(counter, 1);
    if (cap) { cudaGraph_t cg = nullptr; check(cudaStreamEndCapture(cudaStreamPerThread, &cg)); g.launch(owner, layer, n, cg, cudaStreamPerThread); }
    return cap;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    int * counter = nullptr; check(cudaMalloc(&counter, sizeof(int))); check(cudaMemset(counter, 0, sizeof(int)));
    DecodeGraphs g;
    const int owners = 300, layers = 6, steps = 5;
    size_t free0 = 0, total = 0;
    long launched = 0;
    for (int o = 0; o < owners; ++o) {
        const uint64_t owner = next_graph_owner();
        bool first_plain = true;
        for (int s = 0; s < steps; ++s)
            for (int l = 0; l < layers; ++l) { const bool cap = step(g, owner, l, 1 + (o % 4), counter); if (s == 0 && cap) first_plain = false; ++launched; }
        if (!first_plain) { expect(false, "owner " + std::to_string(o) + ": first step runs plain"); break; }
        check(cudaStreamSynchronize(cudaStreamPerThread));
        g.release(owner);  // the request ended
        if (o == 9) check(cudaMemGetInfo(&free0, &total));  // after the CUDA graph runtime warmed up
    }
    check(cudaStreamSynchronize(cudaStreamPerThread));
    size_t free1 = 0; check(cudaMemGetInfo(&free1, &total));
    int ran = 0; check(cudaMemcpy(&ran, counter, sizeof(int), cudaMemcpyDeviceToHost));
    expect(ran == int(launched * 4), "every step ran its kernels, captured or plain (" + std::to_string(ran) + ")");
    expect(g.execs() == 0 && g.warm_entries() == 0, "released owners leave no execs and no warm-up entries");
    expect(free1 + (size_t(8) << 20) >= free0, "device memory after " + std::to_string(owners) + " owners within 8 MiB of after 10 (" +
           std::to_string((long long)(free0 - free1) >> 10) + " KiB used)");
    // not released (the address-keyed tables of 56e34552 kept every finished request's execs)
    for (int o = 0; o < 20; ++o) { const uint64_t owner = next_graph_owner(); for (int s = 0; s < steps; ++s) for (int l = 0; l < layers; ++l) step(g, owner, l, 2, counter); }
    check(cudaStreamSynchronize(cudaStreamPerThread));
    expect(g.execs() >= size_t(20) * layers && g.execs() <= size_t(20) * layers * DecodeGraphs::kExecsPerShape,
           "owners never released keep their execs (" + std::to_string(g.execs()) + ", one or more per shape: one per launch in flight)");
    cudaFree(counter);
    std::puts(failures ? "decode_graphs FAIL" : "decode_graphs PASS");
    return failures ? 1 : 0;
}

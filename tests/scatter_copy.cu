// scatter_copy (one launch for many device-to-device copies; Flash-Next batches each layer's rollback saves with it) must
// equal one cudaMemcpy per item byte for byte: aligned and unaligned sources / destinations / sizes, more items than
// kScatterMax (several launches), and inside a captured graph.
#include "experts/moe_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void check(cudaError_t x) { if (x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
static bool run(int items, bool graph, unsigned seed) {
    std::mt19937 rng(seed);
    const size_t cap = size_t(1) << 22;
    char * src = nullptr, * a = nullptr, * b = nullptr;
    check(cudaMalloc(&src, cap)); check(cudaMalloc(&a, cap)); check(cudaMalloc(&b, cap));
    std::vector<char> h(cap); for (auto & c : h) c = char(rng());
    check(cudaMemcpy(src, h.data(), cap, cudaMemcpyHostToDevice));
    check(cudaMemset(a, 0x5a, cap)); check(cudaMemset(b, 0x5a, cap));
    std::vector<tqllm_cuda::ScatterItem> list; size_t at = 0;
    for (int i = 0; i < items; ++i) {  // disjoint destinations; sizes 1 B .. 64 KB, offsets with every alignment
        const size_t bytes = i % 3 ? size_t(rng() % 65536 + 1) : size_t(16 * (rng() % 4096 + 1));
        const size_t so = rng() % (cap - bytes), d = at + (i % 2 ? rng() % 16 : 0);
        if (d + bytes > cap) break;
        list.push_back({src + so, a + d, bytes}); at = (d + bytes + 15) / 16 * 16;
        check(cudaMemcpy(b + d, src + so, bytes, cudaMemcpyDeviceToDevice));
    }
    check(cudaDeviceSynchronize());  // the fills above run on the default stream; the scatter's stream does not wait for them
    cudaStream_t s; check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (graph) {
        cudaGraph_t g; cudaGraphExec_t e;
        check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); tqllm_cuda::scatter_copy(list.data(), int(list.size()), s);
        check(cudaStreamEndCapture(s, &g)); check(cudaGraphInstantiate(&e, g, 0)); check(cudaGraphLaunch(e, s));
        check(cudaStreamSynchronize(s)); cudaGraphExecDestroy(e); cudaGraphDestroy(g);
    } else { tqllm_cuda::scatter_copy(list.data(), int(list.size()), s); check(cudaStreamSynchronize(s)); }
    std::vector<char> x(cap), y(cap);
    check(cudaMemcpy(x.data(), a, cap, cudaMemcpyDeviceToHost)); check(cudaMemcpy(y.data(), b, cap, cudaMemcpyDeviceToHost));
    cudaFree(src); cudaFree(a); cudaFree(b); cudaStreamDestroy(s);
    return std::memcmp(x.data(), y.data(), cap) == 0;
}
int main() {
    try {
        int fails = 0, cases = 0;
        for (int items : {1, 7, 32, 128, 129, 300}) for (bool graph : {false, true}) { fails += !run(items, graph, unsigned(items * 2 + graph)); ++cases; }
        std::printf("%s: %d cases\n", fails ? "scatter copy FAILED" : "scatter copy PASS", cases);
        return fails ? 1 : 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}

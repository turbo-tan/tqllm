// Flash-Next device buffers inside a CUDA graph capture (decode graphs): a buffer replaced (freed) or created during a
// captured decode step must leave no error and stay valid after the graph runs.
#include "models/qwen/qwen4exp/fn_runtime.hpp"
#include <cstdio>

using namespace tqllm;
static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++failures; } }

int main() {
    cudaStream_t s = cudaStreamPerThread;
    {   // replaced during capture: the old pool buffer is freed inside the capture
        auto old = std::make_unique<Device<float>>(1024);
        cudaGraph_t g = nullptr;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        old = std::make_unique<Device<float>>(4096);
        ck(cudaMemsetAsync(old->p, 0, 4096 * sizeof(float), s));
        const cudaError_t in = cudaPeekAtLastError();
        const cudaError_t end = cudaStreamEndCapture(s, &g);
        check(in == cudaSuccess && end == cudaSuccess && g, "pool buffer freed inside a capture leaves no error");
        cudaGetLastError();
        if (g) { cudaGraphExec_t e = nullptr; check(cudaGraphInstantiate(&e, g, 0) == cudaSuccess && cudaGraphLaunch(e, s) == cudaSuccess &&
                                                    cudaStreamSynchronize(s) == cudaSuccess, "graph with the replaced buffer runs");
                 if (e) cudaGraphExecDestroy(e); cudaGraphDestroy(g); }
    }
    {   // created during capture (lazy state): usable after the graph and after the capture closed
        std::unique_ptr<Device<int>> lazy;
        cudaGraph_t g = nullptr;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        lazy = std::make_unique<Device<int>>(256);
        ck(cudaMemsetAsync(lazy->p, 1, 256 * sizeof(int), s));
        check(cudaStreamEndCapture(s, &g) == cudaSuccess && g && cudaPeekAtLastError() == cudaSuccess, "buffer created inside a capture leaves no error");
        cudaGetLastError();
        cudaGraphExec_t e = nullptr;
        for (int launch = 0; launch < 2 && g; ++launch) {  // the address stays valid across launches (not graph-owned)
            if (!e) ck(cudaGraphInstantiate(&e, g, 0));
            ck(cudaGraphLaunch(e, s)); ck(cudaStreamSynchronize(s));
            std::vector<int> h(256, 0); ck(cudaMemcpy(h.data(), lazy->p, 256 * sizeof(int), cudaMemcpyDeviceToHost));
            check(h[0] == 0x01010101 && h[255] == 0x01010101, "buffer created inside a capture holds the graph's writes");
            ck(cudaMemset(lazy->p, 0, 256 * sizeof(int)));
        }
        if (e) cudaGraphExecDestroy(e); if (g) cudaGraphDestroy(g);
        lazy.reset();
        check(cudaGetLastError() == cudaSuccess, "buffer created inside a capture frees cleanly");
    }
    std::printf("graph capture alloc: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

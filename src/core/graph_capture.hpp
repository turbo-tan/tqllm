#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tqllm {
// Takes ownership of the captured graph, including when instantiation fails.
template <class Create, class Destroy>
cudaGraphExec_t instantiate_graph(cudaGraph_t g, Create create, Destroy destroy) {
    struct Owner {
        cudaGraph_t graph;
        Destroy & destroy;
        ~Owner() { destroy(graph); }
    } owner{g, destroy};
    cudaGraphExec_t exec = nullptr;
    if (const cudaError_t e = create(&exec, g); e != cudaSuccess)
        throw std::runtime_error(std::string("graph instantiate: ") + cudaGetErrorString(e));
    return exec;
}
inline cudaGraphExec_t instantiate_graph(cudaGraph_t g) {
    return instantiate_graph(g, [](cudaGraphExec_t * out, cudaGraph_t graph) {
        return cudaGraphInstantiate(out, graph, 0);
    }, [](cudaGraph_t graph) { return cudaGraphDestroy(graph); });
}

// Records fn() into a CUDA graph on stream s. If fn throws, the capture is ended and its graph discarded before the
// exception propagates: a stream left capturing fails every later call ("operation not permitted when stream is
// capturing"), which turned one failed request into a dead server.
template <class F>
cudaGraph_t capture_graph(cudaStream_t s, F && fn) {
    if (const cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); e != cudaSuccess)
        throw std::runtime_error(std::string("graph capture: ") + cudaGetErrorString(e));
    try {
        fn();
    } catch (...) {
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(s, &g);
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();
        throw;
    }
    cudaGraph_t g = nullptr;
    if (const cudaError_t e = cudaStreamEndCapture(s, &g); e != cudaSuccess)
        throw std::runtime_error(std::string("graph capture: ") + cudaGetErrorString(e));
    return g;
}
}  // namespace tqllm

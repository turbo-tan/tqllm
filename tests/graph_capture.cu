// A failure inside a graph capture must leave the stream usable (a server kept failing every later request).
#include "graph_capture.hpp"

#include <cstdio>
#include <stdexcept>

static int failures = 0;
static void check(bool ok, const char * what) {
    std::printf("%-70s %s\n", what, ok ? "ok" : "FAIL");
    failures += !ok;
}

int main() {
    int * d = nullptr;
    cudaMalloc(&d, 1024);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    // the old pattern (begin capture, the body throws, end never reached) leaves the stream capturing
    {
        cudaStream_t s2;
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        try {
            cudaStreamBeginCapture(s2, cudaStreamCaptureModeThreadLocal);
            cudaMemsetAsync(d, 0, 1024, s2);
            throw std::runtime_error("body failed");
        } catch (const std::exception &) {
        }
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s2, &st);
        check(st == cudaStreamCaptureStatusActive, "without the guard the stream stays capturing (the bug)");
        cudaGraph_t g = nullptr;
        cudaStreamEndCapture(s2, &g);
        if (g) cudaGraphDestroy(g);
        cudaGetLastError();
        cudaStreamDestroy(s2);
    }
    bool threw = false;
    try {
        tqllm::capture_graph(s, [&] {
            cudaMemsetAsync(d, 0, 1024, s);
            throw std::runtime_error("body failed");
        });
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "capture_graph rethrows the body's exception");
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusActive;
    cudaStreamIsCapturing(s, &st);
    check(st == cudaStreamCaptureStatusNone, "stream not capturing after a failed capture");
    check(cudaMemsetAsync(d, 1, 1024, s) == cudaSuccess && cudaStreamSynchronize(s) == cudaSuccess,
          "the stream runs work again after a failed capture");
    cudaGraph_t g = tqllm::capture_graph(s, [&] { cudaMemsetAsync(d, 2, 1024, s); });
    cudaGraphExec_t ex = nullptr;
    const bool ok = g && cudaGraphInstantiate(&ex, g, 0) == cudaSuccess && cudaGraphLaunch(ex, s) == cudaSuccess &&
                    cudaStreamSynchronize(s) == cudaSuccess;
    check(ok, "a later capture records and replays");
    if (ex) cudaGraphExecDestroy(ex);
    if (g) cudaGraphDestroy(g);
    cudaStreamDestroy(s);
    cudaFree(d);
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("graph_capture: ok");
    return 0;
}

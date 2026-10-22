// A request that failed on a CUDA out-of-memory must not break the next one: the pending (non-sticky) error is cleared
// at the start of a request, after which QSA (which checks with cudaPeekAtLastError) works again.
#include "qwen4exp_kernels.cuh"
#include "../src/cuda_recover.hpp"

#include <cstdio>
#include <stdexcept>

int main() {
    int bad = 0;
    void * p = nullptr;
    if (cudaMalloc(&p, size_t(1) << 50) == cudaSuccess) { std::printf("FAIL: a 1 PiB allocation succeeded\n"); return 1; }
    bool broken = false;
    try { tqllm_cuda::QsaSelector s(4096, 8, 2048); } catch (const std::exception & e) { broken = true; std::printf("  stale error, not cleared: %s\n", e.what()); }
    if (!broken) std::printf("  note: this CUDA build does not report the stale error here\n");
    if (cudaMalloc(&p, size_t(1) << 50) == cudaSuccess) { std::printf("FAIL: a 1 PiB allocation succeeded\n"); return 1; }
    if (tqllm::clear_stale_cuda_error() != cudaSuccess) { std::printf("FAIL: a non-sticky error was reported as fatal\n"); ++bad; }
    try { tqllm_cuda::QsaSelector s(4096, 8, 2048); std::printf("PASS: QSA works after the stale error is cleared\n"); }
    catch (const std::exception & e) { std::printf("FAIL: QSA after clearing: %s\n", e.what()); ++bad; }
    std::printf(bad ? "STALE_ERROR_FAILED\n" : "STALE_ERROR_OK\n");
    return bad ? 1 : 0;
}

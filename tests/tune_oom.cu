// Shape autotuning under a full device: the tuning scratch must fail closed (throw, context still usable),
// not launch its timing kernels on null buffers (a sticky illegal address seen by the next unrelated CUDA call).
#include "quant_kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;
static void expect(bool ok, const std::string & what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// takes all free device memory, down to 1 KB pieces
static std::vector<void *> fill_device() {
    std::vector<void *> held;
    for (size_t piece = size_t(1) << 30; piece >= 1024; piece /= 4)
        for (void * p = nullptr; cudaMalloc(&p, piece) == cudaSuccess; p = nullptr) held.push_back(p);
    cudaGetLastError();
    return held;
}

static void check_case(const char * name, const std::function<void()> & tune) {
    auto held = fill_device();
    void * probe = nullptr;
    const bool full = cudaMalloc(&probe, size_t(8) * 4096 * 2) != cudaSuccess;
    if (!full) cudaFree(probe);
    cudaGetLastError();
    expect(full, std::string(name) + ": the device is full (a 64 KB allocation fails)");
    bool threw = false;
    try { tune(); } catch (const std::runtime_error & e) { threw = true; std::printf("  %s threw: %s\n", name, e.what()); }
    for (void * p : held) cudaFree(p);
    const cudaError_t sync = cudaDeviceSynchronize();
    expect(threw, std::string(name) + " throws when its scratch cannot be allocated");
    expect(sync == cudaSuccess, std::string(name) + " leaves the context usable (" + cudaGetErrorString(sync) + ")");
    if (sync != cudaSuccess) return;
    bool ok = true;
    try { tune(); } catch (const std::exception &) { ok = false; }
    expect(ok && cudaDeviceSynchronize() == cudaSuccess, std::string(name) + " tunes once memory is free again");
}

int main() {
    // filling the device on unified memory (GB10) takes the host's memory too and has hung a box: skip there
    int dev = 0, integrated = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) == cudaSuccess && integrated) {
        std::printf("SKIP: tune_oom fills device memory, which is host memory on an integrated GPU\n");
        return 0;
    }
    const int rows = 512, k = 4096;
    void * w = nullptr, * aux = nullptr;
    if (cudaMalloc(&w, size_t(rows) * k) != cudaSuccess || cudaMalloc(&aux, size_t(rows) * k / 8) != cudaSuccess) {
        std::printf("FAIL: weight allocation\n");
        return 1;
    }
    cudaMemset(w, 0, size_t(rows) * k);
    cudaMemset(aux, 0, size_t(rows) * k / 8);
    check_case("iq3_tune", [&] { tqllm_cuda::iq3_tune(static_cast<const uint8_t *>(w), static_cast<const uint8_t *>(aux), rows, k, nullptr); });
    check_case("tq3_tune", [&] { tqllm_cuda::tq3_tune(w, rows + 256, k, nullptr); });
    cudaFree(w);
    cudaFree(aux);
    std::printf("%s\n", failures ? "TUNE_OOM_FAILED" : "TUNE_OOM_OK");
    return failures ? 1 : 0;
}

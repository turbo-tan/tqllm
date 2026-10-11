// The KV cache's device memory must be counted where the memory fit reads it (dev_bytes(), WSL / WDDM): an uncounted
// all-on-GPU KV cache let a 262K context load on a 16 GB 5060 Ti and spill into shared memory during prefill.
#include "dev_alloc.cuh"
#include "kv_store.hpp"

#include <cstdio>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char * what) { std::printf("%-62s %s\n", what, ok ? "ok" : "FAIL"); failures += !ok; };
    auto mem = tqllm::make_cuda_kv_memory(nullptr);
    const size_t n = size_t(64) << 20, before = tqllm_cuda::dev_bytes();
    void * p = mem->device_alloc(n);
    check(p != nullptr, "device_alloc of 64 MiB");
    check(tqllm_cuda::dev_bytes() == before + n, "device_alloc is counted in dev_bytes()");
    mem->device_free(p);
    check(tqllm_cuda::dev_bytes() == before, "device_free gives it back");
    void * h = mem->host_alloc(n);
    check(h != nullptr && tqllm_cuda::dev_bytes() == before, "host_alloc is not device memory");
    mem->host_free(h);
    if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
    std::puts("kv_mem_count: ok");
    return 0;
}

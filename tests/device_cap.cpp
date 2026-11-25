// see device_cap.h
#include "device_cap.h"

#if defined(__linux__)
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {
struct State {
    std::mutex m;
    std::unordered_map<void *, size_t> live;  // allocations made through cudaMalloc since the process started
    size_t used = 0, cap = SIZE_MAX;
};
State & state() { static State s; return s; }
// what the driver takes for a request (RTX 3090, cudaMemGetInfo over 16 allocations each): a power of two up to 1 MiB,
// whole 2 MiB pages above. The tests count on it: a bank of several pieces frees more than a rebuilt one-piece bank needs
size_t footprint(size_t bytes) {
    if (bytes > (size_t(1) << 20)) return (bytes + (size_t(2) << 20) - 1) & ~((size_t(2) << 20) - 1);
    size_t f = 512; while (f < bytes) f *= 2;
    return f;
}
template<class F> F real(const char * name) {
    void * f = dlsym(RTLD_NEXT, name);
    if (!f) { std::fprintf(stderr, "device_cap: no %s in the CUDA runtime\n", name); std::abort(); }
    return reinterpret_cast<F>(f);
}
} // namespace

void device_cap::limit(size_t room) {
    auto & s = state(); std::lock_guard<std::mutex> g(s.m);
    s.cap = room == SIZE_MAX ? SIZE_MAX : s.used + room;
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void ** p, size_t bytes) {
    static const auto next = real<cudaError_t (*)(void **, size_t)>("cudaMalloc");
    auto & s = state();
    std::lock_guard<std::mutex> g(s.m);
    const size_t take = footprint(bytes);
    if (s.cap != SIZE_MAX && (take > s.cap || s.used > s.cap - take)) { if (p) *p = nullptr; return cudaErrorMemoryAllocation; }
    const cudaError_t e = next(p, bytes);
    if (e == cudaSuccess && *p) { s.live[*p] = take; s.used += take; }
    return e;
}
extern "C" cudaError_t CUDARTAPI cudaFree(void * p) {
    static const auto next = real<cudaError_t (*)(void *)>("cudaFree");
    auto & s = state();
    {   std::lock_guard<std::mutex> g(s.m);
        if (const auto it = s.live.find(p); it != s.live.end()) { s.used -= it->second; s.live.erase(it); } }
    return next(p);
}
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(size_t * free_bytes, size_t * total) {
    static const auto next = real<cudaError_t (*)(size_t *, size_t *)>("cudaMemGetInfo");
    const cudaError_t e = next(free_bytes, total);
    auto & s = state();
    std::lock_guard<std::mutex> g(s.m);
    if (e == cudaSuccess && s.cap != SIZE_MAX && free_bytes) {
        const size_t left = s.cap > s.used ? s.cap - s.used : 0;
        if (left < *free_bytes) *free_bytes = left;
    }
    return e;
}
#else
void device_cap::limit(size_t) {}
#endif

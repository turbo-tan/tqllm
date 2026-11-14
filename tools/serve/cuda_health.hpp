#pragma once
// A request that fails may leave the CUDA context in one of two states. A pending non-sticky error (an out-of-memory
// launch or allocation): cleared on the engine's thread, the next request runs normally. A sticky error (a kernel fault):
// the context is lost and every later request fails ("illegal memory access", 11-14 3090 27B, Xid 31), so tqserve fails
// that request and exits with kContextLostStatus for a supervisor (systemd, Docker) to restart it.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#if TQLLM_WITH_CUDA
#include "cuda_recover.hpp"
#endif

namespace tqserve {
constexpr int kContextLostStatus = 70;  // EX_SOFTWARE
inline std::atomic<bool> & context_lost() { static std::atomic<bool> lost{false}; return lost; }
// After the process has answered the failed request (the HTTP threads get `grace_ms`), it exits.
inline void exit_after_context_lost(const std::string & why, int grace_ms = 2000) {
    if (context_lost().exchange(true)) return;
    std::fprintf(stderr, "tqserve: the CUDA context is lost (%s); exiting with status %d so it can be restarted\n", why.c_str(), kContextLostStatus);
    std::fflush(stderr);
    std::thread([grace_ms] {
        std::this_thread::sleep_for(std::chrono::milliseconds(grace_ms));
        std::fflush(nullptr);
        std::_Exit(kContextLostStatus);
    }).detach();
}
// Before a request and after a failed one, on the thread that runs the engine: clears a pending non-sticky error; a lost
// context schedules the exit and returns its error text ("" when the context is usable).
inline std::string check_cuda_context() {
#if TQLLM_WITH_CUDA
    if (const cudaError_t e = tqllm::clear_stale_cuda_error(); e != cudaSuccess) {
        const std::string why = cudaGetErrorString(e);
        exit_after_context_lost(why);
        return why;
    }
#endif
    return context_lost() ? std::string("lost earlier") : std::string();
}
}  // namespace tqserve

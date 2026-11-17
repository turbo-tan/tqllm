#pragma once
// Threadgroup sizes of the custom kernels that dispatch 1024 threads (issue #55). A pipeline's limit
// (maxTotalThreadsPerThreadgroup) depends on the GPU and on the kernel's registers: an M1 Max allowed 384 threads where
// M4 / M5 allow 1024. MLX checks it when the kernel is evaluated and throws, so each specialization of such a kernel
// runs once on small inputs and takes the largest size that runs. TQLLM_DEBUG=no_tg1024 caps the sizes at 384 (as the
// M1 Max) to check the smaller sizes on a GPU that allows 1024.
#include "debug.hpp"
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>

namespace tqllm::mlx_backend {
inline constexpr int kNoTg1024Cap = 384;
inline int threadgroup_cap() {
    static const int cap = tqllm::debug_on("no_tg1024") ? kNoTg1024Cap : 1024;
    return cap;
}
// MLX's "Thread group size (N) is greater than the maximum allowed threads per threadgroup (L)"
inline bool threadgroup_too_large(const std::exception& e) {
    return std::string_view(e.what()).find("maximum allowed threads per threadgroup") != std::string_view::npos;
}
// run(threads) evaluates the kernel at that size; false when MLX turns the size away (other errors propagate)
inline bool threadgroup_runs(const std::function<void(int)>& run, int threads, std::string* why = nullptr) {
    try {
        run(threads);
        return true;
    } catch (const std::exception& e) {
        if (!threadgroup_too_large(e)) throw;
        if (why) *why = e.what();
        return false;
    }
}
// the first of `sizes` (largest first, each <= threadgroup_cap()) that run accepts, once per (kernel, spec); 0 if none.
// A size below 1024 is reported once per kernel.
inline int fit_threadgroup(const char* kernel, const std::string& spec, std::initializer_list<int> sizes,
                           const std::function<void(int)>& run) {
    static std::mutex mu;
    static std::map<std::string, int> fits;
    static std::set<std::string> told;
    std::lock_guard<std::mutex> lock(mu);
    const std::string key = std::string(kernel) + "|" + spec;
    if (const auto it = fits.find(key); it != fits.end()) return it->second;
    int got = 0;
    std::string why;
    for (const int t : sizes) {
        if (t > threadgroup_cap()) {
            if (why.empty()) why = "TQLLM_DEBUG=no_tg1024";
            continue;
        }
        if (threadgroup_runs(run, t, &why)) {got = t; break;}
    }
    if (got < 1024 && told.insert(kernel).second)
        std::fprintf(stderr, "tqllm: %s: %s; using %s (issue #55)\n", kernel, why.c_str(),
                     got ? (std::to_string(got) + " threads per threadgroup").c_str() : "the unfused path");
    fits[key] = got;
    return got;
}
}  // namespace tqllm::mlx_backend

#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

namespace tqllm {
// One caller, which also runs the first share; workers remain alive across all projections in a generation.
// Workers spin briefly between dispatches (a decode step issues hundreds of them back to back) and then sleep,
// so an idle engine does not hold the cores. Android puts background apps in a cpuset with fewer cores; more
// spinning threads than allowed cores starve the ones holding work, so the count is capped at the affinity mask.
class HostExecutor {
public:
    static int usable(int threads) {
#if defined(__linux__)
        cpu_set_t set;
        if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0) return std::min(threads, CPU_COUNT(&set));
#endif
        return threads;
    }
    explicit HostExecutor(int threads) : threads_(threads < 1 || threads > 64 ? threads : usable(threads)) {
        if (threads < 1 || threads > 64) throw std::invalid_argument("host executor thread count");
        try {
            for (int i = 1; i < threads_; ++i) workers_.emplace_back([this, i] { run(i); });
        } catch (...) { stop(); throw; }
    }
    ~HostExecutor() { stop(); }
    HostExecutor(const HostExecutor &) = delete;
    HostExecutor & operator=(const HostExecutor &) = delete;
    void rows(int count, const std::function<void(int, int)> & fn) {
        fn_ = &fn; count_ = count; error_ = nullptr;
        pending_.store(threads_ - 1, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst)) { std::lock_guard<std::mutex> lock(mutex_); ready_.notify_all(); }
        share(0);
        for (int spins = 0; pending_.load(std::memory_order_acquire);) pause(spins);
        fn_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }
private:
    // Spin on the CPU hint; every 256 rounds give the core to the scheduler in case a worker was preempted.
    static void pause(int & spins) {
        if (++spins % 256 == 0) { std::this_thread::yield(); return; }
#if defined(__aarch64__)
        asm volatile("yield");
#endif
    }
public:
    int threads() const { return threads_; }
private:
    void share(int id) {
        const int begin = int(int64_t(count_) * id / threads_), end = int(int64_t(count_) * (id + 1) / threads_);
        try { if (begin < end || count_ == 0) (*fn_)(begin, end); }
        catch (...) { std::lock_guard<std::mutex> lock(mutex_); if (!error_) error_ = std::current_exception(); }
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto & worker : workers_) if (worker.joinable()) worker.join();
    }
    void run(int id) {
        uint64_t seen = 0;
        for (;;) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
            int spins = 0;
            while (epoch_.load(std::memory_order_acquire) == seen && !stopping_.load()) {
                pause(spins);
                if (spins % 1024 == 0 && std::chrono::steady_clock::now() > deadline) {
                    std::unique_lock<std::mutex> lock(mutex_);
                    sleepers_.fetch_add(1, std::memory_order_seq_cst);
                    ready_.wait(lock, [&] { return stopping_.load() || epoch_.load(std::memory_order_seq_cst) != seen; });
                    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
                }
            }
            if (stopping_.load()) return;
            seen = epoch_.load(std::memory_order_acquire);
            share(id);
            pending_.fetch_sub(1, std::memory_order_release);
        }
    }
    const int threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_;
    const std::function<void(int, int)> * fn_ = nullptr;
    std::exception_ptr error_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int> pending_{0}, sleepers_{0};
    std::atomic<bool> stopping_{false};
    int count_ = 0;
};
}

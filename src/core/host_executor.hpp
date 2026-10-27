#pragma once
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tqllm {
// One caller; workers remain alive across all projections in a generation.
class HostExecutor {
public:
    explicit HostExecutor(int threads) {
        if (threads < 1 || threads > 64) throw std::invalid_argument("host executor thread count");
        try {
            for (int i = 0; i < threads; ++i) workers_.emplace_back([this, i] { run(i); });
        } catch (...) { stop(); throw; }
    }
    ~HostExecutor() { stop(); }
    HostExecutor(const HostExecutor &) = delete;
    HostExecutor & operator=(const HostExecutor &) = delete;
    void rows(int count, const std::function<void(int, int)> & fn) {
        std::unique_lock<std::mutex> lock(mutex_);
        count_ = count; fn_ = fn; error_ = nullptr; pending_ = int(workers_.size()); ++epoch_;
        ready_.notify_all();
        done_.wait(lock, [this] { return pending_ == 0; });
        fn_ = {};
        if (error_) std::rethrow_exception(error_);
    }
private:
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto & worker : workers_) if (worker.joinable()) worker.join();
    }
    void run(int id) {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            ready_.wait(lock, [&] { return stopping_ || seen != epoch_; });
            if (stopping_) return;
            seen = epoch_;
            const int n = int(workers_.size());
            const int begin = int(int64_t(count_) * id / n), end = int(int64_t(count_) * (id + 1) / n);
            lock.unlock();
            std::exception_ptr error;
            try { fn_(begin, end); } catch (...) { error = std::current_exception(); }
            lock.lock();
            if (error && !error_) error_ = error;
            if (--pending_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::function<void(int, int)> fn_;
    std::exception_ptr error_;
    uint64_t epoch_ = 0;
    int count_ = 0, pending_ = 0;
    bool stopping_ = false;
};
}

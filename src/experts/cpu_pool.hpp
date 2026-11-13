#pragma once
// A fixed set of worker threads that run one batch of independent tasks at a time: run(n, f) calls f(0) .. f(n - 1) across
// the workers and the caller, and returns when all are done. For the CPU expert path (cpu_kquant), where a decode layer's
// missed experts are split into row chunks. Exceptions in a task are rethrown by run().
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tqllm {
class CpuPool {
public:
    explicit CpuPool(int threads) {
        for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { loop(); });
    }
    ~CpuPool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; ++epoch_; }
        cv_.notify_all();
        for (auto & t : workers_) t.join();
    }
    CpuPool(const CpuPool &) = delete;
    CpuPool & operator=(const CpuPool &) = delete;
    int threads() const { return int(workers_.size()) + 1; }
    void run(int n, const std::function<void(int)> & f) {
        if (n <= 0) return;
        {
            std::lock_guard<std::mutex> l(m_);
            f_ = &f; n_ = n; next_.store(0); left_ = int(workers_.size()); err_ = nullptr; ++epoch_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [this] { return left_ == 0; });
        f_ = nullptr;
        if (err_) std::rethrow_exception(err_);
    }

private:
    void work() {
        for (int i; (i = next_.fetch_add(1)) < n_;) {
            try { (*f_)(i); } catch (...) { std::lock_guard<std::mutex> l(m_); if (!err_) err_ = std::current_exception(); }
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return epoch_ != seen; });
                seen = epoch_;
                if (stop_) return;
            }
            work();
            { std::lock_guard<std::mutex> l(m_); --left_; }
            done_.notify_one();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)> * f_ = nullptr;
    int n_ = 0, left_ = 0;
    std::atomic<int> next_{0};
    uint64_t epoch_ = 0;
    bool stop_ = false;
    std::exception_ptr err_;
};
}  // namespace tqllm

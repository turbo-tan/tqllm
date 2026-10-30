#pragma once
#include <condition_variable>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace tqllm {
// Empty futures mark consumed records. Polling bounds delay without blocking on the first read.
template<class T> size_t next_ready_expert(const std::vector<std::shared_future<T>> & pending) {
    for (;;) {
        size_t first=pending.size();
        for(size_t i=0;i<pending.size();++i)if(pending[i].valid()) {
            if(first==pending.size())first=i;
            const auto status=pending[i].wait_for(std::chrono::microseconds(0));
            if(status==std::future_status::ready || status==std::future_status::deferred)return i;
        }
        if(first==pending.size())throw std::logic_error("expert reads: no pending record");
        pending[first].wait_for(std::chrono::microseconds(50));
    }
}
class ExpertReadPool {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::function<void()>> jobs_;
    bool stop_ = false;
    std::vector<std::thread> workers_;
    void close() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_=true; }
        ready_.notify_all();
        for(auto & w:workers_)if(w.joinable())w.join();
    }
public:
    explicit ExpertReadPool(unsigned count) {
        if(!count || count>8)throw std::invalid_argument("expert readers: expected1..8 workers");
        try {
            for(unsigned i=0;i<count;++i)workers_.emplace_back([this] {
                for(;;) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait(lock,[this]{return stop_ || !jobs_.empty();});
                        if(stop_ && jobs_.empty())return;
                        job=std::move(jobs_.front()); jobs_.pop();
                    }
                    job();
                }
            });
        } catch(...) { close(); throw; }
    }
    ~ExpertReadPool(){close();}
    ExpertReadPool(const ExpertReadPool &)=delete;
    ExpertReadPool & operator=(const ExpertReadPool &)=delete;
    template<class F> auto submit(F f) -> std::future<std::invoke_result_t<F>> {
        using R=std::invoke_result_t<F>;
        auto job=std::make_shared<std::packaged_task<R()>>(std::move(f));auto result=job->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(stop_)throw std::runtime_error("expert readers: pool is closed");
            jobs_.emplace([job]{(*job)();});
        }
        ready_.notify_one();return result;
    }
};
} // namespace tqllm

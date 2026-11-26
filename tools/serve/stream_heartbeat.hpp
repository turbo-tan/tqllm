#pragma once
// Heartbeat for an SSE answer: whenever nothing went out for `every`, `beat()` runs (under the stream's write lock), for
// the whole request - not only until the first token. Agent clients drop a stream that sends no event for a while (Hermes
// on the Responses API: 12 s under 10K prompt tokens), and a stream is quiet during a long or queued prefill and while
// an answer with tools holds its text back until the call is parsed. beat() should send a real event of the stream's API
// (SSE comments are not events to an SDK). No engine and no CUDA here (unit test: tests/api_adapters.cpp).
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace tqserve {

class StreamHeartbeat {
public:
    using Clock = std::chrono::steady_clock;
    StreamHeartbeat(std::chrono::milliseconds every, std::function<bool()> beat)
        : every_(every), beat_(std::move(beat)), last_(Clock::now()), thread_([this] { run(); }) {}
    ~StreamHeartbeat() { stop(); }
    StreamHeartbeat(const StreamHeartbeat &) = delete;
    StreamHeartbeat & operator=(const StreamHeartbeat &) = delete;

    // the stream's write lock: hold it around every send (and the adapter state it touches); beats take it too
    std::mutex & lock() { return write_mx_; }
    // call after each send, with lock() held
    void wrote() { last_ = Clock::now(); }
    void stop() {
        { std::lock_guard<std::mutex> lk(cv_mx_); stop_ = true; }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        const auto tick = std::min<Clock::duration>(every_ / 4, std::chrono::seconds(1));
        std::unique_lock<std::mutex> lk(cv_mx_);
        while (!cv_.wait_for(lk, tick, [&] { return stop_; })) {
            std::lock_guard<std::mutex> w(write_mx_);
            if (Clock::now() - last_ < every_) continue;
            if (!beat_()) return;  // the client went away
            last_ = Clock::now();
        }
    }
    std::chrono::milliseconds every_;
    std::function<bool()> beat_;
    std::mutex write_mx_, cv_mx_;
    std::condition_variable cv_;
    Clock::time_point last_;  // guarded by write_mx_
    bool stop_ = false;
    std::thread thread_;
};

} // namespace tqserve

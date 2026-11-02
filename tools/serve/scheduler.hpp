#pragma once
// Multi-slot request scheduler (tqserve --np N): connection threads enqueue requests and wait; one engine thread
// owns every model call, admits queued requests into free slots and round-robins one draft + verify step per slot.
#include "session.hpp"
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace tqserve {

class Scheduler {
public:
    Scheduler(const tqllm::Tokenizer & tokenizer, tqllm::Model & model, uint32_t slot_context)
        : tokenizer_(tokenizer), model_(model), context_(slot_context), active_(size_t(model.slots())), used_(size_t(model.slots()), 0),
          limit_(model.slots()) {
        thread_ = std::thread([this] { run(); });
    }
    ~Scheduler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

    // Blocks the calling (connection) thread until the request has been generated.
    DecodeResult submit(const ChatRequest & req) {
        auto job = std::make_shared<Job>();
        job->req = req;
        job->prompt = prepare(tokenizer_, req, context_, job->result, job->count, &job->images);
        if (!job->result.error.empty()) return job->result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                job->result.error = "server shutting down";
                return job->result;
            }
            queue_.push_back(job);
        }
        wake_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return job->done; });
        return job->result;
    }

    // Live parallelism: at most `n` (1 .. allocated) requests run at once; running ones finish, new ones queue.
    int set_limit(int n) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            limit_ = std::clamp(n, 1, int(active_.size()));
        }
        wake_.notify_all();
        return limit_;
    }
    std::string status_json() {
        std::lock_guard<std::mutex> lock(mutex_);
        int busy = 0;
        for (const auto & j : active_) busy += j != nullptr;
        return "{\"allocated\":" + std::to_string(active_.size()) + ",\"slots\":" + std::to_string(limit_) +
               ",\"busy\":" + std::to_string(busy) + ",\"queued\":" + std::to_string(queue_.size()) + "}";
    }

    // Streaming submit: `on_tokens` (connection thread) gets the slot's ids after each engine step; returning false
    // cancels the request (the slot is finished at the next step).
    DecodeResult submit_stream(const ChatRequest & req, const std::function<bool(const std::vector<uint32_t> &)> & on_tokens) {
        auto job = std::make_shared<Job>();
        job->req = req;
        job->stream = true;
        job->prompt = prepare(tokenizer_, req, context_, job->result, job->count, &job->images);
        if (!job->result.error.empty()) return job->result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                job->result.error = "server shutting down";
                return job->result;
            }
            queue_.push_back(job);
        }
        wake_.notify_all();
        uint64_t seen = 0;
        for (;;) {
            std::vector<uint32_t> ids;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                done_.wait(lock, [&] { return job->done || job->version != seen; });
                if (job->done) return job->result;
                seen = job->version;
                ids = job->partial;
            }
            if (!on_tokens(ids)) {
                std::lock_guard<std::mutex> lock(mutex_);
                job->cancel = true;
                if (job->req.observation) job->req.observation->finish("cancelled");
            }
        }
    }

private:
    struct Job {
        ChatRequest req;
        std::vector<uint32_t> prompt;
        PreparedImages images;
        uint32_t count = 0;
        DecodeResult result;
        bool done = false;
        bool stream = false, cancel = false;  // stream: partial / version updated after every step
        std::vector<uint32_t> partial;
        uint64_t version = 0;
        RepeatGuard repeat;
    };

    // after a step: the slot's ids go to the stream and through the repeat guard (tripped -> the request ends)
    bool step_done(Job & job, int slot) {
        const auto & ids = model_.spec_tokens(slot);  // read now; publish() copies for the stream thread
        if (job.req.observation) job.req.observation->tokens(static_cast<uint32_t>(ids.size()));
        const bool tripped = job.repeat.check(ids);
        if (job.stream) publish(job, ids);
        return tripped;
    }

    void publish(Job & job, std::vector<uint32_t> ids) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job.partial = std::move(ids);
            ++job.version;
        }
        done_.notify_all();
    }

    void finish_slot(int s) {  // spec_finish + result for the job in slot s
        auto & job = active_[size_t(s)];
        model_.select_slot(s);
        uint64_t st[5] = {};
        finish(tokenizer_, job->req, model_.spec_finish(st), job->result);
        set_timings(job->result, st);
        complete(job);
    }

    void complete(std::shared_ptr<Job> & job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job->done = true;
        }
        done_.notify_all();
        job.reset();
    }

    void run() {
        for (;;) {
            std::vector<std::shared_ptr<Job>> admitted;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                auto idle = [&] {
                    for (const auto & j : active_) if (j) return false;
                    return true;
                };
                wake_.wait(lock, [&] { return stop_ || !queue_.empty() || !idle(); });
                if (stop_) {
                    for (auto & j : queue_) {
                        j->result.error = "server shutting down";
                        j->done = true;
                    }
                    for (auto & j : active_)
                        if (j) {
                            j->result.error = "server shutting down";
                            j->done = true;
                        }
                    done_.notify_all();
                    return;
                }
                int busy = 0;
                for (const auto & j : active_) busy += j != nullptr;
                // first come first served; each request takes the free slot that reuses most of its prompt (an agent's
                // next turn goes back to the slot holding its conversation), else the free slot reused least recently
                while (!queue_.empty() && busy < limit_) {
                    int best = -1; uint32_t reuse = 0;
                    for (size_t s = 0; s < active_.size(); ++s) {
                        if (active_[s]) continue;
                        const uint32_t r = queue_.front()->images.spans.empty() ? model_.prefix_reuse(int(s), queue_.front()->prompt) : 0;
                        if (best < 0 || r > reuse || (r == reuse && used_[s] < used_[size_t(best)])) { best = int(s); reuse = r; }
                    }
                    if (best < 0) break;
                    ++busy;
                    used_[size_t(best)] = ++clock_;
                    active_[size_t(best)] = queue_.front();
                    queue_.pop_front();
                    admitted.push_back(active_[size_t(best)]);
                }
            }
            for (size_t s = 0; s < active_.size(); ++s) {  // prefill newly admitted requests
                auto & job = active_[s];
                if (!job || std::find(admitted.begin(), admitted.end(), job) == admitted.end()) continue;
                try {
                    model_.select_slot(int(s));
                    model_.set_sampling(job->req.temperature, job->req.top_p, job->req.top_k, job->req.seed, job->req.presence_penalty);
                    model_.set_penalties(job->req.repeat_penalty, job->req.frequency_penalty);
                    model_.set_ignore_eos(job->req.ignore_eos);
                    if (!job->images.spans.empty()) model_.set_prompt_images(job->images.spans);
                    if (job->req.observation) job->req.observation->prefill(job->result.prompt_tokens);
                    { tqllm::PrefillObservation progress(observe_prefill(job->req.observation));
                    model_.spec_begin(job->prompt, job->count, draft_length()); }
                    if (job->req.observation) job->req.observation->decoding();
                } catch (const std::exception & e) {
                    job->result.error = std::string("generation failed: ") + e.what();  // tqserve kFailed: HTTP 500
                    complete(job);
                }
            }
            for (size_t s = 0; s < active_.size(); ++s) {  // cancelled streams (client gone)
                bool cancel = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    cancel = active_[s] && active_[s]->cancel;
                }
                if (cancel) finish_slot(int(s));
            }
            std::vector<int> ids;  // one decode step for every active slot
            for (size_t s = 0; s < active_.size(); ++s)
                if (active_[s]) ids.push_back(int(s));
            if (ids.size() == 1) {  // single sequence: the unbatched step (identical to --np 1)
                auto & job = active_[size_t(ids[0])];
                try {
                    model_.select_slot(ids[0]);
                    if (!model_.spec_advance() || step_done(*job, ids[0])) finish_slot(ids[0]);
                } catch (const std::exception & e) {
                    job->result.error = std::string("generation failed: ") + e.what();  // tqserve kFailed: HTTP 500
                    complete(job);
                }
            } else if (!ids.empty()) {  // batched: one verify pass over all slots
                std::vector<char> done;
                try {
                    model_.spec_step(ids, done);
                    for (size_t i = 0; i < ids.size(); ++i) {
                        auto & job = active_[size_t(ids[i])];
                        if (done[i] || step_done(*job, ids[i])) finish_slot(ids[i]);
                    }
                } catch (const std::exception & e) {
                    for (int id : ids) {
                        auto & job = active_[size_t(id)];
                        if (!job) continue;
                        job->result.error = std::string("generation failed: ") + e.what();  // tqserve kFailed: HTTP 500
                        complete(job);
                    }
                }
            }
        }
    }

    const tqllm::Tokenizer & tokenizer_;
    tqllm::Model & model_;
    uint32_t context_;
    std::vector<std::shared_ptr<Job>> active_;  // engine thread only
    std::vector<uint64_t> used_;  // admission clock per slot (least recently used free slot first)
    uint64_t clock_ = 0;
    std::deque<std::shared_ptr<Job>> queue_;
    int limit_ = 1;  // admitted at once (POST /admin/slots)
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace tqserve

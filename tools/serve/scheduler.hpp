#pragma once
// Multi-slot request scheduler (tqserve --np N): connection threads enqueue requests and wait; one engine thread
// owns every model call, admits queued requests into free slots and round-robins one draft + verify step per slot.
#include "admission.hpp"
#include "cuda_health.hpp"
#include "router.hpp"
#include "session.hpp"
#include "tqllm/model.hpp"
#include "tqllm/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
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
        slot_text_.resize(active_.size());
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
        job->start = std::chrono::steady_clock::now();
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
    // What the front door's router needs from this box (tqserve --peer, POST /admin/route): slots, load, measured rates and
    // the prompt tokens a free slot already holds for this rendered prompt (its last conversation's prefix).
    BoxStatus route_status(const std::string & prompt_text) {
        std::lock_guard<std::mutex> lock(mutex_);
        BoxStatus b;
        b.ok = true;
        for (size_t s = 0; s < active_.size(); ++s) {
            if (active_[s]) { ++b.busy; continue; }
            b.reuse = std::max(b.reuse, reuse_tokens(slot_text_[s], prompt_text, cpt_.set ? cpt_.v : 4.0));
        }
        b.free = std::max(0, limit_ - b.busy);
        b.queued = int(queue_.size());
        b.prefill_tps = prefill_tps_.v; b.decode_tps = decode_tps_.v; b.hold_s = hold_s_.v;
        return b;
    }
    double chars_per_token() {
        std::lock_guard<std::mutex> lock(mutex_);
        return cpt_.set ? cpt_.v : 4.0;
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
        job->start = std::chrono::steady_clock::now();
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
        std::chrono::steady_clock::time_point start;
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
        {  // what the router reads: this slot now holds the conversation; measured rates for the cost model
            std::lock_guard<std::mutex> lock(mutex_);
            const auto & r = job->result;
            slot_text_[size_t(s)] = job->req.prompt_text + r.reasoning + r.text;
            if (r.prompt_tokens > 0) cpt_.add(double(job->req.prompt_text.size()) / r.prompt_tokens);
            if (r.prompt_tokens >= 512 && r.prefill_ms > 0) prefill_tps_.add(r.prompt_tokens * 1000.0 / r.prefill_ms);
            int running = 0;
            for (const auto & j : active_) running += j != nullptr;
            if (r.completion_tokens >= 32 && r.decode_ms > 0)  // back to one sequence alone (router.hpp kBatchSlowdown)
                decode_tps_.add(r.completion_tokens * 1000.0 / r.decode_ms * (1.0 + kBatchSlowdown * (running - 1)));
            hold_s_.add(std::chrono::duration<double>(std::chrono::steady_clock::now() - job->start).count());
        }
        complete(job);
    }

    // the engine forgets a failed request's slot (Engine::spec_abort): otherwise it still counts it as running, refuses
    // the next request for memory while the scheduler sees nothing to wait for, and that refusal becomes another 500
    void abort_slot(int s) { try { model_.spec_abort(s); } catch (...) {} }
    // The CUDA context is lost (cuda_health.hpp: tqserve exits shortly): every running and queued request gets its error
    // now, none waits for the exit
    void fail_everything(const std::string & why) {
        std::deque<std::shared_ptr<Job>> queued;
        { std::lock_guard<std::mutex> lock(mutex_); queued.swap(queue_); }
        const std::string error = "generation failed: the CUDA context is lost (" + why + "); tqserve is restarting";
        for (auto & job : queued) { job->result.error = error; complete(job); }
        for (size_t s = 0; s < active_.size(); ++s)
            if (active_[s]) { active_[s]->result.error = error; abort_slot(int(s)); complete(active_[s]); }
    }
    void complete(std::shared_ptr<Job> & job) {
        if (!job->result.error.empty()) std::fprintf(stderr, "tqserve: request failed (HTTP 500): %s\n", job->result.error.c_str());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job->done = true;
            ++finished_;
            job.reset();  // under the lock: status_json / route_status read active_ from connection threads
        }
        done_.notify_all();
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
                while (!queue_.empty() && busy < limit_ && admission_open(finished_, wait_until_)) {
                    int best = -1; uint32_t reuse = 0;
                    for (size_t s = 0; s < active_.size(); ++s) {
                        if (active_[s]) continue;
                        const uint32_t r = queue_.front()->images.spans.empty() ? model_.prefix_reuse(int(s), queue_.front()->prompt) : 0;
                        if (best < 0 || r > reuse || (r == reuse && used_[s] < used_[size_t(best)])) { best = int(s); reuse = r; }
                    }
                    if (best < 0) break;
                    ++busy;
                    used_[size_t(best)] = ++clock_;
                    slot_text_[size_t(best)].clear();  // its K/V is about to be overwritten
                    active_[size_t(best)] = queue_.front();
                    queue_.pop_front();
                    admitted.push_back(active_[size_t(best)]);
                }
            }
            if (!admitted.empty()) {  // this thread runs the engine: clear a pending CUDA error; a lost context fails them all
                if (const std::string lost = check_cuda_context(); !lost.empty()) fail_everything(lost);
#if TQLLM_WITH_CUDA
                {   // TQLLM_DEBUG=cuda_fault:N: the N-th admitted request faults the GPU (the lost-context test)
                    static const int fault_at = tqllm::debug_int("cuda_fault");
                    static int admits = 0;
                    if (fault_at > 0 && admits < fault_at && (admits += int(admitted.size())) >= fault_at) tqllm_cuda::debug_device_fault();
                }
#endif
            }
            for (size_t s = 0; s < active_.size(); ++s) {  // prefill newly admitted requests
                auto & job = active_[s];
                if (!job || std::find(admitted.begin(), admitted.end(), job) == admitted.end()) continue;
                try {
                    model_.select_slot(int(s));
                    model_.set_sampling(job->req.temperature, job->req.top_p, job->req.top_k, job->req.seed, job->req.presence_penalty);
                    model_.set_penalties(job->req.repeat_penalty, job->req.frequency_penalty);
                    model_.set_ignore_eos(job->req.ignore_eos);
                    model_.set_continuation(job->req.continuation);
                    if (!job->images.spans.empty()) model_.set_prompt_images(job->images.spans);
                    if (job->req.observation) job->req.observation->prefill(job->result.prompt_tokens);
                    struct HookReset { tqllm::Model & m; ~HookReset() { m.set_first_token_hook(nullptr); } } hook_reset{model_};  // also on a throw
                    if (job->stream) model_.set_first_token_hook([this, &job, s] { publish(*job, model_.spec_tokens(int(s))); });
                    { tqllm::PrefillObservation progress(observe_prefill(job->req.observation));
                    model_.spec_begin(job->prompt, job->count, draft_length()); }
                    if (job->req.observation) job->req.observation->decoding();
                    // the prefill picked the first token: stream it now, not after every admitted prompt is prefilled
                    // and the first batched step has run (8 prompts at once: each waited for all eight prefills)
                    if (job->stream) publish(*job, model_.spec_tokens(int(s)));
                } catch (const std::exception & e) {
                    int others = 0;
                    for (size_t o = 0; o < active_.size(); ++o) others += o != s && active_[o] != nullptr;
                    if (on_prefill_error(e.what(), others) == OnOom::Requeue) {  // queue, not stop: retried after one finishes
                        std::fprintf(stderr, "tqserve: not enough memory for a %zu-token prompt beside %d running: queued until one finishes\n",
                                     job->prompt.size(), others);
                        std::lock_guard<std::mutex> lock(mutex_);
                        wait_until_ = finished_ + 1;
                        queue_.push_front(job);
                        job.reset();
                        continue;
                    }
                    job->result.error = std::string("generation failed: ") + e.what();  // tqserve kFailed: HTTP 500
                    abort_slot(int(s));
                    complete(job);
                    if (const std::string lost = check_cuda_context(); !lost.empty()) fail_everything(lost);
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
                    abort_slot(ids[0]);
                    complete(job);
                    if (const std::string lost = check_cuda_context(); !lost.empty()) fail_everything(lost);
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
                        abort_slot(id);
                        complete(job);
                    }
                    if (const std::string lost = check_cuda_context(); !lost.empty()) fail_everything(lost);
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
    unsigned long long finished_ = 0, wait_until_ = 0;  // after an out-of-memory requeue, admission waits for one finish
    std::deque<std::shared_ptr<Job>> queue_;
    std::vector<std::string> slot_text_;  // per slot: the conversation its K/V holds (rendered prompt + answer), for routing
    Ema cpt_, prefill_tps_, decode_tps_, hold_s_;  // chars per token, prompt tok/s, one-sequence decode tok/s, request seconds
    int limit_ = 1;  // admitted at once (POST /admin/slots)
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    bool stop_ = false;
    std::thread thread_;
};

} // namespace tqserve

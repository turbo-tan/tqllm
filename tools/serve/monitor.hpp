#pragma once
// Read-only serving telemetry. Writers publish counters; readers never touch an engine.
#include "json_out.hpp"
#include "prefill_observer.hpp"
#include "router.hpp"
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tqserve {
class Monitor {
    using Clock = std::chrono::steady_clock;
    struct Entry {
        uint64_t id;
        std::string phase = "queued";
        std::optional<uint32_t> prompt, output, context, prefill_done;
        uint32_t cached = 0;
        Clock::time_point prefill_start = Clock::now();
        uint32_t previous_output = 0, passes = 1;
        double previous_prefill = 0, previous_decode = 0;
        uint64_t previous_drafted = 0, previous_accepted = 0;
        std::optional<double> first_ms, prefill_ms, decode_ms;
        uint64_t drafted = 0, accepted = 0;
        Clock::time_point start = Clock::now(), decode = start, changed = start;
    };
    struct Shared { std::mutex mutex; std::deque<Entry> entries; uint64_t next = 0; };
public:
    struct Info { std::string model, build, host, device; uint32_t context = 0, slots = 1; };
    struct Hardware {
        std::optional<double> cpu_percent, ram_used_bytes, ram_total_bytes;
        std::optional<double> gpu_percent, gpu_used_bytes, gpu_total_bytes, gpu_celsius, gpu_watts;
        std::string device_name;
    };
    struct ClusterDevice {
        int rank;
        std::string bus;
        std::optional<uint32_t> layer_start, layer_end;
        std::string name;
        std::optional<double> used_bytes, total_bytes;
    };
    void cluster(std::vector<std::string> peers, std::vector<ClusterDevice> devices) {
        std::lock_guard<std::mutex> lock(data_->mutex);
        peers_ = std::move(peers);
        devices_ = std::move(devices);
    }
    uint64_t route(const std::vector<BoxStatus> & boxes, const std::vector<bool> & answered,
                   RoutePick pick, double link_bytes_per_second) {
        std::lock_guard<std::mutex> lock(data_->mutex);
        boxes_ = boxes;
        answered_ = answered;
        route_time_ = Clock::now();
        link_rate_ = link_bytes_per_second;
        if (routes_.size() == 24) routes_.pop_front();
        routes_.push_back({++next_route_, pick, boxes[size_t(pick.box)].reuse, pick.box == 0 ? "local" : "forwarding"});
        return next_route_;
    }
    void route_forwarded(uint64_t id, bool forwarded) {
        std::lock_guard<std::mutex> lock(data_->mutex);
        for (auto & entry : routes_) if (entry.id == id) {
            entry.outcome = forwarded ? "forwarded" : "local fallback";
            break;
        }
    }
    class Request {
    public:
        Request(std::shared_ptr<Shared> data, uint64_t id) : data_(std::move(data)), id_(id) {}
        ~Request() { finish("failed"); }
        Request(const Request &) = delete;
        Request & operator=(const Request &) = delete;
        void prefill(uint32_t total) { update([&](Entry & e) {
            if (e.decode_ms) {
                e.previous_output = e.output.value_or(0); e.previous_prefill = e.prefill_ms.value_or(0);
                e.previous_decode = *e.decode_ms; e.previous_drafted = e.drafted; e.previous_accepted = e.accepted;
                e.decode_ms.reset(); e.prefill_ms.reset(); ++e.passes;
            }
            e.phase = "prefill"; e.prompt = total; e.context = total;
            e.prefill_start=Clock::now(); e.prefill_done.reset(); e.cached=0;
        }); }
        void progress(uint32_t done, uint32_t total, uint32_t cached=0) { update([&](Entry & e) {
            if (e.phase!="prefill" || !e.prompt || total!=*e.prompt || done>total || cached>done) return;
            e.prefill_done=std::max(e.prefill_done.value_or(0),done); e.cached=cached;
        }); }
        void decoding() { update([](Entry & e) {
            e.phase = "decode"; e.decode = Clock::now();
            e.prefill_ms=e.previous_prefill+milliseconds(e.decode-e.prefill_start);
        }); }
        void tokens(uint32_t count) {
            update([&](Entry & e) {
                if (count && !e.first_ms) e.first_ms = milliseconds(Clock::now() - e.start);
                e.output = e.previous_output + count;
                if (e.prompt) e.context = *e.prompt + count;
            });
        }
        void timings(uint32_t prompt, uint32_t output, double prefill, double decode, uint64_t drafted, uint64_t accepted) {
            update([&](Entry & e) { e.prompt = prompt; e.context = prompt + output; e.output = e.previous_output + output;
                e.prefill_ms = e.previous_prefill + prefill; e.decode_ms = e.previous_decode + decode;
                e.drafted = e.previous_drafted + drafted; e.accepted = e.previous_accepted + accepted; });
        }
        void finish(const std::string & phase = "complete") {
            update([&](Entry & e) { e.phase = phase; });
        }
    private:
        template<class F> void update(F && fn) {
            std::lock_guard<std::mutex> lock(data_->mutex);
            for (auto & e : data_->entries) if (e.id == id_) {
                if (!terminal(e.phase)) { fn(e); e.changed = Clock::now(); }
                break;
            }
        }
        std::shared_ptr<Shared> data_;
        uint64_t id_;
    };
    explicit Monitor(Info info) : info_(std::move(info)) {}
    // What the chat panel can offer: thinking default on/off, the reasoning_effort values the chat template understands
    // (empty: only on/off), and the server's default effort. Set once the chat template is known.
    void reasoning(bool default_on, std::vector<std::string> efforts, std::string default_effort) {
        std::lock_guard<std::mutex> lock(data_->mutex);
        reasoning_on_ = default_on; efforts_ = std::move(efforts); default_effort_ = std::move(default_effort);
    }
    std::shared_ptr<Request> begin() {
        std::lock_guard<std::mutex> lock(data_->mutex);
        // Retain 32 completed requests; never evict an active request.
        size_t done = 0;
        for (const auto & e : data_->entries) done += terminal(e.phase);
        for (auto it = data_->entries.begin(); it != data_->entries.end() && done >= 32;)
            if (terminal(it->phase)) { it = data_->entries.erase(it); --done; } else ++it;
        data_->entries.push_back(Entry{++data_->next});
        return std::make_shared<Request>(data_, data_->next);
    }
    void hardware(Hardware value) {
        std::lock_guard<std::mutex> lock(data_->mutex);
        hardware_ = std::move(value); hardware_time_ = Clock::now();
    }
    std::string snapshot() const {
        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(data_->mutex);
        std::string out = "{\"schema_version\":1,\"instance_id\":";
        jstr(out, instance_);
        out += ",\"uptime_ms\":" + std::to_string(milliseconds(now - started_)) + ",\"engine\":{\"model\":";
        jstr(out, info_.model); out += ",\"build\":"; jstr(out, info_.build);
        out += ",\"host\":"; jstr(out, info_.host); out += ",\"device_id\":"; jstr(out, info_.device);
        out += ",\"context_per_slot\":" + std::to_string(info_.context) + ",\"slots\":" + std::to_string(info_.slots);
        out += std::string(",\"reasoning\":{\"default_on\":") + (reasoning_on_ ? "true" : "false") + ",\"efforts\":[";
        for (size_t i = 0; i < efforts_.size(); ++i) { if (i) out += ','; jstr(out, efforts_[i]); }
        out += "],\"default_effort\":"; jstr(out, default_effort_); out += "}},\"requests\":[";
        bool first = true;
        for (const auto & e : data_->entries) {
            if (!first) out += ',';
            first = false;
            out += "{\"id\":" + std::to_string(e.id) + ",\"phase\":"; jstr(out, e.phase);
            out += ",\"elapsed_ms\":" + std::to_string(milliseconds((terminal(e.phase) ? e.changed : now) - e.start));
            field(out, "prompt_tokens", e.prompt); field(out, "output_tokens", e.output); field(out, "context_tokens", e.context);
            field(out, "first_output_ms", e.first_ms); field(out, "prefill_ms", e.prefill_ms);
            field(out, "decode_ms", e.decode_ms);
            field(out, "prefill_processed_tokens", e.prefill_done);
            out += ",\"prefill_cached_tokens\":" + std::to_string(e.cached);
            const double prefill_time=e.phase=="prefill" ? milliseconds(now-e.prefill_start) : e.prefill_ms.value_or(0)-e.previous_prefill;
            std::optional<double> prefill_rate;
            if (prefill_time>0 && e.prompt) {
                const auto processed=e.phase=="prefill" ? e.prefill_done : e.prompt;
                if(processed && *processed>e.cached) prefill_rate=(*processed-e.cached)*1000.0/prefill_time;
            }
            field(out, "prefill_tokens_per_second", prefill_rate);
            const auto n = e.output.value_or(0);
            const double ms = e.decode_ms.value_or(e.previous_decode + milliseconds(now - e.decode));
            std::optional<double> rate;
            if (ms > 0 && n > e.passes && (e.phase == "decode" || e.decode_ms)) rate = (n - e.passes) * 1000.0 / ms;
            field(out, "decode_tokens_per_second", rate);
            out += ",\"drafted\":" + std::to_string(e.drafted) + ",\"accepted\":" + std::to_string(e.accepted) + "}";
        }
        out += "],\"hardware\":{\"sample_age_ms\":";
        out += hardware_time_ ? std::to_string(milliseconds(now - *hardware_time_)) : "null";
        field(out, "cpu_percent", hardware_.cpu_percent); field(out, "ram_used_bytes", hardware_.ram_used_bytes);
        field(out, "ram_total_bytes", hardware_.ram_total_bytes); field(out, "gpu_percent", hardware_.gpu_percent);
        field(out, "gpu_used_bytes", hardware_.gpu_used_bytes); field(out, "gpu_total_bytes", hardware_.gpu_total_bytes);
        field(out, "gpu_celsius", hardware_.gpu_celsius); field(out, "gpu_watts", hardware_.gpu_watts);
        out += ",\"device_name\":"; jstr(out, hardware_.device_name);
        out += "},\"cluster\":";
        append_cluster(out, now);
        return out + ",\"capabilities\":{\"live_tokens\":true,\"prefill_progress\":true,\"expert_cache\":false,\"cancel\":false}}";
    }
private:
    struct RouteEntry {
        uint64_t id;
        RoutePick pick;
        double reuse;
        std::string outcome;
    };
    void append_cluster(std::string & out, Clock::time_point now) const {
        out += "{\"nodes\":[";
        for (size_t i = 0; i <= peers_.size(); ++i) {
            if (i) out += ',';
            out += "{\"id\":" + std::to_string(i) + ",\"label\":";
            jstr(out, i ? peers_[i - 1] : info_.host);
            out += ",\"model\":";
            if (i) out += "null"; else jstr(out, info_.model);
            out += ",\"status\":";
            jstr(out, !i ? "local" : i >= boxes_.size() ? "unsampled" :
                 !boxes_[i].ok ? "unreachable" : answered_[i] ? "reachable" : "last known");
            if (i < boxes_.size()) {
                const auto & b = boxes_[i];
                if (b.ok) {
                    field(out, "busy", std::optional<int>(b.busy));
                    field(out, "queued", std::optional<int>(b.queued));
                    field(out, "prefill", positive(b.prefill_tps));
                    field(out, "decode", positive(b.decode_tps));
                }
            } else if (!i) {
                int busy = 0, queued = 0;
                for (const auto & e : data_->entries) if (!terminal(e.phase)) {
                    if (e.phase == "queued") ++queued; else ++busy;
                }
                field(out, "busy", std::optional<int>(busy));
                field(out, "queued", std::optional<int>(queued));
            }
            out += '}';
        }
        out += "],\"devices\":[";
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (i) out += ',';
            const auto & d = devices_[i];
            out += "{\"rank\":" + std::to_string(d.rank) + ",\"bus\":";
            const bool primary = !d.bus.empty() && d.bus == info_.device;
            jstr(out, d.bus); out += ",\"name\":"; jstr(out, primary ? hardware_.device_name : d.name);
            field(out, "layer_start", d.layer_start); field(out, "layer_end", d.layer_end);
            const bool fresh = hardware_time_ && now - *hardware_time_ < std::chrono::seconds(10);
            field(out, "used_bytes", primary ? (fresh ? hardware_.gpu_used_bytes : std::nullopt) : d.used_bytes);
            field(out, "total_bytes", primary ? (fresh ? hardware_.gpu_total_bytes : std::nullopt) : d.total_bytes);
            out += '}';
        }
        out += "],\"routes\":[";
        for (auto it = routes_.rbegin(); it != routes_.rend(); ++it) {
            if (it != routes_.rbegin()) out += ',';
            out += "{\"id\":" + std::to_string(it->id) + ",\"selected\":" + std::to_string(it->pick.box);
            field(out, "state_source", it->pick.owner < 0 ? std::nullopt : std::optional<int>(it->pick.owner));
            field(out, "reuse_tokens", positive(it->reuse));
            out += ",\"outcome\":"; jstr(out, it->outcome); out += '}';
        }
        out += ']';
        field(out, "sample_age_ms", route_time_ ? std::optional<double>(milliseconds(now - *route_time_)) : std::nullopt);
        field(out, "state_link_bytes_per_second", positive(link_rate_));
        out += '}';
    }
    static std::optional<double> positive(double v) { return v > 0 ? std::optional<double>(v) : std::nullopt; }
    static bool terminal(const std::string & s) { return s == "complete" || s == "failed" || s == "cancelled"; }
    template<class D> static double milliseconds(D d) { return std::chrono::duration<double, std::milli>(d).count(); }
    template<class T> static void field(std::string & out, const char * name, std::optional<T> value) {
        out += ",\"" + std::string(name) + "\":";
        if (value && std::isfinite(double(*value))) out += std::to_string(*value); else out += "null";
    }
    Info info_;
    bool reasoning_on_ = false;
    std::vector<std::string> efforts_;
    std::string default_effort_;
    const Clock::time_point started_ = Clock::now();
    const std::string instance_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::shared_ptr<Shared> data_ = std::make_shared<Shared>();
    Hardware hardware_;
    std::vector<std::string> peers_;
    std::vector<ClusterDevice> devices_;
    std::vector<BoxStatus> boxes_;
    std::vector<bool> answered_;
    std::deque<RouteEntry> routes_;
    std::optional<Clock::time_point> route_time_;
    double link_rate_ = 0;
    uint64_t next_route_ = 0;
    std::optional<Clock::time_point> hardware_time_;
};
inline tqllm::PrefillObserver observe_prefill(const std::shared_ptr<Monitor::Request> & ticket) {
    return ticket ? tqllm::PrefillObserver([ticket](uint32_t done,uint32_t total,uint32_t cached) { ticket->progress(done,total,cached); }) : tqllm::PrefillObserver{};
}
} // namespace tqserve

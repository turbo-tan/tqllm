#pragma once
// Read-only serving telemetry. Writers publish counters; readers never touch an engine.
#include "json_out.hpp"
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
        std::optional<uint32_t> prompt, output, context;
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
        }); }
        void decoding() { update([](Entry & e) { e.phase = "decode"; e.decode = Clock::now(); }); }
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
        out += ",\"context_per_slot\":" + std::to_string(info_.context) + ",\"slots\":" + std::to_string(info_.slots) + "},\"requests\":[";
        bool first = true;
        for (const auto & e : data_->entries) {
            if (!first) out += ',';
            first = false;
            out += "{\"id\":" + std::to_string(e.id) + ",\"phase\":"; jstr(out, e.phase);
            out += ",\"elapsed_ms\":" + std::to_string(milliseconds((terminal(e.phase) ? e.changed : now) - e.start));
            field(out, "prompt_tokens", e.prompt); field(out, "output_tokens", e.output); field(out, "context_tokens", e.context);
            field(out, "first_output_ms", e.first_ms); field(out, "prefill_ms", e.prefill_ms);
            field(out, "decode_ms", e.decode_ms);
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
        return out + "},\"capabilities\":{\"live_tokens\":true,\"prefill_progress\":false,\"expert_cache\":false,\"cancel\":false}}";
    }
private:
    static bool terminal(const std::string & s) { return s == "complete" || s == "failed" || s == "cancelled"; }
    template<class D> static double milliseconds(D d) { return std::chrono::duration<double, std::milli>(d).count(); }
    template<class T> static void field(std::string & out, const char * name, std::optional<T> value) {
        out += ",\"" + std::string(name) + "\":";
        if (value && std::isfinite(double(*value))) out += std::to_string(*value); else out += "null";
    }
    Info info_;
    const Clock::time_point started_ = Clock::now();
    const std::string instance_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::shared_ptr<Shared> data_ = std::make_shared<Shared>();
    Hardware hardware_;
    std::optional<Clock::time_point> hardware_time_;
};
} // namespace tqserve

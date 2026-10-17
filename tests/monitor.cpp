#include "serve/monitor_server.hpp"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <thread>
#include <cmath>

using tqserve::Monitor;
void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
tqllm::Json snapshot(Monitor & m) { return tqllm::parse_json(m.snapshot()); }
const tqllm::Json & latest(const tqllm::Json & j) { return j.get("requests")->arr.back(); }

#if defined(__linux__)
std::string get(uint16_t port, const std::string & path, const std::string & auth = {}, const std::string & method = "GET", const std::string & body = {},
                const std::string & origin = {}, const std::string & type = "application/json") {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    check(fd >= 0, "client socket");
    timeval timeout{3, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port); address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0, "client connect");
    const std::string request = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) + "\r\n" +
        (auth.empty() ? "" : "Authorization: Bearer " + auth + "\r\n") +
        (origin.empty() ? "" : "Origin: " + origin + "\r\n") +
        "Content-Type: " + type + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    check(send(fd, request.data(), request.size(), MSG_NOSIGNAL) == ssize_t(request.size()), "client send");
    std::string result; char buffer[8192]; ssize_t n;
    while ((n = recv(fd, buffer, sizeof(buffer), 0)) > 0) result.append(buffer, size_t(n));
    close(fd); return result;
}
void chat_proxy_gate() {
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    check(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0 && listen(listener, 1) == 0, "upstream bind");
    socklen_t size = sizeof(address); getsockname(listener, reinterpret_cast<sockaddr *>(&address), &size);
    const auto upstream_port = ntohs(address.sin_port);
    Monitor monitor({"fixture", "test", "localhost", "", 2048, 1});
    tqserve::MonitorServer server(monitor, "127.0.0.1", 0, "test-key", "", upstream_port);
    std::mutex mutex; std::condition_variable wake; bool admitted = false, release = false, forwarded = false;
    const std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"private prompt\"}],\"stream\":true}";
    std::thread upstream([&] {
        const int fd = accept(listener, nullptr, nullptr);
        tqserve::RequestReader reader(1024 * 1024); uint8_t data[4096];
        while (!reader.complete() && !reader.failed()) { const auto n = recv(fd, data, sizeof(data), 0); if (n <= 0) break; reader.append(data, size_t(n)); }
        const auto request = reader.request();
        { std::lock_guard<std::mutex> lock(mutex); forwarded = request.ok() && request.path == "/v1/chat/completions" && request.authorization == "Bearer test-key" && std::string(request.body.begin(), request.body.end()) == body; admitted = true; }
        wake.notify_all();
        { std::unique_lock<std::mutex> lock(mutex); wake.wait_for(lock, std::chrono::seconds(5), [&] { return release; }); }
        const std::string response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n";
        send(fd, response.data(), response.size(), MSG_NOSIGNAL); close(fd);
    });
    std::string streamed;
    std::thread chat([&] { streamed = get(server.port(), "/api/monitor/v1/chat/completions", "test-key", "POST", body); });
    { std::unique_lock<std::mutex> lock(mutex); check(wake.wait_for(lock, std::chrono::seconds(3), [&] { return admitted; }), "chat not forwarded"); }
    const auto start = std::chrono::steady_clock::now();
    const auto metrics = get(server.port(), "/api/monitor/v1/snapshot", "test-key");
    check(metrics.find("200 OK") != std::string::npos && std::chrono::steady_clock::now() - start < std::chrono::milliseconds(400), "chat blocks telemetry");
    check(metrics.find("private prompt") == std::string::npos, "chat body leaked into telemetry");
    check(get(server.port(), "/api/monitor/v1/chat/completions", "test-key", "POST", body).find("429") != std::string::npos, "unbounded chat concurrency");
    check(get(server.port(), "/api/monitor/v1/chat/completions", "", "POST", body).find("401") != std::string::npos, "unauthenticated chat");
    check(get(server.port(), "/api/monitor/v1/chat/completions", "test-key", "POST", body, "https://other.invalid").find("403") != std::string::npos, "cross-origin chat");
    check(get(server.port(), "/api/monitor/v1/chat/completions", "test-key", "POST", body, "", "text/plain").find("403") != std::string::npos, "simple-request CSRF");
    { std::lock_guard<std::mutex> lock(mutex); release = true; } wake.notify_all();
    upstream.join(); chat.join(); close(listener);
    check(forwarded && streamed.find("hello") != std::string::npos && streamed.find("[DONE]") != std::string::npos, "SSE/body/auth proxy contract");
}

#endif

int main(int argc, char ** argv) {
    try {
        Monitor m({"Sample Qwen engine", "fixture", "Test host", "", 32768, 2});
#if defined(__linux__)
        if (argc >= 3 && argc <= 5 && std::string(argv[1]) == "--fixture") {
            tqserve::MonitorServer server(m, "127.0.0.1", 0, argc >= 4 ? argv[3] : "", "", argc == 5 ? uint16_t(std::stoul(argv[4])) : 0);
            std::cout << server.port() << std::endl;
            std::ifstream initial_file(argv[2]); std::string initial_mode; initial_file >> initial_mode;
            for (int i = 0; i < (initial_mode == "empty" ? 0 : 6); ++i) {
                auto history = m.begin(); history->prefill(1024 * uint32_t(i + 1)); history->decoding();
                history->tokens(128); history->timings(1024 * uint32_t(i + 1), 128, 500 + i * 100, 1500, 140, 110);
                history->finish(i == 2 ? "cancelled" : "complete");
            }
            std::shared_ptr<Monitor::Request> job;
            std::string last_mode;
            for (int step = 0; step < 1200; ++step) {
                std::ifstream file(argv[2]); std::string mode; file >> mode;
                if (mode != last_mode) {
                    if (job) job->finish();
                    if (mode != "idle" && mode != "empty") job = m.begin();
                    if (mode == "prefill" || mode == "decode") job->prefill(8192);
                    if (mode == "decode") job->decoding();
                    if (mode == "failed") job->finish("failed");
                    if (mode == "cancelled") job->finish("cancelled");
                    last_mode = mode;
                }
                if (mode == "decode") job->tokens(uint32_t(step % 512 + 1));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            return 0;
        }
#else
        (void)argc; (void)argv;
#endif
        auto j = snapshot(m);
        check(j.get("schema_version")->num == 1, "version");
        check(j.get("requests")->arr.empty(), "initial empty");
        check(j.get("hardware")->get("gpu_percent")->kind == tqllm::Json::Null, "unknown is null");
        auto r = m.begin();
        check(latest(snapshot(m)).get("phase")->str == "queued", "queue");
        r->prefill(4096);
        check(latest(snapshot(m)).get("prompt_tokens")->num == 4096, "actual prompt count");
        check(latest(snapshot(m)).get("output_tokens")->kind == tqllm::Json::Null, "no invented output");
        r->decoding(); r->tokens(20);
        r->timings(4096, 20, 100, 190, 30, 18);
        j = snapshot(m);
        check(std::fabs(latest(j).get("decode_tokens_per_second")->num - 100) < .001, "engine-rate denominator");
        check(latest(j).get("context_tokens")->num == 4116, "logical context count");
        // A thinking-budget continuation is still one request, with cumulative outputs and timings.
        r->prefill(4116); r->decoding(); r->tokens(5); r->timings(4116, 5, 50, 40, 8, 4);
        j = snapshot(m);
        check(latest(j).get("output_tokens")->num == 25, "multi-pass output");
        check(latest(j).get("decode_ms")->num == 230, "multi-pass timing");
        r->finish("cancelled"); r->tokens(1000); r->finish(); r.reset();
        check(latest(snapshot(m)).get("phase")->str == "cancelled", "terminal state immutable");
        {
            auto failed = m.begin();
        }
        check(latest(snapshot(m)).get("phase")->str == "failed", "abandoned request fails");
        auto active = m.begin();
        for (int i = 0; i < 100; ++i) m.begin()->finish();
        j = snapshot(m);
        check(j.get("requests")->arr.size() <= 33, "bounded completed history");
        check(j.get("requests")->arr.front().get("phase")->str == "queued", "active not evicted");
        active->finish();
        std::atomic<bool> done{false};
        auto parallel = m.begin();
        std::thread writer([&] { parallel->prefill(2000); parallel->decoding(); for (int n=0;n<1000;++n) parallel->tokens(n); parallel->finish(); done=true; });
        while (!done) check(snapshot(m).get("schema_version")->num == 1, "concurrent snapshot valid");
        writer.join();
        // Tickets can outlive the monitor owner safely.
        std::shared_ptr<Monitor::Request> orphan;
        { Monitor temporary({}); orphan = temporary.begin(); }
        orphan.reset();
#if defined(__linux__)
        chat_proxy_gate();
        tqserve::MonitorServer server(m, "127.0.0.1", 0, "test-key");
        const auto html = get(server.port(), "/");
        check(html.find("200 OK") != std::string::npos && html.find("id=\"chatInput\"") != std::string::npos && html.find("id=\"chartLine\"") != std::string::npos, "embedded page");
        check(get(server.port(), "/api/monitor/v1/snapshot").find("401 Unauthorized") != std::string::npos, "auth enforced");
        const auto before = std::chrono::steady_clock::now();
        auto long_prefill = m.begin(); long_prefill->prefill(32768);
        // The inference owner is blocked; the HTTP listener must still answer from its snapshot.
        std::thread blocked([] { std::this_thread::sleep_for(std::chrono::milliseconds(500)); });
        const auto response = get(server.port(), "/api/monitor/v1/snapshot", "test-key");
        check(response.find("200 OK") != std::string::npos && response.find("prefill") != std::string::npos, "responsive during prefill");
        check(std::chrono::steady_clock::now() - before < std::chrono::milliseconds(400), "monitor blocked by inference");
        blocked.join(); long_prefill->finish();
        check(get(server.port(), "/health", "test-key").find("200 OK") != std::string::npos, "monitor health");
        check(get(server.port(), "/api/monitor/v1/snapshot?key=test-key").find("401 Unauthorized") != std::string::npos, "query keys rejected");
        check(get(server.port(), "/api/monitor/v1/snapshot", "test-key", "POST").find("405") != std::string::npos, "read only");
        check(response.find("test-key") == std::string::npos, "no credentials in snapshot");
#endif
        std::cout << "monitor: all checks passed\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << "monitor: " << e.what() << '\n'; return 1; }
}

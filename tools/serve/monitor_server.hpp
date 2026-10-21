#pragma once
#include "http.hpp"
#include "monitor_host.hpp"
#include "monitor_page.hpp"
#include <atomic>
#include <stdexcept>
#include <thread>
#if defined(__linux__)
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace tqserve {
#if defined(__linux__)
// Independent telemetry listener and bounded same-instance chat bridge; no engine locks.
class MonitorServer {
public:
    MonitorServer(Monitor & monitor, const std::string & host, uint16_t port,
                  std::string key, std::string pci = {}, uint16_t inference_port = 0, bool listen = true)
        : monitor_(monitor), key_(std::move(key)), pci_(std::move(pci)), inference_host_(host == "0.0.0.0" ? "127.0.0.1" : host), inference_port_(inference_port) {
        if (listen) {
        socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (socket_ < 0) throw std::runtime_error("monitor socket failed");
        int one = 1; ::setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1 ||
            ::bind(socket_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0 || ::listen(socket_, 16) < 0) {
            ::close(socket_); throw std::runtime_error("monitor bind/listen failed (use an IPv4 bind address)");
        }
        socklen_t size = sizeof(address); ::getsockname(socket_, reinterpret_cast<sockaddr *>(&address), &size);
        port_ = ntohs(address.sin_port);
        }
        try { thread_ = std::thread([this] {
            try { run(); } catch (...) { stop_ = true; std::fputs("tqserve: monitor listener stopped\n", stderr); }
        }); }
        catch (...) { ::close(socket_); throw; }
    }
    ~MonitorServer() {
        stop_ = true;
        if (socket_ >= 0) ::shutdown(socket_, SHUT_RDWR);
        thread_.join();
        {
            std::lock_guard<std::mutex> lock(chat_mutex_);
            if (chat_client_ >= 0) ::shutdown(chat_client_, SHUT_RDWR);
            if (chat_upstream_ >= 0) ::shutdown(chat_upstream_, SHUT_RDWR);
        }
        if (chat_.joinable()) chat_.join();
        ::close(socket_);
    }
    uint16_t port() const { return port_; }
    bool serve(int fd, const Request & req) {
        timeval timeout{1,0}; ::setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        return handle_request(fd,req);
    }
    MonitorServer(const MonitorServer &) = delete;
    MonitorServer & operator=(const MonitorServer &) = delete;
private:
    static void send_response(int fd, int status, const std::string & body, bool html = false) {
        std::string data = "HTTP/1.1 " + std::to_string(status) + (status == 200 ? " OK" : status == 401 ? " Unauthorized" : " Error");
        data += "\r\nContent-Type: " + std::string(html ? "text/html" : "application/json") + "; charset=utf-8\r\n";
        data += "Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n";
        if (html) data += "Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'\r\n";
        data += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
        size_t offset = 0;
        while (offset < data.size()) {
            const auto n = ::send(fd, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
            if (n <= 0) break;
            offset += size_t(n);
        }
    }
    bool handle(int fd) {
        timeval timeout{1, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        RequestReader reader(1024 * 1024, 8192); uint8_t buffer[2048];
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!reader.complete() && !reader.failed() && !stop_ && std::chrono::steady_clock::now() < deadline) {
            const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
            if (n <= 0) return false;
            reader.append(buffer, size_t(n));
        }
        const auto req = reader.request();
        if (!req.ok()) { send_response(fd, 400, "{\"error\":\"invalid request\"}"); return false; }
        return handle_request(fd,req);
    }
    bool handle_request(int fd, const Request & req) {
        std::lock_guard<std::mutex> request_lock(request_mutex_);
        if (req.method == "GET" && (req.path == "/" || req.path == "/monitor" || req.path == "/monitor/")) { send_response(fd, 200, monitor_page, true); return false; }
        if (!key_.empty() && req.authorization != "Bearer " + key_) {
            send_response(fd, 401, "{\"error\":\"API key required\"}"); return false;
        }
        if (req.method == "POST" && req.path == "/api/monitor/v1/chat/completions") {
            std::string type = req.content_type.substr(0, req.content_type.find(';'));
            for (auto & c : type) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (type != "application/json" || (!req.origin.empty() && req.origin != "http://" + req.host && req.origin != "https://" + req.host)) {
                send_response(fd, 403, "{\"error\":\"same-origin JSON required\"}"); return false;
            }
            if (!inference_port_) { send_response(fd, 503, "{\"error\":\"chat unavailable\"}"); return false; }
            if (chat_active_) { send_response(fd, 429, "{\"error\":\"a monitor chat request is already running\"}"); return false; }
            if (chat_.joinable()) chat_.join();
            { std::lock_guard<std::mutex> lock(chat_mutex_); chat_client_ = fd; }
            chat_active_ = true;
            try { chat_ = std::thread([this, body = std::string(req.body.begin(), req.body.end())] { proxy_chat(body); }); }
            catch (...) { chat_active_ = false; std::lock_guard<std::mutex> lock(chat_mutex_); chat_client_ = -1; throw; }
            return true;
        }
        if (req.method != "GET") { send_response(fd, 405, "{\"error\":\"method not allowed\"}"); return false; }
        if (req.path == "/api/monitor/v1/snapshot") send_response(fd, 200, monitor_.snapshot());
        else if (req.path == "/health") send_response(fd, 200, "{\"status\":\"ok\",\"service\":\"monitor\"}");
        else send_response(fd, 404, "{\"error\":\"not found\"}");
        return false;
    }
    static bool write_bytes(int fd, const char * bytes, size_t size) {
        while (size) {
            const auto n = ::send(fd, bytes, size, MSG_NOSIGNAL);
            if (n <= 0) return false;
            bytes += n; size -= size_t(n);
        }
        return true;
    }
    void proxy_chat(const std::string & body) {
        int upstream = -1, client;
        { std::lock_guard<std::mutex> lock(chat_mutex_); client = chat_client_; }
        bool started = false;
        try {
            upstream = ::socket(AF_INET, SOCK_STREAM, 0);
            if (upstream < 0) throw std::runtime_error("socket");
            { std::lock_guard<std::mutex> lock(chat_mutex_); chat_upstream_ = upstream; }
            timeval timeout{1, 0};
            ::setsockopt(upstream, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            ::setsockopt(upstream, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(inference_port_);
            ::inet_pton(AF_INET, inference_host_.c_str(), &address.sin_addr);
            if (::connect(upstream, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) throw std::runtime_error("connect");
            std::string request = "POST /v1/chat/completions HTTP/1.1\r\nHost: " + inference_host_ + ":" + std::to_string(inference_port_) + "\r\nContent-Type: application/json\r\n";
            if (!key_.empty()) request += "Authorization: Bearer " + key_ + "\r\n";
            request += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
            if (!write_bytes(upstream, request.data(), request.size())) throw std::runtime_error("send");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
            char buffer[8192];
            while (!stop_ && std::chrono::steady_clock::now() < deadline) {
                pollfd ready[2]{{client, POLLIN, 0}, {upstream, POLLIN, 0}};
                const auto available = ::poll(ready, 2, 200);
                if (available < 0) { if (errno == EINTR) continue; break; }
                if (!available) continue;
                if (ready[0].revents) {
                    char probe;
                    if (ready[0].revents & (POLLHUP | POLLERR | POLLNVAL)) break;
                    if ((ready[0].revents & POLLIN) && ::recv(client, &probe, 1, MSG_PEEK | MSG_DONTWAIT) == 0) break;
                }
                if (ready[1].revents & (POLLERR | POLLNVAL)) break;
                if (!(ready[1].revents & (POLLIN | POLLHUP))) continue;
                const auto n = ::recv(upstream, buffer, sizeof(buffer), 0);
                if (n == 0) break;
                if (n < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue; break; }
                started = true;
                if (!write_bytes(client, buffer, size_t(n))) break;
            }
        } catch (...) {
            if (!started && !stop_) send_response(client, 502, "{\"error\":\"inference endpoint unavailable\"}");
        }
        {
            std::lock_guard<std::mutex> lock(chat_mutex_);
            if (upstream >= 0) ::close(upstream);
            ::close(client); chat_client_ = chat_upstream_ = -1;
        }
        chat_active_ = false;
    }
    void run() {
        MonitorHost collector(pci_);
        const auto hardware_interval = std::chrono::seconds(3);
        auto sampled = std::chrono::steady_clock::now() - hardware_interval;
        while (!stop_) {
            const auto now = std::chrono::steady_clock::now();
            if (now - sampled >= hardware_interval) { monitor_.hardware(collector.sample()); sampled = now; }
            if (socket_ < 0) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); continue; }
            pollfd p{socket_, POLLIN, 0};
            if (::poll(&p, 1, 200) <= 0 || !(p.revents & POLLIN)) continue;
            const int fd = ::accept(socket_, nullptr, nullptr);
            if (fd >= 0) {
                try { if (!handle(fd)) ::close(fd); }
                catch (...) { ::close(fd); }
            }
        }
    }
    Monitor & monitor_;
    std::string key_, pci_, inference_host_;
    uint16_t inference_port_ = 0;
    std::mutex chat_mutex_, request_mutex_;
    int chat_client_ = -1, chat_upstream_ = -1;
    std::atomic<bool> chat_active_{false};
    std::thread chat_;
    int socket_ = -1;
    uint16_t port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};
#endif
} // namespace tqserve

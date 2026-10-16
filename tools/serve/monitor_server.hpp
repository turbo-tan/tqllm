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
// An independent, bounded read-only listener; no inference calls or engine locks.
class MonitorServer {
public:
    MonitorServer(Monitor & monitor, const std::string & host, uint16_t port,
                  std::string key, std::string pci = {})
        : monitor_(monitor), key_(std::move(key)), pci_(std::move(pci)) {
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
        try { thread_ = std::thread([this] {
            try { run(); } catch (...) { stop_ = true; std::fputs("tqserve: monitor listener stopped\n", stderr); }
        }); }
        catch (...) { ::close(socket_); throw; }
    }
    ~MonitorServer() {
        stop_ = true;
        ::shutdown(socket_, SHUT_RDWR);
        thread_.join();
        ::close(socket_);
    }
    uint16_t port() const { return port_; }
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
    void handle(int fd) {
        timeval timeout{1, 0};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        RequestReader reader(0, 8192); uint8_t buffer[2048];
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!reader.complete() && !reader.failed() && !stop_ && std::chrono::steady_clock::now() < deadline) {
            const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
            if (n <= 0) return;
            reader.append(buffer, size_t(n));
        }
        const auto req = reader.request();
        if (!req.ok()) { send_response(fd, 400, "{\"error\":\"invalid request\"}"); return; }
        if (req.method != "GET") { send_response(fd, 405, "{\"error\":\"read-only monitor\"}"); return; }
        if (req.path == "/" || req.path == "/monitor") { send_response(fd, 200, monitor_page, true); return; }
        if (!key_.empty() && req.authorization != "Bearer " + key_) {
            send_response(fd, 401, "{\"error\":\"API key required\"}"); return;
        }
        if (req.path == "/api/monitor/v1/snapshot") send_response(fd, 200, monitor_.snapshot());
        else if (req.path == "/health") send_response(fd, 200, "{\"status\":\"ok\",\"service\":\"monitor\"}");
        else send_response(fd, 404, "{\"error\":\"not found\"}");
    }
    void run() {
        MonitorHost collector(pci_);
        auto sampled = std::chrono::steady_clock::now() - std::chrono::seconds(2);
        while (!stop_) {
            const auto now = std::chrono::steady_clock::now();
            if (now - sampled >= std::chrono::seconds(1)) { monitor_.hardware(collector.sample()); sampled = now; }
            pollfd p{socket_, POLLIN, 0};
            if (::poll(&p, 1, 200) <= 0 || !(p.revents & POLLIN)) continue;
            const int fd = ::accept(socket_, nullptr, nullptr);
            if (fd >= 0) { handle(fd); ::close(fd); }
        }
    }
    Monitor & monitor_;
    std::string key_, pci_;
    int socket_ = -1;
    uint16_t port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};
#endif
} // namespace tqserve

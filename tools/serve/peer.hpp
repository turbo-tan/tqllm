#pragma once
// tqserve --peer URL: the other boxes behind this front door. Status queries (POST /admin/route) and request forwarding
// (bytes relayed both ways, so streaming answers pass through unchanged). POSIX sockets.
#include "http.hpp"
#include "router.hpp"
#include <third_party/nlohmann/json.hpp>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

#ifndef POLLRDHUP
#define POLLRDHUP 0  // macOS: POLLHUP reports the client going away
#endif

namespace tqserve {

struct Peer {
    std::string host;
    uint16_t port = 0;
};

// "http://host:port", "host:port"
inline bool parse_peer(std::string s, Peer & p) {
    if (s.rfind("http://", 0) == 0) s = s.substr(7);
    while (!s.empty() && s.back() == '/') s.pop_back();
    const size_t c = s.rfind(':');
    if (c == std::string::npos || c == 0) return false;
    try {
        const unsigned long v = std::stoul(s.substr(c + 1));
        if (v == 0 || v > 65535) return false;
        p.port = uint16_t(v);
    } catch (...) { return false; }
    p.host = s.substr(0, c);
    return true;
}

inline int peer_connect(const Peer & p, int timeout_ms) {
    addrinfo hints{}, * res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (::getaddrinfo(p.host.c_str(), std::to_string(p.port).c_str(), &hints, &res) != 0 || !res) return -1;
    int fd = -1;
    for (auto * a = res; a && fd < 0; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        const int fl = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        int rc = ::connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            pollfd pf{fd, POLLOUT, 0};
            int err = 0; socklen_t len = sizeof err;
            rc = ::poll(&pf, 1, timeout_ms) == 1 && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0 ? 0 : -1;
        }
        if (rc != 0) { ::close(fd); fd = -1; continue; }
        ::fcntl(fd, F_SETFL, fl);
    }
    ::freeaddrinfo(res);
    return fd;
}

inline bool peer_send(int fd, const std::string & d) {
    for (size_t sent = 0; sent < d.size();) {
        const ssize_t n = ::send(fd, d.data() + sent, d.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += size_t(n);
    }
    return true;
}

// The request as this server received it, for the peer; X-Tqllm-Routed stops the peer from routing it again.
inline std::string forward_bytes(const Request & r, const Peer & p) {
    std::string h = r.method + " " + r.path + (r.query.empty() ? "" : "?" + r.query) + " HTTP/1.1\r\nHost: " + p.host + ":" +
                    std::to_string(p.port) + "\r\nX-Tqllm-Routed: 1\r\nConnection: close\r\n";
    if (!r.content_type.empty()) h += "Content-Type: " + r.content_type + "\r\n";
    if (!r.authorization.empty()) h += "Authorization: " + r.authorization + "\r\n";
    h += "Content-Length: " + std::to_string(r.body.size()) + "\r\n\r\n";
    h.append(r.body.begin(), r.body.end());
    return h;
}

inline std::string status_json(const BoxStatus & b) {
    char out[256];
    std::snprintf(out, sizeof out, "{\"free\":%d,\"busy\":%d,\"queued\":%d,\"reuse\":%.0f,\"prefill_tps\":%.1f,\"decode_tps\":%.1f,\"hold_s\":%.2f}",
                  b.free, b.busy, b.queued, b.reuse, b.prefill_tps, b.decode_tps, b.hold_s);
    return out;
}

// POST /admin/route on the peer with the rendered prompt; not ok when it does not answer within timeout_ms.
inline BoxStatus peer_status(const Peer & p, const std::string & prompt_text, const std::string & authorization, int timeout_ms) {
    BoxStatus b;
    const int fd = peer_connect(p, timeout_ms);
    if (fd < 0) return b;
    std::string req = "POST /admin/route HTTP/1.1\r\nHost: " + p.host + "\r\nConnection: close\r\nContent-Type: text/plain\r\n";
    if (!authorization.empty()) req += "Authorization: " + authorization + "\r\n";
    req += "Content-Length: " + std::to_string(prompt_text.size()) + "\r\n\r\n" + prompt_text;
    std::string in;
    if (peer_send(fd, req)) {
        char buf[4096];
        for (;;) {
            pollfd pf{fd, POLLIN, 0};
            if (::poll(&pf, 1, timeout_ms) != 1) { in.clear(); break; }
            const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
            if (n <= 0) break;
            in.append(buf, size_t(n));
        }
    }
    ::close(fd);
    const size_t body = in.find("\r\n\r\n");
    if (in.rfind("HTTP/1.1 200", 0) != 0 || body == std::string::npos) return b;
    try {
        const auto j = tqllm::parse_json(in.substr(body + 4));
        auto num = [&](const char * k) { const auto * v = j.get(k); return v && v->kind == tqllm::Json::Number ? v->num : 0.0; };
        b.free = int(num("free")); b.busy = int(num("busy")); b.queued = int(num("queued"));
        b.reuse = num("reuse"); b.prefill_tps = num("prefill_tps"); b.decode_tps = num("decode_tps"); b.hold_s = num("hold_s");
        b.ok = j.get("free") != nullptr;
    } catch (...) {}
    return b;
}

// Sends the request to the peer and relays its answer to `client` until the peer closes. False when nothing reached the
// client (connect or send failed): the caller may still serve the request itself.
inline bool peer_proxy(const Peer & p, const std::string & bytes, int client) {
    const int fd = peer_connect(p, 1000);
    if (fd < 0) return false;
    if (!peer_send(fd, bytes)) { ::close(fd); return false; }
    char buf[16384];
    bool any = false;
    for (;;) {
        pollfd pf[2] = {{fd, POLLIN, 0}, {client, POLLRDHUP, 0}};
        if (::poll(pf, 2, -1) < 0) { if (errno == EINTR) continue; break; }
        if (pf[1].revents & (POLLRDHUP | POLLHUP | POLLERR)) break;  // client gone: closing the peer side cancels there
        if (!(pf[0].revents & (POLLIN | POLLHUP | POLLERR))) continue;
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        any = true;
        if (!peer_send(client, std::string(buf, size_t(n)))) break;
    }
    ::close(fd);
    return any;
}

} // namespace tqserve

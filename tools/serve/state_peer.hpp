#pragma once
// Conversation states between boxes (task 30): a box that serves a follow-up turn whose kept state another box holds
// fetches it over the link instead of prefilling the shared prefix again. The owner answers on its internal listener
// (tqserve --state-listen LINK_IP:PORT, never the API port): POST /v1/state/fetch {"tokens": [...], "entity": "...",
// "max_bytes": N} -> 200 + the blob (src/kv/conversation_state.hpp encoding) with X-Tqllm-State-Positions / -Conv, or
// 404; POST /v1/state/ack {"conv": C} after the fetcher stored it (the owner then drops its parked copy: two-phase, a
// failed fetch leaves the owner's state alone). A fetch that prices above the prefill it saves, times out, exceeds the
// ceiling or loses the owner falls back to prefilling here; every decision is logged with its numbers.
#include "peer.hpp"
#include "kv/conversation_state.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace tqserve {

constexpr size_t kStateFetchMax = size_t(4) << 30;  // a fetch above this is refused (the park's own budget)

// measured link rate (bytes / s of the last fetches; 0 until one ran: a fetch is then priced at 1 GB/s)
struct LinkRate {
    std::mutex m; double bps = 0;
    void note(size_t bytes, double s) { if (s <= 0) return; std::lock_guard<std::mutex> g(m); const double r = double(bytes) / s; bps = bps > 0 ? 0.7 * bps + 0.3 * r : r; }
    double measured() { std::lock_guard<std::mutex> g(m); return bps; }
    double get() { const double rate = measured(); return rate > 0 ? rate : 1e9; }
};

// "host:port;positions=N" -> peer + N
inline bool parse_state_owner(const std::string & v, Peer & p, double & positions) {
    const size_t semi = v.find(';');
    if (!parse_peer(v.substr(0, semi), p)) return false;
    positions = 0;
    if (semi != std::string::npos && v.compare(semi + 1, 10, "positions=") == 0) {
        try { positions = std::stod(v.substr(semi + 11)); } catch (...) { return false; }
    }
    return true;
}

inline std::string tokens_json(const std::vector<uint32_t> & t, const std::string & entity, size_t max_bytes) {
    std::string s = "{\"entity\":\"";
    for (char c : entity) if (c != '"' && c != '\\' && c >= 0x20) s += c;
    s += "\",\"max_bytes\":" + std::to_string(max_bytes) + ",\"tokens\":[";
    for (size_t i = 0; i < t.size(); ++i) { if (i) s += ','; s += std::to_string(t[i]); }
    return s + "]}";
}

struct StateFetch {
    bool ok = false;
    std::string why;            // why not (logged by the caller)
    size_t bytes = 0; uint32_t positions = 0; uint64_t owner_conv = 0;
    double seconds = 0;
};

// Fetches the owner's best kept state for `tokens`. decide(positions, bytes) runs once the owner has named both and may
// refuse the transfer (prefill is cheaper); adopt(blob) stores it here ("" = stored). Acks the owner after a stored one.
inline StateFetch fetch_state(const Peer & owner, const std::vector<uint32_t> & tokens, const std::string & entity, LinkRate & rate,
                              const std::function<bool(uint32_t, size_t)> & decide, const std::function<std::string(tqllm::StateBlob)> & adopt) {
    StateFetch r;
    const auto t0 = std::chrono::steady_clock::now();
    const int fd = peer_connect(owner, 1000);
    if (fd < 0) { r.why = "owner unreachable"; return r; }
    struct Close { int fd; ~Close() { tqnet::close(fd); } } close_fd{fd};
    const std::string body = tokens_json(tokens, entity, kStateFetchMax);
    const std::string req = "POST /v1/state/fetch HTTP/1.1\r\nHost: " + owner.host + "\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body;
    if (!peer_send(fd, req)) { r.why = "owner closed the request"; return r; }
    // headers, then the body through a bounded reader: every recv waits at most 2 s, the whole body at most 5 s + 1 s per 200 MB
    std::string buf; size_t at = 0;
    auto fill = [&](int wait_ms) {
        char tmp[65536];
        auto pf = tqnet::poll_entry(fd, POLLIN);
        if (tqnet::poll(&pf, 1, wait_ms) != 1) return false;
        const ssize_t n = tqnet::recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf.append(tmp, size_t(n));
        return true;
    };
    size_t head_end = std::string::npos;
    while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > 65536 || !fill(5000)) { r.why = "owner sent no answer"; return r; }
    }
    const std::string head = buf.substr(0, head_end);
    at = head_end + 4;
    if (head.rfind("HTTP/1.1 200", 0) != 0) { r.why = "owner answered " + head.substr(0, head.find("\r\n")); return r; }
    auto field = [&](const char * name) -> double {
        const size_t p = head.find(name);
        if (p == std::string::npos) return -1;
        try { return std::stod(head.substr(p + std::strlen(name))); } catch (...) { return -1; }
    };
    const double length = field("Content-Length: "), positions = field("X-Tqllm-State-Positions: "), conv = field("X-Tqllm-State-Conv: ");
    if (length <= 0 || positions <= 0) { r.why = "owner's answer names no size / positions"; return r; }
    r.bytes = size_t(length); r.positions = uint32_t(positions); r.owner_conv = conv > 0 ? uint64_t(conv) : 0;
    if (r.bytes > kStateFetchMax) { r.why = "state of " + std::to_string(r.bytes) + " bytes is above the fetch ceiling"; return r; }
    if (!decide(r.positions, r.bytes)) { r.why = "prefill is cheaper"; return r; }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(5000 + int64_t(r.bytes / (200u << 20)) * 1000);
    size_t taken = 0;
    auto source = [&](void * p, size_t n) {
        auto * out = static_cast<char *>(p);
        while (n) {
            if (at < buf.size()) {
                const size_t k = std::min(n, buf.size() - at);
                std::memcpy(out, buf.data() + at, k); out += k; n -= k; at += k; taken += k;
                if (at == buf.size()) { buf.clear(); at = 0; }
                continue;
            }
            if (std::chrono::steady_clock::now() > deadline || !fill(2000)) return false;
        }
        return true;
    };
    tqllm::StateBlob blob;
    try { blob = tqllm::decode_state_blob(source, r.bytes, "state from " + owner.host); }
    catch (const std::exception & e) { r.why = e.what(); return r; }
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rate.note(r.bytes, r.seconds);
    if (const std::string why = adopt(std::move(blob)); !why.empty()) { r.why = why; return r; }
    r.ok = true;
    if (r.owner_conv) {  // the owner drops its parked copy now; a lost ack only leaves it parked (an eviction takes it later)
        const std::string ab = "{\"conv\":" + std::to_string(r.owner_conv) + "}";
        const int afd = peer_connect(owner, 500);
        if (afd >= 0) {
            peer_send(afd, "POST /v1/state/ack HTTP/1.1\r\nHost: " + owner.host + "\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: " +
                               std::to_string(ab.size()) + "\r\n\r\n" + ab);
            char tmp[256]; auto pf = tqnet::poll_entry(afd, POLLIN); if (tqnet::poll(&pf, 1, 500) == 1) (void) tqnet::recv(afd, tmp, sizeof tmp, 0);
            tqnet::close(afd);
        }
    }
    return r;
}

// The owner's answer to a fetch: headers, then the blob straight onto the socket. stall_ms (TQLLM_DEBUG=state_fetch_stall,
// the kill test) waits after the first part so the owner can be stopped mid-transfer.
inline bool send_state(int client, const tqllm::StateBlob & b, int stall_ms) {
    const size_t n = tqllm::encoded_size(b);
    const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: " + std::to_string(n) +
                             "\r\nX-Tqllm-State-Positions: " + std::to_string(b.meta.positions) + "\r\nX-Tqllm-State-Conv: " + std::to_string(b.meta.conv) +
                             "\r\nConnection: close\r\n\r\n";
    if (!peer_send(client, head)) return false;
    size_t sent = 0; bool stalled = false;
    return tqllm::encode_state_blob(b, [&](const void * p, size_t k) {
        const auto * c = static_cast<const char *>(p);
        for (size_t off = 0; off < k;) {
            const ssize_t w = tqnet::send(client, c + off, k - off, MSG_NOSIGNAL);
            if (w <= 0) return false;
            off += size_t(w); sent += size_t(w);
        }
        if (stall_ms > 0 && !stalled && sent > (n >> 4)) { stalled = true; std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms)); }
        return true;
    });
}

}  // namespace tqserve

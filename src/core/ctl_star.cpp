#include "ctl_star.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace tqllm {
namespace {
constexpr uint32_t kMagic = 0x74716c32;  // "tql2"
struct Hello { uint32_t magic; int32_t rank, world; char host[64]; char tag[64]; };
struct Reply { int32_t ok, local; uint32_t blob; char err[192]; };

void full_io(int fd, void * p, size_t n, bool send) {
    auto * c = static_cast<char *>(p);
    while (n) {
        const ssize_t k = send ? ::send(fd, c, n, MSG_NOSIGNAL) : ::recv(fd, c, n, 0);
        if (k <= 0) throw std::runtime_error("tp: control socket closed");
        c += k; n -= size_t(k);
    }
}
void put(char (&dst)[64], const std::string & s) { std::memset(dst, 0, sizeof dst); std::strncpy(dst, s.c_str(), sizeof dst - 1); }
std::string str(const char * s, size_t n) { return std::string(s, strnlen(s, n)); }
void nodelay(int fd) { int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); }
}  // namespace

CtlStar::CtlStar(int rank, int world, const std::string & peer, int port, std::vector<char> & blob, const std::string & tag,
                 const std::string & host_in)
    : rank_(rank), world_(world), fd_(size_t(world > 0 ? world : 1), -1) {
    if (world < 2 || rank < 0 || rank >= world) throw std::invalid_argument("tp: rank " + std::to_string(rank) + " of " + std::to_string(world));
    std::string host = host_in;
    if (host.empty()) { char h[256] = {}; gethostname(h, sizeof h - 1); host = h; }
    if (rank == 0) {
        const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port)); a.sin_addr.s_addr = INADDR_ANY;
        if (::bind(ls, reinterpret_cast<sockaddr *>(&a), sizeof a) || ::listen(ls, world)) {
            ::close(ls);
            throw std::runtime_error("tp: cannot listen on port " + std::to_string(port));
        }
        std::vector<Hello> hello(static_cast<size_t>(world));
        std::vector<int> pending;  // connections in arrival order
        std::string err;
        for (int got = 1; got < world; ++got) {
            const int fd = ::accept(ls, nullptr, nullptr);
            if (fd < 0) { ::close(ls); throw std::runtime_error("tp: accept failed"); }
            nodelay(fd);
            pending.push_back(fd);
            Hello h{};
            try { full_io(fd, &h, sizeof h, false); } catch (...) { err = "a rank hung up during the handshake"; break; }
            const std::string who = "rank " + std::to_string(h.rank) + " (" + str(h.host, sizeof h.host) + ")";
            if (h.magic != kMagic) err = "a connection on port " + std::to_string(port) + " is not a tqllm rank";
            else if (h.world != world) err = who + " was started with " + std::to_string(h.world) + " ranks, rank 0 with " + std::to_string(world);
            else if (h.rank < 1 || h.rank >= world) err = who + ": rank out of range";
            else if (fd_[size_t(h.rank)] >= 0) err = "two processes claim rank " + std::to_string(h.rank);
            else if (str(h.tag, sizeof h.tag) != tag) err = who + " runs build " + str(h.tag, sizeof h.tag) + ", rank 0 runs " + tag;
            if (!err.empty()) break;
            fd_[size_t(h.rank)] = fd;
            hello[size_t(h.rank)] = h;
        }
        ::close(ls);
        put(hello[0].host, host);
        for (int fd : pending) {
            int r = -1;
            for (int i = 1; i < world; ++i) if (fd_[size_t(i)] == fd) r = i;
            Reply rep{}; rep.ok = err.empty(); rep.blob = uint32_t(blob.size());
            std::strncpy(rep.err, err.c_str(), sizeof rep.err - 1);
            if (r > 0) for (int i = 0; i < world; ++i) rep.local += std::strcmp(hello[size_t(i)].host, hello[size_t(r)].host) == 0;
            try { full_io(fd, &rep, sizeof rep, true); if (rep.ok) full_io(fd, blob.data(), blob.size(), true); } catch (...) {}
        }
        if (!err.empty()) {
            for (int fd : pending) ::close(fd);
            fd_.assign(fd_.size(), -1);
            throw std::runtime_error("tp: " + err);
        }
        local_ = 0;
        for (int i = 0; i < world; ++i) local_ += std::strcmp(hello[size_t(i)].host, hello[0].host) == 0;
    } else {
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port));
        if (inet_pton(AF_INET, peer.c_str(), &a.sin_addr) != 1) throw std::runtime_error("tp: bad peer address " + peer);
        int fd = -1;
        for (int tries = 0;; ++tries) {
            fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) == 0) break;
            ::close(fd);
            if (tries > 6000) throw std::runtime_error("tp: cannot reach rank 0 at " + peer + ":" + std::to_string(port) + " in 10 minutes");
            usleep(100000);
        }
        nodelay(fd);
        fd_[0] = fd;
        Hello h{}; h.magic = kMagic; h.rank = rank; h.world = world; put(h.host, host); put(h.tag, tag);
        Reply rep{};
        try {
            full_io(fd, &h, sizeof h, true);
            full_io(fd, &rep, sizeof rep, false);
        } catch (...) { ::close(fd); fd_[0] = -1; throw std::runtime_error("tp: rank 0 at " + peer + " closed the handshake"); }
        if (!rep.ok) { ::close(fd); fd_[0] = -1; throw std::runtime_error("tp: rank 0 refused: " + str(rep.err, sizeof rep.err)); }
        blob.resize(rep.blob);
        full_io(fd, blob.data(), blob.size(), false);
        local_ = rep.local;
    }
}
CtlStar::~CtlStar() { for (int fd : fd_) if (fd >= 0) ::close(fd); }
int CtlStar::fd_of(int peer) const {
    if (peer < 0 || peer >= world_ || peer == rank_ || (rank_ != 0 && peer != 0) || fd_[size_t(peer)] < 0)
        throw std::logic_error("tp: no control link from rank " + std::to_string(rank_) + " to rank " + std::to_string(peer));
    return fd_[size_t(peer)];
}
void CtlStar::send(int peer, const void * p, size_t n) { full_io(fd_of(peer), const_cast<void *>(p), n, true); }
void CtlStar::recv(int peer, void * p, size_t n) { full_io(fd_of(peer), p, n, false); }
void CtlStar::fan_out(const void * p, size_t n) {
    if (rank_ != 0) throw std::logic_error("tp: only rank 0 sends to every rank");
    for (int r = 1; r < world_; ++r) send(r, p, n);
}
void CtlStar::all_gather(const void * mine, void * all, size_t bytes) {
    auto * out = static_cast<char *>(all);
    if (rank_ == 0) {
        std::memcpy(out, mine, bytes);
        for (int r = 1; r < world_; ++r) recv(r, out + size_t(r) * bytes, bytes);
        fan_out(out, size_t(world_) * bytes);
    } else {
        send(0, mine, bytes);
        recv(0, out, size_t(world_) * bytes);
    }
}
std::vector<int> CtlStar::fds() const {
    std::vector<int> v;
    for (int fd : fd_) if (fd >= 0) v.push_back(fd);
    return v;
}
void CtlStar::shutdown_all() { for (int fd : fd_) if (fd >= 0) ::shutdown(fd, SHUT_RDWR); }
}  // namespace tqllm

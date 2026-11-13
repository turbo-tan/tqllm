#include "tp_comm.hpp"

#include <nccl.h>

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <poll.h>
#include <atomic>
#include <cstdio>
#include <thread>

namespace tqllm {
namespace {
void nck(ncclResult_t r, const char * what) {
    if (r != ncclSuccess) throw std::runtime_error(std::string("tp: ") + what + ": " + ncclGetErrorString(r));
}
void full_io(int fd, void * p, size_t n, bool send) {
    auto * c = static_cast<char *>(p);
    while (n) {
        const ssize_t k = send ? ::send(fd, c, n, MSG_NOSIGNAL) : ::recv(fd, c, n, 0);
        if (k <= 0) throw std::runtime_error("tp: control socket closed");
        c += k; n -= size_t(k);
    }
}
}  // namespace

struct TpComm::Impl {
    ncclComm_t comm = nullptr;
    int ctl = -1;
    // a rank that dies (killed, out of memory) must not leave the other spinning in a collective: the watchdog sees the
    // control connection hang up and aborts the communicator, so the pending NCCL call returns an error
    std::atomic<bool> stop{false}, aborted{false};
    std::thread watchdog;
    void live() const { if (aborted) throw std::runtime_error("tp: the other rank went away"); }
};

TpComm::TpComm(int rank, int world, const std::string & peer, int port) : impl_(std::make_unique<Impl>()), rank_(rank), world_(world) {
    if (world != 2 || rank < 0 || rank > 1) throw std::invalid_argument("tp: two ranks only");
    ncclUniqueId id{};
    if (rank == 0) {
        nck(ncclGetUniqueId(&id), "unique id");
        const int ls = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port)); a.sin_addr.s_addr = INADDR_ANY;
        if (::bind(ls, reinterpret_cast<sockaddr *>(&a), sizeof a) || ::listen(ls, 1)) throw std::runtime_error("tp: cannot listen");
        const int fd = ::accept(ls, nullptr, nullptr);
        if (fd < 0) throw std::runtime_error("tp: accept failed");
        full_io(fd, &id, sizeof id, true);
        ::close(ls);
        impl_->ctl = fd;
    } else {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port));
        if (inet_pton(AF_INET, peer.c_str(), &a.sin_addr) != 1) throw std::runtime_error("tp: bad peer address " + peer);
        int tries = 0;
        while (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a)) {
            if (++tries > 6000) throw std::runtime_error("tp: cannot reach rank 0 at " + peer + " in 10 minutes");
            usleep(100000);
        }
        full_io(fd, &id, sizeof id, false);
        impl_->ctl = fd;
    }
    int one = 1;
    setsockopt(impl_->ctl, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    nck(ncclCommInitRank(&impl_->comm, world, id, rank), "comm init");
    Impl * im = impl_.get();
    im->watchdog = std::thread([im] {
        while (!im->stop) {
            pollfd p{im->ctl, POLLRDHUP, 0};
            if (::poll(&p, 1, 1000) > 0 && (p.revents & (POLLRDHUP | POLLHUP | POLLERR))) {
                if (im->stop) return;
                std::fprintf(stderr, "tp: the other rank went away; aborting the communicator\n");
                im->aborted = true;
                ncclCommAbort(im->comm);
                im->comm = nullptr;
                return;
            }
        }
    });
}
TpComm::~TpComm() {
    if (!impl_) return;
    impl_->stop = true;
    if (impl_->watchdog.joinable()) impl_->watchdog.join();
    if (impl_->comm) ncclCommDestroy(impl_->comm);
    if (impl_->ctl >= 0) ::close(impl_->ctl);
}
void TpComm::all_reduce_sum(float * x, size_t n, cudaStream_t s) { impl_->live(); nck(ncclAllReduce(x, x, n, ncclFloat, ncclSum, impl_->comm, s), "all-reduce"); }
void TpComm::broadcast(void * buf, size_t bytes, int root, cudaStream_t s) { impl_->live(); nck(ncclBroadcast(buf, buf, bytes, ncclUint8, root, impl_->comm, s), "broadcast"); }
void TpComm::all_gather(const void * send, void * recv, size_t bytes, cudaStream_t s) { impl_->live(); nck(ncclAllGather(send, recv, bytes, ncclUint8, impl_->comm, s), "all-gather"); }
void TpComm::send(const void * buf, size_t bytes, int peer, cudaStream_t s) { impl_->live(); nck(ncclSend(buf, bytes, ncclUint8, peer, impl_->comm, s), "send"); }
void TpComm::recv(void * buf, size_t bytes, int peer, cudaStream_t s) { impl_->live(); nck(ncclRecv(buf, bytes, ncclUint8, peer, impl_->comm, s), "recv"); }
bool TpComm::alive() const { return !impl_->aborted; }
void TpComm::group_begin() { nck(ncclGroupStart(), "group start"); }
void TpComm::group_end() { nck(ncclGroupEnd(), "group end"); }
void TpComm::ctl_send(const void * buf, size_t bytes) { full_io(impl_->ctl, const_cast<void *>(buf), bytes, true); }
void TpComm::ctl_recv(void * buf, size_t bytes) { full_io(impl_->ctl, buf, bytes, false); }
}  // namespace tqllm

#include "tp_comm.hpp"

#include <nccl.h>

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

namespace tqllm {
namespace {
void nck(ncclResult_t r, const char * what) {
    if (r != ncclSuccess) throw std::runtime_error(std::string("tp: ") + what + ": " + ncclGetErrorString(r));
}
void full_io(int fd, void * p, size_t n, bool send) {
    auto * c = static_cast<char *>(p);
    while (n) {
        const ssize_t k = send ? ::send(fd, c, n, 0) : ::recv(fd, c, n, 0);
        if (k <= 0) throw std::runtime_error("tp: bootstrap socket closed");
        c += k; n -= size_t(k);
    }
}
}  // namespace

struct TpComm::Impl { ncclComm_t comm = nullptr; };

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
        ::close(fd); ::close(ls);
    } else {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port));
        if (inet_pton(AF_INET, peer.c_str(), &a.sin_addr) != 1) throw std::runtime_error("tp: bad peer address " + peer);
        int tries = 0;
        while (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a)) {
            if (++tries > 600) throw std::runtime_error("tp: cannot reach rank 0 at " + peer);
            usleep(100000);
        }
        full_io(fd, &id, sizeof id, false);
        ::close(fd);
    }
    nck(ncclCommInitRank(&impl_->comm, world, id, rank), "comm init");
}
TpComm::~TpComm() { if (impl_ && impl_->comm) ncclCommDestroy(impl_->comm); }
void TpComm::all_reduce_sum(float * x, size_t n, cudaStream_t s) { nck(ncclAllReduce(x, x, n, ncclFloat, ncclSum, impl_->comm, s), "all-reduce"); }
void TpComm::broadcast(void * buf, size_t bytes, int root, cudaStream_t s) { nck(ncclBroadcast(buf, buf, bytes, ncclUint8, root, impl_->comm, s), "broadcast"); }
void TpComm::all_gather(const void * send, void * recv, size_t bytes, cudaStream_t s) { nck(ncclAllGather(send, recv, bytes, ncclUint8, impl_->comm, s), "all-gather"); }
}  // namespace tqllm

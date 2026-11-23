#include "tp_comm.hpp"
#include "ctl_star.hpp"

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
#include <vector>

#ifndef TQLLM_COMMIT
#define TQLLM_COMMIT "dev"
#endif

namespace tqllm {
namespace {
void nck(ncclResult_t r, const char * what) {
    if (r != ncclSuccess) throw std::runtime_error(std::string("tp: ") + what + ": " + ncclGetErrorString(r));
}
}  // namespace

struct TpComm::Impl {
    ncclComm_t comm = nullptr;
    std::unique_ptr<CtlStar> ctl;
    // a rank that dies (killed, out of memory) must not leave the others spinning in a collective: the watchdog sees a
    // control connection hang up, aborts the communicator (the pending NCCL call returns an error) and closes the other
    // links, so every remaining rank sees it too
    std::atomic<bool> stop{false}, aborted{false};
    std::thread watchdog;
    void live() const { if (aborted) throw std::runtime_error("tp: another rank went away"); }
};

TpComm::TpComm(int rank, int world, const std::string & peer, int port) : impl_(std::make_unique<Impl>()), rank_(rank), world_(world) {
    if (world < 2 || world > kMaxRanks || rank < 0 || rank >= world)
        throw std::invalid_argument("tp: rank " + std::to_string(rank) + " of " + std::to_string(world) + " (2 .. " + std::to_string(kMaxRanks) + " ranks)");
    ncclUniqueId id{};
    std::vector<char> blob;
    if (rank == 0) {
        nck(ncclGetUniqueId(&id), "unique id");
        blob.assign(reinterpret_cast<const char *>(&id), reinterpret_cast<const char *>(&id) + sizeof id);
    }
    int nv = 0; ncclGetVersion(&nv);
    impl_->ctl = std::make_unique<CtlStar>(rank, world, peer, port, blob, std::string(TQLLM_COMMIT) + "/nccl" + std::to_string(nv));
    if (blob.size() != sizeof id) throw std::runtime_error("tp: bad bootstrap id from rank 0");
    std::memcpy(&id, blob.data(), sizeof id);
    nck(ncclCommInitRank(&impl_->comm, world, id, rank), "comm init");
    Impl * im = impl_.get();
    im->watchdog = std::thread([im] {
        const std::vector<int> fds = im->ctl->fds();
        std::vector<pollfd> p(fds.size());
        while (!im->stop) {
            for (size_t i = 0; i < fds.size(); ++i) p[i] = {fds[i], POLLRDHUP, 0};
            if (::poll(p.data(), p.size(), 1000) <= 0) continue;
            for (const auto & x : p)
                if (x.revents & (POLLRDHUP | POLLHUP | POLLERR)) {
                    if (im->stop) return;
                    std::fprintf(stderr, "tp: a rank went away; aborting the communicator\n");
                    im->aborted = true;
                    ncclCommAbort(im->comm);
                    im->comm = nullptr;
                    im->ctl->shutdown_all();
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
}
void TpComm::all_reduce_sum(float * x, size_t n, cudaStream_t s) { impl_->live(); nck(ncclAllReduce(x, x, n, ncclFloat, ncclSum, impl_->comm, s), "all-reduce"); }
void TpComm::broadcast(void * buf, size_t bytes, int root, cudaStream_t s) { impl_->live(); nck(ncclBroadcast(buf, buf, bytes, ncclUint8, root, impl_->comm, s), "broadcast"); }
void TpComm::all_gather(const void * send, void * recv, size_t bytes, cudaStream_t s) { impl_->live(); nck(ncclAllGather(send, recv, bytes, ncclUint8, impl_->comm, s), "all-gather"); }
void TpComm::send(const void * buf, size_t bytes, int peer, cudaStream_t s) { impl_->live(); nck(ncclSend(buf, bytes, ncclUint8, peer, impl_->comm, s), "send"); }
void TpComm::recv(void * buf, size_t bytes, int peer, cudaStream_t s) { impl_->live(); nck(ncclRecv(buf, bytes, ncclUint8, peer, impl_->comm, s), "recv"); }
bool TpComm::alive() const { return !impl_->aborted; }
void TpComm::group_begin() { nck(ncclGroupStart(), "group start"); }
void TpComm::group_end() { nck(ncclGroupEnd(), "group end"); }
void TpComm::ctl_send(const void * buf, size_t bytes) {
    if (rank_ == 0) impl_->ctl->fan_out(buf, bytes); else impl_->ctl->send(0, buf, bytes);
}
void TpComm::ctl_recv(void * buf, size_t bytes) {
    if (rank_ == 0 && world_ != 2) throw std::logic_error("tp: with more than two ranks rank 0 gathers (ctl_all_gather)");
    impl_->ctl->recv(rank_ == 0 ? 1 : 0, buf, bytes);
}
void TpComm::ctl_all_gather(const void * mine, void * all, size_t bytes) { impl_->ctl->all_gather(mine, all, bytes); }
int TpComm::local_ranks() const { return impl_->ctl->local_ranks(); }
}  // namespace tqllm

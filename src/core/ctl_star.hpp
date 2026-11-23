#pragma once
// The host control network of a multi-GPU run (one process per GPU, TpComm): rank 0 listens on a TCP port and every
// other rank connects to it (a star). The start-up handshake checks that all ranks agree on the world size and build,
// hands rank 0's bootstrap bytes (the NCCL unique id) to everyone and tells each rank how many ranks share its host.
// Then: point-to-point bytes between rank 0 and a rank, rank 0 to every rank, and an all-gather through rank 0.
// No CUDA / NCCL: tests/ctl_star.cpp runs three ranks in threads over loopback.
#include <cstddef>
#include <string>
#include <vector>

namespace tqllm {
class CtlStar {
public:
    // rank 0: `blob` is sent to every rank; others: `blob` receives it. `tag` (the build) must match on every rank.
    // `host` names this machine (empty: gethostname)
    CtlStar(int rank, int world, const std::string & peer, int port, std::vector<char> & blob, const std::string & tag,
            const std::string & host = "");
    ~CtlStar();
    CtlStar(const CtlStar &) = delete;
    CtlStar & operator=(const CtlStar &) = delete;
    int rank() const { return rank_; }
    int world() const { return world_; }
    int local_ranks() const { return local_; }  // ranks on this rank's host (itself included)
    // between rank 0 and rank `peer` (a non-zero rank talks to 0 only)
    void send(int peer, const void * p, size_t n);
    void recv(int peer, void * p, size_t n);
    void fan_out(const void * p, size_t n);  // rank 0: the same bytes to every other rank
    // every rank's `bytes` from `mine`, in rank order, into `all` (world x bytes) on every rank
    void all_gather(const void * mine, void * all, size_t bytes);
    std::vector<int> fds() const;  // the open control sockets (rank 0: one per other rank; others: the one to rank 0)
    void shutdown_all();           // wake every rank blocked on these sockets (a rank went away)
private:
    int fd_of(int peer) const;
    int rank_ = 0, world_ = 1, local_ = 1;
    std::vector<int> fd_;  // by rank (-1: none)
};
}  // namespace tqllm

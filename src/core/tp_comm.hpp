#pragma once
// Multi-GPU / multi-box ranks (one process per GPU): one NCCL communicator. Rank 0 makes the NCCL unique id and hands it
// to every other rank over TCP (peer = rank 0's address; src/core/ctl_star.hpp); those connections stay open as the host
// control channel.
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace tqllm {
class TpComm {
public:
    static constexpr int kMaxRanks = 8;
    // rank 0 listens on port; every other rank connects to peer:port
    TpComm(int rank, int world, const std::string & peer, int port);
    ~TpComm();
    TpComm(const TpComm &) = delete;
    TpComm & operator=(const TpComm &) = delete;
    int rank() const { return rank_; }
    int world() const { return world_; }
    void all_reduce_sum(float * x, size_t n, cudaStream_t s);
    void broadcast(void * buf, size_t bytes, int root, cudaStream_t s);
    // (max value, index) over ranks for each of n rows: vi [n][2] floats (value, index as float), replaced in place
    void all_gather(const void * send, void * recv, size_t bytes_per_rank, cudaStream_t s);
    // point to point (pipeline / layer split): rank `peer` posts the matching recv / send on its stream
    void send(const void * buf, size_t bytes, int peer, cudaStream_t s);
    void recv(void * buf, size_t bytes, int peer, cudaStream_t s);
    // the sends / recvs between group_begin and group_end go as one operation (both ranks post theirs in any order)
    void group_begin();
    bool alive() const;  // false once the other rank went away (the communicator was aborted)
    void group_end();
    // host control channel (the bootstrap TCP connections, kept open): rank 0's ctl_send goes to every other rank (layer
    // split: each engine call, src/engine/lockstep.hpp), another rank's to rank 0; ctl_recv is the reverse (rank 0
    // receives this way only with two ranks)
    void ctl_send(const void * buf, size_t bytes);
    void ctl_recv(void * buf, size_t bytes);
    // every rank's `bytes` in rank order into `all` (world x bytes), on every rank
    void ctl_all_gather(const void * mine, void * all, size_t bytes);
    int local_ranks() const;  // ranks on this machine (they share its RAM)
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int rank_ = 0, world_ = 1;
};
}  // namespace tqllm

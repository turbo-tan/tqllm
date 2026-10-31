#pragma once
// Two-box tensor parallelism (two GB10s linked by their ConnectX ports): one NCCL communicator over RoCE. Rank 0 makes
// the NCCL unique id and hands it to rank 1 over TCP (peer = rank 0's address on the linked network).
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace tqllm {
class TpComm {
public:
    // rank 0 listens on port; rank 1 connects to peer:port
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int rank_ = 0, world_ = 1;
};
}  // namespace tqllm

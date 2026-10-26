#include "tp_comm.hpp"
#include <stdexcept>
// built without NCCL (-DTQLLM_NCCL_DIR unset): tensor parallelism refuses at start
namespace tqllm {
struct TpComm::Impl {};
TpComm::TpComm(int, int, const std::string &, int) { throw std::runtime_error("tp: this build has no NCCL (configure with -DTQLLM_NCCL_DIR=...)"); }
TpComm::~TpComm() = default;
void TpComm::all_reduce_sum(float *, size_t, cudaStream_t) {}
void TpComm::broadcast(void *, size_t, int, cudaStream_t) {}
void TpComm::all_gather(const void *, void *, size_t, cudaStream_t) {}
}  // namespace tqllm

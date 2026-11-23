#pragma once
// Expert parallelism across two linked boxes (--split experts), for any MoE model.
//
// Both boxes hold every expert and serve their own users. While the second box (rank 1) has no request of its own it
// lends its GPU to the first (rank 0): for each MoE layer of a rank 0 decode forward, rank 0 sends the layer's input rows
// and routed expert ids, both ranks compute their half of the assignments (experts e % 2 == rank, plus rank 0 keeps what
// exceeds a fixed result cap), rank 1 sends its results back and rank 0 combines them in the one-box order. The output is
// bit-identical to one box (both ranks run the model's same compute callback on the same rows).
//
// Lease (control channel, ordered): rank 1 grants a lease while it is idle; under the lease rank 0 announces each forward
// and sends it without waiting for an answer. Before a request of its own, rank 1 revokes and waits for the lease end,
// which rank 0 sends in order after its last forward, so rank 1's own users never wait behind rank 0's work. A rank 1
// that goes away fails only the forward in flight; rank 0 then runs alone (every expert is resident there).
//
// A model provides an ExpertShard (sizes, bank readiness, one compute call); the engine calls, on rank 0, before_layer /
// exchange / after_layer around each MoE layer of a decode forward, and on rank 1, lock / request_begin / note_idle around
// its own requests. Pure parts (forward rules, lease protocol: expert_parallel_rules.hpp) are tested without a GPU: tests/expert_parallel.cpp.
#include "expert_parallel_rules.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm {

class TpComm;

// ---- the model's side ----

struct ExpertShard {
    int layers = 0, hidden = 0, topk = 0;
    int max_assignments = 1024;  // rows x topk of the largest forward exchanged
    // this layer may split: the bank holds every expert and its formats have the split kernels
    std::function<bool(int layer)> can_split;
    // rank 1: the layer's bank holds every expert; slot_of = expert -> bank slot (refreshed after this box's requests)
    std::function<bool(int layer, std::vector<int> & slot_of)> full_bank;
    // y [A][hidden] = the experts of the assignments whose local id >= 0, on rows x [A / topk][hidden]; slot_table maps
    // an expert to its bank slot (device). Called by both ranks; the model owns its scratch.
    std::function<void(int layer, const __half * x, const int * local_ids, int A, const int * slot_table,
                       __nv_bfloat16 * y, cudaStream_t)> compute;
};

// ---- the component ----

class ExpertParallel : public std::enable_shared_from_this<ExpertParallel> {
public:
    // made with std::make_shared; start() begins the link's reader (rank 0) or the helper (rank 1, after the engine loaded)
    ExpertParallel(std::shared_ptr<TpComm> link, ExpertShard shard);
    ~ExpertParallel();
    int rank() const;
    bool alive() const;
    int max_rows = 0;  // forwards of more rows run locally (0: no cap)

    // rank 0, around each MoE layer of a decode forward
    EpCall before_layer(int layer, int n, bool can);
    // exchange this layer: out [n * topk][hidden] gets every assignment's expert output (this rank's and rank 1's)
    void exchange(int layer, const __half * x, int n, const int * routed, const int * slot_table, __nv_bfloat16 * out, cudaStream_t s);
    void after_layer(int layer);
    void stop_peer();  // shutdown: rank 1 stops serving forwards

    // rank 1, around this box's own requests: hold lock() while serving; request_begin before taking it, note_idle after
    std::unique_lock<std::recursive_mutex> lock() { return std::unique_lock<std::recursive_mutex>(box_); }
    void request_begin() { if (giver_) giver_->request_begin(); }
    void note_idle() { if (giver_) giver_->note_idle(); }
    void close() { if (giver_) giver_->close(); }  // the engine is going away: lend no more
    // rank 0: the lease reader; rank 1: the helper, lending this box while idle() (no request of its own)
    void start(std::function<bool()> idle = nullptr);

private:
    void reader_loop();
    void serve_loop();
    bool refresh_tables();
    std::shared_ptr<TpComm> link_;
    ExpertShard shard_;
    std::unique_ptr<EpControl> ctl_;
    std::unique_ptr<EpLeaseHolder> holder_;  // rank 0
    std::unique_ptr<EpLeaseGiver> giver_;    // rank 1
    EpForwardState fwd_;
    std::recursive_mutex box_;               // rank 1: this box's serving vs the helper's forwards
    // device buffers sized for max_assignments
    int * local_ = nullptr, * ids_ = nullptr; __half * x_ = nullptr; __nv_bfloat16 * res_ = nullptr, * out_ = nullptr;
    std::vector<int *> tables_; std::vector<std::vector<int>> host_tables_;
    cudaStream_t stream_ = nullptr;
};

}  // namespace tqllm

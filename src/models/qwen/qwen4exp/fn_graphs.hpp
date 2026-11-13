#pragma once
// Flash-Next decode graphs: the part of a decode / verify step before the routing readback is captured each step and
// launched from an exec that is updated in place. Graphs belong to one DecodeModel (a request's sequence), keyed by its
// owner id (decode_rules.hpp next_graph_owner), and are released when the model ends: keyed by the model's address, a
// model built where a finished one had lived inherited its warm-up calls and execs, and every finished model's execs
// stayed (G2). Layers of one kind share an exec (every step updates all kernel parameters, weights included): one exec
// per layer and variant (~11 x 49) held ~1.3 GB the expert banks had taken. A shape keeps a few execs, one per launch in
// flight.
#include "decode_rules.hpp"
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace tqllm::flashnext {

class DecodeGraphs {
    struct Key { uint64_t owner; int kind, n; size_t nodes; bool operator<(const Key & o) const { return std::tie(owner, kind, n, nodes) < std::tie(o.owner, o.kind, o.n, o.nodes); } };
    struct Slot { cudaGraphExec_t exec = nullptr; cudaEvent_t done = nullptr; };
    GraphWarmup warm_;
    std::map<Key, std::vector<Slot>> pools_;
    static void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("decode graphs: ") + cudaGetErrorString(e)); }
public:
    static constexpr size_t kExecsPerShape = 8;
    // calls 1-2 of each (owner, layer, rows) run plain, later ones are captured (GraphWarmup)
    bool capture(uint64_t owner, int layer, int n, bool plain) { return warm_.capture(owner, layer, n, plain); }
    void reset_warmup() { warm_.clear(); }  // lazy scratch was released (a prefill): every key runs plain twice again
    // launches the captured graph cg (taken) on `stream` from an exec of this owner and shape that is not in flight
    void launch(uint64_t owner, int kind, int n, cudaGraph_t cg, cudaStream_t stream) {
        size_t nodes = 0; ck(cudaGraphGetNodes(cg, nullptr, &nodes));
        auto & pool = pools_[Key{owner, kind, n, nodes}];
        Slot * slot = nullptr;
        for (auto & g : pool) if (cudaEventQuery(g.done) == cudaSuccess) { slot = &g; break; }
        if (!slot && pool.size() < kExecsPerShape) { pool.push_back({}); slot = &pool.back(); ck(cudaEventCreateWithFlags(&slot->done, cudaEventDisableTiming)); }
        if (!slot) { slot = &pool.front(); ck(cudaEventSynchronize(slot->done)); }
        if (slot->exec) {
            cudaGraphExecUpdateResultInfo info;
            if (cudaGraphExecUpdate(slot->exec, cg, &info) != cudaSuccess) { cudaGetLastError(); ck(cudaGraphExecDestroy(slot->exec)); slot->exec = nullptr; }
        }
        if (!slot->exec) ck(cudaGraphInstantiate(&slot->exec, cg, 0));
        ck(cudaGraphDestroy(cg));
        ck(cudaGraphLaunch(slot->exec, stream));
        ck(cudaEventRecord(slot->done, stream));
    }
    // a finished model: its warm-up counts and execs go (a launch still queued finishes first)
    void release(uint64_t owner) {
        warm_.forget(owner);
        for (auto it = pools_.lower_bound(Key{owner, -(1 << 30), -(1 << 30), 0}); it != pools_.end() && it->first.owner == owner;) {
            for (auto & g : it->second) {
                if (g.done) { cudaEventSynchronize(g.done); cudaEventDestroy(g.done); }
                if (g.exec) cudaGraphExecDestroy(g.exec);
            }
            it = pools_.erase(it);
        }
    }
    size_t execs() const { size_t k = 0; for (const auto & [key, pool] : pools_) for (const auto & g : pool) k += g.exec != nullptr; return k; }
    size_t warm_entries() const { return warm_.size(); }
};

}  // namespace tqllm::flashnext

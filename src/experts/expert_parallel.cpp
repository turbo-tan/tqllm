#include "expert_parallel.hpp"

#include "core/tp_comm.hpp"
#include "moe_kernels.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(std::string("--split experts: ") + cudaGetErrorString(e)); }
template<class T> T * dalloc(size_t count) { void * p = nullptr; ck(cudaMalloc(&p, std::max<size_t>(count * sizeof(T), 256))); return static_cast<T *>(p); }

// the link's host control socket (the bootstrap TCP connection) as the lease channel
struct TpControl final : EpControl {
    explicit TpControl(TpComm & t) : tp(t) {}
    void send(const EpMsg & m) override { tp.ctl_send(&m, sizeof m); }
    EpMsg recv() override { EpMsg m; tp.ctl_recv(&m, sizeof m); return m; }
    TpComm & tp;
};
}  // namespace

// ---- the component ----

ExpertParallel::ExpertParallel(std::shared_ptr<TpComm> link, ExpertShard shard) : link_(std::move(link)), shard_(std::move(shard)) {
    if (!link_) throw std::invalid_argument("--split experts: no link");
    if (!shard_.compute || !shard_.can_split) throw std::invalid_argument("--split experts: the model gave no expert shard");
    ctl_ = std::make_unique<TpControl>(*link_);
    const size_t amax = size_t(shard_.max_assignments), h = size_t(shard_.hidden);
    local_ = dalloc<int>(amax); ids_ = dalloc<int>(amax);
    res_ = dalloc<__nv_bfloat16>(amax * h);
    if (rank() == 1) { x_ = dalloc<__half>(amax / size_t(std::max(shard_.topk, 1)) * h); out_ = dalloc<__nv_bfloat16>(amax * h); }
}
ExpertParallel::~ExpertParallel() {
    if (giver_) giver_->close();
    for (void * p : {static_cast<void *>(local_), static_cast<void *>(ids_), static_cast<void *>(x_), static_cast<void *>(res_), static_cast<void *>(out_)}) if (p) cudaFree(p);
    for (int * t : tables_) if (t) cudaFree(t);
    if (stream_) cudaStreamDestroy(stream_);
}
int ExpertParallel::rank() const { return link_->rank(); }
bool ExpertParallel::alive() const { return link_->alive(); }

void ExpertParallel::start(std::function<bool()> idle) {
    auto self = shared_from_this();  // the threads keep the component alive while they wait on the link
    if (rank() == 0) {
        holder_ = std::make_unique<EpLeaseHolder>(*ctl_);
        std::thread([self] { self->reader_loop(); }).detach();
        return;
    }
    for (int l = 0; l < shard_.layers; ++l)
        if (!shard_.can_split(l)) throw std::runtime_error("--split experts: layer " + std::to_string(l) + " has an expert format without the split kernels");
    ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    tables_.assign(size_t(shard_.layers), nullptr); host_tables_.assign(size_t(shard_.layers), {});
    giver_ = std::make_unique<EpLeaseGiver>(*ctl_, idle ? std::move(idle) : [] { return true; }, [this] { return refresh_tables(); });
    std::thread([self] {
        try { self->serve_loop(); }
        catch (const std::exception & e) { std::fprintf(stderr, "tqllm: --split experts rank 1 helper stopped: %s\n", e.what()); }
        self->giver_->lease_ended();  // a request of this box waiting for the lease goes on
    }).detach();
}

// ---- rank 0 ----

EpCall ExpertParallel::before_layer(int layer, int n, bool bank_full) {
    const bool was_off = fwd_.off;
    const bool can = bank_full && shard_.can_split(layer) && (max_rows <= 0 || n <= max_rows);
    const auto call = ep_before_layer(fwd_, layer, n, can, link_->alive(), [&](int rows) { return holder_ && holder_->announce(rows); });
    if (fwd_.off && !was_off)
        std::fprintf(stderr, "tqllm: --split experts off: %s; every layer runs locally\n", link_->alive() ? "a forward was left open" : "rank 1 went away");
    if (call == EpCall::Fail) throw std::runtime_error("--split experts: the forward lost its exchanges with rank 1 at layer " + std::to_string(layer));
    return call;
}

void ExpertParallel::exchange(int layer, const __half * x, int n, const int * routed, const int * slot_table, __nv_bfloat16 * out, cudaStream_t s) {
    const int A = n * shard_.topk, cap = ep_result_cap(A), h = shard_.hidden;
    if (A > shard_.max_assignments) throw std::runtime_error("--split experts: forward of " + std::to_string(A) + " assignments over the buffers");
    link_->group_begin();
    link_->send(x, size_t(n) * h * sizeof(__half), 1, s); link_->send(routed, size_t(A) * sizeof(int), 1, s);
    link_->group_end();
    tqllm_cuda::ep_local_ids(routed, A, nullptr, 0, 0, cap, local_, s);  // this rank's half + what exceeds the cap
    shard_.compute(layer, x, local_, A, slot_table, out, s);
    if (cap > 0) { link_->recv(res_, size_t(cap) * h * sizeof(__nv_bfloat16), 1, s); tqllm_cuda::ep_scatter(res_, routed, A, 0, cap, out, h, s); }
    if (layer + 1 == shard_.layers) {  // the forward's last exchange: a rank 1 that went away during it fails the request
        ck(cudaStreamSynchronize(s));
        if (!link_->alive()) throw std::runtime_error("--split experts: rank 1 went away during a forward");
    }
}

void ExpertParallel::after_layer(int layer) { ep_after_layer(fwd_, layer, shard_.layers); }
void ExpertParallel::stop_peer() { if (holder_) { try { holder_->stop(); } catch (const std::exception &) {} } }

void ExpertParallel::reader_loop() {
    for (;;) {
        EpMsg m;
        try { m = ctl_->recv(); } catch (const std::exception &) { holder_->lost(); return; }  // rank 1 went away
        if (!holder_->on_message(m)) return;
    }
}

// ---- rank 1 ----

bool ExpertParallel::refresh_tables() {  // caller holds box_: every bank full, device slot tables current
    std::vector<int> slot_of;
    for (int l = 0; l < shard_.layers; ++l) {
        if (!shard_.full_bank(l, slot_of)) return false;
        auto & dev = tables_[size_t(l)]; auto & host = host_tables_[size_t(l)];
        if (!dev) dev = dalloc<int>(slot_of.size());
        if (host != slot_of) {
            host = slot_of;
            ck(cudaMemcpyAsync(dev, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
            ck(cudaStreamSynchronize(stream_));
        }
    }
    return true;
}

void ExpertParallel::serve_loop() {
    const int h = shard_.hidden, nmax = shard_.max_assignments / std::max(shard_.topk, 1);
    std::fprintf(stderr, "tqllm: --split experts rank 1: computing experts e %% 2 == 1 for rank 0 when this box is idle\n");
    { auto lk = lock(); giver_->note_idle(); }
    uint64_t forwards = 0;
    for (;;) {
        EpMsg m;
        try { m = ctl_->recv(); } catch (const std::exception &) { break; }  // rank 0 went away
        if (m.kind == kEpStop) break;
        if (m.kind == kEpLeaseEnd) { giver_->lease_ended(); continue; }
        if (m.kind != kEpForward || m.n <= 0 || m.n > nmax) { std::fprintf(stderr, "tqllm: --split experts rank 1: bad forward header\n"); break; }
        const int n = m.n, A = n * shard_.topk, cap = ep_result_cap(A);
        // a forward under the lease: this box has no request of its own (its next one waits in request_begin)
        auto lk = lock();
        for (int layer = 0; layer < shard_.layers; ++layer) {
            link_->group_begin(); link_->recv(x_, size_t(n) * h * sizeof(__half), 0, stream_); link_->recv(ids_, size_t(A) * sizeof(int), 0, stream_); link_->group_end();
            tqllm_cuda::ep_local_ids(nullptr, 0, ids_, A, 1, cap, local_, stream_);
            shard_.compute(layer, x_, local_, A, tables_[size_t(layer)], out_, stream_);
            if (cap > 0) { tqllm_cuda::ep_gather(out_, 0, ids_, A, 1, cap, res_, h, stream_); link_->send(res_, size_t(cap) * h * sizeof(__nv_bfloat16), 0, stream_); }
        }
        ck(cudaStreamSynchronize(stream_));  // the forward's work is done before this box's own requests run
        if (++forwards % 1000 == 0) std::fprintf(stderr, "tqllm: --split experts rank 1: %llu forwards computed\n", (unsigned long long) forwards);
    }
    std::fprintf(stderr, "tqllm: --split experts rank 1: rank 0 stopped after %llu forwards computed\n", (unsigned long long) forwards);
}

}  // namespace tqllm

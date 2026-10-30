#pragma once
// Flash-Next per-sequence state: GDN recurrent state, attention K/V buffers and QSA indexer keys.
#include "fn_checkpoint.hpp"
#include "decode_rules.hpp"
struct RecurrentState {
    std::unique_ptr<Device<float>> history, state;
    // speculative mode: state after every token, the block's conv inputs and the history before it (rollback)
    bool spec = false; int last_n = 0; std::unique_ptr<Device<float>> qkv_saved, old_history; std::unique_ptr<Device<float>> state2, history2;  // double buffers
    // TQFN_GDN_REPLAY: keep the block's GDN inputs; accept(m) reruns the first m tokens from state2 (the pre-block state)
    std::unique_ptr<Device<float>> conv_saved, a_saved, b_saved, replay_out;
    const float * neg_a = nullptr, * bias = nullptr; int kh = 0, vh = 0, dim = 0, channels = 0; float eps = 0.f;
};
// Attention K or V rows in one reserved address range, mapped page by page to device or pinned host memory (CUDA VMM,
// the 27B engine's KvMemory): after a long prompt the oldest pages move to the host at the same addresses, so the kernels
// keep one pointer and decode reads the selected host rows over PCIe; the freed device memory goes back to expert banks.
struct KvBuf;
// Host K/V on (decode_rules.hpp kv_host_on): TQFN_KV_HOST or the default for this GPU, decided once
inline bool kv_host_enabled();
struct KvBuf {
    __nv_bfloat16 * p = nullptr; size_t bytes = 0, page = 0, span = 0; std::vector<uint8_t> on_host; bool vmm = false;
    static tqllm::KvMemory & mem() { static auto m = tqllm::make_cuda_kv_memory(nullptr); return *m; }
    static void * staging(size_t n) {  // one pinned page for the moves
        static void * s = nullptr; static size_t cap = 0;
        if (cap < n) { if (s) cudaFreeHost(s); ck(cudaMallocHost(&s, n)); cap = n; }
        return s;
    }
    char * at(size_t i) const { return reinterpret_cast<char *>(p) + i * page; }
    explicit KvBuf(size_t n) : bytes(n) {
        auto & m = mem(); const size_t g = m.granule();
        // host K/V (kv_host_enabled: discrete GPUs by default): VMM pages, so old ones can move to host memory; else
        // plain allocations
        if (g && kv_host_enabled()) {
            page = tqllm::flashnext::kv_page_bytes(g); span = (n + page - 1) / page * page;
            if (void * va = m.reserve(span)) {
                p = static_cast<__nv_bfloat16 *>(va); vmm = true; on_host.assign(span / page, 0);
                for (size_t i = 0; i < on_host.size(); ++i)
                    if (!m.map(at(i), page, tqllm::KvTier::device)) { m.unreserve(p, span); throw std::runtime_error("out of memory: attention KV page"); }
                return;
            }
        }
        ck(cudaMalloc(reinterpret_cast<void **>(&p), n));
    }
    ~KvBuf() { if (vmm) mem().unreserve(p, span); else if (p) cudaFree(p); }
    KvBuf(const KvBuf &) = delete; KvBuf & operator=(const KvBuf &) = delete;
    // page i to the other tier with its data; false (data and tier unchanged) when the target tier is out of memory
    bool move(size_t i, bool to_host) {
        if (!vmm || bool(on_host[i]) == to_host) return true;
        auto & m = mem(); char * a = at(i); void * st = staging(page);
        ck(cudaMemcpy(st, a, page, cudaMemcpyDefault));
        m.unmap(a, page);
        const auto want = to_host ? tqllm::KvTier::host : tqllm::KvTier::device, back = to_host ? tqllm::KvTier::device : tqllm::KvTier::host;
        const bool ok = m.map(a, page, want);
        if (!ok && !m.map(a, page, back)) throw std::runtime_error("attention KV page lost (both tiers out of memory)");
        ck(cudaMemcpy(a, st, page, cudaMemcpyDefault));
        if (ok) on_host[i] = to_host;
        return ok;
    }
    size_t page_rows(size_t row) const { return page / row; }
    bool any_host() const { for (auto h : on_host) if (h) return true; return false; }
    // rows (of row_bytes) of the leading pages in host memory: kv_to_host moves the oldest positions
    int host_rows(size_t row_bytes) const {
        size_t n = 0; while (n < on_host.size() && on_host[n]) ++n;
        return int(n * page / row_bytes);
    }
};
inline bool kv_host_enabled() {
    static const bool on = [] {
        int dev = 0, unified = 0; cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&unified, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) != cudaSuccess) unified = 0;
        return tqllm::flashnext::kv_host_on(std::getenv("TQFN_KV_HOST"), KvBuf::mem().granule() != 0, unified != 0);
    }();
    return on;
}
// The K/V format of the attention layers (src/formats/kv): TQFN_KV=<name>, default bf16. Flash-Next attends sparsely, so
// the format must implement attend_selected().
inline const tqllm::KvFormat & kv_fmt() {
    static const tqllm::KvFormat * f = [] {
        const char * n = std::getenv("TQFN_KV");
        const tqllm::KvFormat * x = tqllm::kv_format(n && *n ? n : "bf16");
        if (!x->selected_ok()) throw std::runtime_error(std::string("Flash-Next: KV format ") + x->name() + " has no selected-position attention");
        return x;
    }();
    return *f;
}
struct AttentionState { std::unique_ptr<KvBuf> k, v; int len = 0;
                        std::unique_ptr<tqllm_cuda::QsaIndexer> qsa; };  // QSA indexer keys (beyond the indexer budget)
// pre: the row-wise projections of this sequence's rows were done for a whole batch (qkv, z, alpha, beta slices);
// raw_out: write the recurrence output there and stop before the gated norm and the output projection (also batched)

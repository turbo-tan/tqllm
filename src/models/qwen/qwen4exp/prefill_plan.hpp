#pragma once
// Flash-Next layer-major prefill: the memory plan and the bank ledger.
//
// PrefillPlan is a pure function of the request and the engine parameters: how many device bytes each prefill buffer
// takes, what the chunk's peak is, and how many expert banks the request must borrow (release) before its first kernel.
// BankLedger owns the borrow: which banks were released, in which state the request is, and that the refill happens
// only after the prefill buffers are gone. Both are CUDA-free so tests/prefill_plan.cpp can check them on any machine.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

struct PrefillShape {
    int hidden = 0, hc = 0, expert_ff = 0, topk = 0, experts = 0, layers = 0;
    size_t expert_block_bytes = 0;  // one staged expert (gate + up + down, data + aux) in the bank layout
};

struct PrefillTuning {
    int chunk = 8192, sub = 2048, stage = 192;
    size_t arena_bytes = 0, dense_cache_bytes = 0;  // --prefill-arena-mb, --dense-cache-mb
    size_t workspace_bytes = size_t(1024) << 20;    // attention / GDN / QSA scratch the kernels allocate on first use
    size_t mtp_bank_bytes = 0;                      // the MTP layer's expert bank, built in the catch-up after the trunk
    size_t context_bytes = 0;                       // prefill_context_bytes(): the attention state the request allocates at first use
    size_t context_bytes_per_token = 0;             // extra per-token headroom on top of it (0: none)         // attention / indexer state that grows with the prompt, not the chunk (the retired
                                                    // heuristic reserved 60 MB per 1K tokens for the same reason; 128K failed at 4 and 16 KB/token;
                                                    // 124 MiB allocation failed with 87 MiB free on 10-17 when the plan had no such term)
};

// Device bytes of one prefill chunk of n tokens (the per-chunk buffers of DecodeModel::prefill_layer and its staging).
struct PrefillChunkBytes {
    size_t tokens = 0, activations = 0, expert_io = 0, staging = 0, residual = 0, arena = 0, dense_cache = 0, workspace = 0;
    size_t total() const { return activations + expert_io + staging + residual + arena + dense_cache + workspace; }
};

inline size_t align256(size_t b) { return (b + 255) / 256 * 256; }
// Hyper-connection norm of a chunk: more than 64 tokens keep only row scales and a BF16 image (the mix recomputes the
// normalized rows), valid only when the mixer projections reading them (down, inject) are BF16 and so take that image;
// a quantized down / inject reads the fp32 rows.
inline bool hc_row_scales_only(int tokens, bool down_bf16, bool inject_bf16) { return tokens > 64 && down_bf16 && inject_bf16; }
// Expert slots per layer from free device memory (slots=auto): what is free with no expert bank, less the room the
// context and decode keep, over the bytes one slot takes in every layer; at least 1 (the fit check then reports), at
// most every expert.
// slots=auto builds the warm-up banks at this provisional size and sizes them after every other buffer (the MTP bank,
// staging, the dense weights) is in place: a large start (64) left a 12 GB card no room for the MTP bank and the warm-up
// failed before the sizing ran
constexpr int kAutoStartSlots = 8;
inline int auto_expert_slots(size_t free_bytes, size_t keep_bytes, size_t slot_bytes_all_layers, int experts) {
    if (!slot_bytes_all_layers) return experts;
    if (free_bytes <= keep_bytes) return 1;
    return int(std::clamp<size_t>((free_bytes - keep_bytes) / slot_bytes_all_layers, 1, size_t(experts)));
}
// the prefill scratch arena a chunk needs (per-layer temporaries of the dense sub-chunks and attention, without the QSA
// block scores the caller adds; measured 10-17: 2880 MB high-water at 16384 tokens with 2048-query blocks, ~2340 MB
// without their scores; 3519+ MB short at 32768 with 3800). 0.15 MB per token + 128 MB; an overflow falls back to the pool.
inline size_t prefill_arena_bytes(int chunk) { return (size_t(std::max(0, chunk)) * 154 / 1024 + 128) << 20; }
// The attention state a request allocates lazily during its first chunk, all sized by the context: per attention layer
// the bf16 K and V rows, the indexer's raw keys (fp32) and pooled scores; plus one QSA batch scratch shared by the layers
// (sort keys and sorted keys for sub-chunk queries x context/ratio blocks, CUB temp of the same order, selected positions).
inline size_t prefill_context_bytes(int context, int sub, int attention_layers, int kv_heads, int head_dim, int index_dim, int ratio = 4, int top_k = 2048) {
    const size_t ctx = size_t(std::max(0, context)), blocks = ctx / size_t(std::max(1, ratio));
    const size_t per_layer = 2 * ctx * size_t(kv_heads) * size_t(head_dim) * 2 + ctx * size_t(index_dim) * 4 + blocks * 4;
    const size_t items = size_t(std::max(16, sub)) * blocks, width = size_t(std::min<int64_t>(int64_t(ctx), int64_t(top_k) + ratio - 1));
    // lean scratch (QsaBatchScratch::lean: top-k only): 32-bit score bits + 1024 sorted keys per query; else sort keys,
    // sorted keys and CUB temp of the same order
    const bool lean = (width - 1) / size_t(std::max(1, ratio)) + 1 <= 1024 && blocks >= 1024;
    const size_t sel = lean ? items * 4 + size_t(std::max(16, sub)) * 1024 * 8 : items * 8 * 2 + items * 16;
    const size_t scratch = sel + size_t(std::max(16, sub)) * (width * 5 + 4 * 128 * 4);
    return size_t(std::max(0, attention_layers)) * per_layer + scratch;
}

inline PrefillChunkBytes prefill_chunk_bytes(const PrefillShape & s, const PrefillTuning & t, int prompt_tokens) {
    if (s.hidden <= 0 || s.hc <= 0 || s.expert_ff <= 0 || s.topk <= 0 || t.chunk < 256 || t.stage < 1)
        throw std::invalid_argument("prefill_chunk_bytes: invalid shape or tuning");
    PrefillChunkBytes b;
    const size_t n = size_t(std::min(prompt_tokens, t.chunk)), A = n * size_t(s.topk), h = size_t(s.hidden), ff = size_t(s.expert_ff);
    b.tokens = n;
    // pf_inject f32 [n hc], pf_weights f32 [A], pf_sw f32 [n], pf_xr half [n h], pf_sd f32 [n h], pf_routed int [A],
    // pf_idx int (pass tables, bounded by 8 A), q8_1 tiles of the gate/up input (144 bytes per 128 values)
    b.activations = n * size_t(s.hc) * 4 + A * 4 + n * 4 + n * h * 2 + n * h * 4 + A * 4 + 8 * A * 4 + n * ((h + 127) / 128) * 144;
    // pf_eg, pf_eu bf16 [A ff], pf_ax half [A ff], pf_ed bf16 [A h]
    b.expert_io = 3 * A * ff * 2 + A * h * 2;
    b.staging = 2 * align256(size_t(t.stage) * s.expert_block_bytes);
    b.residual = n * h * size_t(s.hc) * 4;  // res_keep: the chunk's wide residual rows
    b.arena = std::max(t.arena_bytes, prefill_arena_bytes(t.chunk)); b.dense_cache = t.dense_cache_bytes; b.workspace = t.workspace_bytes;
    return b;
}

// The whole request: the trunk's chunk peak, the MTP catch-up that follows (trunk buffers released first), and how many
// bank bytes must be released before the trunk starts.
struct PrefillPlan {
    PrefillChunkBytes chunk;
    int chunks = 0;
    size_t trunk_peak = 0, catchup_peak = 0;
    size_t need = 0;         // device bytes the request must have free before its first kernel (peak + the prompt-length term)
    size_t release = 0;      // bank bytes to release (0: fits in the free memory as is)
    size_t host_copies = 0;  // expert bytes streamed from host over the whole prompt (all experts once per layer-chunk)
};

enum class BorrowMode { Auto, All, Count };

inline PrefillPlan prefill_plan(const PrefillShape & s, const PrefillTuning & t, int prompt_tokens, size_t free_bytes) {
    if (prompt_tokens <= 0) throw std::invalid_argument("prefill_plan: no tokens");
    PrefillPlan p;
    p.chunk = prefill_chunk_bytes(s, t, prompt_tokens);
    p.chunks = (prompt_tokens + t.chunk - 1) / t.chunk;
    p.trunk_peak = p.chunk.total();
    // the catch-up runs the MTP layer over the prompt in 1024-position pieces: its bank, one piece of residual rows, staging
    p.catchup_peak = t.mtp_bank_bytes + size_t(std::min(prompt_tokens, 1024)) * size_t(s.hidden) * size_t(s.hc) * 4 +
                     p.chunk.staging + p.chunk.arena + t.workspace_bytes;
    p.need = std::max(p.trunk_peak, p.catchup_peak) + t.context_bytes + size_t(prompt_tokens) * t.context_bytes_per_token;
    p.release = p.need > free_bytes ? p.need - free_bytes : 0;
    p.host_copies = size_t(p.chunks) * size_t(s.layers) * size_t(s.experts) * s.expert_block_bytes;
    return p;
}

// How many banks a plan borrows: all of them, a fixed count, or the fewest (least recently used first) that cover the plan.
inline int banks_to_borrow(const PrefillPlan & p, const std::vector<size_t> & bank_bytes_lru, BorrowMode mode, int count = 0) {
    const int n = int(bank_bytes_lru.size());
    if (mode == BorrowMode::All) return n;
    if (mode == BorrowMode::Count) return std::clamp(count, 0, n);
    size_t got = 0; int k = 0;
    while (got < p.release && k < n) got += bank_bytes_lru[size_t(k++)];
    if (got < p.release) throw std::runtime_error("prefill plan: " + std::to_string((p.release - got) >> 20) + " MB short with every bank released (" + std::to_string(p.release >> 20) + " MB to release, " + std::to_string(got >> 20) + " MB of banks)");
    return k;
}

// The borrow of one request. The ledger does not touch CUDA: the engine supplies release / refill / free callbacks
// (tests supply a mock). Order enforced: borrow -> trunk -> buffers released -> catch-up -> refill. A refill before
// the buffers are released is the bug of 2026-10-16 (OOM half way through the refill) and throws here.
class BankLedger {
public:
    struct Ops {
        std::function<std::vector<int>()> lru_layers;        // layers with a bank, least recently used first
        std::function<size_t(int)> bank_bytes;               // device bytes of that layer's bank
        std::function<void(int)> release;                    // free the bank
        std::function<void(int)> refill;                     // rebuild it (may throw: out of memory)
        std::function<size_t()> free_bytes;                  // device bytes free now
    };
    enum class State { Idle, Borrowed, BuffersReleased, Refilled };

    explicit BankLedger(Ops ops) : ops_(std::move(ops)) {}

    // Releases banks for the plan; returns the layers released (prefill streams all of their experts). With mode All the
    // layers that have no bank (released by regrow at the request start) join them: the refill after the prompt rebuilds
    // every layer from one free-memory share, instead of some banks built mid-prompt at whatever the history left free
    // (12 GB: the same 16K prompt twice gave different text).
    std::vector<int> borrow(const PrefillPlan & plan, BorrowMode mode, int count = 0, const std::vector<int> & bankless = {}) {
        if (state_ != State::Idle && state_ != State::Refilled) throw std::logic_error("BankLedger::borrow: previous request not refilled");
        const auto lru = ops_.lru_layers();
        std::vector<size_t> bytes; for (int l : lru) bytes.push_back(ops_.bank_bytes(l));
        const int k = banks_to_borrow(plan, bytes, mode, count);
        borrowed_.assign(lru.begin(), lru.begin() + k); borrowed_bytes_ = 0;
        for (int l : borrowed_) { borrowed_bytes_ += ops_.bank_bytes(l); ops_.release(l); }
        // a release frees a little less than the bank's bytes (allocator rounding, pool blocks): when the plan still does
        // not fit, release the next least recently used bank instead of failing a few MB short (GB10: 12789 MB needed,
        // 12785 MB free after 2 banks)
        size_t next = size_t(k);
        while (ops_.free_bytes() < plan.need && mode != BorrowMode::Count && next < lru.size()) {
            const int l = lru[next++]; borrowed_.push_back(l); borrowed_bytes_ += ops_.bank_bytes(l); ops_.release(l);
        }
        if (mode == BorrowMode::All) for (int l : bankless) if (!borrowed(l)) borrowed_.push_back(l);
        if (ops_.free_bytes() < plan.need)
            throw std::runtime_error("prefill plan: " + std::to_string(plan.need >> 20) + " MB needed, " +
                                     std::to_string(ops_.free_bytes() >> 20) + " MB free after releasing " + std::to_string(next) + " banks");
        state_ = State::Borrowed;
        return borrowed_;
    }
    bool borrowed(int layer) const { return std::find(borrowed_.begin(), borrowed_.end(), layer) != borrowed_.end(); }
    const std::vector<int> & borrowed_layers() const { return borrowed_; }
    size_t borrowed_bytes() const { return borrowed_bytes_; }

    // The engine calls this once the trunk buffers (and the pool's cached blocks) are gone.
    void buffers_released() {
        if (state_ != State::Borrowed) throw std::logic_error("BankLedger::buffers_released: nothing borrowed");
        state_ = State::BuffersReleased;
    }
    // Rebuilds the borrowed banks; returns how many came back this call. With keep_free > 0 it stops before the free
    // memory would drop below keep_free (an overlapped refill beside the MTP catch-up) and leaves the rest for the next
    // call; without it, a refill that runs out of memory stops and the rest refill on their first use in decode.
    int refill(size_t keep_free = 0) {
        if (state_ == State::Idle || borrowed_.empty()) { state_ = State::Refilled; return 0; }
        if (state_ != State::BuffersReleased) throw std::logic_error("BankLedger::refill: prefill buffers still held");
        int done = 0; size_t i = 0;
        for (; i < borrowed_.size(); ++i) {
            const int l = borrowed_[i];
            if (keep_free && ops_.free_bytes() < keep_free + ops_.bank_bytes(l)) break;
            try { ops_.refill(l); ++done; } catch (const std::exception &) { i = borrowed_.size(); break; }
        }
        refilled_ += done;
        if (keep_free && i < borrowed_.size()) { borrowed_.erase(borrowed_.begin(), borrowed_.begin() + long(i)); return done; }
        borrowed_.clear(); borrowed_bytes_ = 0; state_ = State::Refilled;
        return done;
    }
    size_t borrowed_total() const { return borrowed_.size() + refilled_; }  // released this request (refilled or not yet)
    State state() const { return state_; }

private:
    Ops ops_; std::vector<int> borrowed_; size_t borrowed_bytes_ = 0, refilled_ = 0; State state_ = State::Idle;
};

}  // namespace tqllm

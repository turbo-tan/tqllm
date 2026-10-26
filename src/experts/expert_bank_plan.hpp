#pragma once
// Expert banks: turning device bytes into per-layer slot counts, and borrowing bank bytes for a prefill.
//
// Both engines that keep expert banks (Flash-Next, GLM-5.3-Flash) need the same two things: how many experts of each
// layer fit in the device memory that is left, and which banks a layer-major prefill may borrow. This is that arithmetic,
// moved out of src/models/qwen/qwen4exp/prefill_plan.hpp so a model does not include another model. CUDA-free, so
// tests/expert_bank_plan.cpp and tests/prefill_plan.cpp check it on any machine.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

// Expert slots per layer from free device memory (slots=auto): what is free with no expert bank, less the room the
// context and decode keep, over the bytes one slot takes in every layer; at least 1 (the fit check then reports), at
// most every expert.
// slots=auto builds the warm-up banks at this provisional size and sizes them after every other buffer is in place
constexpr int kAutoStartSlots = 8;
inline int auto_expert_slots(size_t free_bytes, size_t keep_bytes, size_t slot_bytes_all_layers, int experts) {
    if (!slot_bytes_all_layers) return experts;
    if (free_bytes <= keep_bytes) return 1;
    return int(std::clamp<size_t>((free_bytes - keep_bytes) / slot_bytes_all_layers, 1, size_t(experts)));
}

// Per-layer caps written for one model (TQFN_LAYER_SLOTS, or a "<experts>.slots" file next to the packed experts: one line
// of comma-separated counts, fitted on a routing trace of other prompts). Counts fitted for another slot count are scaled
// to keep the same total. Empty result: no override.
inline std::vector<int> parse_layer_slots(const std::string & text, int slots, int experts) {
    std::vector<int> caps;
    if (text.empty()) return caps;
    size_t at = 0;
    while (at <= text.size()) {
        const size_t e = std::min(text.find(',', at), text.size());
        caps.push_back(std::stoi(text.substr(at, e - at)));
        at = e + 1;
    }
    long total = 0;
    for (int c : caps) total += c;
    const long want = long(slots) * long(caps.size());
    if (total != want && total > 0)
        for (int & c : caps) c = std::max(1, std::min(experts, int(std::lround(double(c) * double(want) / double(total)))));
    return caps;
}
// the fitted caps of a model: the environment wins over the file beside its packed experts; empty: neither
inline std::string layer_slots_text(const char * env, const std::string & experts_path) {
    if (const char * v = env ? std::getenv(env) : nullptr; v && *v) return v;
    if (experts_path.empty()) return {};
    std::string line;
    std::ifstream f(experts_path + ".slots");
    return (f && std::getline(f, line)) ? line : std::string();
}

// One layer's expert bank sized from a device byte budget.
struct BankSlotPlan {
    std::vector<int> slots;  // per layer: experts resident in that layer's bank (0: no bank there)
    size_t bytes = 0;        // device bytes the plan takes
    bool fits = true;        // false: the budget does not even hold every layer's floor
};

// Per-layer slots from a byte budget. `slot_bytes[l]` is one slot of layer l in the bank row layout (all three
// projections; 0: that layer has no bank, e.g. a dense FFN). `spread[l]` is how far that layer's routing reaches - the
// experts it touches over the window the plan is for; a layer that spreads over more experts needs more slots, and 0
// means "not measured yet" (treated as equal for every banked layer). Every banked layer starts at `min_slots` (the
// model's top-k: one token's experts must fit) and the rest of the budget goes out one slot at a time to the layer with
// the best marginal value per byte, spread[l] / (slots[l] * slot_bytes[l]): slots end up proportional to
// spread / slot_bytes, so a hot layer gets more slots than a cold one and an expensive layer fewer than a cheap one.
// Stops at the budget, or at `experts` slots in every layer.
inline BankSlotPlan expert_layer_slots(size_t budget_bytes, const std::vector<size_t> & slot_bytes,
                                       const std::vector<double> & spread, int min_slots, int experts) {
    if (experts <= 0 || min_slots < 0) throw std::invalid_argument("expert_layer_slots: no experts or negative floor");
    if (slot_bytes.size() != spread.size()) throw std::invalid_argument("expert_layer_slots: spread per layer expected");
    if (min_slots > experts) throw std::invalid_argument("expert_layer_slots: floor above the expert count");
    BankSlotPlan p;
    p.slots.assign(slot_bytes.size(), 0);
    std::vector<char> banked(slot_bytes.size(), 0);
    for (size_t l = 0; l < slot_bytes.size(); ++l) {
        if (!slot_bytes[l] || slot_bytes[l] > budget_bytes) continue;
        banked[l] = 1;
        if (p.bytes + slot_bytes[l] * size_t(min_slots) <= budget_bytes) {
            p.slots[l] = min_slots;
            p.bytes += slot_bytes[l] * size_t(min_slots);
        } else
            p.fits = false;
    }
    if (!p.fits) return p;
    for (;;) {
        int best = -1;
        double bv = 0;
        for (size_t l = 0; l < slot_bytes.size(); ++l) {
            if (!banked[l] || p.slots[l] >= experts || p.bytes + slot_bytes[l] > budget_bytes) continue;
            const double w = spread[l] > 0 ? spread[l] : 1.0;
            const double v = w / (double(p.slots[l]) * double(slot_bytes[l]));
            if (best < 0 || v > bv || (v == bv && slot_bytes[l] < slot_bytes[size_t(best)])) { best = int(l); bv = v; }
        }
        if (best < 0) break;
        p.slots[size_t(best)] += 1;
        p.bytes += slot_bytes[size_t(best)];
    }
    return p;
}

enum class BorrowMode { Auto, All, Count };

// How many banks a prefill borrows: all of them, a fixed count, or the fewest (least recently used first) that cover
// `release_bytes` of device memory.
inline int banks_to_borrow(size_t release_bytes, const std::vector<size_t> & bank_bytes_lru, BorrowMode mode, int count = 0) {
    const int n = int(bank_bytes_lru.size());
    if (mode == BorrowMode::All) return n;
    if (mode == BorrowMode::Count) return std::clamp(count, 0, n);
    size_t got = 0;
    int k = 0;
    while (got < release_bytes && k < n) got += bank_bytes_lru[size_t(k++)];
    if (got < release_bytes) throw std::runtime_error("prefill plan: " + std::to_string((release_bytes - got) >> 20) + " MB short with every bank released (" + std::to_string(release_bytes >> 20) + " MB to release, " + std::to_string(got >> 20) + " MB of banks)");
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

    // Releases banks until `release_bytes` of device memory came back; returns the layers released (prefill streams all
    // of their experts). With mode All the layers that have no bank (released by regrow at the request start) join them:
    // the refill after the prompt rebuilds every layer from one free-memory share, instead of some banks built mid-prompt
    // at whatever the history left free (12 GB: the same 16K prompt twice gave different text).
    std::vector<int> borrow(size_t release_bytes, BorrowMode mode, int count = 0, size_t need_bytes = 0,
                            const std::vector<int> & bankless = {}) {
        if (state_ != State::Idle && state_ != State::Refilled) throw std::logic_error("BankLedger::borrow: previous request not refilled");
        const auto lru = ops_.lru_layers();
        std::vector<size_t> bytes; for (int l : lru) bytes.push_back(ops_.bank_bytes(l));
        const int k = banks_to_borrow(release_bytes, bytes, mode, count);
        borrowed_.assign(lru.begin(), lru.begin() + k); borrowed_bytes_ = 0;
        for (int l : borrowed_) { borrowed_bytes_ += ops_.bank_bytes(l); ops_.release(l); }
        // a release frees a little less than the bank's bytes (allocator rounding, pool blocks): when the plan still does
        // not fit, release the next least recently used bank instead of failing a few MB short (GB10: 12789 MB needed,
        // 12785 MB free after 2 banks)
        size_t next = size_t(k);
        while (ops_.free_bytes() < need_bytes && mode != BorrowMode::Count && next < lru.size()) {
            const int l = lru[next++]; borrowed_.push_back(l); borrowed_bytes_ += ops_.bank_bytes(l); ops_.release(l);
        }
        if (mode == BorrowMode::All) for (int l : bankless) if (!borrowed(l)) borrowed_.push_back(l);
        if (ops_.free_bytes() < need_bytes)
            throw std::runtime_error("prefill plan: " + std::to_string(need_bytes >> 20) + " MB needed, " +
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

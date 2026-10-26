// The shared expert-bank arithmetic (src/experts/expert_bank_plan.hpp): a device byte budget -> per-layer slot counts,
// the fitted per-layer caps, and the ledger that borrows bank bytes for a prefill. The slot plan is checked against mock
// routing (a layer whose routing spreads over more experts must get more slots) and against the rule it implements:
// slots follow spread / slot_bytes, inside the budget, never below the model's top-k, never above the expert count.
#include "expert_bank_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
using namespace tqllm;
int fails = 0;
void ok(bool pass, const char * what) {
    std::printf("%s %s\n", pass ? "PASS" : "FAIL", what);
    if (!pass) ++fails;
}
constexpr size_t MB = size_t(1) << 20;

// mock routing: top-k per token over `pool` of the `experts` experts; returns the distinct experts seen over `tokens`
// tokens, which is what an engine measures per layer to size its banks.
int routing_spread(int experts, int topk, int tokens, int pool, unsigned seed) {
    std::mt19937 r(seed);
    std::vector<int> reach(size_t(std::max(experts, 1)));
    for (int i = 0; i < experts; ++i) reach[size_t(i)] = i;
    std::shuffle(reach.begin(), reach.end(), r);
    reach.resize(size_t(std::clamp(pool, topk, experts)));
    std::set<int> seen;
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < topk; ++k) seen.insert(reach[size_t(r() % reach.size())]);
    return int(seen.size());
}

// a mock device with one bank per layer, as tests/prefill_plan.cpp models one: the ledger's release / refill / free
// callbacks here are the MixedExpertCache calls an engine would make
struct MockDevice {
    size_t total, other = 0, held = 0;
    std::vector<size_t> bank;
    std::vector<int> lru;
    std::vector<char> present;
    MockDevice(size_t total_, std::vector<size_t> bank_bytes) : total(total_), bank(std::move(bank_bytes)) {
        for (size_t l = 0; l < bank.size(); ++l) { lru.push_back(int(l)); present.push_back(bank[l] ? 1 : 0); }
    }
    size_t free_bytes() const {
        size_t held_banks = 0;
        for (size_t l = 0; l < present.size(); ++l) if (present[l]) held_banks += bank[l];
        return total - other - held_banks - held;
    }
    BankLedger::Ops ops() {
        BankLedger::Ops o;
        o.lru_layers = [this] { std::vector<int> v; for (int l : lru) if (present[size_t(l)]) v.push_back(l); return v; };
        o.bank_bytes = [this](int l) { return bank[size_t(l)]; };
        o.release = [this](int l) { present[size_t(l)] = 0; };
        o.refill = [this](int l) {
            if (free_bytes() < bank[size_t(l)]) throw std::runtime_error("out of memory");
            present[size_t(l)] = 1;
        };
        o.free_bytes = [this] { return free_bytes(); };
        return o;
    }
};
}  // namespace

int main() {
    // 1. the byte budget alone: every banked layer starts at the floor (the model's top-k), the rest of the budget goes
    //    out by marginal value per byte, so slots follow the routing spread.
    {
        const std::vector<size_t> bytes = {10 * MB, 10 * MB, 10 * MB};
        const std::vector<double> spread = {40, 10, 20};
        const auto p = expert_layer_slots(1000 * MB, bytes, spread, 8, 288);
        ok(p.fits && p.bytes <= 1000 * MB, "the plan stays inside the budget");
        ok(p.slots[0] > p.slots[2] && p.slots[2] > p.slots[1], "a layer whose routing spreads further gets more slots");
        const double r01 = double(p.slots[0]) / double(p.slots[1]), r12 = double(p.slots[1]) / double(p.slots[2]);
        char what[160];
        std::snprintf(what, sizeof what, "slots follow the spread 40 / 10 / 20: %d / %d / %d (ratios %.2f, %.2f)", p.slots[0], p.slots[1],
                      p.slots[2], r01, r12);
        ok(std::fabs(r01 - 4.0) < 0.5 && std::fabs(r12 - 0.5) < 0.15, what);
        ok(p.slots[0] + p.slots[1] + p.slots[2] == 100, "an equal-cost budget is spent completely");
    }
    // 2. an expensive layer gets fewer slots for the same routing: value per device byte, not per layer
    {
        const std::vector<size_t> bytes = {20 * MB, 5 * MB};
        const std::vector<double> spread = {30, 30};
        const auto p = expert_layer_slots(1000 * MB, bytes, spread, 8, 288);
        char what[160];
        std::snprintf(what, sizeof what, "4x the bytes per slot -> a quarter of the slots: %d / %d", p.slots[0], p.slots[1]);
        ok(p.slots[0] * 4 == p.slots[1], what);
        ok(p.slots[0] * bytes[0] == p.slots[1] * bytes[1], "equal spread -> equal bytes per layer");
        ok(p.bytes == 1000 * MB && p.bytes + std::min(bytes[0], bytes[1]) > 1000 * MB, "the budget is spent to the last affordable slot");
    }
    // 3. floors, ceilings and layers without a bank (a dense FFN)
    {
        const std::vector<size_t> bytes = {10 * MB, 0, 10 * MB};
        const std::vector<double> spread = {100, 0, 1};
        const auto p = expert_layer_slots(180 * MB, bytes, spread, 8, 12);
        ok(p.fits && p.slots[1] == 0, "a layer with no expert bank gets no slot");
        ok(p.slots[2] == 8, "the cold layer stays at the floor (the model's top-k)");
        ok(p.slots[0] > 8 && p.slots[0] <= 12, "the hot layer grows, capped at every expert");
        ok(p.bytes <= 180 * MB && p.bytes + 10 * MB > 180 * MB, "the budget is spent to the last affordable slot");
        const auto tight = expert_layer_slots(100 * MB, bytes, spread, 8, 12);
        ok(!tight.fits && tight.slots[0] == 8 && tight.slots[2] == 0, "a budget below every layer's floor says so and stops");
        bool threw = false;
        try { expert_layer_slots(100 * MB, bytes, spread, 13, 12); } catch (const std::invalid_argument &) { threw = true; }
        ok(threw, "a floor above the expert count is refused");
    }
    // 4. unmeasured routing (no prompt seen yet): every banked layer gets an equal share of the bytes
    {
        const std::vector<size_t> bytes = {8 * MB, 8 * MB, 8 * MB, 8 * MB};
        const std::vector<double> spread(4, 0.0);
        const auto p = expert_layer_slots(400 * MB, bytes, spread, 8, 288);
        const int lo = *std::min_element(p.slots.begin(), p.slots.end()), hi = *std::max_element(p.slots.begin(), p.slots.end());
        ok(hi - lo <= 1, "no measurement: equal shares");
        ok(p.bytes == 400 * MB, "and the whole budget used");
    }
    // 5. mock routing -> measured spread -> slots, at the GLM-5.3-Flash shape (288 experts, top-8, 3 dense layers, the
    //    GSQ quant mix: Q4_K layers cost more per slot than the Q2_K ones). Two layers of the same quant with different
    //    routing pools must come out with different slot counts; the plan must never go below top-k or over the experts.
    {
        const int layers = 12, experts = 288, topk = 8, tokens = 64;
        std::vector<size_t> bytes(size_t(layers), 0);
        std::vector<int> pool(size_t(layers), 0);
        for (int l = 0; l < layers; ++l) {
            if (l < 3) continue;  // dense FFN: no bank
            bytes[size_t(l)] = (l < 6 ? 14.2 : l < 8 ? 10.8 : 8.3) * MB;
            pool[size_t(l)] = 30 + 20 * l;  // later layers spread further
        }
        std::vector<double> spread(size_t(layers), 0.0);
        for (int l = 3; l < layers; ++l) spread[size_t(l)] = double(routing_spread(experts, topk, tokens, pool[size_t(l)], 7u + 13u * unsigned(l)));
        const size_t budget = 12000 * MB;
        const auto p = expert_layer_slots(budget, bytes, spread, topk, experts);
        bool floors = true, ceiling = true, nobank = true;
        for (int l = 0; l < layers; ++l) {
            if (l < 3) nobank = nobank && p.slots[size_t(l)] == 0;
            else floors = floors && p.slots[size_t(l)] >= topk && p.slots[size_t(l)] <= experts;
        }
        ok(nobank && floors && p.bytes <= budget, "every banked layer keeps top-k slots inside the budget");
        std::printf("  spread %d / %d / %d -> slots %d / %d / %d of %d MoE layers\n", int(spread[9]), int(spread[10]),
                    int(spread[11]), p.slots[9], p.slots[10], p.slots[11], int(p.slots.size()) - 3);
        ok(spread[9] < spread[10] && p.slots[9] <= p.slots[10], "the layer with the wider measured spread gets at least as many slots");
        ok(p.slots[9] < p.slots[11], "and clearly fewer than a layer spreading over twice as many experts");
        // the same spread on an expensive and a cheap layer: the cheap one gets the slots
        const std::vector<size_t> two = {14 * MB, 8 * MB};
        const auto q = expert_layer_slots(220 * MB, two, {50, 50}, topk, experts);
        ok(q.slots[1] > q.slots[0], "same spread: the cheaper slots win");
    }
    // 6. the fitted per-layer caps (TQFN_LAYER_SLOTS / "<experts>.slots"): parsed, scaled to the engine's slot count,
    //    clamped to the expert count; the environment wins over the file; neither -> no override.
    {
        const auto caps = parse_layer_slots("176,88,44,22", 64, 256);
        ok(caps.size() == 4 && caps[0] == 137 && caps[1] == 68 && caps[2] == 34 && caps[3] == 17,
           "caps fitted for another slot count are scaled to the same total (330 -> 256, ratios kept)");
        const auto clamped = parse_layer_slots("900,10,10,10", 300, 256);
        ok(clamped[0] == 256 && clamped[1] >= 1, "clamped to the expert count, never to zero");
        ok(parse_layer_slots("", 64, 256).empty(), "an empty list is no override");
        setenv("TQLLM_TEST_LAYER_SLOTS", "5,6,7", 1);
        ok(layer_slots_text("TQLLM_TEST_LAYER_SLOTS", "") == "5,6,7", "the environment is read");
        ok(layer_slots_text("TQLLM_TEST_LAYER_SLOTS_UNSET", "").empty(), "unset environment, no experts file: no override");
        unsetenv("TQLLM_TEST_LAYER_SLOTS");
    }
    // 7. the ledger over per-layer banks of different sizes: it borrows the fewest LRU banks that cover the shortfall,
    //    refuses to refill while the prefill buffers are held, and gives every bank back.
    {
        std::vector<size_t> bank;
        for (int l = 0; l < 12; ++l) bank.push_back(l < 3 ? 0 : size_t(600 + 40 * l) * MB);
        MockDevice dev(24 * size_t(1024) * MB, bank);
        dev.other = 6000 * MB;
        const size_t free_before = dev.free_bytes();
        BankLedger ledger(dev.ops());
        const size_t need = free_before + 1500 * MB;
        const auto got = ledger.borrow(need - free_before, BorrowMode::Auto, 0, need);
        ok(!got.empty() && got[0] == 3, "auto borrows the least recently used banked layer first");
        size_t released = 0;
        for (int l : got) released += bank[size_t(l)];
        ok(released >= need - free_before, "the banks it borrowed cover the shortfall");
        dev.held = need;
        bool logic = false;
        try { ledger.refill(); } catch (const std::logic_error &) { logic = true; }
        ok(logic, "refill while the prefill buffers are held is refused");
        dev.held = 0;
        ledger.buffers_released();
        ok(ledger.refill() == int(got.size()), "every borrowed bank comes back");
        ok(dev.present[3] == 1 && dev.present[11] == 1, "and the layers have their banks again");
        ok(ledger.state() == BankLedger::State::Refilled, "the request is closed");
    }

    std::printf(fails ? "EXPERT_BANK_PLAN FAIL\n" : "EXPERT_BANK_PLAN PASS\n");
    return fails ? 1 : 0;
}

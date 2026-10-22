// Flash-Next prefill memory plan and bank ledger (src/models/qwen/qwen4exp/prefill_plan.hpp), on a mock device: the plan's sizes at the
// Flash-Next shape, the borrow policies, and the request order (the 2026-10-16 refill-while-buffers-held OOM and the
// start-up arena sized by a 32K chunk reproduce here).
#include "prefill_plan.hpp"
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
using namespace tqllm;

namespace {
int fails = 0;
void expect(bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } }
constexpr size_t MB = size_t(1) << 20;

// Flash-Next GSQ-RCO IQ3_S on the 3090 (140 slots): 48 layers x 512 experts, ~1.95 MB per staged expert
PrefillShape shape() { PrefillShape s; s.hidden = 2560; s.hc = 4; s.expert_ff = 640; s.topk = 10; s.experts = 512; s.layers = 48; s.expert_block_bytes = 1950 * 1024; return s; }
PrefillTuning tuning(int chunk) { PrefillTuning t; t.chunk = chunk; t.arena_bytes = 3800 * MB; t.dense_cache_bytes = 600 * MB; t.mtp_bank_bytes = 650 * MB; return t; }

// A device of `total` bytes holding `layers` banks of `bank` bytes each plus `other` fixed bytes; prefill buffers are
// "held" by the test to model the request's life.
struct MockDevice {
    size_t total, bank, other, held = 0; std::set<int> banks; std::vector<int> lru; int refills = 0;
    MockDevice(size_t total_, size_t bank_, size_t other_, int layers) : total(total_), bank(bank_), other(other_) {
        for (int l = 0; l < layers; ++l) { banks.insert(l); lru.push_back(l); }
    }
    size_t free() const { return total - other - banks.size() * bank - held; }
    BankLedger::Ops ops() {
        BankLedger::Ops o;
        o.lru_layers = [this] { std::vector<int> v; for (int l : lru) if (banks.count(l)) v.push_back(l); return v; };
        o.bank_bytes = [this](int) { return bank; };
        o.release = [this](int l) { banks.erase(l); };
        o.refill = [this](int l) { if (free() < bank) throw std::runtime_error("out of memory"); banks.insert(l); ++refills; };
        o.free_bytes = [this] { return free(); };
        return o;
    }
};
}  // namespace

int main() {
    const auto s = shape();
    // 1. sizes: an 8K chunk is ~6-7 GB, a 32K chunk ~9-11 GB, monotonic in tokens and capped by the chunk
    const auto c8 = prefill_chunk_bytes(s, tuning(8192), 30892), c32 = prefill_chunk_bytes(s, tuning(32768), 30892);
    expect(c8.total() > 6000 * MB && c8.total() < 7500 * MB, "8K chunk between 6 and 7.5 GB");
    expect(c32.total() > 11000 * MB && c32.total() < 15000 * MB, "32K chunk between 11 and 15 GB (its arena alone is 6.6 GB)");
    expect(c32.arena == prefill_arena_bytes(32768) && c8.arena == 3800 * MB, "the arena follows the chunk once the parameter is too small (the 10-16 one-chunk OOM)");
    expect(c32.total() > c8.total(), "bigger chunk, bigger peak");
    expect(prefill_chunk_bytes(s, tuning(8192), 4095).total() < c8.total(), "a 4K prompt needs less than a full 8K chunk");
    expect(prefill_chunk_bytes(s, tuning(8192), 100000).tokens == 8192, "the chunk caps the token count");
    bool threw = false; try { prefill_chunk_bytes(s, tuning(100), 1000); } catch (const std::invalid_argument &) { threw = true; }
    expect(threw, "chunk below 256 rejected");

    // 2. the plan: copies fall 4x with one 32K chunk; release = need - free
    const size_t free0 = 2400 * MB;
    const auto p8 = prefill_plan(s, tuning(8192), 30892, free0), p32 = prefill_plan(s, tuning(32768), 30892, free0);
    expect(p8.chunks == 4 && p32.chunks == 1, "4 chunks at 8K, 1 at 32K");
    expect(p32.host_copies * 4 == p8.host_copies, "one chunk streams a quarter of the expert bytes");
    expect(p8.host_copies > 170 * 1000 * MB && p8.host_copies < 200 * 1000 * MB, "8K chunks: ~174-192 GB streamed at 32K");
    expect(p8.release == p8.need - free0 && p32.release > p8.release, "release covers the shortfall, more for 32K");
    // Flash-Next attention state: 12 attention layers, 2 kv heads x 256, indexer dim 128, sub-chunk 2048
    const size_t ctx32 = prefill_context_bytes(33024, 2048, 12, 2, 256, 128), ctx128 = prefill_context_bytes(126976, 2048, 12, 2, 256, 128);
    expect(ctx32 > 1000 * MB && ctx32 < 2500 * MB, "context state at 32K between 1 and 2.5 GB");
    expect(ctx128 > 3800 * MB && ctx128 < 8000 * MB && ctx128 > 3 * ctx32, "context state at 128K between 3.8 and 8 GB (lean top-k scratch: 4 bytes per query x block), more than 3x the 32K one");
    { auto t = tuning(16384); t.context_bytes = ctx128; expect(prefill_plan(s, t, 122555, free0).need == prefill_plan(s, tuning(16384), 122555, free0).need + ctx128, "the context state adds to the need"); }
    expect(prefill_plan(s, tuning(8192), 30892, 64 * size_t(1024) * MB).release == 0, "nothing released on a 64 GB device");

    // 3. borrow policies over 48 LRU banks of 274 MB (13.1 GB)
    const std::vector<size_t> banks(48, 274 * MB);
    const int k_auto = banks_to_borrow(p8, banks, BorrowMode::Auto);
    expect(k_auto > 0 && k_auto < 48 && size_t(k_auto) * 274 * MB >= p8.release && size_t(k_auto - 1) * 274 * MB < p8.release, "auto: the fewest banks that cover the shortfall");
    expect(banks_to_borrow(p8, banks, BorrowMode::All) == 48, "all: every bank");
    expect(banks_to_borrow(p8, banks, BorrowMode::Count, 10) == 10 && banks_to_borrow(p8, banks, BorrowMode::Count, 99) == 48, "count: clamped");
    threw = false; try { banks_to_borrow(prefill_plan(s, tuning(32768), 30892, 0), std::vector<size_t>(2, 274 * MB), BorrowMode::Auto); } catch (const std::runtime_error &) { threw = true; }
    expect(threw, "auto with too few banks reports the shortfall instead of starting");

    // 4. the request order on a 24 GB mock: borrow, hold the buffers, release them, refill
    {
        MockDevice dev(24 * size_t(1024) * MB, 274 * MB, 8500 * MB, 48);  // 13.1 GB banks + 8.5 GB fixed -> 2.3 GB free
        BankLedger ledger(dev.ops());
        const auto plan = prefill_plan(s, tuning(32768), 30892, dev.free());
        const auto got = ledger.borrow(plan, BorrowMode::All);
        expect(got.size() == 48 && dev.banks.empty() && dev.free() >= plan.need, "borrow all: 48 banks released, the 32K plan fits");
        dev.held = plan.trunk_peak;  // the trunk runs
        bool logic = false; try { ledger.refill(); } catch (const std::logic_error &) { logic = true; }
        expect(logic, "refill while the prefill buffers are held is refused (the 10-16 OOM)");
        dev.held = 0; ledger.buffers_released();
        expect(ledger.refill() == 48 && dev.banks.size() == 48 && ledger.state() == BankLedger::State::Refilled, "after the release every bank refills");
    }
    // 4b. borrow all with layers that have no bank (regrow released them): they join the refill, so every layer comes back
    {
        MockDevice dev(24 * size_t(1024) * MB, 274 * MB, 8500 * MB, 48);
        dev.banks.erase(5); dev.banks.erase(9);
        BankLedger ledger(dev.ops());
        const auto plan = prefill_plan(s, tuning(32768), 30892, dev.free());
        const auto got = ledger.borrow(plan, BorrowMode::All, 0, {5, 9});
        expect(got.size() == 48 && ledger.borrowed(5) && ledger.borrowed(9), "borrow all: the bankless layers 5 and 9 join the 46 released");
        ledger.buffers_released();
        expect(ledger.refill() == 48 && dev.banks.size() == 48, "every layer has a bank after the refill (12 GB 16K a != b)");
    }
    // 4c. a release frees a little less than the bank's bytes: auto borrows the next bank instead of failing a few MB short
    {
        MockDevice dev(24 * size_t(1024) * MB, 274 * MB, 8500 * MB, 48);
        BankLedger::Ops o = dev.ops();
        o.release = [&dev](int l) { dev.banks.erase(l); dev.other += 8 * MB; };  // 8 MB of each bank stays allocated
        BankLedger ledger(o);
        auto plan = prefill_plan(s, tuning(8192), 30892, dev.free());
        plan.need = dev.free() + 2 * 274 * MB;  // needs exactly two banks' bytes
        plan.release = 2 * 274 * MB;
        bool ok = true; std::vector<int> got;
        try { got = ledger.borrow(plan, BorrowMode::Auto); } catch (const std::runtime_error &) { ok = false; }
        expect(ok && got.size() == 3 && dev.free() >= plan.need, "auto borrow: two banks fall 16 MB short, the third covers it (GB10 12789 vs 12785 MB)");
    }
    // 5. a refill that runs out of memory stops and reports how many came back; auto borrows the LRU banks first
    {
        MockDevice dev(24 * size_t(1024) * MB, 274 * MB, 8500 * MB, 48);
        dev.lru = {7, 3, 11, 0};  for (int l = 0; l < 48; ++l) if (l != 7 && l != 3 && l != 11 && l != 0) dev.lru.push_back(l);
        BankLedger ledger(dev.ops());
        const auto plan = prefill_plan(s, tuning(8192), 30892, dev.free());
        const auto got = ledger.borrow(plan, BorrowMode::Auto);
        expect(got.size() >= 4 && got[0] == 7 && got[1] == 3 && got[2] == 11 && got[3] == 0, "auto releases least recently used banks first");
        expect(ledger.borrowed(7) && !ledger.borrowed(47), "borrowed() answers per layer");
        ledger.buffers_released();
        dev.other += (dev.free() - got.size() * dev.bank) + 5 * dev.bank;  // room for all but 5 banks: the refill cannot complete
        const int back = ledger.refill();
        expect(back == int(got.size()) - 5 && ledger.state() == BankLedger::State::Refilled, "a short refill stops at the first failure and reports the count");
    }
    // 5b. an overlapped refill keeps the catch-up's room and the second call finishes the rest
    {
        MockDevice dev(24 * size_t(1024) * MB, 274 * MB, 8500 * MB, 48);
        BankLedger ledger(dev.ops());
        const auto plan = prefill_plan(s, tuning(16384), 30892, dev.free());
        const auto got = ledger.borrow(plan, BorrowMode::All);
        ledger.buffers_released();
        const size_t keep = 9000 * MB;
        const int first = ledger.refill(keep);
        expect(first > 0 && first < int(got.size()) && dev.free() >= keep && dev.free() < keep + 274 * MB, "partial refill stops at the kept room");
        expect(ledger.state() == BankLedger::State::BuffersReleased && ledger.borrowed_total() == got.size(), "the rest stays borrowed, total unchanged");
        const int second = ledger.refill();
        expect(first + second == int(got.size()) && dev.banks.size() == 48 && ledger.state() == BankLedger::State::Refilled, "the second call completes the refill");
    }
    // 6. start-up: the arena is sized by the 8K chunk even when the manifest chunk is 32K (the second 10-16 OOM)
    auto startup_arena_mb = [](int chunk) { return std::min(chunk, 8192) / 5 + 64; };
    expect(startup_arena_mb(32768) == startup_arena_mb(8192) && startup_arena_mb(8192) == 1702, "start-up arena capped at the 8K-chunk size");

    // 7. hyper-connection norm: chunks of more than 64 tokens skip the fp32 normalized rows only when the mixer projections
    //    that read them take the BF16 image (BF16 weights). The Flash-Next IQ4_XS file has IQ4_XS hc_attn_down / hc_ffn_down
    //    (the MTP file Q8_0): their native kernels read the fp32 rows, which were never written (NaN routing at 512 tokens).
    expect(hc_row_scales_only(512, true, true), "BF16 mixer, long chunk: row scales only");
    expect(!hc_row_scales_only(64, true, true), "short chunk keeps the fp32 rows");
    expect(!hc_row_scales_only(512, false, true), "IQ4_XS hc down needs the fp32 rows");
    expect(!hc_row_scales_only(512, true, false), "non-BF16 inject needs the fp32 rows");

    // 8. expert slots from free memory (manifest slots=auto). RTX 3090, Flash-Next IQ4_XS on 10-18: one slot of every
    //    layer is 121.9 MB (14625 MB of banks at 120 slots); with the banks released ~17.3 GB were free and a 33K context
    //    needs ~0.6 GB (state) + 0.5 GB (decode reserve). The fixed manifest's 112 slots left 3.5 GB unused.
    {
        const size_t per = size_t(121.9 * double(MB)), keep = (600 + 512 + 256) * MB, free_b = 17300 * MB;
        const int s = auto_expert_slots(free_b, keep, per, 512);
        expect(s >= 130 && s <= 132, "IQ4_XS 3090: ~131 slots from 17.3 GB free");
        expect(size_t(s) * per + keep <= free_b, "the chosen slots leave the kept room");
        expect(auto_expert_slots(keep / 2, keep, per, 512) == 1, "too little memory: one slot, the fit check then reports");
        expect(auto_expert_slots(size_t(100) << 30, keep, per, 512) == 512, "capped at every expert (GB10)");
        expect(auto_expert_slots(free_b, keep, 0, 512) == 512, "no per-slot size known: every expert");
    }

    std::printf(fails ? "PREFILL_PLAN FAIL\n" : "PREFILL_PLAN PASS\n");
    return fails ? 1 : 0;
}

// Parallel-slot memory guard: tqserve's startup precheck (fit_slots) and the bounded graph cache (GraphLru).
#include "../src/fit_context.hpp"
#include "../src/graph_lru.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    const size_t GB = 1000000000ull, MB = size_t(1) << 20;
    int bad = 0;
    auto expect = [&](const char * what, bool ok) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); ++bad; } };

    // .77 3090, 27B TQ4_L, tqserve 49e2fd6 --np 8 --context 32768 (2026-10-13): the fit left 1.63 GB free (margin 0.81 GB)
    // and the server failed mid-run with "graph instantiate: out of memory". The precheck must refuse that np.
    const size_t reserve = 128 * MB;
    const uint32_t n77 = tqllm::fit_slots(8, 4096, 4096, size_t(1.63 * GB), size_t(0.81 * GB), reserve);
    expect(".77 np 8 refused", n77 < 8);
    expect(".77 suggestion fits the memory", n77 >= 1 && 0.81 * GB + n77 * reserve <= 1.63 * GB);
    // the same load after the fit made room (3.46 GB free, .77 10-13 08:33): np 8 at 4096 per slot fits; the context is
    // per slot, so it must not cap np by itself (the first version read it as the total and suggested --np 1)
    expect(".77 np 8 after the fit", tqllm::fit_slots(8, 4096, 4096, size_t(3.46 * GB), size_t(0.81 * GB), reserve) == 8);
    // np 16 at --context 65536 on .77: 2.65 GB free, memory for 13 slots
    expect(".77 np 16 refused, 13 fit", tqllm::fit_slots(16, 4096, 4096, size_t(2.65 * GB), size_t(0.81 * GB), reserve) == 13);
    // np 16 on a 32K context: 2048 positions per slot is below the 4096 floor -> 8 slots of 4096
    expect("np 16 at 32K: context floor", tqllm::fit_slots(16, 2048, 4096, size_t(40 * GB), size_t(0.81 * GB), reserve) == 8);
    expect("np 16 that fits", tqllm::fit_slots(16, 8192, 4096, size_t(8 * GB), size_t(0.81 * GB), reserve) == 16);
    expect("one slot is never refused", tqllm::fit_slots(1, 2048, 4096, 0, size_t(0.81 * GB), reserve) == 1);
    expect("no memory left: one slot suggested", tqllm::fit_slots(8, 65536, 4096, size_t(0.5 * GB), size_t(0.81 * GB), reserve) == 1);

    // GraphLru: fake handles; destroy records what was freed, quiesce must run before any destroy
    std::vector<int> freed;
    int quiesced = 0, destroyed_before_quiesce = 0;
    auto destroy = [&](int e) { if (!quiesced) ++destroyed_before_quiesce; freed.push_back(e); };
    auto quiesce = [&] { ++quiesced; };
    bool low = false;
    auto low_memory = [&] { return low; };
    tqllm::GraphLru<int> lru(3);
    for (int k = 1; k <= 3; ++k) lru.insert(0, uint64_t(k), [&] { return 100 + k; }, low_memory, quiesce, destroy);
    expect("hit returns the graph", lru.find(0, 1) == 101);  // 1 is now the most recent
    expect("family is part of the key", lru.find(1, 1) == 0);
    lru.insert(0, 4, [] { return 104; }, low_memory, quiesce, destroy);
    expect("cap evicts the least recently used", freed.size() == 1 && freed[0] == 102 && lru.size() == 3);
    expect("quiesce before destroy", quiesced == 1 && destroyed_before_quiesce == 0);
    low = true;
    lru.insert(0, 5, [] { return 105; }, low_memory, quiesce, destroy);
    expect("low memory empties the cache before a new graph", lru.size() == 1 && freed.size() == 4);
    low = false;
    // instantiation out of memory: the cache is emptied and the graph made again
    lru.insert(0, 6, [] { return 106; }, low_memory, quiesce, destroy);
    int tries = 0;
    lru.insert(0, 7, [&] { if (++tries == 1) throw std::runtime_error("graph instantiate: out of memory"); return 107; },
               low_memory, quiesce, destroy);
    expect("OOM retried once after emptying", tries == 2 && lru.size() == 1 && lru.find(0, 7) == 107);
    bool rethrown = false;
    try {
        lru.insert(0, 8, [] () -> int { throw std::runtime_error("graph capture: operation not permitted"); }, low_memory, quiesce, destroy);
    } catch (const std::runtime_error &) { rethrown = true; }
    expect("other errors propagate", rethrown && lru.size() == 1);
    lru.clear(destroy);
    expect("clear frees everything", lru.size() == 0 && lru.evictions() == freed.size());

    // lossy fit steps are reported (tqserve refuses them without -fitl on); placement-only loads report nothing.
    // .77 3090, 27B TQ4_L --np 8 --context 262144 (10-13): the fit switched the KV cache to t4k-turbo3v and only 2 of 8
    // answers matched np 1
    expect("same KV and context: nothing", tqllm::fit_result_changes("bf16", "bf16", 32768, 32768).empty());
    expect(".77 np 8 KV switch reported", tqllm::fit_result_changes("bf16", "q4/tcq2", 32768, 32768) == "KV cache q4/tcq2 (asked bf16)");
    expect("context cut reported", tqllm::fit_result_changes("bf16", "bf16", 262144, 131072) == "context 131072 per slot (asked 262144)");
    expect("both reported", tqllm::fit_result_changes("q8/q8", "t4k-turbo3v", 65536, 32768).find(", context 32768") != std::string::npos);
    expect("a longer context is not a cut", tqllm::fit_result_changes("bf16", "bf16", 2048, 4096).empty());

    if (bad) return 1;
    std::puts("slot guard PASS");
    return 0;
}

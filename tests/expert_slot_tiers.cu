// Two expert slot tiers (MixedExpertCache::set_layer_tiers): a bank rebuilt while the free memory cannot hold its target
// (a refill after a long prompt) is sized to an even share of what is free, never below its floor, and does not evict
// other layers' banks, and leaves the 256 MB the decode's lazy state needs; regrow() brings it back to the target. Without
// tiers the same refill takes that room (3090 after a 30K prompt: 75 MB free, out of memory in decode), and where even
// that is short it evicts other layers' banks (each step one bank evicted the next: 67 -> 4.9 tok/s).
#include "tq3_expert_cache.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static int failures = 0;
static void expect(bool ok, const char * what) { if (!ok) { std::printf("FAIL %s\n", what); ++failures; } }

int main() {
    // the scenarios fill the device's free memory; on a GPU that shares system memory (GB10) that is the host's memory
    {   int dev = 0, shared = 0; cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&shared, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) == cudaSuccess && shared) {
            std::printf("expert slot tiers: skipped (device memory is system memory)\n"); return 77; } }
    const int layers = 4, experts = 96, hidden = 2560, ff = 640, target = 40, floor = 8;
    const std::array<uint32_t, 3> types{18, 18, 20};
    std::vector<std::array<std::vector<char>, 3>> raw(size_t(layers) * experts);
    std::mt19937 rng(91);
    for (auto & e : raw) for (int p = 0; p < 3; ++p) {
        uint64_t block, bytes; gguf_block_layout(types[p], block, bytes);
        e[p].resize(uint64_t(hidden) * ff / block * bytes);
        for (auto & b : e[p]) b = char(rng() & 255);
        for (size_t b = 0; b < e[p].size(); b += bytes) { const auto s = __float2half_rn(.001f); std::memcpy(e[p].data() + b, &s, 2); }
    }
    auto source = [&](int l, int id) {
        std::array<ExpertPayload, 3> pl;
        for (int p = 0; p < 3; ++p) { const auto & d = raw[size_t(l) * experts + id][p]; pl[p] = {types[p], {d.data(), d.size()}}; }
        return pl;
    };
    std::vector<int> all(target); for (int i = 0; i < target; ++i) all[i] = i;
    auto scenario = [&](bool tiers, int & shrunk_slots, bool & others_kept, bool & regrown, bool & bytes_ok, size_t & free_after) {
        MixedExpertCache cache(layers, experts, hidden, ff, target, size_t(4) << 30, source, nullptr);
        cache.set_layer_slots(std::vector<int>(layers, target));
        if (tiers) cache.set_layer_tiers(std::vector<int>(layers, floor));
        for (int l = 0; l < layers; ++l) { cache.prepare(l, all); cache.wait_misses(); }
        check(cudaDeviceSynchronize());
        const size_t per = cache.bank_layout(2).block;
        auto expert_bytes = [&](int id) {  // the gate projection of one expert as layer 2's bank holds it
            const auto slots = cache.prepare(2, {id}); cache.wait_misses(); const auto lay = cache.bank_layout(2); const auto m = cache.gate();
            std::vector<char> b(lay.piece[0]); check(cudaMemcpy(b.data(), static_cast<const char *>(m.p) + size_t(slots[0]) * lay.piece[0], b.size(), cudaMemcpyDeviceToHost));
            return b;
        };
        const auto before = expert_bytes(1);
        cache.release_bank(2); cache.release_bank(3);  // a long prompt borrowed them
        // fill the device: room for the 256 MB margin plus half of the two banks
        size_t fr = 0, tot = 0; check(cudaMemGetInfo(&fr, &tot));
        const size_t leave = (size_t(256) << 20) + size_t(target) * per + (size_t(4) << 20);
        void * hog = nullptr; check(cudaMalloc(&hog, fr - leave));
        const uint64_t ev0 = cache.stats().evictions;
        shrunk_slots = tiers ? cache.plan_slots(2) : cache.slots(2);
        std::vector<int> top(all.begin(), all.begin() + shrunk_slots);
        bool built = true;
        try { cache.prepare(2, top); cache.wait_misses(); cache.prepare(3, std::vector<int>(top.begin(), top.begin() + std::min<size_t>(top.size(), size_t(tiers ? cache.plan_slots(3) : target)))); cache.wait_misses(); }
        catch (const std::exception &) { built = false; }
        check(cudaDeviceSynchronize()); cudaGetLastError();
        check(cudaMemGetInfo(&free_after, &tot));
        others_kept = built && cache.bank_ready(0) && cache.bank_ready(1) && cache.stats().evictions == ev0;
        // the shrunk bank holds the same bytes for an expert as the full bank did
        bytes_ok = built && cache.bank_ready(2) && expert_bytes(1) == before;
        check(cudaFree(hog));
        regrown = false;
        if (tiers) { cache.regrow(0); regrown = cache.plan_slots(2) == target && cache.plan_slots(3) == target; }
    };
    int s1 = 0; bool kept1 = false, grow1 = false, bytes1 = false; size_t free1 = 0;
    scenario(true, s1, kept1, grow1, bytes1, free1);
    expect(s1 >= floor && s1 < target, "tiers: the refilled bank is sized to the free memory (between floor and target)");
    expect(s1 >= target / 2 - 2 && s1 <= target / 2 + 2, "tiers: two pending banks share the free memory evenly");
    expect(kept1, "tiers: rebuilding two banks evicts no other layer's bank");
    expect(bytes1, "tiers: the shrunk bank holds the source expert bytes");
    expect(grow1, "tiers: regrow brings the banks back to their target once the memory is free");
    expect(free1 >= (size_t(240) << 20), "tiers: the refill leaves the decode's 256 MB");
    {   // the size plan_slots gives a released bank is the size prepare builds, even when memory frees in between (the prompt
        // path sized its slot maps from slots() and prepare then built a larger bank: writes past them, heap corruption)
        MixedExpertCache cache(layers, experts, hidden, ff, target, size_t(4) << 30, source, nullptr);
        cache.set_layer_slots(std::vector<int>(layers, target)); cache.set_layer_tiers(std::vector<int>(layers, floor));
        for (int l = 0; l < layers; ++l) { cache.prepare(l, all); cache.wait_misses(); }
        const size_t per = cache.bank_layout(1).block;
        cache.release_bank(1);
        size_t fr = 0, tot = 0; check(cudaMemGetInfo(&fr, &tot));
        void * hog = nullptr; check(cudaMalloc(&hog, fr - ((size_t(256) << 20) + size_t(target / 2) * per)));
        const int planned = cache.plan_slots(1);
        check(cudaFree(hog));  // room for the full target now
        const auto slots = cache.prepare(1, std::vector<int>(all.begin(), all.begin() + planned)); cache.wait_misses();
        int top_slot = 0; for (int x : slots) top_slot = std::max(top_slot, x);
        expect(planned < target && cache.slots(1) == planned && top_slot < planned, "planned bank size is the size prepare builds");
    }
    {   // regrow releases a shrunk bank only while the memory also holds the full banks of those it already released: a
        // released bank's slots show as free memory, and counting them twice let the next bank go too, so both rebuilt
        // below their target (residency thrown away for no capacity)
        MixedExpertCache cache(layers, experts, hidden, ff, target, size_t(4) << 30, source, nullptr);
        cache.set_layer_slots(std::vector<int>(layers, target)); cache.set_layer_tiers(std::vector<int>(layers, floor));
        for (int l = 0; l < layers; ++l) { cache.prepare(l, all); cache.wait_misses(); }
        const size_t per = cache.bank_layout(2).block, margin = size_t(256) << 20;
        cache.release_bank(2); cache.release_bank(3);
        size_t fr = 0, tot = 0; check(cudaMemGetInfo(&fr, &tot));
        void * hog = nullptr; check(cudaMalloc(&hog, fr - (margin + size_t(target) * per + (size_t(4) << 20))));
        for (int l : {2, 3}) { const int c = cache.plan_slots(l); cache.prepare(l, std::vector<int>(all.begin(), all.begin() + c)); cache.wait_misses(); }
        const int half = cache.slots(2);
        check(cudaDeviceSynchronize()); check(cudaFree(hog));
        // room beyond the margin for one and a half banks' growth: one bank can regrow, not both
        check(cudaMemGetInfo(&fr, &tot));
        const size_t g = size_t(target - half) * per;
        check(cudaMalloc(&hog, fr - (margin + g * 3 / 2)));
        const int released = cache.regrow(margin);
        bool full = true; for (int l : {2, 3}) if (!cache.bank_ready(l)) full = full && cache.plan_slots(l) == target;
        expect(half < target && released == 1 && full, "regrow releases only the banks the memory can rebuild at their target");
        check(cudaFree(hog));
    }
    int s0 = 0; bool kept0 = false, grow0 = false, bytes0 = false; size_t free0 = 0;
    scenario(false, s0, kept0, grow0, bytes0, free0);
    expect(!kept0 || free0 < (size_t(240) << 20), "control (no tiers): the same refill takes the decode's room or evicts a bank");
    std::printf("expert slot tiers: %s (tiers %d slots %zu MB free; control kept=%d %zu MB free)\n", failures ? "FAIL" : "PASS", s1, free1 >> 20, int(kept0), free0 >> 20);
    return failures ? 1 : 0;
}

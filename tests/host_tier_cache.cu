// MixedExpertCache over a host RAM tier (experts read from the GGUF on demand): every slot a prepare() hands out holds
// the expert's exact bank bytes, through misses read into the tier, evictions copied back to it (device to host) when the
// tier gave their blocks away, and reloads from it, with a tier small enough that blocks are reused while copies fly.
// Expected: bytes equal on every call; an evicted expert needed again comes from the tier (no second drive read) while it
// fits; a tier with room keeps the experts it sent to the card, so nothing is copied back.
#include "tq3_expert_cache.hpp"
#include "host_expert_tier.hpp"
#include "packed_experts.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tqllm;
static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void require(bool ok, const std::string & why) { if (!ok) throw std::runtime_error(why); }

static void run(int tier_blocks, int rounds, bool expect_reuse) {
    const int layers = 2, experts = 12, hidden = 2048, ff = 768, slots = 4;
    const uint32_t type = 21;  // IQ3_S
    uint64_t qk = 0, qb = 0; require(gguf_block_layout(type, qk, qb), "IQ3_S layout");
    std::mt19937 rng(11);
    std::vector<std::array<std::vector<char>, 3>> raw(size_t(layers) * experts);
    for (auto & r : raw) for (int p = 0; p < 3; ++p) {
        r[size_t(p)].resize(size_t(hidden) * ff / qk * qb);
        for (auto & b : r[size_t(p)]) b = char(rng() & 255);
        for (size_t o = 0; o < r[size_t(p)].size(); o += qb) { const auto s = __float2half_rn(0.01f); std::memcpy(r[size_t(p)].data() + o, &s, 2); }
    }
    auto payload = [&](int l, int e) {
        std::array<ExpertPayload, 3> pl;
        for (int p = 0; p < 3; ++p) pl[size_t(p)] = {type, {raw[size_t(l * experts + e)][size_t(p)].data(), raw[size_t(l * experts + e)][size_t(p)].size()}};
        return pl;
    };
    std::vector<std::vector<char>> packed(raw.size());
    for (int l = 0; l < layers; ++l) for (int e = 0; e < experts; ++e) {
        for (const auto & part : repack_expert(payload(l, e), hidden, ff, false)) packed[size_t(l * experts + e)].insert(packed[size_t(l * experts + e)].end(), part.begin(), part.end());
    }
    const size_t block = packed[0].size();
    char * mem = nullptr; check(cudaMallocHost(reinterpret_cast<void **>(&mem), size_t(tier_blocks) * block));
    int reads = 0;
    HostExpertTier tier(layers, experts, std::vector<size_t>(size_t(layers), block), mem, size_t(tier_blocks) * block, [&](int l, int e, char * dst) {
        ++reads; std::memcpy(dst, packed[size_t(l * experts + e)].data(), block); }, 3);
    {
        MixedExpertCache cache(layers, experts, hidden, ff, slots, size_t(256) << 20, payload, nullptr);
        cudaStream_t ms; check(cudaStreamCreateWithFlags(&ms, cudaStreamNonBlocking));
        cache.set_host_tier(&tier); cache.set_miss_stream(ms);
        require(cache.split(), "the tier switches split mode on");
        std::vector<char> got(block);
        for (int r = 0; r < rounds; ++r) for (int l = 0; l < layers; ++l) {
            std::vector<int> ids; for (int k = 0; k < 3; ++k) ids.push_back(int(rng() % experts));
            if (r % 10 == 9 && l == 0) cache.release_bank(1);  // a prompt borrows a bank: its experts go back to the tier
            const auto slot = cache.prepare(l, ids); cache.wait_misses();
            if (r % 8 != 7) continue;  // seven rounds of calls with copies in flight, then a check
            check(cudaDeviceSynchronize());
            const auto lay = cache.bank_layout(l);
            require(lay.host_tier && !lay.src0 && lay.block == block, "the layout streams from the tier, block size matches");
            for (size_t a = 0; a < ids.size(); ++a) {
                size_t off = 0;
                for (int k = 0; k < 6; ++k) if (lay.piece[k]) {
                    check(cudaMemcpy(got.data() + off, lay.dst[k] + size_t(slot[a]) * lay.piece[k], lay.piece[k], cudaMemcpyDeviceToHost));
                    off += lay.piece[k];
                }
                require(off == block && std::memcmp(got.data(), packed[size_t(l * experts + ids[a])].data(), block) == 0,
                        "round " + std::to_string(r) + " layer " + std::to_string(l) + " expert " + std::to_string(ids[a]) + ": slot bytes differ");
            }
        }
        check(cudaDeviceSynchronize());  // the cache copies on its own stream
        tier.for_each([&](int l, int e, const char * p) {  // every block the tier holds has its expert's bytes
            require(std::memcmp(p, packed[size_t(l * experts + e)].data(), block) == 0, "tier block of layer " + std::to_string(l) + " expert " + std::to_string(e) + " holds other bytes");
        });
        cache.set_miss_stream(nullptr); check(cudaStreamDestroy(ms));
    }
    const auto & st = tier.stats(); const auto ix = tier.index_stats();
    std::printf("host_tier_cache: %d blocks: %d drive reads, %llu demoted, %llu tier hits, %llu fences, %llu evictions\n", tier_blocks, reads,
                (unsigned long long) st.demotes, (unsigned long long) ix.hits, (unsigned long long) ix.fences, (unsigned long long) ix.evictions);
    // released blocks wait for their copies, so "everything" needs room beside them: twice the experts
    // a tier with room for everything keeps what it sent to the card (inclusive): each expert read once, nothing copied back
    if (tier_blocks >= 2 * layers * experts)
        require(reads <= layers * experts && st.demotes == 0, "a tier with room for everything reads each expert once and copies nothing back");
    // a small one gives the blocks of experts on the card away (exclusive under pressure): their evictions come back to it
    // a GPU that reads host memory through its page tables (GB10) copies nothing back to the tier (mixed_expert_cache.cpp
    // host_coherent): only a PCIe card's small tier takes its evicted experts back
    int dev = 0, coherent = 0; check(cudaGetDevice(&dev));
    if (cudaDeviceGetAttribute(&coherent, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) != cudaSuccess) coherent = 0;
    if (expect_reuse) require(ix.evictions + ix.fences > 0 && (coherent ? st.demotes == 0 : st.demotes > 0),
                              coherent ? "the small tier reused blocks (coherent host: nothing copied back)" : "the small tier reused blocks and took evicted experts back");
    check(cudaFreeHost(mem));
}

int main() {
    try { run(48, 40, false); run(8, 60, true); }
    catch (const std::exception & e) { std::fprintf(stderr, "host_tier_cache: FAIL %s\n", e.what()); return 1; }
    std::printf("host_tier_cache: ok\n");
    return 0;
}

// Prefill chunk rule: GB10 (128 GB unified) at long context must not fall back to the 24 GB-card 512-token chunks;
// 24 GB and 16 GB cards keep their sizes.
#include "prefill_chunk.hpp"

#include <cstdio>
#include <vector>
#include <algorithm>
#include <string>

namespace {
int failures = 0;
void expect(int got, int want, const char * what) {
    if (got != want) { std::fprintf(stderr, "FAIL %s: %d, want %d\n", what, got, want); ++failures; }
}
constexpr size_t GiB = size_t(1) << 30;
// Qwen3.8-27B, 16 attention layers, tq3 KV: 2 x 4 heads x 66 bytes per position and layer
constexpr size_t kv_tq3_per_pos = 16 * 2 * 4 * 66;
}

int main() {
    using tqllm::prefill_chunk;
    // GB10, TQ_XS loaded (~12 GB of 128), 540K context, all KV on the device: old rule 512
    expect(prefill_chunk(540672, 128 * GiB, 110 * GiB, 540672 * kv_tq3_per_pos), 8192, "GB10 540K device KV");
    expect(prefill_chunk(270336, 128 * GiB, 110 * GiB, 270336 * kv_tq3_per_pos), 8192, "GB10 270K");
    expect(prefill_chunk(8192, 128 * GiB, 110 * GiB, 8192 * kv_tq3_per_pos), 8192, "GB10 8K");
    expect(prefill_chunk(270336, 48 * GiB, 20 * GiB, 270336 * kv_tq3_per_pos), 4096, "48 GB card, 20 GB free");
    // RTX 3090 (24 GB, ~11 GB free after the model): unchanged
    expect(prefill_chunk(32768, 24 * GiB, 11 * GiB, 32768 * kv_tq3_per_pos), 4096, "3090 32K");
    expect(prefill_chunk(131072, 24 * GiB, 11 * GiB, 131072 * kv_tq3_per_pos), 2048, "3090 128K");
    expect(prefill_chunk(262144, 24 * GiB, 11 * GiB, 262144 * kv_tq3_per_pos), 512, "3090 256K");
    // RTX 5060 Ti (16 GB, ~3.5 GB free): unchanged
    expect(prefill_chunk(204800, 16 * GiB, size_t(3.5 * GiB), 204800 * kv_tq3_per_pos), 512, "5060 Ti 200K");
    expect(prefill_chunk(32768, 16 * GiB, size_t(3.5 * GiB), 32768 * kv_tq3_per_pos), 2048, "5060 Ti 32K");
    // KV larger than free memory: no underflow
    expect(prefill_chunk(262144, 24 * GiB, 2 * GiB, 20 * GiB), 512, "KV beyond free");
    // GB10 unified memory: cudaMemGetInfo says 2.9 GiB free (page cache counted as used), /proc/meminfo has the truth
    const std::string gb10 = "MemTotal:       127535116 kB\nMemFree:         3120496 kB\nMemAvailable:   101820244 kB\n";
    const size_t avail = tqllm::meminfo_available(gb10);
    if (avail != size_t(101820244) * 1024) { std::fprintf(stderr, "FAIL meminfo_available: %zu\n", avail); ++failures; }
    expect(prefill_chunk(270336, 121 * GiB, size_t(2.9 * GiB), 270336 * kv_tq3_per_pos), 512, "GB10 cudaMemGetInfo free only");
    expect(prefill_chunk(270336, 121 * GiB, avail, 270336 * kv_tq3_per_pos), 8192, "GB10 MemAvailable");
    if (tqllm::meminfo_available("MemTotal: 1 kB\n") != 0) { std::fprintf(stderr, "FAIL meminfo without MemAvailable\n"); ++failures; }
    // kept prompt states (the per-slot prompt cache): a later turn resumed at the kept state must prefill exactly the
    // chunks a cold prefill of its prompt would (the old rule, prompt - 64, resumed mid-chunk: other kernels, other
    // replies on 2 of 48 agent turns)
    auto cold = [](uint32_t from, uint32_t P, uint32_t ch) {  // chunk starts of a prefill from `from` to P
        std::vector<uint32_t> v;
        for (uint32_t t = from; t < P; t += std::min(ch, P - t)) v.push_back(t);
        return v;
    };
    for (uint32_t ch : {512u, 2048u, 8192u})
        for (uint32_t P = 1; P < 40000; P += 997) {
            const uint32_t k = tqllm::kept_state_at(P, 0, ch, 64);
            if (k && (k % ch || k + 64 > P)) { std::fprintf(stderr, "FAIL kept state %u off the grid (P %u, chunk %u)\n", k, P, ch); ++failures; }
            const uint32_t P2 = P + 300;  // the next turn's prompt shares [0, k)
            if (!k) continue;
            auto all = cold(0, P2, ch), tail = cold(k, P2, ch);
            all.erase(all.begin(), std::find(all.begin(), all.end(), k));
            if (all != tail) { std::fprintf(stderr, "FAIL resume at %u does not replay the cold chunks (P %u, chunk %u)\n", k, P2, ch); ++failures; }
            if (tqllm::kept_state_at(P2, k, ch, 64) && tqllm::kept_state_at(P2, k, ch, 64) <= k) { std::fprintf(stderr, "FAIL newer state not after the resume point\n"); ++failures; }
        }
    if (tqllm::kept_state_at(17846, 0, 8192, 64) != 16384) { std::fprintf(stderr, "FAIL 17846 tokens, chunk 8192: kept at %u\n", tqllm::kept_state_at(17846, 0, 8192, 64)); ++failures; }
    if (tqllm::kept_state_at(16384 + 64, 16384, 8192, 64) != 0) { std::fprintf(stderr, "FAIL no newer grid point: keep nothing new\n"); ++failures; }
    // message-start chunks: a turn resumed at its kept state (the last message start) prefills exactly the chunks a cold
    // prefill of the next turn's prompt would (the grid rule kept states up to a chunk before the end: 1-2K tokens to
    // re-read per follow-up)
    auto chunks = [](const std::vector<uint32_t> & st, uint32_t from, uint32_t P, uint32_t ch) {
        std::vector<uint32_t> v;
        for (uint32_t t = from; t < P; t = tqllm::chunk_end(st, t, P, ch)) v.push_back(t);
        return v;
    };
    for (uint32_t ch : {512u, 2048u, 8192u}) {
        const std::vector<uint32_t> turn1{30, 16800, 16860};            // system, user (document), generation prompt
        const uint32_t P1 = 16864;
        const uint32_t k = tqllm::kept_state_msg(turn1, P1, 0);
        if (k != 16860) { std::fprintf(stderr, "FAIL kept at the last message start: %u\n", k); ++failures; }
        auto cold1 = chunks(turn1, 0, P1, ch);
        if (std::find(cold1.begin(), cold1.end(), k) == cold1.end()) { std::fprintf(stderr, "FAIL a cold prefill starts no chunk at the kept state\n"); ++failures; }
        const std::vector<uint32_t> turn2{30, 16800, 16860, 17030, 17600};  // + the reply, a tool result, the next generation prompt
        const uint32_t P2 = 17604;
        auto cold2 = chunks(turn2, 0, P2, ch), tail2 = chunks(turn2, k, P2, ch);
        cold2.erase(cold2.begin(), std::find(cold2.begin(), cold2.end(), k));
        if (cold2 != tail2) { std::fprintf(stderr, "FAIL the resumed turn does not replay the cold chunks (chunk %u)\n", ch); ++failures; }
        if (tail2.size() > 4) { std::fprintf(stderr, "FAIL resume re-reads more than the new messages\n"); ++failures; }
        if (tqllm::kept_state_msg(turn2, P2, k) != 17600) { std::fprintf(stderr, "FAIL the next kept state\n"); ++failures; }
    }
    if (tqllm::chunk_end({}, 0, 20000, 8192) != 8192 || tqllm::chunk_end({}, 8192, 20000, 8192) != 16384)
        { std::fprintf(stderr, "FAIL no message starts: the plain grid\n"); ++failures; }
    if (tqllm::kept_state_msg({}, 20000, 0) != 0) { std::fprintf(stderr, "FAIL no message starts: nothing kept by this rule\n"); ++failures; }
    if (failures) return 1;
    std::puts("prefill_chunk: ok");
    return 0;
}

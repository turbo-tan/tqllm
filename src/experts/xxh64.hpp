#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace tqllm {
// XXH64 (the published xxHash 64-bit algorithm), bit-identical to libxxhash's XXH64(data, n, seed): the cold store's
// .xxh64 sidecars are written and checked with it on every platform without a runtime library.
inline uint64_t xxh64(const void * data, size_t n, uint64_t seed) {
    constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull,
                       P4 = 0x85EBCA77C2B2AE63ull, P5 = 0x27D4EB2F165667C5ull;
    auto rotl = [](uint64_t x, int r) {return (x << r) | (x >> (64 - r));};
    auto read64 = [](const unsigned char * p) {uint64_t v; std::memcpy(&v, p, 8); return v;};
    auto read32 = [](const unsigned char * p) {uint32_t v; std::memcpy(&v, p, 4); return v;};
    auto round = [&](uint64_t acc, uint64_t in) {return rotl(acc + in * P2, 31) * P1;};
    auto merge = [&](uint64_t h, uint64_t v) {return (h ^ round(0, v)) * P1 + P4;};
    const auto * p = static_cast<const unsigned char *>(data);
    const auto * end = p + n;
    uint64_t h;
    if (n >= 32) {
        uint64_t v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        for (; p + 32 <= end; p += 32) {
            v1 = round(v1, read64(p)); v2 = round(v2, read64(p + 8));
            v3 = round(v3, read64(p + 16)); v4 = round(v4, read64(p + 24));
        }
        h = rotl(v1, 1) + rotl(v2, 7) + rotl(v3, 12) + rotl(v4, 18);
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed + P5;
    }
    h += uint64_t(n);
    for (; p + 8 <= end; p += 8) h = rotl(h ^ round(0, read64(p)), 27) * P1 + P4;
    if (p + 4 <= end) {h = rotl(h ^ (uint64_t(read32(p)) * P1), 23) * P2 + P3; p += 4;}
    for (; p < end; ++p) h = rotl(h ^ (uint64_t(*p) * P5), 11) * P1;
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}
}  // namespace tqllm

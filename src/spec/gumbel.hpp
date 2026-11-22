#pragma once
// Gumbel noise for exact sampling (token_sampling.hpp, head_sampler.cu): g(key, row) from a counter-based hash, the same
// bits on the host and on the device. key = gumbel_key(seed, position of the sampled token): every output position has
// its own noise vector, so a sampled reply depends only on the seed and the target's distributions, not on how many
// tokens a speculative step drafted or which tree was verified.
#include <cstdint>

#if defined(__CUDACC__)
#define TQLLM_HD __host__ __device__ __forceinline__
#else
#define TQLLM_HD inline
#endif

namespace tqllm {

TQLLM_HD uint64_t gumbel_mix(uint64_t z) {  // splitmix64 finaliser
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
TQLLM_HD uint64_t gumbel_key(uint64_t seed, uint32_t pos) { return gumbel_mix(gumbel_mix(seed) ^ pos); }
// 23 bits: u = (bits + 0.5) / 2^23 is exact in FP32, so the device's float draw and the host's double draw see the same u
TQLLM_HD uint32_t gumbel_bits(uint64_t key, uint32_t row) { return uint32_t(gumbel_mix(key ^ row) >> 41); }
constexpr float kGumbelUnit = 1.f / 8388608.f;  // 2^-23

}  // namespace tqllm

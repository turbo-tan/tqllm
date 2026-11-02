// Q5_0 / Q5_1 CPU decode (mmproj files in Q5_K-MIX carry Q5_0 tensors in the ViT): a block whose 32 codes are 0..31 decodes
// to (code - 16) d for Q5_0 and code d + m for Q5_1, and the GGUF block sizes are 22 / 24 bytes.
#include "dequant.hpp"
#include "gguf.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<uint8_t> block(bool q51, uint16_t d, uint16_t m) {
    std::vector<uint8_t> b;
    b.push_back(uint8_t(d)); b.push_back(uint8_t(d >> 8));
    if (q51) { b.push_back(uint8_t(m)); b.push_back(uint8_t(m >> 8)); }
    uint32_t qh = 0;
    for (int j = 0; j < 32; ++j) if (j & 16) qh |= 1u << j;  // code j: high bit = bit 4 of j
    for (int k = 0; k < 4; ++k) b.push_back(uint8_t(qh >> (8 * k)));
    for (int j = 0; j < 16; ++j) b.push_back(uint8_t((j & 15) | ((j + 16) & 15) << 4));  // low nibbles of codes j and j + 16
    return b;
}

int main() {
    int fails = 0;
    uint64_t e = 0, by = 0;
    fails += !(tqllm::gguf_block_layout(6, e, by) && e == 32 && by == 22);
    fails += !(tqllm::gguf_block_layout(7, e, by) && e == 32 && by == 24);
    float out[32];
    const auto b0 = block(false, tqllm::float_to_half(0.5f), 0);
    tqllm::dequantize_host(6, b0.data(), out, 32);
    for (int j = 0; j < 32; ++j) fails += out[j] != 0.5f * float(j - 16);
    const auto b1 = block(true, tqllm::float_to_half(0.25f), tqllm::float_to_half(2.f));
    tqllm::dequantize_host(7, b1.data(), out, 32);
    for (int j = 0; j < 32; ++j) fails += out[j] != 0.25f * float(j) + 2.f;
    std::printf("DEQUANT_Q5 %s (%d mismatches)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

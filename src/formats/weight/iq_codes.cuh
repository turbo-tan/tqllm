// Signed grid codes of the IQ2 / IQ3 formats (no includes: quant_u4h.inc pulls this into its anonymous namespace, and
// tests/iq_code_bytes.cu checks it against the per-byte __vadd4 / __vsub4 form over every grid byte and sign pattern).
// Grid bytes are 1..15 (IQ3_S) and 4..62 (IQ3_XXS, IQ2_*): base + g and base - 2 g stay inside each byte, so plain 32-bit
// arithmetic gives the same 4 code bytes as the per-byte select, in 3 instructions instead of ~12.
__device__ __forceinline__ uint32_t iq3_code(uint32_t g, uint32_t sbits) {  // 4 magnitudes, 4 sign bits -> 16 +- m
    const uint32_t neg = ((sbits * 0x00204081u) & 0x01010101u) * 0xFFu;
    return 0x10101010u + g - ((g << 1) & neg);
}
__device__ __forceinline__ uint32_t iq3x_code(uint32_t g, uint32_t sbits) {  // 4 magnitudes, 4 sign bits -> 64 +- m
    const uint32_t neg = ((sbits * 0x00204081u) & 0x01010101u) * 0xFFu;
    return 0x40404040u + g - ((g << 1) & neg);
}

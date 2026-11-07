#pragma once
// Host dequantisation of GGUF block formats used by drafter checkpoints (load time only).
#include <cstdint>
#include <functional>
#include <vector>

namespace tqllm {

// n elements of GGML type `type` (F32, F16, BF16, Q4_K, Q6_K) -> FP32.
float half_to_float(uint16_t h);
// FP32 -> FP16 bits, round to nearest even (no _Float16: GCC 12 has none in C++ on x86)
uint16_t float_to_half(float f);
// OCP E4M3 byte -> float (0x7F / 0xFF = NaN)
float e4m3_to_float(uint8_t b);
// Q4_K / Q5_K 6-bit scale and min of sub-block j (0..7) from the 12-byte scales field.
void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m);
// Q3_K: 12-byte packed scales -> 16 signed scales (6-bit value - 32).
void q3k_scales(const uint8_t * p, int8_t * sc);
void dequantize_host(uint32_t type, const void * src, float * dst, int64_t n);
// a plugin format's CPU reference (include/tqllm/plugin.h ABI 2), used for types no built-in reference covers
using HostDequant = std::function<void(const void * src, float * dst, int64_t n)>;
void register_host_dequant(uint32_t type, HostDequant f);
// The same for a [rows][cols] matrix; layouts whose rows carry their own scales (MXFP8) need the row length.
void dequantize_host_rows(uint32_t type, const void * src, float * dst, int64_t rows, int64_t cols);
// TQ3 rotation of one 32-value block (model basis -> the rotated basis TQ3 formats store and decode in)
void tq3_rotate_block(float * v);

} // namespace tqllm

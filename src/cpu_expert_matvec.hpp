#pragma once
// CPU (AVX2) expert matvec for Flash-Next GSQ-RCO miss experts.
//
// PURPOSE: the CpuMissPool (Claude, commit 4637b67, TQFN_CPU_MISS) hands a CPU pool the
// GGUF bytes of one expert matrix and needs y[t*rows+r] = dot(W_row_r, x_t) in fp32.
// The pool does its own bf16 rounding + SwiGLU, so this returns raw fp32 dots.
//
// LAYOUT (verified against the artifacts, NOT assumed):
//   GGUF-native IQ4_NL, row-major, row_bytes = cols/32*18 (2-byte fp16 scale + 16 code
//   bytes per 32 weights). Matches dequantize_host case 20 (src/dequant.cpp:380-383),
//   which strides iq4nl(s + b*18, dst + b*32).
//   Shapes from blk.ffn_*_exps.weight headers: gate/up rows=640 cols=2560,
//   down rows=2560 cols=640. (Claude's spec said 2048; hidden is 2560 -- see agora c1f99c07.)
//
// TWO KERNELS:
//   *_fp32  : exact dequant to fp32 + FMA. This is the one that must hit <=1e-5 relative
//             vs dequantize_host, because any x requantisation breaks that budget.
//   *_i16   : codes -> int8 codebook values, x quantised to int8 per 32-block, then
//             _mm256_madd_epi16. This is Claude's "int8/int16 maddubs, the way the GPU dp4a
//             paths do" request -- it is LOSSY, so it gets its own measured error budget,
//             reported separately rather than hidden inside the 1e-5 claim.
//
// WHY int16 madd and not maddubs: _mm256_maddubs_epi16 saturates its pairwise int16 sum and
// treats operand A as UNSIGNED while our codebook values are signed (-127..113), which needs
// a +128 bias and a per-block sum(kv) correction. cvtepi8_epi16 + _mm256_madd_epi16 has no
// saturation (|kv|<=127, |xq|<=127 -> pair sum <= 32258, fits int32) and no bias bookkeeping.
// Costs 2 extra instructions per 32 weights; measured, not guessed.
//
// All functions are single-threaded. The pool owns threading; see cpu_expert_pool.hpp.
#include <cstdint>

namespace tqllm { namespace cpu {

// Exact fp32 path. y[t*rows+r]. x is ntok*cols fp32 (contract: already fp16-rounded).
// ntok must be 1..4.
void iq4nl_expert_matvec_fp32(const uint8_t * W, int rows, int cols, int row_bytes,
                              const float * x, int ntok, float * y);

// Exact fp64-accumulate path: IDENTICAL dequant to the fp32 path, but the dot accumulates
// in double. Exists to separate "my dequant differs from dequantize_host" from "fp32
// summation over `cols` terms is inherently lossy". If fp64 meets the 1e-5 gate and fp32
// does not, the dequant is exact and the residual is pure accumulation error.
void iq4nl_expert_matvec_fp64(const uint8_t * W, int rows, int cols, int row_bytes,
                              const float * x, int ntok, float * y);

// Fast int8 path. Same I/O contract. LOSSY (x requantised to int8 per 32-block); its error
// is reported separately, never folded into the 1e-5 gate.
void iq4nl_expert_matvec_i16(const uint8_t * W, int rows, int cols, int row_bytes,
                             const float * x, int ntok, float * y);

// Bytes of packed weights this call will read -- for bandwidth accounting.
inline int64_t iq4nl_matrix_bytes(int rows, int cols) {
    return (int64_t)rows * (cols / 32) * 18;
}

}} // namespace tqllm::cpu

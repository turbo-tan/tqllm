#pragma once
// Quantized-weight kernels: TQ3_4S (TurboQuant 3-bit, four E3M5 scales per 32) and Q6_K, GGUF block layouts.
// Matvecs take N activation columns (N = 1..8) and read every weight byte once for all N.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda {

constexpr int TQ3_BLOCK = 32;          // weights per TQ3_4S block
constexpr int TQ3_BLOCK_BYTES = 16;    // 4 scale bytes + 12 bytes of 3-bit indices
constexpr int Q6K_BLOCK = 256;
constexpr int Q6K_BLOCK_BYTES = 210;
constexpr int QMAX_COLS = 8;

// xr[n][k] = RHT_forward(x[n][k]) per 32-element block (sign flip, Walsh-Hadamard, 1/sqrt(32)); x: [n][k] BF16.
// TQ3_4S stores weights in the rotated basis, so dot(x, dequant(w)) = dot(xr, codes * scale).
void tq3_rotate(const __nv_bfloat16 * x, float * xr, int k, int ncols, cudaStream_t);
// y[n][r] = sum_k W[r][k] x[n][k]; W: TQ3_4S [rows][k/32 blocks]; xr from tq3_rotate. y BF16.
void tq3_matvec(const void * w, const float * xr, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t);
// y[k] = dequant(W[row]) as BF16 (embedding lookup from a TQ3_4S table).
void tq3_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t);
// y[k] = dequant(W[row]) as BF16 (embedding lookup from a Q6_K table).
void q6k_row(const void * w, int row, __nv_bfloat16 * y, int k, cudaStream_t);
// Q6_K: y[n][r] = sum_k W[r][k] x[n][k]; x FP32 [n][k]; y FP32 (used for the LM head -> logits).
void q6k_matvec(const void * w, const float * x, float * y, int rows, int k, int ncols, cudaStream_t);
// ---- int8 fast path (single column) ----
// Repack TQ3_4S blocks in place into the PRMT layout used by tq3_matvec_i8 (same 16 bytes per block).
void tq3_repack(void * w, int64_t nblocks, cudaStream_t);
// xq[k] int8 + xs[k/32] FP32 scale = int8 quantisation (per 32 block, max-abs) of RHT_forward(x); x BF16 [k].
void tq3_rotate_i8(const __nv_bfloat16 * x, int8_t * xq, float * xs, int k, cudaStream_t);
// y[r] = sum_k W[r][k] x[k] with W repacked (tq3_repack) and x from tq3_rotate_i8. y BF16.
void tq3_matvec_i8(const void * w, const int8_t * xq, const float * xs, __nv_bfloat16 * y, int rows, int k,
                   cudaStream_t);
// Fused residual + RMSNorm prologue for the quantized path. Partial sums of squares are kept per 256 elements
// (k / 256 floats), so the norm is a grid-wide kernel instead of a single-block reduction.
// x += y (BF16), partial[i] = sum over x[256 i .. 256 i + 255] of x^2 (after rounding).
void add_sumsq(__nv_bfloat16 * x, const __nv_bfloat16 * y, float * partial, int k, cudaStream_t);
// partial[i] = sum of x^2 over each 256 chunk (used after the embedding lookup).
void sumsq(const __nv_bfloat16 * x, float * partial, int k, cudaStream_t);
// xq/xs = tq3_rotate_i8(bf16(x * rsqrt(sum(partial)/k + eps) * w)); w FP32 (1 + weight baked in).
void norm_rotate_i8(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, int8_t * xq,
                    float * xs, int k, cudaStream_t);
// Two-stage argmax over n FP32 values (ties -> lowest index). scratch: 2 * 1024 words.
void argmax_big(const float * x, int n, uint32_t * out, void * scratch, cudaStream_t);
// ---- multi-column (verify) path: FP16 tensor cores ----
// xh[n][k] = RHT_forward(x[n][k]) as FP16, n = 0..ncols-1 (x BF16).
void tq3_rotate_h(const __nv_bfloat16 * x, __half * xh, int k, int ncols, cudaStream_t);
// xh[n] = RHT_forward(bf16(x[n] * rsqrt(sum(partial[n]) / k + eps) * w)) as FP16; partial: [ncols][k / 256].
void norm_rotate_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k,
                   int ncols, cudaStream_t);
// y[n][r] = sum_k W[r][k] x[n][k], W repacked TQ3_4S, xh from tq3_rotate_h / norm_rotate_h, ncols 1..12. y BF16.
void tq3_matvec_h(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t);
// Test hook: <= 8 columns through one kernel variant (0 default, 1 / 2 prefetching with 1 / 2 row tiles, 4 row-tiled x4).
void tq3_matvec_h_variant(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, int variant,
                          cudaStream_t);
// Time the bit-identical <= 8-column variants for this matrix shape once and use the fastest.
void tq3_tune(const void * w, int rows, int k, cudaStream_t);
// Prefill GEMM: same as tq3_matvec_h for any number of columns (weights read once per 64 columns).
void tq3_gemm_h(const void * w, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols, cudaStream_t);
// TQ3_4S (repacked) -> FP16 [rows][k] (rotated basis, true scale); large prefill chunks run these through cuBLAS.
void tq3_dequant_h(const void * w, __half * out, int rows, int k, cudaStream_t);
// W4A16 models (u4h weights, un-rotated activations): norm_rotate_h without the rotation; u4h -> FP16 [rows][k];
// Q8_0 embedding rows (dtok on device) -> BF16.
void norm_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k, int ncols,
            cudaStream_t);
void u4h_dequant_h(const uint8_t * w4, const __half2 * sm, __half * out, int rows, int k, cudaStream_t);
// iq3x: native IQ3_XXS (12 B per 32 weights: 8 grid-index bytes + scale/sign word; 4 B per 256: d)
void iq3x_matvec_bf16(const uint8_t * w3, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                      int ncols, cudaStream_t, int ldy = 0);  // ldy: output column stride (0 = rows)
void iq3x_matvec(const uint8_t * w3, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                 cudaStream_t);
void iq3x_tune(const uint8_t * w3, const uint8_t * rec, int rows, int k, cudaStream_t);
void iq3x_dequant_h(const uint8_t * w3, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t);
// iq3: native IQ3_S (12 B per 32 weights: 8 grid-index bytes + 4 sign bytes; 16 B per 256: d, qh[8], scales[4])
void iq3_matvec_bf16(const uint8_t * w3, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t, int ldy = 0);  // ldy: output column stride (0 = rows)
void iq3_matvec(const uint8_t * w3, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t);
void iq3_tune(const uint8_t * w3, const uint8_t * rec, int rows, int k, cudaStream_t);
void iq3_dequant_h(const uint8_t * w3, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t);
// k3: native Q3_K (3-bit planes, 12 B per 32 weights + 20 B per 256: 16 int8 scales + FP16 d)
void k3_matvec_bf16(const uint8_t * w3, const uint8_t * kr, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t, int ldy = 0);  // ldy: output column stride (0 = rows)
void k3_matvec(const uint8_t * w3, const uint8_t * kr, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t);
void k3_tune(const uint8_t * w3, const uint8_t * kr, int rows, int k, cudaStream_t);
void k3_dequant_h(const uint8_t * w3, const uint8_t * kr, __half * out, int rows, int k, cudaStream_t);
// k4: native Q4_K (u4h codes + the 16-byte Q4_K superblock header per 256 weights per row)
void k4_matvec_bf16(const uint8_t * w4, const uint8_t * kh, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t, int ldy = 0);  // ldy: output column stride (0 = rows)
void k4_matvec(const uint8_t * w4, const uint8_t * kh, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t);
void k4_tune(const uint8_t * w4, const uint8_t * kh, int rows, int k, cudaStream_t);
void k4_dequant_h(const uint8_t * w4, const uint8_t * kh, __half * out, int rows, int k, cudaStream_t);
// K5 (native Q5_K): per 32 weights 16 nibble bytes in K4 order + one word of high code bits; kh = Q5_K headers.
void k5_matvec_bf16(const uint8_t * w5, const uint8_t * kh, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                    int ncols, cudaStream_t, int ldy = 0);
void k5_matvec(const uint8_t * w5, const uint8_t * kh, const __half * xh, float * y, int rows, int k, int ncols,
               cudaStream_t);
void k5_tune(const uint8_t * w5, const uint8_t * kh, int rows, int k, cudaStream_t);
void k5_dequant_h(const uint8_t * w5, const uint8_t * kh, __half * out, int rows, int k, cudaStream_t);
void i8h_dequant_h(const int8_t * w8, const __half * sc, __half * out, int rows, int k, cudaStream_t);
// y[n][rows] = W x[n] for BF16 W [rows][k] (k % 256 == 0), FP32 accumulation in a fixed order: each column is
// bit-identical for any ncols (small matrices where a cuBLAS GEMM would change algorithm with the column count).
void bf16_matvec_cols(const __nv_bfloat16 * w, const __nv_bfloat16 * x, __nv_bfloat16 * y, int rows, int k, int ncols,
                      cudaStream_t, int ldy = 0);  // ldy: output row stride (0 = rows)
// W4S: symmetric 4-bit in the u4h code layout, FP16 scale per 128 weights (sc [rows][k / 128]), w = (q - 8) s.
void w4s_matvec_bf16(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t);
void w4s_matvec(const uint8_t * w4, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t);
void w4s_dequant_h(const uint8_t * w4, const __half * sc, __half * out, int rows, int k, cudaStream_t);
// TQ4: rotated 4-bit (codes natural nibble order [rows][k / 2], FP16 scale per 128 [rows][k / 128]); inputs
// rotated per 128 by tq4_rotate_h / tq4_norm_rotate_h (Turbo4 s1 signs, Hadamard / sqrt(128)).
// int8 tensor-core prototype (per-128 int8 activations; not byte-identical to tq4_matvec_bf16)
// IQ4N: IQ4_XS in natural nibble order on the 16-row split-K kernel (<= 16 columns), pf = prefetch mode 0 / 1 / 2
void iq4n_matvec_bf16(const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                      int ncols, int pf, cudaStream_t, int ldy = 0);
void iq4n_matvec(const uint8_t * w, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols, int pf,
                 cudaStream_t);
// IQ3 (IQ3_S) on the 16-row split-K kernel (<= 16 columns)
void iq3_matvec_split_bf16(const uint8_t * w, const uint8_t * rec, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                           int ncols, int pf, cudaStream_t, int ldy = 0);
void iq3_matvec_split(const uint8_t * w, const uint8_t * rec, const __half * xh, float * y, int rows, int k, int ncols,
                      int pf, cudaStream_t);
void iq4n_dequant_h(const uint8_t * w, const uint8_t * rec, __half * out, int rows, int k, cudaStream_t);
void tq4_matvec_i8_bf16(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int ncols,
                        cudaStream_t, int mt = 2);
void tq4_matvec_bf16(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t);
void tq4_matvec(const uint8_t * w4, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, cudaStream_t);
void tq4_dequant_h(const uint8_t * w4, const __half * sc, __half * out, int rows, int k, cudaStream_t);
// TQ3: same rotation / MMA path as TQ4, 3-bit codes as uint32 planes (lo0, lo1, hi) per 32 weights of a row
void tq4b3_matvec_bf16_ld(const uint8_t * w3, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                        int ncols, int ldy, cudaStream_t);
void tq4b3_matvec(const uint8_t * w3, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols, cudaStream_t);
void tq4b3_matvec_variant(const uint8_t * w3, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                          int ncols, int v, cudaStream_t);
void tq4b3_dequant_h(const uint8_t * w3, const __half * sc, __half * out, int rows, int k, cudaStream_t);
// prefill GEMM on TQ4 / TQ3 weights (no FP16 copy of the matrix): y[c][r] (row stride ldy) = W xh[c], c < n; k % 128 == 0
void tq4_gemm_h(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n, int ldy,
                cudaStream_t);
void tq4b3_gemm_h(const uint8_t * w3, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k, int n, int ldy,
                  cudaStream_t);
// TQ4 matvec variants (row tiles / prefetch, bit-identical per column) and the per-shape pick made at load
void tq4_matvec_variant(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                        int ncols, int variant, cudaStream_t);
// y rows ldy apart ([ncols][ldy]; the first `rows` of each written): matvec straight into a wider row layout
void tq4_matvec_bf16_ld(const uint8_t * w4, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                        int ncols, int ldy, cudaStream_t);
// NVFP4 (modelopt, Qwen3.8-Flash-Next experts): codes [rows][k / 2] (E2M1, low nibble = even column), E4M3 scale per
// 16 [rows][k / 16], FP32 global scale. x: [ncols][k] BF16 -> y: [ncols][rows] FP32. Reference kernel (not tuned).
void nvfp4_matvec(const uint8_t * w, const uint8_t * sc, float gscale, const __nv_bfloat16 * x, float * y, int rows,
                  int k, int ncols, cudaStream_t);
void tq4_rotate_h(const __nv_bfloat16 * x, __half * xh, int k, int ncols, cudaStream_t);
// Fused op + TQ4 rotation (identical to the op followed by tq4_rotate_h): swiglu of gu [n][2F]; attention sigmoid gate.
void tq4_swiglu_rot_h(const __nv_bfloat16 * gu, int n, int F, __half * xh, cudaStream_t);
void tq4_swiglu_rot_from_h(const __half * gu, int n, int F, __half * xh, cudaStream_t);  // gu: FP16 (prefill GEMM)
void tq4_sigmoid_gate_rot_h(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int n, int heads, int dim, __half * xh,
                            cudaStream_t);
void tq4_norm_rotate_h(const __nv_bfloat16 * x, const float * w, const float * partial, float eps, __half * xh, int k,
                       int ncols, cudaStream_t);
void embed_multi_q8_0(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t);
void embed_multi_q4k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t);
void embed_multi_q3k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t);
void f32_to_bf16(const float * x, __nv_bfloat16 * y, int64_t n, cudaStream_t);
void h_to_bf16(const __half * x, __nv_bfloat16 * y, int64_t n, cudaStream_t);
// Per-column two-stage argmax: out[n] = argmax(x[n][0..len)).
void argmax_cols(const float * x, int len, int ncols, uint32_t * out, void * scratch, cudaStream_t);
// ---- LM head for the verify path: Q6_K re-laid out as int8 (q - 32) + FP16 scale per 16 weights ----
// Converts Q6_K [rows][k] into w8 [rows][k] int8 and sc [rows][k / 16] half (= d * scale_j, rounded to FP16).
void q6k_to_i8(const void * q6k, int8_t * w8, __half * sc, int rows, int k, cudaStream_t);
// Q5_K [rows][k] -> int8 + FP16 scale per 16 (max-abs / 127; not exact: Q5_K values are affine)
void q5k_to_i8(const void * q5k, int8_t * w8, __half * sc, int rows, int k, cudaStream_t);
// logits[n][r] = sum_k W[r][k] x[n][k], x BF16 [n][k] (not rotated), FP16 tensor cores, ncols 1..12. Output FP32.
void i8h_matvec(const int8_t * w8, const __half * sc, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t);
// FP32 [rows][k] -> int8 + FP16 scale per 16 (symmetric, max-abs / 127), the i8h_matvec format.
void i8h_quantize(const float * x, int8_t * w8, __half * sc, int64_t n, cudaStream_t);
// Same as i8h_matvec with BF16 output.
void i8h_matvec_bf16(const int8_t * w8, const __half * sc, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t, int ldy = 0);  // ldy: output column stride (0 = rows)
// Two-stage top-16 per row (rows <= 8) for long rows: stage 1 top-16 per chunk, stage 2 over the candidates.
// scratch: rows * 64 * 16 * 8 bytes.
void topk16_big(const float * x, int len, int rows, float * vals, uint32_t * idx, void * scratch, cudaStream_t);
// argmax over 16 candidate rows per column, re-scored exactly on the int8 head (w8/sc as i8h_matvec)
void i8h_rescore_argmax(const int8_t * w8, const __half * sc, const __half * xh, const uint32_t * cand, int k,
                        int ncols, uint32_t * out, cudaStream_t, float * scores = nullptr);  // scores: [ncols][16] exact logits
// u4h: w4 [rows][k / 2] 4-bit codes (group-of-16 nibble order, see quant.cu), sm [rows][k / 32] (scale, min) FP16
// pairs, w = q * scale + min. y[n][r] = sum_k W[r][k] x[n][k] (BF16 out), ncols 1..8, k % 64 == 0.
void u4h_matvec_bf16(const uint8_t * w4, const __half2 * sm, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                     int ncols, cudaStream_t, int ldy = 0);  // ldy: output column stride (0 = rows)
// Row-tiled u4h matvec with MT row tiles per CTA (test hook; bit-identical to MT = 1) and per-shape tuning of MT.
void u4h_matvec_mt(const uint8_t * w4, const __half2 * sm, const __half * xh, __nv_bfloat16 * y, int rows, int k,
                   int ncols, int mt, cudaStream_t);
void u4h_tune(const uint8_t * w4, const __half2 * sm, int rows, int k, cudaStream_t);
// Same with FP32 output (draft logits).
// rows rid[0 .. rows) of a larger u4h matrix (same results as on a gathered copy)
void u4h_matvec_ids(const uint8_t * w4, const __half2 * sm, const uint32_t * rid, const __half * xh, float * y, int rows,
                    int k, int ncols, cudaStream_t);
void u4h_matvec(const uint8_t * w4, const __half2 * sm, const __half * xh, float * y, int rows, int k, int ncols,
                cudaStream_t);
// i8h [n] -> u4h (asymmetric 4-bit per 32; lossy, for draft-only copies).
void u4h_from_i8h(const int8_t * w8, const __half * sc, uint8_t * w4, __half2 * sm, int64_t n, cudaStream_t);
// BF16 -> FP16 copy.
void bf16_to_h(const __nv_bfloat16 * x, __half * y, int n, cudaStream_t);
// BF16 -> FP32 copy (activation feed for q6k_matvec).
void bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t);

} // namespace tqllm_cuda

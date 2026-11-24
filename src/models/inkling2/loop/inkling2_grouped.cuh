#pragma once
// Inkling-2-Mini routed experts for prompt chunks (inkling2_grouped.cu): assignments grouped by expert, each expert's
// weights read once per 64 assignments, BF16 tensor-core GEMM. The weights decode to BF16 as the reference dequantises
// them (NVFP4: bf16((E2M1 x E4M3 scale) x global); MXFP8: E4M3 x 2^E8M0, exact), the products are exact in the F32
// accumulator. Same per-assignment outputs as the few-token kernels up to F32 summation order.
// Big cards (>= 96 GB) keep a BF16 copy of the expert stacks (experts_decode_bf16, the same rounding as the WMMA
// decode) and run the prompt chunks through cuBLAS per expert: gather -> gate+up GEMM -> SwiGLU -> down GEMM -> scatter.
#include "inkling2_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda::inkling2 {

constexpr int kGroupTile = 128;  // assignments per GEMM tile (128: each decoded weight tile serves twice the tokens)

// cuBLAS pays off only on large chunks: at 1K prefill (one 1024-token chunk) the gather + host count sync cost more
// than the grouped GEMM saves (1272 -> 1102 tok/s), so chunks below this keep the WMMA grouped kernels.
constexpr int kGroupedBf16MinN = 4096;
inline bool use_grouped_bf16(int n) { return n >= kGroupedBf16MinN; }

// ids [A] (A = tokens x topk) -> order [A] (assignment indices grouped by expert, ascending within an expert),
// counts / offsets [E] and tile offsets [E + 1] (tiles of kGroupTile assignments per expert)
void group_assignments(const int * ids, int A, int E, int * order, int * counts, int * offsets, int * tile_off, cudaStream_t s);
// most tiles A assignments can need
inline int group_max_tiles(int A, int E) { return (A + kGroupTile - 1) / kGroupTile + E; }

// act[a][i] = bf16(softcap_gc(g) sigmoid(g) softcap_uc(u)), g / u = bf16(W_gate / W_up [ids[a]] row i . x[a / topk])
// true when the block-scaled FP4 MMA was compiled for this GPU (else the FP4 expert path must stay off)
bool fp4_mma_available();
void grouped_gate_up(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const __nv_bfloat16 * x, int A, int E, int topk, const int * order,
                     const int * counts, const int * offsets, const int * tile_off, __nv_bfloat16 * act, float gc, float uc, cudaStream_t s);
// y[a][r] = bf16(W_down[ids[a]] row r . act[a])
void grouped_down(const ExpertMat & d, int rows, int cols, const __nv_bfloat16 * act, int A, int E, const int * order, const int * counts,
                  const int * offsets, const int * tile_off, __nv_bfloat16 * y, cudaStream_t s);

// Native NVFP4 W4A4 prompt path (Blackwell): the stored weight codes feed the FP4 MMA directly; the activations are
// quantised once per row into codes [rows][cols / 2], UE4M3 scales [rows][cols / 16] and one FP32 row scale.
void quantize_act_nvfp4(const __nv_bfloat16 * x, int rows, int cols, uint8_t * codes, uint8_t * scales, float * row_scale, cudaStream_t s);
void grouped_gate_up_fp4(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const uint8_t * xcodes, const uint8_t * xscales,
                         const float * xrow, int A, int E, int topk, const int * order, const int * counts, const int * offsets,
                         const int * tile_off, __nv_bfloat16 * act, float gc, float uc, cudaStream_t s);
// gate + up + SwiGLU, the down input written as NVFP4 codes [A][I / 2] and UE4M3 scales [A][I / 16] (no row scale; xrow may be null)
void grouped_gate_up_fp4_q(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const uint8_t * xcodes, const uint8_t * xscales,
                           const float * xrow, int A, int E, int topk, const int * order, const int * counts, const int * offsets,
                           const int * tile_off, uint8_t * acodes, uint8_t * ascales, float gc, float uc, cudaStream_t s);
void grouped_down_fp4(const ExpertMat & d, int rows, int cols, const uint8_t * xcodes, const uint8_t * xscales, const float * xrow, int A, int E,
                      const int * order, const int * counts, const int * offsets, const int * tile_off, __nv_bfloat16 * y, cudaStream_t s);

// BF16 prompt copies (big cards): decode the whole expert stack to BF16 once, out[e][r][c] with row stride ldrows
// (gate + up share one [E][2 * rows][cols] buffer: ldrows = 2 * rows, up at out + rows * cols)
void experts_decode_bf16(const ExpertMat & m, int E, int rows, int cols, int ldrows, __nv_bfloat16 * out, cudaStream_t s);
// xg[j] = x[order[j] / topk]; gu[j][: 2I] = xg[j] Wgu[e]^T (BF16 out, F32 accumulate); act[j] = softcapped SwiGLU of
// gu[j][0 .. I) / gu[j][I .. 2I) (the grouped_k epilogue's arithmetic). hc / ho: counts / offsets of group_assignments
// copied to host memory once per visit (cublasGemmEx needs M per expert).
void grouped_gate_up_bf16(const __nv_bfloat16 * Wgu, int I, int D, const __nv_bfloat16 * x, int A, int E, int topk, const int * order, const int * hc,
                          const int * ho, __nv_bfloat16 * xg, __nv_bfloat16 * gu, __nv_bfloat16 * act, float gc, float uc, cublasHandle_t blas,
                          cudaStream_t s);
// yg[j] = act[j] Wd[e]^T in grouped order (moe_combine_grouped reads it through the inverse of order)
void grouped_down_bf16(const __nv_bfloat16 * Wd, int D, int I, const __nv_bfloat16 * act, int A, int E, const int * hc, const int * ho,
                       __nv_bfloat16 * yg, cublasHandle_t blas, cudaStream_t s);

// Small cards: the same path over experts [e0, e0 + G) at a time, so the BF16 scratch holds one group, not the whole
// layer. expert_slice: m seen from expert e0 (base + stride, or the per-expert pointer arrays of partial residency).
// grouped_gather once per visit, then per group: experts_decode_bf16 of the slice, grouped_gate_up_part_bf16 (GEMM +
// SwiGLU of the group's rows, contiguous in grouped order) and grouped_down_bf16 with the group's hc / ho slices (ho are
// absolute row offsets). The GEMM descriptors live until the GEMM ran: synchronise the stream before the next group.
inline ExpertMat expert_slice(const ExpertMat & m, int e0) {
    ExpertMat r = m;
    if (m.wp) r.wp = m.wp + e0; else r.w = m.w + size_t(e0) * m.stride;
    if (m.ap) r.ap = m.ap + e0; else if (m.aux) r.aux = m.aux + size_t(e0) * m.aux_stride;
    return r;
}
void grouped_gather(const __nv_bfloat16 * x, int A, const int * order, int topk, __nv_bfloat16 * xg, int D, cudaStream_t s);
void grouped_gate_up_part_bf16(const __nv_bfloat16 * Wgu, int I, int D, const __nv_bfloat16 * xg, int G, const int * hc, const int * ho,
                               __nv_bfloat16 * gu, __nv_bfloat16 * act, float gc, float uc, cublasHandle_t blas, cudaStream_t s);
// the largest group of E, E / 2, ... 8 experts whose BF16 scratch (G * per_expert_b) still leaves every expert resident:
// experts_b + keep_b + G * per_expert_b <= free_b; none fits: 8 (the experts then share the rest, partial residency)
int choose_tqt_group(int E, size_t free_b, size_t experts_b, size_t keep_b, size_t per_expert_b);

}  // namespace tqllm_cuda::inkling2

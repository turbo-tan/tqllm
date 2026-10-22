#pragma once
// DFlash2 drafter kernels (block-diffusion speculative drafter, z-lab) and load-time TQ3_4S quantisation.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {

// FP32 weights [rows][k] -> TQ3_4S blocks (GGUF layout), following llama.cpp-tq3 quantize_row_tq3_4s_ref:
// 32-point RHT, per-8 scale fit (4 refinement passes against the codebook), E3M5 scale encoding.
void tq3_quantize(const float * x, void * out, int64_t nblocks, cudaStream_t);
// DFlash2 dynamic depthwise conv inside a block of n tokens (kernel 2, groups of `group` channels):
// y[t][c] = sum_tap (dyn[t][g + G*(tap + 2*side)] + base[c + C*(tap + 2*side)]) * h[t - tap][c], h[-1] = 0.
void dflash_conv(const __nv_bfloat16 * h, const __nv_bfloat16 * dyn, int dyn_stride, const float * base, int side,
                 __nv_bfloat16 * y, int n, int channels, int group, cudaStream_t);
// Top-16 (value, index) per row of x [rows][len], descending; ties -> lower index.
void topk16(const float * x, int len, int rows, float * vals, uint32_t * idx, cudaStream_t);
// DFlash2 selector lattice for block positions 1..n-1 (n <= 8), ranks r = 256:
// score[i][j][k] = unary[i][k] + sum_r next[cand[i][k]][r] * prev[p_ij][r] * gate[i][r], p_1j = anchor (or *anchor_dev),
// p_ij = cand[i-1][j] for i >= 2. cand hold token ids; tables BF16 [vocab][rank]; gate BF16 [n][rank].
// dst row r = src row ids[r] (row_bytes each), device ids.
void gather_rows(const void * src, void * dst, const uint32_t * ids, int n, int64_t row_bytes, cudaStream_t);
// Tree commit: in every described buffer, move row cols[d] to row d (d = 1 .. k - 1; cols[0] = 0, cols ascending).
struct PathGather {
    void * base;
    int64_t row_bytes;  // multiple of 16
    int use_pos;        // rows start at base + pos * row_bytes (KV caches)
};
void gather_path(const PathGather * descs, int n, const int * cols, int k, int pos, cudaStream_t);
// Device-side best-first token tree (anchor + budget nodes) from the DFlash2 lattice: tok / par / dep for the verify.
void dflash_tree(const uint32_t * idx, const float * score, const uint32_t * anchor, int B, int budget, uint32_t * tok,
                 int * par, int * dep, cudaStream_t);
// Device-side greedy chain (want tokens) from the DFlash2 lattice, the same picks as the host chain.
void dflash_chain(const uint32_t * idx, const float * score, int want, uint32_t * tok, cudaStream_t);
// idx[i] = ids[idx[i]] (draft-vocabulary row -> token id, on the device).
void map_ids(uint32_t * idx, const uint32_t * ids, int n, cudaStream_t);
void dflash_lattice(const uint32_t * cand, const float * unary, const __nv_bfloat16 * gate, int gate_stride,
                    const __nv_bfloat16 * next, const __nv_bfloat16 * prev, uint32_t anchor, int n, int rank,
                    float * score, cudaStream_t, const uint32_t * anchor_dev = nullptr);

} // namespace tqllm_cuda

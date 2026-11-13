#pragma once
// QSA (Qwen Sparse Attention, arXiv 2608.30320) indexer for the dense Qwen3.8-27B verify path, with the indexers of
// higa2515/Qwen3.8-27B-QSA-BF16: per full-attention layer a 4-head x 128 indexer scores 4-token key blocks and a row
// attends to its top 512 blocks (2,048 keys) in place of the cache part of the two-part verify attention
// (keys below tq_split_point(qpos)); the keys from there to the row stay dense (attn_blk_k). A row is sparse when its
// split point is >= `from`. Every value of a row depends on its own query and the committed cache only, in a fixed
// order, so a verify row equals its one-token step bit for bit. Same key maths as the Metal backend
// (src/backends/metal/attention_qsa.hpp): token key = RMS norm (k_norm) of the key projection, FP16; block key = mean
// of 4 token keys, RMS norm (block_norm), RoPE on the first 64 dims at the block start.
// Positions come from device memory (dpos[0] = the block's first position, dep = node depths or null for a chain), so
// the launches can be captured in a CUDA graph; grids depend on n and the context only.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda {

constexpr int kQsa27Block = 4, kQsa27Pick = 512, kQsa27Heads = 4, kQsa27Dim = 128, kQsa27Rope = 64;
constexpr int kQsa27Rows = (kQsa27Heads + 1) * kQsa27Dim;  // projection rows: 4 query heads, then the key

int qsa27_split_point(int qpos);  // tq_split_point (common.cuh) for host code

// y[i][r] = sum_j w[r0 + r][j] * x[i][j] for n rows of x [n][hidden] (hidden % 256 == 0), y [n][rows] FP32
void qsa27_project(const __half * x, const __half * w, float * y, int r0, int rows, int n, int hidden, cudaStream_t);
// token keys of n rows -> tok[dpos[0] + i]: proj [n][stride], the key at column `off`
void qsa27_token_keys(const float * proj, int stride, int off, const float * k_norm, __half * tok, const int * dpos, int n,
                      float eps, cudaStream_t);
// block keys [first, first + count) from the token keys of their 4 positions
void qsa27_pool_blocks(const __half * tok, const float * block_norm, float * blocks, int first, int count, float theta,
                       float eps, cudaStream_t);
// indexer queries of n rows: proj [n][stride] (4 x 128 first) -> q [n][4][128], normed and rotated at the row's position
void qsa27_queries(const float * proj, int stride, const float * q_norm, float * q, const int * dpos, const int * dep, int n,
                   float theta, float eps, cudaStream_t);
// top 512 blocks of each sparse row among the blocks below its split point, ascending (ties: the lower block); rows that
// are not sparse get -1. bits: scratch [n][max_blocks]; sel [n][512]
void qsa27_select(const float * q, const float * blocks, uint32_t * bits, int32_t * sel, const int * dpos, const int * dep,
                  int n, int from, int max_blocks, cudaStream_t);
// staged host KV tail: copy the rows' selected positions whose whole K / V row lies in the host copy (hk / hv, the first
// evicted[0] / evicted[1] bytes of the cache) back into the cache (dk / dv: the staging copy), each position once per call
// (mark [max_blocks], tag: device scratch, mark starts at -1). krow / vrow: bytes per position (all KV heads, multiples of 16)
void qsa27_stage_selected(const int32_t * sel, int n, const int * dpos, const int * dep, int from, const void * hk, const void * hv,
                          void * dk, void * dv, const int64_t * evicted, int krow, int vrow, int * mark, int * tag, cudaStream_t);

}  // namespace tqllm_cuda

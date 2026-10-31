#pragma once
// MiniMax-M3.1-Flash kernels: swigluoai, partial NeoX RoPE, GQA attention over an FP32 KV cache. FP32 activations.
// The reference is tools/minimax_ref.py (HF MiniMax-M3 on our dequant); tests/minimax.cu checks each kernel on the CPU.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {
// out = (clamp(u, -limit, limit) + 1) * g' * sigmoid(alpha g'),  g' = min(g, limit)
void mm_swigluoai_f32(const float * g, const float * u, __half * out, int n, float alpha, float limit, cudaStream_t s);
void mm_swigluoai_bf16(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * out, int n, float alpha, float limit,
                       cudaStream_t s);
// x [tokens][heads][dim], rotate the first `rot` dims of every head (NeoX halves: j with j + rot/2) at position pos0 + t;
// inv_freq[i] = theta^(-2i/rot)
void mm_rope(float * x, int tokens, int heads, int dim, int rot, int pos0, float theta, cudaStream_t s);
// dst rows [row0, row0 + n) = src [n][width]
void mm_put_rows(const float * src, float * dst, int width, int n, int row0, cudaStream_t s);
// causal GQA: token t (position pos0 + t) of q [n][hq][dim] attends to cache rows [0, pos0 + t]; k / v [rows][hk][dim];
// out [n][hq][dim]. dim must be 128.
void mm_attn(const float * q, const float * k, const float * v, float * out, int n, int hq, int hk, int dim, int pos0,
             float scale, cudaStream_t s);
// FP16 KV rows: dst rows [row0, row0 + n) = src [n][width] (FP32 in)
// KV cache rows: FP16, or FP8 E4M3 (fp8 = true; saturating) when the context only fits at half the bytes
void mm_put_rows_h(const float * src, void * dst, int width, int n, int row0, cudaStream_t s, bool fp8 = false);
// the lightning indexer's block scores: token t (position pos0 + t) against index keys [rows][dim]: per key the max over
// the index heads of q . k (no scale), future keys -inf; per block of `block` keys the max; the token's own block (and the
// `local` - 1 blocks before it) +inf. iq [n][heads][dim], out [n][nblocks]
// the same scores for a prompt chunk on tensor cores (4 index heads, FP16 products): prefill only
void mm_index_scores_mma(const float * iq, const void * ik, float * out, int n, int heads, int dim, int pos0, int block,
                         int nblocks, int local, cudaStream_t s, bool fp8 = false);
void mm_index_scores(const float * iq, const void * ik, float * out, int n, int heads, int dim, int pos0, int block,
                     int nblocks, int local, cudaStream_t s, bool fp8 = false);
// per token: the indices of the top-k blocks (ties: lower index first), -1 for slots past the blocks with a finite score
// (or past nblocks). scores [n][nblocks], ids [n][k]
void mm_topk_blocks(const float * scores, int * ids, int n, int nblocks, int k, cudaStream_t s);
// causal GQA over FP16 K / V, restricted to the keys of each token's selected blocks (ids [n][k], -1 ends the list;
// null: every key [0, pos0 + t]); same arithmetic as mm_attn over that key set
void mm_attn_h(const float * q, const void * k, const void * v, float * out, int n, int hq, int hk, int dim, int pos0,
               float scale, const int * ids, int topk, int block, cudaStream_t s, bool fp8 = false);
// tensor-core prompt attention for 16 query heads per KV head: the keys of mm_attn_hg, FP16 products (prefill only)
void mm_attn_mma(const float * q, const void * k, const void * v, float * out, int n, int hq, int hk, int dim, int pos0,
                 float scale, const int * ids, int topk, int block, cudaStream_t s, bool fp8 = false);
// the same attention for a prompt chunk: the query heads of one KV head share staged keys / values
void mm_attn_hg(const float * q, const void * k, const void * v, float * out, int n, int hq, int hk, int dim, int pos0,
                float scale, const int * ids, int topk, int block, cudaStream_t, bool fp8 = false);
// DSpark draft attention: query offset o (position anchor + o) of q [B][hq][dim] attends to the context keys
// [max(0, anchor + o - window + 1), anchor) of kc / vc ([rows][hk][dim] FP16, or FP8 E4M3 with fp8) and to every one of the B block keys
// kn / vn ([B][hk][dim] FP32): bidirectional inside the block. out [B][hq][dim]
void mm_attn_draft(const float * q, const void * kc, const void * vc, const float * kn, const float * vn, float * out,
                   int B, int hq, int hk, int dim, int anchor, int window, float scale, cudaStream_t s, bool fp8 = false);
// out[j] = table[*id][j] for j < width (the id is read on the device)
void mm_gather_row(const float * table, const int * id, float * out, int width, cudaStream_t s);
// a[j] += b[j]
void mm_add(float * a, const float * b, int n, cudaStream_t s);
}  // namespace tqllm_cuda

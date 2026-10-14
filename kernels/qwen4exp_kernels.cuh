#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace tqllm_cuda {
// F32 residual and norm tensors use [tokens][streams][hidden], matching reference HC math.
void hc_init(const float * embedding, float * residual, int hidden, int streams, int tokens, cudaStream_t);
void hc_norm(const float * residual, const float * gamma, float * normalized,
              int hidden, int streams, int tokens, float epsilon, cudaStream_t);
void hc_low_silu(float * low_rank, int count, int streams, cudaStream_t);
void hc_mix(const float * normalized, const float * gate_logits, float * mixed,
             int hidden, int streams, int tokens, cudaStream_t);
void hc_combine(const float * residual, const float * block_out, const float * inject_logits, float * out,
                 int hidden, int streams, int tokens, cudaStream_t);
void ple_gate(const float * key_norm, const float * query_norm, float * gates,
               int hidden, int streams, int tokens, cudaStream_t);
void ple_broadcast(const float * value, const float * gates, float * gated,
                    int hidden, int streams, int tokens, cudaStream_t);
// History is oldest-first [(kernel-1)*dilation][channels]. All buffers must be disjoint.
void ple_conv(const float * normalized, const float * history, const float * kernel, float * out, float * next_history,
               int channels, int tokens, int kernel_size, int dilation, cudaStream_t);
void ple_apply(const float * hidden, const float * gated, const float * conv, float * out, int count, cudaStream_t);
// Packed Q/K/V [tokens][2*kheads*dim + vheads*dim]; state [vheads][value][key].
// Inputs and recurrent state stay F32. Key heads use GGUF's tiled order (h % kheads).
void fn_gdn(const float * qkv, const float * alpha, const float * beta, const float * neg_a,
            const float * dt_bias, const float * state, float * next_state, float * out,
            int kheads, int vheads, int dim, int tokens, float epsilon, cudaStream_t);
// Same as fn_gdn, also writing the state after every token to states[t][vheads][value][key] (speculative rollback).
// Prefills of at least n tokens use the chunked gated-delta kernel (0: never; default TQFN_GDN_CHUNK_MIN or 64). Tests.
void fn_gdn_set_chunk_min(int n);
void fn_gated_rms(const float * x, const float * gate, const float * weight, float * out,
                  int heads, int dim, int tokens, float epsilon, cudaStream_t);
// Text-only positions. An interleaved query input stores [q|gate] per head.
void fn_norm_rope(const float * x, const float * weight, float * out, int heads, int dim,
                  int tokens, int position, int rotary, float theta, float epsilon, bool interleaved, cudaStream_t);
// Correctness path below the indexer's budget; refuses longer contexts instead of omitting QSA.
void fn_dense_attention(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                        float * out, int heads, int kvheads, int dim, int tokens, int position, cudaStream_t);
void fn_attention_gate(float * out, const float * query_gate, int heads, int dim, int tokens, cudaStream_t);
// Decode elementwise ops (device-resident step): routed SwiGLU BF16 -> FP16, shared SwiGLU F32, and
// out[t] = sum_k w[t][k] * routed[t][k] + shared_w[t] * shared[t].
// Single-token BF16 GEMV: y[r] = sum_c W[r][c] * bf16(x[c]), F32 accumulate. One warp per row; cols % 8 == 0.
void fn_gemv_bf16(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, cudaStream_t);
// tokens columns at once ([tokens][cols] -> [tokens][rows]); weights read once per 4 tokens, per-token sums as fn_gemv_bf16.
void fn_gemv_bf16_tokens(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, int tokens, cudaStream_t);
// Prefill chunks on the block-per-row layout (rows <= 16, or rows <= 1024 with cols >= 1024): sums bit-identical to
// fn_gemv_bf16_tokens, x read once per 8 rows; output row t at y + t * ldy.
void fn_gemv_bf16_chunk(const __nv_bfloat16 * w, const float * x, float * y, int rows, int cols, int tokens, int64_t ldy, cudaStream_t);
constexpr int kGemvSegs = 4;
struct GemvSegs { int n = 0; int start[kGemvSegs + 1] = {}; const __nv_bfloat16 * w[kGemvSegs] = {}; float * y[kGemvSegs] = {}; int64_t ld[kGemvSegs] = {}; };
// Only for row-blocks that fn_gemv_bf16_tokens runs on its block-per-row path (rows <= 16, or rows <= 1024 with cols >= 1024).
void fn_gemv_bf16_segments(const GemvSegs & g, const float * x, int cols, int tokens, cudaStream_t);
void fn_swiglu_bf16_h(const __nv_bfloat16 * gate, const __nv_bfloat16 * up, __half * out, int64_t count, cudaStream_t);
void fn_swiglu_f32(const float * gate, const float * up, float * out, int64_t count, cudaStream_t);
void fn_moe_combine(const __nv_bfloat16 * routed, const float * weights, const float * shared, const float * shared_weights,
                    float * out, int hidden, int topk, int tokens, cudaStream_t);

// Keys and query are already normalized and rotated: [blocks][dim], [heads][dim].
// Rectify each head's dot product before summing its contribution to a block.
void qsa_score_blocks(const float * pooled_keys, const float * query, float * scores,
                      int blocks, int heads, int dim, cudaStream_t);
void qsa_norm_rope_blocks(const float * pooled, const float * weight, float * normalized,
                          int first_block, int blocks, int ratio, int dim, int rotary,
                          float theta, float epsilon, cudaStream_t);

// Raw indexer keys stay on device. Completed blocks are updated only when new
// keys make them complete; truncate invalidates later blocks for speculative rollback.
class QsaRawKeyCache {
public:
    QsaRawKeyCache(int max_cells, int ratio, int dim);
    ~QsaRawKeyCache();
    QsaRawKeyCache(const QsaRawKeyCache &) = delete;
    QsaRawKeyCache & operator=(const QsaRawKeyCache &) = delete;
    void append(const float * keys, int tokens, cudaStream_t stream);
    void normalize_new(const float * weight, int rotary, float theta, float epsilon, cudaStream_t stream);
    void truncate(int cells);
    int size() const { return size_; }
    int complete_blocks() const { return size_ / ratio_; }
    const float * raw() const { return raw_; }
    const float * pooled() const { return pooled_; }
    const float * normalized() const { return normalized_; }

private:
    int max_cells_, ratio_, dim_, size_ = 0;
    float * raw_ = nullptr;
    float * pooled_ = nullptr;
    float * normalized_ = nullptr;
    int normalized_blocks_ = 0;
};

// Persistent scratch for one stream. Input scores cover complete ratio-sized blocks; the
// incomplete tail is selected first. Output positions are logical sequence positions.
class QsaSelector {
public:
    QsaSelector(int max_cells, int ratio, int top_k);
    ~QsaSelector();
    QsaSelector(const QsaSelector &) = delete;
    QsaSelector & operator=(const QsaSelector &) = delete;
    int select(const float * block_scores, int cells, int query_position,
               int32_t * positions, uint8_t * valid, cudaStream_t stream);

private:
    int max_cells_, ratio_, top_k_;
    uint64_t * keys_ = nullptr;
    uint64_t * sorted_ = nullptr;
    void * temp_ = nullptr;
    size_t temp_bytes_ = 0;
};

struct QsaDeviceSelection {
    const int32_t * positions = nullptr;
    const uint8_t * valid = nullptr;
    int width = 0;
};

// Flat all-device BF16 KV baseline. Invalid selected entries are zero-filled;
// attention must still apply the returned validity mask.
void qsa_gather_flat_bf16(const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                          const QsaDeviceSelection & selected, __nv_bfloat16 * gathered_keys,
                          __nv_bfloat16 * gathered_values, int cells, int heads, int dim, cudaStream_t);
void qsa_attention_selected(const float * query, const __nv_bfloat16 * gathered_keys,
                            const __nv_bfloat16 * gathered_values, const uint8_t * valid,
                            float * output, int width, int heads, int kv_heads, int dim, cudaStream_t);

// Scratch for selecting a chunk of queries at once; one instance can serve every layer on one stream.
class QsaBatchScratch {
public:
    QsaBatchScratch(int max_queries, int max_cells, int ratio, int top_k);
    ~QsaBatchScratch();
    QsaBatchScratch(const QsaBatchScratch &) = delete;
    QsaBatchScratch & operator=(const QsaBatchScratch &) = delete;
    int max_queries, max_blocks, width_cap;
    float * query = nullptr;
    uint64_t * keys = nullptr;
    uint64_t * sorted = nullptr;
    int * offsets = nullptr;
    void * temp = nullptr;
    size_t temp_bytes = 0;
    int32_t * positions = nullptr;  // [max_queries][width_cap]
    uint8_t * valid = nullptr;
};

// Selected attention reading K/V cells in place (no gather): query/output [tokens][heads][dim], selection rows
// [tokens][width]; token t sees cells below first_cells + t. Same arithmetic as gather + qsa_attention_selected.
// Indexer block scores packed as sort keys (score bits | inverted block) for each of `tokens` queries; fixed = 1 the
// 4-head x 128 kernel, 0 the generic one (bit-identical keys), -1 by TQFN_QSA_FIXED.
void qsa_score_pack(const float * pooled_keys, const float * query, uint64_t * keys, int blocks, int first_position, int ratio,
                    int heads, int dim, int tokens, cudaStream_t stream, int fixed = -1);
void qsa_attention_direct(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                          const QsaDeviceSelection & selected, float * output, int tokens, int first_cells,
                          int heads, int kv_heads, int dim, cudaStream_t);

// Same results as qsa_attention_direct, K/V rows read once per kv head: scores [tokens][heads][width] and inv
// [tokens][heads] are caller scratch.
void qsa_attention_grouped(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                           const QsaDeviceSelection & selected, float * scores, float * inv, float * output, int tokens,
                           int first_cells, int heads, int kv_heads, int dim, cudaStream_t);

// Single-sequence text path. The caller supplies projected raw indexer keys
// and query; output is borrowed until the next select on this instance.
class QsaIndexer {
public:
    QsaIndexer(int max_cells, int ratio, int top_k, int heads, int dim, int rotary,
               float theta, float epsilon);
    ~QsaIndexer();
    QsaIndexer(const QsaIndexer &) = delete;
    QsaIndexer & operator=(const QsaIndexer &) = delete;
    void append(const float * raw_keys, int tokens, const float * key_norm, cudaStream_t stream);
    void truncate(int cells) { keys_.truncate(cells); }
    QsaDeviceSelection select(const float * raw_query, const float * query_norm,
                              int query_position, cudaStream_t stream);
    // Queries at first_position .. first_position + tokens - 1 in one pass; rows of the result are strided by its width.
    QsaDeviceSelection select_batch(const float * raw_queries, const float * query_norm, int first_position,
                                    int tokens, QsaBatchScratch & scratch, cudaStream_t stream);
    int size() const { return keys_.size(); }

private:
    QsaRawKeyCache keys_;
    QsaSelector selector_;
    int ratio_, heads_, dim_, rotary_;
    float theta_, epsilon_;
    float * query_ = nullptr;
    float * scores_ = nullptr;
    int32_t * positions_ = nullptr;
    uint8_t * valid_ = nullptr;
};
} // namespace tqllm_cuda

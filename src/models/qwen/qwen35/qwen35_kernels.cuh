#pragma once
// Qwen3.5 (qwen3_5) decode kernels. Numerics follow transformers 5.17 modeling_qwen3_5.py.
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {

// Turbo3 V cache record for one (position, kv head) of a head_dim-256 cache: GGML TURBO3_0 math (per 128 values:
// norm, sign flip + WHT rotation, 8-centroid codebook, corrected FP16 norm) in a layout made for the verify kernel.
// 32-value chunk c (dims 32c .. 32c + 31) is 96 bits in w[3c .. 3c + 2]; dim 32c + 8j + l (l = 0..7, j = 0..3) is
// the 3-bit field at bit 12 l + 3 j of the chunk. Field = (positive << 2) | rank, value = +-centroid_mag[rank] * norm
// (GGML index i: positive = i >= 4, rank = positive ? i - 4 : 3 - i).
struct Turbo3V256 {
    __half norm[2];  // per 128-value half
    uint32_t w[24];
};
static_assert(sizeof(Turbo3V256) == 100);

// Turbo4 K cache record for one (position, kv head), head_dim 256: per 128 values the Turbo3 rotation (sign flip,
// WHT, sign flip) of the unit vector, a 16-level codebook stored as int8 levels {6,18,31,44,58,75,96,127}
// x sign, and a corrected FP16 scale (value = level * scale in the rotated basis; ||k|| preserved). Nibble =
// (negative << 3) | rank. Word 4 kb + t (kb = 32-dim block, t = 0..3): low 16 bits dims 32 kb + 4t .. +3,
// high 16 bits dims 32 kb + 16 + 4t .. +3 (one word = one lane's int8 MMA operand pair).
struct Turbo4K256 {
    __half scale[2];
    uint32_t w[32];
};
static_assert(sizeof(Turbo4K256) == 132);
// llama.cpp-tq3 TQ3_0 for a 256-dim KV row: 8 blocks of 32 values, each rotated (TQ3_0 signs, 32-point WHT / sqrt(32))
// and quantised to the 8 TQ3_0 levels with the block's fp16 RMS. Block b's 32 codes: w[3b .. 3b + 2], code i
// at bits 3i (96-bit little endian), i.e. 12-bit groups of 4 codes.
struct Tq3KV256 {
    __half rms[8];
    uint32_t w[24];
};
static_assert(sizeof(Tq3KV256) == 112);

// 2-bit V cache record for a 256-dim row: eight fp16 RMS values and 32 two-bit codes per 32-value block.
struct Tq2V256 {
    __half rms[8];
    uint32_t w[16];
};
static_assert(sizeof(Tq2V256) == 80);

// Rotated 8-bit / 4-bit KV rows (same TQ3_0 rotation per 32 values): fp16 scale d per block, value = q * d (q8) or
// (q - 8) * d (q4, value i in nibble i % 2 of byte i / 2, low nibble first).
struct Q8KV256 {
    __half d[8];
    int8_t q[256];
};
static_assert(sizeof(Q8KV256) == 272);
struct Q4KV256 {
    __half d[8];
    uint8_t q[128];
};
static_assert(sizeof(Q4KV256) == 144);
// Trellis-coded 2-bit V row (TCQ; idea from spiritbuun/buun-llama-cpp, own implementation): two 128-value groups of 9 words. Group bits 0..261:
// 6 initial-state bits, then 128 two-bit symbols; value t = codebook[bits 2t .. 2t + 7] * norm in the group's 128-point
// WHT basis. Bits 272..287 (high half of word 8): fp16 norm.
struct Tcq2V256 {
    uint32_t w[18];
};
static_assert(sizeof(Tcq2V256) == 72);
// Trellis-coded 3-bit K row: two 128-value groups of 13 words; group bits 0..389: 6 initial-state bits, then 128 three-bit symbols;
// value t = codebook[bits 3t .. 3t + 8] * norm in the group's 128-point WHT basis; bits 400..415 (high half of word 12): fp16 norm.
struct Tcq3K256 {
    uint32_t w[26];
};
static_assert(sizeof(Tcq3K256) == 104);

// RMSNorm over `rows` vectors of length n: y = bf16(x * rsqrt(mean(x^2) + eps) * w), FP32 math. Qwen3.5's
// zero-centred norm is this with w = 1 + weight, baked in at load (GGUF files already store 1 + weight).
// Row r reads x + r*x_stride and writes y + r*y_stride (used to peel q out of the interleaved [q | gate] layout).
void rms_f(const __nv_bfloat16 * x, int x_stride, const float * w, __nv_bfloat16 * y, int y_stride, int n, int rows,
           float eps, cudaStream_t);
// RoPE on the first `rot` dims of each head (rotate-half within those dims), cos/sin rounded to BF16 as in HF.
// RoPE scaling for every qwen35 rotation: frequency factor per rotary pair (pairs <= 256; null: 1) and cos / sin scale
void set_rope_scaling(const float * mul, int pairs, float mscale);
// KV host tail staging: copies *nbytes (device, multiple of 16; read at run time so captured graphs stay valid)
// off (device, bytes, multiple of 16; null = 0): bytes below it are not copied (a TqCache-compacted prefix)
void kv_stage_copy(const void * src, void * dst, const int64_t * nbytes, int blocks, cudaStream_t, const int64_t * off = nullptr);
void rope_partial(__nv_bfloat16 * x, int heads, int dim, int rot, int pos, float theta, cudaStream_t);
// o[h*dim + d] *= sigmoid(qg[h*2*dim + dim + d])  (gated attention output)
void sigmoid_gate(__nv_bfloat16 * o, const __nv_bfloat16 * qg, int heads, int dim, cudaStream_t);
// Causal depthwise conv (kernel 4) decode step + SiLU. state: [3][channels] BF16, oldest first; updated in place.
// w: [channels][4] FP32.
void conv_step(const __nv_bfloat16 * x, __nv_bfloat16 * state, const float * w, __nv_bfloat16 * y, int channels,
               cudaStream_t);
// Same, reading the state from `in` and writing the shifted state to `out` (verify path: per-token state slots).
void conv_step2(const __nv_bfloat16 * x, const __nv_bfloat16 * in, __nv_bfloat16 * out, const float * w,
                __nv_bfloat16 * y, int channels, cudaStream_t);
// Gated delta rule decode step. qkv: conv output [q(kh*dk) | k(kh*dk) | v(vh*dv)]; a, b: [vh]; neg_a = -exp(A_log),
// dt_bias: [vh] FP32; state: [vh][dk][dv] FP32. Value head h reads key head h / (vh/kh) (HF "grouped" order) or
// h % kh ("tiled" order, as written by llama.cpp's GGUF converter).
void gdn_step(const __nv_bfloat16 * qkv, const __nv_bfloat16 * a, const __nv_bfloat16 * b, const float * neg_a,
              const float * dt_bias, float * state, __nv_bfloat16 * out, int kheads, int vheads, int dk, int dv,
              bool tiled, cudaStream_t);
// Same, reading the recurrent state from `in` and writing the updated state to `out`.
void gdn_step2(const __nv_bfloat16 * qkv, const __nv_bfloat16 * a, const __nv_bfloat16 * b, const float * neg_a,
               const float * dt_bias, const float * in, float * out, __nv_bfloat16 * y, int kheads, int vheads, int dk,
               int dv, bool tiled, cudaStream_t);
// ---- multi-token (verify) versions ----
// Causal conv over n tokens: x[i] = xin + i * x_stride (C channels); state read from `in`, final state (after all n)
// written to `out` (may alias `in`); y [n][C].
// Deferred commit (tree verify): the previous step's accepted path is replayed inside the next verify's kernels first
// (pend tokens, device count; 0 = none), its state written to smid, then the tree runs from that state. The final state
// after the tree is not written in this mode.
struct ConvReplay {
    const __nv_bfloat16 * x = nullptr;  // pend rows of conv inputs, same stride as xin
    const int * pend = nullptr;
    __nv_bfloat16 * smid = nullptr;     // conv state after the replay
    __nv_bfloat16 * y = nullptr;        // pend rows of conv outputs (the GDN replay's q / k / v)
};
struct GdnReplay {
    const __nv_bfloat16 * qkv = nullptr;  // pend rows of conv outputs
    const __nv_bfloat16 * a0 = nullptr;
    const __nv_bfloat16 * b0 = nullptr;
    int abs_ = 0;
    const int * pend = nullptr;
    float * smid = nullptr;               // recurrent state after the replay
};
void conv_multi(const __nv_bfloat16 * xin, int x_stride, int n, const __nv_bfloat16 * in, __nv_bfloat16 * out,
                const float * w, __nv_bfloat16 * y, int channels, cudaStream_t, const int * par = nullptr,
                const ConvReplay & rp = {});
// Gated delta rule over n tokens: qkv [n][qkv_dim] (conv output), a[i] = a0 + i * ab_stride, b likewise; state read
// from `in`, final state written to `out` (may alias); y [n][vh * dv]. Each thread keeps one state column in registers.
void gdn_multi(const __nv_bfloat16 * qkv, int n, const __nv_bfloat16 * a0, const __nv_bfloat16 * b0, int ab_stride,
               const float * neg_a, const float * dt_bias, const float * in, float * out, __nv_bfloat16 * y, int kheads,
               int vheads, int dk, int dv, bool tiled, cudaStream_t, const int * par = nullptr,
               const GdnReplay & rp = {});
// Batched helpers for n tokens whose vectors sit `ts` elements apart ("token stride").
// RMSNorm of `rows` vectors (length len) per token: x of token i, row r at x + i*xts + r*xrs; y at y + i*yts + r*yrs.
void rms_f_multi(const __nv_bfloat16 * x, int xts, int xrs, const float * w, __nv_bfloat16 * y, int yts, int yrs,
                 int len, int rows, int n, float eps, cudaStream_t);
// Partial RoPE for n tokens at positions pos0 + i; x of token i at x + i * ts.
// mpos (optional, [n][4]: unused, t, h, w): interleaved M-RoPE per token with sections sec (Qwen3-VL image tokens);
// positions then come from mpos instead of pos0 / dep.
// rms_f_multi (len == head dim, one row per block) + rope_partial_multi of the result: same arithmetic, one kernel.
void rms_rope_multi(const __nv_bfloat16 * x, int xts, int xrs, const float * w, __nv_bfloat16 * y, int yts, int yrs, int len,
                    int rows, int n, float eps, int rot, const int * dpos0, float theta, cudaStream_t, const int * dep = nullptr,
                    const int * mpos = nullptr, int3 sec = {0, 0, 0});
// DFlash retrieval window: scores 16-token far chunks (positions < pos - recent) against the anchor query, keeps the top
// `chunks` per KV head, and builds a contiguous scratch K/V [R retrieved (re-rotated) | recent | nq]; returns R (or -1
// when there is no far context). Attend the scratch with query position R + recent and window R + recent + nq.
// Only the newest kDflRetrMaxChunks chunks are scored, so kc / vc may be a ring (position p at row p % ring, ring a
// multiple of kDflRetrChunk; 0 = full context) that holds kDflRetrMaxChunks x kDflRetrChunk + recent + nq positions.
// score: kvheads x kDflRetrMaxChunks floats, sel: kvheads x min(chunks, 128) ints, sk / sv: (128 x 16 + recent + nq) x kd.
constexpr int kDflRetrChunk = 16, kDflRetrMaxChunks = 4096;
int dflash_retrieve_build(const __nv_bfloat16 * q, const __nv_bfloat16 * kc, const __nv_bfloat16 * vc, int kd, int heads,
                          int kvheads, int pos, int nq, int recent, int chunks, float theta, float * score, int * sel,
                          __nv_bfloat16 * sk, __nv_bfloat16 * sv, cudaStream_t s, int ring = 0);
void rope_partial_multi(__nv_bfloat16 * x, int ts, int n, int heads, int dim, int rot, const int * dpos0, float theta,
                        cudaStream_t, const int * dep = nullptr, const int * mpos = nullptr, int3 sec = {0, 0, 0});
// KV cache write for n tokens at positions pos0 + i; k/v of token i at k0/v0 + i * ts. ring > 0: slot = pos % ring.
void cache_write_multi(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, __nv_bfloat16 * kc,
                       __nv_bfloat16 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t, int ring = 0);
// int8 KV cache write (q8_0 style): one FP16 max-abs scale per 32 values, ksc / vsc [pos][kv head][dim / 32].
void cache_write_multi_q8(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, int8_t * kc, int8_t * vc,
                          __half * ksc, __half * vsc, const int * dpos0, int kvheads, int dim, cudaStream_t);
void cache_write_multi_q8_turbo3v(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, int8_t * kc,
                                  __half * ksc, Turbo3V256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t);
// Quality experiments: quantise -> dequantise n rows of K or V (row i at x + i * ts) in place with a KV format:
// 1 = Turbo3, 2 = Turbo4 (int8 16-level), 3 = llama.cpp-tq3 TQ3_0 (32-value blocks). Other modes: no-op.
void kv_fakequant(__nv_bfloat16 * x, int ts, int n, int kvheads, int dim, int mode, cudaStream_t);
// Turbo4 K + Turbo4 V (V records use the PV layout described in qwen35.cu).
// TQ3_0 K and V (Tq3KV256 records). Attention reads them in the rotated basis; see attention_tq3kv.
// TQ3_0 K + V attention. nq * group <= 64: verify shape (4 warps, nsplit key splits, part = attention_mq_partial_floats);
// larger nq: prefill shape (8 warps, 128 rows per block, no split). y: BF16, rotation already undone.
void attention_tq3kv(const __nv_bfloat16 * q, const Tq3KV256 * kc, const Tq3KV256 * vc, __nv_bfloat16 * y, float * part,
                     const int * dseq0, int nq, int heads, int kvheads, float scale, int nsplit, cudaStream_t);
void cache_write_multi_tq3kv(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Tq3KV256 * kc,
                             Tq3KV256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t);
void attention_tq3k_tq2v(const __nv_bfloat16 * q, const Tq3KV256 * kc, const Tq2V256 * vc, __nv_bfloat16 * y,
                         float * part, const int * dseq0, int nq, int heads, int kvheads, float scale, int nsplit,
                         cudaStream_t);
void cache_write_multi_tq3k_tq2v(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Tq3KV256 * kc,
                                 Tq2V256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t);
// Split cache: K and V codecs chosen independently (same attention kernel, codec policies per side; head_dim 256).
// Tq3: TQ3_0 record (Tq3KV256). Tq2s: scalar 2-bit V record (Tq2V256), V only. Q8 / Q4: rotated Q8KV256 / Q4KV256.
enum class KvCodec : int { Tq3 = 0, Tq2s = 1, Q8 = 2, Q4 = 3, Tcq2 = 4, Tcq3 = 5 };  // Tcq2: Tcq2V256, V only; Tcq3: Tcq3K256, K only
// TqCache stand-in for the compacted prefix of a sequence: the main kernels read cache positions from *start (a multiple
// of 32) and one extra partial covers the stand-in rows. Partial scratch needs one more split than without it.
struct KvStandIn256 {  // one stand-in row of one KV head: int8 values with an fp16 scale per 32 (272 B)
    __half s[8];
    int8_t q[256];
};
struct KvCompactSeg {
    const int * start = nullptr;          // device
    const KvStandIn256 * k = nullptr;     // [rows][kvheads], original basis
    const KvStandIn256 * v = nullptr;     // [rows][kvheads], TQ3 rotated basis
    const float * bias = nullptr;         // [rows][kvheads], log weights
    const int * n = nullptr;              // device: rows in use
    float * split_buf = nullptr;          // optional: row-split partials, split_cap x (256 + 2) floats; null: no row splits
    int split_cap = 0;                    // query-head rows x splits the buffer holds
};
size_t kv_compact_split_floats(int nver, int heads);  // split_buf size for verify blocks of up to nver columns
// TqCache compaction (src/formats/kv/kv_compact.cu). q_ring_write: the n query rows at positions *dpos0.. into a ring of
// ring_rows rows of `width` values. kv_compact_block: an old block of nb tq3 rows (hk / hv, any device-visible memory)
// -> m stand-in rows at row0 of ck / cv / cb, fitted to r_used ring queries; scratch of kv_compact_scratch_bytes.
void q_ring_write(const __nv_bfloat16 * q, int n, int width, const int * dpos0, int ring_rows, __nv_bfloat16 * ring,
                  cudaStream_t);
size_t kv_compact_scratch_bytes(int nb, int nq_rows, int m_max, int kvh);
// top-m of a block's key masses on the device (score desc, index asc), sel ascending; nb <= 8192
void kv_topm_select(const float * mass, int nb, int m, int succ, int * sel, cudaStream_t);
void kv_compact_block(const void * hk, const void * hv, KvCodec kcodec, KvCodec vcodec, int nb, int kvh, int heads, const __nv_bfloat16 * ring,
                      int r_used, float scale, int m, int row0, KvStandIn256 * ck, KvStandIn256 * cv, float * cb,
                      void * scratch, cublasHandle_t blas, cusolverDnHandle_t sol, cudaStream_t, int iters = 10,
                      float ridge = 1e-3f, int succ = 0, int ring_rows = 0,
                      int ring_end = 0,  // ring_rows > 0: ring written at position % ring_rows, newest row before ring_end
                      int solve = 0);    // value solve: 0 auto (nq x nq when nq < m), 1 m x m, 2 nq x nq
void attention_kv_split(KvCodec kcodec, KvCodec vcodec, const __nv_bfloat16 * q, const void * kc, const void * vc,
                        __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int heads, int kvheads, float scale,
                        int nsplit, cudaStream_t, const int * par = nullptr, const int * dep = nullptr,
                        const KvCompactSeg * cs = nullptr);
void cache_write_kv_side(KvCodec codec, const __nv_bfloat16 * x0, int ts, int n, void * c, const int * dpos0, int kvheads,
                         int dim, cudaStream_t);
void cache_write_multi_t4kv(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Turbo4K256 * kc,
                            Turbo4K256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t);
void attention_mq256_t4kv(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo4K256 * vc, __nv_bfloat16 * y,
                          float * part, const int * dseq0, int nq, int heads, int kvheads, float scale, int nsplit,
                          cudaStream_t, int query_offset = 0);
// Prefill form: all nq queries of a chunk in one launch (FA-style, 128 rows per block, K/V tiles decoded once per
// block); nsplit and part are used only by the old path (TQLLM_PREFILL_ATTN_OLD=1).
void attention_mq256_t4kv_prefill(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo4K256 * vc,
                                  __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int heads, int kvheads,
                                  float scale, int nsplit, cudaStream_t);
void cache_write_multi_t4k_turbo3v(const __nv_bfloat16 * k0, const __nv_bfloat16 * v0, int ts, int n, Turbo4K256 * kc,
                                   Turbo3V256 * vc, const int * dpos0, int kvheads, int dim, cudaStream_t);
void attention_mq256_t4k_turbo3v(const __nv_bfloat16 * q, const Turbo4K256 * kc, const Turbo3V256 * vc,
                                 __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int heads, int kvheads,
                                 float scale, int nsplit, cudaStream_t, int query_offset = 0);
void attention_mq256_q8_turbo3v(const __nv_bfloat16 * q, const int8_t * kc, const Turbo3V256 * vc, const __half * ksc,
                               __nv_bfloat16 * y, float * part, const int * dseq0, int nq, int heads, int kvheads,
                               float scale, int nsplit, cudaStream_t, int query_offset = 0);
// o[i] *= sigmoid(gate) for n tokens: o [n][heads*dim] contiguous, qg of token i at qg0 + i * ts.
void sigmoid_gate_multi(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int n, int heads, int dim, cudaStream_t);
// act[i][f] = silu(gu[i][f]) * gu[i][F + f] for n tokens, gu rows 2F apart, act [n][F].
void swiglu_multi(const __nv_bfloat16 * gu, int n, int F, __nv_bfloat16 * act, cudaStream_t);
// Fused with the TQ3 32-point input rotation (identical to the op followed by tq3_rotate_h): FP16 matmul input in xh.
void swiglu_rot_h(const __nv_bfloat16 * gu, int n, int F, __half * xh, cudaStream_t);
void gated_rms_rot_h(const __nv_bfloat16 * x, const __nv_bfloat16 * z0, int z_stride, int n, const float * w, __half * xh,
                     int heads, int dim, float eps, cudaStream_t);
void sigmoid_gate_rot_h(__nv_bfloat16 * o, const __nv_bfloat16 * qg0, int ts, int n, int heads, int dim, __half * xh,
                        cudaStream_t);
// Embedding rows for n tokens whose ids are on the device (graph-capturable). Table TQ3_4S or Q6_K.
void embed_multi_tq3(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t);
void embed_multi_q6k(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t);
void embed_multi_iq3s(const void * w, const uint32_t * dtok, int n, __nv_bfloat16 * y, int k, cudaStream_t);
// Per-head gated RMSNorm: y = bf16(w * bf16(x * rsqrt(mean(x^2)+eps)) * silu(z)), w FP32. [heads][dim].
void gated_rms(const __nv_bfloat16 * x, const __nv_bfloat16 * z, const float * w, __nv_bfloat16 * y, int heads, int dim,
               float eps, cudaStream_t);
// Same over n tokens: x, y [n][heads * dim] contiguous; z of token i at z0 + i * z_stride.
void gated_rms_multi(const __nv_bfloat16 * x, const __nv_bfloat16 * z0, int z_stride, int n, const float * w,
                     __nv_bfloat16 * y, int heads, int dim, float eps, cudaStream_t);

} // namespace tqllm_cuda

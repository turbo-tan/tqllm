#pragma once
// Inkling-2-Mini elementwise / small kernels (inkling2_kernels.cu). Activations: F32 buffers holding the values the
// transformers reference holds (BF16-rounded where it stores BF16), BF16 buffers as GEMM inputs. n = tokens of a chunk.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda::inkling2 {

constexpr int kMaxStates = 8;  // depth-attention entries (x0, three blocks, the partial; messages: two)
struct States { const float * s[kMaxStates]; int count; };

// depth attention: out[q][t][:] = sum_j softmax_j(query_q . rms(S_j[t]) / sqrt(D)) rms(S_j[t]), F32 (queries F32 [nq][D], nq <= 4);
// one block per token (attn_res_norm without norms)
// scratch: attn_res_norm_scratch_words(kSmallN, D) floats, owned by the caller, zeroed once before first use
void attn_res(const States & st, const float * queries, int nq, float * out, int n, int D, float * scratch, cudaStream_t s);
// y = bf16(w * rms(x)) as F32 and / or BF16 (either output may be null); w F32 [D]
void rms_norm(const float * x, const float * w, float * y32, __nv_bfloat16 * y16, int n, int D, float eps, cudaStream_t s);
void to_bf16(const float * x, __nv_bfloat16 * y, size_t n, cudaStream_t s);
void to_f32(const __nv_bfloat16 * x, float * y, size_t n, cudaStream_t s);
// embedding rows of `ids` (BF16 table [vocab][D]) -> rms with weight -> BF16-valued F32
void embed(const __nv_bfloat16 * table, const int * ids, const float * w, float * y, int n, int D, float eps, cudaStream_t s);
// static causal depthwise conv over n tokens, then (conv + x) / sqrt(5), BF16-rounded: x BF16 [n][C], w F32 [C][4]
// (tap 3 = current token), state F32 [C][3] (oldest first) updated in place
// y32 may add a residual: y32 = add + bf16(conv), rounded to BF16 again with round_sum
// hist (few-token calls only, may be null): the n inputs as BF16 [n][C] (spec-decode rollback)
void conv_static(const __nv_bfloat16 * x, const float * w, float * state, float * y32, __nv_bfloat16 * y16, int n, int C, cudaStream_t s,
                 const float * add = nullptr, bool round_sum = false, __nv_bfloat16 * hist = nullptr);
// dynamic: taps w_t[c][lag] = stat[lag][c] + dyn[t][(c / 128) * 4 + lag] (lag 0 = current), (sum + x) / 3
void conv_dynamic(const __nv_bfloat16 * x, const float * stat, const __nv_bfloat16 * dyn, float * state, __nv_bfloat16 * y16, int n, int C, cudaStream_t s,
                  __nv_bfloat16 * hist = nullptr);
// spec decode: the state after `keep` of the inputs recorded in hist, from the state before them (snap)
void conv_rollback(const float * snap, const __nv_bfloat16 * hist, float * state, int keep, int C, cudaStream_t s);
// message gate, first half: z = bf16(cat(rms(h), rms(m))) [n][2D]
void msg_gate_in(const float * h, const float * m, __nv_bfloat16 * z, int n, int D, float eps, cudaStream_t s);
// second half: g = F32(bf16 [n][rank]); f = tanh(F g), a = tanh(G g) (FT, GT: F, G transposed, BF16 [rank][D]); y = (1 - f) h + a m
void msg_gate_out(const __nv_bfloat16 * g, const __nv_bfloat16 * FT, const __nv_bfloat16 * GT, const float * h, const float * m, float * y, int n, int D,
                  int rank, cudaStream_t s);
// r = boundary ? o : h + o (F32); out = bf16-valued F32 of r + y   (two separate helpers)
void add(const float * a, const float * b, float * y, size_t n, bool round_bf16, cudaStream_t s);
// out[t] = bf16(sum_k w[t][k] * y[t * topk + k])  (BF16 as F32 / BF16)
void moe_combine(const __nv_bfloat16 * y, const float * wts, __nv_bfloat16 * out, int n, int D, int topk, cudaStream_t s);
// same with y in grouped order: row of assignment a = inv[a], inv[order[j]] = j (computed into inv, A ints)
void moe_combine_grouped(const __nv_bfloat16 * yg, const int * order, int A, const float * wts, __nv_bfloat16 * out, int n, int D, int topk, int * inv,
                         cudaStream_t s);
// logits = c * tanh(x / c) in place
void softcap(float * x, size_t n, float c, cudaStream_t s);

// ---- few-token path (decode, verify blocks): every token's arithmetic is independent of n (n <= kSmallN) ----
constexpr int kSmallN = 8;
// BF16 matrix-vector products for up to kMaxSegs (W [rows][K], x [n][ldx]) pairs in one launch; y[t][row] (ldy) as BF16 or
// F32, F32 accumulate; K % 8 == 0, 16-byte aligned rows
constexpr int kMaxSegs = 8;
// scale non-null: W is INT8 [rows][K] (the bytes behind the pointer), row r's weights = int8 x scale[r] (fast mode)
struct GemvSeg { const __nv_bfloat16 * W; const __nv_bfloat16 * x; void * y; int rows, K, ldx, ldy, f32; const float * scale = nullptr; };
// fast mode: rows of a BF16 matrix -> INT8 symmetric per row (scale = max |w| / 127, round to nearest)
void quantize_rows_i8(const __nv_bfloat16 * w, int8_t * q, float * scale, int rows, int K, cudaStream_t s);
// and back to BF16 (prompt chunks: cuBLAS on the dequantised rows, the same weights as the INT8 gemv)
void dequant_rows_i8(const int8_t * q, const float * scale, __nv_bfloat16 * w, int rows, int K, cudaStream_t s);
void gemv(const GemvSeg * segs, int nseg, int n, cudaStream_t s);
// a stack of routed experts on the device: fmt 1 MXFP8 rows (w: per expert rows x (cols + cols / 32) bytes), fmt 2 NVF4
// planes (w: codes [rows][cols / 2]; aux: [rows][cols / 16 E4M3 scales, F32 global]); stride / aux_stride: bytes per expert
// With wp / ap (device arrays [experts]) each expert's data / aux sits at its own address (a device slot or mapped host
// memory: the experts that do not fit stream over the bus), else at w + e * stride.
struct ExpertMat {
    const uint8_t * w = nullptr; const uint8_t * aux = nullptr; size_t stride = 0, aux_stride = 0; int fmt = 0;
    const uint8_t * const * wp = nullptr; const uint8_t * const * ap = nullptr;
};
// routed experts, gate + up + softcapped SwiGLU: act[a][i] for assignment a = t * topk + k (expert ids[a]), x [n][cols]
void expert_gate_up(const ExpertMat & g, const ExpertMat & u, int rows, int cols, const __nv_bfloat16 * x, const int * ids, __nv_bfloat16 * act, int n,
                    int topk, float gc, float uc, cudaStream_t s);
// down + combine: out[t][r] = bf16(sum_k w[t][k] * bf16(W_down[ids[t][k]][r] . act[t * topk + k]))
void expert_down(const ExpertMat & d, int rows, int cols, const __nv_bfloat16 * act, const int * ids, const float * wts, __nv_bfloat16 * out, int n,
                 int topk, cudaStream_t s);
// attn_res() for nq <= 4 streams, fused with rms_norm() of each stream i that has a weight w[i]:
// y16[i] = bf16(w[i] * rms(stream i)); out (F32 [nq][n][D]) may be null. n <= kSmallN splits D over blocks (two kernels,
// internal scratch); larger n runs one block per token. The per-token arithmetic does not depend on n either way.
// scratch (may be null for n > kSmallN) is owned by the caller and must be zeroed once before use, so capture never allocates.
void attn_res_norm(const States & st, const float * queries, int nq, float * out, const float * const * w, __nv_bfloat16 * const * y16, int n, int D,
                   float eps, float * scratch, cudaStream_t s);
// scratch words attn_res_norm needs for n <= kSmallN (0 otherwise); allocate the kSmallN maximum once.
int attn_res_norm_scratch_words(int n, int D);
// verify blocks (1 < n <= kSmallN): each distinct routed expert read once for all its assignments, every assignment with
// the arithmetic of expert_gate_up / expert_down (outputs equal those of single-token calls bitwise); groups: device
// scratch of 1 + n * topk * (1 + kSmallN) ints; act [n * topk][I], yd [n * topk][D] scratch
void expert_dedup(const ExpertMat & g, const ExpertMat & u, const ExpertMat & d, int I, int D, const __nv_bfloat16 * x, const int * ids, const float * wts,
                  int * groups, __nv_bfloat16 * act, __nv_bfloat16 * yd, __nv_bfloat16 * out, int n, int topk, float gc, float uc, cudaStream_t s);
// counts[ids[a]] += 1 for a < A (routed-expert use per layer)
void count_routes(const int * ids, int A, int * counts, cudaStream_t s);
// router selection from F32 logits [n][E]: s = sigmoid; top-k of s + bias (lowest id on ties); w = s / sum * scale
void route_topk(const float * logits, const float * bias, int * ids, float * wts, int n, int E, int topk, float scale, cudaStream_t s);

}  // namespace tqllm_cuda::inkling2

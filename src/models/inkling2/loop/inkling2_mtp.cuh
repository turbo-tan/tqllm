#pragma once
// Inkling-2-Mini MTP drafter kernels (inkling2_mtp.cu): the DSpark block draft (3 dense layers over context K / V built
// from four target taps, RoPE, q / k head norms) and the rnn_hidden Markov head (sglang inkling2_mtp.py / dspark_draft.py).
// The draft only proposes tokens; the target verifies them, so these kernels do not need the reference's rounding.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tqllm_cuda::inkling2 {

// out[r] = bf16(w * rms(sum_j alpha[j] taps[r][j]))  (taps BF16 [m][taps][D], alpha F32 [taps], w F32 [D])
void mtp_mix_norm(const __nv_bfloat16 * taps, int ntaps, const float * alpha, const float * w, __nv_bfloat16 * out, int m, int D, float eps,
                  cudaStream_t s);
// in place on x [n][heads][128]: per head RMSNorm (weight w [128]), then NeoX RoPE at position pos0 + t (theta)
void mtp_head_norm_rope(__nv_bfloat16 * x, int ld, int n, int heads, const float * w, int pos0, float theta, float eps, cudaStream_t s);
// k / v rows [n][kvheads * 128] -> cache rows (pos0 + t) % ring (ring 0: absolute)
void mtp_store_kv(const __nv_bfloat16 * k, const __nv_bfloat16 * v, int ldkv, __nv_bfloat16 * kc, __nv_bfloat16 * vc, int n, int kvheads, int pos0, int ring,
                  cudaStream_t s);
// block attention: query i (of nb, at position pos0 + i) over the context keys [lo, pos0) (lo = max(0, pos0 - window)
// with a window, else 0) and every block key (non-causal); out [nb][heads * 128] BF16; part: scratch
// mtp_attention_floats(...)
int mtp_attention_floats(int heads, int nb, int splits);
void mtp_attention(const __nv_bfloat16 * q, const __nv_bfloat16 * kb, const __nv_bfloat16 * vb, int ldq, int ldkv, const __nv_bfloat16 * kc,
                   const __nv_bfloat16 * vc, int ring, int pos0, int nb, int heads, int kvheads, int window, int splits, float * part,
                   __nv_bfloat16 * out, cudaStream_t s);
// act = bf16(silu(g) * u)
void mtp_silu_mul(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __nv_bfloat16 * act, size_t n, cudaStream_t s);
// h += F32(x) (residual stream F32, x BF16)
void mtp_add(float * h, const __nv_bfloat16 * x, size_t n, cudaStream_t s);
// z = bf16 cat(state, w1[token], hidden, f_rnn)   (state F32 [R], w1 BF16 [vocab][R], hidden / f_rnn BF16 [D])
void mtp_rnn_inputs(const float * state, const int * token, const __nv_bfloat16 * w1, const __nv_bfloat16 * hidden, const __nv_bfloat16 * f_rnn,
                    __nv_bfloat16 * z, int R, int D, cudaStream_t s);
// proj F32 [3R] (gate, candidate, output) + bias: state = sigmoid(g) state + (1 - sigmoid(g)) tanh(c); latent = bf16 tanh(o)
void mtp_rnn_gates(const float * proj, const __nv_bfloat16 * bias, float * state, __nv_bfloat16 * latent, int R, cudaStream_t s);
// token = argmax over v < V of softcap_c(base[v] + fold_b[v] + fold[v] . latent)  (fold BF16 [V][R], base F32 [V])
// partial: scratch of mtp_step_floats(V) floats; writes the token to *token (device)
int mtp_step_floats(int V);
void mtp_step_argmax(const float * base, const __nv_bfloat16 * fold, const float * fold_b, const __nv_bfloat16 * latent, int V, int R, float cap,
                     float * partial, int * token, cudaStream_t s);
// taps[t][j][:] = bf16(x[t][:]) for t < n (x F32 [n][D])
void mtp_tap(const float * x, int n, int D, __nv_bfloat16 * taps, int j, int ntaps, cudaStream_t s);
// x0 for the block: row 0 = src (BF16-valued F32 [D]), rows 1 .. nb - 1 = mask (BF16 [D]), as F32 [nb][D]
void mtp_block_init(const float * src, const __nv_bfloat16 * mask, float * h, int nb, int D, cudaStream_t s);

}  // namespace tqllm_cuda::inkling2

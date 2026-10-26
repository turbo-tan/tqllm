#pragma once
// GLM-5.3-Flash (glm5-next) decode kernels: mHC hyper-connections, KDA (Kimi delta attention, per-key-channel decay),
// nope MLA over a latent cache, sigmoid MoE routing. FP32 activations; one token per call. The FP32 CPU reference is
// tools/glm_ref.cpp; tests/glm5next.cu checks each kernel against the same math.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {
// y[r] = W[r] . x for an FP32 row-major [rows][cols] matrix
// y[t][r] = W[r] . x[t] for t < n
void glm_mv_f32(const float * w, const float * x, float * y, int rows, int cols, int n, cudaStream_t s);
// same result for matrices with few rows: each row split over `splits` blocks; partials [n][rows][splits] summed in
// order (deterministic, no atomics). partial must hold n * rows * splits floats.
void glm_mv_f32_splitk(const float * w, const float * x, float * y, float * partial, int rows, int cols, int n, int splits,
                       cudaStream_t s);
void glm_bf16_to_f32(const __nv_bfloat16 * x, float * y, int n, cudaStream_t s);
// y = x / sqrt(mean(x^2) + eps) * w (w may be null); writes FP32 and / or FP16 copies (either may be null)
void glm_rmsnorm(const float * x, const float * w, float * y, __half * yh, int n, float eps, cudaStream_t s, int rows = 1);
// rows of length n: each normalised on its own (KDA output heads)
void glm_rmsnorm_rows(const float * x, const float * w, float * y, int rows, int n, float eps, cudaStream_t s);
void glm_f32_to_half(const float * x, __half * y, int n, cudaStream_t s);
// out[t][j] = in[t * stride + off + j], j < width, t < n
void glm_gather_cols(const float * in, int stride, int off, int width, int n, float * out, cudaStream_t s);

// mHC (hc = 4): mix [24] -> pre [4] (sigmoid + eps), post [4] (2 sigmoid), comb [4][4] = Sinkhorn(softmax over dst
// for each src + eps) with `iters` alternating normalisations; then x = sum_i pre[i] S[i]
// n tokens: mix [n][mix_stride], S [n][4][hidden], x [n][hidden], post [n][4], comb [n][16]
void glm_hc_pre(const float * mix, const float * scale, const float * base, const float * S, float * x, float * post,
                float * comb, int hidden, int iters, float eps, cudaStream_t s, int n = 1, int mix_stride = 24);
// out[d] = post[d] f + sum_s comb[d][s] S[s]   (out must not alias S)
void glm_hc_post(const float * f, const float * S, const float * post, const float * comb, float * out, int hidden,
                 cudaStream_t s, int n = 1);
// mean of the 4 streams
void glm_hc_mean(const float * S, float * x, int hidden, cudaStream_t s, int n = 1);

// KDA causal conv1d + SiLU over q / k / v: proj [3][d], hist [3][K-1][d] (oldest first, updated), w [3][d][K]
// n tokens in order: proj / out [3][n][d] (q, k, v blocks of n rows)
// ts / ps: element stride between tokens / between the q, k, v projections (0: [3][n][d], i.e. ts = d, ps = n d)
void glm_kda_conv(const float * proj, float * hist, const float * w, float * out, int d, int K, cudaStream_t s, int n = 1,
                  int ts = 0, int ps = 0);
// g = lb * sigmoid(-((g + dt_bias) * A[h]))   g [heads][dim]
void glm_kda_gate(float * g, const float * dt_bias, const float * A, float lb, int heads, int dim, cudaStream_t s, int n = 1);
void glm_sigmoid(float * x, int n, cudaStream_t s);
// delta rule, per head: q, k L2-normed (eps 1e-6), q /= sqrt(dim); S[a][b] *= exp(g[a]); d = (v - S^T k) beta;
// S += k d^T; o = S^T q. q, k, v, g [heads][dim]; beta [heads]; S [heads][dim][dim]
// n tokens in order: q / k / v at token t start at q + t * qkv_stride; g, o [n][heads][dim]; beta [n][heads]
// S_out: the state after the n tokens in a second buffer (the double buffer a speculative block rolls back to);
// null = update S in place. Reading S and writing S_out keeps the arithmetic identical either way.
void glm_kda_recur(const float * q, const float * k, const float * v, const float * g, const float * beta, const float * S,
                   float * S_out, float * o, int heads, int dim, cudaStream_t s, int n = 1, int qkv_stride = 0);
// the conv history after m accepted tokens of a saved block: the last K-1 rows of (old history ++ block inputs [0, m)).
// old hist / hist [3][K-1][d] (oldest first), the saved block inputs with the same token / projection strides glm_kda_conv
// took (ts / ps); m <= the saved rows.
void glm_kda_hist_rollback(const float * old_hist, const float * proj, float * hist, int d, int K, int m, cudaStream_t s,
                           int ts, int ps);
// o[h] = rmsnorm(o[h], w) * sigmoid(g2[h]); also writes FP16
void glm_kda_out(const float * o, const float * g2, const float * w, float * y, __half * yh, int heads, int dim, float eps,
                 cudaStream_t s, int n = 1);

// MLA: qabs[h][j] = sum_i kb[h][j][i] q[h][i]   kb FP16 [heads][R][dk]
void glm_mla_qabs(const float * q, const __half * kb, float * qabs, int heads, int R, int dk, cudaStream_t s, int n = 1);
// kv[h] = softmax_t(qabs[h] . lat[t] * scale) . lat   lat [n][R]
// token t (of `tokens`) attends to positions [0, n + t): causal over the chunk whose first position is n - 1
void glm_mla_attn(const float * qabs, const float * lat, float * kv, int heads, int R, int n, float scale, cudaStream_t s,
                  int tokens = 1, const int * d_pos = nullptr);  // d_pos: n = *d_pos + 1 read on the device (graphs)
// dst rows [*d_pos, *d_pos + n) = src [n][R]
void glm_copy_rows_at(const float * src, float * dst, int R, int n, const int * d_pos, cudaStream_t s);
// FP16 latent cache (the engine's): the same kernels reading / writing FP16 rows
void glm_mla_attn(const float * qabs, const __half * lat, float * kv, int heads, int R, int n, float scale, cudaStream_t s,
                  int tokens = 1, const int * d_pos = nullptr);
void glm_copy_rows_at(const float * src, __half * dst, int R, int n, const int * d_pos, cudaStream_t s);
// out[h][i] = sum_j vb[h][i][j] kv[h][j]   vb FP16 [heads][dv][R]; writes FP32 and FP16
void glm_mla_vb(const float * kv, const __half * vb, float * out, __half * outh, int heads, int R, int dv, cudaStream_t s,
                int n = 1);

// sigmoid routing: p = sigmoid(logits); select top-k of p + bias; weights = p[sel] (normalised if `norm`, times `scale`
// unless it is 0 or 1). ids / w [k]
// n tokens: logits [n][experts], ids / w [n][k]
void glm_route(const float * logits, const float * bias, int * ids, float * w, int experts, int k, bool norm, float scale,
               cudaStream_t s, int n = 1);
// out[i] = silu(min(g, limit)) * clamp(u, -limit, limit) (no clamp when limit <= 1e-6); g, u BF16 [n]
void glm_swiglu(const __nv_bfloat16 * g, const __nv_bfloat16 * u, __half * out, int n, float limit, cudaStream_t s);
// same on FP32 inputs
void glm_swiglu_f32(const float * g, const float * u, __half * out, int n, float limit, cudaStream_t s);
// out[j] = shared[j] + sum_i w[i] y[i][j]   y BF16 [k][hidden]
// n tokens: y [n][k][hidden], w [n][k], shared / out [n][hidden]
void glm_moe_sum(const __nv_bfloat16 * y, const float * w, const float * shared, float * out, int k, int hidden,
                 cudaStream_t s, int n = 1);
// a += b
void glm_add(float * a, const float * b, int n, cudaStream_t s);
// DSA k-pool indexer (glm5next_dsa.cu). Positions [*d_pos or pos0, + n) of the FP16 key|gate ring kg [ring][2D] (position
// p in row p % ring: a key is only read to build its pool, so ring >= n + r holds every row a pool needs) get
// LayerNorm(k) | g; a token at position p with p % r == r - 1 writes pooled[p / r] = softmax_m(g + ape) . k over its pool.
void glm_dsa_store(const float * k, const float * g, const float * ln_w, const float * ln_b, __half * kg, const float * ape,
                   float * pooled, int D, int r, float eps, int n, const int * d_pos, int pos0, int ring, cudaStream_t s);
// scratch bytes glm_dsa_select needs for `tokens` queries over P pools
size_t glm_dsa_select_bytes(int tokens, int P);
// queries at positions pos0 .. pos0 + tokens - 1 (q [tokens][32 heads][D], w [tokens][32]): cells [tokens][stride] =
// the members of the top min(pickable, pools) pools by sum_h w_h relu(q_h . pooled) / sqrt(32 D), then the trailing
// incomplete pool; count [tokens] cells each
void glm_dsa_select(const float * q, const float * w, const float * pooled, int D, int r, int pools, int tokens, int pos0,
                    void * scratch, size_t scratch_bytes, int * cells, int * count, int stride, cudaStream_t s);
// glm_mla_attn over each token's cell list instead of every position before it
void glm_mla_attn_sparse(const float * qabs, const float * lat, float * kv, int heads, int R, const int * cells,
                         const int * count, int stride, float scale, int tokens, cudaStream_t s);
void glm_mla_attn_sparse(const float * qabs, const __half * lat, float * kv, int heads, int R, const int * cells,
                         const int * count, int stride, float scale, int tokens, cudaStream_t s);
// index of the largest value (first on ties)
void glm_argmax(const float * x, int n, int * out, cudaStream_t s);
// per row of x [rows][n]
void glm_argmax_rows(const float * x, int n, int rows, int * out, cudaStream_t s);
}  // namespace tqllm_cuda

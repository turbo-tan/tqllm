#pragma once
// Inkling-2-Mini attention (src/models/inkling2/loop/inkling2_attention.cu), head_dim 128, d_rel 16, no RoPE:
//   q  = clip_c(q_raw * q_scale) [* tau(pos)]                 per head, L2 clip to c (qk_clip), BF16 like the reference
//   rs = clip_r(rs_raw * rel_scale) [* tau(pos)]              rel_clip; tau(p) = 1 + alpha ln(max((p + 1) / floor, 1))
//   score(t) = q . k_t / sqrt(128) + (rs . rel_proj[:, pos - t] when pos - t < rel_extent)
//   keys t <= pos (causal), t > pos - window when window > 0; one extra softmax column per head: sink (its value is 0)
//   out = (softmax-weighted sum of v) * sigmoid(gate_raw)     (the output projection wo runs after, outside)
// [*] only with log_scale (the full-attention layers). K / V in the cache are final (inkling2_kv_finish: k clipped to
// qk_clip, v RMS-normalised without a weight), at cache row = absolute position. Scores, softmax and P.V run in F32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {

struct Inkling2AttnArgs {
    const __nv_bfloat16 * q = nullptr;      // [nq][heads * 128] raw q projection
    const __nv_bfloat16 * rs = nullptr;     // [nq][heads * 16] raw relative-state projection
    const __nv_bfloat16 * sink = nullptr;   // [nq][heads] sink logits
    const __nv_bfloat16 * gate = nullptr;   // [nq][heads * 128] raw output-gate projection
    const __nv_bfloat16 * q_scale = nullptr;    // [heads][128]
    const __nv_bfloat16 * rel_scale = nullptr;  // [heads][16]
    const __nv_bfloat16 * rel_proj = nullptr;   // [16][rel_extent]
    const __nv_bfloat16 * kc = nullptr;     // [>= pos0 + nq][kvheads][128]
    const __nv_bfloat16 * vc = nullptr;
    __nv_bfloat16 * out = nullptr;          // [nq][heads * 128]
    float * part = nullptr;                 // inkling2_attention_partial_floats(heads, nq, pos0 + nq) / _decode_floats
    int nq = 1, pos0 = 0, heads = 24, kvheads = 8, window = 0, rel_extent = 1024;
    int ring = 0;                           // > 0: key t sits in cache row t % ring (windowed layers), else row t
    const int * dpos = nullptr;             // decode: device copy of pos0 (graph-invariant launches)
    int splits = 0;                         // decode: fixed split count of every query's span
    bool log_scale = false;
    // the transformers reference's eager BF16 arithmetic (scores, scaling, bias, probabilities, P.V rounded to BF16), for
    // gating against its golden; needs stats: 2 floats per (query, head)
    bool hf_bf16 = false;
    float * stats = nullptr;
    float qk_clip = 22.627417f, rel_clip = 8.0f, log_alpha = 0.1f, log_floor = 8192.0f;
};

int inkling2_attention_partial_floats(int heads, int nq, int context);
void inkling2_attention(const Inkling2AttnArgs & a, cudaStream_t s);

// Decode / verify (few queries): one block per (split, kv head, query) serves the kv head's query heads, so each K / V row
// is read once per group; the span a query sees (window or context) is cut into a.splits equal parts. Every query's
// arithmetic depends only on its own position, never on nq: verifying k tokens at once equals k single steps bitwise.
// Position from a.dpos (device), F32 only (no hf_bf16).
int inkling2_attention_decode_floats(int heads, int nq, int splits);
void inkling2_attention_decode(const Inkling2AttnArgs & a, cudaStream_t s);

// Prompt chunks on tensor cores (inkling2_prompt_attn.cu): flash-attention over 64-key tiles, no splits, P.V in FP16; positions from a.dpos. Scratch: qbuf BF16 [nq][heads][128], rbuf F32 [nq][heads][16].
void inkling2_attention_prompt(const Inkling2AttnArgs & a, __nv_bfloat16 * qbuf, float * rbuf, cudaStream_t s);

// n rows: k = clip_{qk_clip}(k) per head, v = v * rsqrt(mean(v^2) + eps) per head, read from k / v [n][kvheads * 128] and
// written to cache row (pos + t) (or (pos + t) % ring) of kc / vc, pos = *dpos (0 without dpos); in place when k == kc
void inkling2_kv_finish(const __nv_bfloat16 * k, const __nv_bfloat16 * v, __nv_bfloat16 * kc, __nv_bfloat16 * vc, int n, int kvheads,
                        float qk_clip, float eps, const int * dpos, int ring, cudaStream_t s);

}  // namespace tqllm_cuda

#pragma once
// Vision encoder (Qwen3-VL ViT + 2x2 merger) element-wise kernels; FP32 activations [n][d] row-major.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace tqllm_cuda {

// y = LayerNorm(x) * w + b per row of d
void vis_layer_norm(const float * x, const float * w, const float * b, float * y, int n, int d, float eps, cudaStream_t);
// y[i][j] += b[j]; with gelu: y = gelu_tanh(y + b)
void vis_bias(float * y, const float * b, int n, int d, bool gelu, cudaStream_t);
// y += x
void vis_add(float * y, const float * x, int64_t n, cudaStream_t);
void vis_f32_to_h(const float * x, __half * y, int64_t n, cudaStream_t);
// rotary on q (columns [0, heads * hd)) and k (next heads * hd) of qkv rows (stride ld): pair (j, j + hd / 2), angle
// pos[r][0] * base^(-4j / hd) for j < hd / 4, else pos[r][1] * base^(-4(j - hd / 4) / hd) (ggml rope VISION mode)
void vis_rope(float * qkv, const int * pos, int n, int heads, int hd, int ld, float base, cudaStream_t);
// in-place softmax of each row (cols contiguous)
void vis_softmax_rows(float * s, int rows, int cols, cudaStream_t);

// prompt embeddings: row j of x (width h) <- img[tab[4 j]] where tab[4 j] >= 0 (image tokens; tab layout as the
// M-RoPE positions: [n][4] = source row, t, h, w)
void mm_scatter(__nv_bfloat16 * x, const int * tab, int n, const __nv_bfloat16 * img, int h, cudaStream_t);

}  // namespace tqllm_cuda

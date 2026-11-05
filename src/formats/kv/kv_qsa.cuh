#pragma once
// QSA selected attention on BF16 K/V rows (kv_qsa.cu): the indexer's chosen positions in, attention out.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tqllm_cuda {

struct QsaDeviceSelection {
    const int32_t * positions = nullptr;
    const uint8_t * valid = nullptr;
    int width = 0;
};

void qsa_set_split(int splits);  // TQFN_QSA_SPLIT override: S > 1 splits decode attention over the selected positions (numeric)
void qsa_set_tc(int on);  // 0: scalar instead of tensor-core prefill selected attention (12 heads per kv head, dim 256)
void qsa_reserve_split_scratch();  // allocate the split-attention scratch now (start-up)
void qsa_attention_direct(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                          const QsaDeviceSelection & selected, float * output, int tokens, int first_cells,
                          int heads, int kv_heads, int dim, cudaStream_t);

// Same results as qsa_attention_direct, K/V rows read once per kv head: scores [tokens][heads][width] and inv
// [tokens][heads] are caller scratch.
// Causal dense attention on the tensor-core prefill kernel (query t sees cells 0 .. first_cells + t - 1): the numerics of the
// selected prefill path, so a prompt's first top-k tokens are computed as its later ones. false: shape not covered.
bool qsa_attention_dense_tc(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values, float * output,
                            int tokens, int first_cells, int heads, int kv_heads, int dim, cudaStream_t);
void qsa_attention_grouped(const float * query, const __nv_bfloat16 * keys, const __nv_bfloat16 * values,
                           const QsaDeviceSelection & selected, float * scores, float * inv, float * output, int tokens,
                           int first_cells, int heads, int kv_heads, int dim, cudaStream_t);

}  // namespace tqllm_cuda

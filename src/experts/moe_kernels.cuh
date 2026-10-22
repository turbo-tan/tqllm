#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tqllm_cuda {
void moe_router(const float * router, const float * shared_gate, const __half * x,
                float * logits, int hidden, int experts, int n, cudaStream_t);
void moe_router_f32(const float * router, const float * shared_gate, const float * x,
                    float * logits, int hidden, int experts, int n, cudaStream_t);
// Device-side scatter of staged expert pieces into their bank slots: one launch for up to kScatterMax pieces (the
// descriptors travel as kernel parameters, so no descriptor buffer has to outlive the launch).
struct ScatterItem { const char * src; char * dst; unsigned long long bytes; };
constexpr int kScatterMax = 128;
// parts: blocks per item; zero-copy reads of pinned memory hold their SMs while PCIe-bound, so callers may use fewer.
void scatter_copy(const ScatterItem * items, int n, cudaStream_t stream, int parts = 8);
constexpr int kParamCopyMax = 8192;  // bytes per launch (kernel parameters, CUDA 12.1+ allows 32 KB)
void param_copy(const void * host, size_t bytes, void * dst, cudaStream_t stream);
void moe_topk(const float * logits, int * ids, float * weights, float * shared_weights,
              int experts, int topk, int n, float scale, cudaStream_t);
// TQFN_KEEP_MAP pruning: logits[t][e] += mask[e] (0 kept / -inf pruned) for e < experts, rows of stride ld
void logits_mask(float * logits, const float * mask, int n, int ld, int experts, cudaStream_t);
// moe_topk on two routings in one launch (this layer's and the next layer's prediction); e <= 1024, warp path.
void moe_topk2(const float * logits, int * ids, float * weights, float * shared_weights,
               const float * logits2, int * ids2, float * weights2, float * shared_weights2, int e, int k, int n, float scale, cudaStream_t);
void moe_topk_variant(const float * logits, int * ids, float * weights, float * shared_weights,
                      int e, int k, int n, float scale, bool warp, cudaStream_t stream);
// One assignment per expert per token; groups [experts][n], counts [experts]. Negative ids are skipped.
void moe_group(const int * ids, int * counts, int * groups, int experts, int n, int topk, cudaStream_t);
// Hot / cold expert split: hot_ids[i] = hot_slot[ids[i]] and cold_ids[i] = cold_slot[ids[i]] (-1 where the expert
// lives in the other set), so one expert launch per set covers every assignment exactly once.
void moe_split_ids(const int * ids, const int * hot_slot, const int * cold_slot, int * hot_ids, int * cold_ids,
                   int count, cudaStream_t stream);
// counts[ids[i]] += 1 (routing profile)
void moe_count(const int * ids, unsigned long long * counts, int count, cudaStream_t stream);
// --moe-host-mb: copy the routed host (cold) experts of one layer into a device stage, one slot per assignment.
// span z: expert c of src (c = cold_ids[a] >= 0) -> dst slot a, `bytes` per expert (16-byte multiples, 16-byte aligned).
// stage_ids[a] = a when cold_ids[a] >= 0, else -1: the ids of the staged copy for the unchanged expert kernels.
struct MoeStageSpan { const void * src = nullptr; void * dst = nullptr; long long bytes = 0; };
constexpr int kMoeStageSpans = 6;
void moe_stage(const MoeStageSpan (&spans)[kMoeStageSpans], int nspans, const int * cold_ids, int * stage_ids, int count,
               cudaStream_t stream);
// --moe-cache-mb: CLOCK cache of host experts, one pool per expert layout. For each assignment a with a host expert
// (cold_ids[a] = c >= 0, key = base + c): hit -> slot_ids[a] = its slot; miss -> a victim slot (never one used by this
// step), fill_src[a] = c, slot_ids[a] = victim. Single thread: count = n * topk assignments <= slots / 2.
struct MoeCachePool { int * slot_key; int * ref; int * hand; int slots; };
void moe_cache_lookup(const int * cold_ids, int count, int base, int * table, MoeCachePool pool, int * slot_ids, int * fill_src,
                      cudaStream_t stream);
// copy the missed experts (fill_src[a] >= 0) of every span into their cache slots (slot_ids[a]); spans as moe_stage
void moe_cache_fill(const MoeStageSpan (&spans)[kMoeStageSpans], int nspans, const int * fill_src, const int * slot_ids,
                    int count, cudaStream_t stream);
void moe_swiglu(const __nv_bfloat16 * gate, const __nv_bfloat16 * up, __half * act,
                int count, cudaStream_t);
void moe_sum(const __nv_bfloat16 * routed, const __nv_bfloat16 * shared, const float * weights,
             const float * shared_weights, __nv_bfloat16 * out, int hidden, int topk, int n, cudaStream_t);
// decode with the shared expert fused as assignment k of every token: out[t * (k + 1) + s] = s < k ? map(ids[t * k + s])
// : extra (map = hot_slot when given, else the id); cold_out[...] = cold_slot[id] for s < k, -1 for the shared slot
void moe_extend_ids(const int * ids, const int * hot_slot, const int * cold_slot, int * out, int * cold_out, int k, int n,
                    int extra, cudaStream_t stream);
// moe_sum with `slots` routed rows per token (the first topk weighted) and the shared rows `shared_ld` elements apart
void moe_sum_strided(const __nv_bfloat16 * routed, int slots, const __nv_bfloat16 * shared, int shared_ld, const float * weights,
                     const float * shared_weights, __nv_bfloat16 * out, int hidden, int topk, int n, cudaStream_t);
}  // namespace tqllm_cuda

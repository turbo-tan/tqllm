#pragma once
// Device helpers shared by the format kernels (src/formats/*/*.cu) and models: tensor-core MMA wrappers, packed half2 math,
// ldmatrix, async copies. Each backend implements the hardware-specific ones in src/core/platform/<backend>_device.cuh, with
// capability macros (TQLLM_DEV_PTX, TQLLM_DEV_CP_ASYNC, TQLLM_DEV_LDS_64K) that kernels branch on instead of the backend's
// name (AGENTS.md rule 13). This is the only device header that selects a backend.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

#if defined(TQLLM_WITH_HIP)
#include "platform/hip_device.cuh"
#else
#include "platform/cuda_device.cuh"
#endif

namespace tqllm_dev {

__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// Two-part (tree) verify attention: a query at position qpos takes keys [0, split) from the cache kernel and [split, qpos]
// from the in-block kernel. split <= qpos - TREE_MAX_DEPTH keeps it at or below the block start for any path of up to
// 16 nodes, and depends only on qpos.
constexpr int TREE_MAX_DEPTH = 15;
__host__ __device__ inline int tq_split_point(int qpos) {
    return qpos >= TREE_MAX_DEPTH ? (qpos - TREE_MAX_DEPTH) / 32 * 32 : 0;
}

}  // namespace tqllm_dev

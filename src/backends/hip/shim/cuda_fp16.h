#pragma once
#include "tq_hip_compat.h"
// fp16 / half2 on HIP: hip_fp16.h defines __half, __half2 and all the intrinsics tqLLM's kernels use
// (__hmul2, __hfma2, __floats2half2_rn, per-warp shuffles on half2). Includes are plain aliases.
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>  // several half<->bf16 conversion intrinsics live here

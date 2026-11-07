#pragma once
#include "tq_hip_compat.h"
// bfloat16 on HIP: hip ships the AMD-side equivalent of cuda_bf16.h as hip_bf16.h with the same
// intrinsic spellings (__bfloat162float, __float2bfloat16, __low2bfloat16, __bfloat16_as_ushort,
// __hmul2 / __hfma2 for bf162...). Only the type names and the *_rn spellings differ, aliased here.
#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// CUDA's round-to-nearest spellings: HIP's default conversion rounds to nearest even already.
__host__ __device__ inline __nv_bfloat16 __float2bfloat16_rn(float x) { return __float2bfloat16(x); }
__host__ __device__ inline __nv_bfloat162 __floats2bfloat162_rn(float x, float y) {
    __nv_bfloat162 r;
    r.x = __float2bfloat16_rn(x);
    r.y = __float2bfloat16_rn(y);
    return r;
}
__host__ __device__ inline float __nv_expf(float x) { return expf(x); }

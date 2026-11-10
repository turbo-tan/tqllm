#pragma once
#include "tq_hip_compat.h"
// fp8 (E4M3/E5M2) on HIP: hip_fp8.h exposes the OCP/IEEE interpretation (__HIP_E4M3, bias 7, signed zero)
// that matches NVIDIA's __nv_fp8_e4m3, plus the conversion intrinsics. On RDNA the conversion is software
// (CDNA has hardware cvt); the fp8 GGUF formats keep compiling, decoding costs what it costs.
#include <hip/hip_fp8.h>

using __nv_fp8_storage_t = __hip_fp8_storage_t;
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;

// CUDA's __NV_E4M3 (bias-7 IEEE style E4M3, signed zero) maps to __HIP_E4M3
enum __nv_fp8_interpretation_t { __NV_E5M2 = 0, __NV_E4M3 = 1 };
__host__ __device__ inline __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t x, __nv_fp8_interpretation_t interp) {
    return __hip_cvt_fp8_to_halfraw(x, interp == __NV_E4M3 ? __HIP_E4M3 : __HIP_E5M2);
}

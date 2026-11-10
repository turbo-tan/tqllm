#pragma once
// cuBLAS / hipBLAS and device-capability differences in one place, so no model or component branches on the backend
// (AGENTS.md rule 2; the same reason src/core/common.cuh holds the AMD device math).
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstring>

namespace tqllm_blas {

// FP32 accumulation setup every engine uses after cublasCreate: NVIDIA asks for tensor-op mode explicitly, hipBLAS has
// no equivalent knob (it always uses the matrix cores where the type allows).
inline cublasStatus_t enable_tensor_ops(cublasHandle_t h) {
#if defined(TQLLM_WITH_HIP)
    (void)h;
    return CUBLAS_STATUS_SUCCESS;
#else
    return cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
#endif
}

// BF16 dense GEMM needs matrix cores: sm_80 and newer on NVIDIA; on AMD the wave32 RDNA3 / RDNA4 parts (gfx11, gfx12);
// HIP reports a target id, not an sm_XX compute capability, so the check is per backend rather than read from the prop.
inline bool bf16_device_ok(const cudaDeviceProp & p, int major_min) {
#if defined(TQLLM_WITH_HIP)
    (void)major_min;  // wave32 RDNA3 / RDNA4 only: what the device code is written for (the build refuses other targets too)
    return std::strncmp(p.gcnArchName, "gfx11", 5) == 0 || std::strncmp(p.gcnArchName, "gfx12", 5) == 0;
#else
    (void)major_min;
    return p.major >= 8;
#endif
}

}  // namespace tqllm_blas
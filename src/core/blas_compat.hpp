#pragma once
// cuBLAS / hipBLAS and device-capability differences in one place, so no model or component branches on the backend
// (AGENTS.md rule 2; the same reason src/core/common.cuh holds the AMD device math).
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

// BF16 dense GEMM needs matrix cores: sm_80 and newer on NVIDIA. gfx11 (RDNA3) and gfx90a+ (CDNA2) do BF16 natively;
// HIP reports a target id, not an sm_XX compute capability, so the check is per backend rather than read from the prop.
inline bool bf16_device_ok(const cudaDeviceProp & p, int major_min) {
#if defined(TQLLM_WITH_HIP)
    (void)p;
    (void)major_min;
    return true;
#else
    (void)major_min;
    return p.major >= 8;
#endif
}

}  // namespace tqllm_blas
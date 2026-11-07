#pragma once
#include "tq_hip_compat.h"
// cuBLAS -> hipBLAS shim. Same handle/call shapes; only the dtype enum names and the Sgemv / GemmEx
// argument spellings differ, handled by the alias block and the one-line wrappers below.
#include <hipblas/hipblas.h>

using cublasHandle_t = hipblasHandle_t;
using cublasStatus_t = hipblasStatus_t;
using cublasOperation_t = hipblasOperation_t;
using cublasMath_t = hipblasMath_t;
using cublasPointerMode_t = hipblasPointerMode_t;

// CUBLAS_STATUS_* and HIPBLAS_STATUS_* share values up to 8.
constexpr auto CUBLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
constexpr auto CUBLAS_STATUS_NOT_INITIALIZED = HIPBLAS_STATUS_NOT_INITIALIZED;
constexpr auto CUBLAS_STATUS_ALLOC_FAILED = HIPBLAS_STATUS_ALLOC_FAILED;
constexpr auto CUBLAS_STATUS_INVALID_VALUE = HIPBLAS_STATUS_INVALID_VALUE;
constexpr auto CUBLAS_STATUS_ARCH_MISMATCH = HIPBLAS_STATUS_ARCH_MISMATCH;
constexpr auto CUBLAS_STATUS_EXECUTION_FAILED = HIPBLAS_STATUS_EXECUTION_FAILED;
constexpr auto CUBLAS_STATUS_INTERNAL_ERROR = HIPBLAS_STATUS_INTERNAL_ERROR;
constexpr auto CUBLAS_STATUS_NOT_SUPPORTED = HIPBLAS_STATUS_NOT_SUPPORTED;
constexpr auto CUBLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;
constexpr auto CUBLAS_GEMM_DEFAULT_TENSOR_OP = HIPBLAS_GEMM_DEFAULT;
constexpr auto CUBLAS_OP_N = HIPBLAS_OP_N;
constexpr auto CUBLAS_OP_T = HIPBLAS_OP_T;
constexpr auto CUBLAS_OP_C = HIPBLAS_OP_C;
constexpr auto CUBLAS_DEFAULT_MATH = HIPBLAS_DEFAULT_MATH;
constexpr auto CUBLAS_PEDANTIC_MATH = HIPBLAS_PEDANTIC_MATH;
constexpr auto CUBLAS_TENSOR_OP_MATH = HIPBLAS_TENSOR_OP_MATH;
constexpr auto CUBLAS_TF32_TENSOR_OP_MATH = HIPBLAS_TF32_TENSOR_OP_MATH;
constexpr auto cublasSetMathMode = hipblasSetMathMode;
constexpr auto cublasGetMathMode = hipblasGetMathMode;

constexpr auto cublasCreate = hipblasCreate;
constexpr auto cublasDestroy = hipblasDestroy;
constexpr auto cublasSetStream = hipblasSetStream;
constexpr auto cublasGetStream = hipblasGetStream;
constexpr auto cublasSgemv = hipblasSgemv;
constexpr auto cublasSgemm = hipblasSgemm;
constexpr auto cublasSetWorkspace = hipblasSetWorkspace;
constexpr auto CUBLAS_STATUS_MAPPING_ERROR = HIPBLAS_STATUS_MAPPING_ERROR;
constexpr auto cublasSgeam = hipblasSgeam;

// GemmEx: the only compute type tqLLM passes is CUBLAS_COMPUTE_32F; only dtypes are R_16F, R_16BF, R_32F,
// R_32I. hipblasGemmEx takes (hipDataType, hipblasComputeType_t, hipblasGemmAlgo_t) -- alias the numeric
// dtypes and call the HIP spellings; ops carry the HIPBLAS values already.
using cublasComputeType_t = hipblasComputeType_t;
constexpr auto CUBLAS_COMPUTE_32F = HIPBLAS_COMPUTE_32F;
constexpr auto CUBLAS_COMPUTE_16F = HIPBLAS_COMPUTE_16F;
constexpr auto CUDA_R_16F = HIP_R_16F;
constexpr auto CUDA_R_16BF = HIP_R_16BF;
constexpr auto CUDA_R_32F = HIP_R_32F;
constexpr auto CUDA_R_32I = HIP_R_32I;
constexpr auto CUDA_R_8I = HIP_R_8I;
constexpr auto cublasGemmEx = hipblasGemmEx;

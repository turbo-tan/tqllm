#pragma once
#include "tq_hip_compat.h"
// math constants: HIP's amd_hip_math_constants.h ships the HIP_* spellings; tqLLM uses CUDART_*. Keep every
// constant the kernel files reference so a picky internal header compile does not silently de-math them.
#include <hip/hip_math_constants.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F HIP_INF_F
#endif
#ifndef CUDART_INF
#define CUDART_INF HIP_INF
#endif
#define CUDART_VERSION 12000  // HIP: pretend CUDA 12.0 so the CUDA-12.8 batch-copy path falls through to plain cudaMemcpyAsync (HIP lacks cudaMemcpyBatchAsync with these flags)

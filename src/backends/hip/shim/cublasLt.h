#pragma once
// cuBLASLt is documented as out of scope for the HIP backend in this iteration: prefill quantised GEMM
// (FP8, sm_89+ / INT8, sm_80+) runs through cuBLASLt on NVIDIA and falls back to the native BF16 path
// (hipBLAS / own kernels) on AMD. Fail loudly if a new site starts including this header.
#error "cublasLt: the AMD/HIP build ships the native BF16 prefill path; cuBLASLt-specific code is excluded (see CMakeLists, TQLLM_HIP)"

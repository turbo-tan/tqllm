#pragma once
// cuSOLVER stub for HIP: tqLLM's only solver use is the opts-in TqvCache compaction path
// (src/formats/kv/kv_compact.cu Sp-otRf/Sp-otRs, --kv-compact N). It is turned off at engine setup on AMD
// (qwen35_load.cpp vendors: hip build), so the stub keeps the library linkable without hipsolver and any
// accidental call answers NOT_SUPPORTED with the message, instead of a link error.
using cusolverStatus_t = int;
using cusolverDnHandle_t = void *;
using cusolverDnParams_t = void *;
constexpr int CUSOLVER_STATUS_SUCCESS = 0;
constexpr int CUSOLVER_STATUS_NOT_SUPPORTED = 15;
constexpr int CUBLAS_FILL_MODE_LOWER = 1;
constexpr int CUBLAS_FILL_MODE_UPPER = 2;
inline int cusolverDnCreate(cusolverDnHandle_t *) { return CUSOLVER_STATUS_NOT_SUPPORTED; }
inline int cusolverDnDestroy(cusolverDnHandle_t) { return CUSOLVER_STATUS_SUCCESS; }
inline int cusolverDnSetStream(cusolverDnHandle_t, void *) { return CUSOLVER_STATUS_NOT_SUPPORTED; }
inline int cusolverDnSpotrf_bufferSize(cusolverDnHandle_t, int, int, float *, int, int * lwork) {
    *lwork = 0;
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
inline int cusolverDnSpotrf(cusolverDnHandle_t, int, int, float *, int, float *, int, int * ) {
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
inline int cusolverDnSpotrs(cusolverDnHandle_t, int, int, int, float *, int, float *, int, int * ) {
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}

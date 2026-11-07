#pragma once
// ROCm/HIP compatibility shim (src/backends/hip/shim): maps the CUDA *host* API surface tqLLM uses onto
// the HIP runtime. Alias names only -- at runtime every call is the HIP call, and any place that needs a
// different *device math* implementation is a native AMD kernel next to the CUDA one (search TQLLM_WITH_HIP).
// See tq_hip_compat.h for the device intrinsics this header carries as well.
#include "tq_hip_compat.h"

// --- types and enums that only differ by name ----------------------------------------------
using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaGraphNode_t = hipGraphNode_t;
using cudaGraph = ihipGraph;
using cudaGraphExec = hipGraphExec;
using cudaDeviceProp = hipDeviceProp_t;
using cudaMemPool_t = hipMemPool_t;
using cudaPointerAttributes = hipPointerAttribute_t;
using cudaFuncAttributes = hipFuncAttributes;
using cudaMemcpyKind = hipMemcpyKind;
using cudaStreamCaptureMode = hipStreamCaptureMode;
using cudaStreamCaptureStatus = hipStreamCaptureStatus;
using cudaDevAttr = hipDeviceAttribute_t;
using cudaFuncAttribute = hipFuncAttribute;

constexpr auto cudaSuccess = hipSuccess;
constexpr auto cudaErrorNotReady = hipErrorNotReady;
constexpr auto cudaErrorMemoryAllocation = hipErrorOutOfMemory;
constexpr auto cudaErrorInvalidValue = hipErrorInvalidValue;
constexpr auto cudaErrorLaunchFailure = hipErrorLaunchFailure;

constexpr auto cudaMemcpyHostToHost = hipMemcpyHostToHost;
constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;
constexpr auto cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
constexpr auto cudaMemcpyDefault = hipMemcpyDefault;

constexpr auto cudaHostAllocDefault = hipHostMallocDefault;
constexpr auto cudaHostAllocPortable = hipHostMallocPortable;
constexpr auto cudaHostAllocMapped = hipHostMallocMapped;
constexpr auto cudaHostRegisterDefault = hipHostRegisterDefault;

constexpr auto cudaStreamNonBlocking = hipStreamNonBlocking;
#define cudaStreamPerThread hipStreamPerThread
constexpr auto cudaEventDisableTiming = hipEventDisableTiming;
constexpr auto cudaEventBlockingSync = hipEventBlockingSync;

constexpr auto cudaStreamCaptureModeGlobal = hipStreamCaptureModeGlobal;
constexpr auto cudaStreamCaptureModeThreadLocal = hipStreamCaptureModeThreadLocal;
constexpr auto cudaStreamCaptureModeRelaxed = hipStreamCaptureModeRelaxed;
constexpr auto cudaStreamCaptureStatusNone = hipStreamCaptureStatusNone;
constexpr auto cudaStreamCaptureStatusActive = hipStreamCaptureStatusActive;
constexpr auto cudaStreamCaptureStatusInvalidated = hipStreamCaptureStatusInvalidated;

constexpr auto cudaDevAttrComputeCapabilityMajor = hipDeviceAttributeComputeCapabilityMajor;
constexpr auto cudaDevAttrComputeCapabilityMinor = hipDeviceAttributeComputeCapabilityMinor;
constexpr auto cudaDevAttrMultiProcessorCount = hipDeviceAttributeMultiprocessorCount;
constexpr auto cudaDevAttrL2CacheSize = hipDeviceAttributeL2CacheSize;
constexpr auto cudaDevAttrIntegrated = hipDeviceAttributeIntegrated;
constexpr auto cudaFuncAttributeMaxDynamicSharedMemorySize = hipFuncAttributeMaxDynamicSharedMemorySize;

constexpr auto cudaMemPoolAttrReleaseThreshold = hipMemPoolAttrReleaseThreshold;
constexpr auto cudaMemPoolAttrReservedMemCurrent = hipMemPoolAttrReservedMemCurrent;
constexpr auto cudaMemPoolAttrUsedMemCurrent = hipMemPoolAttrUsedMemCurrent;
constexpr auto cudaMemoryTypeDevice = hipMemoryTypeDevice;
constexpr auto cudaMemoryTypeManaged = hipMemoryTypeManaged;

using cudaGraphInstantiateFlagType = hipGraphInstantiateFlags;
using cudaGraphExecUpdateResultInfo = hipGraphExecUpdateResult;

// --- calls: macro aliases so overload resolution matches the CUDA spelling -----------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostAlloc hipHostMalloc
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaMemGetInfo hipMemGetInfo
#define cudaGetDevice hipGetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolGetAttribute hipMemPoolGetAttribute
#define cudaMemPoolTrimTo hipMemPoolTrimTo
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
// CUDA's modern 3-arg form (exec, graph, flags); HIP keeps the legacy 5-arg entry point.
inline hipError_t cudaGraphInstantiate(hipGraphExec_t * exec, hipGraph_t graph, unsigned long long flags) {
    (void)flags;
    return hipGraphInstantiate(exec, graph, nullptr, nullptr, 0);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t * exec, hipGraph_t graph, hipGraphNode_t * errorNode,
                                       char * logBuffer, size_t bufferSize) {
    return hipGraphInstantiate(exec, graph, errorNode, logBuffer, bufferSize);
}
#define cudaProfilerStart hipProfilerStart
#define cudaProfilerStop hipProfilerStop
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaLaunchKernel __hipLaunchKernel
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaStreamAddCallback hipStreamAddCallback

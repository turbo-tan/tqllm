#pragma once
// Driver-API (cuda.h) shim: the whole CUDA driver surface tqLLM uses is KV virtual memory
// (src/kv/kv_store_cuda.cpp). HIP >= 5.3 exposes the same calls with hip-prefixed names in
// hip/hip_runtime_api.h, including the host-NUMA tier, so only the names alias here.
#include <hip/hip_runtime_api.h>

using CUdeviceptr = hipDeviceptr_t;
using CUresult = hipError_t;
using CUdevice = int;
using CUmemGenericAllocationHandle = hipMemGenericAllocationHandle_t;
using CUmemAllocationProp = hipMemAllocationProp;
using CUmemAccessDesc = hipMemAccessDesc;
using CUdevice_attribute = hipDeviceAttribute_t;

constexpr auto CU_MEM_ALLOCATION_TYPE_PINNED = hipMemAllocationTypePinned;
constexpr auto CU_MEM_LOCATION_TYPE_DEVICE = hipMemLocationTypeDevice;
constexpr auto CU_MEM_LOCATION_TYPE_HOST_NUMA = hipMemLocationTypeHost;  // same NUMA-interleave semantics on HIP
constexpr auto CU_MEM_ALLOC_GRANULARITY_MINIMUM = hipMemAllocationGranularityMinimum;
constexpr auto CU_MEM_ACCESS_FLAGS_PROT_READWRITE = hipMemAccessFlagsProtReadWrite;
constexpr auto CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED =
    hipDeviceAttributeVirtualMemoryManagementSupported;

constexpr auto cuDeviceGetAttribute = hipDeviceGetAttribute;
constexpr auto cuMemGetAllocationGranularity = hipMemGetAllocationGranularity;
constexpr auto cuMemAddressReserve = hipMemAddressReserve;
constexpr auto cuMemCreate = hipMemCreate;
constexpr auto cuMemMap = hipMemMap;
constexpr auto cuMemSetAccess = hipMemSetAccess;
constexpr auto cuMemUnmap = hipMemUnmap;
constexpr auto cuMemRelease = hipMemRelease;
constexpr auto cuMemAddressFree = hipMemAddressFree;

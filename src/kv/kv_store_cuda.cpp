#include "dev_alloc.cuh"
#include "kv_store.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

namespace tqllm {
namespace {
struct CudaKvMemory : KvMemory {
    cudaStream_t stream;
    explicit CudaKvMemory(cudaStream_t s) : stream(s) {
        cudaGetDevice(&dev);
        int vmm = 0;
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
        if (vmm) {
            size_t gd = 0, gh = 0;
            CUmemAllocationProp pd = prop(KvTier::device), ph = prop(KvTier::host);
            if (cuMemGetAllocationGranularity(&gd, &pd, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
                cuMemGetAllocationGranularity(&gh, &ph, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS)
                gran = std::max(gd, gh);
        }
    }
    void * device_alloc(size_t n) override {
        void * p = nullptr;
        if (tqllm_cuda::dev_malloc_v(&p, n) != cudaSuccess) {  // counted: the memory fit reads dev_bytes() on WSL
            cudaGetLastError();  // clear the error so the caller's next CUDA call does not see it
            return nullptr;
        }
        return p;
    }
    void device_free(void * p) override { tqllm_cuda::dev_free(p); }
    void * host_alloc(size_t n) override {
        void * p = nullptr;
        if (cudaHostAlloc(&p, n, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        return p;
    }
    void host_free(void * p) override { cudaFreeHost(p); }
    void copy(void * dst, const void * src, size_t n) override {
        if (cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, stream) != cudaSuccess)
            throw std::runtime_error("KV store: page copy failed");
    }

    // Virtual memory management: device chunks (pinned on this GPU) and host chunks (pinned host memory the GPU reads
    // over PCIe) mapped into one reserved range.
    int dev = 0;
    size_t gran = 0;
    struct Mapping { CUdeviceptr at; size_t bytes; CUmemGenericAllocationHandle h; bool owned; bool device; };
    struct Phys { CUmemGenericAllocationHandle h; size_t bytes; };
    std::map<CUdeviceptr, std::vector<Mapping>> ranges;  // reserved base -> its mappings
    CUmemAllocationProp prop(KvTier where) const {
        CUmemAllocationProp p{};
        p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location.type = where == KvTier::host ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
        p.location.id = where == KvTier::host ? 0 : dev;
        return p;
    }
    size_t granule() const override { return gran; }
    void * reserve(size_t n) override {
        CUdeviceptr va = 0;
        if (cuMemAddressReserve(&va, n, gran, 0, 0) != CUDA_SUCCESS) return nullptr;
        ranges[va];
        return reinterpret_cast<void *>(va);
    }
    bool map(void * at, size_t n, KvTier where) override {
        const CUdeviceptr a = reinterpret_cast<CUdeviceptr>(at);
        auto it = ranges.upper_bound(a);
        if (it == ranges.begin()) return false;
        --it;
        const CUmemAllocationProp p = prop(where);
        CUmemGenericAllocationHandle h;
        if (cuMemCreate(&h, n, &p, 0) != CUDA_SUCCESS) return false;
        if (cuMemMap(a, n, 0, h, 0) != CUDA_SUCCESS) {
            cuMemRelease(h);
            return false;
        }
        CUmemAccessDesc d{};
        d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        d.location.id = dev;
        d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        it->second.push_back({a, n, h, true, where == KvTier::device});
        if (where == KvTier::device) tqllm_cuda::dev_bytes_counter() += n;  // the memory fit counts device granules
        return cuMemSetAccess(a, n, &d, 1) == CUDA_SUCCESS;
    }
    void * phys_alloc(size_t n) override {
        const CUmemAllocationProp p = prop(KvTier::device);
        CUmemGenericAllocationHandle h;
        if (cuMemCreate(&h, n, &p, 0) != CUDA_SUCCESS) return nullptr;
        tqllm_cuda::dev_bytes_counter() += n;
        return new Phys{h, n};
    }
    void phys_free(void * ph) override {
        if (!ph) return;
        cuMemRelease(static_cast<Phys *>(ph)->h);
        tqllm_cuda::dev_bytes_counter() -= static_cast<Phys *>(ph)->bytes;
        delete static_cast<Phys *>(ph);
    }
    bool map_phys(void * at, void * ph) override {
        const CUdeviceptr a = reinterpret_cast<CUdeviceptr>(at);
        auto it = ranges.upper_bound(a);
        if (it == ranges.begin()) return false;
        --it;
        const auto * P = static_cast<Phys *>(ph);
        const size_t n = P->bytes;
        if (cuMemMap(a, n, 0, P->h, 0) != CUDA_SUCCESS) return false;
        CUmemAccessDesc d{};
        d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        d.location.id = dev;
        d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        it->second.push_back({a, n, P->h, false, true});
        return cuMemSetAccess(a, n, &d, 1) == CUDA_SUCCESS;
    }
    void unmap(void * at, size_t n) override {
        const CUdeviceptr a = reinterpret_cast<CUdeviceptr>(at);
        auto it = ranges.upper_bound(a);
        if (it == ranges.begin()) return;
        --it;
        auto & v = it->second;
        for (size_t i = 0; i < v.size(); ++i)
            if (v[i].at == a && v[i].bytes == n) {
                cuMemUnmap(a, n);
                if (v[i].owned) cuMemRelease(v[i].h);
                if (v[i].owned && v[i].device) tqllm_cuda::dev_bytes_counter() -= n;
                v.erase(v.begin() + ptrdiff_t(i));
                return;
            }
    }
    void sync() override { cudaStreamSynchronize(stream); }
    void unreserve(void * base, size_t n) override {
        const auto it = ranges.find(reinterpret_cast<CUdeviceptr>(base));
        if (it == ranges.end()) return;
        cudaDeviceSynchronize();  // not the stream: the engine may have destroyed it already (WSL then loses the context)
        for (const auto & m : it->second) {
            cuMemUnmap(m.at, m.bytes);
            if (m.owned) cuMemRelease(m.h);
            if (m.owned && m.device) tqllm_cuda::dev_bytes_counter() -= m.bytes;
        }
        cuMemAddressFree(it->first, n);
        ranges.erase(it);
    }
};
}  // namespace

std::unique_ptr<KvMemory> make_cuda_kv_memory(void * stream) {
    return std::make_unique<CudaKvMemory>(static_cast<cudaStream_t>(stream));
}

}  // namespace tqllm

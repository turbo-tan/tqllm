#pragma once
#include "debug.hpp"
#include "wired_budget.hpp"
#include <mlx/device.h>
#include <mlx/memory.h>
#include <cstdio>
#include <cstdlib>
#include <optional>
namespace tqllm::mlx_backend {
class ScopedWiredLimit {
    std::optional<size_t> previous;
public:
    ScopedWiredLimit() {
        if(const char* value=tqllm::debug_cstr("wired_mib")) {
            auto recommended=std::get<size_t>(mlx::core::device_info(mlx::core::Device::gpu).at("max_recommended_working_set_size"));
            size_t requested=wired_bytes(value,recommended);
            previous=mlx::core::set_wired_limit(requested);
            std::fprintf(stderr,"MLX_WIRED bytes=%zu previous=%zu recommended=%zu\n",requested,*previous,recommended);
        }
    }
    ScopedWiredLimit(const ScopedWiredLimit&)=delete;
    ScopedWiredLimit& operator=(const ScopedWiredLimit&)=delete;
    bool overridden() const {return previous.has_value();}
    ~ScopedWiredLimit() {
        if(previous)try{mlx::core::set_wired_limit(*previous);}
        catch(...){std::fprintf(stderr,"MLX_WIRED restore failed\n");}
    }
};
// Without a wired limit macOS may page a long request's GPU buffers out between steps (256K: 19.0 vs 23.2 tok/s, j1824).
// Each request wires only what it holds; the limit goes back when the request ends.
class RequestResidency {
    std::optional<size_t> previous;
public:
    size_t hold(size_t active,size_t headroom) {
        auto recommended=std::get<size_t>(mlx::core::device_info(mlx::core::Device::gpu).at("max_recommended_working_set_size"));
        size_t bytes=request_wired_bytes(active,headroom,recommended);
        size_t was=mlx::core::set_wired_limit(bytes);
        if(!previous)previous=was;
        return bytes;
    }
    void release() {
        if(previous){mlx::core::set_wired_limit(*previous);previous.reset();}
    }
    ~RequestResidency() {
        try{release();}catch(...){std::fprintf(stderr,"MLX_WIRED restore failed\n");}
    }
};
}

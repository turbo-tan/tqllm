#pragma once
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
        if(const char* value=std::getenv("TQLLM_MLX_WIRED_MIB")) {
            auto recommended=std::get<size_t>(mlx::core::device_info(mlx::core::Device::gpu).at("max_recommended_working_set_size"));
            size_t requested=wired_bytes(value,recommended);
            previous=mlx::core::set_wired_limit(requested);
            std::fprintf(stderr,"MLX_WIRED bytes=%zu previous=%zu recommended=%zu\n",requested,*previous,recommended);
        }
    }
    ScopedWiredLimit(const ScopedWiredLimit&)=delete;
    ScopedWiredLimit& operator=(const ScopedWiredLimit&)=delete;
    ~ScopedWiredLimit() {
        if(previous)try{mlx::core::set_wired_limit(*previous);}
        catch(...){std::fprintf(stderr,"MLX_WIRED restore failed\n");}
    }
};
}

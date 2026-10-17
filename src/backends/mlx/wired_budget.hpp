#pragma once
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
namespace tqllm::mlx_backend {
inline size_t wired_bytes(std::string_view text,size_t recommended) {
    uint64_t mib=0;auto r=std::from_chars(text.data(),text.data()+text.size(),mib);
    constexpr uint64_t unit=1024*1024;
    if(text.empty() || r.ec!=std::errc{} || r.ptr!=text.data()+text.size() ||
       mib>std::numeric_limits<size_t>::max()/unit || mib*unit>recommended)
        throw std::invalid_argument("invalid MLX wired MiB budget or exceeds device recommendation");
    return size_t(mib*unit);
}
}

#pragma once
#include <charconv>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace tqllm::mlx_backend {
inline int kld_integer(std::string_view value) {
    int result = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc() || end != value.data() + value.size())
        throw std::invalid_argument("invalid accuracy window integer");
    return result;
}
inline size_t kld_window_begin(int first, int count, size_t token_count) {
    if (first < 0 || count < 1 || count > 100 || first > 100 - count)
        throw std::invalid_argument("accuracy window must lie within100 chunks");
    const size_t begin = size_t(first) * 512;
    if (token_count < begin + size_t(count) * 512 + 1)
        throw std::invalid_argument("accuracy corpus lacks window or lookahead token");
    return begin;
}
}

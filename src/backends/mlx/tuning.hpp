#pragma once
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tqllm::mlx_backend {
inline int matmul_columns(int rows) {
    int limit=4;
    if(const char* value=std::getenv("TQLLM_MLX_MATMUL_COLS")) {
        const std::string_view text(value);
        if(text=="4")limit=4;
        else if(text=="8")limit=8;
        else if(text=="16")limit=16;
        else throw std::invalid_argument("TQLLM_MLX_MATMUL_COLS must be 4, 8 or 16");
    }
    return rows<=1?1:rows<=2?2:rows<=4?4:rows<=8?std::min(limit,8):limit;
}
inline int matmul_simds() {
    const char* value = std::getenv("TQLLM_MLX_MATMUL_SIMDS");
    if (!value) return 2;
    for (int n : {1, 2, 4, 8})
        if (std::string_view(value) == std::to_string(n)) return n;
    throw std::invalid_argument("TQLLM_MLX_MATMUL_SIMDS must be 1, 2, 4 or 8");
}
}

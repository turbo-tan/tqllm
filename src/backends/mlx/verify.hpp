#pragma once
#include "weights.hpp"
namespace tqllm::mlx_backend {
mlx::core::array verify_tiled(const Matrix& weights,const mlx::core::array& input);
}

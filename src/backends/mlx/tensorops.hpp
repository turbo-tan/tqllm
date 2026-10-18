#pragma once
#include "weights.hpp"
namespace tqllm::mlx_backend {
mlx::core::array verify_expanded_tensorops(const Tq3Expanded& weights,const mlx::core::array& input,bool fp16=true);
mlx::core::array verify_tensorops(const Matrix& weights,const mlx::core::array& input, bool fp16 = false);
}

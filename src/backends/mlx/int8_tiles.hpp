#pragma once
#include "tq3.hpp"

namespace tqllm::mlx_backend {
// Experimental projection; three residual planes retain near-FP32 activations.
mlx::core::array verify_tq3_int8(const Tq3Weights& weights,
                               const mlx::core::array& input, int planes = 3);
}

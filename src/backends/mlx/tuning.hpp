#pragma once
#include <algorithm>

namespace tqllm::mlx_backend {
// columns per tile and simdgroups per threadgroup of the row-kernel matmuls (TQ3 / TQ4 / generic linear)
inline int matmul_columns(int rows) {return rows<=1?1:rows<=2?2:4;}
inline int matmul_simds() {return 2;}
}

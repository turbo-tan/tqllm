#pragma once
#include <mlx/mlx.h>

namespace tqllm::mlx_backend {
struct GdnHeadOrder {
    mlx::core::array to_native,from_native;
    int key_heads,value_heads;
};
GdnHeadOrder gdn_head_order(int key_heads,int value_heads);
// Recurrent state stays in native head order; readout returns model head order.
std::vector<mlx::core::array> gdn_native(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,const GdnHeadOrder& order);
}

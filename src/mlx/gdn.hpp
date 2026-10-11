#pragma once
#include <mlx/mlx.h>

namespace tqllm::mlx_backend {
mlx::core::array gdn_convolve(const mlx::core::array& joined,
    const mlx::core::array& weights,bool compiled=false);
mlx::core::array gdn_normalize(const mlx::core::array& input,float scale,bool compiled=false);
mlx::core::array gdn_decay(const mlx::core::array& alpha,const mlx::core::array& bias,
    const mlx::core::array& rate,bool compiled=false);
struct GdnHeadOrder {
    mlx::core::array to_native,from_native;
    int key_heads,value_heads;
};
GdnHeadOrder gdn_head_order(int key_heads,int value_heads);
struct GdnScan {
    mlx::core::array output,state;
    std::optional<mlx::core::array> snapshots;
};
GdnScan gdn_scan(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,bool snapshots=false);
// Recurrent state stays in native head order; readout returns model head order.
std::vector<mlx::core::array> gdn_native(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,const GdnHeadOrder& order);
}

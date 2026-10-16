#pragma once
#include <mlx/mlx.h>
#include <optional>

namespace tqllm::mlx_backend {
mlx::core::array gdn_convolve(const mlx::core::array& joined,
    const mlx::core::array& weights,bool compiled=false);
mlx::core::array gdn_convolve_native(const mlx::core::array& joined,
    const mlx::core::array& weights);
std::pair<mlx::core::array,mlx::core::array> gdn_gates_native(
    const mlx::core::array& alpha,const mlx::core::array& beta,
    const mlx::core::array& bias,const mlx::core::array& rate);
mlx::core::array gdn_normalize(const mlx::core::array& input,float scale,bool compiled=false);
mlx::core::array gdn_decay(const mlx::core::array& alpha,const mlx::core::array& bias,
    const mlx::core::array& rate,bool compiled=false);
// conv + silu + q/k normalization in one launch (output layout of gdn_convolve with q, k normalized in place)
mlx::core::array gdn_pre_fused(const mlx::core::array& joined,const mlx::core::array& weights,int key_heads,float query_scale);
// bf16 -> rms_norm -> * silu(z) -> bf16 in one launch; y, z [1, m, heads, 128]
mlx::core::array gdn_post_fused(const mlx::core::array& y,const mlx::core::array& z,const mlx::core::array& weight,float eps);
// prefill-only chunked form of gdn_scan (output [1, m, vh, 128], final state); chunk 16 / 32 / 64
std::pair<mlx::core::array,mlx::core::array> gdn_chunked(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,int chunk=64);
struct GdnHeadOrder {
    mlx::core::array to_native,from_native;
    int key_heads,value_heads;
};
GdnHeadOrder gdn_head_order(int key_heads,int value_heads);
struct GdnScan {
    mlx::core::array output,state;
    std::optional<mlx::core::array> snapshots;
    std::optional<mlx::core::array> deltas;
};
GdnScan gdn_scan(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,bool snapshots=false,bool compact=false);
mlx::core::array gdn_replay_prefix(const mlx::core::array& keys,
    const mlx::core::array& decay,const mlx::core::array& deltas,
    const std::optional<mlx::core::array>& initial,int count);
// Recurrent state stays in native head order; readout returns model head order.
std::vector<mlx::core::array> gdn_native(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,const GdnHeadOrder& order);
}

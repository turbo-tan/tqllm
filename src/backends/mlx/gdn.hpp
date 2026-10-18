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
// One launch for a 1-8 token GDN step: conv + silu, q / k normalization, gates, the scan, the compact tape
// (normalized keys, decay, deltas), the conv history and the gated norm (bf16 -> rms_norm * silu(z) -> bf16).
// joined [m + 3, width] (conv history + projection), z [m, vh * 128]; out [m, vh * 128] holds bf16 values as fp32.
struct GdnStep {
    mlx::core::array out,state,deltas,keys,decay,history;
};
GdnStep gdn_step_fused(const mlx::core::array& joined,const mlx::core::array& weights,
    const mlx::core::array& alpha,const mlx::core::array& beta,const mlx::core::array& bias,const mlx::core::array& rate,
    const mlx::core::array& z,const mlx::core::array& norm_weight,float eps,float query_scale,
    const std::optional<mlx::core::array>& state,int key_heads,int value_heads);
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
// prefill scan (no snapshots) with 8 lanes per state row; gdn_scan routes m >= 64 here (TQLLM_MLX_GDN_SCAN8=0 off)
GdnScan gdn_scan8(const mlx::core::array& q,const mlx::core::array& k,const mlx::core::array& v,
    const mlx::core::array& decay,const mlx::core::array& beta,const mlx::core::array& initial);
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

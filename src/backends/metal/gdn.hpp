#pragma once
#include <array>
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
// prefill pack variants: q / k / v (and z) read in place from the pack's FP16 product, bitwise equal to the above
std::array<mlx::core::array,3> gdn_pre_pack(const mlx::core::array& history,const mlx::core::array& product,int width,
                                            const mlx::core::array& weights,int key_heads,float query_scale);
mlx::core::array gdn_post_pack(const mlx::core::array& y,const mlx::core::array& product,int offset,
                               const mlx::core::array& weight,float eps);
// One launch for a 1-8 token GDN step: conv + silu, q / k normalization, gates, the scan, the compact tape
// (normalized keys, decay, deltas), the conv history and the gated norm (bf16 -> rms_norm * silu(z) -> bf16).
// joined [m + 3, width] (conv history + projection), z [m, vh * 128]; out [m, vh * 128] holds bf16 values as FP16.
struct GdnStep {
    mlx::core::array out,state,deltas,keys,decay,history;
};
GdnStep gdn_step_fused(const mlx::core::array& joined,const mlx::core::array& weights,
    const mlx::core::array& alpha,const mlx::core::array& beta,const mlx::core::array& bias,const mlx::core::array& rate,
    const mlx::core::array& z,const mlx::core::array& norm_weight,float eps,float query_scale,
    const std::optional<mlx::core::array>& state,int key_heads,int value_heads);
// the same with the conv history prev [3, width] and the projection rows [m, width] as separate inputs (no joined copy).
// input / ab are either the projected gates alpha, beta [m, vh] or the layer input [m, hidden] and the stacked alpha | beta
// weights [2 vh, hidden] (FP32, the stored basis): then the kernel projects them itself (j1038: the two 48-row GEMVs and
// their casts were ~8 dispatches and ~50 us of launch latency per layer)
GdnStep gdn_step_fused(const mlx::core::array& prev,const mlx::core::array& rows,const mlx::core::array& weights,
    const mlx::core::array& input,const mlx::core::array& ab,const mlx::core::array& bias,const mlx::core::array& rate,
    const mlx::core::array& z,const mlx::core::array& norm_weight,float eps,float query_scale,
    const std::optional<mlx::core::array>& state,int key_heads,int value_heads,int row_stride=0,int z_stride=0,int z_offset=0);
// out holds BF16 values as FP16 (the ssm_out matmul's input). row_stride / z_stride / z_offset: rows and z read in place
// from one packed qkv | z product [m, row_stride] (z at column z_offset), so the projection is a single launch.
struct GdnScan {
    mlx::core::array output,state;
    std::optional<mlx::core::array> snapshots;
    std::optional<mlx::core::array> deltas;
};
// prefill scan (no snapshots) with 8 lanes per state row; gdn_scan routes m >= 64 here
GdnScan gdn_scan8(const mlx::core::array& q,const mlx::core::array& k,const mlx::core::array& v,
    const mlx::core::array& decay,const mlx::core::array& beta,const mlx::core::array& initial);
// gdn_scan8 with the k / q / v / gate blocks staged in threadgroup memory, equal output; rows 16 or 32 per threadgroup
GdnScan gdn_scan8_staged(const mlx::core::array& q,const mlx::core::array& k,const mlx::core::array& v,
                         const mlx::core::array& decay,const mlx::core::array& beta,const mlx::core::array& initial,int rows);
GdnScan gdn_scan(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& decay,const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state,bool snapshots=false,bool compact=false);
mlx::core::array gdn_replay_prefix(const mlx::core::array& keys,
    const mlx::core::array& decay,const mlx::core::array& deltas,
    const std::optional<mlx::core::array>& initial,int count);
}

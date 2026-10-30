#pragma once
#include "qwen35.hpp"
#include <memory>
#include <optional>
#include <utility>
#include <vector>
namespace tqllm::mlx_backend {
// side 0: input taps of the block, side 1: output taps; with a residual, returns bf(residual + conv)
mlx::core::array dflash_conv(const mlx::core::array& input,const mlx::core::array& dynamic,
                            const mlx::core::array& base,int group,int side,
                            const std::optional<mlx::core::array>& residual=std::nullopt);
mlx::core::array dflash_select(const mlx::core::array& mapped,const mlx::core::array& unary,
    const mlx::core::array& gate,const mlx::core::array& successors,
    const mlx::core::array& predecessors,uint32_t anchor,int count);
// the same selection from an anchor id already on the GPU (uint32 scalar)
mlx::core::array dflash_select(const mlx::core::array& mapped,const mlx::core::array& unary,
    const mlx::core::array& gate,const mlx::core::array& successors,
    const mlx::core::array& predecessors,const mlx::core::array& anchor,int count);
// the 16 largest logits per row, largest first (ties: lower column first): {column ids (uint32), values (float32)} [rows, 16]
std::pair<mlx::core::array,mlx::core::array> dflash_top16(const mlx::core::array& logits);
class DFlash2 {
public:
    DFlash2(const std::string& path,Qwen35& target);
    ~DFlash2();
    const std::vector<int>& feature_layers() const;
    void reset();
    void inject(const mlx::core::array& features,int position);
    std::vector<uint32_t> draft(uint32_t anchor,int position,int count);
    // the same draft with the selector's ids left unevaluated on the GPU (uint32 [count]); needs
    // feed it to Qwen35::verify_device without reading it back.
    mlx::core::array draft_device(uint32_t anchor,int position,int count);
    // draft-ahead (ring drafter, device selector): inject `count` (device int32, 1..rows) of the feature rows at `position`
    // (device int32, the first row's position) without a readback, and draft from an anchor id and position on the GPU,
    // so the next draft is queued behind the verify that produced them
    void inject_device(const mlx::core::array& features,const mlx::core::array& count,const mlx::core::array& position);
    mlx::core::array draft_device_at(const mlx::core::array& anchor,const mlx::core::array& position,int count);
    // after the step's readback: the last inject_device wrote `rows` rows (tightens the host's bound on the GPU fill)
    void settle_device_inject(int rows);
    // TQLLM_MLX_DRAFT_STAGES: mean ms per draft_device_at of layers, head, top-16 and selector (each stage synced)
    std::vector<double> stage_ms_per_call() const;
    // bytes the far-context caches hold per position (full K / V per layer + chunk key sums); 0 without retrieval
    size_t far_row_bytes() const;
    // per request (after reset): keep the far-context caches and retrieval, or run the window alone
    void far_context(bool enabled);
    // on: a draft below the native block runs the native 8-row block; off: every draft runs the full block (a narrow
    // verify takes its head)
    void native_block(bool on);
    // size the far caches once for the request's rows: the x1.5 growth step at the first decode inject would add half
    // the prefill's caches again (248K: +3 GB)
    void reserve_far(int rows);
    // with TQLLM_MLX_DFLASH_RANKSTATS: the last draft's 16 candidates per proposed token, best unary score first
    const std::vector<std::vector<uint32_t>>& candidates() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

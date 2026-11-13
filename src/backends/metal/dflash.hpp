#pragma once
#include "qwen35.hpp"
#include <memory>
#include <optional>
#include <utility>
#include <vector>
namespace tqllm::mlx_backend {
struct DSpineBridgeResult { mlx::core::array output,raw,rms,packed; };
DSpineBridgeResult dspine_bridge(const mlx::core::array& raw,const mlx::core::array& anchor,
    const mlx::core::array& read,const mlx::core::array& write,const mlx::core::array& codebook,
    const mlx::core::array& packed_weights,const mlx::core::array& bias,float epsilon);
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
    // block_rows: the largest draft block (16 served; 8 for block-8 pricing tests)
    DFlash2(const std::string& path,Qwen35& target,int block_rows=16);
    ~DFlash2();
    const std::vector<int>& feature_layers() const;
    bool dspine() const;
    void reset();
    void inject(const mlx::core::array& features,int position);
    std::vector<uint32_t> draft(uint32_t anchor,int position,int count);
    // the selector lattice of a host draft (token-tree drafting): per drafted position the top-16 candidate ids, their
    // unary logits, the selector's gate row and each candidate's successor codebook row, plus the predecessor rows of
    // the candidates and the anchor (sorted by token). The drafter state advances as for draft()
    struct HostLattice {
        int positions=0,topk=0,rank=0;
        std::vector<uint32_t> tokens,predecessor_tokens;
        std::vector<float> unary,gate,successor,predecessor,dense_scores;
        const float* predecessor_row(uint32_t token) const;
    };
    HostLattice draft_lattice(uint32_t anchor,int position,int count);
    // the same draft with the selector's ids left unevaluated on the GPU (uint32 [count]); needs
    // feed it to Qwen35::verify_device without reading it back.
    mlx::core::array draft_device(uint32_t anchor,int position,int count);
    // draft-ahead (ring drafter, device selector): inject `count` (device int32, 1..rows) of the feature rows at `position`
    // (device int32, the first row's position) without a readback, and draft from an anchor id and position on the GPU,
    // so the next draft is queued behind the verify that produced them. Up to two draft blocks (32 rows) go in two ring
    // advances, so a tree step's accepted path may exceed the drafter's own block
    void inject_device(const mlx::core::array& features,const mlx::core::array& count,const mlx::core::array& position);
    mlx::core::array draft_device_at(const mlx::core::array& anchor,const mlx::core::array& position,int count);
    // after the step's readback: the last inject_device wrote `rows` rows (tightens the host's bound on the GPU fill)
    void settle_device_inject(int rows);
    // the ring's device head / fill back to the host at `position` (the target's position), so a host inject can continue
    // the same sequence (a request that extends the previous one's text)
    void settle_to_host(int position);
    // the ring window with its head / fill and the far caches' partial chunk-key sum at the injected position (host
    // injects only, i.e. during a prefill), for a later request that resumes the prompt there; the far rows below the
    // position stay in the caches, so it holds until the next reset
    class Kept {
        struct State;
        std::shared_ptr<const State> state_;
        friend class DFlash2;
    };
    Kept keep() const;
    // evaluate the caches (window, far context, chunk sums) and wait: see Qwen35::settle
    void settle() const;
    void restore(const Kept& kept);
    // rows one inject may carry (the pipeline's verify rows: 16, or 32 for a 31-node tree); sizes the attention span of a
    // draft issued before its inject settles
    void inject_block(int rows);
    void reset_inject_block();
    // draft bench: time_stages(true) syncs each stage of draft_device_at; stage_ms_per_call() returns the mean ms of
    // layers, head, top-16 and selector
    void time_stages(bool on);
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
    // with TQLLM_DEBUG=dflash_rankstats: the last draft's 16 candidates per proposed token, best unary score first
    const std::vector<std::vector<uint32_t>>& candidates() const;
    // the selector lattice of a block (rows 0 .. block - 1, row 0 the anchor's): candidate ids [rows, 16], unary logits,
    // gate rows [rows, rank] and successor rows [rows, 16, rank], all left on the device
    struct LatticeArrays {mlx::core::array mapped,unary,gate,successors;};
    // draft_device_at's lattice before the chain selector: the input of a device token tree (tree_device.hpp); the
    // drafter state advances as for draft_device_at
    LatticeArrays draft_lattice_device(const mlx::core::array& anchor,const mlx::core::array& position,int count);
    const mlx::core::array& selector_predecessor() const;  // [vocab, rank]: the predecessor codebook table
private:
    LatticeArrays lattice_arrays(uint32_t anchor,int position,int count);
    LatticeArrays device_lattice(const mlx::core::array& anchor,const mlx::core::array& position,int count);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

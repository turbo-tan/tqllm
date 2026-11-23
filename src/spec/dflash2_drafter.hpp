#pragma once
// DFlash2 block-diffusion drafter (llama.cpp `dflash` GGUF layout) as a module any engine can drive: the engine hands it
// the target's features of committed positions (inject) and asks for a draft chain after an anchor token (draft). The
// drafter owns its weights (via the weight-format registry), its sliding-window K/V ring and its scratch; the engine
// supplies its token embedding and its logits head as callbacks. RoPE (NeoX, full head) and theta come from the drafter
// GGUF: Flash-Next's drafter uses 10000 while the target uses 1e7.
#include "weight_format.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tqllm {

class Dflash2Drafter {
public:
    // host token ids -> out [n][hidden] BF16 on the device
    using EmbedFn = std::function<void(const std::vector<uint32_t> & tokens, __nv_bfloat16 * out, cudaStream_t)>;
    // x [n][hidden] BF16 -> logits [n][vocab] FP32 (the target's full LM head)
    using HeadFn = std::function<void(const __nv_bfloat16 * x, int n, float * logits, cudaStream_t)>;

    Dflash2Drafter(const std::string & path, int target_hidden, int vocab, float eps, cudaStream_t stream);
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter &) = delete;
    Dflash2Drafter & operator=(const Dflash2Drafter &) = delete;

    // zero-based target layers whose completed outputs form the features (the GGUF lists them one-based)
    const std::vector<int> & target_layers() const { return layers_; }
    int features() const { return int(layers_.size()) * hidden_; }  // per position
    int block() const { return block_; }
    size_t device_bytes() const { return bytes_; }
    // stop a chain at the first draft below this probability among its position's 16 best (0: always block - 1)
    void set_tau(float t) { tau_ = t; }

    // features [n][features()] BF16 of positions pos0 .. pos0 + n - 1 into every layer's K/V ring
    void inject(const __nv_bfloat16 * feats, int n, uint32_t pos0);
    // a chain of up to `want` (< block) draft tokens after `anchor`, the token at position pos
    std::vector<uint32_t> draft(uint32_t anchor, uint32_t pos, int want, const EmbedFn & embed, const HeadFn & head);

private:
    // row-concatenated weights; one part per run of equal types (K-quant files mix Q4_K / Q6_K within one group)
    struct Rows { std::vector<Mat> parts; int rows = 0, cols = 0; };
    struct Layer {
        const float * attn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr, * ffn_norm = nullptr;
        const float * attn_conv_base = nullptr, * ffn_conv_base = nullptr;
        Mat conv_attn, conv_ffn, wo, down;
        Rows w_qkv, w_gu;
        __nv_bfloat16 * kc = nullptr, * vc = nullptr;
    };
    void mm(const Mat & m, const __nv_bfloat16 * x, __nv_bfloat16 * y, int n);  // y [n][rows] = W x, x [n][cols]
    void mm(const Rows & m, const __nv_bfloat16 * x, __nv_bfloat16 * y, int n);
    void * dev(size_t bytes);  // owned device allocation

    cudaStream_t s_;
    int hidden_ = 0, vocab_ = 0, ffn_ = 0, heads_ = 0, kvheads_ = 0, hd_ = 0, block_ = 0, group_ = 0, window_ = 0;
    int rank_ = 0, ring_ = 0, conv_rows_ = 0;
    float eps_ = 1e-6f, theta_ = 10000.f, tau_ = 0.f;
    uint32_t mask_id_ = 0;
    std::vector<int> layers_;
    std::vector<Layer> L_;
    Mat fc_, sel_hidden_;
    Rows w_kv_all_;
    const float * enc_norm_ = nullptr, * out_norm_ = nullptr;
    const __nv_bfloat16 * sel_prev_ = nullptr, * sel_next_ = nullptr;  // [vocab][rank]
    std::vector<void *> owned_;
    size_t bytes_ = 0;
    // scratch
    __nv_bfloat16 * x_ = nullptr, * r_ = nullptr, * a_ = nullptr, * b_ = nullptr, * t_ = nullptr, * o_ = nullptr, * q_ = nullptr;
    __nv_bfloat16 * act_ = nullptr, * dyn_ = nullptr, * dynf_ = nullptr, * te_ = nullptr, * gate_ = nullptr;
    __half * xh_ = nullptr; size_t xh_cap_ = 0;
    __nv_bfloat16 * part_y_ = nullptr; size_t part_y_cap_ = 0;
    float * part_ = nullptr, * logits_ = nullptr, * vals_ = nullptr, * score_ = nullptr; void * topk_scratch_ = nullptr;
    uint32_t * idx_ = nullptr, * tok_ = nullptr; int * pos_ = nullptr;
    __nv_bfloat16 * inj_g_ = nullptr, * inj_kv_ = nullptr; int inj_cap_ = 0;
};

}  // namespace tqllm

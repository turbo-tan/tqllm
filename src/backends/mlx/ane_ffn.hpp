#pragma once
#include <memory>
#include <string>

namespace tqllm::mlx_backend {
// Neural Engine share of the prefill FFN: one compiled Core ML model per layer (dir/ffn_l<layer>.mlmodelc) computing
// silu(x Wg_s) * (x Wu_s) @ Wd_s for the last C intermediate channels, fp16 [rows, hidden] in and out.
// Models have a fixed row count; other chunk sizes stay on the GPU.
class AneFfn {
public:
    static std::unique_ptr<AneFfn> open(const std::string& dir,int layers);
    virtual ~AneFfn()=default;
    virtual bool has(int layer) const=0;
    virtual int rows() const=0;
    virtual int columns() const=0;  // C
    // blocking; x and y are row-major fp16 [rows, hidden] buffers that stay alive for the call
    virtual void predict(int layer,const void* x,void* y,int hidden)=0;
};
}

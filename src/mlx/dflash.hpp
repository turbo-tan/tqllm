#pragma once
#include "qwen35.hpp"
#include <memory>
#include <vector>
namespace tqllm::mlx_backend {
mlx::core::array dflash_conv(const mlx::core::array& input,const mlx::core::array& dynamic,
                            const mlx::core::array& base,int group,int side);
class DFlash2 {
public:
    DFlash2(const std::string& path,Qwen35& target);
    ~DFlash2();
    const std::vector<int>& feature_layers() const;
    void reset();
    void inject(const mlx::core::array& features,int position);
    std::vector<uint32_t> draft(uint32_t anchor,int position,int count);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

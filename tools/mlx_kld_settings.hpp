#pragma once
#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace tqllm::mlx_backend {
class KldKernelSettings {
    inline static constexpr std::array names{
        "TQLLM_MLX_TQ3_SUBGROUP", "TQLLM_MLX_MATMUL_COLS",
        "TQLLM_MLX_TILED_TQ3_FP16", "TQLLM_MLX_TILED_TQ3_FP32",
        "TQLLM_MLX_TILED_TQ4_FP16", "TQLLM_MLX_TILED_TQ4_FP32",
        "TQLLM_MLX_TILED_MIXER_FP16", "TQLLM_MLX_TILED_MIXER_FP32",
        "TQLLM_MLX_PAIRED_GATE_UP", "TQLLM_MLX_ATTN_DECODE_ROWS"};
    std::array<std::optional<std::string>,names.size()> values;
public:
    KldKernelSettings() {
        for(size_t i=0;i<names.size();++i)if(auto v=std::getenv(names[i]))values[i]=v;
    }
    KldKernelSettings(const KldKernelSettings&)=delete;
    KldKernelSettings& operator=(const KldKernelSettings&)=delete;
    void clear() const {
        for(auto name:names)if(unsetenv(name)!=0)throw std::runtime_error("failed to clear KLD kernel setting");
    }
    void restore() const {
        for(size_t i=0;i<names.size();++i) {
            int r=values[i]?setenv(names[i],values[i]->c_str(),1):unsetenv(names[i]);
            if(r!=0)throw std::runtime_error("failed to restore KLD kernel setting");
        }
    }
    ~KldKernelSettings() {
        for(size_t i=0;i<names.size();++i) {
            if(values[i])setenv(names[i],values[i]->c_str(),1);else unsetenv(names[i]);
        }
    }
};
}

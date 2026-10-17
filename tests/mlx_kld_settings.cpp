#include "../tools/mlx_kld_settings.hpp"
#include <iostream>
int main() {
    using tqllm::mlx_backend::KldKernelSettings;
    KldKernelSettings original;
    original.clear();
    setenv("TQLLM_MLX_PAIRED_GATE_UP","1",1);
    setenv("TQLLM_MLX_TILED_MIXER_FP16","1",1);
    setenv("TQLLM_MLX_TILED_TQ4_FP16","1",1);
    {
        KldKernelSettings candidate;candidate.clear();
        for(auto name:{"TQLLM_MLX_PAIRED_GATE_UP","TQLLM_MLX_TILED_MIXER_FP16","TQLLM_MLX_TILED_TQ4_FP16"})
            if(std::getenv(name))throw std::runtime_error("candidate leaked into reference");
        setenv("TQLLM_MLX_TILED_MIXER_FP32","1",1);
        candidate.restore();
        if(std::getenv("TQLLM_MLX_TILED_MIXER_FP32"))throw std::runtime_error("reference setting leaked into candidate");
        for(auto name:{"TQLLM_MLX_PAIRED_GATE_UP","TQLLM_MLX_TILED_MIXER_FP16","TQLLM_MLX_TILED_TQ4_FP16"})
            if(!std::getenv(name)||std::string(std::getenv(name))!="1")throw std::runtime_error("candidate not restored");
        candidate.clear();
    }
    if(!std::getenv("TQLLM_MLX_PAIRED_GATE_UP"))throw std::runtime_error("scope restore failed");
    try { KldKernelSettings temporary;temporary.clear();throw std::runtime_error("test unwind"); }
    catch(const std::runtime_error&) {}
    if(!std::getenv("TQLLM_MLX_PAIRED_GATE_UP"))throw std::runtime_error("exception-safe scope restore failed");
    std::cout<<"mlx_kld_settings: PASS\n";
}

#pragma once
#include "../../engine/engine.hpp"

namespace tqllm {
std::unique_ptr<Engine> make_mlx_inkling2_engine(const std::string& model,const std::string& tokenizer,
                                               uint32_t context,int slots,const EngineOptions& options);
}

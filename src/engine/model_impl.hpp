#pragma once
// tqllm::Model's state: the engine the registry picked for the model (src/engine/registry.hpp).
#include "engine.hpp"
#include <memory>

namespace tqllm {
struct Model::Impl {
    std::unique_ptr<Engine> e;
};
}  // namespace tqllm

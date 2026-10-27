#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tqllm::android_backend {
struct Options { int context = 4096; int threads = 4; };
// Native Arm CPU baseline. Standard GGUF weights stay packed and memory-mapped.
class Model {
public:
    Model(const std::string & path, Options options = {});
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    void reset();
    // Input must be consecutive. The returned logits belong to this instance.
    const std::vector<float> & forward(uint32_t token, bool head = true);
    int position() const;
    int context() const;
    int vocabulary() const;
    uint32_t eos() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

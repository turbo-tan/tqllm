#pragma once
#include "plugin.h"
#include <memory>
#include <string>
#include <vector>
namespace tqllm {
struct Plan {
    tq_invoke_fn invoke = nullptr;
    uint64_t elements = 0;
    std::shared_ptr<void> owner;
    void run(const float * a, const float * b, float * out) const;
};
class Runtime {
public:
    void load(const std::string & path);
    Plan resolve(const std::string & id, uint64_t elements) const;
    std::vector<std::string> kernels() const;
private:
    struct Entry { std::string id; uint64_t max_elements; tq_invoke_fn invoke; std::shared_ptr<void> owner; };
    std::vector<Entry> entries_;
};
}

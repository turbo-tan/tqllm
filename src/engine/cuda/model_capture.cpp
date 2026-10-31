// Model::capture reads the engine's device logits: CUDA builds only.
#include "../model_impl.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
}  // namespace

void Model::capture(const std::vector<uint32_t> & prompt, std::vector<float> & hidden, std::vector<float> & logits) {
    auto & e = *impl_->e;
    if (prompt.empty() || prompt.size() > e.context()) throw std::invalid_argument("capture: bad prompt length");
    const auto n = uint32_t(prompt.size());
    for (uint32_t i = 0; i + 1 < n; ++i) e.step(prompt[i], i, false, nullptr);
    hidden.assign(size_t(e.layers()) * e.hidden(), 0.f);
    e.step(prompt[n - 1], n - 1, true, hidden.data());
    logits.resize(e.vocab());
    ck(cudaMemcpy(logits.data(), e.logits(), logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace tqllm

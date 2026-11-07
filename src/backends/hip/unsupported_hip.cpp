// AMD/HIP: the parts of tqLLM that are out of scope for the ROCm backend in this PR, kept linkable with
// a clear error instead of a missing symbol. Every call site is gated by a feature the AMD build turns
// off (Flash-Next QSA KV, vision/multimodal input); nothing here silently changes results.
#include "kv_qsa.cuh"
#include "vision_kernels.cuh"
#include "engine.hpp"

#include <stdexcept>

namespace tqllm_cuda {

static void hip_unsupported(const char * what) {
    throw std::runtime_error(std::string(what) + " is not supported by the AMD (HIP) backend yet: "
                             "Flash-Next and vision stay CUDA-only in this release");
}

void qsa_attention_direct(const float *, const __nv_bfloat16 *, const __nv_bfloat16 *, const QsaDeviceSelection &, float *,
                          int, int, int, int, int, cudaStream_t) {
    hip_unsupported("QSA attention");
}

bool qsa_attention_dense_tc(const float *, const __nv_bfloat16 *, const __nv_bfloat16 *, float *, int, int, int, int, int,
                            cudaStream_t) {
    return false;  // "shape not covered": the caller falls back to the dense path
}

void qsa_attention_grouped(const float *, const __nv_bfloat16 *, const __nv_bfloat16 *, const QsaDeviceSelection &, float *,
                           float *, float *, int, int, int, int, int, cudaStream_t) {
    hip_unsupported("QSA attention");
}

void mm_scatter(__nv_bfloat16 *, const int *, int, const __nv_bfloat16 *, int, cudaStream_t) {
    hip_unsupported("multimodal image-token scatter");
}

}  // namespace tqllm_cuda

// The GLM-5next and MiniMax-M3 engines are not part of the AMD build in this PR: the server still probes
// model files, and a file for one of them gets a clear error instead of a missing symbol.
namespace tqllm {
bool is_glm5next_gguf(const std::string &) { return false; }
bool is_minimax_gguf(const std::string &) { return false; }
std::unique_ptr<Engine> make_glm5next_engine(const std::string &, uint32_t) {
    throw std::runtime_error("GLM-5next is not supported by the AMD (HIP) backend yet");
}
std::unique_ptr<Engine> make_minimax_engine(const std::string &, uint32_t, const std::string &, const std::string &,
                                            const TpConfig &, const std::string &, const std::string &) {
    throw std::runtime_error("MiniMax-M3 is not supported by the AMD (HIP) backend yet");
}
}  // namespace tqllm

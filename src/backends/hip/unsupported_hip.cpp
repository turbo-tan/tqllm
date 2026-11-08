// AMD/HIP: the parts of tqLLM that are out of scope for the ROCm backend in this PR, kept linkable with
// a clear error instead of a missing symbol. Every call site is gated by a feature the AMD build turns
// off (Flash-Next QSA KV, vision/multimodal input); nothing here silently changes results.
#include "kv_qsa.cuh"
#include "vision_kernels.cuh"
#include <cuda_runtime.h>

// MMQ expert GEMM is CUDA-only (third-party ggml_mmq). On HIP, mmq_supports returns false so the
// dense u4h path is used instead. These stubs keep the Flash-Next engine linkable.
namespace tqllm_cuda {
size_t mmq_warm() { return 0; }
int mmq_max_tiles(int, int) { return 0; }
bool mmq_supports(int, int) { return false; }
size_t mmq_act_bytes(int, int) { return 0; }
void mmq_quantize_act(const __half *, int, int, void *, cudaStream_t, bool) {}
void mmq_quantize_act_rows(const __half *, int, int, const int *, const int *, int, int, void *, cudaStream_t) {}
bool mmq_expert_gemm(int, const uint8_t *, const void *, bool, int, int, const int *, const int *, const int *,
                     int, int, int, const void *, int, __nv_bfloat16 *, int2 *, int, cudaStream_t,
                     const __nv_bfloat16 *, __half *) { return false; }
}  // namespace tqllm_cuda

#include "engine.hpp"
#include "gguf.hpp"

#include <stdexcept>

namespace tqllm_cuda {

void qsa_reserve_split_scratch() {}  // QSA split scratch: Flash-Next-only, not built on HIP

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

void vis_f32_to_h(const float *, __half *, int64_t, cudaStream_t) {
    throw std::runtime_error("vision input is not supported by the AMD (HIP) backend yet: --mmproj needs the CUDA build");
}

}  // namespace tqllm_cuda

// The GLM-5next and MiniMax-M3 engines are not part of the AMD build in this PR. The *detection* is host-only GGUF
// metadata (src/core/gguf.cpp), so it stays exact here: a file for one of them must be recognised and refused by name,
// not fall through to the qwen35 loader (which would report a confusing tensor error instead of the truth).
namespace tqllm {
bool is_glm5next_gguf(const std::string & path) {
    const std::string a = gguf_architecture(path);
    return a == "glm5-next" || a == "glm5next";
}
bool is_minimax_gguf(const std::string & path) { return gguf_architecture(path) == "minimax-m3"; }
std::unique_ptr<Engine> make_glm5next_engine(const std::string &, uint32_t) {
    throw std::runtime_error("GLM-5next is not supported by the AMD (HIP) backend yet");
}
std::unique_ptr<Engine> make_minimax_engine(const std::string &, uint32_t, const std::string &, const std::string &,
                                            const TpConfig &, const std::string &, const std::string &) {
    throw std::runtime_error("MiniMax-M3 is not supported by the AMD (HIP) backend yet");
}
}  // namespace tqllm

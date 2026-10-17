#include "weight_format.hpp"
#include "quant_kernels.cuh"

#include <map>
#include <mutex>
#include <stdexcept>

namespace tqllm {
void WeightFormat::expert_matvec(const Mat &, const __half *, __nv_bfloat16 *, int,
                                 const int *, int, int, bool, cudaStream_t) const {
    throw std::runtime_error(std::string(name()) + ": grouped experts are unsupported");
}


void WeightFormat::expert_matvec_dedup(const Mat & m, const __half * x, __nv_bfloat16 * y, int rows, const int * dmap,
                                       int groups, cudaStream_t s) const {
    if (u4h_id() < 0) throw std::runtime_error(std::string(name()) + ": dedup grouped experts are unsupported");
    tqllm_cuda::k_expert_matvec_dedup(u4h_id(), static_cast<const uint8_t *>(m.p), m.aux, x, y, rows, m.cols, dmap, groups, s);
}

void WeightFormat::expert_matvec_dedup2(const Mat & a, const Mat & b, const __half * x, __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows,
                                        const int * dmap, int groups, cudaStream_t s) const {
    if (u4h_id() < 0 || a.cols != b.cols) throw std::runtime_error(std::string(name()) + ": paired dedup experts are unsupported");
    tqllm_cuda::k_expert_matvec_dedup2(u4h_id(), static_cast<const uint8_t *>(a.p), a.aux, static_cast<const uint8_t *>(b.p), b.aux, x, ya, yb,
                                       rows, a.cols, dmap, groups, s);
}

void WeightFormat::expert_matvec2(const Mat & a, const Mat & b, const __half * x, __nv_bfloat16 * ya, __nv_bfloat16 * yb, int rows,
                                  const int * ids, int n, int topk, cudaStream_t s) const {
    if (u4h_id() < 0 || a.cols != b.cols) throw std::runtime_error(std::string(name()) + ": paired grouped experts are unsupported");
    tqllm_cuda::k_expert_matvec2(u4h_id(), static_cast<const uint8_t *>(a.p), a.aux, static_cast<const uint8_t *>(b.p), b.aux, x, ya, yb,
                                 rows, a.cols, ids, n, topk, s);
}

void WeightFormat::matvec_f32(const Mat &, const __half *, float *, int, cudaStream_t) const {
    throw std::runtime_error(std::string("weight format ") + name() + " has no FP32-output matvec");
}

namespace formats {  // built-in formats; a new format adds one declaration and one call below
void register_tq3_4s();
void register_q4();
void register_tq4();
void register_tq3();
void register_i8h();
void register_gsq_q2();
}

namespace {
std::map<uint32_t, std::unique_ptr<WeightFormat>> & table() {
    static std::map<uint32_t, std::unique_ptr<WeightFormat>> t;
    return t;
}
void builtins() {
    static std::once_flag once;
    std::call_once(once, [] {
        formats::register_tq3_4s();
        formats::register_q4();
        formats::register_tq4();
        formats::register_tq3();
        formats::register_i8h();
        formats::register_gsq_q2();
    });
}
}  // namespace

void register_weight_format(uint32_t gguf_type, std::unique_ptr<WeightFormat> f) {
    if (!table().emplace(gguf_type, std::move(f)).second)
        throw std::runtime_error("weight format id " + std::to_string(gguf_type) + " registered twice");
}

const WeightFormat * weight_format(uint32_t gguf_type) {
    builtins();
    const auto it = table().find(gguf_type);
    return it == table().end() ? nullptr : it->second.get();
}

}  // namespace tqllm

namespace tqllm {
bool WeightFormat::gemm_f32(const Mat & m, const __half * x, float * y, int n, cudaStream_t s) const {
    const int id = u4h_id();
    if (id < 0 || m.cols % 64) return false;
    return tqllm_cuda::u4h_family_gemm_f32(id, static_cast<const uint8_t *>(m.p), m.aux, x, y, m.rows, m.cols, n, s, m.rows);
}
}  // namespace tqllm

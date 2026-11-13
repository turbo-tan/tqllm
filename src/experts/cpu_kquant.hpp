#pragma once
// CPU expert matvec for K-quant GGUF rows (Q2_K / Q3_K / Q4_K): the formats of GLM-5.3-Flash GSQ-RCO's experts, so a
// missed expert runs on the CPU straight from the mapped GGUF bytes (no re-pack, no copy over PCIe).
// x is quantised once to int8 per 256-value block (Xq); each row is an int8 dot on the codes (AVX2 maddubs / madd where
// the CPU has it, a scalar loop otherwise) with the K-quant scales and mins applied per sub-block. Lossy only by the int8
// activation, as the GPU's quantised-activation paths are. Single-threaded: callers split rows. tests/cpu_kquant.cpp.
#include <cstdint>
#include <vector>

namespace tqllm { namespace cpu {

// x quantised per 256 values: q[k], d[k / 256] (x = q * d), sums of q per 16 values (the K-quant min terms)
struct Xq {
    int cols = 0;
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<int16_t> sum16;
};
void quantize_x(const float * x, int cols, Xq & out);  // cols: a multiple of 256

// GGML type ids: Q2_K 10, Q3_K 11, Q4_K 12
bool kq_supported(uint32_t type);
int64_t kq_row_bytes(uint32_t type, int cols);
// y[r] = dot(row r of W, x) for rows [r0, r1); W is rows x cols in the GGUF layout of `type`
void kq_matvec(uint32_t type, const uint8_t * W, int cols, const Xq & x, int r0, int r1, float * y);
// the same with the portable loop (reference, and the path on CPUs without AVX2)
void kq_matvec_scalar(uint32_t type, const uint8_t * W, int cols, const Xq & x, int r0, int r1, float * y);

}}  // namespace tqllm::cpu

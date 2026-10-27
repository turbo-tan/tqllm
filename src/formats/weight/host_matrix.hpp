#pragma once
#include "dequant.hpp"
#include "gguf.hpp"
#include "host_executor.hpp"
#include "mapped_file.hpp"
#include "host_packed_dot.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tqllm {
inline float host_dot(const float * a, const float * b, int n) {
    float sum = 0.f;
    int j = 0;
#if defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0.f);
    for (; j + 4 <= n; j += 4) acc = vfmaq_f32(acc, vld1q_f32(a + j), vld1q_f32(b + j));
    sum = vaddvq_f32(acc);
#endif
    for (; j < n; ++j) sum += a[j] * b[j];
    return sum;
}
struct HostMatrix {
    int rows, cols;
    uint32_t type;
    size_t row_bytes;
    const unsigned char * data;
    HostMatrix(const GgufTensor & t, const MappedFile & mapping, int expected_cols, int expected_rows)
        : rows(expected_rows), cols(expected_cols), type(t.type), row_bytes(0), data(nullptr) {
        if (rows <= 0 || cols <= 0 || t.ne != std::vector<uint64_t>{uint64_t(cols), uint64_t(rows)})
            throw std::invalid_argument("native matrix shape mismatch");
        switch (type) {
            case GGML_F32: case GGML_F16: case GGML_BF16: case GGML_Q8_0:
            case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: break;
            default: throw std::invalid_argument("native Android unsupported weight type: " + std::to_string(type));
        }
        uint64_t block, bytes;
        if (!gguf_block_layout(type, block, bytes) || uint64_t(cols) % block)
            throw std::invalid_argument("native matrix row is not block aligned");
        row_bytes = size_t(cols) / size_t(block) * size_t(bytes);
        if (uint64_t(row_bytes) * uint64_t(rows) != t.bytes) throw std::invalid_argument("native matrix byte size");
        data = mapping.at(t.offset, t.bytes);
    }
    std::vector<float> row(int r) const {
        if (r < 0 || r >= rows) throw std::out_of_range("native matrix row");
        std::vector<float> out(cols);
        dequantize_host(type, data + size_t(r) * row_bytes, out.data(), cols);
        return out;
    }
    std::vector<float> multiply(const std::vector<float> & x, HostExecutor & executor, bool packed = true) const {
        if (x.size() != size_t(cols)) throw std::invalid_argument("native matrix input shape");
        std::vector<float> out(rows);
        executor.rows(rows, [&](int begin, int end) {
#if defined(__aarch64__)
            if (packed && (type == GGML_Q4_K || type == GGML_Q5_K || type == GGML_Q6_K || type == GGML_Q8_0)) {
                for (int r = begin; r < end; ++r)
                    out[r] = host_packed_dot(type, data + size_t(r) * row_bytes, x.data(), cols);
                return;
            }
#else
            (void) packed;
#endif
            thread_local std::vector<float> scratch;
            scratch.resize(cols);
            for (int r = begin; r < end; ++r) {
                dequantize_host(type, data + size_t(r) * row_bytes, scratch.data(), cols);
                out[r] = host_dot(scratch.data(), x.data(), cols);
            }
        });
        return out;
    }
};
}

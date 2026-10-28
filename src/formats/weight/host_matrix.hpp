#pragma once
#include "dequant.hpp"
#include "gguf.hpp"
#include "host_executor.hpp"
#include "mapped_file.hpp"
#include "host_packed_dot.hpp"
#include "host_q8_dot.hpp"
#include "host_q8_mmla.hpp"
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
            case GGML_Q3_K: case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K:
            case GGML_IQ3_XXS: case GGML_IQ3_S: case GGML_IQ2_S: case GGML_IQ4_XS: break;
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
    bool int8_ready() const { return host_q8_supported(type, cols); }
    // Approximate: the activation is int8 per 256 elements; weights are read exactly as stored.
    std::vector<float> multiply(const HostQ8 & x, HostExecutor & executor) const {
        return std::move(multiply(std::vector<const HostQ8 *>{&x}, executor)[0]);
    }
    // A batch reads each row once for every activation; result[i] belongs to xs[i], bit-identical to one at a time.
    // limit > 0 computes only the first `limit` rows (a draft head over the common vocabulary); the rest stay 0.
    std::vector<std::vector<float>> multiply(const std::vector<const HostQ8 *> & xs, HostExecutor & executor,
                                             int limit = 0) const {
        if (!int8_ready() || xs.empty()) throw std::invalid_argument("native int8 matrix input");
        for (auto * x : xs) if (x->n != cols) throw std::invalid_argument("native int8 matrix input");
        const int count = int(xs.size());
        const int rows = limit > 0 ? std::min(limit, this->rows) : this->rows;
        std::vector<float> flat(size_t(rows) * count);
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
        if (count > 1 && host_mmla_supported(type, cols)) {
            std::vector<HostQ8Pair> pairs((count + 1) / 2);
            std::vector<const HostQ8Pair *> ps;
            for (int i = 0; i < int(pairs.size()); ++i) {
                host_pair_q8(*xs[2 * i], 2 * i + 1 < count ? xs[2 * i + 1] : nullptr, pairs[i]);
                ps.push_back(&pairs[i]);
            }
            executor.rows((rows + 1) / 2, [&](int begin, int end) {
                std::vector<float32x4_t> lanes(ps.size());
                for (int pr = begin; pr < end; ++pr) {
                    const int r0 = 2 * pr, r1 = std::min(r0 + 1, rows - 1);
                    host_mmla_batch(type, data + size_t(r0) * row_bytes, data + size_t(r1) * row_bytes,
                                    ps.data(), int(ps.size()), lanes.data());
                    for (int i = 0; i < count; ++i) {
                        const float32x4_t v = lanes[i / 2];
                        flat[size_t(r0) * count + i] = i % 2 ? vgetq_lane_f32(v, 1) : vgetq_lane_f32(v, 0);
                        if (r1 != r0) flat[size_t(r1) * count + i] = i % 2 ? vgetq_lane_f32(v, 3) : vgetq_lane_f32(v, 2);
                    }
                }
            });
        } else
#endif
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
        executor.rows(rows, [&](int begin, int end) {
            for (int r = begin; r < end; ++r)
                host_q8_dot_batch(type, data + size_t(r) * row_bytes, xs.data(), count, flat.data() + size_t(r) * count);
        });
#else
        (void) executor;
#endif
        std::vector<std::vector<float>> out(count, std::vector<float>(this->rows));
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < count; ++i) out[i][r] = flat[size_t(r) * count + i];
        return out;
    }
};
}

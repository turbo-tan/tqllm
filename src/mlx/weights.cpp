#include "weights.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
mx::Shape checked_shape(const GgufTensor& tensor) {
    if (tensor.ne.empty()) throw std::invalid_argument("empty GGUF tensor dimensions");
    mx::Shape shape;
    for (auto it = tensor.ne.rbegin(); it != tensor.ne.rend(); ++it) {
        if (*it == 0 || *it > uint64_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("GGUF tensor dimension exceeds MLX shape range");
        shape.push_back(int(*it));
    }
    return shape;
}
uint64_t elements(const mx::Shape& shape) {
    uint64_t n = 1;
    for (int d : shape) {
        if (n > std::numeric_limits<uint64_t>::max() / uint64_t(d))
            throw std::invalid_argument("GGUF tensor element count overflow");
        n *= uint64_t(d);
    }
    return n;
}
}
mx::array load_plain_tensor(const GgufTensor& tensor, std::span<const uint8_t> payload) {
    auto shape = checked_shape(tensor);
    const auto n = elements(shape);
    const uint64_t width = tensor.type == GGML_F32 ? 4 : 2;
    if (n > std::numeric_limits<uint64_t>::max() / width || payload.size() != n * width)
        throw std::invalid_argument("plain tensor payload length mismatch");
    if (tensor.type == GGML_F32) {
        std::vector<float> data(static_cast<size_t>(n));
        std::memcpy(data.data(), payload.data(), payload.size());
        return mx::array(data.data(), shape);
    }
    if (tensor.type != GGML_F16 && tensor.type != GGML_BF16)
        throw std::invalid_argument("unsupported plain tensor type");
    std::vector<uint16_t> data(static_cast<size_t>(n));
    std::memcpy(data.data(), payload.data(), payload.size());
    return mx::view(mx::array(data.data(), shape, mx::uint16),
                    tensor.type == GGML_F16 ? mx::float16 : mx::bfloat16);
}
Matrix load_matrix(const GgufTensor& tensor, std::span<const uint8_t> payload) {
    auto shape = checked_shape(tensor);
    if (shape.size() != 2) throw std::invalid_argument("linear weights must be a matrix");
    switch (tensor.type) {
        case GGML_F32: case GGML_F16: case GGML_BF16: return load_plain_tensor(tensor, payload);
        case GGML_TQ3: return load_tq3_blocks(payload, shape[0], shape[1]);
        case GGML_TQ4: return load_tq4_blocks(payload, shape[0], shape[1]);
        case GGML_Q6_K: return load_q6_blocks(payload, shape[0], shape[1]);
        case GGML_Q8_0: return load_q8_blocks(payload, shape[0], shape[1]);
        default: throw std::invalid_argument("unsupported MLX linear weight type " + std::to_string(tensor.type));
    }
}
Matrix load_matrix(const GgufFile& file, const std::string& name) {
    const auto& tensor = file.tensors.at(name);
    auto payload = file.read(tensor);
    return load_matrix(tensor, {reinterpret_cast<const uint8_t*>(payload.data()), payload.size()});
}
mx::array linear(const Matrix& weights, const mx::array& input) {
    if (input.ndim() != 2 || input.dtype() != mx::float32 || input.shape(0) <= 0)
        throw std::invalid_argument("linear input must be a nonempty FP32 matrix");
    return std::visit([&](const auto& w) -> mx::array {
        using T = std::decay_t<decltype(w)>;
        if constexpr (std::is_same_v<T, mx::array>) {
            if (w.ndim() != 2 || w.shape(1) != input.shape(1))
                throw std::invalid_argument("incompatible dense linear dimensions");
            return mx::matmul(input, mx::transpose(mx::astype(w, mx::float32)));
        } else if constexpr (std::is_same_v<T, Tq3Weights>) {
            return tq3_matmul(w, tq4_rotate(input));
        } else if constexpr (std::is_same_v<T, Tq4Weights>) {
            return tq4_matmul(w, tq4_rotate(input));
        } else if constexpr (std::is_same_v<T, Q6Weights>) {
            return q6_matmul(w, input);
        } else {
            return q8_matmul(w, input);
        }
    }, weights);
}
}

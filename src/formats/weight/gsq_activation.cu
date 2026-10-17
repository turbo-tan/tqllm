#include "gsq_activation.cuh"
#include <cmath>
#include <stdexcept>

namespace {
template<int Block> __global__ void round_h(const float * x, __half * y) {
    const int lane = threadIdx.x;
    const int64_t offset = int64_t(blockIdx.x) * Block;
    __shared__ float maxima[Block];
    __shared__ int positions[Block];
    const float value = x[offset + lane];
    maxima[lane] = value;
    positions[lane] = lane;
    __syncthreads();
    for (int step = Block / 2; step; step >>= 1) {
        if (lane < step) {
            const float candidate = maxima[lane + step];
            if (fabsf(candidate) > fabsf(maxima[lane]) ||
                (fabsf(candidate) == fabsf(maxima[lane]) && positions[lane + step] < positions[lane])) {
                maxima[lane] = candidate;
                positions[lane] = positions[lane + step];
            }
        }
        __syncthreads();
    }
    float scale = 0.f, inverse = 0.f;
    if (maxima[0] != 0.f) {
        if constexpr (Block == 256) {
            inverse = -127.f / maxima[0];
            scale = 1.f / inverse;
        } else {
            const float original = fabsf(maxima[0]) / 127.f;
            inverse = 1.f / original;
            scale = __half2float(__float2half_rn(original));
        }
    }
    const float code = fminf(127.f, roundf(value * inverse));
    y[offset + lane] = __float2half_rn(code * scale);
}

template<int Block> void launch(const float * x, __half * y, int64_t count, cudaStream_t stream) {
    if (!x || !y || count < 0 || count % Block || count / Block > 65535)
        throw std::invalid_argument("GSQ activation: invalid pointer or block count");
    if (count) round_h<Block><<<unsigned(count / Block), Block, 0, stream>>>(x, y);
}
}
namespace tqllm_cuda {
void gsq_q8k_round_h(const float * x, __half * y, int64_t count, cudaStream_t stream) {
    launch<256>(x, y, count, stream);
}
void gsq_q8_round_h(const float * x, __half * y, int64_t count, cudaStream_t stream) {
    launch<32>(x, y, count, stream);
}
}

#include "tq4.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
const char* source = R"METAL(
    constant float* values = INTEGER_LEVELS ? LVI : LV;
    // Decode each row's packed weights once, then reuse them across the M activation columns.
    uint lane = thread_position_in_threadgroup.x;
    uint col0 = threadgroup_position_in_grid.z * MCOLS;
    uint row0 = (threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y) * RPS;
    if (row0 >= R) return;
    const uint KV = K / 32;                      // uint4 chunks (32 codes) per row
    const uint KS = K / 128;                     // scales per row
    float acc[RPS][MCOLS];
    for (uint r = 0; r < RPS; ++r) for (uint m = 0; m < MCOLS; ++m) acc[r][m] = 0.0f;
    for (uint v = lane; v < KV; v += 32) {
        uint k = v * 32;
        float4 wv[RPS][8];
        for (uint r = 0; r < RPS; ++r) {
            uint row = row0 + r;
            if (row >= R) break;
            uint4 c4 = ((const device uint4 *)codes)[row * KV + v];
            float s = float(scale[row * KS + (k >> 7)]);
            uint cw[4] = {c4.x, c4.y, c4.z, c4.w};
            for (uint q = 0; q < 8; ++q) {
                uint packed = cw[q >> 1] >> ((q & 1u) * 16u);
                wv[r][q] = float4(values[packed & 15u], values[(packed >> 4) & 15u],
                                  values[(packed >> 8) & 15u], values[(packed >> 12) & 15u]) * s;
            }
        }
        for (uint m = 0; m < MCOLS; ++m) {
            if (col0 + m >= M) break;
            float4 xv[8];
            const device float4 * xp = (const device float4 *)(x + (col0 + m) * K + k);
            for (uint q = 0; q < 8; ++q) xv[q] = xp[q];
            for (uint r = 0; r < RPS; ++r) {
                if (row0 + r >= R) break;
                float a = 0.0f;
                for (uint q = 0; q < 8; ++q) a += dot(wv[r][q], xv[q]);
                acc[r][m] += a;
            }
        }
    }
    for (uint r = 0; r < RPS; ++r) {
        uint row = row0 + r;
        if (row >= R) break;
        for (uint m = 0; m < MCOLS; ++m) {
            if (col0 + m >= M) break;
            float t = simd_sum(acc[r][m]);
            if (lane == 0) y[(col0 + m) * R + row] = t;
        }
    }
)METAL";
const char* header = R"METAL(constant float LV[16] = {0.129099213f, 0.387297638f, 0.667012598f, 0.946727559f, 1.247959055f, 1.613740157f, 2.065587402f, 2.732600000f, -0.129099213f, -0.387297638f, -0.667012598f, -0.946727559f, -1.247959055f, -1.613740157f, -2.065587402f, -2.732600000f}; constant float LVI[16] = {6.0f,18.0f,31.0f,44.0f,58.0f,75.0f,96.0f,127.0f,-6.0f,-18.0f,-31.0f,-44.0f,-58.0f,-75.0f,-96.0f,-127.0f};)METAL";
}

Tq4Weights load_tq4_blocks(std::span<const uint8_t> blocks, int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % 128)
        throw std::invalid_argument("invalid TQ4 block dimensions");
    const uint64_t count = uint64_t(rows) * uint64_t(columns / 128);
    if (blocks.size() != count * 66)
        throw std::invalid_argument("TQ4 block payload size mismatch");
    std::vector<uint8_t> codes(size_t(count) * 64);
    std::vector<uint16_t> scales(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        scales[i] = uint16_t(blocks[i * 66]) | (uint16_t(blocks[i * 66 + 1]) << 8);
        std::memcpy(codes.data() + i * 64, blocks.data() + i * 66 + 2, 64);
    }
    return {mx::array(codes.data(), {rows, columns / 2}, mx::uint8),
            mx::view(mx::array(scales.data(), {rows, columns / 128}, mx::uint16), mx::float16)};
}

mx::array tq4_rotate(const mx::array& x) {
    if (x.ndim() != 2 || x.shape(0) <= 0 || x.shape(1) <= 0 || x.shape(1) % 128 ||
        (x.dtype() != mx::float32 && x.dtype() != mx::float16 && x.dtype() != mx::bfloat16))
        throw std::invalid_argument("TQ4 rotation requires floating point matrices with 128-column blocks");
    static const auto signs = [] {
        constexpr uint32_t words[] = {0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9};
        std::vector<float> s(128);
        for (int i = 0; i < 128; ++i) s[i] = 1.0f - 2.0f * ((words[i / 32] >> (i % 32)) & 1);
        return mx::array(s.data(), {128});
    }();
    auto blocks = mx::reshape(mx::astype(x, mx::float32), {x.shape(0), x.shape(1) / 128, 128});
    return mx::reshape(mx::hadamard_transform(mx::multiply(blocks, signs), 1.0f / std::sqrt(128.0f)), x.shape());
}

mx::array tq4_matmul(const mx::array& codes, const mx::array& scales,
                    const mx::array& x, int rps, int cols, int simds, bool integer_levels) {
    if (codes.ndim() != 2 || scales.ndim() != 2 || x.ndim() != 2 ||
        codes.dtype() != mx::uint8 || scales.dtype() != mx::float16 ||
        x.dtype() != mx::float32)
        throw std::invalid_argument("TQ4 requires uint8 codes, float16 scales and float32 rotated input matrices");
    const int rows = codes.shape(0), k = codes.shape(1) * 2, m = x.shape(0);
    if (rows <= 0 || k <= 0 || k % 128 || m <= 0 || x.shape(1) != k ||
        scales.shape(0) != rows || scales.shape(1) != k / 128)
        throw std::invalid_argument("TQ4 matrix shapes are incompatible");
    if ((rps != 1 && rps != 2 && rps != 4 && rps != 8) ||
        (cols != 1 && cols != 2 && cols != 4 && cols != 8 && cols != 16) ||
        (simds != 1 && simds != 2 && simds != 4 && simds != 8))
        throw std::invalid_argument("invalid TQ4 Metal tile");
    static auto kernel = mx::fast::metal_kernel("tqllm_tq4_matmul", {"codes", "scale", "x"}, {"y"}, source, header);
    const int groups = (rows + rps - 1) / rps;
    return kernel({codes, scales, x}, {{m, rows}}, {mx::float32},
                  {32, (groups + simds - 1) / simds * simds, (m + cols - 1) / cols}, {32, simds, 1},
                  {{"R", rows}, {"K", k}, {"M", m}, {"MCOLS", cols}, {"RPS", rps}, {"INTEGER_LEVELS", integer_levels}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
mx::array tq4_matmul(const Tq4Weights& weights, const mx::array& x, int rps, int cols, int simds) {
    return tq4_matmul(weights.codes, weights.scales, x, rps, cols, simds, weights.integer_levels);
}

}

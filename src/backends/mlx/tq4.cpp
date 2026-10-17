#include <string>
#include "tq4.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Affine8Weights tq4_affine8(const Tq4Weights& w) {
    if (!w.integer_levels || w.codes.ndim() != 2 || w.codes.dtype() != mx::uint8 ||
        w.codes.shape(0) <= 0 || w.codes.shape(1) <= 0 || w.codes.shape(1) % 64 ||
        w.scales.dtype() != mx::float16 ||
        w.scales.shape() != mx::Shape{w.codes.shape(0), w.codes.shape(1) / 64})
        throw std::invalid_argument("TQ4 affine expansion requires integer-level group128 weights");
    const int r = w.codes.shape(0), k = w.codes.shape(1) * 2;
    static auto expand = mx::fast::metal_kernel("tqllm_tq4_affine8", {"packed"}, {"codes"}, R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K)return;
uint row=i/K,col=i%K;
uint code=(packed[row*(K/2u)+col/2u]>>(4u*(col%2u)))&15u;
int level=int(L4[code&7u]);
codes[i]=uchar(128+(code<8u?level:-level));
)METAL", "constant uchar L4[8]={6,18,31,44,58,75,96,127};");
    auto bytes = expand({w.codes}, {{r, k}}, {mx::uint8}, {r * k, 1, 1}, {256, 1, 1},
        {{"R", r}, {"K", k}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    auto codes = mx::view(bytes, mx::uint32), scales = mx::astype(w.scales, mx::float32);
    auto biases = mx::multiply(scales, mx::array(-128.0f));
    auto half_scales = mx::astype(scales, mx::float16), half_biases = mx::astype(biases, mx::float16);
    mx::eval(codes, scales, biases, half_scales, half_biases);
    return {codes, scales, biases, half_scales, half_biases};
}
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
    static const bool skip = std::getenv("TQLLM_MLX_ABLATE_SMALL") && std::string(std::getenv("TQLLM_MLX_ABLATE_SMALL")).find("rot") != std::string::npos;
    if (skip) return mx::astype(x, mx::float32);
    auto blocks = mx::reshape(mx::astype(x, mx::float32), {x.shape(0), x.shape(1) / 128, 128});
    return mx::reshape(mx::hadamard_transform(mx::multiply(blocks, signs), 1.0f / std::sqrt(128.0f)), x.shape());
}

// tq4_rotate in one launch, FP16 out (the matrix-unit kernel's input): one simdgroup per 128-column block, lane l
// holds columns 4l..4l+3; signs, then the Sylvester-order Walsh-Hadamard butterflies (bits 0-1 in registers, bits
// 2-6 across lanes), then the 1/sqrt(128) scale. Per-row work only, so every row is independent of the batch.
mx::array tq4_rotate_half(const mx::array& x) {
    if (x.ndim() != 2 || x.shape(0) <= 0 || x.shape(1) <= 0 || x.shape(1) % 128 ||
        (x.dtype() != mx::float32 && x.dtype() != mx::float16 && x.dtype() != mx::bfloat16))
        throw std::invalid_argument("TQ4 rotation requires floating point matrices with 128-column blocks");
    static auto kernel = mx::fast::metal_kernel("tqllm_tq4_rotate_half", {"x"}, {"y"}, R"METAL(
const uint lane = thread_index_in_simdgroup;
const uint block = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
const uint row = threadgroup_position_in_grid.y;
if (block >= uint(K / 128)) return;
constexpr uint W[4] = {0xE46A0359u, 0x42F85949u, 0xA4C63BBBu, 0x49F250A9u};
const size_t base = (size_t)row * K + block * 128u + lane * 4u;
float v[4];
for (uint e = 0; e < 4u; ++e) {
    const uint i = lane * 4u + e;
    const float sign = ((W[i / 32u] >> (i % 32u)) & 1u) ? -1.0f : 1.0f;
    v[e] = float(x[base + e]) * sign;
}
float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
for (uint m = 1u; m < 32u; m <<= 1u)
    for (uint e = 0; e < 4u; ++e) {
        const float other = simd_shuffle_xor(v[e], ushort(m));
        v[e] = (lane & m) ? other - v[e] : v[e] + other;
    }
const float scale = 0.08838834764831845f;
for (uint e = 0; e < 4u; ++e) y[base + e] = half(v[e] * scale);
)METAL");
    const int blocks = x.shape(1) / 128;
    return kernel({mx::contiguous(x)}, {x.shape()}, {mx::float16}, {((blocks + 3) / 4) * 128, x.shape(0), 1}, {128, 1, 1},
                  {{"K", x.shape(1)}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
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

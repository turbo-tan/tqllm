#include "tq3_tiled.hpp"
#include <limits>
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Tq3TiledWeights tq3_tile(const Tq3Weights& w) {
    if (w.interleaved || w.planes.ndim() != 3 || w.planes.dtype() != mx::uint32 ||
        w.planes.shape(1) != 3 || w.scales.ndim() != 2 || w.scales.dtype() != mx::float16)
        throw std::invalid_argument("TQ3 tiling requires planar codes and FP16 scales");
    const int rows = w.planes.shape(0), blocks = w.planes.shape(2);
    if (rows <= 0 || rows > std::numeric_limits<int>::max() - 31 || blocks <= 0 ||
        blocks > std::numeric_limits<int>::max() / 32 || blocks % 4 ||
        w.scales.shape(0) != rows || w.scales.shape(1) != blocks / 4)
        throw std::invalid_argument("invalid TQ3 tiling dimensions");
    const int tiles = (rows + 31) / 32, pad = tiles * 32 - rows;
    auto codes = w.planes, scales = w.scales;
    if (pad) {
        codes = mx::concatenate({codes, mx::zeros({pad, 3, blocks}, mx::uint32)}, 0);
        scales = mx::concatenate({scales, mx::zeros({pad, blocks / 4}, mx::float16)}, 0);
    }
    codes = mx::contiguous(mx::transpose(mx::reshape(codes, {tiles, 32, 3, blocks}), {0, 3, 2, 1}));
    scales = mx::contiguous(mx::transpose(mx::reshape(scales, {tiles, 32, blocks / 4}), {0, 2, 1}));
    return {codes, scales, rows, blocks * 32};
}

TqTiledWeights tq4_tile(const Tq4Weights& w) {
    if (w.codes.ndim() != 2 || w.codes.dtype() != mx::uint8 || w.scales.ndim() != 2 ||
        w.scales.dtype() != mx::float16 || w.codes.shape(0) <= 0 ||
        w.codes.shape(0) > std::numeric_limits<int>::max() - 31 || w.codes.shape(1) <= 0 ||
        w.codes.shape(1) > std::numeric_limits<int>::max() / 2 || w.codes.shape(1) % 64 ||
        w.scales.shape() != mx::Shape{w.codes.shape(0), w.codes.shape(1) / 64})
        throw std::invalid_argument("invalid TQ4 tiling layout");
    const int rows = w.codes.shape(0), k = w.codes.shape(1) * 2, tiles = (rows + 31) / 32;
    auto codes = w.codes, scales = w.scales;
    if (tiles * 32 != rows) {
        codes = mx::concatenate({codes, mx::zeros({tiles * 32 - rows, k / 2}, mx::uint8)}, 0);
        scales = mx::concatenate({scales, mx::zeros({tiles * 32 - rows, k / 128}, mx::float16)}, 0);
    }
    codes = mx::contiguous(mx::transpose(mx::reshape(mx::view(mx::contiguous(codes), mx::uint32),
        {tiles, 32, k / 32, 4}), {0, 2, 3, 1}));
    scales = mx::contiguous(mx::transpose(mx::reshape(scales, {tiles, 32, k / 128}), {0, 2, 1}));
    return {codes, scales, rows, k, 4, w.integer_levels};
}

mx::array tq_tiled_matmul(const TqTiledWeights& w, const mx::array& input, int splits, bool prefetch,
    int tile_k, bool full_precision) {
    if (w.rows <= 0 || w.rows > std::numeric_limits<int>::max() - 63 || w.columns <= 0 ||
        w.columns % 128 || w.planes.ndim() != 4 || w.scales.ndim() != 3 ||
        w.planes.dtype() != mx::uint32 || w.scales.dtype() != mx::float16 ||
        (w.bits != 3 && w.bits != 4) || (w.bits == 3 && !w.integer_levels) ||
        w.planes.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 32, w.bits, 32} ||
        w.scales.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 128, 32} ||
        input.ndim() != 2 || input.dtype() != mx::float32 || input.shape(0) < 1 ||
        input.shape(0) > 8 || input.shape(1) != w.columns ||
        (splits != 1 && splits != 2 && splits != 4) || w.columns % (128 * splits) ||
        (tile_k != 32 && tile_k != 64 && tile_k != 128) || (full_precision && tile_k == 128))
        throw std::invalid_argument("invalid tiled TQ3 projection");
    const int m = input.shape(0);
    auto padded = input;
    if (m < 8) padded = mx::concatenate({padded, mx::zeros({8 - m, w.columns}, mx::float32)}, 0);
    auto operands = mx::contiguous(full_precision ? padded : mx::astype(padded, mx::float16));
    static auto kernel = mx::fast::metal_kernel("tqllm_tq3_coalesced_stage",
        {"planes", "scales", "x"}, {"partials"}, R"METAL(
uint lane = thread_position_in_threadgroup.x % 32u;
uint tile = threadgroup_position_in_grid.y;
uint part = threadgroup_position_in_grid.z;
if (tile >= (R + 31u) / 32u) return;
threadgroup Operand storage[2 * 32 * BK];
threadgroup Operand* first = storage;
threadgroup Operand* second = first + 32u * BK;
// TensorOps requires an unqualified operand type; this view is only read.
tensor<device Operand, dextents<int, 2>, tensor_inline> inputs(
    (device Operand*)x, dextents<int, 2>{int(K), 8}, array<int, 2>{1, int(K)});
tensor<threadgroup Operand, extents<int, 32, BK>, tensor_inline> stage0(first, extents<int, 32, BK>());
tensor<threadgroup Operand, extents<int, 32, BK>, tensor_inline> stage1(second, extents<int, 32, BK>());
constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(8, 32, BK, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<1>> operation;
auto initial = inputs.template slice<BK, 8>(0, 0);
auto accumulator = operation.template get_destination_cooperative_tensor<decltype(initial), decltype(stage0), float>();
for (uint i = 0; i < accumulator.get_capacity(); ++i)
    if (accumulator.is_valid_element(i)) accumulator[i] = 0.0f;
uint begin = part * (K / BK / SPLITS), end = (part + 1u) * (K / BK / SPLITS);
ulong origin = ulong(tile) * (K / 32u) * (32u * BITS) + lane;
ulong meta = ulong(tile) * (K / 128u) * 32u + lane;
uint low[BK/32], high[BK/32], sign[BK/32], fourth[BK/32];
for (uint g = 0; g < BK/32u; ++g) {
    ulong at = origin + ulong(begin * (BK/32u) + g) * (32u * BITS);
    low[g] = planes[at]; high[g] = planes[at + 32u]; sign[g] = planes[at + 64u];
    if (BITS == 4) fourth[g] = planes[at + 96u];
}
float scale = float(scales[meta + ulong(begin / (128u / BK)) * 32u]);
for (uint block = begin; block < end; ++block) {
    if (!PREFETCH) {
        for (uint g = 0; g < BK/32u; ++g) {
            ulong at = origin + ulong(block * (BK/32u) + g) * (32u * BITS);
            low[g] = planes[at]; high[g] = planes[at + 32u]; sign[g] = planes[at + 64u];
            if (BITS == 4) fourth[g] = planes[at + 96u];
        }
        scale = float(scales[meta + ulong(block / (128u / BK)) * 32u]);
    }
    threadgroup Operand* destination = (block & 1u) ? second : first;
    #pragma clang loop unroll(full)
    for (uint g = 0; g < BK/32u; ++g) {
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32u; ++j) {
        float value;
        if (BITS == 3) {
            uint ranks = j < 16u ? low[g] : high[g];
            uint rank = (ranks >> (4u * (j % 8u) + 2u * ((j % 16u) / 8u))) & 3u;
            uint negative = (sign[g] >> (4u * (j % 8u) + j / 8u)) & 1u;
            int level = int((0x7f4f2d0eu >> (rank * 8u)) & 255u);
            value = float(negative ? -level : level);
        } else {
            uint packed = j < 8u ? low[g] : j < 16u ? high[g] : j < 24u ? sign[g] : fourth[g];
            uint code = (packed >> (4u * (j % 8u))) & 15u;
            value = INTEGER_LEVELS ? L4I[code] : L4F[code];
        }
        destination[(g * 32u + j) * 32u + lane] = Operand(scale * value);
    }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (PREFETCH && block + 1u < end) {
        for (uint g = 0; g < BK/32u; ++g) {
            ulong at = origin + ulong((block + 1u) * (BK/32u) + g) * (32u * BITS);
            low[g] = planes[at]; high[g] = planes[at + 32u]; sign[g] = planes[at + 64u];
            if (BITS == 4) fourth[g] = planes[at + 96u];
        }
        if ((block + 1u) % (128u / BK) == 0u)
            scale = float(scales[meta + ulong((block + 1u) / (128u / BK)) * 32u]);
    }
    auto activation = inputs.template slice<BK, 8>(block * BK, 0);
    if (block & 1u) operation.run(activation, stage1, accumulator);
    else operation.run(activation, stage0, accumulator);
}
for (uint i = 0; i < accumulator.get_capacity(); ++i) {
    if (!accumulator.is_valid_element(i)) continue;
    auto position = accumulator.get_multidimensional_index(i);
    uint row = tile * 32u + uint(position[0]), column = uint(position[1]);
    if (row < R && column < M) partials[(part * M + column) * R + row] = accumulator[i];
}
)METAL", R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma clang fp reassociate(off)
constant float L4I[16] = {6.0f,18.0f,31.0f,44.0f,58.0f,75.0f,96.0f,127.0f,-6.0f,-18.0f,-31.0f,-44.0f,-58.0f,-75.0f,-96.0f,-127.0f};
constant float L4F[16] = {0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};
)METAL");
    auto output = kernel({w.planes, w.scales, operands}, {{splits, m, w.rows}}, {mx::float32},
        {32, (w.rows + 31) / 32, splits}, {32, 1, 1},
        {{"R", w.rows}, {"K", w.columns}, {"M", m}, {"SPLITS", splits}, {"PREFETCH", prefetch},
         {"BK", tile_k}, {"Operand", full_precision ? mx::float32 : mx::float16},
         {"BITS", w.bits}, {"INTEGER_LEVELS", w.integer_levels}},
        std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return splits == 1 ? mx::reshape(output, {m, w.rows}) : mx::sum(output, 0);
}
mx::array tq_compensated_matmul(const TqTiledWeights& w, const mx::array& input, int splits) {
    if (w.rows <= 0 || w.rows > std::numeric_limits<int>::max() - 31 || w.columns <= 0 ||
        (splits != 1 && splits != 2 && splits != 4) || w.columns % (128 * splits) ||
        (w.bits != 3 && w.bits != 4) || !w.integer_levels ||
        w.planes.dtype() != mx::uint32 || w.scales.dtype() != mx::float16 ||
        w.planes.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 32, w.bits, 32} ||
        w.scales.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 128, 32} ||
        input.ndim() != 2 || input.dtype() != mx::float32 || input.shape(0) < 1 ||
        input.shape(0) > 8 || input.shape(1) != w.columns)
        throw std::invalid_argument("invalid compensated integer TQ projection");
    const int m = input.shape(0);
    auto padded = input;
    if (m < 8) padded = mx::concatenate({padded, mx::zeros({8 - m, w.columns}, mx::float32)}, 0);
    auto high = mx::astype(padded, mx::float16);
    auto low = mx::astype((padded - mx::astype(high, mx::float32)) * mx::array(2048.0f), mx::float16);
    auto operands = mx::contiguous(mx::concatenate({high, low}, 0));
    static auto kernel = mx::fast::metal_kernel("tqllm_tq_compensated_integer_stage",
        {"planes", "scales", "x"}, {"partials"}, R"METAL(
uint lane = thread_position_in_threadgroup.x;
uint tile = threadgroup_position_in_grid.y;
uint part = threadgroup_position_in_grid.z;
threadgroup half storage[2 * 32 * 32];
threadgroup float dots[16 * 32];
tensor<device half, dextents<int, 2>, tensor_inline> inputs(
    (device half*)x, dextents<int, 2>{int(K), 16}, array<int, 2>{1, int(K)});
tensor<threadgroup half, extents<int, 32, 32>, tensor_inline> stage0(storage, extents<int, 32, 32>());
tensor<threadgroup half, extents<int, 32, 32>, tensor_inline> stage1(storage + 32 * 32, extents<int, 32, 32>());
constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, false, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<1>> operation;
auto initial = inputs.template slice<32, 16>(0, 0);
auto accumulator = operation.template get_destination_cooperative_tensor<decltype(initial), decltype(stage0), float>();
float sums[8];
for (uint col = 0; col < 8u; ++col) sums[col] = 0.0f;
uint begin = part * (K / 32u / SPLITS), end = (part + 1u) * (K / 32u / SPLITS);
ulong origin = ulong(tile) * (K / 32u) * (32u * BITS) + lane;
ulong meta = ulong(tile) * (K / 128u) * 32u + lane;
uint low_code = planes[origin + ulong(begin) * (32u * BITS)];
uint high_code = planes[origin + ulong(begin) * (32u * BITS) + 32u];
uint sign_code = planes[origin + ulong(begin) * (32u * BITS) + 64u];
uint fourth = 0u;
if (BITS == 4) fourth = planes[origin + ulong(begin) * (32u * BITS) + 96u];
for (uint block = begin; block < end; ++block) {
    if (block % 4u == 0u)
        for (uint i = 0; i < accumulator.get_capacity(); ++i)
            if (accumulator.is_valid_element(i)) accumulator[i] = 0.0f;
    threadgroup half* destination = storage + (block & 1u) * (32u * 32u);
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32u; ++j) {
        int level;
        if (BITS == 3) {
            uint ranks = j < 16u ? low_code : high_code;
            uint rank = (ranks >> (4u * (j % 8u) + 2u * ((j % 16u) / 8u))) & 3u;
            uint negative = (sign_code >> (4u * (j % 8u) + j / 8u)) & 1u;
            level = int((0x7f4f2d0eu >> (rank * 8u)) & 255u);
            if (negative) level = -level;
        } else {
            uint packed = j < 8u ? low_code : j < 16u ? high_code : j < 24u ? sign_code : fourth;
            uint code = (packed >> (4u * (j % 8u))) & 15u;
            level = LEVELS4[code];
        }
        destination[j * 32u + lane] = half(level);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (block + 1u < end) {
        ulong at = origin + ulong(block + 1u) * (32u * BITS);
        low_code = planes[at]; high_code = planes[at + 32u]; sign_code = planes[at + 64u];
        if (BITS == 4) fourth = planes[at + 96u];
    }
    auto activation = inputs.template slice<32, 16>(block * 32u, 0);
    if (block & 1u) operation.run(activation, stage1, accumulator);
    else operation.run(activation, stage0, accumulator);
    if (block % 4u == 3u) {
        for (uint i = 0; i < accumulator.get_capacity(); ++i) {
            if (!accumulator.is_valid_element(i)) continue;
            auto pos = accumulator.get_multidimensional_index(i);
            dots[uint(pos[1]) * 32u + uint(pos[0])] = accumulator[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float scale = float(scales[meta + ulong(block / 4u) * 32u]);
        for (uint col = 0; col < 8u; ++col)
            sums[col] += scale * (dots[col * 32u + lane] + dots[(col + 8u) * 32u + lane] * (1.0f / 2048.0f));
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
uint row = tile * 32u + lane;
if (row < R) for (uint col = 0; col < M; ++col)
    partials[(part * M + col) * R + row] = sums[col];
)METAL", R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma clang fp reassociate(off)
constant int LEVELS4[16] = {6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};
)METAL");
    auto output = kernel({w.planes, w.scales, operands}, {{splits, m, w.rows}}, {mx::float32},
        {32, (w.rows + 31) / 32, splits}, {32, 1, 1},
        {{"R", w.rows}, {"K", w.columns}, {"M", m}, {"SPLITS", splits}, {"BITS", w.bits}},
        std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return splits == 1 ? mx::reshape(output, {m, w.rows}) : mx::sum(output, 0);
}

}

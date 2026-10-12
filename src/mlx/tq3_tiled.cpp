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

mx::array tq3_tiled_matmul(const Tq3TiledWeights& w, const mx::array& input, int splits, bool prefetch) {
    if (w.rows <= 0 || w.rows > std::numeric_limits<int>::max() - 63 || w.columns <= 0 ||
        w.columns % 128 || w.planes.ndim() != 4 || w.scales.ndim() != 3 ||
        w.planes.dtype() != mx::uint32 || w.scales.dtype() != mx::float16 ||
        w.planes.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 32, 3, 32} ||
        w.scales.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 128, 32} ||
        input.ndim() != 2 || input.dtype() != mx::float32 || input.shape(0) < 1 ||
        input.shape(0) > 8 || input.shape(1) != w.columns ||
        (splits != 1 && splits != 2 && splits != 4) || w.columns % (128 * splits))
        throw std::invalid_argument("invalid tiled TQ3 projection");
    const int m = input.shape(0);
    auto padded = input;
    if (m < 8) padded = mx::concatenate({padded, mx::zeros({8 - m, w.columns}, mx::float32)}, 0);
    auto operands = mx::contiguous(mx::astype(padded, mx::float16));
    static auto kernel = mx::fast::metal_kernel("tqllm_tq3_coalesced_stage",
        {"planes", "scales", "x"}, {"partials"}, R"METAL(
uint lane = thread_position_in_threadgroup.x % 32u;
uint simd = thread_position_in_threadgroup.x / 32u;
uint tile = threadgroup_position_in_grid.y * 2u + simd;
uint part = threadgroup_position_in_grid.z;
if (tile >= (R + 31u) / 32u) return;
threadgroup half storage[4096];
threadgroup half* first = storage + simd * 2048u;
threadgroup half* second = first + 1024u;
// TensorOps requires an unqualified operand type; this view is only read.
tensor<device half, dextents<int, 2>, tensor_inline> inputs(
    (device half*)x, dextents<int, 2>{int(K), 8}, array<int, 2>{1, int(K)});
tensor<threadgroup half, extents<int, 32, 32>, tensor_inline> stage0(first, extents<int, 32, 32>());
tensor<threadgroup half, extents<int, 32, 32>, tensor_inline> stage1(second, extents<int, 32, 32>());
constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(8, 32, 32, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<1>> operation;
auto initial = inputs.template slice<32, 8>(0, 0);
auto accumulator = operation.template get_destination_cooperative_tensor<decltype(initial), decltype(stage0), float>();
for (uint i = 0; i < accumulator.get_capacity(); ++i)
    if (accumulator.is_valid_element(i)) accumulator[i] = 0.0f;
uint begin = part * (K / 32u / SPLITS), end = (part + 1u) * (K / 32u / SPLITS);
ulong origin = ulong(tile) * (K / 32u) * 96u + lane;
ulong meta = ulong(tile) * (K / 128u) * 32u + lane;
uint low = planes[origin + ulong(begin) * 96u];
uint high = planes[origin + ulong(begin) * 96u + 32u];
uint sign = planes[origin + ulong(begin) * 96u + 64u];
float scale = float(scales[meta + ulong(begin / 4u) * 32u]);
for (uint block = begin; block < end; ++block) {
    if (!PREFETCH) {
        low = planes[origin + ulong(block) * 96u];
        high = planes[origin + ulong(block) * 96u + 32u];
        sign = planes[origin + ulong(block) * 96u + 64u];
        scale = float(scales[meta + ulong(block / 4u) * 32u]);
    }
    threadgroup half* destination = (block & 1u) ? second : first;
    for (uint j = 0; j < 32u; ++j) {
        uint ranks = j < 16u ? low : high;
        uint rank = (ranks >> (4u * (j % 8u) + 2u * ((j % 16u) / 8u))) & 3u;
        uint negative = (sign >> (4u * (j % 8u) + j / 8u)) & 1u;
        int level = int((0x7f4f2d0eu >> (rank * 8u)) & 255u);
        destination[lane * 32u + j] = half(scale * float(negative ? -level : level));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (PREFETCH && block + 1u < end) {
        low = planes[origin + ulong(block + 1u) * 96u];
        high = planes[origin + ulong(block + 1u) * 96u + 32u];
        sign = planes[origin + ulong(block + 1u) * 96u + 64u];
        if ((block + 1u) % 4u == 0u) scale = float(scales[meta + ulong((block + 1u) / 4u) * 32u]);
    }
    auto activation = inputs.template slice<32, 8>(block * 32u, 0);
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
)METAL");
    auto output = kernel({w.planes, w.scales, operands}, {{splits, m, w.rows}}, {mx::float32},
        {64, (w.rows + 63) / 64, splits}, {64, 1, 1},
        {{"R", w.rows}, {"K", w.columns}, {"M", m}, {"SPLITS", splits}, {"PREFETCH", prefetch}},
        std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return splits == 1 ? mx::reshape(output, {m, w.rows}) : mx::sum(output, 0);
}
}

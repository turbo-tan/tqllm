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

mx::array tq3_tiled_matmul(const Tq3TiledWeights& w, const mx::array& input, int splits, bool prefetch,
    int tile_k, bool full_precision) {
    if (w.rows <= 0 || w.rows > std::numeric_limits<int>::max() - 63 || w.columns <= 0 ||
        w.columns % 128 || w.planes.ndim() != 4 || w.scales.ndim() != 3 ||
        w.planes.dtype() != mx::uint32 || w.scales.dtype() != mx::float16 ||
        w.planes.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 32, 3, 32} ||
        w.scales.shape() != mx::Shape{(w.rows + 31) / 32, w.columns / 128, 32} ||
        input.ndim() != 2 || input.dtype() != mx::float32 || input.shape(0) < 1 ||
        input.shape(0) > 8 || input.shape(1) != w.columns ||
        (splits != 1 && splits != 2 && splits != 4) || w.columns % (128 * splits) ||
        (tile_k != 32 && tile_k != 128) || (full_precision && tile_k != 32))
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
ulong origin = ulong(tile) * (K / 32u) * 96u + lane;
ulong meta = ulong(tile) * (K / 128u) * 32u + lane;
uint low[BK/32], high[BK/32], sign[BK/32];
for (uint g = 0; g < BK/32u; ++g) {
    ulong at = origin + ulong(begin * (BK/32u) + g) * 96u;
    low[g] = planes[at]; high[g] = planes[at + 32u]; sign[g] = planes[at + 64u];
}
float scale = float(scales[meta + ulong(begin / (128u / BK)) * 32u]);
for (uint block = begin; block < end; ++block) {
    if (!PREFETCH) {
        for (uint g = 0; g < BK/32u; ++g) {
            ulong at = origin + ulong(block * (BK/32u) + g) * 96u;
            low[g] = planes[at]; high[g] = planes[at + 32u]; sign[g] = planes[at + 64u];
        }
        scale = float(scales[meta + ulong(block / (128u / BK)) * 32u]);
    }
    threadgroup Operand* destination = (block & 1u) ? second : first;
    #pragma clang loop unroll(full)
    for (uint g = 0; g < BK/32u; ++g) {
    #pragma clang loop unroll(full)
    for (uint j = 0; j < 32u; ++j) {
        uint ranks = j < 16u ? low[g] : high[g];
        uint rank = (ranks >> (4u * (j % 8u) + 2u * ((j % 16u) / 8u))) & 3u;
        uint negative = (sign[g] >> (4u * (j % 8u) + j / 8u)) & 1u;
        int level = int((0x7f4f2d0eu >> (rank * 8u)) & 255u);
        destination[(g * 32u + j) * 32u + lane] = Operand(scale * float(negative ? -level : level));
    }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (PREFETCH && block + 1u < end) {
        for (uint g = 0; g < BK/32u; ++g) {
            ulong at = origin + ulong((block + 1u) * (BK/32u) + g) * 96u;
            low[g] = planes[at]; high[g] = planes[at + 32u]; sign[g] = planes[at + 64u];
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
)METAL");
    auto output = kernel({w.planes, w.scales, operands}, {{splits, m, w.rows}}, {mx::float32},
        {32, (w.rows + 31) / 32, splits}, {32, 1, 1},
        {{"R", w.rows}, {"K", w.columns}, {"M", m}, {"SPLITS", splits}, {"PREFETCH", prefetch},
         {"BK", tile_k}, {"Operand", full_precision ? mx::float32 : mx::float16}},
        std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return splits == 1 ? mx::reshape(output, {m, w.rows}) : mx::sum(output, 0);
}
}

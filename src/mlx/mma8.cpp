#include "mma8.hpp"
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
constexpr int NT = 2;
constexpr uint8_t L4[8] = {6, 18, 31, 44, 58, 75, 96, 127};
constexpr uint8_t L3[4] = {14, 45, 79, 127};
const char* header = R"METAL(
#include <metal_simdgroup_matrix>
using namespace metal;
)METAL";
const char* source = R"METAL(
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint g = threadgroup_position_in_grid.x * 4 + sg;
    const uint qid = lane >> 2;
    const uint fn = (qid & 2) * 2 + (lane & 1) * 2;
    threadgroup uint tlut[256];
    for (uint b = sg * 32 + lane; b < 256; b += 128) tlut[b] = lut[b];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 C[2];
    C[0] = simdgroup_float8x8(0);
    C[1] = simdgroup_float8x8(0);
    for (uint kb = 0; kb < K / 128; ++kb) {
        uint4 w4[2];
        float2 s2[2];
        simdgroup_float8x8 Ct[2];
        for (uint t = 0; t < 2; ++t) {
            ulong base = ((ulong)g * (K / 128) + kb) * 2 + t;
            w4[t] = ((const device uint4*)(W + base * 128))[lane];
            s2[t] = float2(S[base * 8 + fn], S[base * 8 + fn + 1]);
            Ct[t] = simdgroup_float8x8(0);
        }
        for (uint kk = 0; kk < 16; ++kk) {
            simdgroup_half8x8 A;
            simdgroup_load(A, X + kb * 128 + kk * 8, K);
            for (uint t = 0; t < 2; ++t) {
                uint word = w4[t][kk >> 2];
                uint byte = (word >> ((kk & 3) * 8)) & 255u;
                half2 hv = as_type<half2>(tlut[byte]);
                simdgroup_half8x8 B;
                B.thread_elements()[0] = hv.x;
                B.thread_elements()[1] = hv.y;
                simdgroup_multiply_accumulate(Ct[t], A, B, Ct[t]);
            }
        }
        for (uint t = 0; t < 2; ++t) {
            C[t].thread_elements()[0] += Ct[t].thread_elements()[0] * s2[t].x;
            C[t].thread_elements()[1] += Ct[t].thread_elements()[1] * s2[t].y;
        }
    }
    if (BF) {
        for (uint t = 0; t < 2; ++t) {
            C[t].thread_elements()[0] = float(bfloat16_t(C[t].thread_elements()[0]));
            C[t].thread_elements()[1] = float(bfloat16_t(C[t].thread_elements()[1]));
        }
    }
    for (uint t = 0; t < 2; ++t)
        simdgroup_store(C[t], out + (g * 2 + t) * 8, N);
)METAL";

std::map<std::pair<int,int>, mx::array>& gather_cache() {
    static std::map<std::pair<int,int>, mx::array> cache;
    return cache;
}
// Source position, in the flattened [rows_padded, K] nibble array, of every packed nibble.
const mx::array& gather_index(int rows_padded, int k) {
    auto& cache = gather_cache();
    auto key = std::make_pair(rows_padded, k);
    if (auto found = cache.find(key); found != cache.end()) return found->second;
    const int groups = rows_padded / (8 * NT), blocks = k / 128;
    std::vector<int32_t> index(size_t(rows_padded) * k);
    size_t at = 0;
    for (int g = 0; g < groups; ++g)
    for (int kb = 0; kb < blocks; ++kb)
    for (int t = 0; t < NT; ++t)
    for (int lane = 0; lane < 32; ++lane) {
        const int qid = lane / 4, fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
        for (int pos = 0; pos < 32; ++pos) {
            const int w = pos / 8, q = pos % 8, e = q % 2, b = q / 2, kk = 4 * w + b;
            const int row = (g * NT + t) * 8 + fn + e, col = kb * 128 + kk * 8 + fm;
            index[at++] = int32_t(size_t(row) * k + col);
        }
    }
    return cache.emplace(key, mx::array(index.data(), {rows_padded * k}, mx::int32)).first->second;
}
}

void mma8_clear_pack_cache() {gather_cache().clear();}

namespace {
mx::array level_table(const uint8_t* levels, int rank_mask, int sign_bit) {
    std::vector<uint32_t> table(256);
    for (uint32_t b = 0; b < 256; ++b) {
        uint16_t bits[2];
        for (int i = 0; i < 2; ++i) {
            const uint32_t c = (b >> (4 * i)) & 15u;
            const mx::float16_t value(float(levels[c & rank_mask]) * ((c & sign_bit) ? -1.0f : 1.0f));
            std::memcpy(&bits[i], &value, sizeof(uint16_t));
        }
        table[b] = uint32_t(bits[0]) | (uint32_t(bits[1]) << 16);
    }
    return mx::array(table.data(), {256}, mx::uint32);
}
// nibbles: [rows_padded * K] uint32 codes, scales: [rows_padded, K/128] fp16.
Mma8Weights finish_pack(const mx::array& nibbles, mx::array scales, const mx::array& lut, int rows, int rp, int k) {
    auto permuted = mx::reshape(mx::take(nibbles, gather_index(rp, k)), {-1, 8});
    std::vector<uint32_t> shifts;
    for (int i = 0; i < 8; ++i) shifts.push_back(uint32_t(4 * i));
    auto words = mx::sum(mx::left_shift(permuted, mx::array(shifts.data(), {8}, mx::uint32)), -1);
    words = mx::contiguous(mx::reshape(mx::astype(words, mx::uint32), {-1}));
    const int groups = rp / (8 * NT);
    auto packed_scales = mx::contiguous(mx::reshape(mx::transpose(
        mx::reshape(scales, {groups, NT, 8, k / 128}), {0, 3, 1, 2}), {-1}));
    mx::eval(words, packed_scales, lut);
    return {words, packed_scales, lut, rows, rp, k};
}
}

Mma8Weights mma8_from_parts(mx::array words, mx::array scales, int rows, int columns, int bits) {
    if ((bits != 3 && bits != 4) || rows <= 0 || columns <= 0 || columns % 128)
        throw std::invalid_argument("invalid stored MMA8 pack");
    const int rp = (rows + 63) / 64 * 64;
    if (words.dtype() != mx::uint32 || words.size() != size_t(rp) * columns / 8 ||
        scales.dtype() != mx::float16 || scales.size() != size_t(rp) * (columns / 128))
        throw std::invalid_argument("stored MMA8 pack has the wrong sizes");
    static const mx::array lut4 = level_table(L4, 7, 8), lut3 = level_table(L3, 3, 4);
    return {words, scales, bits == 4 ? lut4 : lut3, rows, rp, columns};
}

Mma8Weights mma8_pack(const Tq4Weights& w) {
    if (!w.integer_levels || w.codes.ndim() != 2 || w.codes.dtype() != mx::uint8 || w.scales.dtype() != mx::float16 ||
        w.codes.shape(1) % 64 || w.scales.shape() != mx::Shape{w.codes.shape(0), w.codes.shape(1) / 64})
        throw std::invalid_argument("MMA8 packing requires integer-level TQ4 group128 weights");
    const int rows = w.codes.shape(0), k = w.codes.shape(1) * 2, rp = (rows + 63) / 64 * 64;
    auto codes = w.codes, scales = w.scales;
    if (rp != rows) {
        codes = mx::concatenate({codes, mx::zeros({rp - rows, k / 2}, mx::uint8)}, 0);
        scales = mx::concatenate({scales, mx::zeros({rp - rows, k / 128}, mx::float16)}, 0);
    }
    auto low = mx::bitwise_and(codes, mx::array(uint8_t(15))), high = mx::right_shift(codes, mx::array(uint8_t(4)));
    auto nibbles = mx::astype(mx::reshape(mx::stack({low, high}, -1), {rp * k}), mx::uint32);
    static const mx::array lut = level_table(L4, 7, 8);
    return finish_pack(nibbles, scales, lut, rows, rp, k);
}

Mma8Weights mma8_pack(const Tq3Weights& w) {
    if (w.interleaved || w.planes.ndim() != 3 || w.planes.dtype() != mx::uint32 || w.planes.shape(1) != 3 ||
        w.scales.dtype() != mx::float16 || w.planes.shape(2) % 4 ||
        w.scales.shape() != mx::Shape{w.planes.shape(0), w.planes.shape(2) / 4})
        throw std::invalid_argument("MMA8 packing requires planar TQ3 weights");
    const int rows = w.planes.shape(0), k = w.planes.shape(2) * 32, rp = (rows + 63) / 64 * 64;
    auto planes = w.planes, scales = w.scales;
    if (rp != rows) {
        planes = mx::concatenate({planes, mx::zeros({rp - rows, 3, k / 32}, mx::uint32)}, 0);
        scales = mx::concatenate({scales, mx::zeros({rp - rows, k / 128}, mx::float16)}, 0);
    }
    auto plane = [&](int p) {return mx::reshape(mx::slice(planes, {0, p, 0}, {rp, p + 1, k / 32}), {rp, k / 32});};
    auto low = plane(0), high = plane(1), sign = plane(2);
    std::vector<mx::array> columns;
    columns.reserve(32);
    for (uint32_t j = 0; j < 32; ++j) {
        auto ranks = j < 16 ? low : high;
        const uint32_t rank_shift = 4 * (j % 8) + 2 * ((j % 16) / 8), sign_shift = 4 * (j % 8) + j / 8;
        auto rank = mx::bitwise_and(mx::right_shift(ranks, mx::array(rank_shift, mx::uint32)), mx::array(3u, mx::uint32));
        auto negative = mx::bitwise_and(mx::right_shift(sign, mx::array(sign_shift, mx::uint32)), mx::array(1u, mx::uint32));
        columns.push_back(mx::bitwise_or(rank, mx::left_shift(negative, mx::array(2u, mx::uint32))));
    }
    auto nibbles = mx::reshape(mx::stack(columns, -1), {rp * k});
    static const mx::array lut = level_table(L3, 3, 4);
    return finish_pack(nibbles, scales, lut, rows, rp, k);
}

mx::array mma8_input(const mx::array& x) {
    if (x.ndim() != 2 || x.dtype() != mx::float32 || x.shape(0) < 1 || x.shape(0) > 8)
        throw std::invalid_argument("MMA8 requires a 1..8 row FP32 rotated input");
    auto half_x = mx::astype(x, mx::float16);
    if (x.shape(0) < 8) half_x = mx::concatenate({half_x, mx::zeros({8 - x.shape(0), x.shape(1)}, mx::float16)}, 0);
    return half_x;
}

mx::array mma8_matmul(const Mma8Weights& w, const mx::array& half_x, int m, bool round_bf16) {
    if (half_x.ndim() != 2 || half_x.dtype() != mx::float16 || half_x.shape(0) != 8 || half_x.shape(1) != w.columns ||
        m < 1 || m > 8)
        throw std::invalid_argument("MMA8 requires a padded [8, K] FP16 input");
    const int rp = w.rows_padded, k = w.columns;
    static auto kernel = mx::fast::metal_kernel("tqllm_mma8", {"W", "S", "X", "lut"}, {"out"}, source, header);
    auto y = kernel({w.words, w.scales, half_x, w.lut}, {{8, rp}}, {mx::float32},
                    {rp / (8 * NT) * 32, 1, 1}, {128, 1, 1}, {{"K", k}, {"N", rp}, {"BF", round_bf16}},
                    std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return (m == 8 && rp == w.rows) ? y : mx::slice(y, {0, 0}, {m, w.rows});
}

mx::array mma8_matmul(const Mma8Weights& w, const mx::array& x) {
    return mma8_matmul(w, mma8_input(x), x.shape(0), false);
}
}

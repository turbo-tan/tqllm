#include "nax_affine.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mx = mlx::core;
namespace tqllm::mlx_backend {
namespace {
void check_shape(uint64_t rows, uint64_t columns, int bits, int group) {
    if ((bits != 4 && bits != 6) || (group != 64 && group != 128) || rows == 0 || rows % 128 || columns % group)
        throw std::invalid_argument("nax affine needs 4/6 bits, group 64/128, rows % 128 == 0, columns % group == 0");
}
// Lane layout of one 16x16 fragment (MLX steel NAX): element i of lane l sits at row fm + (i/4)*8, column fn + i%4.
inline int lane_fm(int lane) {return ((lane >> 2) & 4) | ((lane >> 1) & 3);}
inline int lane_fn(int lane) {return ((((lane >> 2) & 2) | (lane & 1)) * 4);}

const char* kHeader = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline float4 bf16x4(ushort4 v) {return float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16),
                                               as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));}
)METAL";

const char* kSource = R"METAL(
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.x * 4 + sg;
    const int split = int(threadgroup_position_in_grid.z), KS = K / SPLIT;
    const int row0 = int(threadgroup_position_in_grid.y) * MR;
    const int rows = min(int(R[0]) - row0, MR);
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ca2 = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto cc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < 16; i++) { cc[i] = 0.0f; cc2[i] = 0.0f; }
    const device uint2* wp = (const device uint2*)W + ((size_t)blk * (K / 16) + split * (KS / 16)) * 32 + lane;
    const device uint* hp = H + (BITS == 6 ? ((size_t)blk * (K / 16) + split * (KS / 16)) * 32 + lane : 0);
    const device ushort* sp = S + ((size_t)blk * (K / GS) + split * (KS / GS)) * 32;
    const device ushort* bp = Bz + ((size_t)blk * (K / GS) + split * (KS / GS)) * 32;
    const device half* xs = X + (size_t)(row0 + fm) * K + fn + split * KS;
    for (int g = 0; g < KS / GS; g++) {
        const float4 s0 = bf16x4(*(const device ushort4*)(sp + g * 32 + fn)), s1 = bf16x4(*(const device ushort4*)(sp + g * 32 + 16 + fn));
        const float4 b0 = bf16x4(*(const device ushort4*)(bp + g * 32 + fn)), b1 = bf16x4(*(const device ushort4*)(bp + g * 32 + 16 + fn));
        for (short t = 0; t < GS / 16; t++) {
            const int ks = g * (GS / 16) + t;
            half4 a0 = fm < rows ? *(const device half4*)(xs + ks * 16) : half4(0.0h);
            half4 a1 = fm + 8 < rows ? *(const device half4*)(xs + 8 * K + ks * 16) : half4(0.0h);
            for (short i = 0; i < 4; i++) { ca[i] = a0[i]; ca[4 + i] = a1[i]; }
            uint2 w = wp[ks * 32];
            uint h = BITS == 6 ? hp[ks * 32] : 0u;
            for (short p = 0; p < 4; p++) {
                uint l0 = (w.x >> (4 * p)) & 0x000F000Fu, l1 = (w.y >> (4 * p)) & 0x000F000Fu;
                if (BITS == 6) {
                    l0 |= ((h >> (2 * p)) & 0x00030003u) << 4;
                    l1 |= ((h >> (8 + 2 * p)) & 0x00030003u) << 4;
                }
                const float2 q0 = float2(as_type<half2>(l0 | 0x64006400u)) - 1024.0f;
                const float2 q1 = float2(as_type<half2>(l1 | 0x64006400u)) - 1024.0f;
                const half2 v0 = half2(fma(q0, float2(s0[p]), float2(b0[p])));
                const half2 v1 = half2(fma(q1, float2(s1[p]), float2(b1[p])));
                cb[p] = v0.x; cb[p + 4] = v0.y; cb[8 + p] = v1.x; cb[12 + p] = v1.y;
            }
            op.run(ca, cb, cc);
            if (MR > 16) {
                half4 a2 = fm + 16 < rows ? *(const device half4*)(xs + 16 * K + ks * 16) : half4(0.0h);
                half4 a3 = fm + 24 < rows ? *(const device half4*)(xs + 24 * K + ks * 16) : half4(0.0h);
                for (short i = 0; i < 4; i++) { ca2[i] = a2[i]; ca2[4 + i] = a3[i]; }
                op.run(ca2, cb, cc2);
            }
        }
    }
    const uint col0 = blk * 32 + fn;
    device float* out = Y + (size_t)split * int(R[0]) * N;
    for (short i = 0; i < 16; i++) {
        const short r = fm + ((i & 7) >> 2) * 8;
        const uint c = col0 + (i >> 3) * 16 + (i & 3);
        if (r < rows) out[(size_t)(row0 + r) * N + c] = cc[i];
        if (MR > 16 && r + 16 < rows) out[(size_t)(row0 + r + 16) * N + c] = cc2[i];
    }
)METAL";
}

uint64_t nax_affine_bytes(uint64_t rows, uint64_t columns, int bits, int group) {
    check_shape(rows, columns, bits, group);
    return rows * columns * bits / 8 + rows * (columns / group) * 4;
}

std::vector<uint8_t> nax_affine_pack(std::span<const uint8_t> codes, std::span<const uint16_t> scales,
                                     std::span<const uint16_t> biases, int rows, int columns, int bits, int group) {
    check_shape(rows, columns, bits, group);
    const size_t n = size_t(rows) * columns, groups = size_t(rows) * (columns / group);
    if (codes.size() != n || scales.size() != groups || biases.size() != groups)
        throw std::invalid_argument("nax affine pack: wrong input sizes");
    const int blocks = rows / 32, steps = columns / 16, gpr = columns / group;
    std::vector<uint32_t> words(size_t(blocks) * steps * 32 * 2, 0), high(bits == 6 ? size_t(blocks) * steps * 32 : 0, 0);
    std::vector<uint16_t> s(groups), b(groups);
    for (int blk = 0; blk < blocks; ++blk)
        for (int ks = 0; ks < steps; ++ks)
            for (int lane = 0; lane < 32; ++lane)
                for (int w = 0; w < 2; ++w)
                    for (int i = 0; i < 8; ++i) {
                        const int r = blk * 32 + w * 16 + lane_fn(lane) + (i & 3);
                        const int c = ks * 16 + lane_fm(lane) + (i >> 2) * 8;
                        const uint32_t q = codes[size_t(r) * columns + c];
                        if (q >> bits) throw std::invalid_argument("nax affine pack: code out of range");
                        const size_t at = (size_t(blk) * steps + ks) * 32 + lane;
                        words[at * 2 + w] |= (q & 15u) << (4 * i);
                        if (bits == 6) high[at] |= (q >> 4) << (w * 8 + 2 * (i & 3) + 16 * (i >> 2));
                    }
    for (int blk = 0; blk < blocks; ++blk)
        for (int g = 0; g < gpr; ++g)
            for (int j = 0; j < 32; ++j) {
                const size_t from = size_t(blk * 32 + j) * gpr + g, to = (size_t(blk) * gpr + g) * 32 + j;
                s[to] = scales[from];
                b[to] = biases[from];
            }
    std::vector<uint8_t> out(nax_affine_bytes(rows, columns, bits, group));
    uint8_t* at = out.data();
    auto put = [&](const void* p, size_t bytes) {std::memcpy(at, p, bytes); at += bytes;};
    put(words.data(), words.size() * 4);
    if (bits == 6) put(high.data(), high.size() * 4);
    put(s.data(), s.size() * 2);
    put(b.data(), b.size() * 2);
    return out;
}

NaxAffine nax_affine_load(std::span<const uint8_t> payload, int rows, int columns, int bits, int group) {
    if (payload.size() != nax_affine_bytes(rows, columns, bits, group))
        throw std::invalid_argument("nax affine payload has the wrong size");
    const int blocks = rows / 32, steps = columns / 16, gpr = columns / group;
    const uint8_t* at = payload.data();
    auto take = [&](size_t bytes) {const uint8_t* p = at; at += bytes; return p;};
    auto words = mx::array(reinterpret_cast<const uint32_t*>(take(size_t(blocks) * steps * 32 * 8)),
                           {blocks, steps, 32, 2}, mx::uint32);
    auto high = bits == 6 ? mx::array(reinterpret_cast<const uint32_t*>(take(size_t(blocks) * steps * 32 * 4)),
                                      {blocks, steps, 32}, mx::uint32)
                          : mx::zeros({64}, mx::uint32);
    const size_t sb = size_t(blocks) * gpr * 32 * 2;
    auto scales = mx::array(reinterpret_cast<const uint16_t*>(take(sb)), {blocks, gpr, 32}, mx::uint16);
    auto biases = mx::array(reinterpret_cast<const uint16_t*>(take(sb)), {blocks, gpr, 32}, mx::uint16);
    mx::eval(words, high, scales, biases);
    return {words, high, scales, biases, rows, columns, bits, group};
}

// K splits for one matrix: enough simdgroups to keep the memory system busy when each matmul waits for the previous
// one (N/32 simdgroups per split), each split a whole number of groups. Depends only on the shape, never on the row
// count, so a single row and a verify batch sum the same partial products in the same order.
int nax_affine_splits(const NaxAffine& w) {
    int split = 1;
    while (split < 8 && (w.rows / 32) * split < 512 && (w.columns / w.group) % (split * 2) == 0) split *= 2;
    return split;
}

mx::array nax_affine_matmul(const NaxAffine& w, const mx::array& input) {
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax affine matmul: input must be [m, columns]");
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_affine", {"W", "H", "S", "Bz", "X", "R"}, {"Y"},
                                                kSource, kHeader);
    const int m = input.shape(0), mr = m <= 16 ? 16 : 32, split = nax_affine_splits(w);
    auto x = mx::contiguous(mx::astype(input, mx::float16));
    const uint32_t m32 = uint32_t(m);
    auto count = mx::array(&m32, {1}, mx::uint32);
    auto y = kernel({w.words, w.high, w.scales, w.biases, x, count}, {{split, m, w.rows}}, {mx::float32},
                    {w.rows, (m + mr - 1) / mr, split}, {128, 1, 1},
                    {{"K", w.columns}, {"N", w.rows}, {"MR", mr}, {"BITS", w.bits}, {"GS", w.group}, {"SPLIT", split}},
                    std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return split == 1 ? mx::reshape(y, {m, w.rows}) : mx::sum(y, 0);
}

uint64_t affine8_embedding_bytes(uint64_t rows, uint64_t columns) {
    if (rows == 0 || columns % 64) throw std::invalid_argument("affine8 embedding needs columns % 64 == 0");
    return rows * columns + rows * (columns / 64) * 4;
}

Affine8Embedding affine8_embedding_load(std::span<const uint8_t> payload, int rows, int columns) {
    if (payload.size() != affine8_embedding_bytes(rows, columns))
        throw std::invalid_argument("affine8 embedding payload has the wrong size");
    const uint8_t* at = payload.data();
    auto codes = mx::array(at, {rows, columns}, mx::uint8);
    at += size_t(rows) * columns;
    const int g = columns / 64;
    auto scales = mx::array(reinterpret_cast<const mx::bfloat16_t*>(at), {rows, g}, mx::bfloat16);
    at += size_t(rows) * g * 2;
    auto biases = mx::array(reinterpret_cast<const mx::bfloat16_t*>(at), {rows, g}, mx::bfloat16);
    mx::eval(codes, scales, biases);
    return {codes, scales, biases, rows, columns};
}

mx::array affine8_embedding(const Affine8Embedding& e, std::span<const uint32_t> tokens) {
    const int m = int(tokens.size());
    for (auto t : tokens) if (t >= uint32_t(e.rows)) throw std::invalid_argument("token id out of range");
    auto ids = mx::array(tokens.data(), {m}, mx::uint32);
    auto q = mx::reshape(mx::astype(mx::take(e.codes, ids, 0), mx::float32), {m, e.columns / 64, 64});
    auto s = mx::expand_dims(mx::astype(mx::take(e.scales, ids, 0), mx::float32), 2);
    auto b = mx::expand_dims(mx::astype(mx::take(e.biases, ids, 0), mx::float32), 2);
    return mx::reshape(mx::add(mx::multiply(q, s), b), {m, e.columns});
}
}

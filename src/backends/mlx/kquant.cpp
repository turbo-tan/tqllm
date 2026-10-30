#include "kquant.hpp"
#include "gguf.hpp"
#include <stdexcept>
#include <string>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
const uint32_t iq3s_grid[512] = {
#include "../../formats/weight/iq3s_grid.inc"
};
const int8_t kvalues_iq4nl[16] = {
#include "../../formats/weight/kvalues_iq4nl.inc"
};
struct Layout {int elements, bytes;};
Layout layout(uint32_t type) {
    switch (type) {
        case GGML_Q2_K: return {256, 84};
        case GGML_Q3_K: return {256, 110};
        case GGML_Q4_K: return {256, 144};
        case GGML_Q5_K: return {256, 176};
        case GGML_IQ3_S: return {256, 110};
        case GGML_IQ4_XS: return {256, 136};
        case GGML_IQ4_NL: return {32, 18};
        default: return {0, 0};
    }
}
const mx::array& grid() {
    static const mx::array g = [] {auto a = mx::array(iq3s_grid, {512}, mx::uint32); mx::eval(a); return a;}();
    return g;
}
const mx::array& levels() {
    static const mx::array v = [] {auto a = mx::array(kvalues_iq4nl, {16}, mx::int8); mx::eval(a); return a;}();
    return v;
}
// element e of one block, in the operation order of dequantize_host (no contraction, so the values match bitwise)
const char* kHeader = R"METAL(
inline float kq_half(const device uchar* p) {return float(as_type<half>(ushort(uint(p[0]) | (uint(p[1]) << 8))));}
inline void kq_scale_min(uint j, const device uchar* q, thread uint& d, thread uint& m) {
    if (j < 4u) {d = q[j] & 63u; m = q[j + 4u] & 63u;}
    else {d = (q[j + 4u] & 15u) | ((q[j - 4u] >> 6) << 4); m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);}
}
template <int TYPE>
inline float kq_value(const device uchar* p, uint e, const device uint32_t* grid, const device int8_t* kv) {
#pragma clang fp contract(off)
    if (TYPE == 12) {  // Q4_K
        const float d = kq_half(p), mn = kq_half(p + 2);
        const uint sub = e / 32u; uint s, m; kq_scale_min(sub, p + 4, s, m);
        const uint q = (p[16u + (sub / 2u) * 32u + e % 32u] >> (4u * (sub % 2u))) & 15u;
        const float d1 = d * float(s), m1 = mn * float(m);
        return d1 * float(q) - m1;
    } else if (TYPE == 13) {  // Q5_K
        const float d = kq_half(p), mn = kq_half(p + 2);
        const uint j = e / 32u, l = e % 32u; uint s, m; kq_scale_min(j, p + 4, s, m);
        const uint b = p[48u + 32u * (j / 2u) + l];
        const int v = int(j % 2u ? b >> 4 : b & 15u) + int((p[16u + l] >> j) & 1u) * 16;
        return d * float(s) * float(v) - mn * float(m);
    } else if (TYPE == 11) {  // Q3_K
        const uint n = e / 128u, j = (e % 128u) / 32u, w = e % 32u;
        const uint i = 8u * n + 2u * j + (w >= 16u ? 1u : 0u), grp = i / 4u, r = i % 4u;
        const device uchar* s = p + 96;
        const uint low = grp < 2u ? s[grp * 4u + r] & 15u : s[(grp - 2u) * 4u + r] >> 4;
        const int sc = int(low | (((s[8u + r] >> (2u * grp)) & 3u) << 4)) - 32;
        const int q = int((p[32u + 32u * n + w] >> (2u * j)) & 3u) - (((p[w] >> (4u * n + j)) & 1u) ? 0 : 4);
        return kq_half(p + 108) * float(sc) * float(q);
    } else if (TYPE == 10) {  // Q2_K
        const float d = kq_half(p + 80), mn = kq_half(p + 82);
        const uint n = e / 128u, jj = (e % 128u) / 32u, h = (e % 32u) / 16u, l = e % 16u;
        const uint sc = p[n * 8u + jj * 2u + h];
        const float dl = d * float(sc & 15u), ml = mn * float(sc >> 4);
        return dl * float((p[16u + n * 32u + l + 16u * h] >> (2u * jj)) & 3u) - ml;
    } else if (TYPE == 21) {  // IQ3_S
        const float d = kq_half(p);
        const uint ib = e / 32u, w = (e % 32u) / 4u, j = e % 4u;
        const float db = d * float(1u + 2u * ((p[106u + ib / 2u] >> (4u * (ib % 2u))) & 15u));
        const uint g = grid[p[2u + ib * 8u + w] | (((p[66u + ib] >> w) & 1u) << 8)];
        const uint sg = p[74u + ib * 4u + w / 2u] >> (4u * (w % 2u));
        return db * float((g >> (8u * j)) & 255u) * (((sg >> j) & 1u) ? -1.0f : 1.0f);
    } else if (TYPE == 23) {  // IQ4_XS
        const float d = kq_half(p);
        const uint sh = uint(p[2]) | (uint(p[3]) << 8), ib = e / 32u, r = e % 32u;
        const int ls = int(((p[4u + ib / 2u] >> (4u * (ib % 2u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4));
        const float dl = d * float(ls - 32);
        const uint b = p[8u + ib * 16u + (r % 16u)];
        return dl * float(kv[r < 16u ? b & 15u : b >> 4]);
    } else {  // IQ4_NL, 32-element blocks
        const uint r = e % 32u, b = p[2u + (r % 16u)];
        return kq_half(p) * float(kv[r < 16u ? b & 15u : b >> 4]);
    }
}
)METAL";
}  // namespace

bool kquant_supported(uint32_t type) {return layout(type).elements > 0;}

KQuantWeights load_kquant_blocks(std::span<const uint8_t> blocks, uint32_t type, int rows, int columns) {
    const auto [elements, bytes] = layout(type);
    if (!elements || rows <= 0 || columns <= 0 || columns % elements ||
        blocks.size() != uint64_t(rows) * uint64_t(columns / elements) * uint64_t(bytes))
        throw std::invalid_argument("invalid K-quant / IQ block payload or dimensions (type " + std::to_string(type) + ")");
    return {mx::array(blocks.data(), {rows, columns / elements, bytes}, mx::uint8), type};
}

mx::array kquant_dequant(const KQuantWeights& w) {
    const auto [elements, bytes] = layout(w.type);
    if (!elements || w.blocks.ndim() != 3 || w.blocks.shape(2) != bytes) throw std::invalid_argument("invalid K-quant weights");
    const int rows = w.blocks.shape(0), k = w.blocks.shape(1) * elements;
    static auto kernel = mx::fast::metal_kernel("tqllm_kquant_dequant", {"blocks", "grid", "kv"}, {"y"}, R"METAL(
const uint i = thread_position_in_grid.x;
if (i >= uint(R) * uint(K)) return;
const uint row = i / uint(K), col = i % uint(K);
const device uchar* p = blocks + ((size_t)row * (uint(K) / uint(BE)) + col / uint(BE)) * uint(BB);
y[i] = kq_value<TYPE>(p, col % uint(BE), grid, kv);
)METAL", kHeader);
    return kernel({w.blocks, grid(), levels()}, {{rows, k}}, {mx::float32}, {rows * k, 1, 1}, {256, 1, 1},
                  {{"R", rows}, {"K", k}, {"BE", elements}, {"BB", bytes}, {"TYPE", int(w.type)}}, std::nullopt, false,
                  mx::Device(mx::Device::gpu))[0];
}

mx::array kquant_matmul(const KQuantWeights& w, const mx::array& x) {
    const auto [elements, bytes] = layout(w.type);
    if (!elements || w.blocks.ndim() != 3 || w.blocks.shape(2) != bytes || x.ndim() != 2 || x.dtype() != mx::float32)
        throw std::invalid_argument("invalid K-quant matrix types");
    const int rows = w.blocks.shape(0), k = w.blocks.shape(1) * elements, m = x.shape(0);
    if (m <= 0 || x.shape(1) != k) throw std::invalid_argument("incompatible K-quant matrix dimensions");
    if (m > 16) return mx::matmul(x, mx::transpose(kquant_dequant(w)));
    static auto kernel = mx::fast::metal_kernel("tqllm_kquant_matmul", {"blocks", "x", "grid", "kv"}, {"y"}, R"METAL(
const uint lane = thread_position_in_threadgroup.x;
const uint row = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;
const uint col = threadgroup_position_in_grid.z * uint(MCOLS);
if (row >= uint(R)) return;
float acc[MCOLS];
for (uint c = 0; c < uint(MCOLS); ++c) acc[c] = 0.0f;
for (uint b = lane; b < uint(K) / uint(BE); b += 32u) {
    const device uchar* p = blocks + ((size_t)row * (uint(K) / uint(BE)) + b) * uint(BB);
    for (uint e = 0; e < uint(BE); ++e) {
        const float v = kq_value<TYPE>(p, e, grid, kv);
        for (uint c = 0; c < uint(MCOLS); ++c) {
            if (col + c >= uint(M)) break;
            acc[c] += v * x[(size_t)(col + c) * uint(K) + b * uint(BE) + e];
        }
    }
}
for (uint c = 0; c < uint(MCOLS); ++c) {
    if (col + c >= uint(M)) break;
    const float s = simd_sum(acc[c]);
    if (lane == 0) y[(size_t)(col + c) * uint(R) + row] = s;
}
)METAL", kHeader);
    const int cols = m <= 1 ? 1 : m <= 2 ? 2 : 4;
    return kernel({w.blocks, mx::contiguous(x), grid(), levels()}, {{m, rows}}, {mx::float32},
                  {32, (rows + 1) / 2 * 2, (m + cols - 1) / cols}, {32, 2, 1},
                  {{"R", rows}, {"K", k}, {"M", m}, {"MCOLS", cols}, {"BE", elements}, {"BB", bytes}, {"TYPE", int(w.type)}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}

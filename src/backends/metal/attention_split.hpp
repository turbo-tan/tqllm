#pragma once
// Tree-verify attention for 1-16 query rows over the BF16 KV cache (Qwen3.8-27B full-attention shape: 4 KV heads,
// 6 query heads per KV head, head dim 256), split over the keys: one threadgroup per (KV head, key chunk, pair of query
// heads), one simdgroup per query head (its 16 token rows, all 256 output columns), scores and P x V on the matrix unit
// with the probabilities kept in registers, an online softmax per row; a merge kernel folds the chunks in key order.
// Many more threadgroups than one call per 1024-key segment, so the K / V stream fills the GPU.
// Block keys (the last n of `length`) are visible to row t when bit j of vis[t] is set; keys before the block to every
// row. K / V must be the cache with capacity a multiple of 64 past `length` holding finite values (zero or stale rows).
#include <mlx/mlx.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {

// the split kernel only above 4K keys: the short prompts keep the folded / segmented path and their texts
// (j1783: doc2k 93.1 -> 81.8 tok/s on a changed text with the split kernel; j1784: 8K 49.4 split vs 46.7 without)
constexpr int kSplitAttentionMin = 4096;
// keys per chunk: about 24 chunks per KV head (96 threadgroups), at least 256 keys, a multiple of 64
inline int split_attention_chunk(int length) {
    return length <= 12288 ? 1024 : std::max(256, ((length + 23) / 24 + 63) / 64 * 64);  // j1764: 1024 at 8K, ~24 chunks at 16K
}
inline bool split_attention_fits(const mlx::core::array& q, const mlx::core::array& k) {
    return q.ndim() == 4 && k.ndim() == 4 && q.shape(2) <= 16 && q.shape(3) == 256 && q.shape(1) == 6 * k.shape(1) &&
           k.dtype() == mlx::core::bfloat16 && k.shape(2) % 64 == 0;
}

// bits [16] int32: bit j of row t = block key j visible to row t (rows past n: 0)
inline mlx::core::array split_attention_bits(const mlx::core::array* visible, int n) {
    namespace mx = mlx::core;
    if (!visible) {
        std::vector<int32_t> b(16, 0);
        for (int t = 0; t < n; ++t) b[size_t(t)] = int32_t((2u << t) - 1u);
        return mx::array(b.data(), {16}, mx::int32);
    }
    auto w = mx::left_shift(mx::array(1, mx::int32), mx::arange(visible->shape(1), mx::int32));
    auto bits = mx::sum(mx::multiply(mx::astype(*visible, mx::int32), mx::reshape(w, {1, visible->shape(1)})), 1);
    if (n < 16) bits = mx::concatenate({bits, mx::zeros({16 - n}, mx::int32)}, 0);
    return mx::astype(bits, mx::int32);
}

inline mlx::core::array decode_attention_split(const mlx::core::array& q, const mlx::core::array& k,
                                               const mlx::core::array& v, float scale, int length,
                                               const mlx::core::array& bits) {
    namespace mx = mlx::core;
    const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), kvh = k.shape(1), cap = k.shape(2);
    if (q.ndim() != 4 || q.shape(0) != 1 || n < 1 || n > 16 || dim != 256 || kvh * 6 != heads || k.dtype() != mx::bfloat16 ||
        v.shape() != k.shape() || v.dtype() != k.dtype() || length < n || length > cap || cap % 64 || bits.shape() != mx::Shape{16})
        throw std::invalid_argument("split attention: 1-16 rows, 6 query heads per KV head, head dim 256, BF16 cache");
    const int ch = split_attention_chunk(length), nch = (length + ch - 1) / ch;
    static const std::string header = R"METAL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    // one simdgroup per query head over its chunk: two 128-column accumulators, the 16 x 64 score tile turned into the
    // P x V operand in registers (no threadgroup memory, no barriers)
    static auto attend = mx::fast::metal_kernel("tqllm_attn_split", {"QB", "K", "V", "BITS", "P", "SC"}, {"PO", "PM", "PL"}, R"METAL(
    using namespace mpp::tensor_ops;
    const ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const int c = int(threadgroup_position_in_grid.y), kh = int(threadgroup_position_in_grid.z) / (6 / HPT);
    const int L = P[0], PRE = P[1], CH = P[2], NCH = P[3], CAP = P[4];
    const int head = int(threadgroup_position_in_grid.z) % (6 / HPT) * HPT + sg;
    const short fm = ((lane >> 2) & 4) | ((lane >> 1) & 3), fn = ((((lane >> 2) & 2) | (lane & 1)) * 4);
    const uint b0 = uint(BITS[fm]), b1 = uint(BITS[fm + 8]);
    const float sc = SC;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)QB + (size_t(kh) * 96 + head * 16) * 256,
                                                                 dextents<int32_t, 2>(256, 16));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tK((device bfloat*)K + size_t(kh) * CAP * 256, dextents<int32_t, 2>(256, CAP));
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tV((device bfloat*)V + size_t(kh) * CAP * 256, dextents<int32_t, 2>(256, CAP));
    threadgroup half pbuf[16 * 64];
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(pbuf, dextents<int32_t, 2>(64, 16));
    constexpr auto dS = matmul2d_descriptor(16, 64, 256, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto dO = matmul2d_descriptor(16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroup> opS;
    matmul2d<dO, execution_simdgroup> opO;
    auto v0 = tV.slice(0, 0);
    auto Oa = opO.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tP)>, metal::remove_addrspace_t<decltype(v0)>, float>();
    auto Ob = opO.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tP)>, metal::remove_addrspace_t<decltype(v0)>, float>();
    for (int i = 0; i < 64; i++) {Oa[i] = 0.0f; Ob[i] = 0.0f;}
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int kbeg = c * CH, kend = min(kbeg + CH, L);
    for (int kt = kbeg; kt < kend; kt += 64) {
        auto kslice = tK.slice(0, kt);
        auto S = opS.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tQ)>, metal::remove_addrspace_t<decltype(kslice)>, float>();
        opS.run(tQ, kslice, S);
        float s[32];
        for (int i = 0; i < 32; i++) {
            const int key = kt + (i >> 3) * 16 + fn + (i & 3);
            const uint b = (i & 4) ? b1 : b0;
            const bool see = b != 0u && key < L && (key < PRE || ((b >> uint(key - PRE)) & 1u) != 0u);
            s[i] = see ? S[i] * sc : -INFINITY;
        }
        float x0 = -INFINITY, x1 = -INFINITY;
        for (int i = 0; i < 32; i++) {if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]);}
        x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
        x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
        const float n0 = max(m0, x0), n1 = max(m1, x1);
        const float f0 = x0 == -INFINITY ? 1.0f : fast::exp(m0 - n0), f1 = x1 == -INFINITY ? 1.0f : fast::exp(m1 - n1);
        auto Pc = opS.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(tQ)>, metal::remove_addrspace_t<decltype(kslice)>, half>();
        float y0 = 0.0f, y1 = 0.0f;
        for (int i = 0; i < 32; i++) {
            const float p = s[i] == -INFINITY ? 0.0f : fast::exp(s[i] - ((i & 4) ? n1 : n0));
            Pc[i] = half(p);
            if (i & 4) y1 += p; else y0 += p;
        }
        y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
        y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
        if (x0 != -INFINITY) {l0 = l0 * f0 + y0; m0 = n0;}
        if (x1 != -INFINITY) {l1 = l1 * f1 + y1; m1 = n1;}
        for (int i = 0; i < 64; i++) {const float f = (i & 4) ? f1 : f0; Oa[i] *= f; Ob[i] *= f;}
        auto Pin = opO.template get_left_input_cooperative_tensor<half, bfloat, float>(Pc);
        auto va = tV.slice(0, kt), vb = tV.slice(128, kt);
        opO.run(Pin, va, Oa);
        opO.run(Pin, vb, Ob);
    }
    const size_t base = (size_t(kh) * NCH + c) * 96 + head * 16;
    for (int g = 0; g < 16; g++) {
        const int r = fm + (g & 1) * 8, col = (g >> 1) * 16 + fn;
        *(device float4*)(PO + (base + r) * 256 + col) = float4(Oa[4 * g], Oa[4 * g + 1], Oa[4 * g + 2], Oa[4 * g + 3]);
        *(device float4*)(PO + (base + r) * 256 + 128 + col) = float4(Ob[4 * g], Ob[4 * g + 1], Ob[4 * g + 2], Ob[4 * g + 3]);
    }
    if ((lane & 9) == 0) {PM[base + fm] = m0; PL[base + fm] = l0; PM[base + fm + 8] = m1; PL[base + fm + 8] = l1;}
)METAL", header);
    static auto merge = mx::fast::metal_kernel("tqllm_attn_split_merge", {"PO", "PM", "PL", "P"}, {"Y"}, R"METAL(
    // one thread per (column, folded row, KV head): fold the chunks in key order; rows that saw no key give 0
    const uint e = thread_position_in_grid.x, r = thread_position_in_grid.y, kh = thread_position_in_grid.z;
    const int NCH = P[3], N = P[5], t = int(r) & 15, h = int(r) >> 4;
    if (t >= N) return;
    float m = -INFINITY, den = 0.0f, acc = 0.0f;
    for (int c = 0; c < NCH; ++c) {
        const size_t i = (size_t(kh) * NCH + c) * 96 + r;
        const float mc = PM[i];
        if (!(mc > -INFINITY)) continue;
        const float nm = max(m, mc), a = m == nm ? 1.0f : exp(m - nm), b = mc == nm ? 1.0f : exp(mc - nm);
        den = den * a + PL[i] * b;
        acc = acc * a + PO[i * 256 + e] * b;
        m = nm;
    }
    Y[((size_t(kh) * 6 + h) * N + t) * 256 + e] = den > 0.0f ? float(bfloat16_t(acc / den)) : 0.0f;
)METAL");
    auto qb = mx::astype(q, mx::bfloat16);
    if (n < 16) qb = mx::concatenate({qb, mx::zeros({1, heads, 16 - n, dim}, mx::bfloat16)}, 2);
    qb = mx::reshape(qb, {kvh, 96, dim});
    std::vector<int32_t> p = {length, length - n, ch, nch, cap, n};
    auto P = mx::array(p.data(), {6}, mx::int32);
    constexpr int hpt = 2;  // query heads per threadgroup (j1762 / j1763)
    auto parts = attend({qb, k, v, bits, P, mx::array(scale)}, {{kvh, nch, 96, dim}, {kvh, nch, 96}, {kvh, nch, 96}},
                        {mx::float32, mx::float32, mx::float32}, {32 * hpt, nch, kvh * (6 / hpt)}, {32 * hpt, 1, 1},
                        {{"HPT", hpt}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return merge({parts[0], parts[1], parts[2], P}, {{1, heads, n, dim}}, {mx::float32}, {256, 96, kvh}, {256, 1, 1}, {},
                 std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// compile the split-attention kernels once at load, so the first verify of a request does not pay the MPP compile
// (j1769: the first prompt's steps averaged 102.5 ms against 86 on the next four)
inline void warm_split_attention_kernels(int heads, int kvheads) {
    namespace mx = mlx::core;
    if (heads != 6 * kvheads) return;
    auto q = mx::astype(mx::random::normal({1, heads, 16, 256}), mx::bfloat16);
    auto k = mx::astype(mx::random::normal({1, kvheads, 1024, 256}), mx::bfloat16);
    mx::eval(decode_attention_split(q, k, k, 0.0625f, 1000, split_attention_bits(nullptr, 16)));
}
}  // namespace tqllm::mlx_backend

#pragma once
// Decode / verify attention for 1-16 query rows over the BF16 KV cache (Qwen3.8-27B full-attention shape: 4 KV heads,
// 6 query heads per KV head, head dim 256), where a row's arithmetic depends only on its query, its own last key x and
// the keys it sees in AR order, so a token-tree row equals its one-token step bit for bit with no key copies:
// - keys below T(x) = 64 * floor((x - 15) / 64) run on the matrix unit in 64-key tiles, split over chunks counted from
//   key 0 and merged in key order, as decode_attention_split; the chunk size exact_chunk(T) depends on T alone (1024
//   keys up to 32K, then doubling so a row has at most 32 chunks);
// - the keys from T(x) to the row (16 to 79 of them) run as a per-row loop in key order: the prefix keys T(x) .. P - 1,
//   then the block rows on the row's path in depth order (for AR and a chain: the next positions).
// A block row t at depth d has x = P + d with d <= 15, so every block key falls in the loop, and T(x) is the same for the
// row and its AR step: the two runs do the same operations on the same values.
#include "attention_decode.hpp"
#include "attention_qsa.hpp"
#include "shape_constants.hpp"
#include <mlx/mlx.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm::mlx_backend {
constexpr int kExactTile = 64, kExactChunk = 1024, kExactTail = 80;

// keys per matrix-unit chunk for a row with bound T; the kernel repeats this rule
inline int exact_chunk(int t) {
    int ch = kExactChunk;
    while (ch * 32 < t) ch *= 2;
    return ch;
}

inline bool exact_attention_fits(const mlx::core::array& q, const mlx::core::array& k) {
    return q.ndim() == 4 && k.ndim() == 4 && q.shape(2) >= 1 && q.shape(2) <= 16 && q.shape(3) == 256 &&
           q.shape(1) == 6 * k.shape(1) && k.dtype() == mlx::core::bfloat16 && k.shape(2) % kExactTile == 0;
}

// Per-row plan [16, 2 + kExactTail] int32: T (keys below run on the matrix unit), the loop's key count, then its key
// positions. prefix: keys before the block; visible: the tree's [n, n] ancestor mask (nullptr: causal rows).
inline mlx::core::array exact_rows_plan(int prefix, int n, const mlx::core::array* visible) {
    namespace mx = mlx::core;
    auto index = mx::arange(n, mx::int32);
    auto vis = visible ? mx::astype(*visible, mx::float32)
                       : mx::astype(mx::less_equal(mx::reshape(index, {1, n}), mx::reshape(index, {n, 1})), mx::float32);
    auto depth = mx::astype(mx::subtract(mx::sum(vis, 1), mx::array(1.0f)), mx::int32);
    // path[t][d]: the block row at depth d on row t's path (0 past its depth)
    auto at_depth = mx::astype(mx::equal(mx::reshape(depth, {n, 1}), mx::reshape(index, {1, n})), mx::float32);
    auto path = mx::astype(mx::matmul(mx::multiply(vis, mx::reshape(mx::astype(index, mx::float32), {1, n})), at_depth), mx::int32);
    auto x = mx::add(depth, mx::array(prefix, mx::int32));
    auto low = mx::subtract(x, mx::array(15, mx::int32));
    auto t = mx::where(mx::greater(low, mx::array(0, mx::int32)),
                       mx::multiply(mx::floor_divide(low, mx::array(kExactTile, mx::int32)), mx::array(kExactTile, mx::int32)),
                       mx::array(0, mx::int32));
    auto pre = mx::subtract(mx::array(prefix, mx::int32), t);          // prefix keys in the loop
    auto count = mx::add(pre, mx::add(depth, mx::array(1, mx::int32)));
    auto j = mx::reshape(mx::arange(kExactTail, mx::int32), {1, kExactTail});
    auto pre2 = mx::reshape(pre, {n, 1});
    auto step = mx::clip(mx::subtract(j, pre2), mx::array(0, mx::int32), mx::array(n - 1, mx::int32));
    auto keys = mx::where(mx::less(j, pre2), mx::add(mx::reshape(t, {n, 1}), j),
                          mx::add(mx::take_along_axis(path, step, 1), mx::array(prefix, mx::int32)));
    keys = mx::where(mx::less(j, mx::reshape(count, {n, 1})), keys, mx::array(0, mx::int32));
    auto plan = mx::concatenate({mx::reshape(t, {n, 1}), mx::reshape(count, {n, 1}), keys}, 1);
    if (n < 16) plan = mx::concatenate({plan, mx::zeros({16 - n, 2 + kExactTail}, mx::int32)}, 0);
    return plan;
}

inline mlx::core::array decode_attention_exact(const mlx::core::array& q, const mlx::core::array& k, const mlx::core::array& v,
                                               float scale, int length, const mlx::core::array* tree = nullptr,
                                               const QsaRows* qsa = nullptr) {
    namespace mx = mlx::core;
    const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), kvh = k.shape(1), cap = k.shape(2);
    if (!exact_attention_fits(q, k) || v.shape() != k.shape() || v.dtype() != k.dtype() || length < n || length > cap)
        throw std::invalid_argument("exact attention: 1-16 rows, 6 query heads per KV head, head dim 256, BF16 cache");
    const int prefix = length - n;
    // the matrix-unit keys of any row end at or below the deepest row's T: chunks up to that bound
    const int deepest = prefix + n - 1 - 15, bound = deepest > 0 ? deepest / kExactTile * kExactTile : 0;
    const int shallow = prefix - 15, low = shallow > 0 ? shallow / kExactTile * kExactTile : 0;
    // rows on both sides of a chunk-size step (once per doubling of the context) take one launch per size
    std::vector<int> sizes = {exact_chunk(low)};
    if (exact_chunk(bound) != sizes[0]) sizes.push_back(exact_chunk(bound));
    // QSA: rows with T >= qsa->from attend their selected blocks instead (attention_qsa.hpp); the dense launches skip them
    const bool sparse = qsa && bound >= qsa->from;
    const int qmin = sparse ? qsa->from : std::numeric_limits<int>::max();
    if (sparse && low >= qsa->from) sizes.clear();
    const auto plan = tree ? cached_tree_mask(*tree, {11, prefix, n, 0, 0, 0, 0}, [&] {return exact_rows_plan(prefix, n, tree);})
                           : shape_constant("exact_rows", {prefix, n}, [&] {return exact_rows_plan(prefix, n, nullptr);});
    static const std::string header = R"METAL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    // matrix-unit part: decode_attention_split's kernel with the per-row key bound T in place of the block bits
    static auto attend = mx::fast::metal_kernel("tqllm_attn_exact", {"QB", "K", "V", "PLAN", "P", "SC"}, {"PO", "PM", "PL"}, R"METAL(
    using namespace mpp::tensor_ops;
    const ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const int c = int(threadgroup_position_in_grid.y), kh = int(threadgroup_position_in_grid.z) / (6 / HPT);
    const int BOUND = P[0], CH = P[1], NCH = P[2], CAP = P[3], QMIN = P[5];
    const int head = int(threadgroup_position_in_grid.z) % (6 / HPT) * HPT + sg;
    const short fm = ((lane >> 2) & 4) | ((lane >> 1) & 3), fn = ((((lane >> 2) & 2) | (lane & 1)) * 4);
    // rows whose own chunk size is not this launch's see no matrix-unit keys here
    const int r0 = PLAN[fm * PW], r1 = PLAN[(fm + 8) * PW];
    int c0 = CH0, c1 = CH0;
    while (c0 * 32 < r0) c0 *= 2;
    while (c1 * 32 < r1) c1 *= 2;
    const int t0 = c0 == CH && r0 < QMIN ? r0 : 0, t1 = c1 == CH && r1 < QMIN ? r1 : 0;
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
    auto Oa = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(v0), float>();
    auto Ob = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(v0), float>();
    for (int i = 0; i < 64; i++) {Oa[i] = 0.0f; Ob[i] = 0.0f;}
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const int kbeg = c * CH, kend = min(kbeg + CH, BOUND);
    for (int kt = kbeg; kt < kend; kt += 64) {
        auto kslice = tK.slice(0, kt);
        auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(kslice), float>();
        opS.run(tQ, kslice, S);
        float s[32];
        for (int i = 0; i < 32; i++) {
            const int key = kt + (i >> 3) * 16 + fn + (i & 3);
            s[i] = key < ((i & 4) ? t1 : t0) ? S[i] * sc : -INFINITY;
        }
        float x0 = -INFINITY, x1 = -INFINITY;
        for (int i = 0; i < 32; i++) {if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]);}
        x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
        x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
        const float n0 = max(m0, x0), n1 = max(m1, x1);
        const float f0 = x0 == -INFINITY ? 1.0f : fast::exp(m0 - n0), f1 = x1 == -INFINITY ? 1.0f : fast::exp(m1 - n1);
        auto Pc = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(kslice), half>();
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
    // loop part, scores: one thread per (key slot, folded row, KV head), the 256-term dot in order
    static auto scores = mx::fast::metal_kernel("tqllm_attn_exact_scores", {"QB", "K", "PLAN", "P", "SC"}, {"TS"}, R"METAL(
    const uint j = thread_position_in_grid.x, r = thread_position_in_grid.y, kh = thread_position_in_grid.z;
    const int t = int(r) & 15, CAP = P[3];
    const size_t o = (size_t(kh) * 96 + r) * TAIL + j;
    if (int(j) >= PLAN[t * PW + 1]) {TS[o] = -INFINITY; return;}
    const device bfloat16_t* qr = QB + (size_t(kh) * 96 + r) * 256;
    const device bfloat16_t* kr = K + (size_t(kh) * CAP + size_t(PLAN[t * PW + 2 + j])) * 256;
    float s = 0.0f;
    for (int d = 0; d < 256; ++d) s = fma(float(qr[d]), float(kr[d]), s);
    TS[o] = s * SC;
)METAL");
    // merge: one thread per (column, folded row, KV head): the matrix-unit chunks in key order, then the loop's keys in order
    static auto merge = mx::fast::metal_kernel("tqllm_attn_exact_merge", {"PO", "PM", "PL", "TS", "V", "PLAN", "P"}, {"Y"}, R"METAL(
    const uint e = thread_position_in_grid.x, r = thread_position_in_grid.y, kh = thread_position_in_grid.z;
    const int NCH = P[2], CAP = P[3], N = P[4], t = int(r) & 15, h = int(r) >> 4;
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
    const int count = PLAN[t * PW + 1];
    const device float* ts = TS + (size_t(kh) * 96 + r) * TAIL;
    for (int j = 0; j < count; ++j) {
        const float s = ts[j], nm = max(m, s), a = m == nm ? 1.0f : exp(m - nm), p = s == nm ? 1.0f : exp(s - nm);
        const float vv = float(V[(size_t(kh) * CAP + size_t(PLAN[t * PW + 2 + j])) * 256 + e]);
        den = den * a + p;
        acc = acc * a + p * vv;
        m = nm;
    }
    Y[((size_t(kh) * 6 + h) * N + t) * 256 + e] = den > 0.0f ? float(bfloat16_t(acc / den)) : 0.0f;
)METAL");
    auto qb = mx::astype(q, mx::bfloat16);
    if (n < 16) qb = mx::concatenate({qb, mx::zeros({1, heads, 16 - n, dim}, mx::bfloat16)}, 2);
    qb = mx::reshape(qb, {kvh, 96, dim});
    const auto SC = mx::array(scale);
    constexpr int hpt = 2;  // query heads per threadgroup, as the split kernel (j1762 / j1763)
    const std::vector<std::pair<std::string, mx::fast::TemplateArg>> consts = {{"PW", 2 + kExactTail}, {"TAIL", kExactTail}};
    // partials of every launch side by side on the chunk axis; a row's chunks outside its own launch hold no keys
    std::vector<mx::array> po, pm, pl;
    int nch = 0;
    for (const int ch : sizes) {
        const int nc = std::max(1, (bound + ch - 1) / ch);
        const std::vector<int32_t> p = {bound, ch, nc, cap, n, qmin};
        auto parts = attend({qb, k, v, plan, mx::array(p.data(), {6}, mx::int32), SC},
                            {{kvh, nc, 96, dim}, {kvh, nc, 96}, {kvh, nc, 96}}, {mx::float32, mx::float32, mx::float32},
                            {32 * hpt, nc, kvh * (6 / hpt)}, {32 * hpt, 1, 1},
                            {{"HPT", hpt}, {"PW", 2 + kExactTail}, {"CH0", kExactChunk}}, std::nullopt, false,
                            mx::Device(mx::Device::gpu));
        po.push_back(parts[0]); pm.push_back(parts[1]); pl.push_back(parts[2]);
        nch += nc;
    }
    if (sparse) {
        auto qq = qsa->q;
        if (n < 16) qq = mx::concatenate({qq, mx::zeros({16 - n, kQsaHeads, kQsaDim}, mx::float32)}, 0);
        const auto sel = qsa_select(qq, qsa->blocks, plan, n, bound / kQsaBlock, 2 + kExactTail, qsa->from);
        auto parts = qsa_partials(qb, k, v, sel, plan, n, 2 + kExactTail, SC, qsa->from);
        po.push_back(parts[0]); pm.push_back(parts[1]); pl.push_back(parts[2]);
        nch += kQsaChunks;
    }
    if (po.size() > 1) po = {mx::concatenate(po, 1)}, pm = {mx::concatenate(pm, 1)}, pl = {mx::concatenate(pl, 1)};
    const std::vector<int32_t> p = {bound, 0, nch, cap, n};
    const auto P = mx::array(p.data(), {5}, mx::int32);
    auto ts = scores({qb, k, plan, P, SC}, {{kvh, 96, kExactTail}}, {mx::float32}, {kExactTail, 96, kvh}, {kExactTail, 1, 1},
                     consts, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return merge({po[0], pm[0], pl[0], ts, v, plan, P}, {{1, heads, n, dim}}, {mx::float32}, {256, 96, kvh}, {256, 1, 1},
                 consts, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// the exact kernel runs on the GPU's matrix unit through Metal Performance Primitives tensor ops, our own kernels: it
// needs no patched MLX. Probed once by compiling and running it on a tiny cache (false: the GPU has no tensor ops)
inline bool matrix_unit_available() {
    static const bool ok = [] {
        namespace mx = mlx::core;
        try {
            auto q = mx::zeros({1, 6, 1, 256}, mx::float32);
            auto k = mx::zeros({1, 1, 128, 256}, mx::bfloat16);
            mx::eval(decode_attention_exact(q, k, k, 0.0625f, 100));
            return true;
        } catch (const std::exception&) {return false;}
    }();
    return ok;
}

// compile the exact-attention kernels once at load, so the first request does not pay the Metal compile
inline void warm_exact_attention_kernels(int heads, int kvheads) {
    namespace mx = mlx::core;
    if (heads != 6 * kvheads || !matrix_unit_available()) return;
    auto q = mx::astype(mx::astype(mx::random::normal({1, heads, 16, 256}), mx::bfloat16), mx::float32);
    auto k = mx::astype(mx::random::normal({1, kvheads, 1024, 256}), mx::bfloat16);
    mx::eval(decode_attention_exact(q, k, k, 0.0625f, 1000));
}

// QSA rows over the int8 KV cache (K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]):
// the same plan and block selection as decode_attention_exact's sparse rows; each row's selected keys and then its loop
// keys (T .. the row) in FP32, one key at a time, merged in that order. Rows below qsa.from come back as zeros: the
// caller takes them from decode_attention_int8. A row depends only on its query and the committed cache.
inline mlx::core::array decode_attention_qsa_int8(const mlx::core::array& q, const mlx::core::array& k8t, const mlx::core::array& v8,
                                                  const mlx::core::array& kscale, const mlx::core::array& vscale, float scale,
                                                  int length, const QsaRows& qsa) {
    namespace mx = mlx::core;
    const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), kvh = v8.shape(0), cap = v8.shape(1);
    if (n < 1 || n > 16 || dim != 256 || heads != 6 * kvh || k8t.ndim() != 4 || k8t.shape(2) != 256 || k8t.shape(3) != 1024 ||
        length < n || length > cap)
        throw std::invalid_argument("QSA int8: 1-16 rows, 6 query heads per KV head, head dim 256, int8 K^T blocks of 1024");
    const int prefix = length - n, deepest = prefix + n - 1 - 15, bound = deepest > 0 ? deepest / kExactTile * kExactTile : 0;
    constexpr int pw = 2 + kExactTail, chunks = kQsaChunks + 1;  // the selected keys, then the loop keys
    const auto plan = shape_constant("exact_rows", {prefix, n}, [&] {return exact_rows_plan(prefix, n, nullptr);});
    auto qq = qsa.q;
    if (n < 16) qq = mx::concatenate({qq, mx::zeros({16 - n, kQsaHeads, kQsaDim}, mx::float32)}, 0);
    const auto sel = qsa_select(qq, qsa.blocks, plan, n, std::max(1, bound / kQsaBlock), pw, qsa.from);
    auto qf = mx::astype(q, mx::float32);
    if (n < 16) qf = mx::concatenate({qf, mx::zeros({1, heads, 16 - n, dim}, mx::float32)}, 2);
    qf = mx::contiguous(mx::reshape(qf, {kvh, 96, dim}));
    static auto partials = mx::fast::metal_kernel("tqllm_qsa_partials_int8", {"QF", "KT", "V", "KS", "VS", "SEL", "PLAN", "P", "SC"},
                                                  {"PO", "PM", "PL"}, R"METAL(
    const uint lane = thread_index_in_simdgroup, h = simdgroup_index_in_threadgroup;
    const uint c = threadgroup_position_in_grid.x, t = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int N = P[0], CAP = P[2], NBA = P[3];
    const int T = PLAN[t * uint(PW)];
    const size_t base = (size_t(kh) * uint(NCH) + c) * 96u + h * 16u + t;
    float acc[8];
    for (uint i = 0; i < 8u; ++i) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    if (int(t) < N && T >= P[1]) {
        const device float* qr = QF + (size_t(kh) * 96u + h * 16u + t) * 256u + lane * 8u;
        float qv[8];
        for (uint i = 0; i < 8u; ++i) qv[i] = qr[i];
        const float sc = SC;
        const bool loop = c == uint(NCH) - 1u;
        const uint count = loop ? uint(PLAN[t * uint(PW) + 1u]) : uint(SPAN);
        for (uint j = 0; j < count; ++j) {
            const uint at = c * uint(SPAN) + j;
            const uint key = loop ? uint(PLAN[t * uint(PW) + 2u + j]) : uint(SEL[t * uint(PICK) + (at >> 2)]) * 4u + (at & 3u);
            const device int8_t* kc = KT + ((size_t(kh) * uint(NBA) + key / 1024u) * 256u + lane * 8u) * 1024u + key % 1024u;
            float d = 0.0f;
            for (uint i = 0; i < 8u; ++i) d = fma(qv[i], float(kc[size_t(i) * 1024u]), d);
            const float s = simd_sum(d) * float(KS[size_t(kh) * uint(CAP) + key]) * sc;
            const device int8_t* vc = V + (size_t(kh) * uint(CAP) + key) * 256u + lane * 8u;
            const float vs = float(VS[size_t(kh) * uint(CAP) + key]);
            if (s > m) {
                const float a = m == -INFINITY ? 0.0f : exp(m - s);
                l = l * a + 1.0f;
                for (uint i = 0; i < 8u; ++i) acc[i] = acc[i] * a + float(vc[i]) * vs;
                m = s;
            } else {
                const float p = exp(s - m);
                l += p;
                for (uint i = 0; i < 8u; ++i) acc[i] = fma(p, float(vc[i]) * vs, acc[i]);
            }
        }
    }
    for (uint i = 0; i < 8u; ++i) PO[base * 256u + lane * 8u + i] = acc[i];
    if (lane == 0u) {PM[base] = m; PL[base] = l;}
)METAL");
    static auto merge = mx::fast::metal_kernel("tqllm_qsa_merge", {"PO", "PM", "PL", "P"}, {"Y"}, R"METAL(
    const uint e = thread_position_in_grid.x, r = thread_position_in_grid.y, kh = thread_position_in_grid.z;
    const int N = P[0], t = int(r) & 15, h = int(r) >> 4;
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
    const std::vector<int32_t> p = {n, qsa.from, cap, k8t.shape(1)};
    const auto P = mx::array(p.data(), {4}, mx::int32);
    auto parts = partials({qf, k8t, v8, kscale, vscale, sel, plan, P, mx::array(scale)},
                          {{kvh, chunks, 96, dim}, {kvh, chunks, 96}, {kvh, chunks, 96}}, {mx::float32, mx::float32, mx::float32},
                          {32 * 6 * chunks, 16, kvh}, {32 * 6, 1, 1},
                          {{"NCH", chunks}, {"SPAN", kQsaPick * kQsaBlock / kQsaChunks}, {"PICK", kQsaPick}, {"PW", pw}},
                          std::nullopt, false, mx::Device(mx::Device::gpu));
    return merge({parts[0], parts[1], parts[2], P}, {{1, heads, n, dim}}, {mx::float32}, {256, 96, kvh}, {256, 1, 1},
                 {{"NCH", chunks}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

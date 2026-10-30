#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace tqllm::mlx_backend {
// Decode / verify attention for 1-16 query rows over the BF16 KV cache: every call has the shape of a 16-row call, so a
// row's result does not depend on how many rows share the call (AR and the 16-row verify agree bit for bit).
inline void check_decode_rows_shape(const mlx::core::array& q, const mlx::core::array& k, const mlx::core::array& v,
                                    int valid) {
    namespace mx = mlx::core;
    if (q.ndim() != 4 || k.ndim() != 4 || q.shape(0) != 1 || k.shape(0) != 1 || q.shape(2) < 1 || q.shape(2) > 16 ||
        (valid > 0 ? valid : k.shape(2)) < q.shape(2) || valid > k.shape(2) || v.shape() != k.shape() || q.shape(1) % k.shape(1) ||
        (q.shape(3) != 128 && q.shape(3) != 256) || k.shape(3) != q.shape(3) || q.dtype() != mx::float32 ||
        (k.dtype() != mx::float32 && k.dtype() != mx::bfloat16) || v.dtype() != k.dtype())
        throw std::invalid_argument("unsupported decode rows attention shape");
}

// v4: MLX's fused SDPA matrix kernel (the 9+ query-row path, j700: 8K x 16 rows 0.49 vs v1 3.05 ms) for every call, AR
// included. Queries are padded at the front to 16 rows (fewer only while the cache is shorter) so the shape never changes;
// with the bottom-right causal mask each row's result depends on its own query and visible keys only (masked blocks add
// exact zeros), so a row agrees bit for bit across row counts. BF16 queries and output, like the prefill SDPA.
inline mlx::core::array decode_rows_attention_v4(const mlx::core::array& q, const mlx::core::array& k,
                                                 const mlx::core::array& v, float scale, int valid = -1) {
    namespace mx = mlx::core;
    check_decode_rows_shape(q, k, v, valid);
    const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), length = valid > 0 ? valid : k.shape(2);
    // MLX's SDPA takes another kernel above 4096 keys (j743 / j751: rows of a call straddling 4096 were not bit equal to
    // the same rows alone). Rows that see <= 4096 keys run as their own call over 4096 keys, the rest over the full
    // length, so every row gets the kernel its AR call gets.
    constexpr int kSplit = 4096;
    if (length > kSplit && length - n + 1 <= kSplit) {
        const int below = kSplit - (length - n);
        auto lo = decode_rows_attention_v4(mx::slice(q, {0, 0, 0, 0}, {1, heads, below, dim}), k, v, scale, kSplit);
        auto hi = decode_rows_attention_v4(mx::slice(q, {0, 0, below, 0}, {1, heads, n, dim}), k, v, scale, length);
        return mx::concatenate({lo, hi}, 2);
    }
    const int rows = std::min(16, length);
    auto query = mx::astype(q, mx::bfloat16);
    if (rows > n) {
        auto pad = mx::broadcast_to(mx::slice(query, {0, 0, 0, 0}, {1, heads, 1, dim}), {1, heads, rows - n, dim});
        query = mx::concatenate({pad, query}, 2);
    }
    auto keys = mx::slice(k, {0, 0, 0, 0}, {1, k.shape(1), length, dim});
    auto values = mx::slice(v, {0, 0, 0, 0}, {1, v.shape(1), length, dim});
    auto out = mx::fast::scaled_dot_product_attention(query, keys, values, scale, "causal");
    return mx::astype(mx::slice(out, {0, 0, rows - n, 0}, {1, heads, rows, dim}), mx::float32);
}

// Query rows per decode / verify attention call: 8 for up to 8 rows (AR and a 7-token verify), else 16. A row's result
// does not depend on the shape (test-mlx-decode-rows DECODE_ROWS_SHAPES, bitwise), so a verify width may change per step;
// MLX's fused attention is compute-bound in the folded rows (j990), so 8 rows halve the attention work at long context.
inline int decode_row_shape(int n) {return n <= 8 ? 8 : 16;}

// v7: GQA folded into query rows, so each KV head's keys and values are read once for its G query heads: q [1, H, n, d]
// -> [1, KH, G * R, d] (rows padded at the front to R = decode_row_shape(n) per head as in v4), one MLX SDPA call per
// layer with an explicit per-row causal mask (row r sees keys <= L - R + r % R). Rows stay independent of each other, and
// AR uses the same R-row shape, so a row agrees bit for bit across row counts.
inline mlx::core::array decode_rows_attention_v7(const mlx::core::array& q, const mlx::core::array& k,
                                                 const mlx::core::array& v, float scale, int valid = -1) {
    namespace mx = mlx::core;
    check_decode_rows_shape(q, k, v, valid);
    const int heads = q.shape(1), kvh = k.shape(1), group = heads / kvh, n = q.shape(2), dim = q.shape(3);
    const int shape = decode_row_shape(n);
    const int length = valid > 0 ? valid : k.shape(2), rows = std::min(shape, length);
    constexpr int kSplit = 4096;  // as v4: rows <= 4096 keys run as their own call
    if (length > kSplit && length - n + 1 <= kSplit) {
        const int below = kSplit - (length - n);
        auto lo = decode_rows_attention_v7(mx::slice(q, {0, 0, 0, 0}, {1, heads, below, dim}), k, v, scale, kSplit);
        auto hi = decode_rows_attention_v7(mx::slice(q, {0, 0, below, 0}, {1, heads, n, dim}), k, v, scale, length);
        return mx::concatenate({lo, hi}, 2);
    }
    auto query = mx::astype(q, mx::bfloat16);
    if (rows > n) {
        auto pad = mx::broadcast_to(mx::slice(query, {0, 0, 0, 0}, {1, heads, 1, dim}), {1, heads, rows - n, dim});
        query = mx::concatenate({pad, query}, 2);
    }
    // [1, H, rows, d] -> [1, KH, G * rows, d]: query head h = kv head h / G, row index g * rows + t
    query = mx::reshape(query, {1, kvh, group * rows, dim});
    auto keys = mx::slice(k, {0, 0, 0, 0}, {1, kvh, length, dim});
    auto values = mx::slice(v, {0, 0, 0, 0}, {1, kvh, length, dim});
    auto t = mx::remainder(mx::arange(group * rows), mx::array(rows));                       // token of each folded row
    auto visible = mx::add(t, mx::array(length - rows));                                       // last visible key per row
    auto mask = mx::less_equal(mx::reshape(mx::arange(length), {1, length}), mx::reshape(visible, {group * rows, 1}));
    auto out = mx::fast::scaled_dot_product_attention(query, keys, values, scale, "array", mask);
    out = mx::reshape(out, {1, heads, rows, dim});
    return mx::astype(mx::slice(out, {0, 0, rows - n, 0}, {1, heads, rows, dim}), mx::float32);
}

constexpr int kAttentionSegment = 1024;  // the KV cache capacity is a multiple of it (qwen35.cpp)
#ifdef MLX_HAS_SDPA_LSE
// v11: v7 split over fixed 1024-key segments counted from key 0, all segments as the batch of one MLX fused attention
// call (our MLX build's scaled_dot_product_attention_lse: the M5 matrix-unit kernel plus each row's log-sum-exp), merged
// per row in segment order. v7 runs one call over the whole cache: (folded rows / 64-row query tile) x 4 kv heads
// threadgroups, 4 at the 8-row shape (j991: 8 rows only -26 % vs 16 at 64K for half the work), here that many per segment.
// Rows are padded at the front to decode_row_shape(n) as in v7; segments a row cannot see return -inf and are skipped, so
// a row's result does not depend on how many rows share the call. K / V are strided views of the cache (its capacity is a multiple of the segment, zero past the filled rows); a
// shorter buffer (tests) is zero padded. Needs the matrix unit (M5): elsewhere decode_rows_attention keeps v7.
inline mlx::core::array decode_rows_attention_v11(const mlx::core::array& q, const mlx::core::array& k,
                                                  const mlx::core::array& v, float scale, int valid = -1,
                                                  int seg = kAttentionSegment) {
    namespace mx = mlx::core;
    check_decode_rows_shape(q, k, v, valid);
    if (k.dtype() != mx::bfloat16) throw std::invalid_argument("decode rows attention v11 reads a BF16 KV cache");
    if (seg < 64 || seg % kAttentionSegment) throw std::invalid_argument("decode rows attention v11 segment must be a multiple of the cache granule");
    const int heads = q.shape(1), kvh = k.shape(1), group = heads / kvh, n = q.shape(2), dim = q.shape(3);
    const int length = valid > 0 ? valid : k.shape(2), rows = std::min(decode_row_shape(n), length);
    const int segments = (length + seg - 1) / seg;
    auto keys = k, values = v;
    if (keys.shape(2) < segments * seg) {
        const int extra = segments * seg - keys.shape(2);
        keys = mx::concatenate({keys, mx::zeros({1, kvh, extra, dim}, keys.dtype())}, 2);
        values = mx::concatenate({values, mx::zeros({1, kvh, extra, dim}, values.dtype())}, 2);
    }
    keys = mx::contiguous(keys); values = mx::contiguous(values);
    const int64_t cap = keys.shape(2);
    keys = mx::as_strided(keys, {segments, kvh, seg, dim}, {int64_t(seg) * dim, cap * dim, dim, 1}, 0);
    values = mx::as_strided(values, {segments, kvh, seg, dim}, {int64_t(seg) * dim, cap * dim, dim, 1}, 0);
    auto query = mx::astype(q, mx::bfloat16);
    if (rows > n) {
        auto pad = mx::broadcast_to(mx::slice(query, {0, 0, 0, 0}, {1, heads, 1, dim}), {1, heads, rows - n, dim});
        query = mx::concatenate({pad, query}, 2);
    }
    query = mx::broadcast_to(mx::reshape(query, {1, kvh, group * rows, dim}), {segments, kvh, group * rows, dim});
    // row r (token r % rows) sees keys <= length - rows + r % rows
    auto key_index = mx::reshape(mx::arange(segments * seg), {segments, 1, 1, seg});
    auto last = mx::add(mx::remainder(mx::arange(group * rows), mx::array(rows)), mx::array(length - rows));
    auto mask = mx::less_equal(key_index, mx::reshape(last, {1, 1, group * rows, 1}));
    auto parts = mx::fast::scaled_dot_product_attention_lse(query, keys, values, scale, mask);
    static auto merge = mx::fast::metal_kernel("tqllm_attn_v11_merge", {"O", "LSE", "P"}, {"Y"}, R"METAL(
    // one thread per (kv head row, dim element): fold the segments in order; a segment with no visible key is skipped
    const uint e = thread_position_in_grid.x, r = thread_position_in_grid.y;
    const int SEG = P[0], ROWS = P[1];
    float m = -INFINITY, den = 0.0f, acc = 0.0f;
    for (int s = 0; s < SEG; ++s) {
        const float l = LSE[size_t(s) * ROWS + r];
        if (!(l > -INFINITY)) continue;
        const float nm = max(m, l), a = m == nm ? 1.0f : exp(m - nm), c = l == nm ? 1.0f : exp(l - nm);
        den = den * a + c;
        acc = acc * a + float(O[(size_t(s) * ROWS + r) * DIM + e]) * c;
        m = nm;
    }
    Y[size_t(r) * DIM + e] = acc / den;
)METAL");
    std::vector<int32_t> p = {segments, kvh * group * rows};
    auto P = mx::array(p.data(), {2}, mx::int32);
    auto y = merge({parts[0], parts[1], P}, {{1, kvh, group * rows, dim}}, {mx::float32}, {dim, kvh * group * rows, 1}, {std::min(dim, 256), 1, 1},
                   {{"DIM", dim}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    y = mx::reshape(y, {1, heads, rows, dim});
    return mx::astype(mx::astype(mx::slice(y, {0, 0, rows - n, 0}, {1, heads, rows, dim}), mx::bfloat16), mx::float32);
}

// v11 needs our MLX build's log-sum-exp attention on the matrix unit, which MLX has for some head dims only (256 / 512
// on 0.32.3); probed once per head dim (false: v7)
inline bool attention_v11_available(int dim = 256) {
    namespace mx = mlx::core;
    auto probe = [](int d) {
        try {
            auto q = mx::zeros({1, 24, 1, d}, mx::float32), k = mx::zeros({1, 4, 64, d}, mx::bfloat16);
            mx::eval(decode_rows_attention_v11(q, k, k, 0.0625f, 64));
            return true;
        } catch (const std::exception&) {return false;}
    };
    static const bool ok128 = probe(128), ok256 = probe(256);
    return dim == 128 ? ok128 : dim == 256 ? ok256 : false;
}
#else
inline bool attention_v11_available(int = 256) {return false;}
#endif

// v12 (test and bench, j1069-j1071; dispatch candidate for 8 rows at >= 16K keys): own 8-row attention on an int8 KV cache. K is stored transposed in 1024-key blocks
// ([kv head][block][dim][1024] int8) and V row-major ([kv head][key][dim] int8), both with one FP16 scale per (kv head,
// key). One threadgroup per (block, kv head) holds every query row of the kv head (G heads x n rows, 16-row tiles, two
// simdgroups per tile = halves of the head dim). Per 32-key chunk: S = Q K^T on the matrix unit (16 x 32 x 16 ops, the
// NAX lane layout), each simdgroup of the pair over its half of the head dim (Q fragments held in registers), the two
// partials added in the softmax; the int8 codes go into the right operand as they are (exact in FP16) and the scales are
// applied outside the matrix ops: K's per key on S, V's per key folded into P. Softmax by 4 threads per row over the
// row's visible keys (key <= L - n + t), then O += (P * vscale) V. A row's arithmetic depends only on its own query and
// visible keys, so the n-row verify equals n single-row calls bit for bit.
inline mlx::core::array decode_rows_attention_v12(const mlx::core::array& q, const mlx::core::array& k8t,
                                                  const mlx::core::array& v8, const mlx::core::array& kscale,
                                                  const mlx::core::array& vscale, float scale, int length, int variant = 0) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.dtype() != mx::float32 || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode rows attention v12: q [1, heads, n <= 16, 256] FP32");
    const int heads = q.shape(1), kvh = v8.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = v8.shape(1);
    if (kvh < 1 || heads % kvh || k8t.ndim() != 4 || k8t.shape(0) != kvh || k8t.shape(1) < blocks || k8t.shape(2) != dim || k8t.shape(3) != kBlock ||
        k8t.dtype() != mx::int8 || v8.ndim() != 3 || v8.shape(2) != dim || v8.dtype() != mx::int8 || cap < length || length < 1 ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 || vscale.dtype() != mx::float16)
        throw std::invalid_argument("decode rows attention v12: int8 K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = tiles * 16;
    // [1, H, n, d] is [kv head][g][t][d]: each kv head's rows are contiguous; FP16 operands (BF16 values are exact in FP16)
    auto query = mx::reshape(mx::astype(mx::astype(q, mx::bfloat16), mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_v12_pass1", {"Q", "KT", "V", "KS", "VS", "P"}, {"M", "S", "O"}, R"METAL(
    constexpr int C = 32, RP = TILES * 16, HD = DIM / 2, CQ = C / 4;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint blk = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8];
    const float sc = as_type<float>(P[6]);
    const int k0 = int(blk) * BLOCK, kn = min(BLOCK, L - k0);
    const int tile = int(sg) / 2, half_d = int(sg) % 2;
    const int row = int(tid) / 4, quarter = int(tid) % 4;               // softmax: 4 threads per row, CQ keys each
    const int last = row < ROWS ? L - NR + row % NR : -1;               // last visible key of the row
    // NAX lane geometry of the 16 x 32 x 16 op: left a[i] = A[fm + 8 (i >> 2)][fn + (i & 3)], right b[w * 8 + i] =
    // B[n = w * 16 + fn + (i & 3)][k = fm + 8 (i >> 2)], destination index via get_multidimensional_index
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    threadgroup float st[2][RP * C];
    threadgroup half pt[RP * C];
    threadgroup float arow[RP];
    threadgroup float kscl[C], vscl[C];
    threadgroup float pad[VAR == 4 ? 4096 : 1];   // bench: +16 KB threadgroup memory, nothing else (occupancy probe)
    if (VAR == 4) pad[tid] = float(tid);
    const device half* qrow = Q + (size_t(kh) * QROWS + tile * 16) * DIM;
    const device int8_t* kt = KT + (size_t(kh) * P[4] + blk) * DIM * BLOCK;    // [dim][key in block]; P[4] = blocks allocated
    const device int8_t* vv = V + size_t(kh) * CAP * DIM;                       // [key][dim]
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (ushort i = 0; i < 16; ++i) {acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f;}
    // destination element -> (n, m) once; the same layout for S (m = row, n = key) and O (m = row, n = dim)
    short dn[16], dm[16];
    for (ushort i = 0; i < 16; ++i) {
        if (acc0.is_valid_element(i)) {const auto idx = acc0.get_multidimensional_index(i); dn[i] = short(idx[0]); dm[i] = short(idx[1]);}
        else {dn[i] = -1; dm[i] = -1;}
    }
    // this simdgroup's Q fragments: 8 k-steps over its half of the head dim
    half qa[8][8];
    for (int ks = 0; ks < 8; ++ks) for (short i = 0; i < 4; ++i) {
        const int d = (half_d * 8 + ks) * 16 + fn + i;
        qa[ks][i] = qrow[size_t(fm) * DIM + d]; qa[ks][4 + i] = qrow[size_t(fm + 8) * DIM + d];
    }
    float m = -INFINITY, s = 0.0f;                                      // the row's running max and sum (same in its 4 threads)
    for (int c0 = 0; c0 < kn; c0 += C) {
        // chunk scales to threadgroup memory (keys past the view read as 0 and are masked anyway)
        if (tid < uint(C)) {
            const int key = k0 + c0 + int(tid);
            kscl[tid] = key < L ? float(KS[size_t(kh) * CAP + key]) : 0.0f;
            vscl[tid] = key < L ? float(VS[size_t(kh) * CAP + key]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // partial S tile [16 rows x 32 keys] over this half of the head dim; K^T fragment: 4 consecutive keys per load
            auto sacc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0.0f;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int d = (half_d * 8 + ks) * 16 + fm + 8 * h, key0 = c0 + w * 16 + fn;   // keys key0 .. key0 + 3 (within the block)
                    char4 kv = char4(0);                                                            // keys past the view read as 0 (masked)
                    if (VAR == 1) kv = char4(1);                                                    // bench: operands without loads
                    else if (key0 + 3 < kn) kv = *(const device char4*)(kt + size_t(d) * BLOCK + key0);
                    else for (short i = 0; i < 4; ++i) kv[i] = key0 + i < kn ? kt[size_t(d) * BLOCK + key0 + i] : char(0);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(kv[i]);
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * C + dn[i]] = sacc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: each thread its CQ keys, the row's max and sum over the 4 threads by fixed xor shuffles
        if (VAR != 3) {
        float x[CQ];
        float lm = -INFINITY;
        for (int j = 0; j < CQ; ++j) {
            const int key = k0 + c0 + quarter * CQ + j;
            const int e = row * C + quarter * CQ + j;
            x[j] = key <= last ? (st[0][e] + st[1][e]) * kscl[quarter * CQ + j] * sc : -INFINITY;
            lm = max(lm, x[j]);
        }
        lm = max(lm, simd_shuffle_xor(lm, 1)); lm = max(lm, simd_shuffle_xor(lm, 2));
        const float mn = max(m, lm);
        const float alpha = mn == m ? 1.0f : exp(m - mn);
        float ls = 0.0f;
        for (int j = 0; j < CQ; ++j) {
            const float pr = x[j] == -INFINITY ? 0.0f : exp(x[j] - mn);
            ls += pr;
            pt[row * C + quarter * CQ + j] = half(pr * vscl[quarter * CQ + j]);
        }
        ls += simd_shuffle_xor(ls, 1); ls += simd_shuffle_xor(ls, 2);
        s = s * alpha + ls; m = mn;
        if (quarter == 0) arow[row] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (VAR == 2) {threadgroup_barrier(mem_flags::mem_threadgroup); continue;}   // bench: S + softmax only
        // rescale the accumulators of rows whose max moved
        if (VAR != 3) for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
            const float a = arow[tile * 16 + dm[i]];
            if (a != 1.0f) {acc0[i] *= a; acc1[i] *= a; acc2[i] *= a; acc3[i] *= a;}
        }
        // O[16 rows x 128 dims (this half)] += P[16 x 32 keys] V[32 keys x 128]: 2 k-steps x 4 dim blocks of 32
        const threadgroup half* prow = pt + (tile * 16) * C;
        for (int ks = 0; ks < 2; ++ks) {
            for (short i = 0; i < 4; ++i) {ca[i] = prow[fm * C + ks * 16 + fn + i]; ca[4 + i] = prow[(fm + 8) * C + ks * 16 + fn + i];}
            for (short nb = 0; nb < 4; ++nb) {
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int kc = ks * 16 + fm + 8 * h, key = k0 + c0 + kc;                // chunk key kc, dims d0 .. d0 + 3
                    const int d0 = half_d * HD + nb * 32 + w * 16 + fn;
                    const char4 vv4 = VAR == 1 ? char4(1) : kc < kn ? *(const device char4*)(vv + size_t(key) * DIM + d0) : char4(0);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(vv4[i]);
                }
                if (nb == 0) op.run(ca, cb, acc0); else if (nb == 1) op.run(ca, cb, acc1); else if (nb == 2) op.run(ca, cb, acc2); else op.run(ca, cb, acc3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
        const int r = tile * 16 + dm[i];
        if (r >= ROWS) continue;
        const size_t base = ((size_t(kh) * ROWS + r) * NB + blk) * DIM + half_d * HD;
        O[base + dn[i]] = acc0[i]; O[base + 32 + dn[i]] = acc1[i]; O[base + 64 + dn[i]] = acc2[i]; O[base + 96 + dn[i]] = acc3[i];
    }
    if (row < ROWS && quarter == 0) {const size_t base = (size_t(kh) * ROWS + row) * NB + blk; M[base] = m; S[base] = s;}
    if (VAR == 4 && L < 0) M[0] = pad[(tid + 1) % 4096];   // never true: keeps the probe's allocation
)METAL", header);
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_v12_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
    // one simdgroup per (row, kv head): combine the row's blocks 0..NB-1 in order
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = P[3], NB = P[5];
    float m = -INFINITY, s = 0.0f, o[DIM / 32];
    for (int e = 0; e < DIM / 32; ++e) o[e] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t base = (size_t(kh) * ROWS + r) * NB + b;
        const float mb = M[base], sb = S[base];
        if (sb == 0.0f) continue;                         // no visible key of this row in the block
        const float mn = max(m, mb), a = m == mn ? 1.0f : exp(m - mn), c = mb == mn ? 1.0f : exp(mb - mn);
        s = s * a + sb * c;
        for (int e = 0; e < DIM / 32; ++e) o[e] = o[e] * a + O[base * DIM + e * 32 + lane] * c;
        m = mn;
    }
    const size_t out = (size_t(kh) * ROWS + r) * DIM;    // [kv head][g][t][d] = [1, H, n, d]
    for (int e = 0; e < DIM / 32; ++e) Y[out + e * 32 + lane] = o[e] / s;
)METAL");
    // the allocated block count is a runtime parameter: a template on it compiled a new kernel per cache size inside the
    // first decode step (j1152: 7.6 s at 249K)
    std::vector<int32_t> p = {length, n, group, rows, k8t.shape(1), blocks, 0, cap, qrows};
    float scf = scale; std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {9}, mx::int32);
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TILES", tiles}, {"VAR", variant}};
    auto part = pass1({query, k8t, v8, kscale, vscale, P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {64 * tiles, blocks, kvh}, {64 * tiles, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // BF16-rounded values through FP32, as the SDPA versions return them
    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}
// v15: v12's arithmetic (same operands, same op order: results equal v12 bit for bit) with each 32-key K / V chunk loaded
// with 16-byte reads and converted to FP16 once per threadgroup into threadgroup memory, then read by every query tile
// (v12: every tile loads and converts its own copy; j1184: the loads are 44 % of the 64K 8-row call). At most 3 tiles
// per threadgroup (the 32 KB limit, j1185); a 16-row call is two threadgroups per (block, kv head).
inline mlx::core::array decode_rows_attention_v15(const mlx::core::array& q, const mlx::core::array& k8t,
                                                  const mlx::core::array& v8, const mlx::core::array& kscale,
                                                  const mlx::core::array& vscale, float scale, int length) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.dtype() != mx::float32 || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode rows attention v15: q [1, heads, n <= 16, 256] FP32");
    const int heads = q.shape(1), kvh = v8.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = v8.shape(1);
    if (kvh < 1 || heads % kvh || k8t.ndim() != 4 || k8t.shape(0) != kvh || k8t.shape(1) < blocks || k8t.shape(2) != dim || k8t.shape(3) != kBlock ||
        k8t.dtype() != mx::int8 || v8.ndim() != 3 || v8.shape(2) != dim || v8.dtype() != mx::int8 || cap < length || length < 1 ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 || vscale.dtype() != mx::float16)
        throw std::invalid_argument("decode rows attention v15: int8 K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = (tiles + 2) / 3 * 3 * 16;   // whole 3-tile groups of query rows
    // [1, H, n, d] is [kv head][g][t][d]: each kv head's rows are contiguous; FP16 operands (BF16 values are exact in FP16)
    auto query = mx::reshape(mx::astype(mx::astype(q, mx::bfloat16), mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_v15_pass1", {"Q", "KT", "V", "KS", "VS", "P"}, {"M", "S", "O"}, R"METAL(
    constexpr int C = 32, RP = TPG * 16, HD = DIM / 2, CQ = C / 4, NT = TPG * 64;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint grp = threadgroup_position_in_grid.x, blk = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int r0 = int(grp) * RP;                                       // this threadgroup's first fused row
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8];
    const float sc = as_type<float>(P[6]);
    const int k0 = int(blk) * BLOCK, kn = min(BLOCK, L - k0);
    const int tile = int(sg) / 2, half_d = int(sg) % 2;
    const int row = int(tid) / 4, quarter = int(tid) % 4;               // softmax: 4 threads per row, CQ keys each
    const int grow = r0 + row;                                          // the row in the kv head's fused rows
    const int last = grow < ROWS ? L - NR + grow % NR : -1;             // last visible key of the row
    // NAX lane geometry of the 16 x 32 x 16 op: left a[i] = A[fm + 8 (i >> 2)][fn + (i & 3)], right b[w * 8 + i] =
    // B[n = w * 16 + fn + (i & 3)][k = fm + 8 (i >> 2)], destination index via get_multidimensional_index
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    threadgroup float st[2][RP * C];
    threadgroup half pt[RP * C];
    threadgroup float arow[RP];
    threadgroup float kscl[C], vscl[C];
    threadgroup half kv16[DIM * C];   // the chunk's K^T [dim][key], then its V [key][dim], as FP16, shared by every tile
    const device half* qrow = Q + (size_t(kh) * QROWS + r0 + tile * 16) * DIM;
    const device int8_t* kt = KT + (size_t(kh) * P[4] + blk) * DIM * BLOCK;    // [dim][key in block]; P[4] = blocks allocated
    const device int8_t* vv = V + size_t(kh) * CAP * DIM;                       // [key][dim]
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (ushort i = 0; i < 16; ++i) {acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f;}
    // destination element -> (n, m) once; the same layout for S (m = row, n = key) and O (m = row, n = dim)
    short dn[16], dm[16];
    for (ushort i = 0; i < 16; ++i) {
        if (acc0.is_valid_element(i)) {const auto idx = acc0.get_multidimensional_index(i); dn[i] = short(idx[0]); dm[i] = short(idx[1]);}
        else {dn[i] = -1; dm[i] = -1;}
    }
    // this simdgroup's Q fragments: 8 k-steps over its half of the head dim
    half qa[8][8];
    for (int ks = 0; ks < 8; ++ks) for (short i = 0; i < 4; ++i) {
        const int d = (half_d * 8 + ks) * 16 + fn + i;
        qa[ks][i] = qrow[size_t(fm) * DIM + d]; qa[ks][4 + i] = qrow[size_t(fm + 8) * DIM + d];
    }
    float m = -INFINITY, s = 0.0f;                                      // the row's running max and sum (same in its 4 threads)
    for (int c0 = 0; c0 < kn; c0 += C) {
        // chunk scales to threadgroup memory (keys past the view read as 0 and are masked anyway)
        if (tid < uint(C)) {
            const int key = k0 + c0 + int(tid);
            kscl[tid] = key < L ? float(KS[size_t(kh) * CAP + key]) : 0.0f;
            vscl[tid] = key < L ? float(VS[size_t(kh) * CAP + key]) : 0.0f;
        }
        // stage K^T [256 dims][32 keys]: 16 bytes per load (half a dim row), converted once for all tiles
        for (int e = int(tid); e < DIM * 2; e += NT) {
            const int d = e / 2, part = e % 2;
            const char4 a0 = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + part * 16);
            const char4 a1 = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + part * 16 + 4);
            const char4 a2 = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + part * 16 + 8);
            const char4 a3 = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + part * 16 + 12);
            threadgroup half4* dst = (threadgroup half4*)(kv16 + d * C + part * 16);
            dst[0] = half4(a0); dst[1] = half4(a1); dst[2] = half4(a2); dst[3] = half4(a3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // partial S tile [16 rows x 32 keys] over this half of the head dim; K^T fragment: 4 consecutive keys per load
            auto sacc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0.0f;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int d = (half_d * 8 + ks) * 16 + fm + 8 * h;
                    // keys past the view hold finite codes of the allocated block: their scores are masked below
                    const half4 kv = *(const threadgroup half4*)(kv16 + d * C + w * 16 + fn);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = kv[i];
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * C + dn[i]] = sacc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: each thread its CQ keys, the row's max and sum over the 4 threads by fixed xor shuffles
        {
        float x[CQ];
        float lm = -INFINITY;
        for (int j = 0; j < CQ; ++j) {
            const int key = k0 + c0 + quarter * CQ + j;
            const int e = row * C + quarter * CQ + j;
            x[j] = key <= last ? (st[0][e] + st[1][e]) * kscl[quarter * CQ + j] * sc : -INFINITY;
            lm = max(lm, x[j]);
        }
        lm = max(lm, simd_shuffle_xor(lm, 1)); lm = max(lm, simd_shuffle_xor(lm, 2));
        const float mn = max(m, lm);
        const float alpha = mn == m ? 1.0f : exp(m - mn);
        float ls = 0.0f;
        for (int j = 0; j < CQ; ++j) {
            const float pr = x[j] == -INFINITY ? 0.0f : exp(x[j] - mn);
            ls += pr;
            pt[row * C + quarter * CQ + j] = half(pr * vscl[quarter * CQ + j]);
        }
        ls += simd_shuffle_xor(ls, 1); ls += simd_shuffle_xor(ls, 2);
        s = s * alpha + ls; m = mn;
        if (quarter == 0) arow[row] = alpha;
        }
        // stage V [32 keys][256 dims] into the same buffer (every S read of it is behind the barrier above); rows past the
        // cache read as 0 (P is 0 there)
        for (int e = int(tid); e < C * DIM / 16; e += NT) {
            const int kc = e / (DIM / 16), d0 = (e % (DIM / 16)) * 16, key = k0 + c0 + kc;
            threadgroup half4* dst = (threadgroup half4*)(kv16 + kc * DIM + d0);
            if (key < CAP) {
                const device char4* src = (const device char4*)(vv + size_t(key) * DIM + d0);
                dst[0] = half4(src[0]); dst[1] = half4(src[1]); dst[2] = half4(src[2]); dst[3] = half4(src[3]);
            } else {dst[0] = half4(0.0h); dst[1] = half4(0.0h); dst[2] = half4(0.0h); dst[3] = half4(0.0h);}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rescale the accumulators of rows whose max moved
        for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
            const float a = arow[tile * 16 + dm[i]];
            if (a != 1.0f) {acc0[i] *= a; acc1[i] *= a; acc2[i] *= a; acc3[i] *= a;}
        }
        // O[16 rows x 128 dims (this half)] += P[16 x 32 keys] V[32 keys x 128]: 2 k-steps x 4 dim blocks of 32
        const threadgroup half* prow = pt + (tile * 16) * C;
        for (int ks = 0; ks < 2; ++ks) {
            for (short i = 0; i < 4; ++i) {ca[i] = prow[fm * C + ks * 16 + fn + i]; ca[4 + i] = prow[(fm + 8) * C + ks * 16 + fn + i];}
            for (short nb = 0; nb < 4; ++nb) {
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int kc = ks * 16 + fm + 8 * h;                                       // chunk key kc, dims d0 .. d0 + 3
                    const int d0 = half_d * HD + nb * 32 + w * 16 + fn;
                    const half4 vv4 = *(const threadgroup half4*)(kv16 + kc * DIM + d0);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = vv4[i];
                }
                if (nb == 0) op.run(ca, cb, acc0); else if (nb == 1) op.run(ca, cb, acc1); else if (nb == 2) op.run(ca, cb, acc2); else op.run(ca, cb, acc3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
        const int r = r0 + tile * 16 + dm[i];
        if (r >= ROWS) continue;
        const size_t base = ((size_t(kh) * ROWS + r) * NB + blk) * DIM + half_d * HD;
        O[base + dn[i]] = acc0[i]; O[base + 32 + dn[i]] = acc1[i]; O[base + 64 + dn[i]] = acc2[i]; O[base + 96 + dn[i]] = acc3[i];
    }
    if (grow < ROWS && quarter == 0) {const size_t base = (size_t(kh) * ROWS + grow) * NB + blk; M[base] = m; S[base] = s;}
)METAL", header);
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_v15_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
    // one simdgroup per (row, kv head): combine the row's blocks 0..NB-1 in order
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = P[3], NB = P[5];
    float m = -INFINITY, s = 0.0f, o[DIM / 32];
    for (int e = 0; e < DIM / 32; ++e) o[e] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t base = (size_t(kh) * ROWS + r) * NB + b;
        const float mb = M[base], sb = S[base];
        if (sb == 0.0f) continue;                         // no visible key of this row in the block
        const float mn = max(m, mb), a = m == mn ? 1.0f : exp(m - mn), c = mb == mn ? 1.0f : exp(mb - mn);
        s = s * a + sb * c;
        for (int e = 0; e < DIM / 32; ++e) o[e] = o[e] * a + O[base * DIM + e * 32 + lane] * c;
        m = mn;
    }
    const size_t out = (size_t(kh) * ROWS + r) * DIM;    // [kv head][g][t][d] = [1, H, n, d]
    for (int e = 0; e < DIM / 32; ++e) Y[out + e * 32 + lane] = o[e] / s;
)METAL");
    // the allocated block count is a runtime parameter: a template on it compiled a new kernel per cache size inside the
    // first decode step (j1152: 7.6 s at 249K)
    std::vector<int32_t> p = {length, n, group, rows, k8t.shape(1), blocks, 0, cap, qrows};
    float scf = scale; std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {9}, mx::int32);
    // at most 3 tiles (48 fused rows) per threadgroup: st + pt + the FP16 chunk stay under 32 KB (j1185: 16 rows x 6 GQA
    // heads in one threadgroup leave no room); a 16-row call runs its 6 tiles as 2 threadgroups over the same keys
    const int tpg = std::min(tiles, 3), groups = (tiles + tpg - 1) / tpg;
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TPG", tpg}};
    auto part = pass1({query, k8t, v8, kscale, vscale, P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {64 * tpg * groups, blocks, kvh}, {64 * tpg, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // BF16-rounded values through FP32, as the SDPA versions return them
    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}
// v16 (bench / test): v15's tiling (<= 3 tiles per threadgroup) with `chunk` keys per softmax round instead of 32 and
// K / V read straight from the cache as v12 (no FP16 staging: room for the larger S tile). The barriers per key halve
// at 64 (j1186: the per-chunk barrier chain, not the loads, is the cost). Each key's score is v12's; only the online
// softmax's rounds differ, so a row equals its 1-row call bit for bit and v12 within rounding.
inline mlx::core::array decode_rows_attention_v16(const mlx::core::array& q, const mlx::core::array& k8t,
                                                  const mlx::core::array& v8, const mlx::core::array& kscale,
                                                  const mlx::core::array& vscale, float scale, int length, int chunk = 64) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.dtype() != mx::float32 || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode rows attention v16: q [1, heads, n <= 16, 256] FP32");
    const int heads = q.shape(1), kvh = v8.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = v8.shape(1);
    if (kvh < 1 || heads % kvh || k8t.ndim() != 4 || k8t.shape(0) != kvh || k8t.shape(1) < blocks || k8t.shape(2) != dim || k8t.shape(3) != kBlock ||
        k8t.dtype() != mx::int8 || v8.ndim() != 3 || v8.shape(2) != dim || v8.dtype() != mx::int8 || cap < length || length < 1 ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 || vscale.dtype() != mx::float16)
        throw std::invalid_argument("decode rows attention v16: int8 K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = (tiles + 2) / 3 * 3 * 16;   // whole 3-tile groups of query rows
    // [1, H, n, d] is [kv head][g][t][d]: each kv head's rows are contiguous; FP16 operands (BF16 values are exact in FP16)
    auto query = mx::reshape(mx::astype(mx::astype(q, mx::bfloat16), mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_v16_pass1", {"Q", "KT", "V", "KS", "VS", "P"}, {"M", "S", "O"}, R"METAL(
    constexpr int C = CK, RP = TPG * 16, HD = DIM / 2, CQ = C / 4, NT = TPG * 64;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint grp = threadgroup_position_in_grid.x, blk = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int r0 = int(grp) * RP;                                       // this threadgroup's first fused row
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8];
    const float sc = as_type<float>(P[6]);
    const int k0 = int(blk) * BLOCK, kn = min(BLOCK, L - k0);
    const int tile = int(sg) / 2, half_d = int(sg) % 2;
    const int row = int(tid) / 4, quarter = int(tid) % 4;               // softmax: 4 threads per row, CQ keys each
    const int grow = r0 + row;                                          // the row in the kv head's fused rows
    const int last = grow < ROWS ? L - NR + grow % NR : -1;             // last visible key of the row
    // NAX lane geometry of the 16 x 32 x 16 op: left a[i] = A[fm + 8 (i >> 2)][fn + (i & 3)], right b[w * 8 + i] =
    // B[n = w * 16 + fn + (i & 3)][k = fm + 8 (i >> 2)], destination index via get_multidimensional_index
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    threadgroup float st[2][RP * C];
    threadgroup half pt[RP * C];
    threadgroup float arow[RP];
    threadgroup float kscl[C], vscl[C];
    const device half* qrow = Q + (size_t(kh) * QROWS + r0 + tile * 16) * DIM;
    const device int8_t* kt = KT + (size_t(kh) * P[4] + blk) * DIM * BLOCK;    // [dim][key in block]; P[4] = blocks allocated
    const device int8_t* vv = V + size_t(kh) * CAP * DIM;                       // [key][dim]
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (ushort i = 0; i < 16; ++i) {acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f;}
    // destination element -> (n, m) once; the same layout for S (m = row, n = key) and O (m = row, n = dim)
    short dn[16], dm[16];
    for (ushort i = 0; i < 16; ++i) {
        if (acc0.is_valid_element(i)) {const auto idx = acc0.get_multidimensional_index(i); dn[i] = short(idx[0]); dm[i] = short(idx[1]);}
        else {dn[i] = -1; dm[i] = -1;}
    }
    // this simdgroup's Q fragments: 8 k-steps over its half of the head dim
    half qa[8][8];
    for (int ks = 0; ks < 8; ++ks) for (short i = 0; i < 4; ++i) {
        const int d = (half_d * 8 + ks) * 16 + fn + i;
        qa[ks][i] = qrow[size_t(fm) * DIM + d]; qa[ks][4 + i] = qrow[size_t(fm + 8) * DIM + d];
    }
    float m = -INFINITY, s = 0.0f;                                      // the row's running max and sum (same in its 4 threads)
    for (int c0 = 0; c0 < kn; c0 += C) {
        // chunk scales to threadgroup memory (keys past the view read as 0 and are masked anyway)
        if (tid < uint(C)) {
            const int key = k0 + c0 + int(tid);
            kscl[tid] = key < L ? float(KS[size_t(kh) * CAP + key]) : 0.0f;
            vscl[tid] = key < L ? float(VS[size_t(kh) * CAP + key]) : 0.0f;
        }
        {
            // partial S tile [16 rows x 32 keys] over this half of the head dim; K^T fragment: 4 consecutive keys per load
            for (int sub = 0; sub < C / 32; ++sub) {
            auto sacc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0.0f;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int d = (half_d * 8 + ks) * 16 + fm + 8 * h;
                    // keys past the view hold finite codes of the allocated block (a whole number of chunks): masked below
                    const char4 kv = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + sub * 32 + w * 16 + fn);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(kv[i]);
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * C + sub * 32 + dn[i]] = sacc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: each thread its CQ keys, the row's max and sum over the 4 threads by fixed xor shuffles
        {
        float x[CQ];
        float lm = -INFINITY;
        for (int j = 0; j < CQ; ++j) {
            const int key = k0 + c0 + quarter * CQ + j;
            const int e = row * C + quarter * CQ + j;
            x[j] = key <= last ? (st[0][e] + st[1][e]) * kscl[quarter * CQ + j] * sc : -INFINITY;
            lm = max(lm, x[j]);
        }
        lm = max(lm, simd_shuffle_xor(lm, 1)); lm = max(lm, simd_shuffle_xor(lm, 2));
        const float mn = max(m, lm);
        const float alpha = mn == m ? 1.0f : exp(m - mn);
        float ls = 0.0f;
        for (int j = 0; j < CQ; ++j) {
            const float pr = x[j] == -INFINITY ? 0.0f : exp(x[j] - mn);
            ls += pr;
            pt[row * C + quarter * CQ + j] = half(pr * vscl[quarter * CQ + j]);
        }
        ls += simd_shuffle_xor(ls, 1); ls += simd_shuffle_xor(ls, 2);
        s = s * alpha + ls; m = mn;
        if (quarter == 0) arow[row] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rescale the accumulators of rows whose max moved
        for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
            const float a = arow[tile * 16 + dm[i]];
            if (a != 1.0f) {acc0[i] *= a; acc1[i] *= a; acc2[i] *= a; acc3[i] *= a;}
        }
        // O[16 rows x 128 dims (this half)] += P[16 x 32 keys] V[32 keys x 128]: 2 k-steps x 4 dim blocks of 32
        const threadgroup half* prow = pt + (tile * 16) * C;
        for (int ks = 0; ks < C / 16; ++ks) {
            for (short i = 0; i < 4; ++i) {ca[i] = prow[fm * C + ks * 16 + fn + i]; ca[4 + i] = prow[(fm + 8) * C + ks * 16 + fn + i];}
            for (short nb = 0; nb < 4; ++nb) {
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int kc = ks * 16 + fm + 8 * h;                                       // chunk key kc, dims d0 .. d0 + 3
                    const int d0 = half_d * HD + nb * 32 + w * 16 + fn;
                    const int key = k0 + c0 + kc;
                    const char4 vv4 = key < CAP ? *(const device char4*)(vv + size_t(key) * DIM + d0) : char4(0);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(vv4[i]);
                }
                if (nb == 0) op.run(ca, cb, acc0); else if (nb == 1) op.run(ca, cb, acc1); else if (nb == 2) op.run(ca, cb, acc2); else op.run(ca, cb, acc3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
        const int r = r0 + tile * 16 + dm[i];
        if (r >= ROWS) continue;
        const size_t base = ((size_t(kh) * ROWS + r) * NB + blk) * DIM + half_d * HD;
        O[base + dn[i]] = acc0[i]; O[base + 32 + dn[i]] = acc1[i]; O[base + 64 + dn[i]] = acc2[i]; O[base + 96 + dn[i]] = acc3[i];
    }
    if (grow < ROWS && quarter == 0) {const size_t base = (size_t(kh) * ROWS + grow) * NB + blk; M[base] = m; S[base] = s;}
)METAL", header);
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_v16_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
    // one simdgroup per (row, kv head): combine the row's blocks 0..NB-1 in order
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = P[3], NB = P[5];
    float m = -INFINITY, s = 0.0f, o[DIM / 32];
    for (int e = 0; e < DIM / 32; ++e) o[e] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t base = (size_t(kh) * ROWS + r) * NB + b;
        const float mb = M[base], sb = S[base];
        if (sb == 0.0f) continue;                         // no visible key of this row in the block
        const float mn = max(m, mb), a = m == mn ? 1.0f : exp(m - mn), c = mb == mn ? 1.0f : exp(mb - mn);
        s = s * a + sb * c;
        for (int e = 0; e < DIM / 32; ++e) o[e] = o[e] * a + O[base * DIM + e * 32 + lane] * c;
        m = mn;
    }
    const size_t out = (size_t(kh) * ROWS + r) * DIM;    // [kv head][g][t][d] = [1, H, n, d]
    for (int e = 0; e < DIM / 32; ++e) Y[out + e * 32 + lane] = o[e] / s;
)METAL");
    // the allocated block count is a runtime parameter: a template on it compiled a new kernel per cache size inside the
    // first decode step (j1152: 7.6 s at 249K)
    std::vector<int32_t> p = {length, n, group, rows, k8t.shape(1), blocks, 0, cap, qrows};
    float scf = scale; std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {9}, mx::int32);
    // at most 3 tiles (48 fused rows) per threadgroup: st + pt + the FP16 chunk stay under 32 KB (j1185: 16 rows x 6 GQA
    // heads in one threadgroup leave no room); a 16-row call runs its 6 tiles as 2 threadgroups over the same keys
    if (chunk != 32 && chunk != 64) throw std::invalid_argument("decode rows attention v16: chunk 32 or 64");
    const int tpg = std::min(tiles, 3), groups = (tiles + tpg - 1) / tpg;
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TPG", tpg}, {"CK", chunk}};
    auto part = pass1({query, k8t, v8, kscale, vscale, P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {64 * tpg * groups, blocks, kvh}, {64 * tpg, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // BF16-rounded values through FP32, as the SDPA versions return them
    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}
// v17: v16 at 32-key rounds (j1187: the fastest, bitwise = v12) with the scores' and probabilities' threadgroup rows
// padded off the 32 memory banks (stride 33 floats / 38 halves; 32 gave 8-way conflicts on the S stores and the softmax
// reads, 4-way on P). Same arithmetic as v12 for every row.
inline mlx::core::array decode_rows_attention_v17(const mlx::core::array& q, const mlx::core::array& k8t,
                                                  const mlx::core::array& v8, const mlx::core::array& kscale,
                                                  const mlx::core::array& vscale, float scale, int length) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.dtype() != mx::float32 || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode rows attention v17: q [1, heads, n <= 16, 256] FP32");
    const int heads = q.shape(1), kvh = v8.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = v8.shape(1);
    if (kvh < 1 || heads % kvh || k8t.ndim() != 4 || k8t.shape(0) != kvh || k8t.shape(1) < blocks || k8t.shape(2) != dim || k8t.shape(3) != kBlock ||
        k8t.dtype() != mx::int8 || v8.ndim() != 3 || v8.shape(2) != dim || v8.dtype() != mx::int8 || cap < length || length < 1 ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 || vscale.dtype() != mx::float16)
        throw std::invalid_argument("decode rows attention v17: int8 K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = (tiles + 2) / 3 * 3 * 16;   // whole 3-tile groups of query rows
    // [1, H, n, d] is [kv head][g][t][d]: each kv head's rows are contiguous; FP16 operands (BF16 values are exact in FP16)
    auto query = mx::reshape(mx::astype(mx::astype(q, mx::bfloat16), mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_v17_pass1", {"Q", "KT", "V", "KS", "VS", "P"}, {"M", "S", "O"}, R"METAL(
    constexpr int C = 32, SS = C + 1, PS = C + 6, RP = TPG * 16, HD = DIM / 2, CQ = C / 4, NT = TPG * 64;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint grp = threadgroup_position_in_grid.x, blk = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int r0 = int(grp) * RP;                                       // this threadgroup's first fused row
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8];
    const float sc = as_type<float>(P[6]);
    const int k0 = int(blk) * BLOCK, kn = min(BLOCK, L - k0);
    const int tile = int(sg) / 2, half_d = int(sg) % 2;
    const int row = int(tid) / 4, quarter = int(tid) % 4;               // softmax: 4 threads per row, CQ keys each
    const int grow = r0 + row;                                          // the row in the kv head's fused rows
    const int last = grow < ROWS ? L - NR + grow % NR : -1;             // last visible key of the row
    // NAX lane geometry of the 16 x 32 x 16 op: left a[i] = A[fm + 8 (i >> 2)][fn + (i & 3)], right b[w * 8 + i] =
    // B[n = w * 16 + fn + (i & 3)][k = fm + 8 (i >> 2)], destination index via get_multidimensional_index
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    threadgroup float st[2][RP * SS];   // row strides off the 32 banks (j1187 bank-conflict count: 8-way -> 2 / 1)
    threadgroup half pt[RP * PS];
    threadgroup float arow[RP];
    threadgroup float kscl[C], vscl[C];
    const device half* qrow = Q + (size_t(kh) * QROWS + r0 + tile * 16) * DIM;
    const device int8_t* kt = KT + (size_t(kh) * P[4] + blk) * DIM * BLOCK;    // [dim][key in block]; P[4] = blocks allocated
    const device int8_t* vv = V + size_t(kh) * CAP * DIM;                       // [key][dim]
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc0 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc1 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto acc3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (ushort i = 0; i < 16; ++i) {acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f;}
    // destination element -> (n, m) once; the same layout for S (m = row, n = key) and O (m = row, n = dim)
    short dn[16], dm[16];
    for (ushort i = 0; i < 16; ++i) {
        if (acc0.is_valid_element(i)) {const auto idx = acc0.get_multidimensional_index(i); dn[i] = short(idx[0]); dm[i] = short(idx[1]);}
        else {dn[i] = -1; dm[i] = -1;}
    }
    // this simdgroup's Q fragments: 8 k-steps over its half of the head dim
    half qa[8][8];
    for (int ks = 0; ks < 8; ++ks) for (short i = 0; i < 4; ++i) {
        const int d = (half_d * 8 + ks) * 16 + fn + i;
        qa[ks][i] = qrow[size_t(fm) * DIM + d]; qa[ks][4 + i] = qrow[size_t(fm + 8) * DIM + d];
    }
    float m = -INFINITY, s = 0.0f;                                      // the row's running max and sum (same in its 4 threads)
    for (int c0 = 0; c0 < kn; c0 += C) {
        // chunk scales to threadgroup memory (keys past the view read as 0 and are masked anyway)
        if (tid < uint(C)) {
            const int key = k0 + c0 + int(tid);
            kscl[tid] = key < L ? float(KS[size_t(kh) * CAP + key]) : 0.0f;
            vscl[tid] = key < L ? float(VS[size_t(kh) * CAP + key]) : 0.0f;
        }
        {
            // partial S tile [16 rows x 32 keys] over this half of the head dim; K^T fragment: 4 consecutive keys per load
            for (int sub = 0; sub < C / 32; ++sub) {
            auto sacc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0.0f;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int d = (half_d * 8 + ks) * 16 + fm + 8 * h;
                    // keys past the view hold finite codes of the allocated block (a whole number of chunks): masked below
                    const char4 kv = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + sub * 32 + w * 16 + fn);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(kv[i]);
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * SS + sub * 32 + dn[i]] = sacc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: each thread its CQ keys, the row's max and sum over the 4 threads by fixed xor shuffles
        {
        float x[CQ];
        float lm = -INFINITY;
        for (int j = 0; j < CQ; ++j) {
            const int key = k0 + c0 + quarter * CQ + j;
            const int e = row * SS + quarter * CQ + j;
            x[j] = key <= last ? (st[0][e] + st[1][e]) * kscl[quarter * CQ + j] * sc : -INFINITY;
            lm = max(lm, x[j]);
        }
        lm = max(lm, simd_shuffle_xor(lm, 1)); lm = max(lm, simd_shuffle_xor(lm, 2));
        const float mn = max(m, lm);
        const float alpha = mn == m ? 1.0f : exp(m - mn);
        float ls = 0.0f;
        for (int j = 0; j < CQ; ++j) {
            const float pr = x[j] == -INFINITY ? 0.0f : exp(x[j] - mn);
            ls += pr;
            pt[row * PS + quarter * CQ + j] = half(pr * vscl[quarter * CQ + j]);
        }
        ls += simd_shuffle_xor(ls, 1); ls += simd_shuffle_xor(ls, 2);
        s = s * alpha + ls; m = mn;
        if (quarter == 0) arow[row] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // rescale the accumulators of rows whose max moved
        for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
            const float a = arow[tile * 16 + dm[i]];
            if (a != 1.0f) {acc0[i] *= a; acc1[i] *= a; acc2[i] *= a; acc3[i] *= a;}
        }
        // O[16 rows x 128 dims (this half)] += P[16 x 32 keys] V[32 keys x 128]: 2 k-steps x 4 dim blocks of 32
        const threadgroup half* prow = pt + (tile * 16) * PS;
        for (int ks = 0; ks < C / 16; ++ks) {
            for (short i = 0; i < 4; ++i) {ca[i] = prow[fm * PS + ks * 16 + fn + i]; ca[4 + i] = prow[(fm + 8) * PS + ks * 16 + fn + i];}
            for (short nb = 0; nb < 4; ++nb) {
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int kc = ks * 16 + fm + 8 * h;                                       // chunk key kc, dims d0 .. d0 + 3
                    const int d0 = half_d * HD + nb * 32 + w * 16 + fn;
                    const int key = k0 + c0 + kc;
                    const char4 vv4 = key < CAP ? *(const device char4*)(vv + size_t(key) * DIM + d0) : char4(0);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(vv4[i]);
                }
                if (nb == 0) op.run(ca, cb, acc0); else if (nb == 1) op.run(ca, cb, acc1); else if (nb == 2) op.run(ca, cb, acc2); else op.run(ca, cb, acc3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
        const int r = r0 + tile * 16 + dm[i];
        if (r >= ROWS) continue;
        const size_t base = ((size_t(kh) * ROWS + r) * NB + blk) * DIM + half_d * HD;
        O[base + dn[i]] = acc0[i]; O[base + 32 + dn[i]] = acc1[i]; O[base + 64 + dn[i]] = acc2[i]; O[base + 96 + dn[i]] = acc3[i];
    }
    if (grow < ROWS && quarter == 0) {const size_t base = (size_t(kh) * ROWS + grow) * NB + blk; M[base] = m; S[base] = s;}
)METAL", header);
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_v17_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
    // one simdgroup per (row, kv head): combine the row's blocks 0..NB-1 in order
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = P[3], NB = P[5];
    float m = -INFINITY, s = 0.0f, o[DIM / 32];
    for (int e = 0; e < DIM / 32; ++e) o[e] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t base = (size_t(kh) * ROWS + r) * NB + b;
        const float mb = M[base], sb = S[base];
        if (sb == 0.0f) continue;                         // no visible key of this row in the block
        const float mn = max(m, mb), a = m == mn ? 1.0f : exp(m - mn), c = mb == mn ? 1.0f : exp(mb - mn);
        s = s * a + sb * c;
        for (int e = 0; e < DIM / 32; ++e) o[e] = o[e] * a + O[base * DIM + e * 32 + lane] * c;
        m = mn;
    }
    const size_t out = (size_t(kh) * ROWS + r) * DIM;    // [kv head][g][t][d] = [1, H, n, d]
    for (int e = 0; e < DIM / 32; ++e) Y[out + e * 32 + lane] = o[e] / s;
)METAL");
    // the allocated block count is a runtime parameter: a template on it compiled a new kernel per cache size inside the
    // first decode step (j1152: 7.6 s at 249K)
    std::vector<int32_t> p = {length, n, group, rows, k8t.shape(1), blocks, 0, cap, qrows};
    float scf = scale; std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {9}, mx::int32);
    // at most 3 tiles (48 fused rows) per threadgroup: st + pt + the FP16 chunk stay under 32 KB (j1185: 16 rows x 6 GQA
    // heads in one threadgroup leave no room); a 16-row call runs its 6 tiles as 2 threadgroups over the same keys
    const int tpg = std::min(tiles, 3), groups = (tiles + tpg - 1) / tpg;
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TPG", tpg}};
    auto part = pass1({query, k8t, v8, kscale, vscale, P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {64 * tpg * groups, blocks, kvh}, {64 * tpg, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // BF16-rounded values through FP32, as the SDPA versions return them
    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}

// Omega candidate slot (rung 6, docs/steering/tqllm/mac/bundle/omega_attn_iter.sh): a candidate patch replaces this
// body (and may add helpers above it); the pin forwards to v16 at 32-key rounds (the best kernel, bitwise = v12).
// Contract: q [1, heads, n <= 16, 256] FP32, the v12 int8 cache layout, returns [1, heads, n, 256] FP32 (BF16-rounded);
// a row's result must depend only on its own query and visible keys (row t sees length - n + t + 1 keys).
inline mlx::core::array decode_rows_attention_omega(const mlx::core::array& q, const mlx::core::array& k8t,
                                                    const mlx::core::array& v8, const mlx::core::array& kscale,
                                                    const mlx::core::array& vscale, float scale, int length) {
    return decode_rows_attention_v16(q, k8t, v8, kscale, vscale, scale, length, 32);
}

// Bench floor for the attention kernels: read the int8 K and V bytes of `length` keys once (16-byte loads, 1024 x 256
// threads) and fold them into a word per thread, no matrix work. Its time is what a kernel reading each byte once costs.
inline mlx::core::array stream_kv_int8(const mlx::core::array& k8t, const mlx::core::array& v8, int length) {
    namespace mx = mlx::core;
    const long words = long(v8.shape(0)) * length * v8.shape(2) / 16;
    constexpr int kGroups = 1024;
    static auto kernel = mx::fast::metal_kernel("tqllm_kv_stream_int8", {"KT", "V", "P"}, {"Y"}, R"METAL(
    const uint t = thread_position_in_threadgroup.x, g = threadgroup_position_in_grid.x;
    const long n = long(P[0]) | (long(P[1]) << 31), stride = long(P[2]) * 256;
    const device uint4* a = (const device uint4*)KT;
    const device uint4* b = (const device uint4*)V;
    uint acc = 0;
    for (long i = long(g) * 256 + t; i < n; i += stride) {const uint4 x = a[i], y = b[i]; acc += (x.x ^ x.y ^ x.z ^ x.w) + (y.x ^ y.y ^ y.z ^ y.w);}
    Y[g * 256 + t] = acc;
)METAL");
    std::vector<int32_t> p = {int32_t(words & 0x7fffffff), int32_t(words >> 31), kGroups};
    auto P = mx::array(p.data(), {3}, mx::int32);
    return kernel({k8t, v8, P}, {{kGroups * 256}}, {mx::uint32}, {kGroups * 256, 1, 1}, {256, 1, 1}, {}, std::nullopt, false,
                  mx::Device(mx::Device::gpu))[0];
}

// int8 KV for v12 from BF16 / FP32 K [1, kvh, cap, d] and V: one FP16 scale per (kv head, key) = max |x| / 127, codes
// round-to-nearest; K transposed into 1024-key blocks. Test / bench helper (the engine's cache append will quantize in place).
struct Int8KV {mlx::core::array k8t, v8, kscale, vscale;};
inline Int8KV quantize_kv_int8(const mlx::core::array& k, const mlx::core::array& v) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024;
    const int kvh = k.shape(1), cap = k.shape(2), dim = k.shape(3), blocks = (cap + kBlock - 1) / kBlock;
    auto quant = [&](const mx::array& x) {
        auto xf = mx::astype(mx::reshape(x, {kvh, cap, dim}), mx::float32);
        auto amax = mx::max(mx::abs(xf), -1, true);                                                   // [kvh, cap, 1]
        auto scale16 = mx::astype(mx::maximum(mx::divide(amax, mx::array(127.0f)), mx::array(1e-8f)), mx::float16);
        auto scale = mx::astype(scale16, mx::float32);
        auto codes = mx::astype(mx::clip(mx::round(mx::divide(xf, scale)), mx::array(-127.0f), mx::array(127.0f)), mx::int8);
        return std::pair{codes, mx::reshape(scale16, {kvh, cap})};
    };
    auto [k8, ks] = quant(k);
    auto [v8, vs] = quant(v);
    // K^T blocks: [kvh, blocks, dim, 1024] (zero padded past cap)
    auto kpad = blocks * kBlock > cap ? mx::concatenate({k8, mx::zeros({kvh, blocks * kBlock - cap, dim}, mx::int8)}, 1) : k8;
    auto k8t = mx::contiguous(mx::transpose(mx::reshape(kpad, {kvh, blocks, kBlock, dim}), {0, 1, 3, 2}));
    return {k8t, mx::contiguous(v8), mx::contiguous(ks), mx::contiguous(vs)};
}
// the cache append's int8 rows in one dispatch (the MLX-op chain is ~12 ops each for K and V): K / V [1, kvh, m, 256]
// (any float type) -> K^T codes [kvh, 256, m], V codes [kvh, m, 256], FP16 scales [kvh, m]; the same values as
// quantize_kv_int8 (max |x| / 127 rounded to FP16, codes x / scale round half to even, clipped to +-127)
struct Int8Rows {mlx::core::array kt, v, kscale, vscale;};
inline Int8Rows quantize_rows_int8(const mlx::core::array& k, const mlx::core::array& v) {
    namespace mx = mlx::core;
    if (k.ndim() != 4 || k.shape(0) != 1 || k.shape(3) != 256 || v.shape() != k.shape())
        throw std::invalid_argument("quantize_rows_int8: K / V [1, kvh, m, 256]");
    const int kvh = k.shape(1), m = k.shape(2);
    static auto kernel = mx::fast::metal_kernel("tqllm_int8_rows", {"K", "V", "P"}, {"KT", "VO", "KS", "VS"}, R"METAL(
    const uint t = thread_position_in_threadgroup.x, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint M = uint(P[0]), g = threadgroup_position_in_grid.y;   // (K | V, kv head, row)
    const uint row = g % M, kh = (g / M) % KVH, which = g / (M * KVH);
    const size_t off = (size_t(kh) * M + row) * 256 + t;
    const float x = which == 0 ? float(K[off]) : float(V[off]);
    threadgroup float red[8];
    const float a = simd_max(fabs(x));
    if (lane == 0) red[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = red[0];
    for (uint i = 1; i < 8; ++i) amax = max(amax, red[i]);
    const half s16 = half(max(metal::precise::divide(amax, 127.0f), 1e-8f));
    const int8_t c = int8_t(clamp(metal::rint(metal::precise::divide(x, float(s16))), -127.0f, 127.0f));
    if (which == 0) {KT[(size_t(kh) * 256 + t) * M + row] = c; if (t == 0) KS[size_t(kh) * M + row] = s16;}
    else {VO[off] = c; if (t == 0) VS[size_t(kh) * M + row] = s16;}
)METAL");
    const auto p = mx::array({m}, {1}, mx::int32);
    auto out = kernel({k, v, p}, {{kvh, 256, m}, {kvh, m, 256}, {kvh, m}, {kvh, m}}, {mx::int8, mx::int8, mx::float16, mx::float16},
                      {256, 2 * kvh * m, 1}, {256, 1, 1}, {{"KVH", kvh}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return {out[0], out[1], out[2], out[3]};
}
// v13 (research, bench only): int8 K / V packed per 32-key chunk in the matrix unit's lane order, so each lane loads its
// MMA operand fragment with one 16-byte load (the pattern that puts the weight kernels at the DRAM rate); one simdgroup
// owns a 16-row tile end to end: S in its registers, the row softmax by two shuffles, P fed straight into the PV operand.
// No threadgroup memory, no barriers in the key loop (v12's per-32-key chain S -> barrier -> softmax -> barrier -> PV set
// its rate, j1085). Two simdgroups per tile compute S twice and each accumulates half of the head dim.
// Lane geometry of the 16 x 32 x 16 op (nax_affine.cpp): fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4,
// left a[i] = A[fm + 8 (i >> 2)][fn + (i & 3)], right b[w * 8 + i] = B[n = w * 16 + fn + (i & 3)][k = fm + 8 (i >> 2)],
// destination c[w * 8 + i] = C[m = fm + 8 (i >> 2)][n = w * 16 + fn + (i & 3)] (checked by nax_mma_layout_probe).
struct Int8KVNax {mlx::core::array kn, vn, kscale, vscale;};
// lane-order tables: entry lane * 16 + j -> (key within the chunk / step, dim within the step / block)
inline std::pair<std::vector<int32_t>, std::vector<int32_t>> nax_lane_tables() {
    std::vector<int32_t> kfrag(512), vfrag(512);
    for (int lane = 0; lane < 32; ++lane) {
        const int qid = lane >> 2, fm = (qid & 4) | ((lane >> 1) & 3), fn = ((qid & 2) | (lane & 1)) * 4;
        for (int j = 0; j < 16; ++j) {
            const int i = j & 7, w = j >> 3;
            kfrag[lane * 16 + j] = (w * 16 + fn + (i & 3)) * 16 + fm + 8 * (i >> 2);   // K^T: n = key (32), k = dim (16)
            vfrag[lane * 16 + j] = (fm + 8 * (i >> 2)) * 32 + w * 16 + fn + (i & 3);   // V: n = dim (32), k = key (16)
        }
    }
    return {kfrag, vfrag};
}
// from the v12 cache (K^T 1024-key blocks, V rows, FP16 per-key scales): KN [kvh, chunks, 16 dim-steps, 32 lanes, 16],
// VN [kvh, chunks, 2 key-steps, 8 dim-blocks, 32 lanes, 16], chunks = cap / 32 (zero padded)
inline Int8KVNax pack_kv_int8_nax(const Int8KV& kv) {
    namespace mx = mlx::core;
    const int kvh = kv.v8.shape(0), cap = kv.v8.shape(1), dim = kv.v8.shape(2), chunks = (cap + 31) / 32, cap32 = chunks * 32;
    if (dim != 256) throw std::invalid_argument("pack_kv_int8_nax: head dim 256");
    auto k = mx::reshape(mx::transpose(kv.k8t, {0, 1, 3, 2}), {kvh, -1, dim});                       // [kvh, blocks * 1024, dim]
    k = mx::slice(k, {0, 0, 0}, {kvh, cap32 <= k.shape(1) ? cap32 : k.shape(1), dim});
    auto v = kv.v8;
    auto pad = [&](mx::array x) {
        if (x.shape(1) < cap32) x = mx::concatenate({x, mx::zeros({kvh, cap32 - x.shape(1), dim}, mx::int8)}, 1);
        return x;
    };
    k = pad(k); v = pad(v);
    const auto [kfrag, vfrag] = nax_lane_tables();
    auto ki = mx::array(kfrag.data(), {512}, mx::int32), vi = mx::array(vfrag.data(), {512}, mx::int32);
    // K: [kvh, chunks, 32 keys, 16 steps, 16 dims] -> [kvh, chunks, steps, keys, dims] -> [.., 512] -> lane order
    auto kt = mx::reshape(mx::transpose(mx::reshape(k, {kvh, chunks, 32, 16, 16}), {0, 1, 3, 2, 4}), {kvh, chunks, 16, 512});
    auto kn = mx::reshape(mx::take(kt, ki, 3), {kvh, chunks, 16, 32, 16});
    // V: [kvh, chunks, 2 steps, 16 keys, 8 blocks, 32 dims] -> [.., steps, blocks, keys, dims] -> [.., 512] -> lane order
    auto vt = mx::reshape(mx::transpose(mx::reshape(v, {kvh, chunks, 2, 16, 8, 32}), {0, 1, 2, 4, 3, 5}), {kvh, chunks, 2, 8, 512});
    auto vn = mx::reshape(mx::take(vt, vi, 4), {kvh, chunks, 2, 8, 32, 16});
    auto sc = [&](const mx::array& x) {
        auto y = x;
        if (y.shape(1) < cap32) y = mx::concatenate({y, mx::zeros({kvh, cap32 - y.shape(1)}, mx::float16)}, 1);
        return mx::contiguous(y);
    };
    return {mx::contiguous(kn), mx::contiguous(vn), sc(kv.kscale), sc(kv.vscale)};
}
// the destination layout of the 16 x 32 x 16 op: entry lane * 16 + j = (idx0 << 8) | idx1 of get_multidimensional_index
inline mlx::core::array nax_mma_layout_probe() {
    namespace mx = mlx::core;
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_mma_layout_probe", {"X"}, {"Y"}, R"METAL(
    const uint lane = thread_index_in_simdgroup;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (ushort i = 0; i < 16; ++i) {
        int code = -1;
        if (cc.is_valid_element(i)) {const auto idx = cc.get_multidimensional_index(i); code = (int(idx[0]) << 8) | int(idx[1]);}
        Y[lane * 16 + i] = code + int(X[0]) * 0;
    }
)METAL", R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL");
    return kernel({mx::array({0}, {1}, mx::int32)}, {{512}}, {mx::int32}, {32, 1, 1}, {32, 1, 1},
                  std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
// sg1: one simdgroup per tile holds all eight output accumulators (no second S); else two simdgroups per tile, half the
// head dim each
inline mlx::core::array decode_rows_attention_v13(const mlx::core::array& q, const mlx::core::array& kn, const mlx::core::array& vn,
                                                  const mlx::core::array& kscale, const mlx::core::array& vscale, float scale, int valid,
                                                  bool sg1 = false, int qks = 1) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024;
    const int heads = q.shape(1), kvh = kn.shape(0), group = heads / kvh, n = q.shape(2), dim = q.shape(3);
    const int chunks = kn.shape(1), cap = chunks * 32, length = valid > 0 ? valid : cap, blocks = (length + kBlock - 1) / kBlock;
    if (q.ndim() != 4 || q.shape(0) != 1 || dim != 256 || n < 1 || n > 16 || kn.dtype() != mx::int8 || vn.dtype() != mx::int8 ||
        kn.shape() != mx::Shape{kvh, chunks, 16, 32, 16} || vn.shape() != mx::Shape{kvh, chunks, 2, 8, 32, 16} ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 ||
        length < n || length > cap || blocks * kBlock > cap + kBlock - 1)
        throw std::invalid_argument("decode rows attention v13: KN [kvh, chunks, 16, 32, 16], VN [kvh, chunks, 2, 8, 32, 16], FP16 scales [kvh, cap]");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = tiles * 16;
    auto query = mx::reshape(mx::astype(mx::astype(q, mx::bfloat16), mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_v13_pass1", {"Q", "KN", "VN", "KS", "VS", "P"}, {"M", "S", "O"}, R"METAL(
    constexpr int C = 32, HD = DIM / 2, CPB = BLOCK / C;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint blk = threadgroup_position_in_grid.y, tz = threadgroup_position_in_grid.z;
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8], CHUNKS = P[9];
    const float sc = as_type<float>(P[6]);
    const int tile = int(tz) % TILES, kh = int(tz) / TILES, half_d = SG1 ? 0 : int(sg);
    constexpr int DB = SG1 ? 8 : 4;   // output dim blocks of 32 per simdgroup
    const int k0 = int(blk) * BLOCK, kn_ = min(BLOCK, L - k0), nch = (kn_ + C - 1) / C;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    // this lane's two rows of the tile and their last visible key
    const int ra = tile * 16 + fm, rb = ra + 8;
    const int last_a = ra < ROWS ? L - NR + ra % NR : -1, last_b = rb < ROWS ? L - NR + rb % NR : -1;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ss = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    // QKS independent S accumulators over the 16 dim steps (the chain of 16 dependent MMAs into one was the rate, j1170),
    // summed in a fixed order
    auto s1 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto s2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto s3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o0 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o1 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o4 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o5 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o6 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto o7 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (ushort i = 0; i < 16; ++i) {o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; o4[i] = 0.0f; o5[i] = 0.0f; o6[i] = 0.0f; o7[i] = 0.0f;}
    const device half* qa = Q + (size_t(kh) * QROWS + tile * 16 + fm) * DIM + fn;
    const device half* qb = qa + 8 * DIM;
    const device uchar4* kbase = (const device uchar4*)KN + (size_t(kh) * CHUNKS) * 16 * 32 * 4 + lane * 4;
    const device uchar4* vbase = (const device uchar4*)VN + (size_t(kh) * CHUNKS) * 2 * 8 * 32 * 4 + lane * 4;
    const device half* ks_ = KS + size_t(kh) * CAP + k0;
    const device half* vs_ = VS + size_t(kh) * CAP + k0;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    for (int c = 0; c < nch; ++c) {
        const int chunk = (k0 + c * C) / C;
        // S = Q K^T over the 16 dim steps; K fragment: 16 codes per lane per step, one load
        for (ushort i = 0; i < 16; ++i) {ss[i] = 0.0f; if (QKS > 1) s1[i] = 0.0f; if (QKS > 2) {s2[i] = 0.0f; s3[i] = 0.0f;}}
        const device uchar4* kc = kbase + size_t(chunk) * 16 * 32 * 4;
        for (int st = 0; st < 16; ++st) {
            const half4 a0 = *(const device half4*)(qa + st * 16), a1 = *(const device half4*)(qb + st * 16);
            for (short i = 0; i < 4; ++i) {ca[i] = a0[i]; ca[4 + i] = a1[i];}
            const device uchar4* kf = kc + st * 32 * 4;
            for (short u = 0; u < 4; ++u) {const uchar4 w = kf[u]; for (short i = 0; i < 4; ++i) cb[u * 4 + i] = half(char(w[i]));}
            if (QKS == 1 || st % QKS == 0) op.run(ca, cb, ss); else if (QKS == 2 || st % QKS == 1) op.run(ca, cb, s1);
            else if (st % QKS == 2) op.run(ca, cb, s2); else op.run(ca, cb, s3);
        }
        if (QKS > 1) for (ushort i = 0; i < 16; ++i) {float t = ss[i] + s1[i]; if (QKS > 2) {t += s2[i]; t += s3[i];} ss[i] = t;}
        // per-key K scale, visibility, the running row max over the chunk (8 keys per lane per row, then the 3 other lanes)
        float p[16];
        float ca_max = -INFINITY, cb_max = -INFINITY;
        for (short j = 0; j < 16; ++j) {
            const int key = c * C + (j >> 3) * 16 + fn + (j & 3);                         // within the block
            const bool rowb = (j >> 2) & 1;
            const bool vis = key < kn_ && k0 + key <= (rowb ? last_b : last_a);
            const float x = vis ? ss[j] * float(ks_[key]) * sc : -INFINITY;
            p[j] = x;
            if (rowb) cb_max = max(cb_max, x); else ca_max = max(ca_max, x);
        }
        ca_max = max(ca_max, simd_shuffle_xor(ca_max, 1)); ca_max = max(ca_max, simd_shuffle_xor(ca_max, 8));
        cb_max = max(cb_max, simd_shuffle_xor(cb_max, 1)); cb_max = max(cb_max, simd_shuffle_xor(cb_max, 8));
        const float na = max(ma, ca_max), nb = max(mb, cb_max);
        const float aa = na == ma ? 1.0f : exp(ma - na), ab = nb == mb ? 1.0f : exp(mb - nb);
        float sa = 0.0f, sb = 0.0f;
        for (short j = 0; j < 16; ++j) {
            const bool rowb = (j >> 2) & 1;
            const float e = p[j] == -INFINITY ? 0.0f : exp(p[j] - (rowb ? nb : na));
            p[j] = e;
            if (rowb) sb += e; else sa += e;
        }
        sa += simd_shuffle_xor(sa, 1); sa += simd_shuffle_xor(sa, 8);
        sb += simd_shuffle_xor(sb, 1); sb += simd_shuffle_xor(sb, 8);
        la = la * aa + sa; lb = lb * ab + sb; ma = na; mb = nb;
        if (aa != 1.0f || ab != 1.0f) for (ushort i = 0; i < 16; ++i) {
            const float a = ((i >> 2) & 1) ? ab : aa;
            o0[i] *= a; o1[i] *= a; o2[i] *= a; o3[i] *= a;
            if (SG1) {o4[i] *= a; o5[i] *= a; o6[i] *= a; o7[i] *= a;}
        }
        // O += P V: P[row][key] from the same lane's S elements (w = key step), V scale folded into P
        const device uchar4* vc = vbase + size_t(chunk) * 2 * 8 * 32 * 4;
        for (short w = 0; w < 2; ++w) {
            for (short i = 0; i < 8; ++i) {
                const int key = c * C + w * 16 + fn + (i & 3);
                ca[i] = half(p[w * 8 + i] * (key < kn_ ? float(vs_[key]) : 0.0f));
            }
            for (short db = 0; db < DB; ++db) {
                const device uchar4* vf = vc + (w * 8 + half_d * 4 + db) * 32 * 4;
                for (short u = 0; u < 4; ++u) {const uchar4 x = vf[u]; for (short i = 0; i < 4; ++i) cb[u * 4 + i] = half(char(x[i]));}
                if (db == 0) op.run(ca, cb, o0); else if (db == 1) op.run(ca, cb, o1); else if (db == 2) op.run(ca, cb, o2); else if (db == 3) op.run(ca, cb, o3);
                else if (db == 4) op.run(ca, cb, o4); else if (db == 5) op.run(ca, cb, o5); else if (db == 6) op.run(ca, cb, o6); else op.run(ca, cb, o7);
            }
        }
    }
    // partials: O rows [kh][row][block][dim], M / S per (row, block)
    for (ushort j = 0; j < 16; ++j) {
        const bool rowb = (j >> 2) & 1;
        const int r = rowb ? rb : ra;
        if (r >= ROWS) continue;
        const int d = half_d * HD + (j >> 3) * 16 + fn + (j & 3);
        const size_t base = ((size_t(kh) * ROWS + r) * NB + blk) * DIM;
        O[base + d] = o0[j]; O[base + 32 + d] = o1[j]; O[base + 64 + d] = o2[j]; O[base + 96 + d] = o3[j];
        if (SG1) {O[base + 128 + d] = o4[j]; O[base + 160 + d] = o5[j]; O[base + 192 + d] = o6[j]; O[base + 224 + d] = o7[j];}
    }
    if (half_d == 0 && fn == 0) {
        if (ra < ROWS) {const size_t b = (size_t(kh) * ROWS + ra) * NB + blk; M[b] = ma; S[b] = la;}
        if (rb < ROWS) {const size_t b = (size_t(kh) * ROWS + rb) * NB + blk; M[b] = mb; S[b] = lb;}
    }
)METAL", header);
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_v13_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = P[3], NB = P[5];
    float m = -INFINITY, s = 0.0f, o[DIM / 32];
    for (int e = 0; e < DIM / 32; ++e) o[e] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t base = (size_t(kh) * ROWS + r) * NB + b;
        const float mb = M[base], sb = S[base];
        if (sb == 0.0f) continue;
        const float mn = max(m, mb), a = m == mn ? 1.0f : exp(m - mn), c = mb == mn ? 1.0f : exp(mb - mn);
        s = s * a + sb * c;
        for (int e = 0; e < DIM / 32; ++e) o[e] = o[e] * a + O[base * DIM + e * 32 + lane] * c;
        m = mn;
    }
    const size_t out = (size_t(kh) * ROWS + r) * DIM;
    for (int e = 0; e < DIM / 32; ++e) Y[out + e * 32 + lane] = o[e] / s;
)METAL");
    std::vector<int32_t> p = {length, n, group, rows, 0, blocks, 0, cap, qrows, chunks};
    float scf = scale; std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {10}, mx::int32);
    if (qks != 1 && qks != 2 && qks != 4) throw std::invalid_argument("decode rows attention v13: qks 1 / 2 / 4");
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TILES", tiles}, {"SG1", sg1 ? 1 : 0}, {"QKS", qks}};
    const int tg = sg1 ? 32 : 64;
    auto part = pass1({query, kn, vn, kscale, vscale, P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {tg, blocks, kvh * tiles}, {tg, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}
inline mlx::core::array decode_rows_attention(const mlx::core::array& q, const mlx::core::array& k,
                                              const mlx::core::array& v, float scale, int valid) {
    if (k.dtype() != mlx::core::bfloat16) throw std::invalid_argument("decode rows attention reads a BF16 KV cache");
#ifdef MLX_HAS_SDPA_LSE
    if (attention_v11_available(q.shape(3))) {
        // inside one segment the split gains nothing (v11 = v7 + merge, j1030: 98 vs 32 us at 64 keys): rows that see at
        // most one segment run v7, each row on the kernel its AR call gets
        namespace mx = mlx::core;
        const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), length = valid > 0 ? valid : k.shape(2);
        constexpr int seg = kAttentionSegment;
        if (length <= seg) return decode_rows_attention_v7(q, k, v, scale, length);
        if (length - n + 1 <= seg) {
            const int below = seg - (length - n);
            auto lo = decode_rows_attention_v7(mx::slice(q, {0, 0, 0, 0}, {1, heads, below, dim}), k, v, scale, seg);
            auto hi = decode_rows_attention_v11(mx::slice(q, {0, 0, below, 0}, {1, heads, n, dim}), k, v, scale, length);
            return mx::concatenate({lo, hi}, 2);
        }
        return decode_rows_attention_v11(q, k, v, scale, valid);
    }
#endif
    return decode_rows_attention_v7(q, k, v, scale, valid);
}
}  // namespace tqllm::mlx_backend

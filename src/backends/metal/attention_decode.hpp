#pragma once
#include <array>
#include <functional>
#include <optional>
#include <vector>
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

#include "attention_split.hpp"
#include "shape_constants.hpp"

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

// Query rows per decode / verify attention call: 8 for up to 8 rows (AR and a 7-token verify), else 16. A row's result
// does not depend on the shape (test-mlx-decode-rows DECODE_ROWS_SHAPES, bitwise), so a verify width may change per step;
// MLX's fused attention is compute-bound in the folded rows (j990), so 8 rows halve the attention work at long context.
inline int decode_row_shape(int n) {return n <= 8 ? 8 : 16;}

// Token-tree verify: block row t sees every key before the block and the block keys of its ancestors and itself
// (visible [n, nb] bool: query row t, block key j; nb = n, or the whole block when a 17-32 row verify runs its rows as
// two calls). Rows padded at the front see what row 0 sees (their outputs are dropped). With a chain (visible lower
// triangular) every real row's mask equals the causal one, so a chain verifies bit for bit as before.
inline mlx::core::array tree_key_mask(const mlx::core::array& visible, int rows, int n, int length, int total) {
    namespace mx = mlx::core;
    const int nb = visible.shape(1);
    auto block = visible;
    if (rows > n) block = mx::concatenate({mx::broadcast_to(mx::slice(visible, {0, 0}, {1, nb}), {rows - n, nb}), visible}, 0);
    std::vector<mx::array> parts{mx::full({rows, length - nb}, mx::array(true)), block};
    if (total > length) parts.push_back(mx::full({rows, total - length}, mx::array(false)));
    return mx::concatenate(parts, 1);
}

// Every attention layer of one tree verify asks for the same mask (same rows, keys and ancestor table), so it is built
// on the first layer and reused by the others instead of once per layer. The entry holds the ancestor table it was built
// from, so its id cannot be reused by a later table while it is cached; a new table replaces the entries.
// Entries made from an entry (the row halves of a 17-32 row verify) belong to the same verify. Entries hold the arrays
// they were made from, so no id is reused while cached.
inline mlx::core::array cached_tree_mask(const mlx::core::array& visible, const std::array<int, 7>& shape,
                                         const std::function<mlx::core::array()>& make) {
    struct Entry {mlx::core::array visible, mask; std::array<int, 7> shape;};
    struct Cache {std::optional<mlx::core::array> root; std::vector<Entry> entries;};
    // never destroyed: an array must not outlive MLX's allocator at exit
    static thread_local auto* cache = new Cache();
    bool known = cache->root && cache->root->id() == visible.id();
    for (const auto& e : cache->entries) known = known || e.mask.id() == visible.id();
    if (!known) {cache->entries.clear(); cache->root = visible;}
    for (const auto& e : cache->entries) if (e.visible.id() == visible.id() && e.shape == shape) return e.mask;
    cache->entries.push_back({visible, make(), shape});
    return cache->entries.back().mask;
}

// bf16_folded: GQA folded into query rows, so each KV head's keys and values are read once for its G query heads: q [1,
// H, n, d] -> [1, KH, G * R, d] (rows padded at the front to R = decode_row_shape(n) per head), one MLX SDPA call per
// layer with an explicit per-row causal mask (row r sees keys <= L - R + r % R). Rows stay independent of each other,
// and AR uses the same R-row shape, so a row agrees bit for bit across row counts.
inline mlx::core::array decode_attention_bf16_folded(const mlx::core::array& q, const mlx::core::array& k,
                                                 const mlx::core::array& v, float scale, int valid = -1,
                                                 const mlx::core::array* tree = nullptr) {
    namespace mx = mlx::core;
    check_decode_rows_shape(q, k, v, valid);
    const int heads = q.shape(1), kvh = k.shape(1), group = heads / kvh, n = q.shape(2), dim = q.shape(3);
    const int shape = decode_row_shape(n);
    const int length = valid > 0 ? valid : k.shape(2), rows = std::min(shape, length);
    constexpr int kSplit = 4096;  // rows <= 4096 keys run as their own call
    if (!tree && length > kSplit && length - n + 1 <= kSplit) {
        const int below = kSplit - (length - n);
        auto lo = decode_attention_bf16_folded(mx::slice(q, {0, 0, 0, 0}, {1, heads, below, dim}), k, v, scale, kSplit);
        auto hi = decode_attention_bf16_folded(mx::slice(q, {0, 0, below, 0}, {1, heads, n, dim}), k, v, scale, length);
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
    auto mask = tree ? cached_tree_mask(*tree, {0, rows, n, length, length, group, 0},
                                        [&] {return mx::tile(tree_key_mask(*tree, rows, n, length, length), {group, 1});})
                     : shape_constant("folded_mask", {rows, length, group}, [&] {   // folded row r sees keys <= length - rows + r % rows
                           auto visible = mx::add(mx::remainder(mx::arange(group * rows), mx::array(rows)), mx::array(length - rows));
                           return mx::less_equal(mx::reshape(mx::arange(length), {1, length}), mx::reshape(visible, {group * rows, 1}));
                       });
    auto out = mx::fast::scaled_dot_product_attention(query, keys, values, scale, "array", mask);
    out = mx::reshape(out, {1, heads, rows, dim});
    return mx::astype(mx::slice(out, {0, 0, rows - n, 0}, {1, heads, rows, dim}), mx::float32);
}

constexpr int kAttentionSegment = 1024;  // the KV cache capacity is a multiple of it (qwen35.cpp)
#ifdef MLX_HAS_SDPA_LSE
// bf16_segmented: bf16_folded split over fixed 1024-key segments counted from key 0, all segments as the batch of one
// MLX fused attention call (our MLX build's scaled_dot_product_attention_lse: the M5 matrix-unit kernel plus each row's
// log-sum-exp), merged per row in segment order. bf16_folded runs one call over the whole cache: (folded rows / 64-row
// query tile) x 4 kv heads threadgroups, 4 at the 8-row shape (j991: 8 rows only -26 % vs 16 at 64K for half the work),
// here that many per segment. Rows are padded at the front to decode_row_shape(n) as in bf16_folded; segments a row
// cannot see return -inf and are skipped, so a row's result does not depend on how many rows share the call. K / V are
// strided views of the cache (its capacity is a multiple of the segment, zero past the filled rows); a shorter buffer
// (tests) is zero padded. Needs the matrix unit (M5): elsewhere decode_attention_bf16 keeps bf16_folded.
// Side branches of a token tree (host parents, row 0 = the anchor): the main chain is rows 0, 1, ... while each row's parent
// is the row before it; every other row belongs to the first leaf path (in row order) that contains it.
struct TreeBranch {std::vector<int32_t> path; std::vector<int32_t> rows, depths;};  // path: block rows in depth order
inline std::vector<TreeBranch> tree_side_branches(const std::vector<int32_t>& parents) {
    const int n = int(parents.size());
    std::vector<char> on_chain(size_t(n), 0), has_child(size_t(n), 0), taken(size_t(n), 0);
    for (int t = 0; t < n; ++t) {
        on_chain[size_t(t)] = t == 0 || (parents[size_t(t)] == t - 1 && on_chain[size_t(t - 1)]);
        if (parents[size_t(t)] >= 0) has_child[size_t(parents[size_t(t)])] = 1;
    }
    std::vector<TreeBranch> out;
    for (int leaf = 0; leaf < n; ++leaf) {
        if (has_child[size_t(leaf)] || on_chain[size_t(leaf)]) continue;
        TreeBranch b;
        for (int a = leaf; a >= 0; a = parents[size_t(a)]) b.path.insert(b.path.begin(), a);
        for (size_t d = 0; d < b.path.size(); ++d) {
            const int row = b.path[d];
            if (!on_chain[size_t(row)] && !taken[size_t(row)]) {taken[size_t(row)] = 1; b.rows.push_back(row); b.depths.push_back(int32_t(d));}
        }
        if (!b.rows.empty()) out.push_back(std::move(b));
    }
    return out;
}

inline mlx::core::array decode_attention_bf16_segmented(const mlx::core::array& q, const mlx::core::array& k,
                                                  const mlx::core::array& v, float scale, int valid = -1,
                                                  int seg = kAttentionSegment, const mlx::core::array* tree = nullptr,
                                                  const std::vector<int32_t>* tree_parents = nullptr,
                                                  int block_rows = 0, int block_offset = 0) {
    namespace mx = mlx::core;
    check_decode_rows_shape(q, k, v, valid);
    if (k.dtype() != mx::bfloat16) throw std::invalid_argument("decode attention bf16_segmented reads a BF16 KV cache");
    if (seg < 64 || seg % kAttentionSegment) throw std::invalid_argument("decode attention bf16_segmented segment must be a multiple of the cache granule");
    const int heads = q.shape(1), kvh = k.shape(1), group = heads / kvh, n = q.shape(2), dim = q.shape(3);
    const int length = valid > 0 ? valid : k.shape(2), rows = std::min(decode_row_shape(n), length);
    const int segments = (length + seg - 1) / seg;
    const auto source_keys = k, source_values = v;
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
    auto key_index = shape_constant("segment_keys", {segments, seg}, [&] {return mx::reshape(mx::arange(segments * seg), {segments, 1, 1, seg});});
    auto mask = shape_constant("segment_mask", {segments, seg, rows, length, group}, [&] {
        auto last = mx::add(mx::remainder(mx::arange(group * rows), mx::array(rows)), mx::array(length - rows));
        return mx::less_equal(key_index, mx::reshape(last, {1, 1, group * rows, 1}));
    });
    if (tree)
        mask = cached_tree_mask(*tree, {1, rows, n, length, segments * seg, group, seg}, [&] {
            return mx::reshape(mx::transpose(mx::reshape(mx::tile(tree_key_mask(*tree, rows, n, length, segments * seg), {group, 1}),
                                                         {group * rows, segments, seg}), {1, 0, 2}), {segments, 1, group * rows, seg});
        });
    // A token tree's side-branch rows must see their ancestors' keys where their AR step has them (prefix, then the path
    // in depth order): the matrix-unit kernel's rounding depends on where a visible key sits in its tiles, so a hole
    // before a key changes the result (mlx_tree_kernels). Those rows take only the full segments below the prefix's last
    // segment boundary from this call, and their tail segments from a call on the prefix tail followed by their branch's
    // path keys; the merge folds each row's visible segments in key order, as for AR.
    const auto branches = tree && tree_parents ? tree_side_branches(*tree_parents) : std::vector<TreeBranch>{};
    // a 17-32 row verify's halves: `length` counts every key of the block, and the hi half's rows are block rows 16.. of
    // a 32-row block, so neither the prefix nor a row's block index follows from this call's own row count
    const int block = block_rows > 0 ? block_rows : n;
    const int prefix = length - block, shared = prefix / seg * seg;
    std::vector<int32_t> row_of_slot(size_t(group * rows), -1);  // folded row -> block row (-1: front padding)
    for (int r = 0; r < group * rows; ++r) {const int t = r % rows - (rows - n) + block_offset; row_of_slot[size_t(r)] = t;}
    if (!branches.empty()) {
        std::vector<int32_t> limit(size_t(group * rows), segments * seg);
        for (const auto& b : branches)
            for (int r = 0; r < group * rows; ++r)
                if (std::find(b.rows.begin(), b.rows.end(), row_of_slot[size_t(r)]) != b.rows.end()) limit[size_t(r)] = shared;
        mask = mx::logical_and(mask, mx::less(key_index, mx::reshape(mx::array(limit.data(), {group * rows}, mx::int32), {1, 1, group * rows, 1})));
    }
    auto parts = mx::fast::scaled_dot_product_attention_lse(query, keys, values, scale, mask);
    std::vector<mx::array> outputs{parts[0]}, lses{parts[1]};
    int total = segments;
    for (const auto& b : branches) {
        const int tail = prefix - shared + int(b.path.size()), count = (tail + seg - 1) / seg;
        std::vector<int32_t> picks;
        for (int key = shared; key < prefix; ++key) picks.push_back(key);
        for (int row : b.path) picks.push_back(prefix + row);
        const auto index = mx::array(picks.data(), {int(picks.size())}, mx::int32);
        auto tk = mx::take(source_keys, index, 2), tv = mx::take(source_values, index, 2);
        if (tail < count * seg) {
            tk = mx::concatenate({tk, mx::zeros({1, kvh, count * seg - tail, dim}, tk.dtype())}, 2);
            tv = mx::concatenate({tv, mx::zeros({1, kvh, count * seg - tail, dim}, tv.dtype())}, 2);
        }
        tk = mx::reshape(mx::transpose(mx::reshape(tk, {kvh, count, seg, dim}), {1, 0, 2, 3}), {count, kvh, seg, dim});
        tv = mx::reshape(mx::transpose(mx::reshape(tv, {kvh, count, seg, dim}), {1, 0, 2, 3}), {count, kvh, seg, dim});
        // a branch row at depth d sees tail keys <= (prefix - shared) + d; every other row sees none of this call
        std::vector<int32_t> last_key(size_t(group * rows), -1);
        for (int r = 0; r < group * rows; ++r)
            for (size_t j = 0; j < b.rows.size(); ++j)
                if (b.rows[j] == row_of_slot[size_t(r)]) last_key[size_t(r)] = prefix - shared + b.depths[j];
        auto tail_index = mx::reshape(mx::arange(count * seg), {count, 1, 1, seg});
        auto tail_mask = mx::less_equal(tail_index, mx::reshape(mx::array(last_key.data(), {group * rows}, mx::int32), {1, 1, group * rows, 1}));
        auto branch_query = mx::broadcast_to(mx::slice(query, {0, 0, 0, 0}, {1, kvh, group * rows, dim}), {count, kvh, group * rows, dim});
        auto bp = mx::fast::scaled_dot_product_attention_lse(branch_query, tk, tv, scale, tail_mask);
        outputs.push_back(bp[0]); lses.push_back(bp[1]); total += count;
    }
    if (total > segments) parts = {mx::concatenate(outputs, 0), mx::concatenate(lses, 0)};
    static auto merge = mx::fast::metal_kernel("tqllm_attn_bf16_segment_merge", {"O", "LSE", "P"}, {"Y"}, R"METAL(
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
    std::vector<int32_t> p = {total, kvh * group * rows};
    auto P = mx::array(p.data(), {2}, mx::int32);
    auto y = merge({parts[0], parts[1], P}, {{1, kvh, group * rows, dim}}, {mx::float32}, {dim, kvh * group * rows, 1}, {std::min(dim, 256), 1, 1},
                   {{"DIM", dim}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    y = mx::reshape(y, {1, heads, rows, dim});
    return mx::astype(mx::astype(mx::slice(y, {0, 0, rows - n, 0}, {1, heads, rows, dim}), mx::bfloat16), mx::float32);
}

// bf16_segmented needs our MLX build's log-sum-exp attention on the matrix unit, which MLX has for some head dims only
// (256 / 512 on 0.32.3); probed once per head dim (false: bf16_folded)
inline bool segmented_attention_available(int dim = 256) {
    namespace mx = mlx::core;
    auto probe = [](int d) {
        try {
            auto q = mx::zeros({1, 24, 1, d}, mx::float32), k = mx::zeros({1, 4, 64, d}, mx::bfloat16);
            mx::eval(decode_attention_bf16_segmented(q, k, k, 0.0625f, 64));
            return true;
        } catch (const std::exception&) {return false;}
    };
    static const bool ok128 = probe(128), ok256 = probe(256);
    return dim == 128 ? ok128 : dim == 256 ? ok256 : false;
}
#else
inline bool segmented_attention_available(int = 256) {return false;}
#endif

// int8_reference (test and bench, j1069-j1071; dispatch candidate for 8 rows at >= 16K keys): own 8-row attention on an
// int8 KV cache. K is stored transposed in 1024-key blocks ([kv head][block][dim][1024] int8) and V row-major ([kv
// head][key][dim] int8), both with one FP16 scale per (kv head, key). One threadgroup per (block, kv head) holds every
// query row of the kv head (G heads x n rows, 16-row tiles, two simdgroups per tile = halves of the head dim). Per
// 32-key chunk: S = Q K^T on the matrix unit (16 x 32 x 16 ops, the NAX lane layout), each simdgroup of the pair over
// its half of the head dim (Q fragments held in registers), the two partials added in the softmax; the int8 codes go
// into the right operand as they are (exact in FP16) and the scales are applied outside the matrix ops: K's per key on
// S, V's per key folded into P. Softmax by 4 threads per row over the row's visible keys (key <= L - n + t), then O +=
// (P * vscale) V. A row's arithmetic depends only on its own query and visible keys, so the n-row verify equals n
// single-row calls bit for bit.
inline mlx::core::array decode_attention_int8_reference(const mlx::core::array& q, const mlx::core::array& k8t,
                                                  const mlx::core::array& v8, const mlx::core::array& kscale,
                                                  const mlx::core::array& vscale, float scale, int length) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.dtype() != mx::float32 || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode attention int8_reference: q [1, heads, n <= 16, 256] FP32");
    const int heads = q.shape(1), kvh = v8.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = v8.shape(1);
    if (kvh < 1 || heads % kvh || k8t.ndim() != 4 || k8t.shape(0) != kvh || k8t.shape(1) < blocks || k8t.shape(2) != dim || k8t.shape(3) != kBlock ||
        k8t.dtype() != mx::int8 || v8.ndim() != 3 || v8.shape(2) != dim || v8.dtype() != mx::int8 || cap < length || length < 1 ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 || vscale.dtype() != mx::float16)
        throw std::invalid_argument("decode attention int8_reference: int8 K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = tiles * 16;
    // [1, H, n, d] is [kv head][g][t][d]: each kv head's rows are contiguous; FP16 operands (BF16 values are exact in FP16)
    auto query = mx::reshape(mx::astype(mx::astype(q, mx::bfloat16), mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_int8_reference_pass1", {"Q", "KT", "V", "KS", "VS", "P"}, {"M", "S", "O"}, R"METAL(
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
    const device half* qrow = Q + (size_t(kh) * QROWS + tile * 16) * DIM;
    const device int8_t* kt = KT + (size_t(kh) * P[4] + blk) * DIM * BLOCK;    // [dim][key in block]; P[4] = blocks allocated
    const device int8_t* vv = V + size_t(kh) * CAP * DIM;                       // [key][dim]
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    auto acc1 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    auto acc2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    auto acc3 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
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
            auto sacc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0.0f;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int d = (half_d * 8 + ks) * 16 + fm + 8 * h, key0 = c0 + w * 16 + fn;   // keys key0 .. key0 + 3 (within the block)
                    char4 kv = char4(0);                                                            // keys past the view read as 0 (masked)
                    if (key0 + 3 < kn) kv = *(const device char4*)(kt + size_t(d) * BLOCK + key0);
                    else for (short i = 0; i < 4; ++i) kv[i] = key0 + i < kn ? kt[size_t(d) * BLOCK + key0 + i] : char(0);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(kv[i]);
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * C + dn[i]] = sacc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax: each thread its CQ keys, the row's max and sum over the 4 threads by fixed xor shuffles
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
                    const int kc = ks * 16 + fm + 8 * h, key = k0 + c0 + kc;                // chunk key kc, dims d0 .. d0 + 3
                    const int d0 = half_d * HD + nb * 32 + w * 16 + fn;
                    const char4 vv4 = kc < kn ? *(const device char4*)(vv + size_t(key) * DIM + d0) : char4(0);
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
)METAL", header);
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_int8_reference_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
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
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TILES", tiles}};
    auto part = pass1({query, k8t, v8, kscale, vscale, P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {64 * tiles, blocks, kvh}, {64 * tiles, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // BF16-rounded values through FP32, as the SDPA versions return them
    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}
// decode_attention_int8 (served for int8 KV): S = int8 Q x int8 K on the matrix unit (int32 accumulate, Q quantized per
// row by one kernel), softmax and PV over 32-key rounds (FP16 P x V codes). Not bitwise int8_reference: error vs FP64
// ~2.7x its error (j1438); 15-19 % faster than the FP16-operand kernel it replaced at 16K-256K, 8 and 16 rows (j1439 /
// j1443). Used for every int8 call (1 row too), so spec == AR stays exact.
inline mlx::core::array decode_attention_int8(const mlx::core::array& q, const mlx::core::array& k8t,
                                                    const mlx::core::array& v8, const mlx::core::array& kscale,
                                                    const mlx::core::array& vscale, float scale, int length) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.dtype() != mx::float32 || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode attention int8: q [1, heads, n <= 16, 256] FP32");
    const int heads = q.shape(1), kvh = v8.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = v8.shape(1);
    if (kvh < 1 || heads % kvh || k8t.ndim() != 4 || k8t.shape(0) != kvh || k8t.shape(1) < blocks || k8t.shape(2) != dim || k8t.shape(3) != kBlock ||
        k8t.dtype() != mx::int8 || v8.ndim() != 3 || v8.shape(2) != dim || v8.dtype() != mx::int8 || cap < length || length < 1 ||
        kscale.shape() != mx::Shape{kvh, cap} || vscale.shape() != kscale.shape() || kscale.dtype() != mx::float16 || vscale.dtype() != mx::float16)
        throw std::invalid_argument("decode attention int8: int8 K^T blocks [kvh, blocks, 256, 1024], V [kvh, cap, 256], FP16 scales [kvh, cap]");

    const int rows = group * n, tiles = (rows + 15) / 16, qrows = tiles * 16;

    // Q to int8 in one launch: per row (BF16-rounded) scale max(max |q| / 127, 1e-8), codes rint(q / scale) clamped to
    // +-127, rows past `rows` zero. The scale depends only on the row, so a row still equals its 1-row call bit for bit.
    static auto qquant = mx::fast::metal_kernel("tqllm_attn_int8_qquant", {"Q", "R"}, {"Q8", "QS"}, R"METAL(
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = R[0], QROWS = R[1];
    const size_t dst = (size_t(kh) * QROWS + r) * DIM;
    if (int(r) >= ROWS) {
        for (int e = 0; e < DIM / 32; ++e) Q8[dst + e * 32 + lane] = 0;
        if (lane == 0) QS[size_t(kh) * QROWS + r] = 0.0f;
        return;
    }
    const size_t src = (size_t(kh) * ROWS + r) * DIM;
    float x[DIM / 32], amax = 0.0f;
    for (int e = 0; e < DIM / 32; ++e) {
        const uint u = as_type<uint>(Q[src + e * 32 + lane]);
        x[e] = as_type<float>((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);   // BF16 round to nearest even
        amax = max(amax, fabs(x[e]));
    }
    amax = simd_max(amax);
    const float sc = max(amax / 127.0f, 1e-8f);
    for (int e = 0; e < DIM / 32; ++e) Q8[dst + e * 32 + lane] = int8_t(clamp(rint(x[e] / sc), -127.0f, 127.0f));
    if (lane == 0) QS[size_t(kh) * QROWS + r] = sc;
)METAL");
    const auto rq = mx::array({rows, qrows}, {2}, mx::int32);
    auto qo = qquant({mx::reshape(q, {kvh, rows, dim}), rq}, {{kvh, qrows, dim}, {kvh, qrows}}, {mx::int8, mx::float32},
                     {qrows * 32, kvh, 1}, {32, 1, 1}, {{"DIM", dim}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    auto q8 = qo[0], qsc = qo[1];

    static const char* header = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
)METAL";

    static auto pass1 = mx::fast::metal_kernel(
        "tqllm_attn_int8_pass1",
        {"Q", "QS", "KT", "V", "KS", "VS", "P"},
        {"M", "S", "O"},
        R"METAL(
    constexpr int C = 32, RP = 16, HD = DIM / 2, CQ = C / 4;   // one 16-row tile per threadgroup, 32-key rounds
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint grp = threadgroup_position_in_grid.x, blk = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int r0 = int(grp) * RP;
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8];
    const float sc = as_type<float>(P[6]);
    const int k0 = int(blk) * BLOCK, kn = min(BLOCK, L - k0);
    const int tile = int(sg) / 2, half_d = int(sg) % 2;
    const int row = int(tid) / 4, quarter = int(tid) % 4;
    const int grow = r0 + row;
    const int last = grow < ROWS ? L - NR + grow % NR : -1;

    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    threadgroup int32_t st[2][RP * C];
    threadgroup half pt[RP * C];
    threadgroup float arow[RP];
    threadgroup float kscl[C], vscl[C];

    const device int8_t* qrow = Q + (size_t(kh) * QROWS + r0 + tile * 16) * DIM;
    const float qsc = QS[size_t(kh) * QROWS + grow];
    const device int8_t* kt = KT + (size_t(kh) * P[4] + blk) * DIM * BLOCK;
    const device int8_t* vv = V + size_t(kh) * CAP * DIM;

    // S = int8 Q x int8 K on the matrix unit (int32); PV stays FP16 P x half(V) with FP32 accumulators.
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;

    auto ca = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
    auto cb = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();

    auto pa = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto pb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto acc0 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto acc1 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto acc2 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto acc3 = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    for (ushort i = 0; i < 16; ++i) {acc0[i] = 0.0f; acc1[i] = 0.0f; acc2[i] = 0.0f; acc3[i] = 0.0f;}

    short dn[16], dm[16];
    for (ushort i = 0; i < 16; ++i) {
        if (acc0.is_valid_element(i)) {
            const auto idx = acc0.get_multidimensional_index(i);
            dn[i] = short(idx[0]); dm[i] = short(idx[1]);
        } else {
            dn[i] = -1; dm[i] = -1;
        }
    }

    // This simdgroup's int8 Q fragments: 8 k-steps over its half of the head dim.
    int8_t qa[8][8];
    for (int ks = 0; ks < 8; ++ks) for (short i = 0; i < 4; ++i) {
        const int d = (half_d * 8 + ks) * 16 + fn + i;
        qa[ks][i] = qrow[size_t(fm) * DIM + d];
        qa[ks][4 + i] = qrow[size_t(fm + 8) * DIM + d];
    }

    float m = -INFINITY, s = 0.0f;
    for (int c0 = 0; c0 < kn; c0 += C) {
        if (tid < uint(C)) {
            const int key = k0 + c0 + int(tid);
            kscl[tid] = key < L ? float(KS[size_t(kh) * CAP + key]) : 0.0f;
            vscl[tid] = key < L ? float(VS[size_t(kh) * CAP + key]) : 0.0f;
        }

        {
            // partial S tile [16 rows x 32 keys] over this half of the head dim, int32
            auto sacc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, int32_t>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int d = (half_d * 8 + ks) * 16 + fm + 8 * h;
                    // keys past the view stay inside the allocated 1024-key block and are masked in the softmax
                    const char4 kv = *(const device char4*)(kt + size_t(d) * BLOCK + c0 + w * 16 + fn);
                    for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = int8_t(kv[i]);
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * C + dn[i]] = sacc[i];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float x[CQ];
            float lm = -INFINITY;
            for (int j = 0; j < CQ; ++j) {
                const int key = k0 + c0 + quarter * CQ + j;
                const int e = row * C + quarter * CQ + j;
                const float sdot = float(st[0][e]) + float(st[1][e]);
                x[j] = key <= last ? sdot * qsc * kscl[quarter * CQ + j] * sc : -INFINITY;
                lm = max(lm, x[j]);
            }
            lm = max(lm, simd_shuffle_xor(lm, 1));
            lm = max(lm, simd_shuffle_xor(lm, 2));

            const float mn = max(m, lm);
            const float alpha = mn == m ? 1.0f : exp2(m - mn);

            float ls = 0.0f;
            for (int j = 0; j < CQ; ++j) {
                const float pr = x[j] == -INFINITY ? 0.0f : exp2(x[j] - mn);
                ls += pr;
                pt[row * C + quarter * CQ + j] = half(pr * vscl[quarter * CQ + j]);
            }
            ls += simd_shuffle_xor(ls, 1);
            ls += simd_shuffle_xor(ls, 2);

            s = s * alpha + ls;
            m = mn;
            if (quarter == 0) arow[row] = alpha;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
            const float a = arow[tile * 16 + dm[i]];
            if (a != 1.0f) {acc0[i] *= a; acc1[i] *= a; acc2[i] *= a; acc3[i] *= a;}
        }

        const threadgroup half* prow = pt + (tile * 16) * C;
        for (int ks = 0; ks < C / 16; ++ks) {
            for (short i = 0; i < 4; ++i) {
                pa[i] = prow[fm * C + ks * 16 + fn + i];
                pa[4 + i] = prow[(fm + 8) * C + ks * 16 + fn + i];
            }

            for (short nb = 0; nb < 4; ++nb) {
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int kc = ks * 16 + fm + 8 * h;
                    const int d0 = half_d * HD + nb * 32 + w * 16 + fn;
                    const int key = k0 + c0 + kc;
                    const char4 vv4 = key < CAP ? *(const device char4*)(vv + size_t(key) * DIM + d0) : char4(0);
                    for (short i = 0; i < 4; ++i) pb[w * 8 + h * 4 + i] = half(vv4[i]);
                }

                if (nb == 0) op.run(pa, pb, acc0);
                else if (nb == 1) op.run(pa, pb, acc1);
                else if (nb == 2) op.run(pa, pb, acc2);
                else op.run(pa, pb, acc3);
            }
        }
    }

    for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
        const int r = r0 + tile * 16 + dm[i];
        if (r >= ROWS) continue;
        const size_t base = ((size_t(kh) * ROWS + r) * NB + blk) * DIM + half_d * HD;
        O[base + dn[i]] = acc0[i];
        O[base + 32 + dn[i]] = acc1[i];
        O[base + 64 + dn[i]] = acc2[i];
        O[base + 96 + dn[i]] = acc3[i];
    }

    if (grow < ROWS && quarter == 0) {
        const size_t base = (size_t(kh) * ROWS + grow) * NB + blk;
        M[base] = m;
        S[base] = s;
    }
)METAL", header);

    static auto pass2 = mx::fast::metal_kernel(
        "tqllm_attn_int8_pass2",
        {"M", "S", "O", "P"},
        {"Y"},
        R"METAL(
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x, kh = threadgroup_position_in_grid.y;
    const int ROWS = P[3], NB = P[5];

    float m = -INFINITY, s = 0.0f, o[DIM / 32];
    for (int e = 0; e < DIM / 32; ++e) o[e] = 0.0f;

    for (int b = 0; b < NB; ++b) {
        const size_t base = (size_t(kh) * ROWS + r) * NB + b;
        const float mb = M[base], sb = S[base];
        if (sb == 0.0f) continue;
        const float mn = max(m, mb);
        const float a = m == mn ? 1.0f : exp2(m - mn);
        const float c = mb == mn ? 1.0f : exp2(mb - mn);
        s = s * a + sb * c;
        for (int e = 0; e < DIM / 32; ++e) o[e] = o[e] * a + O[base * DIM + e * 32 + lane] * c;
        m = mn;
    }

    const size_t out = (size_t(kh) * ROWS + r) * DIM;
    for (int e = 0; e < DIM / 32; ++e) Y[out + e * 32 + lane] = o[e] / s;
)METAL");

    std::vector<int32_t> p = {length, n, group, rows, k8t.shape(1), blocks, 0, cap, qrows};
    float scf = scale * 1.4426950408889634f;
    std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {9}, mx::int32);

    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{
        {"DIM", dim}, {"BLOCK", kBlock}
    };

    auto part = pass1(
        {q8, qsc, k8t, v8, kscale, vscale, P},
        {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
        {mx::float32, mx::float32, mx::float32},
        {64 * tiles, blocks, kvh},
        {64, 1, 1},
        tpl, std::nullopt, false, mx::Device(mx::Device::gpu)
    );

    auto y = pass2(
        {part[0], part[1], part[2], P},
        {{1, heads, n, dim}},
        {mx::float32},
        {rows * 32, kvh, 1},
        {32, 1, 1},
        tpl, std::nullopt, false, mx::Device(mx::Device::gpu)
    )[0];

    return mx::astype(mx::astype(y, mx::bfloat16), mx::float32);
}

// int8 KV for int8_reference from BF16 / FP32 K [1, kvh, cap, d] and V: one FP16 scale per (kv head, key) = max |x| /
// 127, codes round-to-nearest; K transposed into 1024-key blocks. Test / bench helper (the engine's cache append will
// quantize in place).
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
inline mlx::core::array decode_attention_bf16(const mlx::core::array& q, const mlx::core::array& k,
                                              const mlx::core::array& v, float scale, int valid,
                                              const mlx::core::array* tree = nullptr,
                                              const std::vector<int32_t>* tree_parents = nullptr,
                                              int block_rows = 0, int block_offset = 0) {
    if (k.dtype() != mlx::core::bfloat16) throw std::invalid_argument("decode rows attention reads a BF16 KV cache");
#ifdef MLX_HAS_SDPA_LSE
    if (segmented_attention_available(q.shape(3))) {
        // inside one segment the split gains nothing (bf16_segmented = bf16_folded + merge, j1030: 98 vs 32 us at 64
        // keys): rows that see at most one segment run bf16_folded, each row on the kernel its AR call gets
        namespace mx = mlx::core;
        const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), length = valid > 0 ? valid : k.shape(2);
        constexpr int seg = kAttentionSegment;
        if (length <= seg) return decode_attention_bf16_folded(q, k, v, scale, length, tree);
        // token-tree verify (no host side branches, one call of <= 16 rows): split-K over key chunks
        if (tree && (!tree_parents || tree_parents->empty()) && block_rows == 0 && length > kSplitAttentionMin && split_attention_fits(q, k)) {
            auto bits = cached_tree_mask(*tree, {3, n, 0, 0, 0, 0, 0}, [&] {return split_attention_bits(tree, n);});
            return decode_attention_split(q, k, v, scale, length, bits);
        }
        if (tree) return decode_attention_bf16_segmented(q, k, v, scale, length, seg, tree, tree_parents, block_rows, block_offset);
        if (length - n + 1 <= seg) {
            const int below = seg - (length - n);
            auto lo = decode_attention_bf16_folded(mx::slice(q, {0, 0, 0, 0}, {1, heads, below, dim}), k, v, scale, seg);
            auto hi = decode_attention_bf16_segmented(mx::slice(q, {0, 0, below, 0}, {1, heads, n, dim}), k, v, scale, length);
            return mx::concatenate({lo, hi}, 2);
        }
        return decode_attention_bf16_segmented(q, k, v, scale, valid);
    }
#endif
    return decode_attention_bf16_folded(q, k, v, scale, valid, tree);
}
}  // namespace tqllm::mlx_backend

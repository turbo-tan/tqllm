#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace tqllm::mlx_backend {
// Decode attention for 1-8 query rows. Keys are split into fixed partitions and merged in partition order, so a
// row's result does not depend on how many rows share the call (AR and the 8-row verify agree bit for bit).
inline void check_decode_rows_shape(const mlx::core::array& q, const mlx::core::array& k, const mlx::core::array& v,
                                    int valid) {
    namespace mx = mlx::core;
    if (q.ndim() != 4 || k.ndim() != 4 || q.shape(0) != 1 || k.shape(0) != 1 || q.shape(2) < 1 || q.shape(2) > 8 ||
        (valid > 0 ? valid : k.shape(2)) < q.shape(2) || valid > k.shape(2) || v.shape() != k.shape() || q.shape(1) % k.shape(1) ||
        (q.shape(3) != 128 && q.shape(3) != 256) || k.shape(3) != q.shape(3) || q.dtype() != mx::float32 ||
        (k.dtype() != mx::float32 && k.dtype() != mx::bfloat16) || v.dtype() != k.dtype())
        throw std::invalid_argument("unsupported decode rows attention shape");
}

// v1: one simdgroup per row, lanes across the head dimension, one simd_sum per (key, head); 64-key partitions.
inline mlx::core::array decode_rows_attention_v1(const mlx::core::array& q, const mlx::core::array& k,
                                                 const mlx::core::array& v, float scale, int valid = -1) {
    namespace mx = mlx::core;
    constexpr int PB = 64;
    check_decode_rows_shape(q, k, v, valid);
    const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), kvheads = k.shape(1), length = valid > 0 ? valid : k.shape(2);
    const int group = heads / kvheads, parts = (length + PB - 1) / PB;
    // k / v may be a KV buffer longer than `length`: rows are addressed with the buffer's own head stride
    const uint32_t layout[] = {uint32_t(length), uint32_t(n), uint32_t(parts), uint32_t(k.shape(2))};
    const auto layout_array = mx::array(layout, {4}, mx::uint32);
    static auto partial = mx::fast::metal_kernel("tqllm_decode_rows_partial", {"q", "k", "v", "layout", "scale"},
                                                 {"ms", "ls", "accs"}, R"METAL(
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint p = threadgroup_position_in_grid.x;
uint g = threadgroup_position_in_grid.y;
uint length = layout[0], rows = layout[1], parts = layout[2], stride = layout[3];
if (sg >= rows) return;
uint vis = length - rows + sg + 1u;
uint start = p * uint(PB);
uint slot = ((g * parts + p) * rows + sg) * uint(GROUP);
if (start >= vis) {
    if (lane == 0u) for (uint h = 0; h < uint(GROUP); ++h) {ms[slot + h] = -INFINITY; ls[slot + h] = 0.0f;}
    return;
}
uint end = min(start + uint(PB), vis);
constexpr uint E = DIM / 32u;
float qv[GROUP][E], m[GROUP], l[GROUP], a[GROUP][E];
for (uint h = 0; h < uint(GROUP); ++h) {
    const device float* qp = q + ((g * uint(GROUP) + h) * rows + sg) * uint(DIM) + lane * E;
    for (uint e = 0; e < E; ++e) {qv[h][e] = scale * qp[e]; a[h][e] = 0.0f;}
    m[h] = -INFINITY; l[h] = 0.0f;
}
for (uint j = start; j < end; ++j) {
    const device KT* kp = k + (g * stride + j) * uint(DIM) + lane * E;
    const device KT* vp = v + (g * stride + j) * uint(DIM) + lane * E;
    float kr[E], vr[E];
    for (uint e = 0; e < E; ++e) {kr[e] = float(kp[e]); vr[e] = float(vp[e]);}
    for (uint h = 0; h < uint(GROUP); ++h) {
        float s = 0.0f;
        for (uint e = 0; e < E; ++e) s += qv[h][e] * kr[e];
        s = simd_sum(s);
        float next = max(m[h], s), retain = fast::exp(m[h] - next), weight = fast::exp(s - next);
        l[h] = l[h] * retain + weight;
        for (uint e = 0; e < E; ++e) a[h][e] = a[h][e] * retain + weight * vr[e];
        m[h] = next;
    }
}
for (uint h = 0; h < uint(GROUP); ++h) {
    if (lane == 0u) {ms[slot + h] = m[h]; ls[slot + h] = l[h];}
    device float* ap = accs + (slot + h) * uint(DIM) + lane * E;
    for (uint e = 0; e < E; ++e) ap[e] = a[h][e];
}
)METAL");
    static auto merge = mx::fast::metal_kernel("tqllm_decode_rows_merge", {"ms", "ls", "accs", "layout"}, {"out"}, R"METAL(
uint d = thread_position_in_threadgroup.x;
uint hg = threadgroup_position_in_grid.y;
uint t = threadgroup_position_in_grid.z;
uint rows = layout[1], parts = layout[2];
uint g = hg / uint(GROUP), h = hg % uint(GROUP);
float peak = -INFINITY;
for (uint p = 0; p < parts; ++p) peak = max(peak, ms[((g * parts + p) * rows + t) * uint(GROUP) + h]);
float total = 0.0f, numerator = 0.0f;
for (uint p = 0; p < parts; ++p) {
    uint slot = ((g * parts + p) * rows + t) * uint(GROUP) + h;
    float mp = ms[slot];
    if (mp == -INFINITY) continue;
    float factor = fast::exp(mp - peak);
    total += ls[slot] * factor;
    numerator += accs[slot * uint(DIM) + d] * factor;
}
out[(hg * rows + t) * uint(DIM) + d] = numerator / total;
)METAL");
    const auto stream = mx::Device(mx::Device::gpu);
    auto scratch = partial({mx::contiguous(q), mx::contiguous(k), mx::contiguous(v), layout_array, mx::array(scale)},
                           {{kvheads, parts, n, group}, {kvheads, parts, n, group}, {kvheads, parts, n, group, dim}},
                           {mx::float32, mx::float32, mx::float32}, {parts * 256, kvheads, 1}, {256, 1, 1},
                           {{"DIM", dim}, {"GROUP", group}, {"PB", PB}, {"KT", k.dtype()}}, std::nullopt, false, stream);
    return merge({scratch[0], scratch[1], scratch[2], layout_array}, {q.shape()}, {mx::float32}, {dim, heads, n},
                 {dim, 1, 1}, {{"DIM", dim}, {"GROUP", group}}, std::nullopt, false, stream)[0];
}

// v2: the (row, head) queries of one KV head form 8-query tiles; scores and the P.V product run on the simdgroup
// matrix unit as 8x8 fragments (S^T = K.Q^T, O^T += V^T.P^T), 16 keys per step, 256-key partitions. Each of the 4
// simdgroups owns 64 consecutive keys of the partition with its own online softmax; the 4 states merge in a fixed
// tree, then the partitions merge in order. Every reduction order depends on the key position only, never on the
// number of rows. Fragment lanes: element (fm, fn / fn+1) with fm = (lane/4 & 4) + (lane/2 % 4), fn = (lane/4 & 2)*2
// + (lane % 2)*2 (the layout MLX's steel kernels rely on).
inline mlx::core::array decode_rows_attention_v2(const mlx::core::array& q, const mlx::core::array& k,
                                                 const mlx::core::array& v, float scale, int valid = -1) {
    namespace mx = mlx::core;
    constexpr int SG = 4, KSUB = 16, NSUB = 4, PB = SG * KSUB * NSUB;
    check_decode_rows_shape(q, k, v, valid);
    const int heads = q.shape(1), n = q.shape(2), dim = q.shape(3), kvheads = k.shape(1), length = valid > 0 ? valid : k.shape(2);
    const int group = heads / kvheads, parts = (length + PB - 1) / PB, qtiles = (n * group + 7) / 8;
    const uint32_t layout[] = {uint32_t(length), uint32_t(n), uint32_t(parts), uint32_t(k.shape(2)), uint32_t(qtiles)};
    const auto layout_array = mx::array(layout, {5}, mx::uint32);
    static auto partial = mx::fast::metal_kernel("tqllm_decode_rows_partial_v2", {"q", "k", "v", "layout", "scale"},
                                                 {"ms", "ls", "accs"}, R"METAL(
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint part = threadgroup_position_in_grid.x;
uint g = threadgroup_position_in_grid.y;
uint qt = threadgroup_position_in_grid.z;
uint length = layout[0], rows = layout[1], parts = layout[2], stride = layout[3], qtiles = layout[4];
uint nq = rows * uint(GROUP);
uint qid = lane / 4u;
uint fm = (qid & 4u) + ((lane / 2u) % 4u);
uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
uint qa = qt * 8u + fn, qb = qa + 1u;
uint visa = qa < nq ? length - rows + qa / uint(GROUP) + 1u : 0u;
uint visb = qb < nq ? length - rows + qb / uint(GROUP) + 1u : 0u;
constexpr uint DB = DIM / 8u;
// Q^T fragments (B operand): element (dim fm, query fn / fn+1), held for the whole key loop
simdgroup_float8x8 qf[DB];
{
    uint ca = min(qa, nq - 1u), cb = min(qb, nq - 1u);
    const device float* pa = q + ((g * uint(GROUP) + ca % uint(GROUP)) * rows + ca / uint(GROUP)) * uint(DIM) + fm;
    const device float* pb = q + ((g * uint(GROUP) + cb % uint(GROUP)) * rows + cb / uint(GROUP)) * uint(DIM) + fm;
    for (uint d = 0; d < DB; ++d) {
        qf[d].thread_elements()[0] = qa < nq ? scale * pa[d * 8u] : 0.0f;
        qf[d].thread_elements()[1] = qb < nq ? scale * pb[d * 8u] : 0.0f;
    }
}
simdgroup_float8x8 o[DB];
for (uint d = 0; d < DB; ++d) o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
const device KT* kh = k + g * stride * uint(DIM);
const device KT* vh = v + g * stride * uint(DIM);
uint base = part * uint(PB) + sg * uint(KSUB * NSUB);
for (uint t = 0; t < uint(NSUB); ++t) {
    uint kk = base + t * uint(KSUB);
    if (kk >= length) break;
    // S^T = K . Q^T for keys kk..kk+15: K fragment (A operand) element (key fm, dims fn / fn+1)
    simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint key0 = kk + fm, key1 = kk + 8u + fm;
    bool ok0 = key0 < length, ok1 = key1 < length;
    const device KT* kp0 = kh + min(key0, length - 1u) * uint(DIM) + fn;
    const device KT* kp1 = kh + min(key1, length - 1u) * uint(DIM) + fn;
    for (uint d = 0; d < DB; ++d) {
        simdgroup_float8x8 k0, k1;
        k0.thread_elements()[0] = ok0 ? float(kp0[d * 8u]) : 0.0f;
        k0.thread_elements()[1] = ok0 ? float(kp0[d * 8u + 1u]) : 0.0f;
        k1.thread_elements()[0] = ok1 ? float(kp1[d * 8u]) : 0.0f;
        k1.thread_elements()[1] = ok1 ? float(kp1[d * 8u + 1u]) : 0.0f;
        simdgroup_multiply_accumulate(s0, k0, qf[d], s0);
        simdgroup_multiply_accumulate(s1, k1, qf[d], s1);
    }
    // online softmax per query (a fragment column): lanes sharing fn differ in bits 1, 2 and 4
    float2 e0 = s0.thread_elements(), e1 = s1.thread_elements();
    e0.x = key0 < visa ? e0.x : -INFINITY; e0.y = key0 < visb ? e0.y : -INFINITY;
    e1.x = key1 < visa ? e1.x : -INFINITY; e1.y = key1 < visb ? e1.y : -INFINITY;
    float xa = max(e0.x, e1.x), xb = max(e0.y, e1.y);
    xa = max(xa, simd_shuffle_xor(xa, 2u)); xa = max(xa, simd_shuffle_xor(xa, 4u)); xa = max(xa, simd_shuffle_xor(xa, 16u));
    xb = max(xb, simd_shuffle_xor(xb, 2u)); xb = max(xb, simd_shuffle_xor(xb, 4u)); xb = max(xb, simd_shuffle_xor(xb, 16u));
    float na = max(ma, xa), nb = max(mb, xb);
    float ra = na == -INFINITY ? 1.0f : fast::exp(ma - na);
    float rb = nb == -INFINITY ? 1.0f : fast::exp(mb - nb);
    float2 p0, p1;
    p0.x = na == -INFINITY ? 0.0f : fast::exp(e0.x - na); p1.x = na == -INFINITY ? 0.0f : fast::exp(e1.x - na);
    p0.y = nb == -INFINITY ? 0.0f : fast::exp(e0.y - nb); p1.y = nb == -INFINITY ? 0.0f : fast::exp(e1.y - nb);
    float sa = p0.x + p1.x, sb = p0.y + p1.y;
    sa += simd_shuffle_xor(sa, 2u); sa += simd_shuffle_xor(sa, 4u); sa += simd_shuffle_xor(sa, 16u);
    sb += simd_shuffle_xor(sb, 2u); sb += simd_shuffle_xor(sb, 4u); sb += simd_shuffle_xor(sb, 16u);
    la = la * ra + sa; lb = lb * rb + sb; ma = na; mb = nb;
    for (uint d = 0; d < DB; ++d) {o[d].thread_elements()[0] *= ra; o[d].thread_elements()[1] *= rb;}
    simdgroup_float8x8 pf0, pf1;
    pf0.thread_elements()[0] = p0.x; pf0.thread_elements()[1] = p0.y;
    pf1.thread_elements()[0] = p1.x; pf1.thread_elements()[1] = p1.y;
    // O^T += V^T . P^T: V^T fragment (A operand) element (dim fm, keys fn / fn+1)
    uint kva = kk + fn, kvb = kva + 1u, kvc = kk + 8u + fn, kvd = kvc + 1u;
    const device KT* vp = vh + fm;
    for (uint d = 0; d < DB; ++d) {
        simdgroup_float8x8 v0, v1;
        v0.thread_elements()[0] = kva < length ? float(vp[kva * uint(DIM) + d * 8u]) : 0.0f;
        v0.thread_elements()[1] = kvb < length ? float(vp[kvb * uint(DIM) + d * 8u]) : 0.0f;
        v1.thread_elements()[0] = kvc < length ? float(vp[kvc * uint(DIM) + d * 8u]) : 0.0f;
        v1.thread_elements()[1] = kvd < length ? float(vp[kvd * uint(DIM) + d * 8u]) : 0.0f;
        simdgroup_multiply_accumulate(o[d], v0, pf0, o[d]);
        simdgroup_multiply_accumulate(o[d], v1, pf1, o[d]);
    }
}
// merge the 4 simdgroup states: common peak, own factor, then (O0 + O1) + (O2 + O3)
threadgroup float mt[SG * 8];
threadgroup float lt[SG * 8];
threadgroup float ot[2 * 8 * DIM];
if (fm == 0u) {mt[sg * 8u + fn] = ma; mt[sg * 8u + fn + 1u] = mb; lt[sg * 8u + fn] = la; lt[sg * 8u + fn + 1u] = lb;}
threadgroup_barrier(mem_flags::mem_threadgroup);
float pa = -INFINITY, pb = -INFINITY;
for (uint s = 0; s < uint(SG); ++s) {pa = max(pa, mt[s * 8u + fn]); pb = max(pb, mt[s * 8u + fn + 1u]);}
float fa = pa == -INFINITY ? 0.0f : fast::exp(ma - pa);
float fb = pb == -INFINITY ? 0.0f : fast::exp(mb - pb);
float La = 0.0f, Lb = 0.0f;
for (uint s = 0; s < uint(SG); ++s) {
    float m0 = mt[s * 8u + fn], m1 = mt[s * 8u + fn + 1u];
    if (m0 != -INFINITY) La += lt[s * 8u + fn] * fast::exp(m0 - pa);
    if (m1 != -INFINITY) Lb += lt[s * 8u + fn + 1u] * fast::exp(m1 - pb);
}
for (uint d = 0; d < DB; ++d) {o[d].thread_elements()[0] *= fa; o[d].thread_elements()[1] *= fb;}
uint oa = fn * uint(DIM) + fm, ob = (fn + 1u) * uint(DIM) + fm;
if (sg == 1u || sg == 3u) {
    threadgroup float* dst = ot + (sg / 2u) * 8u * uint(DIM);
    for (uint d = 0; d < DB; ++d) {dst[oa + d * 8u] = o[d].thread_elements()[0]; dst[ob + d * 8u] = o[d].thread_elements()[1];}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u || sg == 2u) {
    threadgroup const float* src = ot + (sg / 2u) * 8u * uint(DIM);
    for (uint d = 0; d < DB; ++d) {o[d].thread_elements()[0] += src[oa + d * 8u]; o[d].thread_elements()[1] += src[ob + d * 8u];}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 2u) for (uint d = 0; d < DB; ++d) {ot[oa + d * 8u] = o[d].thread_elements()[0]; ot[ob + d * 8u] = o[d].thread_elements()[1];}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg != 0u) return;
uint slot = ((g * parts + part) * qtiles + qt) * 8u;
if (fm == 0u) {ms[slot + fn] = pa; ms[slot + fn + 1u] = pb; ls[slot + fn] = La; ls[slot + fn + 1u] = Lb;}
device float* acc = accs + slot * uint(DIM);
for (uint d = 0; d < DB; ++d) {
    acc[oa + d * 8u] = o[d].thread_elements()[0] + ot[oa + d * 8u];
    acc[ob + d * 8u] = o[d].thread_elements()[1] + ot[ob + d * 8u];
}
)METAL");
    static auto merge = mx::fast::metal_kernel("tqllm_decode_rows_merge_v2", {"ms", "ls", "accs", "layout"}, {"out"}, R"METAL(
uint d = thread_position_in_threadgroup.x;
uint y = threadgroup_position_in_grid.y;
uint rows = layout[1], parts = layout[2], qtiles = layout[4];
uint g = y / (qtiles * 8u), qi = y % (qtiles * 8u);
if (qi >= rows * uint(GROUP)) return;
uint r = qi / uint(GROUP), h = qi % uint(GROUP);
float peak = -INFINITY;
for (uint p = 0; p < parts; ++p) peak = max(peak, ms[((g * parts + p) * qtiles) * 8u + qi]);
float total = 0.0f, numerator = 0.0f;
for (uint p = 0; p < parts; ++p) {
    uint slot = ((g * parts + p) * qtiles) * 8u + qi;
    float mp = ms[slot];
    if (mp == -INFINITY) continue;
    float factor = fast::exp(mp - peak);
    total += ls[slot] * factor;
    numerator += accs[slot * uint(DIM) + d] * factor;
}
out[((g * uint(GROUP) + h) * rows + r) * uint(DIM) + d] = numerator / total;
)METAL");
    const auto stream = mx::Device(mx::Device::gpu);
    auto scratch = partial({mx::contiguous(q), mx::contiguous(k), mx::contiguous(v), layout_array, mx::array(scale)},
                           {{kvheads, parts, qtiles, 8}, {kvheads, parts, qtiles, 8}, {kvheads, parts, qtiles, 8, dim}},
                           {mx::float32, mx::float32, mx::float32}, {parts * SG * 32, kvheads, qtiles}, {SG * 32, 1, 1},
                           {{"DIM", dim}, {"GROUP", group}, {"PB", PB}, {"SG", SG}, {"KSUB", KSUB}, {"NSUB", NSUB}, {"KT", k.dtype()}},
                           std::nullopt, false, stream);
    return merge({scratch[0], scratch[1], scratch[2], layout_array}, {q.shape()}, {mx::float32}, {dim, kvheads * qtiles * 8, 1},
                 {dim, 1, 1}, {{"DIM", dim}, {"GROUP", group}}, std::nullopt, false, stream)[0];
}

inline mlx::core::array decode_rows_attention(const mlx::core::array& q, const mlx::core::array& k,
                                              const mlx::core::array& v, float scale, int valid = -1) {
    static const bool v1 = std::getenv("TQLLM_MLX_ATTN_DECODE_V1") != nullptr;
    return v1 ? decode_rows_attention_v1(q, k, v, scale, valid) : decode_rows_attention_v2(q, k, v, scale, valid);
}
}  // namespace tqllm::mlx_backend

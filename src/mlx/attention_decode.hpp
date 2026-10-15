#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace tqllm::mlx_backend {
// Decode attention for 1-8 query rows. Keys are split into fixed partitions and merged in partition order, so a
// row's result does not depend on how many rows share the call (AR and the 8-row verify agree bit for bit).
inline mlx::core::array decode_rows_attention(const mlx::core::array& q, const mlx::core::array& k,
                                              const mlx::core::array& v, float scale, int valid = -1) {
    namespace mx = mlx::core;
    constexpr int PB = 64;
    if (q.ndim() != 4 || k.ndim() != 4 || q.shape(0) != 1 || k.shape(0) != 1 || q.shape(2) < 1 || q.shape(2) > 8 ||
        (valid > 0 ? valid : k.shape(2)) < q.shape(2) || valid > k.shape(2) || v.shape() != k.shape() || q.shape(1) % k.shape(1) ||
        (q.shape(3) != 128 && q.shape(3) != 256) || k.shape(3) != q.shape(3) || q.dtype() != mx::float32 ||
        k.dtype() != mx::float32 || v.dtype() != mx::float32)
        throw std::invalid_argument("unsupported decode rows attention shape");
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
    const device float* kp = k + (g * stride + j) * uint(DIM) + lane * E;
    const device float* vp = v + (g * stride + j) * uint(DIM) + lane * E;
    float kr[E], vr[E];
    for (uint e = 0; e < E; ++e) {kr[e] = kp[e]; vr[e] = vp[e];}
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
                           {{"DIM", dim}, {"GROUP", group}, {"PB", PB}}, std::nullopt, false, stream);
    return merge({scratch[0], scratch[1], scratch[2], layout_array}, {q.shape()}, {mx::float32}, {dim, heads, n},
                 {dim, 1, 1}, {{"DIM", dim}, {"GROUP", group}}, std::nullopt, false, stream)[0];
}
}  // namespace tqllm::mlx_backend

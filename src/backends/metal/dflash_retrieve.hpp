#pragma once
// DFlash2 far-context retrieval: the drafter attends a fixed window; past it a document's names and facts are out of view.
// Retrieval adds `chunks` x 16 rows: the far chunks the anchor query scores highest, moved (keys re-rotated) to sit right
// before the window, so the drafter sees them at distances it was trained on (CUDA lane: --draft-retrieve).
// Score of a 16-key chunk per kv head: max over its keys of <anchor query averaged over the head's group, key> (rotated
// vectors, as stored); chunks [0, scored) are scored, those not wholly below F (device scalar: the window's first
// position) masked out, the top `chunks` kept per kv head in ascending order. Row i of the result holds chunk sel[i / 16]
// key i % 16 rotated from its position to to - R + i (to = F unless given); values copied. The caller guarantees
// F >= chunks x 16.
#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace tqllm::mlx_backend {
struct DflashRetrieval {
    mlx::core::array k, v;       // [1, kvh, R, hd] (BF16, as the cache)
    mlx::core::array chunks;     // [kvh, chunks] selected chunk ids, ascending
    int retrieved = 0;           // R
};
// The side GPU stream for look-ahead selections, one per thread: MLX streams belong to the thread that made them
// (a server request on another thread failed with "There is no Stream(gpu, N) in current thread", j1096)
inline mlx::core::Stream retrieval_stream() {
    thread_local const mlx::core::Stream stream = mlx::core::new_stream(mlx::core::Device::gpu);
    return stream;
}
// Selection: the top `chunks` chunk ids per kv head [kvh, chunks] (ascending) for the anchor query q0 [1, heads, 1, hd]
// over kfull's chunks [0, scored), chunks not wholly below F masked. Separate from the gather so a step can select ahead
// (from the previous draft's anchor queries) while the GPU runs the verify, off the draft's critical path.
inline mlx::core::array dflash_retrieve_select(const mlx::core::array& q0, const mlx::core::array& kfull, const mlx::core::array& F,
                                               int scored, int chunks) {
    namespace mx = mlx::core;
    constexpr int kChunk = 16;
    const int heads = q0.shape(1), kvh = kfull.shape(1), hd = kfull.shape(3), group = heads / kvh, take = std::min(chunks, scored);
    auto anchor = mx::astype(mx::slice(q0, {0, 0, 0, 0}, {1, heads, 1, hd}), mx::float32);
    auto qm = mx::astype(mx::mean(mx::reshape(anchor, {kvh, group, hd}), 1), kfull.dtype());           // [kvh, hd]
    // the keys are read once in their stored type (an FP32 copy of the cache tripled the bytes: j1094 +17 ms at 64K)
    auto keys = mx::reshape(mx::slice(kfull, {0, 0, 0, 0}, {1, kvh, scored * kChunk, hd}), {kvh, scored * kChunk, hd});
    auto dots = mx::matmul(keys, mx::expand_dims(qm, -1));                                              // [kvh, scored*16, 1]
    auto score = mx::astype(mx::max(mx::reshape(dots, {kvh, scored, kChunk}), -1), mx::float32);      // [kvh, scored]
    auto far = mx::less(mx::arange(scored, mx::int32), mx::floor_divide(mx::astype(F, mx::int32), mx::array(kChunk, mx::int32)));
    score = mx::where(mx::expand_dims(far, 0), score, mx::array(-INFINITY));
    auto top = mx::slice(mx::argpartition(mx::negative(score), take - 1, -1), {0, 0}, {kvh, take});
    return mx::sort(mx::astype(top, mx::int32), -1);
}
// Selection from chunk key sums (the cache's keys summed per 16-key chunk, kept as rows arrive): <query, mean key> per
// chunk instead of the max over the chunk's keys, 1/16 of the bytes of a cache scan. ksum [1, kvh, chunks_cap, hd] FP32.
inline mlx::core::array dflash_retrieve_select_sums(const mlx::core::array& q0, const mlx::core::array& ksum, const mlx::core::array& F,
                                                    int scored, int chunks) {
    namespace mx = mlx::core;
    constexpr int kChunk = 16;
    const int heads = q0.shape(1), kvh = ksum.shape(1), hd = ksum.shape(3), group = heads / kvh, take = std::min(chunks, scored);
    auto anchor = mx::astype(mx::slice(q0, {0, 0, 0, 0}, {1, heads, 1, hd}), mx::float32);
    auto qm = mx::mean(mx::reshape(anchor, {kvh, group, hd}), 1);                                       // [kvh, hd]
    auto sums = mx::reshape(mx::slice(ksum, {0, 0, 0, 0}, {1, kvh, scored, hd}), {kvh, scored, hd});
    auto score = mx::squeeze(mx::matmul(sums, mx::expand_dims(qm, -1)), -1);                            // [kvh, scored]
    auto far = mx::less(mx::arange(scored, mx::int32), mx::floor_divide(mx::astype(F, mx::int32), mx::array(kChunk, mx::int32)));
    score = mx::where(mx::expand_dims(far, 0), score, mx::array(-INFINITY));
    auto top = mx::slice(mx::argpartition(mx::negative(score), take - 1, -1), {0, 0}, {kvh, take});
    return mx::sort(mx::astype(top, mx::int32), -1);
}
// int8 far caches: codes [1, kvh, cap, hd] with one FP16 scale per (kv head, row) [1, kvh, cap] = max |x| / 127 (the
// drafter's own K / V, not the target's: a quantized far row only moves the draft, never the verified text). Half the
// BF16 bytes: at 248K the BF16 caches (6.7 GB) did not fit beside the int8 KV (j1166), the int8 ones do.
struct FarInt8 {mlx::core::array codes, scales;};
inline FarInt8 dflash_far_quantize(const mlx::core::array& x) {   // x [1, kvh, n, hd] (BF16 values)
    namespace mx = mlx::core;
    if (x.ndim() != 4 || x.shape(0) != 1 || x.shape(3) % 32 || x.shape(3) > 256) throw std::invalid_argument("dflash_far_quantize: [1, kvh, n, hd]");
    const int kvh = x.shape(1), n = x.shape(2), hd = x.shape(3);
    // one threadgroup per (kv head, row): the row's max |x| over its simdgroups, FP16 scale, codes round half to even.
    // One dispatch per layer and tensor instead of ~11 MLX ops (the decode inject quantizes 5 layers x K, V every step)
    static auto kernel = mx::fast::metal_kernel("tqllm_dflash_far_int8", {"X"}, {"C", "S"}, R"METAL(
    const uint t = thread_position_in_threadgroup.x, sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint g = threadgroup_position_in_grid.y;
    const size_t off = size_t(g) * HD + t;
    const float x = float(X[off]);
    threadgroup float red[8];
    const float a = simd_max(fabs(x));
    if (lane == 0) red[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = red[0];
    for (uint i = 1; i < HD / 32; ++i) amax = max(amax, red[i]);
    const half s16 = half(max(metal::precise::divide(amax, 127.0f), 1e-8f));
    C[off] = int8_t(clamp(metal::rint(metal::precise::divide(x, float(s16))), -127.0f, 127.0f));
    if (t == 0) S[g] = s16;
)METAL");
    auto out = kernel({mx::contiguous(mx::astype(x, mx::bfloat16))}, {{1, kvh, n, hd}, {1, kvh, n}}, {mx::int8, mx::float16},
                      {hd, kvh * n, 1}, {hd, 1, 1}, {{"HD", hd}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return {out[0], out[1]};
}
inline mlx::core::array dflash_far_dequantize(const mlx::core::array& codes, const mlx::core::array& scales) {   // -> FP32
    namespace mx = mlx::core;
    return mx::multiply(mx::astype(codes, mx::float32), mx::astype(mx::expand_dims(scales, -1), mx::float32));
}
// Gather: rows of the selected chunks [kvh, take], keys rotated from their positions to to - R + i. int8 caches come with
// their scales and are dequantized to BF16 rows first.
inline DflashRetrieval dflash_retrieve_gather(const mlx::core::array& sel, const mlx::core::array& kfull, const mlx::core::array& vfull,
                                              const mlx::core::array& to, float theta,
                                              const std::optional<mlx::core::array>& kscale = std::nullopt,
                                              const std::optional<mlx::core::array>& vscale = std::nullopt) {
    namespace mx = mlx::core;
    constexpr int kChunk = 16;
    const int kvh = kfull.shape(1), hd = kfull.shape(3), cap = kfull.shape(2), R = sel.shape(1) * kChunk;
    auto src = mx::reshape(mx::add(mx::multiply(mx::expand_dims(sel, -1), mx::array(kChunk, mx::int32)), mx::arange(kChunk, mx::int32)),
                           {kvh, R});                                                                    // [kvh, R]
    auto dest = mx::add(mx::arange(R, mx::int32), mx::subtract(mx::astype(to, mx::int32), mx::array(R, mx::int32)));
    auto delta = mx::astype(mx::subtract(dest, src), mx::float32);                                      // [kvh, R]
    auto idx = mx::expand_dims(src, -1);
    auto kr = mx::take_along_axis(mx::reshape(kfull, {kvh, cap, hd}), idx, 1);                          // [kvh, R, hd]
    auto vr = mx::take_along_axis(mx::reshape(vfull, {kvh, cap, hd}), idx, 1);
    if (kscale) {
        auto ks = mx::squeeze(mx::take_along_axis(mx::reshape(*kscale, {kvh, cap, 1}), idx, 1), -1);   // [kvh, R]
        auto vs = mx::squeeze(mx::take_along_axis(mx::reshape(*vscale, {kvh, cap, 1}), idx, 1), -1);
        kr = mx::astype(dflash_far_dequantize(kr, ks), mx::bfloat16);
        vr = mx::astype(dflash_far_dequantize(vr, vs), mx::bfloat16);
    }
    // NeoX rotation by delta (angles add): x[d], x[d + hd / 2]
    const int half = hd / 2;
    auto d = mx::astype(mx::arange(half, mx::int32), mx::float32);
    auto inv = mx::exp(mx::multiply(d, mx::array(-2.0f * std::log(theta) / float(hd))));               // theta^(-2d / hd)
    auto ang = mx::multiply(mx::expand_dims(delta, -1), inv);                                           // [kvh, R, half]
    auto c = mx::cos(ang), s = mx::sin(ang);
    auto kf = mx::astype(kr, mx::float32);
    auto a = mx::slice(kf, {0, 0, 0}, {kvh, R, half}), b = mx::slice(kf, {0, 0, half}, {kvh, R, hd});
    auto rot = mx::concatenate({mx::subtract(mx::multiply(a, c), mx::multiply(b, s)), mx::add(mx::multiply(b, c), mx::multiply(a, s))}, -1);
    // delta 0 (a chunk already where it lands) keeps the stored key bit for bit
    auto krr = mx::where(mx::expand_dims(mx::equal(delta, mx::array(0.0f)), -1), kr, mx::astype(rot, kr.dtype()));
    return {mx::reshape(krr, {1, kvh, R, hd}), mx::reshape(vr, {1, kvh, R, hd}), sel, R};
}
// q [1, heads, nq, hd] (row 0 = the anchor, rotated at its position), kfull / vfull [1, kvh, cap, hd] (keys rotated at
// their positions), F / to int32 scalars, scored * 16 <= cap
inline DflashRetrieval dflash_retrieve(const mlx::core::array& q, const mlx::core::array& kfull, const mlx::core::array& vfull,
                                       const mlx::core::array& F, int scored, int chunks, float theta,
                                       const std::optional<mlx::core::array>& to = std::nullopt,
                                       const std::optional<mlx::core::array>& kscale = std::nullopt,
                                       const std::optional<mlx::core::array>& vscale = std::nullopt) {
    if (kfull.dtype() == mlx::core::int8) throw std::invalid_argument("dflash_retrieve: int8 caches select from the chunk key sums (dflash_retrieve_select_sums)");
    return dflash_retrieve_gather(dflash_retrieve_select(q, kfull, F, scored, chunks), kfull, vfull, to ? *to : F, theta, kscale, vscale);
}
}  // namespace tqllm::mlx_backend

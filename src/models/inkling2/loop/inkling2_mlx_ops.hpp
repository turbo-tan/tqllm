#pragma once
#include <mlx/mlx.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <optional>
#include <vector>
#include <algorithm>

namespace tqllm::mlx_backend {
// keys/values already end with the zero virtual sink row (MlxKvCache::append); mask is [1,H,n,rows] with the
// per-query sink logit in its last column.
inline mlx::core::array inkling2_sink_attention(const mlx::core::array& q,const mlx::core::array& keys,
    const mlx::core::array& values,const mlx::core::array& mask,float scale) {
    namespace mx=mlx::core;
    if(q.ndim()!=4 || keys.ndim()!=4 || keys.shape()!=values.shape() || q.shape(0)!=1 || keys.shape(0)!=1 ||
       mask.shape()!=mx::Shape{1,q.shape(1),q.shape(2),keys.shape(2)}) throw std::invalid_argument("Inkling2 sink attention shape");
    return mx::fast::scaled_dot_product_attention(q,keys,values,scale,"",mx::astype(mask,q.dtype()));
}
inline mlx::core::array inkling2_masked_attention(const mlx::core::array& q,const mlx::core::array& k,
    const mlx::core::array& v,const mlx::core::array& bias,const mlx::core::array& sinks,float scale) {
    namespace mx=mlx::core;
    if(q.ndim()!=4 || k.ndim()!=4 || k.shape()!=v.shape() || q.shape(0)!=1 || k.shape(0)!=1 ||
       bias.shape()!=mx::Shape{1,q.shape(1),q.shape(2),k.shape(2)} ||
       sinks.shape()!=mx::Shape{1,q.shape(1),q.shape(2),1}) throw std::invalid_argument("Inkling2 attention shape");
    auto zero=mx::zeros({1,k.shape(1),1,k.shape(3)},k.dtype());
    auto keys=mx::concatenate({k,zero},2),values=mx::concatenate({v,zero},2);
    auto mask=mx::concatenate({mx::astype(bias,q.dtype()),mx::astype(sinks,q.dtype())},-1);
    // A zero-value virtual key represents the per-query sink without a fixed-head sink restriction.
    return inkling2_sink_attention(q,keys,values,mask,scale);
}
inline std::pair<mlx::core::array,mlx::core::array> inkling2_static_convolution(
    const mlx::core::array& input,const mlx::core::array& taps,const mlx::core::array& history) {
    namespace mx=mlx::core;
    if (input.ndim()!=2 || taps.shape()!=mx::Shape{input.shape(1),4} || history.shape()!=mx::Shape{3,input.shape(1)})
        throw std::invalid_argument("Inkling2 static convolution shape");
    const int n=input.shape(0),c=input.shape(1);
    auto x=mx::astype(input,mx::float32);
    auto values=mx::concatenate({history,x},0);
    auto term=[&](int tap) {
        return mx::multiply(mx::slice(values,{tap,0},{tap+n,c}),mx::squeeze(mx::slice(taps,{0,tap},{c,tap+1}),1));
    };
    auto sum=term(0);
    for (int tap=1;tap<4;++tap) sum=mx::add(sum,term(tap));
    auto y=mx::astype(mx::multiply(mx::add(sum,x),mx::array(1.f/std::sqrt(5.f))),mx::bfloat16);
    auto next=mx::slice(values,{n,0},{n+3,c});
    return {y,next};
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// A row count as a kernel input ([1] int32) rather than a template argument, so a new chunk length (every prompt's last
// chunk) does not compile new Metal kernels.
inline mlx::core::array inkling2_count(int n) { return mlx::core::array(&n, {1}, mlx::core::int32); }
// A one-element F32 array for kernel inputs a variant does not read (evaluated once: no fill kernel per call).
inline const mlx::core::array& inkling2_unused() {
    static const mlx::core::array a = [] { auto z = mlx::core::zeros({1}, mlx::core::float32); mlx::core::eval(z); return z; }();
    return a;
}
// Depth attention (block attention residuals) in one kernel per row: the states are the finished blocks
// ([n, D] F32 each, at most 4) and the partial state normalised (unweighted RMS, eps 1e-6); each query [Q <= 4, D]
// scores them (dot / sqrt(D)), a softmax over the states mixes them. Returns {mix [n, Q, D] F32, the normalised
// partial [n, D] F32}. One threadgroup of 256 a row: the norm, the Q x (blocks + 1) dot products and the mix in place
// of a stack, two matmuls, a scale and a softmax (about 8 MLX kernels a call, 2.5 calls a visit).
// With norm weights (one F32 [D] per query): the third output is each mix row RMS-normalised (eps 1e-6) times its
// weight in BF16 [n, Q, D], the same arithmetic as inkling2_norm(row, weight, bfloat16) in place of a kernel each.
inline std::array<mlx::core::array, 3> inkling2_depth_normed(const std::vector<mlx::core::array>& blocks,
    const mlx::core::array& partial, const mlx::core::array& query, const std::vector<mlx::core::array>& norm_weights) {
    namespace mx = mlx::core;
    const int n = partial.shape(0), D = partial.shape(1), NB = int(blocks.size());
    auto q = mx::reshape(query, {-1, D});
    const int Q = q.shape(0);
    const bool normed_out = !norm_weights.empty();
    if (partial.ndim() != 2 || NB > 4 || Q > 4 || D % 256 || (normed_out && int(norm_weights.size()) != Q))
        throw std::invalid_argument("Inkling2 depth shape");
    for (const auto& b : blocks) if (b.shape() != partial.shape()) throw std::invalid_argument("Inkling2 depth block shape");
    static const char* source = R"METAL(
    constexpr int TG = 256, PER = D / TG, S = NB + 1;
    const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int row = int(threadgroup_position_in_grid.y);
    threadgroup float red[8][Q * S + 1];
    float x[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) { x[i] = float(partial[size_t(row) * D + tid + i * TG]); ss += x[i] * x[i]; }
    // the partial's RMS
    float t = simd_sum(ss);
    if (lane == 0) red[sg][0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int g = 0; g < TG / 32; ++g) total += red[g][0];
    const float inv = rsqrt(total / float(D) + 1e-6f);
    for (int i = 0; i < PER; ++i) { x[i] *= inv; normed[size_t(row) * D + tid + i * TG] = x[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // scores: every query against every state
    float dots[Q * S];
    for (int j = 0; j < Q * S; ++j) dots[j] = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int c = int(tid) + i * TG;
        float st[S];
        if (NB > 0) st[0] = b0[size_t(row) * D + c];
        if (NB > 1) st[1] = b1[size_t(row) * D + c];
        if (NB > 2) st[2] = b2[size_t(row) * D + c];
        if (NB > 3) st[3] = b3[size_t(row) * D + c];
        st[NB] = x[i];
        for (int qq = 0; qq < Q; ++qq) {
            const float qv = query[qq * D + c];
            for (int b = 0; b < S; ++b) dots[qq * S + b] += st[b] * qv;
        }
    }
    for (int j = 0; j < Q * S; ++j) { const float v = simd_sum(dots[j]); if (lane == 0) red[sg][j] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[Q * S];
    float m[NORMED ? Q : 1][NORMED ? PER : 1];
    const float scale = rsqrt(float(D));
    for (int qq = 0; qq < Q; ++qq) {
        float m = -INFINITY;
        for (int b = 0; b < S; ++b) {
            float v = 0.0f;
            for (int g = 0; g < TG / 32; ++g) v += red[g][qq * S + b];
            w[qq * S + b] = v * scale;
            m = max(m, w[qq * S + b]);
        }
        float z = 0.0f;
        for (int b = 0; b < S; ++b) { w[qq * S + b] = exp(w[qq * S + b] - m); z += w[qq * S + b]; }
        for (int b = 0; b < S; ++b) w[qq * S + b] /= z;
    }
    // the mix
    for (int i = 0; i < PER; ++i) {
        const int c = int(tid) + i * TG;
        float st[S];
        if (NB > 0) st[0] = b0[size_t(row) * D + c];
        if (NB > 1) st[1] = b1[size_t(row) * D + c];
        if (NB > 2) st[2] = b2[size_t(row) * D + c];
        if (NB > 3) st[3] = b3[size_t(row) * D + c];
        st[NB] = x[i];
        for (int qq = 0; qq < Q; ++qq) {
            float v = 0.0f;
            for (int b = 0; b < S; ++b) v += w[qq * S + b] * st[b];
            out[(size_t(row) * Q + qq) * D + c] = v;
            if (NORMED) m[qq][i] = v;
        }
    }
    if (NORMED) {
        // each mix row's RMS: per thread in channel order, then the simdgroups in order (as inkling2_norm)
        float sq[Q];
        for (int qq = 0; qq < Q; ++qq) { sq[qq] = 0.0f; for (int i = 0; i < PER; ++i) sq[qq] += m[qq][i] * m[qq][i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int qq = 0; qq < Q; ++qq) { const float v = simd_sum(sq[qq]); if (lane == 0) red[sg][qq] = v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int qq = 0; qq < Q; ++qq) {
            float total = 0.0f;
            for (int g = 0; g < TG / 32; ++g) total += red[g][qq];
            const float inv = rsqrt(total / float(D) + 1e-6f);
            for (int i = 0; i < PER; ++i) {
                const int c = int(tid) + i * TG;
                const float nw = qq == 0 ? float(n0[c]) : qq == 1 ? float(n1[c]) : qq == 2 ? float(n2[c]) : float(n3[c]);
                ready[(size_t(row) * Q + qq) * D + c] = bfloat16_t(m[qq][i] * inv * nw);
            }
        }
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_depth", {"partial", "query", "b0", "b1", "b2", "b3", "n0", "n1", "n2", "n3"},
                                                {"out", "normed", "ready"}, source);
    std::vector<mx::array> in{partial, mx::astype(q, mx::float32)};
    for (int i = 0; i < 4; ++i) in.push_back(i < NB ? mx::astype(blocks[size_t(i)], mx::float32) : inkling2_unused());  // unused slots
    for (int i = 0; i < 4; ++i) in.push_back(i < int(norm_weights.size()) ? mx::astype(norm_weights[size_t(i)], mx::float32) : inkling2_unused());
    auto r = kernel(in, {{n, Q, D}, {n, D}, normed_out ? mx::Shape{n, Q, D} : mx::Shape{1}}, {mx::float32, mx::float32, mx::bfloat16},
                    {256, n, 1}, {256, 1, 1}, {{"D", D}, {"Q", Q}, {"NB", NB}, {"NORMED", normed_out}}, std::nullopt, false,
                    mx::Device(mx::Device::gpu));
    return {r[0], r[1], r[2]};
}
inline std::pair<mlx::core::array, mlx::core::array> inkling2_depth(const std::vector<mlx::core::array>& blocks,
    const mlx::core::array& partial, const mlx::core::array& query) {
    auto r = inkling2_depth_normed(blocks, partial, query, {});
    return {r[0], r[1]};
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// Row RMSNorm (eps 1e-6) read in any float dtype, optionally times a weight (F32, length D), written as `out_type`:
// one kernel in place of a cast in, MLX's rms_norm and a cast out. Rows of D (a multiple of 32, <= 4096) over the last
// axis; one threadgroup a row (D / 32 threads up to 256).
inline mlx::core::array inkling2_norm(const mlx::core::array& x, const std::optional<mlx::core::array>& weight,
                                      mlx::core::Dtype out_type) {
    namespace mx = mlx::core;
    const int D = x.shape(-1), rows = int(x.size() / size_t(D));
    if (D % 32 || D > 4096 || (weight && weight->size() != size_t(D))) throw std::invalid_argument("Inkling2 norm shape");
    static const char* source = R"METAL(
    constexpr int PER = (D + TG - 1) / TG;
    const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const size_t row = threadgroup_position_in_grid.y;
    threadgroup float red[TG / 32];
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int c = int(tid) + i * TG;
        v[i] = c < D ? float(x[row * D + c]) : 0.0f;
        ss += v[i] * v[i];
    }
    const float t = simd_sum(ss);
    if (lane == 0) red[sg] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int g = 0; g < TG / 32; ++g) total += red[g];
    const float inv = rsqrt(total / float(D) + 1e-6f);
    for (int i = 0; i < PER; ++i) {
        const int c = int(tid) + i * TG;
        if (c < D) out[row * D + c] = OUT(WEIGHTED ? v[i] * inv * float(w[c]) : v[i] * inv);
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_norm", {"x", "w"}, {"out"}, source);
    const int tg = std::min(256, D / 32 * 32 >= 256 ? 256 : D);
    auto w = weight ? mx::astype(*weight, mx::float32) : inkling2_unused();
    return kernel({x, w}, {x.shape()}, {out_type}, {tg, rows, 1}, {tg, 1, 1},
                  {{"D", D}, {"TG", tg}, {"WEIGHTED", bool(weight)}, {"OUT", out_type}}, std::nullopt, false,
                  mx::Device(mx::Device::gpu))[0];
}
// L2 clip of each vector of the last axis (length d <= 128): x (any float dtype) optionally times a scale (F32 [d], or
// [heads, d] for x [..., heads, d]),
// then scaled by min(limit / (|x| + 1e-6), 1); written BF16. One simdgroup a vector.
// With row_scale ([rows] F32, x [rows, heads, d]): the clipped BF16 value times row_scale[row], rounded again (the full
// layers' log-position query scale), as bf16(f32(clip(x)) * row_scale) in one kernel.
inline mlx::core::array inkling2_clip(const mlx::core::array& x, float limit, const std::optional<mlx::core::array>& scale,
                                      const std::optional<mlx::core::array>& row_scale = std::nullopt) {
    namespace mx = mlx::core;
    const int d = x.shape(-1), vectors = int(x.size() / size_t(d));
    const int groups = scale ? int(scale->size() / size_t(d)) : 1;
    if (d > 128 || (scale && (scale->size() % size_t(d) || (groups > 1 && x.shape(-2) != groups))))
        throw std::invalid_argument("Inkling2 clip shape");
    static const char* source = R"METAL(
    constexpr int PER = (DIM + 31) / 32;
    const uint lane = thread_index_in_simdgroup;
    const size_t vec_ = size_t(threadgroup_position_in_grid.y) * VPT + simdgroup_index_in_threadgroup;
    if (vec_ >= size_t(count[0])) return;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int c = int(lane) + i * 32;
        v[i] = c < DIM ? float(x[vec_ * DIM + c]) * (SCALED ? float(s[(vec_ % GROUPS) * DIM + c]) : 1.0f) : 0.0f;
        ss += v[i] * v[i];
    }
    const float size = sqrt(simd_sum(ss));
    const float factor = min(float(limit) / (size + 1e-6f), 1.0f);
    for (int i = 0; i < PER; ++i) {
        const int c = int(lane) + i * 32;
        if (c < DIM) {
            const bfloat16_t y = bfloat16_t(v[i] * factor);
            out[vec_ * DIM + c] = ROWED ? bfloat16_t(float(y) * float(rs[vec_ / PER_ROW])) : y;
        }
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_clip", {"x", "s", "limit", "rs", "count"}, {"out"}, source);
    const int per_row = x.ndim() >= 2 ? x.shape(-2) : 1;
    if (row_scale && (x.ndim() < 2 || row_scale->size() * size_t(per_row) != size_t(vectors))) throw std::invalid_argument("Inkling2 clip row scale");
    constexpr int VPT = 8;  // vectors (simdgroups) a threadgroup
    auto s = scale ? mx::astype(*scale, mx::float32) : inkling2_unused();
    return kernel({x, s, mx::array(limit), row_scale ? mx::astype(*row_scale, mx::float32) : inkling2_unused(), inkling2_count(vectors)},
                  {x.shape()}, {mx::bfloat16},
                  {32 * VPT, (vectors + VPT - 1) / VPT, 1}, {32 * VPT, 1, 1},
                  {{"DIM", d}, {"VPT", VPT}, {"SCALED", bool(scale)}, {"GROUPS", groups}, {"ROWED", bool(row_scale)},
                   {"PER_ROW", per_row}}, std::nullopt, false,
                  mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend



namespace tqllm::mlx_backend {
// MoE router after its logits: scores = sigmoid(logits), the top K of scores + bias (largest first, the lowest
// expert on a tie), routes = chosen scores / (their sum + 1e-20) x route_scale. logits [n, E] F32 (E = 32 x PER),
// bias [E]. Returns {ids [n, K] uint32, routes [n, K] F32}. One simdgroup a row: in place of a sigmoid, an add,
// an argsort, a gather, a sum and the scaling.
inline std::pair<mlx::core::array, mlx::core::array> inkling2_route(const mlx::core::array& logits, const mlx::core::array& bias,
                                                                    int k, float route_scale) {
    namespace mx = mlx::core;
    const int n = logits.shape(0), E = logits.shape(1);
    if (logits.ndim() != 2 || E % 32 || E > 256 || bias.size() != size_t(E) || k > 16) throw std::invalid_argument("Inkling2 route shape");
    static const char* source = R"METAL(
    constexpr int PER = E / 32;
    const uint lane = thread_index_in_simdgroup;
    const size_t row = size_t(threadgroup_position_in_grid.y) * 4 + simdgroup_index_in_threadgroup;
    if (row >= size_t(count[0])) return;
    float sc[PER], key[PER];
    for (int i = 0; i < PER; ++i) {
        const int e = int(lane) * PER + i;
        sc[i] = 1.0f / (1.0f + exp(-logits[row * E + e]));
        key[i] = sc[i] + bias[e];
    }
    float chosen[K];
    uint pick[K];
    float total = 0.0f;
    for (int r = 0; r < K; ++r) {
        float best = -INFINITY; uint bi = 0xffffffffu; float bs = 0.0f;
        for (int i = 0; i < PER; ++i) {
            const uint e = lane * PER + uint(i);
            if (key[i] > best || (key[i] == best && e < bi)) { best = key[i]; bi = e; bs = sc[i]; }
        }
        // the simdgroup's best: largest key, then lowest expert
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ob = simd_shuffle_xor(best, o);
            const uint oi = simd_shuffle_xor(bi, o);
            const float os = simd_shuffle_xor(bs, o);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; bs = os; }
        }
        pick[r] = bi; chosen[r] = bs; total += bs;
        if (bi / PER == lane) key[bi % PER] = -INFINITY;
    }
    if (lane == 0)
        for (int r = 0; r < K; ++r) {
            ids[row * K + r] = pick[r];
            routes[row * K + r] = chosen[r] / (total + 1e-20f) * float(rscale);
        }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_route", {"logits", "bias", "rscale", "count"}, {"ids", "routes"}, source);
    auto r = kernel({mx::astype(logits, mx::float32), mx::astype(bias, mx::float32), mx::array(route_scale), inkling2_count(n)}, {{n, k}, {n, k}},
                    {mx::uint32, mx::float32}, {128, (n + 3) / 4, 1}, {128, 1, 1}, {{"E", E}, {"K", k}},
                    std::nullopt, false, mx::Device(mx::Device::gpu));
    return {r[0], r[1]};
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// The static causal convolution in one kernel: y = bf16((sum_tap values[t + tap] * taps[c, tap] + x[t]) / sqrt(5))
// over values = [history (3 rows); x], next history = the last 3 rows of values. x [n, C] in any float dtype, or, with
// experts / routes given, x = bf16(sum_k experts[t, k] * routes[t, k]) (the MoE combine, experts [n, K, C]).
// history [3, C] F32, taps [C, 4] F32. Returns {y [n, C] BF16, next [3, C] F32, x [n, C] BF16 (the input as used)}.
// With sum_type: y is instead (residual + y) written as sum_type, residual [n, C] in any float dtype (y alone in
// sum_type without one), the residual add of the block folded in.
// One thread a channel, the 3-row window in registers: in place of a concatenate and the elementwise chain (and,
// combining, a multiply, a sum and a cast).
inline std::array<mlx::core::array, 3> inkling2_conv(const std::optional<mlx::core::array>& x,
    const std::optional<mlx::core::array>& experts, const std::optional<mlx::core::array>& routes,
    const mlx::core::array& history, const mlx::core::array& taps,
    const std::optional<mlx::core::array>& residual = std::nullopt, std::optional<mlx::core::Dtype> sum_type = std::nullopt,
    const std::optional<mlx::core::array>& order = std::nullopt) {
    // order ([n K] uint32, combining): experts holds the rows in another order ([n K, ..., C]) and pair (t, k) is row
    // order[t K + k] (a grouped expert product read without restoring route order first)
    namespace mx = mlx::core;
    const bool combine = experts.has_value();
    const auto& in = combine ? *experts : *x;
    const int n = order ? routes->shape(0) : in.shape(0), C = in.shape(-1), K = combine ? (order ? routes->shape(1) : in.shape(1)) : 1;
    if (order && (!combine || order->size() != size_t(n) * K || in.size() != size_t(n) * K * C))
        throw std::invalid_argument("Inkling2 conv order");
    if (history.shape() != mx::Shape{3, C} || taps.shape() != mx::Shape{C, 4} || (combine && (!routes || routes->shape() != mx::Shape{n, K})))
        throw std::invalid_argument("Inkling2 conv shape");
    if (residual && (!sum_type || residual->shape() != mx::Shape{n, C})) throw std::invalid_argument("Inkling2 conv residual shape");
    static const char* source = R"METAL(
    const int c = int(thread_position_in_grid.x);
    if (c >= C) return;
    const float t0 = taps[c * 4], t1 = taps[c * 4 + 1], t2 = taps[c * 4 + 2], t3 = taps[c * 4 + 3];
    float h0 = history[c], h1 = history[C + c], h2 = history[2 * C + c];
    const float inv = 1.0f / sqrt(5.0f);
    const int N = count[0];
    for (int t = 0; t < N; ++t) {
        float xv;
        if (COMBINE) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += float(src[size_t(ORDERED ? ord[t * K + k] : uint(t * K + k)) * C + c]) * routes[t * K + k];
            xv = float(bfloat16_t(acc));
        } else {
            xv = float(src[size_t(t) * C + c]);
        }
        xin[size_t(t) * C + c] = bfloat16_t(xv);
        const float yv = float(bfloat16_t((h0 * t0 + h1 * t1 + h2 * t2 + xv * t3 + xv) * inv));
        y[size_t(t) * C + c] = OUT(RESIDUAL ? float(res[size_t(t) * C + c]) + yv : yv);
        h0 = h1; h1 = h2; h2 = xv;
    }
    next[c] = h0; next[C + c] = h1; next[2 * C + c] = h2;
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_conv", {"src", "routes", "history", "taps", "res", "ord", "count"}, {"y", "next", "xin"}, source);
    // Prompt chunks: a thread per (token, channel), each recomputing its 4-row window (the same arithmetic, so the same
    // bits as the serial form, which keeps decode and verify blocks).
    static const char* parallel_source = R"METAL(
    const int c = int(thread_position_in_grid.x), t = int(thread_position_in_grid.y), N = count[0];
    if (c >= C || t >= N) return;
    auto value = [&](int u) -> float {  // row u of [history; x], u from -3
        if (u < 0) return history[(3 + u) * C + c];
        if (COMBINE) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += float(src[size_t(ORDERED ? ord[u * K + k] : uint(u * K + k)) * C + c]) * routes[u * K + k];
            return float(bfloat16_t(acc));
        }
        return float(src[size_t(u) * C + c]);
    };
    const float t0 = taps[c * 4], t1 = taps[c * 4 + 1], t2 = taps[c * 4 + 2], t3 = taps[c * 4 + 3];
    const float h0 = value(t - 3), h1 = value(t - 2), h2 = value(t - 1), xv = value(t);
    const float inv = 1.0f / sqrt(5.0f);
    xin[size_t(t) * C + c] = bfloat16_t(xv);
    const float yv = float(bfloat16_t((h0 * t0 + h1 * t1 + h2 * t2 + xv * t3 + xv) * inv));
    y[size_t(t) * C + c] = OUT(RESIDUAL ? float(res[size_t(t) * C + c]) + yv : yv);
    if (t == N - 1) { next[c] = h1; next[C + c] = h2; next[2 * C + c] = xv; }
)METAL";
    static auto parallel = mx::fast::metal_kernel("tqllm_inkling2_conv_rows", {"src", "routes", "history", "taps", "res", "ord", "count"}, {"y", "next", "xin"},
                                                  parallel_source);
    const bool wide = n > 8;
    const auto out = sum_type.value_or(mx::bfloat16);
    auto r = (wide ? parallel : kernel)({in, combine ? mx::astype(*routes, mx::float32) : inkling2_unused(), mx::astype(history, mx::float32),
                     mx::astype(taps, mx::float32), residual ? *residual : inkling2_unused(),
                     order ? mx::astype(*order, mx::uint32) : inkling2_unused(), inkling2_count(n)},
                    {{n, C}, {3, C}, {n, C}}, {out, mx::float32, mx::bfloat16},
                    wide ? std::tuple<int, int, int>{(C + 255) / 256 * 256, n, 1} : std::tuple<int, int, int>{(C + 255) / 256 * 256, 1, 1},
                    {256, 1, 1},
                    {{"C", C}, {"K", K}, {"COMBINE", combine}, {"RESIDUAL", residual.has_value()}, {"OUT", out},
                     {"ORDERED", order.has_value()}},
                    std::nullopt, false, mx::Device(mx::Device::gpu));
    return {r[0], r[1], r[2]};
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// The attention mask of n queries at positions base .. base + n - 1 over s keys at start .. start + s - 1 plus the
// zero sink key: [1, H, n, s + 1] BF16, the relative bias (bias [n, H, extent]) below rel_extent, 0 beyond it, -inf
// where not visible (causal, and within `window` when window > 0), the sink logit (sinks [n, H]) in the last column.
// One kernel in place of the bias table's concatenate and fills, the column gather and its transpose.
inline mlx::core::array inkling2_mask(const mlx::core::array& bias, const mlx::core::array& sinks, int base, int start, int s,
                                      int window) {
    namespace mx = mlx::core;
    const int n = bias.shape(0), H = bias.shape(1), E = bias.shape(2);
    if (bias.ndim() != 3 || sinks.shape() != mx::Shape{n, H}) throw std::invalid_argument("Inkling2 mask shape");
    static const char* source = R"METAL(
    const int c = int(thread_position_in_grid.x);
    const int N = params[3];
    const int hr = int(thread_position_in_grid.y), h = hr / N, r = hr % N;
    const int s = params[2];
    if (c > s) return;
    float v;
    if (c == s) {
        v = float(sinks[r * H + h]);
    } else {
        const int d = params[0] + r - (params[1] + c);
        const bool visible = d >= 0 && (WINDOW == 0 || d < WINDOW);
        v = !visible ? -INFINITY : d < E ? float(bias[(size_t(r) * H + h) * E + d]) : 0.0f;
    }
    out[(size_t(h) * N + r) * (s + 1) + c] = bfloat16_t(v);
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_mask", {"bias", "sinks", "params"}, {"out"}, source);
    const int32_t params[4] = {base, start, s, n};
    return kernel({bias, sinks, mx::array(params, {4}, mx::int32)}, {{1, H, n, s + 1}}, {mx::bfloat16},
                  {(s + 1 + 255) / 256 * 256, H * n, 1}, {256, 1, 1}, {{"H", H}, {"E", E}, {"WINDOW", window}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// The exact greedy LM head from an approximate one, for each of n inputs x [n, D]. approx [n, rows] F32 = x times an
// approximation of each row of the BF16 head w [rows, D]; bound [rows] F32 with |x . w_r - approx_r| <= |x| * bound_r
// (the approximation's row error norm plus a float accumulation term); norm [n] = |x|; best [n] = argmax approx.
// A row can only win when its upper bound reaches the best row's lower bound, less two BF16 ulps of separation: those
// rows get the reference logit cap * tanh(bf16(x . w_r) / cap), every other row -inf. Returns [n, V] F32 whose argmax
// (lowest index on ties, as for the full head) is the full BF16 head's greedy token. One simdgroup a row; rows ruled
// out read 8 bytes.
inline mlx::core::array inkling2_head_select(const mlx::core::array& x, const mlx::core::array& w, const mlx::core::array& approx,
                                             const mlx::core::array& bound, const mlx::core::array& norm,
                                             const mlx::core::array& best, int V, float cap) {
    namespace mx = mlx::core;
    const int D = w.shape(1), n = int(x.size() / size_t(D)), rows = w.shape(0);
    if (x.size() != size_t(n) * D || D % 256 || rows < V || approx.size() != size_t(n) * rows || bound.size() < size_t(V) ||
        norm.size() != size_t(n) || best.size() != size_t(n))
        throw std::invalid_argument("Inkling2 head select shape");
    static const char* source = R"METAL(
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup, i = threadgroup_position_in_grid.y;
    if (r >= uint(V)) return;
    const device float* a = approx + size_t(i) * ROWS;
    const device bfloat16_t* xi = x + size_t(i) * D;
    const float nx = norm[i];
    const uint b = best[i];
    const float A = a[b], margin = nx * bound[b];
    const float floor_ = A - margin - (fabs(A) + margin + 1.0f) * (1.0f / 64.0f) - 1e-5f;
    if (a[r] + nx * bound[r] + 1e-5f < floor_) {
        if (lane == 0) out[size_t(i) * V + r] = -INFINITY;
        return;
    }
    float acc = 0.0f;
    const device bfloat16_t* row = w + size_t(r) * D;
    for (int k = int(lane) * 8; k < D; k += 256)
        for (int j = 0; j < 8; ++j) acc += float(xi[k + j]) * float(row[k + j]);
    acc = simd_sum(acc);
    if (lane == 0) out[size_t(i) * V + r] = float(cap) * metal::precise::tanh(float(bfloat16_t(acc)) / float(cap));
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_head_select", {"x", "w", "approx", "bound", "norm", "best", "cap"},
                                                {"out"}, source);
    return kernel({mx::reshape(mx::astype(x, mx::bfloat16), {n, D}), w, mx::reshape(approx, {n, rows}), bound, mx::reshape(norm, {n}),
                   mx::reshape(mx::astype(best, mx::uint32), {n}), mx::array(cap)},
                  {{n, V}}, {mx::float32}, {(V + 7) / 8 * 256, n, 1}, {256, 1, 1}, {{"D", D}, {"V", V}, {"ROWS", rows}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// Prompt-chunk attention on the matrix unit, the bias and mask computed in place (no mask tensor, no key copies):
// q [1, H, n, 128] for rows at positions base ..; keys / values the cache views [1, G, s + 1, 128] (rows at positions
// start .., the last the zero sink row, read in place through their strides); bias [n, H, E] the relative bias by
// distance (0 at distance >= E); sinks [n, H] each row's sink logit; window > 0 for a sliding layer. Row r sees key
// c when 0 <= d < window (or d >= 0), d = base + r - start - c; logits q . k * scale + bias, softmax with the sink
// (logit sink, value 0). A threadgroup owns 32 rows of the query heads that share a key / value head (6 simdgroups for
// Inkling's 3) and walks only the 32-key tiles they can see, staged once in threadgroup memory for all of them; each
// simdgroup: 16 rows, scores with 8 16x32x16 products, online softmax, then 8 products into O [16, 128] (F32).
inline mlx::core::array inkling2_prompt_attention(const mlx::core::array& q, const mlx::core::array& keys, const mlx::core::array& values,
                                                  const mlx::core::array& bias, const mlx::core::array& sinks, int base, int start, int s,
                                                  int window, float scale) {
    namespace mx = mlx::core;
    const int H = q.shape(1), n = q.shape(2), G = keys.shape(1), E = bias.shape(2);
    if (q.ndim() != 4 || q.shape(3) != 128 || keys.shape(3) != 128 || values.shape() != keys.shape() || keys.shape(2) < s ||
        H % G || H / G > 8 || bias.shape(0) != n || bias.shape(1) != H || sinks.shape() != mx::Shape{n, H})
        throw std::invalid_argument("Inkling2 prompt attention shape");
    static const char* source = R"METAL(
    const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    // a threadgroup: the HPG query heads of key/value head g times RB blocks of 16 rows, sharing each K/V tile
    constexpr int HPG = H / G;
    const int g = int(threadgroup_position_in_grid.z), h = g * HPG + int(sg) % HPG;
    const int base = params[0], start = params[1], s = params[2], n = params[3];
    const int r_tg = int(threadgroup_position_in_grid.y) * (16 * RB), r0 = r_tg + (int(sg) / HPG) * 16;
    const int r_last = min(n, r_tg + 16 * RB) - 1;
    const int c_lo = WINDOW > 0 ? max(0, base + r_tg - (WINDOW - 1) - start) : 0;
    const int c_hi = min(s, base + r_last - start + 1);
    // the cache views as MLX lays them out at evaluation (keys_strides / values_strides: [1, G, rows, 128])
    const device bfloat16_t* kh = keys + size_t(g) * keys_strides[1];
    const device bfloat16_t* vh = values + size_t(g) * values_strides[1];
    threadgroup bfloat16_t Ks[32][136];   // [key][dim]: both operands read as 4 consecutive dims a lane
    threadgroup bfloat16_t Vs[32][136];
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    // scores: the right operand transposed, so K fragments come straight from the cache rows (4 dims a lane)
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto descv = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<bfloat16_t, bfloat16_t, float>();
    auto cb = op.template get_right_input_cooperative_tensor<bfloat16_t, bfloat16_t, float>();
    // the value product in FP16: probabilities in [0, 1] (11-bit, finer than BF16), values RMS-normalised (|v| <= sqrt(128)),
    // so both fit FP16 with the values' BF16 bits kept
    mpp::tensor_ops::matmul2d<descv, metal::execution_simdgroup> opv;
    auto cpf = opv.template get_left_input_cooperative_tensor<float, float, float>();
    auto cbv = opv.template get_right_input_cooperative_tensor<float, float, float>();
    auto cs = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    auto o0 = opv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(cpf)>, metal::remove_addrspace_t<decltype(cbv)>, float>();
    auto o1 = opv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(cpf)>, metal::remove_addrspace_t<decltype(cbv)>, float>();
    auto o2 = opv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(cpf)>, metal::remove_addrspace_t<decltype(cbv)>, float>();
    auto o3 = opv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(cpf)>, metal::remove_addrspace_t<decltype(cbv)>, float>();
    for (short i = 0; i < 16; i++) { o0[i] = 0.0f; o1[i] = 0.0f; o2[i] = 0.0f; o3[i] = 0.0f; }
    // this lane's two rows: fm and fm + 8 of the simdgroup's 16
    const int ra = r0 + fm, rb = r0 + fm + 8;
    const bool oka = ra < n, okb = rb < n;
    // the query view as laid out ([1, H, n, 128], any head / row strides, e.g. the transposed [n, H, 128] rows)
    const device bfloat16_t* qa = q + size_t(h) * q_strides[1] + size_t(oka ? ra : 0) * q_strides[2] + fn;
    const device bfloat16_t* qb = q + size_t(h) * q_strides[1] + size_t(okb ? rb : 0) * q_strides[2] + fn;
    float m[2] = {oka ? float(sinks[ra * H + h]) : 0.0f, okb ? float(sinks[rb * H + h]) : 0.0f};
    float l[2] = {1.0f, 1.0f};
    for (int c0 = c_lo & ~31; c0 < c_hi; c0 += 32) {
        {   // the tile's K and V rows into threadgroup memory, 256 key-parts (one key, 16 dims) over the threads
            for (uint q = tid; q < 256u; q += uint(32 * RB * HPG)) {
                const int key = int(q >> 3), part = int(q & 7u) * 16, c = min(c0 + key, s);
                const device bfloat16_t* kr = kh + size_t(c) * keys_strides[2] + part;
                const device bfloat16_t* vr = vh + size_t(c) * values_strides[2] + part;
                typedef vec<bfloat16_t, 4> b4l;
                for (int j = 0; j < 16; j += 4) {
                    *(threadgroup b4l*)&Ks[key][part + j] = *(const device b4l*)(kr + j);
                    *(threadgroup b4l*)&Vs[key][part + j] = *(const device b4l*)(vr + j);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) cs[i] = 0.0f;
        for (int ks = 0; ks < 128; ks += 16) {
            typedef vec<bfloat16_t, 4> b4;
            const b4 a0 = oka ? *(const device b4*)(qa + ks) : b4(bfloat16_t(0.0f));
            const b4 a1 = okb ? *(const device b4*)(qb + ks) : b4(bfloat16_t(0.0f));
            const b4 b0 = *(const threadgroup b4*)&Ks[fm][ks + fn], b1 = *(const threadgroup b4*)&Ks[8 + fm][ks + fn];
            const b4 b2 = *(const threadgroup b4*)&Ks[16 + fm][ks + fn], b3 = *(const threadgroup b4*)&Ks[24 + fm][ks + fn];
            for (short i = 0; i < 4; i++) { ca[i] = a0[i]; ca[4 + i] = a1[i]; cb[i] = b0[i]; cb[4 + i] = b1[i]; cb[8 + i] = b2[i]; cb[12 + i] = b3[i]; }
            op.run(ca, cb, cs);
        }
        // logits with bias and mask; element i: row fm + 8 ((i & 7) >> 2), key c0 + fn + 16 (i >> 3) + (i & 3)
        float p[16], mx_[2] = {-INFINITY, -INFINITY};
        // a tile every row of the threadgroup sees whole, at distance >= E from all of them: no bias, no mask
        const bool far = c0 + 31 < s && base + r_tg - start - (c0 + 31) >= 0 && base + r_last - start - c0 < (WINDOW > 0 ? WINDOW : 1 << 30) &&
                         base + r_tg - start - (c0 + 31) >= E;
        // a tile every row sees whole within the window (and at distance < E, so always biased): no mask checks
        const bool inner = !far && c0 + 31 < s && base + r_tg - start - (c0 + 31) >= 0 &&
                           base + r_last - start - c0 < min(WINDOW > 0 ? WINDOW : 1 << 30, E);
        if (inner) {
            for (short i = 0; i < 16; i++) {
                const int rr = ((i & 7) >> 2), r = rr ? rb : ra, c = c0 + fn + (i >> 3) * 16 + (i & 3);
                p[i] = (rr ? okb : oka) ? float(cs[i]) * scale + float(bias[(size_t(r) * H + h) * E + (base + r - start - c)]) : -INFINITY;
                mx_[rr] = max(mx_[rr], p[i]);
            }
        } else if (far) {
            for (short i = 0; i < 16; i++) {
                const int rr = ((i & 7) >> 2);
                p[i] = (rr ? okb : oka) ? float(cs[i]) * scale : -INFINITY;
                mx_[rr] = max(mx_[rr], p[i]);
            }
        } else
        for (short i = 0; i < 16; i++) {
            const int rr = ((i & 7) >> 2), r = rr ? rb : ra, c = c0 + fn + (i >> 3) * 16 + (i & 3);
            const int d = base + r - start - c;
            const bool vis = (rr ? okb : oka) && c < s && d >= 0 && (WINDOW == 0 || d < WINDOW);
            p[i] = vis ? float(cs[i]) * scale + (d < E ? float(bias[(size_t(r) * H + h) * E + d]) : 0.0f) : -INFINITY;
            mx_[rr] = max(mx_[rr], p[i]);
        }
        float alpha[2], sum[2] = {0.0f, 0.0f};
        for (int rr = 0; rr < 2; rr++) {
            float v = mx_[rr];
            v = max(v, simd_shuffle_xor(v, 1)); v = max(v, simd_shuffle_xor(v, 8));
            const float mn = max(m[rr], v);
            alpha[rr] = exp(m[rr] - mn);
            m[rr] = mn;
        }
        for (short i = 0; i < 16; i++) { const int rr = ((i & 7) >> 2); p[i] = exp(p[i] - m[rr]); sum[rr] += p[i]; }
        for (int rr = 0; rr < 2; rr++) {
            float v = sum[rr];
            v += simd_shuffle_xor(v, 1); v += simd_shuffle_xor(v, 8);
            l[rr] = l[rr] * alpha[rr] + v;
        }
        for (short i = 0; i < 16; i++) {
            const float a = alpha[(i & 7) >> 2];
            o0[i] *= a; o1[i] *= a; o2[i] *= a; o3[i] *= a;
        }
        // O += P V over the block's two 16-key halves; P's score layout is the left operand's
        for (int kh2 = 0; kh2 < 2; kh2++) {
            for (short i = 0; i < 8; i++) cpf[i] = p[kh2 * 8 + i];
            for (int b = 0; b < 4; b++) {
                const int d0 = b * 32;
                typedef vec<bfloat16_t, 4> b4;
                const b4 b0 = *(const threadgroup b4*)&Vs[kh2 * 16 + fm][d0 + fn], b1 = *(const threadgroup b4*)&Vs[kh2 * 16 + 8 + fm][d0 + fn];
                const b4 b2 = *(const threadgroup b4*)&Vs[kh2 * 16 + fm][d0 + 16 + fn], b3 = *(const threadgroup b4*)&Vs[kh2 * 16 + 8 + fm][d0 + 16 + fn];
                for (short i = 0; i < 4; i++) { cbv[i] = float(b0[i]); cbv[4 + i] = float(b1[i]); cbv[8 + i] = float(b2[i]); cbv[12 + i] = float(b3[i]); }
                if (b == 0) opv.run(cpf, cbv, o0); else if (b == 1) opv.run(cpf, cbv, o1); else if (b == 2) opv.run(cpf, cbv, o2); else opv.run(cpf, cbv, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short i = 0; i < 16; i++) {
        const int rr = ((i & 7) >> 2), r = rr ? rb : ra;
        if (!(rr ? okb : oka)) continue;
        const int dim = fn + (i >> 3) * 16 + (i & 3);
        device bfloat16_t* orow = out + (size_t(h) * n + r) * 128;
        const float inv = 1.0f / l[rr];
        orow[dim] = bfloat16_t(o0[i] * inv); orow[32 + dim] = bfloat16_t(o1[i] * inv);
        orow[64 + dim] = bfloat16_t(o2[i] * inv); orow[96 + dim] = bfloat16_t(o3[i] * inv);
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_prompt_attention", {"q", "keys", "values", "bias", "sinks", "params", "scale"},
                                                {"out"}, source,
                                                "#include <metal_simdgroup>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n",
                                                false);
    const int32_t params[4] = {base, start, s, n};
    // row blocks of 16 a head in a threadgroup: 2 (32 rows, 6 simdgroups) measured fastest, sliding and full (1, 3, 4
    // slower; K / V read straight from device memory instead of staged slower still)
    constexpr int RB = 2;
    if (q.dtype() != mx::bfloat16 || q.strides(3) != 1) throw std::invalid_argument("Inkling2 prompt attention query layout");
    return kernel({q, keys, values, mx::contiguous(mx::astype(bias, mx::bfloat16)),
                   mx::contiguous(mx::astype(sinks, mx::float32)), mx::array(params, {4}, mx::int32), mx::array(scale)},
                  {{1, H, n, 128}}, {mx::bfloat16}, {32 * RB * (H / G), (n + 16 * RB - 1) / (16 * RB), G}, {32 * RB * (H / G), 1, 1},
                  {{"H", H}, {"G", G}, {"E", E}, {"WINDOW", window}, {"RB", RB}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// A message gate's merge and the RMSNorm after it, for prompt chunks: fg [n, 2D] F32 = low times [forget; gain],
// y = (1 - tanh(fg[:, :D])) h + tanh(fg[:, D:]) m in F32, then y RMS-normalised (eps 1e-6) times weight, written BF16.
// One threadgroup a row (the norm's layout, inkling2_norm): in place of the tanh / merge kernel and the norm, without the
// F32 merge in memory.
inline mlx::core::array inkling2_gate_merge(const mlx::core::array& fg, const mlx::core::array& h, const mlx::core::array& m,
                                            const mlx::core::array& weight) {
    namespace mx = mlx::core;
    const int D = h.shape(-1), rows = int(h.size() / size_t(D));
    if (fg.shape() != mx::Shape{rows, 2 * D} || m.size() != h.size() || weight.size() != size_t(D) || D % 32 || D > 4096)
        throw std::invalid_argument("Inkling2 gate merge shape");
    static const char* source = R"METAL(
    constexpr int PER = (D + TG - 1) / TG;
    const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const size_t row = threadgroup_position_in_grid.y;
    threadgroup float red[TG / 32];
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int c = int(tid) + i * TG;
        if (c < D) {
            const float f = metal::precise::tanh(fg[row * 2 * D + c]), g = metal::precise::tanh(fg[row * 2 * D + D + c]);
            v[i] = (1.0f - f) * float(h[row * D + c]) + g * float(m[row * D + c]);
        } else {
            v[i] = 0.0f;
        }
        ss += v[i] * v[i];
    }
    const float t = simd_sum(ss);
    if (lane == 0) red[sg] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int k = 0; k < TG / 32; ++k) total += red[k];
    const float inv = rsqrt(total / float(D) + 1e-6f);
    for (int i = 0; i < PER; ++i) {
        const int c = int(tid) + i * TG;
        if (c < D) out[row * D + c] = bfloat16_t(v[i] * inv * float(w[c]));
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_gate_merge", {"fg", "h", "m", "w"}, {"out"}, source);
    const int tg = std::min(256, D);
    return kernel({mx::astype(fg, mx::float32), h, mx::astype(m, mx::float32), mx::astype(weight, mx::float32)}, {h.shape()}, {mx::bfloat16},
                  {tg, rows, 1}, {tg, 1, 1}, {{"D", D}, {"TG", tg}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// A prompt chunk's key or value preparation in one kernel: the causal convolution of x [n, C] (C = heads x 128) over
// [history (3 rows F32); x], static (taps [C, 4]: y = (sum_tap v[t - 3 + tap] taps[c, tap] + x) / sqrt(5)) or dynamic
// (taps [C, 4] by lag plus dyn [n, 4 heads] by row, read in F32: y = (sum_lag v[t - lag] (taps[c, lag] + dyn[t, g, lag]) + x) / 3),
// rounded to BF16, then per head of 128 either the L2 clip to `limit` (keys) or the unweighted RMSNorm (values).
// Returns {out [n, C] BF16, next history [3, C] F32}. A threadgroup of 128 a (row, head): in place of the convolution,
// the reshape and the clip or norm kernels, with the same arithmetic as inkling2_conv / inkling2_clip / inkling2_norm.
inline std::pair<mlx::core::array, mlx::core::array> inkling2_kv_prep(const mlx::core::array& x, const mlx::core::array& history,
    const mlx::core::array& taps, const std::optional<mlx::core::array>& dyn, bool clip, float limit) {
    namespace mx = mlx::core;
    const int n = x.shape(0), C = x.shape(1), heads = C / 128;
    if (x.ndim() != 2 || C % 128 || history.shape() != mx::Shape{3, C} || taps.shape() != mx::Shape{C, 4} ||
        (dyn && dyn->shape() != mx::Shape{n, 4 * heads}))
        throw std::invalid_argument("Inkling2 kv prep shape");
    static const char* source = R"METAL(
    const int c_in = int(thread_position_in_threadgroup.x), g = int(threadgroup_position_in_grid.x), t = int(threadgroup_position_in_grid.y);
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int c = g * 128 + c_in;
    auto value = [&](int u) -> float { return u < 0 ? history[(3 + u) * C + c] : float(src[size_t(u) * C + c]); };
    const float h0 = value(t - 3), h1 = value(t - 2), h2 = value(t - 1), xv = value(t);
    float yv;
    if (DYNAMIC) {
        const auto dr = dyn + size_t(t) * (4 * HEADS) + g * 4;  // constant space when unused
        float sum = xv * (taps[c * 4] + float(dr[0]));
        sum += h2 * (taps[c * 4 + 1] + float(dr[1]));
        sum += h1 * (taps[c * 4 + 2] + float(dr[2]));
        sum += h0 * (taps[c * 4 + 3] + float(dr[3]));
        yv = float(bfloat16_t((sum + xv) / 3.0f));
    } else {
        yv = float(bfloat16_t((h0 * taps[c * 4] + h1 * taps[c * 4 + 1] + h2 * taps[c * 4 + 2] + xv * taps[c * 4 + 3] + xv) *
                              (1.0f / sqrt(5.0f))));
    }
    if (t == count[0] - 1) { next[c] = h1; next[C + c] = h2; next[2 * C + c] = xv; }
    threadgroup float row[128];
    threadgroup float red[4];
    float factor;
    if (CLIP) {
        // inkling2_clip's order: lane l sums channels l, l + 32, l + 64, l + 96, then one simd_sum
        row[c_in] = yv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float ss = 0.0f;
            for (int i = 0; i < 4; ++i) { const float v = row[lane + i * 32]; ss += v * v; }
            const float size = sqrt(simd_sum(ss));
            if (lane == 0) red[0] = min(float(limit[0]) / (size + 1e-6f), 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        factor = red[0];
    } else {
        // inkling2_norm's order at D = 128: one channel a thread, simd_sum, then the 4 simdgroups in order
        const float s = simd_sum(yv * yv);
        if (lane == 0) red[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total = red[0] + red[1] + red[2] + red[3];
        factor = rsqrt(total / 128.0f + 1e-6f);
    }
    out[size_t(t) * C + c] = bfloat16_t(yv * factor);
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_kv_prep", {"src", "history", "taps", "dyn", "limit", "count"}, {"out", "next"}, source);
    return [&] {
        auto r = kernel({x, mx::astype(history, mx::float32), mx::astype(taps, mx::float32), dyn ? mx::astype(*dyn, mx::float32) : inkling2_unused(),
                         mx::array(&limit, {1}, mx::float32), inkling2_count(n)},
                        {{n, C}, {3, C}}, {mx::bfloat16, mx::float32}, {128 * heads, n, 1}, {128, 1, 1},
                        {{"C", C}, {"HEADS", heads}, {"DYNAMIC", dyn.has_value()}, {"CLIP", clip}}, std::nullopt, false,
                        mx::Device(mx::Device::gpu));
        return std::pair<mx::array, mx::array>{r[0], r[1]};
    }();
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// A verify block's routed pairs sorted by expert in one kernel: ids [n, K] (P = n K <= 1024) -> [3, P] uint32 rows
// {sorted expert ids, each sorted pair's token row, restore: each pair's sorted position}. A stable sort (pairs of one
// expert keep their route order), one thread a pair counting the pairs before it: in place of two argsorts, a take,
// a divide and a cast.
inline mlx::core::array inkling2_pair_sort(const mlx::core::array& ids) {
    namespace mx = mlx::core;
    const int n = ids.shape(0), K = ids.shape(1), P = n * K;
    if (ids.ndim() != 2 || P > 1024) throw std::invalid_argument("Inkling2 pair sort shape");
    static const char* source = R"METAL(
    const uint p = thread_position_in_threadgroup.x;
    const uint P = uint(count[0]);
    threadgroup uint keys[1024];
    if (p < P) keys[p] = ids[p];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p >= P) return;
    const uint key = keys[p];
    uint rank = 0;
    for (uint q = 0; q < P; ++q) rank += (keys[q] < key || (keys[q] == key && q < p)) ? 1u : 0u;
    out[rank] = key;
    out[P + rank] = p / uint(K);
    out[2 * P + p] = rank;
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_pair_sort", {"ids", "count"}, {"out"}, source);
    const int tg = (P + 31) / 32 * 32;
    return kernel({mx::astype(mx::reshape(ids, {P}), mx::uint32), inkling2_count(P)}, {{3, P}}, {mx::uint32}, {tg, 1, 1}, {tg, 1, 1},
                  {{"K", K}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

namespace tqllm::mlx_backend {
// A verify block's rollback of one convolution history: the rows after the first `keep` of [before (3 rows F32); input
// (n rows, any float dtype)], i.e. the history once only the kept inputs have been consumed: next[j] = before[j + keep]
// for j + keep < 3, else input[j + keep - 3]. [3, C] F32, one kernel in place of a cast, a concatenate and a slice.
inline mlx::core::array inkling2_history_rollback(const mlx::core::array& before, const mlx::core::array& input, int keep) {
    namespace mx = mlx::core;
    const int C = before.shape(1), n = input.shape(0);
    if (before.shape() != mx::Shape{3, C} || input.ndim() != 2 || input.shape(1) != C || keep < 0 || keep > n)
        throw std::invalid_argument("Inkling2 history rollback shape");
    static const char* source = R"METAL(
    const uint c = thread_position_in_grid.x, j = thread_position_in_grid.y;
    if (c >= uint(C)) return;
    const int at = int(j) + keep_[0];
    next[j * C + c] = at < 3 ? before[size_t(at) * C + c] : float(input[size_t(at - 3) * C + c]);
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_history_rollback", {"before", "input", "keep_"}, {"next"}, source);
    return kernel({mx::astype(before, mx::float32), input, inkling2_count(keep)}, {{3, C}}, {mx::float32}, {(C + 255) / 256 * 256, 3, 1},
                  {256, 1, 1}, {{"C", C}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

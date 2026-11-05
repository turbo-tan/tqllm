#pragma once
// QSA (Qwen Sparse Attention, arXiv 2608.30320) for the BF16 decode / verify path, with the indexers of
// higa2515/Qwen3.8-27B-QSA-BF16 (trained on the frozen Qwen3.8-27B backbone, so they read any quantization of it):
// per full-attention layer a 4-head MQA indexer scores 4-token key blocks, and a row attends to its top 512 blocks
// (2,048 keys) plus its recent keys. Our variant keeps decode_attention_exact's split: blocks are chosen only among
// the keys below the row's T(x) (attention_exact.hpp), which are committed and the same for a tree row and its AR step,
// and the keys from T(x) to the row (16 to 79) stay dense in the exact loop. A row is sparse when T(x) >= `from`;
// below that it runs the dense exact kernel. Every value a row uses depends only on its own query and the committed
// cache, in a fixed order, so a tree row equals its one-token step bit for bit.
#include <mlx/mlx.h>
#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace tqllm::mlx_backend {
constexpr int kQsaBlock = 4, kQsaPick = 512, kQsaHeads = 4, kQsaDim = 128, kQsaRope = 64, kQsaChunks = 4, kQsaSplit = 8;
constexpr int kQsaFrom = 49152;  // default --qsa-from: below 48K keys the dense 16-row kernel reads about as much (j1887)

struct QsaLayer {
    mlx::core::array w;                           // [512 query | 128 key, hidden] FP16, one projection launch
    mlx::core::array q_norm, k_norm, block_norm;  // [128] FP32, the zero-centered scale folded (1 + w)
};

// what decode_attention_exact needs for sparse rows: the rows' indexer queries [n, 4, 128] (normed, rotated) and the
// layer's block keys [>= T / 4, 128] (pooled, normed, rotated at the block start)
struct QsaRows {
    mlx::core::array q, blocks;
    int from = kQsaFrom;  // rows with T >= from are sparse (--qsa-from)
};

// x [m, hidden] FP32 times rows o0 .. o0 + count - 1 of w [out, hidden]^T, as kQsaSplit partial sums over hidden slices:
// [kQsaSplit, m, count] (qsa_norm_rope adds them in order). A threadgroup stages its slice of up to 16 rows of x once
// (two 320-wide halves) and its 8 simdgroups take 4 outputs each; every row's dot products run in a fixed order whatever
// m is, so a verify row and its AR step get the same bits.
inline mlx::core::array qsa_project(const mlx::core::array& x, const mlx::core::array& w, int o0, int count) {
    namespace mx = mlx::core;
    static auto kernel = mx::fast::metal_kernel("tqllm_qsa_project", {"X", "W", "P"}, {"Y"}, R"METAL(
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = thread_position_in_threadgroup.x;
    const uint split = threadgroup_position_in_grid.z % uint(SPLIT), r0 = threadgroup_position_in_grid.z / uint(SPLIT) * 16u;
    const int M = P[0], O0 = P[1], N = P[2];
    const uint ob = threadgroup_position_in_grid.y * 32u + sg * 4u;  // this simdgroup's first output (of N)
    constexpr uint SLICE = uint(H / SPLIT), HALF = SLICE / 2u;
    threadgroup float xs[16 * HALF];
    float acc[4][16];
    for (uint k = 0; k < 4u; ++k) for (uint i = 0; i < 16u; ++i) acc[k][i] = 0.0f;
    for (uint part = 0; part < 2u; ++part) {
        const uint j0 = split * SLICE + part * HALF;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < 16u * HALF; e += 256u) {
            const uint i = e / HALF, j = e % HALF;
            xs[e] = r0 + i < uint(M) ? X[size_t(r0 + i) * uint(H) + j0 + j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = lane; j < HALF; j += 32u) {
            for (uint k = 0; k < 4u; ++k) {
                if (ob + k >= uint(N)) continue;
                const float wv = float(W[size_t(uint(O0) + ob + k) * uint(H) + j0 + j]);
                for (uint i = 0; i < 16u; ++i) acc[k][i] = fma(wv, xs[i * HALF + j], acc[k][i]);
            }
        }
    }
    for (uint k = 0; k < 4u; ++k)
        for (uint i = 0; i < 16u; ++i) {
            const float v = simd_sum(acc[k][i]);
            if (lane == 0u && ob + k < uint(N) && r0 + i < uint(M)) Y[(size_t(split) * uint(M) + r0 + i) * uint(N) + ob + k] = v;
        }
)METAL");
    const int m = x.shape(0), hidden = x.shape(1);
    if (hidden % (64 * kQsaSplit)) throw std::invalid_argument("QSA projection: hidden size must split into 64-wide slices");
    const std::vector<int32_t> p = {m, o0, count};
    return kernel({mx::contiguous(mx::astype(x, mx::float32)), w, mx::array(p.data(), {3}, mx::int32)}, {{kQsaSplit, m, count}},
                  {mx::float32}, {256, (count + 31) / 32, (m + 15) / 16 * kQsaSplit}, {256, 1, 1},
                  {{"H", hidden}, {"SPLIT", kQsaSplit}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// rows [r, 128] (or [r, 4 x 128] by head): RMS norm with `scale`, then (rope) NeoX RoPE on the first 64 channels at
// pos[r] (blocks: 4 x the block index). Each row in a fixed order. pool: rows are 4-token groups of `x` averaged first;
// otherwise x may be qsa_project's [kQsaSplit, r, 128] partial sums, added in order first.
inline mlx::core::array qsa_norm_rope(const mlx::core::array& x, const mlx::core::array& scale, const mlx::core::array& pos,
                                      float theta, float eps, bool rope, bool pool) {
    namespace mx = mlx::core;
    static auto kernel = mx::fast::metal_kernel("tqllm_qsa_norm_rope", {"X", "S", "POS", "TH", "EPS"}, {"Y"}, R"METAL(
    const uint d = thread_position_in_threadgroup.x, r = threadgroup_position_in_grid.y;
    threadgroup float part[4];
    threadgroup float y[128];
    float v;
    if (POOL) {
        const size_t at = size_t(r) * 512u + d;
        v = (((float(X[at]) + float(X[at + 128u])) + float(X[at + 256u])) + float(X[at + 384u])) * 0.25f;
    } else {
        v = 0.0f;
        for (uint s = 0; s < uint(SPLIT); ++s) v += float(X[(size_t(s) * uint(ROWS) + r) * 128u + d]);
    }
    const float ss = simd_sum(v * v);
    if (thread_index_in_simdgroup == 0u) part[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = ((part[0] + part[1]) + part[2]) + part[3];
    const float yv = v * rsqrt(total / 128.0f + EPS) * S[d];
    y[d] = yv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = yv;
    if (ROPE && d < 64u) {
        const uint i = d & 31u;
        const float angle = float(POS[r / uint(PER)]) * pow(TH, -float(2u * i) / 64.0f);
        const float c = cos(angle), s = sin(angle);
        o = d < 32u ? y[d] * c - y[d + 32u] * s : y[d] * c + y[d - 32u] * s;
    }
    Y[size_t(r) * 128u + d] = o;
)METAL");
    const int split = !pool && x.ndim() == 3 ? x.shape(0) : 1;
    const int rows = pool ? int(x.size() / 512) : int(x.size() / 128) / split;
    const int per = pool ? 1 : std::max(1, rows / std::max(1, int(pos.size())));  // heads per position (query rows)
    return kernel({mx::contiguous(x), scale, pos, mx::array(theta), mx::array(eps)}, {{rows, 128}}, {mx::float32},
                  {128, rows, 1}, {128, 1, 1}, {{"ROPE", int(rope)}, {"POOL", int(pool)}, {"PER", per}, {"SPLIT", split}, {"ROWS", rows}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// top kQsaPick blocks (ties: the lower block) among the blocks below each sparse row's T, ascending; -1 rows elsewhere.
// q [16, 4, 128], blocks [>= nbmax, 128], plan as decode_attention_exact's; returns [16, kQsaPick] int32
inline mlx::core::array qsa_select(const mlx::core::array& q, const mlx::core::array& blocks, const mlx::core::array& plan,
                                   int n, int nbmax, int pw, int from) {
    namespace mx = mlx::core;
    // scores of every row against every block below the bound: a threadgroup takes 16 blocks x 8 rows, both staged in
    // threadgroup memory, so the block keys are read once per 8 rows (coalesced); each score in the same fixed order
    static auto scores = mx::fast::metal_kernel("tqllm_qsa_scores", {"Q", "BLK", "P"}, {"S"}, R"METAL(
    const uint tid = thread_position_in_threadgroup.x, b0 = threadgroup_position_in_grid.x * 16u, half_ = threadgroup_position_in_grid.y;
    const uint NB = uint(P[2]);
    threadgroup float q[8 * 512];
    threadgroup float kb[16 * 129];
    for (uint i = tid; i < 4096u; i += 128u) q[i] = Q[half_ * 4096u + i];
    for (uint i = tid; i < 2048u; i += 128u) {
        const uint b = b0 + (i >> 7);
        kb[(i >> 7) * 129u + (i & 127u)] = b < NB ? BLK[size_t(b) * 128u + (i & 127u)] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bl = tid & 15u, r = tid >> 4, b = b0 + bl, t = half_ * 8u + r;
    if (b >= NB) return;
    float s = 0.0f;
    for (uint h = 0; h < 4u; ++h) {
        float d = 0.0f;
        for (uint e = 0; e < 128u; ++e) d = fma(q[r * 512u + h * 128u + e], kb[bl * 129u + e], d);
        s += max(d, 0.0f);
    }
    S[size_t(t) * NB + b] = s * 0.08838834764831845f;  // 128^-0.5; scores >= +0, so their bits order as the values
)METAL");
    static auto kernel = mx::fast::metal_kernel("tqllm_qsa_select", {"SC", "PLAN", "P"}, {"SEL"}, R"METAL(
    const uint tid = thread_position_in_threadgroup.x, t = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const int T = PLAN[t * uint(PW)];
    if (int(t) >= P[0] || T < P[1]) {
        for (uint i = tid; i < uint(PICK); i += 1024u) SEL[t * uint(PICK) + i] = -1;
        return;
    }
    const uint nb = uint(T) / 4u;
    const device float* S = SC + size_t(t) * uint(P[2]);
    // radix select of the PICK-th largest bit pattern, 8 bits at a time from the top
    threadgroup atomic_uint hist[256];
    threadgroup uint sh[3];  // prefix, mask, still wanted at the prefix
    if (tid == 0u) {sh[0] = 0u; sh[1] = 0u; sh[2] = uint(PICK);}
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256u) atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint pre = sh[0], msk = sh[1];
        for (uint b = tid; b < nb; b += 1024u) {
            const uint u = as_type<uint>(S[b]);
            if ((u & msk) == pre) atomic_fetch_add_explicit(&hist[(u >> uint(shift)) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint cum = 0u, want = sh[2];
            int d = 255;
            for (; d > 0; --d) {
                const uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (cum + c >= want) break;
                cum += c;
            }
            sh[2] = want - cum; sh[0] = pre | (uint(d) << uint(shift)); sh[1] = msk | (255u << uint(shift));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint thr = sh[0], need = sh[2];  // every block above thr, and the first `need` blocks equal to it
    threadgroup uint wtake[32], wtie[32], base[2];
    if (tid == 0u) {base[0] = 0u; base[1] = 0u;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c0 = 0; c0 < nb; c0 += 1024u) {
        const uint b = c0 + tid;
        const uint u = b < nb ? as_type<uint>(S[b]) : 0u;
        const bool gt = b < nb && u > thr, eq = b < nb && u == thr;
        const uint eqx = simd_prefix_exclusive_sum(uint(eq)), eqs = simd_sum(uint(eq));
        if (lane == 0u) wtie[sg] = eqs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint eqb = base[1];
        for (uint w = 0; w < sg; ++w) eqb += wtie[w];
        const bool take = gt || (eq && eqb + eqx < need);
        const uint tx = simd_prefix_exclusive_sum(uint(take)), ts = simd_sum(uint(take));
        if (lane == 0u) wtake[sg] = ts;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tb = base[0];
        for (uint w = 0; w < sg; ++w) tb += wtake[w];
        if (take) SEL[t * uint(PICK) + tb + tx] = int(b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint a = 0u, e = 0u;
            for (uint w = 0; w < 32u; ++w) {a += wtake[w]; e += wtie[w];}
            base[0] += a; base[1] += e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
)METAL");
    const std::vector<int32_t> p = {n, from, nbmax};
    const auto P = mx::array(p.data(), {3}, mx::int32);
    auto sc = scores({mx::contiguous(q), blocks, P}, {{16, nbmax}}, {mx::float32}, {128 * ((nbmax + 15) / 16), 2, 1}, {128, 1, 1},
                     {}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return kernel({sc, plan, P}, {{16, kQsaPick}}, {mx::int32}, {1024, 16, 1}, {1024, 1, 1}, {{"PICK", kQsaPick}, {"PW", pw}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// attention partials of the sparse rows over their selected keys, in the merge layout of decode_attention_exact
// ([kvh, kQsaChunks, 96, 256] + max / sum): chunk c holds selected keys c * 512 .. c * 512 + 511 in key order, one key at
// a time (online softmax in FP32); rows that are not sparse leave empty chunks
inline std::vector<mlx::core::array> qsa_partials(const mlx::core::array& qb, const mlx::core::array& k, const mlx::core::array& v,
                                                  const mlx::core::array& sel, const mlx::core::array& plan, int n, int pw,
                                                  const mlx::core::array& sc, int from) {
    namespace mx = mlx::core;
    static auto kernel = mx::fast::metal_kernel("tqllm_qsa_partials", {"QB", "K", "V", "SEL", "PLAN", "P", "SC"}, {"PO", "PM", "PL"}, R"METAL(
    const uint lane = thread_index_in_simdgroup, h = simdgroup_index_in_threadgroup;
    const uint c = threadgroup_position_in_grid.x, t = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int N = P[0], CAP = P[2];
    const int T = PLAN[t * uint(PW)];
    const size_t base = (size_t(kh) * uint(NCH) + c) * 96u + h * 16u + t;
    float acc[8];
    for (uint i = 0; i < 8u; ++i) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    if (int(t) < N && T >= P[1]) {
        const device bfloat16_t* qr = QB + (size_t(kh) * 96u + h * 16u + t) * 256u + lane * 8u;
        float qv[8];
        for (uint i = 0; i < 8u; ++i) qv[i] = float(qr[i]);
        const float sc = SC;
        for (uint j = c * uint(SPAN); j < (c + 1u) * uint(SPAN); ++j) {
            const uint key = uint(SEL[t * uint(PICK) + (j >> 2)]) * 4u + (j & 3u);
            const size_t row = (size_t(kh) * uint(CAP) + key) * 256u + lane * 8u;
            float d = 0.0f;
            for (uint i = 0; i < 8u; ++i) d = fma(qv[i], float(K[row + i]), d);
            const float s = simd_sum(d) * sc;
            if (s > m) {
                const float a = m == -INFINITY ? 0.0f : exp(m - s);
                l = l * a + 1.0f;
                for (uint i = 0; i < 8u; ++i) acc[i] = acc[i] * a + float(V[row + i]);
                m = s;
            } else {
                const float p = exp(s - m);
                l += p;
                for (uint i = 0; i < 8u; ++i) acc[i] = fma(p, float(V[row + i]), acc[i]);
            }
        }
    }
    for (uint i = 0; i < 8u; ++i) PO[base * 256u + lane * 8u + i] = acc[i];
    if (lane == 0u) {PM[base] = m; PL[base] = l;}
)METAL");
    const int kvh = k.shape(1), cap = k.shape(2);
    const std::vector<int32_t> p = {n, from, cap};
    return kernel({qb, k, v, sel, plan, mx::array(p.data(), {3}, mx::int32), sc},
                  {{kvh, kQsaChunks, 96, 256}, {kvh, kQsaChunks, 96}, {kvh, kQsaChunks, 96}}, {mx::float32, mx::float32, mx::float32},
                  {32 * 6 * kQsaChunks, 16, kvh}, {32 * 6, 1, 1},
                  {{"NCH", kQsaChunks}, {"SPAN", kQsaPick * kQsaBlock / kQsaChunks}, {"PICK", kQsaPick}, {"PW", pw}}, std::nullopt,
                  false, mx::Device(mx::Device::gpu));
}
}  // namespace tqllm::mlx_backend

#pragma once
// Row-invariant products for the Inkling-2 decode step and speculative verify block (1..8 rows).
//
// Greedy speculative decoding is exact only when row j of an n-row verify block equals, bit for bit, the j-th one-row
// decode step. MLX's products are not: a 1-row matmul runs gemv and an 8-row one a tiled gemm, gather_qmm's
// qmv and its sorted (grouped) form sum in other orders. These kernels give every row the same arithmetic whatever
// the number of rows: one simdgroup owns TO outputs, its lanes walk K in a fixed stride (8 elements per lane step,
// explicit fma), and simd_sum reduces the lanes; the rows are an outer loop that never changes a row's own sums.
// Each weight is read once per block for all its rows (the routed experts: once per run of rows sharing an expert).
#include <mlx/mlx.h>
#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace tqllm::mlx_backend {
namespace rows_detail {
inline const std::string& fp8_header() {
    static const std::string h = [] {
        // E2M1 (NVFP4 codes) and E4M3 (NVFP4 scales, MXFP8 codes) as exact halves
        std::string out = "constant half E2M1[16] = {0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h, -0.0h, -0.5h, -1.0h, "
                          "-1.5h, -2.0h, -3.0h, -4.0h, -6.0h};\nconstant half E4M3[256] = {";
        for (int b = 0; b < 256; ++b) {
            const int ex = (b >> 3) & 15, man = b & 7;
            double v = ex ? (1.0 + man / 8.0) * double(1 << ex) / 128.0 : man / 8.0 / 64.0;
            if (v > 448.0) v = 448.0;  // 0x7f / 0xff are NaN in E4M3; never valid
            out += std::string(b ? ", " : "") + (b & 128 ? "-" : "") + std::to_string(v) + "h";
        }
        out += "};\n";
        // eight consecutive weights as floats from 16 bytes (BF16 / F16) or 32 bytes (F32)
        out += R"METAL(
inline void load8(const device bfloat16_t* p, thread float* o) {
    const uint4 u = *(const device uint4*)p;
    const uint w[4] = {u.x, u.y, u.z, u.w};
    for (int i = 0; i < 4; ++i) { o[2 * i] = as_type<float>(w[i] << 16); o[2 * i + 1] = as_type<float>(w[i] & 0xffff0000u); }
}
inline void load8(const device half* p, thread float* o) {
    const uint4 u = *(const device uint4*)p;
    const uint w[4] = {u.x, u.y, u.z, u.w};
    for (int i = 0; i < 4; ++i) { const half2 h = as_type<half2>(w[i]); o[2 * i] = float(h.x); o[2 * i + 1] = float(h.y); }
}
inline void load8(const device float* p, thread float* o) {
    const float4 a = *(const device float4*)p, b = *(const device float4*)(p + 4);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
// E2M1 / E4M3 codes by their bits as halves (exact): E2M1 m at the half's bit 9 is the value / 2^14 (0.5 is the
// subnormal 2^-15, 1 .. 6 normals); E4M3's 7 magnitude bits at bit 7 are the value / 2^8 (normals and subnormals)
inline float e2m1_scaled(uint n) { return float(as_type<half>(ushort(((n & 7u) << 9) | ((n & 8u) << 12)))); }
inline float e4m3_scaled(uint b) { return float(as_type<half>(ushort(((b & 127u) << 7) | ((b & 128u) << 8)))); }
inline float e4m3(uint b) { return e4m3_scaled(b) * 256.0f; }
inline void xload8(const device bfloat16_t* p, thread float* o) { load8(p, o); }
inline void xload8(const device half* p, thread float* o) { load8(p, o); }
inline void xload8(const device float* p, thread float* o) { load8(p, o); }
inline void tile8(const threadgroup float* p, thread float* o) {
    const float4 a = *(const threadgroup float4*)p, b = *(const threadgroup float4*)(p + 4);
    o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w; o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
)METAL";
        return out;
    }();
    return h;
}
// threadgroup: SG simdgroups of TO outputs each; every lane reads its x columns straight from device memory (16-byte
// loads, L1 hits after the first simdgroup; a threadgroup-memory stage was slower: bank conflicts and barriers)
constexpr int kSG = 4, kTO = 2, kTK = 512;  // kernel bench 11-24: TO = 4 spills (experts 110 vs 210 GB/s)
}  // namespace rows_detail

// y [n, O] = x [n, K] w [O, K]^T for n <= 8, in w's dtype (BF16, F16 or F32), x cast to it first (as linear()).
// sg / to: simdgroups per threadgroup and outputs per simdgroup (bench sweeps; a row's arithmetic does not depend on them)
inline mlx::core::array inkling2_rows_linear(const mlx::core::array& x, const mlx::core::array& w, int sg = rows_detail::kSG,
                                             int to = rows_detail::kTO) {
    namespace mx = mlx::core;
    const int n = x.ndim() == 2 ? x.shape(0) : 0, K = w.ndim() == 2 ? w.shape(1) : 0, O = w.ndim() == 2 ? w.shape(0) : 0;
    if (n < 1 || n > 8 || x.shape(1) != K || K % 8 || (w.dtype() != mx::bfloat16 && w.dtype() != mx::float16 && w.dtype() != mx::float32))
        throw std::invalid_argument("Inkling2 rows linear shape");
    // A narrow product (few outputs, long K: the controllers' down products, the router, the sink logits) splits K over
    // SPLIT simdgroups of one output tile, whole 256-column lane steps each, their sums added in slice order: SPLIT is a
    // function of the shape only, so a row's arithmetic is the same at every row count.
    static const char* source = R"METAL(
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint slice = sg % SPLIT;
    const uint o0 = (threadgroup_position_in_grid.x * (SG / SPLIT) + sg / SPLIT) * TO;
    constexpr uint STEPS = (uint(K) + 255u) / 256u, PER = (STEPS + SPLIT - 1) / SPLIT;
    threadgroup float part[SG][N * TO];
    const bool live = o0 < uint(O);
    float acc[N][TO];
    for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) acc[r][t] = 0.0f;
    if (live)
    for (uint st = slice * PER; st < min(STEPS, (slice + 1) * PER); ++st) {
        const uint k = st * 256u + lane * 8u;
        if (k >= uint(K)) continue;
        float wv[TO][8];
        for (int t = 0; t < TO; ++t) {
            if (o0 + t < uint(O)) load8(w + size_t(o0 + t) * K + k, wv[t]);
            else for (int i = 0; i < 8; ++i) wv[t][i] = 0.0f;
        }
        for (int r = 0; r < N; ++r) {
            float xv[8];
            xload8(x + size_t(r) * K + k, xv);
            for (int t = 0; t < TO; ++t) for (int i = 0; i < 8; ++i) acc[r][t] = fma(xv[i], wv[t][i], acc[r][t]);
        }
    }
    float v[N][TO];
    for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) v[r][t] = simd_sum(acc[r][t]);
    if (SPLIT > 1) {
        if (lane == 0) for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) part[sg][r * TO + t] = v[r][t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slice != 0) return;
        for (uint j = 1; j < SPLIT; ++j) for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) v[r][t] += part[sg + j][r * TO + t];
    }
    if (!live || lane != 0) return;
    for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) if (o0 + t < uint(O)) y[size_t(r) * O + o0 + t] = T(v[r][t]);
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_rows_linear", {"x", "w"}, {"y"}, source, rows_detail::fp8_header());
    // split: until there are >= 512 simdgroups or a slice would be under 2 lane steps (8 at most)
    const int steps = (K + 255) / 256, tiles = (O + to - 1) / to;
    int split = 1;
    while (split < 8 && tiles * split < 512 && steps / (2 * split) >= 2) split *= 2;
    const int threads = std::max(sg, split), per_tg = threads / split, groups = (tiles + per_tg - 1) / per_tg;
    return kernel({mx::astype(x, w.dtype()), w}, {{n, O}}, {w.dtype()}, {groups * 32 * threads, 1, 1}, {32 * threads, 1, 1},
                  {{"T", w.dtype()}, {"N", n}, {"K", K}, {"O", O}, {"SG", threads}, {"TO", to}, {"SPLIT", split}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// The same for an MLX affine matrix (mx::quantize: codes uint32 [O, K * bits / 32], scales / biases [O, K / group]),
// w = q * scale + bias; x cast to the scales' dtype (as quantized_matmul), output in it.
inline mlx::core::array inkling2_rows_affine(const mlx::core::array& x, const mlx::core::array& codes, const mlx::core::array& scales,
                                             const mlx::core::array& biases, int group, int bits) {
    namespace mx = mlx::core;
    const int n = x.ndim() == 2 ? x.shape(0) : 0, O = codes.shape(0), K = codes.shape(1) * 32 / bits;
    if (n < 1 || n > 8 || x.shape(1) != K || (bits != 4 && bits != 8) || group % 8 || K % group ||
        scales.shape() != mx::Shape{O, K / group} || biases.shape() != scales.shape())
        throw std::invalid_argument("Inkling2 rows affine shape");
    static const char* source = R"METAL(
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint o0 = (threadgroup_position_in_grid.x * SG + sg) * TO;
    if (o0 >= uint(O)) return;
    float acc[N][TO];
    for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) acc[r][t] = 0.0f;
    constexpr int WORDS = K * BITS / 32;
    for (uint k = lane * 8u; k < uint(K); k += 256u) {
        // codes as integers, the step's scale and bias once (a step is within one group): q . x scaled, plus bias sum(x)
        float wv[TO][8], sv[TO], bv[TO];
        for (int t = 0; t < TO; ++t) {
            const size_t row = size_t(min(o0 + t, uint(O) - 1));
            sv[t] = float(scales[row * (K / GROUP) + k / GROUP]); bv[t] = float(biases[row * (K / GROUP) + k / GROUP]);
            if (BITS == 8) {
                const uint2 c = *(const device uint2*)(codes + row * WORDS + k / 4);
                for (int i = 0; i < 4; ++i) { wv[t][i] = float((c.x >> (8 * i)) & 255u); wv[t][4 + i] = float((c.y >> (8 * i)) & 255u); }
            } else {
                const uint c = codes[row * WORDS + k / 8];
                for (int i = 0; i < 8; ++i) wv[t][i] = float((c >> (4 * i)) & 15u);
            }
        }
        for (int r = 0; r < N; ++r) {
            float xv[8];
            xload8(x + size_t(r) * K + k, xv);
            float sx = 0.0f;
            for (int i = 0; i < 8; ++i) sx += xv[i];
            for (int t = 0; t < TO; ++t) {
                float part = 0.0f;
                for (int i = 0; i < 8; ++i) part = fma(xv[i], wv[t][i], part);
                acc[r][t] = fma(sx, bv[t], fma(part, sv[t], acc[r][t]));
            }
        }
    }
    for (int r = 0; r < N; ++r) for (int t = 0; t < TO; ++t) {
        const float v = simd_sum(acc[r][t]);
        if (lane == 0 && o0 + t < uint(O)) y[size_t(r) * O + o0 + t] = T(v);
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_rows_affine", {"x", "codes", "scales", "biases"}, {"y"}, source,
                                                rows_detail::fp8_header());
    using namespace rows_detail;
    const int groups = (O + kSG * kTO - 1) / (kSG * kTO);
    return kernel({mx::astype(x, scales.dtype()), codes, scales, biases}, {{n, O}}, {scales.dtype()},
                  {groups * 32 * kSG, 1, 1}, {32 * kSG, 1, 1},
                  {{"T", scales.dtype()}, {"N", n}, {"K", K}, {"O", O}, {"GROUP", group}, {"BITS", bits}, {"SG", kSG}, {"TO", kTO}, {"TK", kTK}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// Routed experts of a block, pairs sorted by expert (inkling2_pair_sort): y [P, O] (BF16) = x[xrow[p]] times expert
// ids[p], each expert's weights read once for its run of pairs (<= 8: a token routes to an expert once). NVFP4: codes
// [E, O, K / 2] E2M1 pairs, scales [E, O, K / 16] E4M3, w = E2M1 x E4M3 x outer[e] / 2688 (outer: FloatxExperts::global);
// MXFP8: codes [E, O, K] E4M3, scales [E, O, K / 32] E8M0. With swiglu (gate_up: gate and up interleaved by column),
// the capped SwiGLU of each pair instead: act [P, O / 2] = (tanh(g / gcap) gcap sigmoid(g)) (tanh(u / ucap) ucap), g and u
// rounded to BF16 first.
inline mlx::core::array inkling2_rows_experts(const mlx::core::array& x, const mlx::core::array& xrow, const mlx::core::array& ids,
                                              const mlx::core::array& codes, const mlx::core::array& scales,
                                              const std::optional<mlx::core::array>& outer, int bits, bool swiglu = false,
                                              float gate_cap = 0.f, float up_cap = 0.f, int sg = rows_detail::kSG,
                                              int to = rows_detail::kTO) {
    namespace mx = mlx::core;
    // codes: FloatxExperts::codes, the stored bytes viewed as uint32 [E, O, K * bits / 32]
    const int P = ids.ndim() == 1 ? ids.shape(0) : 0, E = codes.shape(0), O = codes.shape(1), K = codes.shape(2) * 32 / bits;
    const int group = bits == 4 ? 16 : 32;
    if (P < 1 || P > 64 || x.ndim() != 2 || x.shape(1) != K || x.dtype() != mx::bfloat16 || xrow.shape() != ids.shape() ||
        (bits != 4 && bits != 8) || codes.dtype() != mx::uint32 || scales.dtype() != mx::uint8 || scales.shape() != mx::Shape{E, O, K / group} ||
        K % 16 || to % 2 || O % (sg * to) || (bits == 4) != outer.has_value())
        throw std::invalid_argument("Inkling2 rows experts shape");
    // a lane takes 16 weights per step (NVFP4: 8 code bytes + 1 scale; MXFP8: 16 code bytes + 1 scale)
    static const char* body = R"METAL(
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint p0 = threadgroup_position_in_grid.y;
    const uint e = ids[p0];
    if (p0 > 0 && ids[p0 - 1] == e) return;  // the run's first pair computes the run
    uint count = 1;
    while (count < 8u && p0 + count < uint(P) && ids[p0 + count] == e) ++count;
    const uint o0 = (threadgroup_position_in_grid.x * SG + sg) * TO;
    if (o0 >= uint(O)) return;
    float acc[8][TO];
    for (int r = 0; r < 8; ++r) for (int t = 0; t < TO; ++t) acc[r][t] = 0.0f;
    const device uchar* cbase = codes + size_t(e) * O * (K * BITS / 8);
    const device uchar* sbase = scales + size_t(e) * O * (K / GROUP);
    for (uint k = lane * 16u; k < uint(K); k += 512u) {
        // codes as their unscaled values, the lane step's scale once (a step is within one scale group)
        float wv[TO][16], sc[TO];
        for (int t = 0; t < TO; ++t) {
            const size_t row = size_t(o0 + t);
            if (BITS == 4) {
                const uint2 c = *(const device uint2*)(cbase + row * (K / 2) + k / 2);
                sc[t] = float(E4M3[sbase[row * (K / 16) + k / 16]]);
                for (int i = 0; i < 8; ++i) { wv[t][i] = float(E2M1[(c.x >> (4 * i)) & 15u]); wv[t][8 + i] = float(E2M1[(c.y >> (4 * i)) & 15u]); }
            } else {
                const uint4 c = *(const device uint4*)(cbase + row * K + k);
                sc[t] = as_type<float>(uint(sbase[row * (K / 32) + k / 32]) << 23) * 256.0f;  // e4m3_scaled() is the value / 2^8
                const uint w[4] = {c.x, c.y, c.z, c.w};
                for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i) wv[t][4 * j + i] = e4m3_scaled(w[j] >> (8 * i));
            }
        }
        for (int r = 0; r < 8; ++r) {  // constant bounds: acc stays in registers (a runtime-indexed array spills)
            if (uint(r) >= count) break;
            const device bfloat16_t* xr = x + size_t(xrow[p0 + r]) * K + k;
            float xv[16];
            load8(xr, xv);
            load8(xr + 8, xv + 8);
            for (int t = 0; t < TO; ++t) {
                float part = 0.0f;
                for (int i = 0; i < 16; ++i) part = fma(xv[i], wv[t][i], part);
                acc[r][t] = fma(part, sc[t], acc[r][t]);
            }
        }
    }
    const float g = BITS == 4 ? outer[e] * (1.0f / 2688.0f) : 1.0f;
    for (int r = 0; r < 8; ++r) {
        if (uint(r) >= count) break;
        float v[TO];
        for (int t = 0; t < TO; ++t) v[t] = float(bfloat16_t(simd_sum(acc[r][t]) * g));
        if (lane != 0) continue;
        if (SWIGLU) {
            for (int t = 0; t < TO; t += 2) {
                const float gv = v[t], uv = v[t + 1];
                const float sig = 1.0f / (1.0f + metal::precise::exp(-gv));
                const float gc = caps[0], uc = caps[1];
                const float a = metal::precise::tanh(gv / gc) * gc * sig * (metal::precise::tanh(uv / uc) * uc);
                y[size_t(p0 + r) * (O / 2) + (o0 + t) / 2] = bfloat16_t(a);
            }
        } else {
            for (int t = 0; t < TO; ++t) y[size_t(p0 + r) * O + o0 + t] = bfloat16_t(v[t]);
        }
    }
)METAL";
    static auto kernel = mx::fast::metal_kernel("tqllm_inkling2_rows_experts", {"x", "xrow", "ids", "codes", "scales", "outer", "caps"}, {"y"},
                                                body, rows_detail::fp8_header());
    const int blocks = (O + sg * to - 1) / (sg * to);
    const float cap[2] = {swiglu ? gate_cap : 1.f, swiglu ? up_cap : 1.f};
    return kernel({x, mx::astype(xrow, mx::uint32), mx::astype(ids, mx::uint32), mx::view(codes, mx::uint8), scales,
                   outer ? mx::astype(*outer, mx::float32) : mx::zeros({1}, mx::float32), mx::array(cap, {2}, mx::float32)},
                  {{P, swiglu ? O / 2 : O}}, {mx::bfloat16}, {blocks * 32 * sg, P, 1}, {32 * sg, 1, 1},
                  {{"P", P}, {"K", K}, {"O", O}, {"BITS", bits}, {"GROUP", group}, {"SG", sg}, {"TO", to}, {"SWIGLU", swiglu}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

// Sink attention of 1..8 query rows (a decode step or a verify block) with the relative bias computed in place:
// q [1, H, n, HD], keys / values [1, G, >= s, HD] (cache rows 0 .. s - 1 at positions start ..), rel [n, H, DR] and
// proj [DR, E] (bias of distance d < E: BF16(rel . proj[:, d])), sinks [n, H] (a zero-value key per row), row r at
// position base + r sees keys at distance 0 <= d < window (0: all). Row-invariant: keys go in blocks of 32 absolute
// positions (one per lane), 8 blocks a chunk merged in order, then the sink and the chunks in order; a row's visible keys are the same whether later rows of its block (masked) or a trimmed window prefix are in
// the cache, so a block row equals its decode step bit for bit.
inline mlx::core::array inkling2_rows_attention(const mlx::core::array& q, const mlx::core::array& keys, const mlx::core::array& values,
                                                const mlx::core::array& rel, const mlx::core::array& proj, const mlx::core::array& sinks,
                                                int base, int start, int s, int window, float scale) {
    namespace mx = mlx::core;
    const int H = q.shape(1), n = q.shape(2), HD = q.shape(3), G = keys.shape(1), DR = rel.ndim() == 3 ? rel.shape(2) : 0;
    const int E = proj.ndim() == 2 ? proj.shape(1) : 0;
    if (q.ndim() != 4 || q.shape(0) != 1 || n < 1 || n > 8 || HD != 128 || keys.ndim() != 4 || keys.shape() != values.shape() ||
        H % G || keys.shape(2) < s || rel.shape() != mx::Shape{n, H, DR} || proj.shape(0) != DR || DR > 32 ||
        sinks.shape() != mx::Shape{n, H} || s < 1)
        throw std::invalid_argument("Inkling2 rows attention shape");
    constexpr int CH = 256;
    const int first = start / CH, last = (base + n - 1) / CH, chunks = last - first + 1;
    // A threadgroup serves up to 8 queries of one key head (its query heads x the block's rows), so each key / value row
    // is read once for all of them. A chunk's partial per query (M, L, A[4] per lane of simdgroup 0): simdgroup j takes
    // the chunk's 32 positions 32 j .., one key per lane (its 128-term dot with each query), softmax over the lanes,
    // values added in lane order; then simdgroup 0 merges j = 0 .. 7 in order. A query's operations do not depend on
    // the other queries of its threadgroup or on which of the two kernels below runs the chunk.
    static const std::string prelude = R"METAL(
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    constexpr int QN = (H / G) * N, QB = (QN + 7) / 8;  // queries per key head, their blocks of <= 8
    const uint g = threadgroup_position_in_grid.y / QB, qb = threadgroup_position_in_grid.y % QB;
    const int q0 = int(qb) * 8, qc = min(8, QN - q0);
    const int base = params[0], start = params[1], s = params[2], window = params[3], first = params[4], chunks = params[6];
    const float scale = as_type<float>(params[5]);
    const size_t ks1 = keys_strides[1], ks2 = keys_strides[2], vs1 = values_strides[1], vs2 = values_strides[2];
    threadgroup float qs[8][128];
    threadgroup float tm[8], tl[8], tacc[8][128];
    int hq[8], rq[8];  // query qi: head g * (H / G) + (q0 + qi) / N, row (q0 + qi) % N
    for (int qi = 0; qi < 8; ++qi) { const int k = min(q0 + qi, QN - 1); hq[qi] = int(g) * (H / G) + k / N; rq[qi] = k % N; }
    for (int qi = int(sg); qi < qc; qi += 8)
        for (int i = 0; i < 4; ++i)  // q read in place by strides (the attention's transposed view: no copy)
            qs[qi][lane * 4 + i] = float(q[size_t(hq[qi]) * q_strides[1] + size_t(rq[qi]) * q_strides[2] + size_t(lane * 4 + i) * q_strides[3]]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
)METAL";
    static const std::string chunk_body = R"METAL(
    float M[8], Lsum[8], A[8][4];
    for (int qi = 0; qi < 8; ++qi) { M[qi] = -INFINITY; Lsum[qi] = 0.0f; for (int t = 0; t < 4; ++t) A[qi][t] = 0.0f; }
    {
        const int p = chunk * 256 + int(sg) * 32 + int(lane);
        const int c = p - start;
        bool visible[8];
        bool any = false;
        for (int qi = 0; qi < 8; ++qi) {
            const int d = base + rq[qi] - p;
            visible[qi] = qi < qc && c >= 0 && c < s && d >= 0 && (window <= 0 || d < window);
            any = any || visible[qi];
        }
        float dot[8];
        for (int qi = 0; qi < 8; ++qi) dot[qi] = 0.0f;
        if (any) {
            const device T* kr = keys + g * ks1 + size_t(c) * ks2;
            for (int i = 0; i < 128; i += 8) {
                float kv[8];
                load8(kr + i, kv);
                for (int qi = 0; qi < 8; ++qi)
                    if (qi < qc) for (int t = 0; t < 8; ++t) dot[qi] = fma(qs[qi][i + t], kv[t], dot[qi]);
            }
        }
        float sc[8];
        for (int qi = 0; qi < 8; ++qi) {
            sc[qi] = -INFINITY;
            if (!visible[qi]) continue;
            const int d = base + rq[qi] - p;
            float bias = 0.0f;
            if (d < E) {
                float b = 0.0f;
                for (int i = 0; i < DR; ++i) b = fma(float(rel[(size_t(rq[qi]) * H + hq[qi]) * DR + i]), float(proj[size_t(i) * E + d]), b);
                bias = float(bfloat16_t(b));
            }
            sc[qi] = fma(dot[qi], scale, bias);
        }
        float m[8], l[8], acc[8][4], pe[8];
        for (int qi = 0; qi < 8; ++qi) {
            m[qi] = simd_max(sc[qi]);
            l[qi] = 0.0f; for (int t = 0; t < 4; ++t) acc[qi][t] = 0.0f;
            pe[qi] = visible[qi] && m[qi] > -INFINITY ? metal::precise::exp(sc[qi] - m[qi]) : 0.0f;
            if (m[qi] > -INFINITY) l[qi] = simd_sum(pe[qi]);
        }
        // no branch per key: a masked (or underflowed) key adds p = 0 times v = 0, the same ops in a decode step and a
        // verify block; each value row is read once for all queries
        const int c0 = chunk * 256 + int(sg) * 32 - start;
        bool live = false;
        for (int qi = 0; qi < 8; ++qi) live = live || m[qi] > -INFINITY;
        if (live) for (int i = 0; i < 32; ++i) {
            const int ci = clamp(c0 + i, 0, s - 1);
            const device T* vr = values + g * vs1 + size_t(ci) * vs2 + lane * 4;
            float vv[4];
            for (int t = 0; t < 4; ++t) vv[t] = float(vr[t]);
            for (int qi = 0; qi < 8; ++qi) {
                if (!(m[qi] > -INFINITY)) continue;
                const float pi = simd_shuffle(pe[qi], ushort(i));
                for (int t = 0; t < 4; ++t) acc[qi][t] = fma(pi, pi != 0.0f ? vv[t] : 0.0f, acc[qi][t]);
            }
        }
        // per query: the 8 simdgroups' partials merged in order by simdgroup 0
        for (int qi = 0; qi < 8; ++qi) {
            if (qi >= qc) break;
            if (lane == 0) { tm[sg] = m[qi]; tl[sg] = l[qi]; }
            for (int t = 0; t < 4; ++t) tacc[sg][lane * 4 + t] = acc[qi][t];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                for (int j = 0; j < 8; ++j) {
                    const float lj = tl[j];
                    if (lj == 0.0f) continue;
                    const float mj = tm[j], mn = max(M[qi], mj), a = metal::precise::exp(M[qi] - mn), b = metal::precise::exp(mj - mn);
                    Lsum[qi] = fma(Lsum[qi], a, lj * b);
                    for (int t = 0; t < 4; ++t) A[qi][t] = fma(A[qi][t], a, tacc[j][lane * 4 + t] * b);
                    M[qi] = mn;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
)METAL";
    // the sink first, then the chunks in order (both paths), query qi
    static const char* merge = R"METAL(
        if (Lsum[qi] != 0.0f) {
            const float mn = max(m0[qi], M[qi]), a = metal::precise::exp(m0[qi] - mn), b = metal::precise::exp(M[qi] - mn);
            l0[qi] = fma(l0[qi], a, Lsum[qi] * b);
            for (int t = 0; t < 4; ++t) acc0[qi][t] = fma(acc0[qi][t], a, A[qi][t] * b);
            m0[qi] = mn;
        }
)METAL";
    // short spans: one threadgroup per (key head, query block) walks the chunks and merges them as it goes
    static const std::string fused_source = prelude + R"METAL(
    float m0[8], l0[8], acc0[8][4];
    for (int qi = 0; qi < 8; ++qi) {
        m0[qi] = float(bfloat16_t(float(sinks[rq[qi] * H + hq[qi]]))); l0[qi] = 1.0f;
        for (int t = 0; t < 4; ++t) acc0[qi][t] = 0.0f;
    }
    for (int ci = 0; ci < chunks; ++ci) {
        const int chunk = first + ci;
)METAL" + chunk_body + "        if (sg == 0) for (int qi = 0; qi < qc; ++qi) {\n" + merge + R"METAL(
        }
    }
    if (sg == 0) for (int qi = 0; qi < qc; ++qi)
        for (int t = 0; t < 4; ++t) out[(size_t(hq[qi]) * N + rq[qi]) * 128 + lane * 4 + t] = T(acc0[qi][t] / l0[qi]);
)METAL";
    // long spans: one threadgroup per (key head, query block, chunk), then a merge kernel per (head, row)
    static const std::string chunk_source = prelude + R"METAL(
    const int chunk = first + int(threadgroup_position_in_grid.x);
)METAL" + chunk_body + R"METAL(
    if (sg != 0) return;
    for (int qi = 0; qi < qc; ++qi) {
        const size_t at = (size_t(hq[qi]) * N + rq[qi]) * chunks + threadgroup_position_in_grid.x;
        if (lane == 0) { ml[at * 2] = M[qi]; ml[at * 2 + 1] = Lsum[qi]; }
        for (int t = 0; t < 4; ++t) part[at * 128 + lane * 4 + t] = A[qi][t];
    }
)METAL";
    static const std::string combine_source = std::string(R"METAL(
    const uint lane = thread_position_in_threadgroup.x, hr = threadgroup_position_in_grid.y, h = hr / N, r = hr % N;
    const int chunks = params[6];
    constexpr int qi = 0;
    float m0[1] = {float(bfloat16_t(float(sinks[r * H + h])))}, l0[1] = {1.0f}, acc0[1][4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (int c = 0; c < chunks; ++c) {
        const size_t at = size_t(hr) * chunks + c;
        float M[1] = {ml[at * 2]}, Lsum[1] = {ml[at * 2 + 1]}, A[1][4];
        for (int t = 0; t < 4; ++t) A[0][t] = part[at * 128 + lane * 4 + t];
)METAL") + merge + R"METAL(
    }
    for (int t = 0; t < 4; ++t) out[(size_t(h) * N + r) * 128 + lane * 4 + t] = T(acc0[0][t] / l0[0]);
)METAL";
    static auto fused_kernel = mx::fast::metal_kernel("tqllm_inkling2_rows_attn_gqa", {"q", "keys", "values", "rel", "proj", "sinks", "params"},
                                                      {"out"}, fused_source, rows_detail::fp8_header(), false);
    static auto chunk_kernel = mx::fast::metal_kernel("tqllm_inkling2_rows_attn_gqa_chunks", {"q", "keys", "values", "rel", "proj", "params"},
                                                      {"ml", "part"}, chunk_source, rows_detail::fp8_header(), false);
    static auto combine_kernel = mx::fast::metal_kernel("tqllm_inkling2_rows_attn_combine", {"ml", "part", "sinks", "params"}, {"out"},
                                                        combine_source);
    int32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    const int32_t params[7] = {base, start, s, window, first, scale_bits, chunks};
    const auto T = keys.dtype();
    if (keys.strides(3) != 1 || values.strides(3) != 1) throw std::invalid_argument("Inkling2 rows attention: head dim not contiguous");
    const auto prm = mx::array(params, {7}, mx::int32);
    const int QB = ((H / G) * n + 7) / 8;
    const auto qc = mx::astype(q, T), rc = mx::contiguous(mx::astype(rel, proj.dtype())), pc = proj;  // a weight: contiguous
    const auto sk = mx::contiguous(sinks);
    // one fused launch only for spans of 1-2 chunks: a decode step's H threadgroups walking 5 window chunks serially
    // were slower (attention phase 18.7 vs 16.0 ms a token) than the chunk-parallel pair
    if (chunks <= 2)
        return fused_kernel({qc, keys, values, rc, pc, sk, prm}, {{1, H, n, HD}}, {q.dtype()}, {256, G * QB, 1}, {256, 1, 1},
                            {{"T", T}, {"N", n}, {"H", H}, {"G", G}, {"DR", DR}, {"E", E}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    auto parts = chunk_kernel({qc, keys, values, rc, pc, prm}, {{H * n, chunks, 2}, {H * n, chunks, 128}}, {mx::float32, mx::float32},
                              {256 * chunks, G * QB, 1}, {256, 1, 1},
                              {{"T", T}, {"N", n}, {"H", H}, {"G", G}, {"DR", DR}, {"E", E}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return combine_kernel({parts[0], parts[1], sk, prm}, {{1, H, n, HD}}, {q.dtype()}, {32, H * n, 1}, {32, 1, 1},
                          {{"T", q.dtype()}, {"N", n}, {"H", H}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

}  // namespace tqllm::mlx_backend

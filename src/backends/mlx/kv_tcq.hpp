#pragma once
// Trellis KV cache on Metal (format: tcq_codec.hpp, the CUDA lane's tcq3 K / tcq2 V): the row encoder (Viterbi, one
// threadgroup per (row, kv head, 128-value group), one thread per trellis state) and decode-rows attention that reads the
// codes directly (the v12 structure: 16-row tiles, two simdgroups per tile = the two 128-value groups, so each half of S
// takes its group's norm; values decoded per fragment from the 9 (K) / 8 (V) bit state window). Queries are rotated into
// the trellis basis before the kernel and the output rotated back after it (K and V are stored rotated). A row's
// arithmetic depends only on its own query and visible keys: the n-row verify equals n single-row calls bit for bit.
#include "attention_decode.hpp"
#include "tcq_codec.hpp"
#include <mlx/mlx.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
// R [128, 128] FP32 with rotate_group(x) = R x (orthonormal)
inline mlx::core::array tcq_rotation() {
    static const std::vector<float> r = [] {
        std::vector<float> m(128 * 128);
        for (int j = 0; j < 128; ++j) {
            float e[128] = {}; e[j] = 1.0f; tcq::rotate_group(e);
            for (int i = 0; i < 128; ++i) m[i * 128 + j] = e[i];
        }
        return m;
    }();
    return mlx::core::array(r.data(), {128, 128}, mlx::core::float32);
}
inline mlx::core::array tcq_table(int kb) {
    return kb == 2 ? mlx::core::array(tcq::kCodebookV, {256}, mlx::core::float32) : mlx::core::array(tcq::kCodebookK, {512}, mlx::core::float32);
}
// rows [r, 128] FP32 -> rotated (R x) or, inverse, R^T x: signs, unnormalised Walsh-Hadamard by reshape butterflies
// (pairs i, i + h with bit h of i clear: the host's order), 1 / sqrt(128), signs
inline mlx::core::array tcq_rotate_rows(const mlx::core::array& x, bool inverse) {
    namespace mx = mlx::core;
    static const auto s1 = mx::array(tcq::kSigns1, {1, 128}, mx::float32), s2 = mx::array(tcq::kSigns2, {1, 128}, mx::float32);
    const int r = x.shape(0);
    auto y = mx::multiply(x, inverse ? s2 : s1);
    for (int h = 1; h < 128; h <<= 1) {
        auto v = mx::reshape(y, {r, 128 / (2 * h), 2, h});
        auto a = mx::slice(v, {0, 0, 0, 0}, {r, 128 / (2 * h), 1, h}), b = mx::slice(v, {0, 0, 1, 0}, {r, 128 / (2 * h), 2, h});
        y = mx::reshape(mx::concatenate({mx::add(a, b), mx::subtract(a, b)}, 2), {r, 128});
    }
    return mx::multiply(mx::multiply(y, mx::array(tcq::kInvSqrt128)), inverse ? s1 : s2);
}
// x [n, kvh, 256] (any float type) -> codes [n, kvh, 2 * words_per_group(kb)] uint32
inline mlx::core::array tcq_encode(const mlx::core::array& x, int kb) {
    namespace mx = mlx::core;
    if (x.ndim() != 3 || x.shape(2) != 256 || (kb != 2 && kb != 3)) throw std::invalid_argument("tcq_encode: x [n, kvh, 256], kb 2 or 3");
    const int n = x.shape(0), kvh = x.shape(1), wpg = tcq::words_per_group(kb);
    static auto kernel = mx::fast::metal_kernel("tqllm_tcq_encode", {"X", "CB", "S1", "S2"}, {"C"}, R"METAL(
    constexpr int STATES = 64 << KB, NP = 1 << KB, WPG = (6 + 128 * KB + 31) / 32;
    const uint sid = thread_position_in_threadgroup.x;
    const uint g = threadgroup_position_in_grid.x;               // (row, kv head, group)
    const uint grp = g % 2, kh = (g / 2) % KVH, row = g / (2 * KVH);
    threadgroup float x[128];
    threadgroup float cost[STATES], costb[STATES], red[STATES];
    threadgroup int redi[STATES];
    threadgroup uchar bt[128 * 64];
    threadgroup uchar sym[128];
    threadgroup int init_state;
    if (sid < 128) x[sid] = float(X[(size_t(row) * KVH + kh) * 256 + grp * 128 + sid]);
    red[sid] = sid < 128 ? x[sid] * x[sid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = STATES / 2; h > 0; h >>= 1) {if (sid < h) red[sid] += red[sid + h]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    const float gnorm = sqrt(red[0]), inv = gnorm > 1e-10f ? 1.0f / gnorm : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid < 128) x[sid] = x[sid] * inv * S1[sid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 1; h < 128; h <<= 1) {                         // unnormalised Walsh-Hadamard, the host's butterfly order
        float v = 0.0f, o = 0.0f;
        if (sid < 128) {v = x[sid]; o = x[sid ^ h];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sid < 128) x[sid] = (sid & h) ? o - v : v + o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sid < 128) x[sid] *= 0.08838834764831845f * S2[sid];
    cost[sid] = 0.0f;
    const float cbv = CB[sid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < 128; ++t) {
        threadgroup float* rd = (t & 1) ? costb : cost;
        threadgroup float* wr = (t & 1) ? cost : costb;
        const uint base = (sid & 0x3F) << KB;
        float best = rd[base]; int bp = 0;
        for (int p = 1; p < NP; ++p) {const float c = rd[base | p]; if (c < best) {best = c; bp = p;}}
        if (sid < 64) bt[t * 64 + sid] = uchar(bp);
        const float d = x[t] - cbv;
        wr[sid] = best + d * d;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    red[sid] = cost[sid]; redi[sid] = int(sid);                   // 128 steps: the final costs are in `cost`
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = STATES / 2; h > 0; h >>= 1) {
        if (sid < h && (red[sid + h] < red[sid] || (red[sid + h] == red[sid] && redi[sid + h] < redi[sid]))) {red[sid] = red[sid + h]; redi[sid] = redi[sid + h];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sid == 0) {
        int state = redi[0];
        for (int t = 127; t >= 0; --t) {sym[t] = uchar(state >> 6); state = ((state & 0x3F) << KB) | bt[t * 64 + (state & 0x3F)];}
        init_state = state;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rsq = 0.0f;
    if (sid < 128) {int cs = init_state; for (int t = 0; t <= int(sid); ++t) cs = (cs >> KB) | (int(sym[t]) << 6); rsq = CB[cs] * CB[cs];}
    red[sid] = rsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = STATES / 2; h > 0; h >>= 1) {if (sid < h) red[sid] += red[sid + h]; threadgroup_barrier(mem_flags::mem_threadgroup);}
    const float rn = sqrt(red[0]);
    const float norm = (rn > 1e-10f ? gnorm / rn : gnorm) * (KB == 2 ? 1.04f : 1.0f);
    if (sid < uint(WPG)) {
        const int ib = (init_state >> KB) & 0x3F, nbits = 6 + 128 * KB;
        uint word = 0;
        for (int bit = 0; bit < 32; ++bit) {
            const int pos = int(sid) * 32 + bit; int v = 0;
            if (pos < 6) v = (ib >> pos) & 1;
            else if (pos < nbits) v = (sym[(pos - 6) / KB] >> ((pos - 6) % KB)) & 1;
            word |= uint(v) << bit;
        }
        if (sid == uint(WPG - 1)) word |= uint(as_type<ushort>(half(norm))) << 16;
        C[(size_t(row) * KVH + kh) * (2 * WPG) + WPG * grp + sid] = word;
    }
)METAL");
    static const auto s1 = mx::array(tcq::kSigns1, {128}, mx::float32), s2 = mx::array(tcq::kSigns2, {128}, mx::float32);
    const int states = 64 << kb;
    return kernel({mx::contiguous(x), tcq_table(kb), s1, s2}, {{n, kvh, 2 * wpg}}, {mx::uint32}, {n * kvh * 2 * states, 1, 1}, {states, 1, 1},
                  {{"KB", kb}, {"KVH", kvh}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
// codes [kvh, cap, 2 * words_per_group(kb)] -> rows [0, rows) decoded to [1, kvh, rows, 256] FP16 in the original basis
// (the prefill's attention reads the prefix this way, one layer at a time)
inline mlx::core::array tcq_decode(const mlx::core::array& codes, int kb, int rows) {
    namespace mx = mlx::core;
    const int kvh = codes.shape(0), cap = codes.shape(1), w = codes.shape(2);
    if (codes.ndim() != 3 || w != 2 * tcq::words_per_group(kb) || rows < 1 || rows > cap) throw std::invalid_argument("tcq_decode: codes [kvh, cap, words]");
    static auto kernel = mx::fast::metal_kernel("tqllm_tcq_decode", {"C", "CB", "P"}, {"Y"}, R"METAL(
    // one thread per value: (kv head, row, dim); value t of the dim's group from the KB-bit state window, x the group norm
    const uint gid = thread_position_in_grid.x;
    const int ROWS = P[0], CAP = P[1];
    const int d = int(gid % 256), r = int((gid / 256) % uint(ROWS)), kh = int(gid / (256 * uint(ROWS)));
    constexpr int WG = (6 + 128 * KB + 31) / 32;
    const device uint* g = C + (size_t(kh) * CAP + r) * (2 * WG) + (d / 128) * WG;
    const int t = d % 128, bit = KB * t, w0 = bit >> 5;
    const ulong win = ulong(g[w0]) | (w0 + 1 < WG ? ulong(g[w0 + 1]) << 32 : 0ul);
    const float v = CB[uint(win >> (bit & 31)) & ((1u << (6 + KB)) - 1u)] * float(as_type<half>(ushort(g[WG - 1] >> 16)));
    Y[gid] = half(v);
)METAL");
    auto P = mx::array(std::vector<int32_t>{rows, cap}.data(), {2}, mx::int32);
    auto rot = kernel({codes, tcq_table(kb), P}, {{kvh * rows * 256}}, {mx::float16}, {kvh * rows * 256, 1, 1}, {256, 1, 1},
                      {{"KB", kb}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // x = R^T y in FP16 (prefill operand precision): no FP32 copy of the prefix
    static const auto R16 = mx::astype(tcq_rotation(), mx::float16);
    return mx::reshape(mx::matmul(mx::reshape(rot, {kvh * rows * 2, 128}), R16), {1, kvh, rows, 256});
}
// q [1, heads, n <= 16, 256] FP32 (original basis); kc [kvh, cap, 26], vc [kvh, cap, 18] uint32; keys [0, length): the
// last row sees `length` keys, row t of n sees length - n + 1 + t. Returns [1, heads, n, 256] FP32 (BF16-rounded).
// diag (bench only): 1 skips the K decode, 2 the V decode, 4 the query / output rotations
inline mlx::core::array decode_rows_attention_tcq(const mlx::core::array& q, const mlx::core::array& kc, const mlx::core::array& vc,
                                                  float scale, int length, int diag = 0) {
    namespace mx = mlx::core;
    constexpr int kBlock = 1024, kDim = 256;
    if (q.ndim() != 4 || q.shape(0) != 1 || q.shape(3) != kDim || q.shape(2) < 1 || q.shape(2) > 16)
        throw std::invalid_argument("decode rows attention tcq: q [1, heads, n <= 16, 256]");
    const int heads = q.shape(1), kvh = kc.shape(0), group = heads / kvh, n = q.shape(2), dim = kDim;
    const int blocks = (length + kBlock - 1) / kBlock, cap = kc.shape(1);
    if (kvh < 1 || heads % kvh || kc.ndim() != 3 || kc.shape(2) != 26 || kc.dtype() != mx::uint32 || vc.ndim() != 3 || vc.shape(0) != kvh ||
        vc.shape(1) != cap || vc.shape(2) != 18 || vc.dtype() != mx::uint32 || cap < length || length < 1)
        throw std::invalid_argument("decode rows attention tcq: K codes [kvh, cap, 26], V codes [kvh, cap, 18] uint32");
    const int rows = group * n, tiles = (rows + 15) / 16, qrows = tiles * 16;
    // queries into the trellis basis per 128-value group (elementwise butterflies: every row the same arithmetic whatever
    // n is, so the n-row call stays bitwise equal to single rows), FP16 operands
    auto qr = (diag & 4) ? mx::reshape(mx::astype(q, mx::float32), {heads * n * 2, 128}) : tcq_rotate_rows(mx::reshape(mx::astype(q, mx::float32), {heads * n * 2, 128}), false);
    auto query = mx::reshape(mx::astype(qr, mx::float16), {kvh, rows, dim});
    if (qrows > rows) query = mx::concatenate({query, mx::zeros({kvh, qrows - rows, dim}, mx::float16)}, 1);
    query = mx::contiguous(query);
    // codebooks as constant-memory literals (cached broadcast reads; device-memory gathers were the decode's cost)
    static const std::string header = [] {
        std::string h = "#include <metal_simdgroup>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
        auto table = [&](const char* name, const float* v, int n) {
            h += std::string("constant half ") + name + "[" + std::to_string(n) + "] = {";
            for (int i = 0; i < n; ++i) {char b[32]; std::snprintf(b, sizeof b, "%.9g%s", v[i], i + 1 < n ? "," : ""); h += b;}
            h += "};\n";
        };
        table("TCQ_CBK", tcq::kCodebookK, 512); table("TCQ_CBV", tcq::kCodebookV, 256);
        return h;
    }();
    static auto pass1 = mx::fast::metal_kernel("tqllm_attn_tcq_pass1", {"Q", "KC", "VC", "CBK", "CBV", "P"}, {"M", "S", "O"}, R"METAL(
    constexpr int C = 32, RP = TILES * 16, HD = DIM / 2, CQ = C / 4, KW = 26, VW = 18;
    const uint sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const uint blk = threadgroup_position_in_grid.y, kh = threadgroup_position_in_grid.z;
    const int L = P[0], NR = P[1], ROWS = P[3], NB = P[5], CAP = P[7], QROWS = P[8];
    const float sc = as_type<float>(P[6]);
    const int k0 = int(blk) * BLOCK, kn = min(BLOCK, L - k0);
    const int tile = int(sg) / 2, half_d = int(sg) % 2;
    const int row = int(tid) / 4, quarter = int(tid) % 4;
    const int last = row < ROWS ? L - NR + row % NR : -1;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    threadgroup float st[2][RP * C];
    threadgroup half pt[RP * C];
    threadgroup float arow[RP];
    threadgroup float knorm[2][C], vnorm[2][C];
    // STAGE (up to 8 rows: threadgroup memory has room): each 32-key chunk is decoded once into `dec` (K codebook values,
    // then V's), shared by every tile; at 16 rows each fragment decodes from device memory. Same values both ways
    // (FP16 codebook entries; norms applied on S for K, on the fragment for V), so rows match whatever n is
    threadgroup half dec[STAGE ? C * DIM : 1];
    const device half* qrow = Q + (size_t(kh) * QROWS + tile * 16) * DIM;
    const device uint* kcb = KC + size_t(kh) * CAP * KW;
    const device uint* vcb = VC + size_t(kh) * CAP * VW;
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
    short dn[16], dm[16];
    for (ushort i = 0; i < 16; ++i) {
        if (acc0.is_valid_element(i)) {const auto idx = acc0.get_multidimensional_index(i); dn[i] = short(idx[0]); dm[i] = short(idx[1]);}
        else {dn[i] = -1; dm[i] = -1;}
    }
    half qa[8][8];
    for (int ks = 0; ks < 8; ++ks) for (short i = 0; i < 4; ++i) {
        const int d = (half_d * 8 + ks) * 16 + fn + i;
        qa[ks][i] = qrow[size_t(fm) * DIM + d]; qa[ks][4 + i] = qrow[size_t(fm + 8) * DIM + d];
    }
    float m = -INFINITY, s = 0.0f;
    for (int c0 = 0; c0 < kn; c0 += C) {
        // chunk norms (the last word's high half of each group record); keys past the view read as 0 (masked anyway)
        if (tid < uint(2 * C)) {
            const int j = int(tid) % C, gsel = int(tid) / C, key = k0 + c0 + j;
            knorm[gsel][j] = key < L ? float(as_type<half>(ushort(kcb[size_t(key) * KW + gsel * 13 + 12] >> 16))) : 0.0f;
            vnorm[gsel][j] = key < L ? float(as_type<half>(ushort(vcb[size_t(key) * VW + gsel * 9 + 8] >> 16))) : 0.0f;
        }
        if (STAGE) {
            // K codebook values transposed [dim][key] (a fragment reads 4 consecutive keys); one item = 8 consecutive
            // values of one (key, group) from one 64-bit window
            for (uint i = tid; i < uint(C * 32); i += 64 * TILES) {
                const int j = int(i) % C, slot = int(i) / C, grp = slot / 16, t0 = (slot % 16) * 8, key = k0 + c0 + j;
                ulong win = 0ul;
                if (c0 + j < kn && !(DIAG & 1)) {
                    const device uint* g = kcb + size_t(key) * KW + grp * 13;
                    const int bit = 3 * t0, w0 = bit >> 5;
                    win = (ulong(g[w0]) | (w0 + 1 < 13 ? ulong(g[w0 + 1]) << 32 : 0ul)) >> (bit & 31);
                }
                for (int e = 0; e < 8; ++e) dec[(grp * 128 + t0 + e) * C + j] = (c0 + j < kn && !(DIAG & 1)) ? TCQ_CBK[uint(win >> (3 * e)) & 511u] : 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            // partial S [16 rows x 32 keys] over this half = trellis group half_d: codebook values, the norm applied on S
            auto sacc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
            for (ushort i = 0; i < 16; ++i) sacc[i] = 0.0f;
            for (int ks = 0; ks < 8; ++ks) {
                for (short i = 0; i < 8; ++i) ca[i] = qa[ks][i];
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int t = ks * 16 + fm + 8 * h, key0 = c0 + w * 16 + fn;   // value t of the group, keys key0 .. key0 + 3
                    const int bit = 3 * t, w0 = bit >> 5, sh = bit & 31;
                    for (short i = 0; i < 4; ++i) {
                        half v = 0.0h;
                        if (key0 + i < kn) {
                            if (STAGE) {
                                v = dec[(half_d * 128 + t) * C + key0 - c0 + i];
                            } else {
                                const device uint* g = kcb + size_t(k0 + key0 + i) * KW + half_d * 13;
                                const ulong win = ulong(g[w0]) | (w0 + 1 < 13 ? ulong(g[w0 + 1]) << 32 : 0ul);
                                v = TCQ_CBK[uint(win >> sh) & 511u];
                            }
                        }
                        cb[w * 8 + h * 4 + i] = v;
                    }
                }
                op.run(ca, cb, sacc);
            }
            for (ushort i = 0; i < 16; ++i) if (dn[i] >= 0) st[half_d][(tile * 16 + dm[i]) * C + dn[i]] = sacc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float x[CQ];
        float lm = -INFINITY;
        for (int j = 0; j < CQ; ++j) {
            const int jj = quarter * CQ + j, key = k0 + c0 + jj, e = row * C + jj;
            x[j] = key <= last ? (st[0][e] * knorm[0][jj] + st[1][e] * knorm[1][jj]) * sc : -INFINITY;
            lm = max(lm, x[j]);
        }
        lm = max(lm, simd_shuffle_xor(lm, 1)); lm = max(lm, simd_shuffle_xor(lm, 2));
        const float mn = max(m, lm);
        const float alpha = mn == m ? 1.0f : exp(m - mn);
        float ls = 0.0f;
        for (int j = 0; j < CQ; ++j) {
            const int jj = quarter * CQ + j;
            const float pr = x[j] == -INFINITY ? 0.0f : exp(x[j] - mn);
            ls += pr;
            pt[row * C + jj] = half(pr);
        }
        ls += simd_shuffle_xor(ls, 1); ls += simd_shuffle_xor(ls, 2);
        s = s * alpha + ls; m = mn;
        if (quarter == 0) arow[row] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = 0; i < 16; ++i) if (dm[i] >= 0) {
            const float a = arow[tile * 16 + dm[i]];
            if (a != 1.0f) {acc0[i] *= a; acc1[i] *= a; acc2[i] *= a; acc3[i] *= a;}
        }
        if (STAGE) {   // the chunk's V codebook values into the same buffer (every S read of `dec` is behind the barrier above)
            for (uint i = tid; i < uint(C * 32); i += 64 * TILES) {   // [key][dim], 8 values per item
                const int j = int(i) / 32, slot = int(i) % 32, grp = slot / 16, t0 = (slot % 16) * 8, key = k0 + c0 + j;
                ulong win = 0ul;
                if (c0 + j < kn && !(DIAG & 2)) {
                    const device uint* g = vcb + size_t(key) * VW + grp * 9;
                    const int bit = 2 * t0, w0 = bit >> 5;
                    win = (ulong(g[w0]) | (w0 + 1 < 9 ? ulong(g[w0 + 1]) << 32 : 0ul)) >> (bit & 31);
                }
                for (int e = 0; e < 8; ++e) dec[j * DIM + grp * 128 + t0 + e] = (c0 + j < kn && !(DIAG & 2)) ? TCQ_CBV[uint(win >> (2 * e)) & 255u] : 0.0h;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // O[16 x 128 (this group)] += P[16 x 32 keys] (V codebook x the key's group norm)[32 keys x 128]
        const threadgroup half* prow = pt + (tile * 16) * C;
        for (int ks = 0; ks < 2; ++ks) {
            for (short i = 0; i < 4; ++i) {ca[i] = prow[fm * C + ks * 16 + fn + i]; ca[4 + i] = prow[(fm + 8) * C + ks * 16 + fn + i];}
            for (short nb = 0; nb < 4; ++nb) {
                for (short w = 0; w < 2; ++w) for (short h = 0; h < 2; ++h) {
                    const int kcj = ks * 16 + fm + 8 * h, key = k0 + c0 + kcj;          // chunk key, group values t0 .. t0 + 3
                    const int t0 = nb * 32 + w * 16 + fn, bit = 2 * t0, w0 = bit >> 5, sh = bit & 31;
                    if (kcj < kn) {
                        const float vn = vnorm[half_d][kcj];
                        if (STAGE) {
                            for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(float(dec[kcj * DIM + half_d * 128 + t0 + i]) * vn);
                        } else {
                            const device uint* g = vcb + size_t(key) * VW + half_d * 9;
                            const ulong win = (ulong(g[w0]) | (w0 + 1 < 9 ? ulong(g[w0 + 1]) << 32 : 0ul)) >> sh;
                            // the staged path's arithmetic (FP16 codebook value x norm): rows equal whatever n is
                            for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = half(float(TCQ_CBV[uint(win >> (2 * i)) & 255u]) * vn);
                        }
                    } else for (short i = 0; i < 4; ++i) cb[w * 8 + h * 4 + i] = 0.0h;
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
    static auto pass2 = mx::fast::metal_kernel("tqllm_attn_tcq_pass2", {"M", "S", "O", "P"}, {"Y"}, R"METAL(
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
    std::vector<int32_t> p = {length, n, group, rows, 0, blocks, 0, cap, qrows};
    float scf = scale; std::memcpy(&p[6], &scf, 4);
    auto P = mx::array(p.data(), {9}, mx::int32);
    const auto tpl = std::vector<std::pair<std::string, mx::fast::TemplateArg>>{{"DIM", dim}, {"BLOCK", kBlock}, {"TILES", tiles}, {"STAGE", tiles <= 3}, {"DIAG", diag & 3}};
    auto part = pass1({query, kc, vc, tcq_table(3), tcq_table(2), P}, {{kvh * rows * blocks}, {kvh * rows * blocks}, {kvh * rows * blocks * dim}},
                      {mx::float32, mx::float32, mx::float32}, {64 * tiles, blocks, kvh}, {64 * tiles, 1, 1}, tpl, std::nullopt,
                      false, mx::Device(mx::Device::gpu));
    auto y = pass2({part[0], part[1], part[2], P}, {{1, heads, n, dim}}, {mx::float32}, {rows * 32, kvh, 1}, {32, 1, 1},
                   tpl, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    // V was decoded in the trellis basis: rotate the output back per group (x = R^T y, i.e. y R as rows)
    auto back = (diag & 4) ? mx::reshape(y, {heads * n * 2, 128}) : tcq_rotate_rows(mx::reshape(y, {heads * n * 2, 128}), true);
    return mx::astype(mx::astype(mx::reshape(back, {1, heads, n, dim}), mx::bfloat16), mx::float32);
}
// compile the compressed-KV kernels once at load (one Metal compile per row-tile count), so a long request does not pay
// it inside its first decode step (j1153: 1.2 s); `heads` / `kvheads` as the model's attention
inline void warm_compressed_kv_kernels(int heads, int kvheads) {
    namespace mx = mlx::core;
    constexpr int rows = 1024;
    auto x = mx::random::normal({rows, kvheads, 256}, mx::float32);
    auto kc = mx::contiguous(mx::transpose(tcq_encode(x, 3), {1, 0, 2})), vc = mx::contiguous(mx::transpose(tcq_encode(x, 2), {1, 0, 2}));
    auto kv = quantize_kv_int8(mx::reshape(mx::transpose(x, {1, 0, 2}), {1, kvheads, rows, 256}), mx::reshape(mx::transpose(x, {1, 0, 2}), {1, kvheads, rows, 256}));
    std::vector<mx::array> out{tcq_decode(kc, 3, rows), tcq_decode(vc, 2, rows)};
    const int group = heads / kvheads;
    for (int n = 1; n <= 16; ++n) {
        if (n > 1 && (group * (n - 1) + 15) / 16 == (group * n + 15) / 16) continue;   // one compile per tile count
        auto q = mx::random::normal({1, heads, n, 256}, mx::float32);
        out.push_back(decode_rows_attention_tcq(q, kc, vc, 0.0625f, rows));
        out.push_back(decode_rows_attention_v12(q, kv.k8t, kv.v8, kv.kscale, kv.vscale, 0.0625f, rows));
    }
    mx::eval(out);
}
}  // namespace tqllm::mlx_backend

#include "nax_affine.hpp"
#include "tq4.hpp"
#include "q6.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>

namespace mx = mlx::core;
namespace tqllm::mlx_backend {
namespace {
void check_shape(uint64_t rows, uint64_t columns, int bits, int group) {
    if ((bits != 4 && bits != 6) || (group != 16 && group != 64 && group != 128) || rows == 0 || rows % 128 || columns % group)
        throw std::invalid_argument("nax affine needs 4/6 bits, group 16/64/128, rows % 128 == 0, columns % group == 0");
}
// Lane layout of one 16x16 fragment (MLX steel NAX): element i of lane l sits at row fm + (i/4)*8, column fn + i%4.
inline int lane_fm(int lane) {return ((lane >> 2) & 4) | ((lane >> 1) & 3);}
inline int lane_fn(int lane) {return ((((lane >> 2) & 2) | (lane & 1)) * 4);}

const char* kHeader = R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
constant half TQ4_LEVEL[16] = {6.0h, 18.0h, 31.0h, 44.0h, 58.0h, 75.0h, 96.0h, 127.0h,
                                -6.0h, -18.0h, -31.0h, -44.0h, -58.0h, -75.0h, -96.0h, -127.0h};
inline float4 bf16x4(ushort4 v) {return float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16),
                                               as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));}
)METAL";

const char* kSource = R"METAL(
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    // BPT 32-column blocks per threadgroup, SPLIT simdgroups per block, each over K / SPLIT
    constexpr int BPT = SPLIT >= 4 ? 1 : 4 / SPLIT, KS = K / SPLIT;
    const uint local = sg / SPLIT;
    const int split = int(sg % SPLIT);
    const uint blk = threadgroup_position_in_grid.x * BPT + local;
    constexpr int E = MR > 32 ? 64 : (MR > 16 ? 32 : 16);
    threadgroup float part[SPLIT > 1 ? BPT * SPLIT * 32 * E : 1];
    const int row0 = int(threadgroup_position_in_grid.y) * MR;
    const int rows = min(int(R[0]) - row0, MR);
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < 16; i++) cc[i] = 0.0f;
#if NAX_WIDE
    auto ca2 = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cc2 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < 16; i++) cc2[i] = 0.0f;
#endif
#if NAX_WIDE > 1
    // 64-row prefill tile: four row tiles share every decoded weight fragment
    auto ca3 = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cc3 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    auto ca4 = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cc4 = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < 16; i++) {cc3[i] = 0.0f; cc4[i] = 0.0f;}
#endif
    const device uint2* wp = (const device uint2*)W + ((size_t)blk * (K / 16) + split * (KS / 16)) * 32 + lane;
    const device uint* hp = H + (BITS == 6 ? ((size_t)blk * (K / 16) + split * (KS / 16)) * 32 + lane : 0);
    const device ushort* sp = S + (Q6 ? 2 : 1) * (((size_t)blk * (K / GS) + split * (KS / GS)) * 32);
    const device ushort* bp = Bz + ((size_t)blk * (K / GS) + split * (KS / GS)) * 32;
    const device half* xs = X + (size_t)(row0 + fm) * K + fn + split * KS;
    for (int g = 0; g < KS / GS; g++) {
        // Q6: FP32 scales (Q6_K d * sub-block scale, exact), codes 0..63 with the implied bias -32 * scale
        const float4 s0 = Q6 ? *(const device float4*)(sp + 2 * (g * 32 + fn))
                        : TQ4 ? float4(as_type<half4>(*(const device ushort4*)(sp + g * 32 + fn)))
                              : bf16x4(*(const device ushort4*)(sp + g * 32 + fn));
        const float4 s1 = Q6 ? *(const device float4*)(sp + 2 * (g * 32 + 16 + fn))
                        : TQ4 ? float4(as_type<half4>(*(const device ushort4*)(sp + g * 32 + 16 + fn)))
                              : bf16x4(*(const device ushort4*)(sp + g * 32 + 16 + fn));
        const float4 b0 = Q6 ? -32.0f * s0 : TQ4 ? float4(0.0f) : bf16x4(*(const device ushort4*)(bp + g * 32 + fn));
        const float4 b1 = Q6 ? -32.0f * s1 : TQ4 ? float4(0.0f) : bf16x4(*(const device ushort4*)(bp + g * 32 + 16 + fn));
        for (short t = 0; t < GS / 16; t++) {
            const int ks = g * (GS / 16) + t;
            half4 a0 = fm < rows ? *(const device half4*)(xs + ks * 16) : half4(0.0h);
            half4 a1 = fm + 8 < rows ? *(const device half4*)(xs + 8 * K + ks * 16) : half4(0.0h);
            for (short i = 0; i < 4; i++) { ca[i] = a0[i]; ca[4 + i] = a1[i]; }
            uint2 w = wp[ks * 32];
            uint h = BITS == 6 ? hp[ks * 32] : 0u;
            if (TQ4 && !NAX_WIDE) {
                // 16-row kernel (decode, verify): the constant table is cheaper here (j582: ALU decode 97.9 vs 70.1 ms)
                for (short i = 0; i < 8; i++) {
                    const float sa = s0[i & 3], sb = s1[i & 3];
                    cb[i] = half(float(TQ4_LEVEL[(w.x >> (4 * i)) & 15u]) * sa);
                    cb[8 + i] = half(float(TQ4_LEVEL[(w.y >> (4 * i)) & 15u]) * sb);
                }
            } else if (TQ4) {
                // 32-row kernel (prefill, compute-bound): levels as bytes of two words, select + shift, same values
                // (j582: 8K prefill 255 vs 100 tok/s)
                for (short i = 0; i < 8; i++) {
                    const uint ca = (w.x >> (4 * i)) & 15u, cbits = (w.y >> (4 * i)) & 15u;
                    const float la = float(((ca & 4u ? 0x7F604B3Au : 0x2C1F1206u) >> (8u * (ca & 3u))) & 0xFFu);
                    const float lb = float(((cbits & 4u ? 0x7F604B3Au : 0x2C1F1206u) >> (8u * (cbits & 3u))) & 0xFFu);
                    cb[i] = half((ca & 8u ? -la : la) * s0[i & 3]);
                    cb[8 + i] = half((cbits & 8u ? -lb : lb) * s1[i & 3]);
                }
            } else
            for (short p = 0; p < 4; p++) {
                uint l0 = (w.x >> (4 * p)) & 0x000F000Fu, l1 = (w.y >> (4 * p)) & 0x000F000Fu;
                if (BITS == 6) {
                    l0 |= ((h >> (2 * p)) & 0x00030003u) << 4;
                    l1 |= ((h >> (8 + 2 * p)) & 0x00030003u) << 4;
                }
                const float2 q0 = float2(as_type<half2>(l0 | 0x64006400u)) - 1024.0f;
                const float2 q1 = float2(as_type<half2>(l1 | 0x64006400u)) - 1024.0f;
                const half2 v0 = half2(fma(q0, float2(s0[p]), float2(b0[p])));
                const half2 v1 = half2(fma(q1, float2(s1[p]), float2(b1[p])));
                cb[p] = v0.x; cb[p + 4] = v0.y; cb[8 + p] = v1.x; cb[12 + p] = v1.y;
            }
            op.run(ca, cb, cc);
#if NAX_WIDE
            {
                half4 a2 = fm + 16 < rows ? *(const device half4*)(xs + 16 * K + ks * 16) : half4(0.0h);
                half4 a3 = fm + 24 < rows ? *(const device half4*)(xs + 24 * K + ks * 16) : half4(0.0h);
                for (short i = 0; i < 4; i++) { ca2[i] = a2[i]; ca2[4 + i] = a3[i]; }
                op.run(ca2, cb, cc2);
            }
#endif
#if NAX_WIDE > 1
            {
                half4 a4 = fm + 32 < rows ? *(const device half4*)(xs + 32 * K + ks * 16) : half4(0.0h);
                half4 a5 = fm + 40 < rows ? *(const device half4*)(xs + 40 * K + ks * 16) : half4(0.0h);
                for (short i = 0; i < 4; i++) { ca3[i] = a4[i]; ca3[4 + i] = a5[i]; }
                op.run(ca3, cb, cc3);
                half4 a6 = fm + 48 < rows ? *(const device half4*)(xs + 48 * K + ks * 16) : half4(0.0h);
                half4 a7 = fm + 56 < rows ? *(const device half4*)(xs + 56 * K + ks * 16) : half4(0.0h);
                for (short i = 0; i < 4; i++) { ca4[i] = a6[i]; ca4[4 + i] = a7[i]; }
                op.run(ca4, cb, cc4);
            }
#endif
        }
    }
    if (SPLIT > 1) {
        // partials meet in threadgroup memory; split 0 adds them in split order, so the sum order is fixed per element
        threadgroup float* mine = part + ((local * SPLIT + split) * 32 + lane) * E;
        for (short i = 0; i < 16; i++) mine[i] = cc[i];
#if NAX_WIDE
        for (short i = 0; i < 16; i++) mine[16 + i] = cc2[i];
#endif
#if NAX_WIDE > 1
        for (short i = 0; i < 16; i++) {mine[32 + i] = cc3[i]; mine[48 + i] = cc4[i];}
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split != 0) return;
        for (int p = 1; p < SPLIT; p++) {
            const threadgroup float* other = part + ((local * SPLIT + p) * 32 + lane) * E;
            for (short i = 0; i < 16; i++) cc[i] = cc[i] + other[i];
#if NAX_WIDE
            for (short i = 0; i < 16; i++) cc2[i] = cc2[i] + other[16 + i];
#endif
#if NAX_WIDE > 1
            for (short i = 0; i < 16; i++) {cc3[i] = cc3[i] + other[32 + i]; cc4[i] = cc4[i] + other[48 + i];}
#endif
        }
    }
    const uint col0 = blk * 32 + fn;
    for (short i = 0; i < 16; i++) {
        const short r = fm + ((i & 7) >> 2) * 8;
        const uint c = col0 + (i >> 3) * 16 + (i & 3);
        const float v = BF ? float(bfloat16_t(float(cc[i]))) : float(cc[i]);
        if (r < rows) Y[(size_t)(row0 + r) * N + c] = v;
#if NAX_WIDE
        if (r + 16 < rows) Y[(size_t)(row0 + r + 16) * N + c] = BF ? float(bfloat16_t(float(cc2[i]))) : float(cc2[i]);
#endif
#if NAX_WIDE > 1
        if (r + 32 < rows) Y[(size_t)(row0 + r + 32) * N + c] = BF ? float(bfloat16_t(float(cc3[i]))) : float(cc3[i]);
        if (r + 48 < rows) Y[(size_t)(row0 + r + 48) * N + c] = BF ? float(bfloat16_t(float(cc4[i]))) : float(cc4[i]);
#endif
    }
)METAL";

// One thread per (32-row block, 16-column step, lane): the 16 codes of its matmul fragment, written as FP16 values
// exactly as the matmul kernel decodes them (half(fma(code, scale, bias)) or half(level * scale)).
const char* kDequant = R"METAL(
    const uint t = thread_position_in_grid.x;
    const uint lane = t % 32u;
    const uint ks = (t / 32u) % uint(STEPS);
    const uint blk = t / 32u / uint(STEPS);
    if (blk >= uint(BLOCKS)) return;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const uint2 w = ((const device uint2*)W)[((size_t)blk * STEPS + ks) * 32 + lane];
    const uint h = BITS == 6 ? H[((size_t)blk * STEPS + ks) * 32 + lane] : 0u;
    const int g = int(ks * 16u) / GS;
    const device ushort* sp = S + ((size_t)blk * GPR + g) * 32;
    const device ushort* bp = Bz + ((size_t)blk * GPR + g) * 32;
    device half* out = Y + (size_t)(blk * 32 + fn) * K + ks * 16u + fm;
    for (short p = 0; p < 4; p++) {
        float sa, sb, ba = 0.0f, bb = 0.0f;
        if (TQ4) {
            sa = float(as_type<half>(sp[fn + p])); sb = float(as_type<half>(sp[16 + fn + p]));
        } else {
            sa = as_type<float>(uint(sp[fn + p]) << 16); sb = as_type<float>(uint(sp[16 + fn + p]) << 16);
            ba = as_type<float>(uint(bp[fn + p]) << 16); bb = as_type<float>(uint(bp[16 + fn + p]) << 16);
        }
        for (short j = 0; j < 2; j++) {
            const short i = p + 4 * j;
            uint qa = (w.x >> (4 * i)) & 15u, qb = (w.y >> (4 * i)) & 15u;
            half va, vb;
            if (TQ4) {
                va = half(float(TQ4_LEVEL[qa]) * sa); vb = half(float(TQ4_LEVEL[qb]) * sb);
            } else {
                if (BITS == 6) {qa |= ((h >> (2 * p + 16 * j)) & 3u) << 4; qb |= ((h >> (8 + 2 * p + 16 * j)) & 3u) << 4;}
                va = half(fma(float(qa), sa, ba)); vb = half(fma(float(qb), sb, bb));
            }
            out[(size_t)p * K + 8 * j] = va;
            out[(size_t)(16 + p) * K + 8 * j] = vb;
        }
    }
)METAL";
}

uint64_t nax_affine_bytes(uint64_t rows, uint64_t columns, int bits, int group) {
    check_shape(rows, columns, bits, group);
    return rows * columns * bits / 8 + rows * (columns / group) * 4;
}

std::vector<uint8_t> nax_affine_pack(std::span<const uint8_t> codes, std::span<const uint16_t> scales,
                                     std::span<const uint16_t> biases, int rows, int columns, int bits, int group) {
    check_shape(rows, columns, bits, group);
    const size_t n = size_t(rows) * columns, groups = size_t(rows) * (columns / group);
    if (codes.size() != n || scales.size() != groups || biases.size() != groups)
        throw std::invalid_argument("nax affine pack: wrong input sizes");
    const int blocks = rows / 32, steps = columns / 16, gpr = columns / group;
    std::vector<uint32_t> words(size_t(blocks) * steps * 32 * 2, 0), high(bits == 6 ? size_t(blocks) * steps * 32 : 0, 0);
    std::vector<uint16_t> s(groups), b(groups);
    for (int blk = 0; blk < blocks; ++blk)
        for (int ks = 0; ks < steps; ++ks)
            for (int lane = 0; lane < 32; ++lane)
                for (int w = 0; w < 2; ++w)
                    for (int i = 0; i < 8; ++i) {
                        const int r = blk * 32 + w * 16 + lane_fn(lane) + (i & 3);
                        const int c = ks * 16 + lane_fm(lane) + (i >> 2) * 8;
                        const uint32_t q = codes[size_t(r) * columns + c];
                        if (q >> bits) throw std::invalid_argument("nax affine pack: code out of range");
                        const size_t at = (size_t(blk) * steps + ks) * 32 + lane;
                        words[at * 2 + w] |= (q & 15u) << (4 * i);
                        if (bits == 6) high[at] |= (q >> 4) << (w * 8 + 2 * (i & 3) + 16 * (i >> 2));
                    }
    for (int blk = 0; blk < blocks; ++blk)
        for (int g = 0; g < gpr; ++g)
            for (int j = 0; j < 32; ++j) {
                const size_t from = size_t(blk * 32 + j) * gpr + g, to = (size_t(blk) * gpr + g) * 32 + j;
                s[to] = scales[from];
                b[to] = biases[from];
            }
    std::vector<uint8_t> out(nax_affine_bytes(rows, columns, bits, group));
    uint8_t* at = out.data();
    auto put = [&](const void* p, size_t bytes) {std::memcpy(at, p, bytes); at += bytes;};
    put(words.data(), words.size() * 4);
    if (bits == 6) put(high.data(), high.size() * 4);
    put(s.data(), s.size() * 2);
    put(b.data(), b.size() * 2);
    return out;
}

NaxAffine nax_affine_load(std::span<const uint8_t> payload, int rows, int columns, int bits, int group) {
    if (payload.size() != nax_affine_bytes(rows, columns, bits, group))
        throw std::invalid_argument("nax affine payload has the wrong size");
    const int blocks = rows / 32, steps = columns / 16, gpr = columns / group;
    const uint8_t* at = payload.data();
    auto take = [&](size_t bytes) {const uint8_t* p = at; at += bytes; return p;};
    auto words = mx::array(reinterpret_cast<const uint32_t*>(take(size_t(blocks) * steps * 32 * 8)),
                           {blocks, steps, 32, 2}, mx::uint32);
    auto high = bits == 6 ? mx::array(reinterpret_cast<const uint32_t*>(take(size_t(blocks) * steps * 32 * 4)),
                                      {blocks, steps, 32}, mx::uint32)
                          : mx::zeros({64}, mx::uint32);
    const size_t sb = size_t(blocks) * gpr * 32 * 2;
    auto scales = mx::array(reinterpret_cast<const uint16_t*>(take(sb)), {blocks, gpr, 32}, mx::uint16);
    auto biases = mx::array(reinterpret_cast<const uint16_t*>(take(sb)), {blocks, gpr, 32}, mx::uint16);
    mx::eval(words, high, scales, biases);
    return {words, high, scales, biases, rows, columns, bits, group};
}

// K splits for one matrix: enough simdgroups to keep the memory system busy when each matmul waits for the previous
// one (N/32 simdgroups per split), each split a whole number of groups. Depends only on the shape, never on the row
// count, so a single row and a verify batch sum the same partial products in the same order.
int nax_affine_splits(const NaxAffine& w) {
    static const int cap = std::getenv("TQLLM_MLX_NAX_MAX_SPLIT") ? std::atoi(std::getenv("TQLLM_MLX_NAX_MAX_SPLIT")) : 8;
    int split = 1;
    while (split < cap && (w.rows / 32) * split < 512 && (w.columns / w.group) % (split * 2) == 0) split *= 2;
    return split;
}

mx::array nax_affine_matmul(const NaxAffine& w, const mx::array& input, bool round_bf16) {
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax affine matmul: input must be [m, columns]");
    // the 16-row kernel does not declare the second row tile, so it holds fewer registers per simdgroup
    static auto kernel16 = mx::fast::metal_kernel("tqllm_nax_affine16", {"W", "H", "S", "Bz", "X", "R"}, {"Y"},
                                                  std::string("#define NAX_WIDE 0\n") + kSource, kHeader);
    static auto kernel32 = mx::fast::metal_kernel("tqllm_nax_affine32", {"W", "H", "S", "Bz", "X", "R"}, {"Y"},
                                                  std::string("#define NAX_WIDE 1\n") + kSource, kHeader);
    // TQLLM_MLX_NAX_MR64=1: chunks over 32 rows use the 64-row tile without split-K (row tiles give the parallelism)
    static auto kernel64 = mx::fast::metal_kernel("tqllm_nax_affine64", {"W", "H", "S", "Bz", "X", "R"}, {"Y"},
                                                  std::string("#define NAX_WIDE 2\n") + kSource, kHeader);
    static const bool mr64 = std::getenv("TQLLM_MLX_NAX_MR64") != nullptr;
    const int m = input.shape(0), mr = m <= 16 ? 16 : (m <= 32 || !mr64 ? 32 : 64);
    const int split = mr == 64 ? 1 : nax_affine_splits(w);
    auto& kernel = mr == 16 ? kernel16 : mr == 32 ? kernel32 : kernel64;
    auto x = mx::contiguous(mx::astype(input, mx::float16));
    const uint32_t m32 = uint32_t(m);
    auto count = mx::array(&m32, {1}, mx::uint32);
    const int bpt = split >= 4 ? 1 : 4 / split, threads = 32 * bpt * split;
    return kernel({w.words, w.high, w.scales, w.biases, x, count}, {{m, w.rows}}, {mx::float32},
                    {w.rows / 32 / bpt * threads, (m + mr - 1) / mr, 1}, {threads, 1, 1},
                    {{"K", w.columns}, {"N", w.rows}, {"MR", mr}, {"BITS", w.bits}, {"GS", w.group}, {"SPLIT", split},
                     {"BF", round_bf16}, {"TQ4", w.tq4}, {"Q6", w.q6}},
                    std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

mx::array nax_affine_dequant(const NaxAffine& w) {
    if (w.q6) throw std::invalid_argument("nax affine dequant: Q6 mode not supported");
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_affine_dequant", {"W", "H", "S", "Bz"}, {"Y"}, kDequant, kHeader);
    const int blocks = w.rows / 32, steps = w.columns / 16, gpr = w.columns / w.group;
    return kernel({w.words, w.high, w.scales, w.biases}, {{w.rows, w.columns}}, {mx::float16}, {blocks * steps * 32, 1, 1},
                  {256, 1, 1}, {{"K", w.columns}, {"STEPS", steps}, {"BLOCKS", blocks}, {"GPR", gpr}, {"BITS", w.bits},
                  {"GS", w.group}, {"TQ4", w.tq4}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

mx::array nax_affine_gemm(const NaxAffine& w, const mx::array& input, bool round_bf16) {
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax affine gemm: input must be [m, columns]");
    auto x = input.dtype() == mx::float16 ? input : mx::astype(input, mx::float16);
    auto y = mx::matmul(x, mx::transpose(nax_affine_dequant(w)));
    // FP16 product -> BF16 rounding -> FP32 as one fused elementwise pass
    static auto to_bf = mx::compile([](const std::vector<mx::array>& a) {
        return std::vector<mx::array>{mx::astype(mx::astype(mx::astype(a[0], mx::float32), mx::bfloat16), mx::float32)};
    }, true);
    return round_bf16 ? to_bf({y})[0] : mx::astype(y, mx::float32);
}

// Prefill on the matrix unit's int8 path (j638: 48 TOPS int8 vs 31 TFLOPS fp16 when the operands are device tensors and
// matmul2d streams a whole K slice per run). TQ4 levels are integers (+-6..127), so the weights are exact as int8; the FP16
// scale of each 128-column group and the input's per-(row, group) int8 scale are applied to the int32 tile after each
// group's run. Both int8 operands are group-major, [K/128][rows][128], so a group's slice of a row tile is one packed tensor.
const char* kLevels = R"METAL(
    const uint t = thread_position_in_grid.x;
    const uint lane = t % 32u;
    const uint ks = (t / 32u) % uint(STEPS);
    const uint blk = t / 32u / uint(STEPS);
    if (blk >= uint(BLOCKS)) return;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const uint2 w = ((const device uint2*)W)[((size_t)blk * STEPS + ks) * 32 + lane];
    const uint col = ks * 16u + fm, g = col / 128u, cg = col % 128u;
    device int8_t* out = Y + ((size_t)g * uint(ROWS) + blk * 32 + fn) * 128u + cg;
    for (short p = 0; p < 4; p++) {
        for (short j = 0; j < 2; j++) {
            const short i = p + 4 * j;
            const uint qa = (w.x >> (4 * i)) & 15u, qb = (w.y >> (4 * i)) & 15u;
            out[(size_t)p * 128u + 8 * j] = int8_t(TQ4_LEVEL[qa]);
            out[(size_t)(16 + p) * 128u + 8 * j] = int8_t(TQ4_LEVEL[qb]);
        }
    }
)METAL";
mx::array nax_tq4_levels_int8(const NaxAffine& w) {
    if (!w.tq4 || w.columns % 128) throw std::invalid_argument("nax int8 levels: TQ4 matrices only");
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_tq4_levels", {"W"}, {"Y"}, kLevels, kHeader);
    const int blocks = w.rows / 32, steps = w.columns / 16;
    return kernel({w.words}, {{w.columns / 128, w.rows, 128}}, {mx::int8}, {blocks * steps * 32, 1, 1}, {256, 1, 1},
                  {{"STEPS", steps}, {"BLOCKS", blocks}, {"ROWS", w.rows}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

std::pair<mx::array, mx::array> nax_quantize_rows_int8(const mx::array& input, int row_pad) {
    if (input.ndim() != 2 || input.shape(1) % 128) throw std::invalid_argument("nax int8 quantize: input must be [m, k], k a multiple of 128");
    auto x = mx::contiguous(input.dtype() == mx::float16 ? input : mx::astype(input, mx::float16));
    const int m = x.shape(0), k = x.shape(1), gpr = k / 128, mp = ((m + row_pad - 1) / row_pad) * row_pad;
    // one threadgroup per (row, 128-column block): symmetric int8 with the block's own scale; rows past m are zero
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_quantize_blocks_int8", {"x"}, {"q", "s"}, R"METAL(
    const uint g = threadgroup_position_in_grid.x, row = threadgroup_position_in_grid.y, tid = thread_position_in_threadgroup.x;
    const float v = row < uint(M) ? float(x[size_t(row) * uint(K) + g * 128u + tid]) : 0.0f;
    float amax = simd_max(abs(v));
    threadgroup float red[4];
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = max(max(red[0], red[1]), max(red[2], red[3]));
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    q[(size_t(g) * uint(MP) + row) * 128u + tid] = int8_t(clamp(rint(v / scale), -127.0f, 127.0f));
    if (tid == 0) s[size_t(row) * uint(GPR) + g] = row < uint(M) ? scale : 0.0f;
)METAL");
    auto out = kernel({x}, {{gpr, mp, 128}, {mp, gpr}}, {mx::int8, mx::float32}, {gpr * 128, mp, 1}, {128, 1, 1},
                      {{"K", k}, {"GPR", gpr}, {"M", m}, {"MP", mp}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return {out[0], out[1]};
}

const char* kInt8Gemm = R"METAL(
    const uint tid = thread_position_in_threadgroup.x;
    // tile order: SB column tiles of one row tile are dispatched together, so they share the input rows in cache
    const uint tiles_m = uint(MP) / uint(MT), tiles_n = uint(N) / uint(NT);
    const uint l = threadgroup_position_in_grid.x, block = l / (uint(SB) * tiles_m), rem = l - block * uint(SB) * tiles_m;
    const uint width = min(uint(SB), tiles_n - block * uint(SB));
    const uint n0 = (block * uint(SB) + rem % width) * uint(NT), m0 = (rem / width) * uint(MT);
    threadgroup half S[NT];
    threadgroup float SA[MT];
    using TX = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, 128, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
    auto acc = op.template get_destination_cooperative_tensor<TX, TX, int32_t>();
    constexpr uint CAP = uint(MT) * uint(NT) / (32u * uint(SIMDS));
    float facc[CAP];
    ushort colidx[CAP], rowidx[CAP];
#pragma unroll
    for (uint i = 0; i < CAP; ++i) {
        facc[i] = 0.0f; colidx[i] = 0; rowidx[i] = 0;
        if (acc.is_valid_element(i)) {auto c = acc.get_multidimensional_index(i); colidx[i] = ushort(c[0]); rowidx[i] = ushort(c[1]);}
    }
    for (uint g = 0; g < uint(GPR); ++g) {
        for (uint i = tid; i < uint(NT); i += 32u * uint(SIMDS)) {
            const uint n = n0 + i;
            S[i] = as_type<half>(sw[((n >> 5) * uint(GPR) + g) * 32u + (n & 31u)]);
        }
        for (uint i = tid; i < uint(MT); i += 32u * uint(SIMDS)) SA[i] = sa[(m0 + i) * uint(GPR) + g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tx = TX((device int8_t*)(xq + (size_t(g) * uint(MP) + m0) * 128u), dextents<int, 2>(128, MT));
        auto tw = TX((device int8_t*)(wq + (size_t(g) * uint(N) + n0) * 128u), dextents<int, 2>(128, NT));
#pragma unroll
        for (uint i = 0; i < CAP; ++i) if (acc.is_valid_element(i)) acc[i] = 0;
        op.run(tx, tw, acc);
#pragma unroll
        for (uint i = 0; i < CAP; ++i) if (acc.is_valid_element(i)) facc[i] += float(acc[i]) * float(S[colidx[i]]) * SA[rowidx[i]];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (uint i = 0; i < CAP; ++i) {
        if (!acc.is_valid_element(i)) continue;
        const uint col = n0 + colidx[i], row = m0 + rowidx[i];
        if (row < uint(M)) y[row * uint(N) + col] = ROUND_BF16 ? float(bfloat16_t(facc[i])) : facc[i];
    }
)METAL";
// Fast variant: one matmul2d run over the whole K per tile (48 TOPS, j638) admits one scale per operand row, so the
// weights are re-quantized per row (level * s_g / max_g s_g, ~1 % rms error) and the input per token.
const char* kLevelsRows = R"METAL(
    const uint t = thread_position_in_grid.x;
    const uint lane = t % 32u;
    const uint ks = (t / 32u) % uint(STEPS);
    const uint blk = t / 32u / uint(STEPS);
    if (blk >= uint(BLOCKS)) return;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const uint2 w = ((const device uint2*)W)[((size_t)blk * STEPS + ks) * 32 + lane];
    const int g = int(ks * 16u) / 128;
    const device ushort* sp = S + ((size_t)blk * GPR + g) * 32;
    const uint c0 = ks * 16u + fm;
    const float s0 = SM[c0], s1 = SM[c0 + 8];
    device int8_t* out = Y + (size_t)(blk * 32 + fn) * K + c0;
    for (short p = 0; p < 4; p++) {
        const float ra = float(as_type<half>(sp[fn + p])) / R[blk * 32 + fn + p];
        const float rb = float(as_type<half>(sp[16 + fn + p])) / R[blk * 32 + 16 + fn + p];
        for (short j = 0; j < 2; j++) {
            const short i = p + 4 * j;
            const float sj = j ? s1 : s0;
            const uint qa = (w.x >> (4 * i)) & 15u, qb = (w.y >> (4 * i)) & 15u;
            out[(size_t)p * K + 8 * j] = int8_t(clamp(rint(float(TQ4_LEVEL[qa]) * ra * sj), -127.0f, 127.0f));
            out[(size_t)(16 + p) * K + 8 * j] = int8_t(clamp(rint(float(TQ4_LEVEL[qb]) * rb * sj), -127.0f, 127.0f));
        }
    }
)METAL";
// per-row int8 weights [rows, columns] of w * diag(sm) and the row scale [rows]: max over groups of the group scale
// times the group's largest sm (an upper bound on the row's |w * sm|, so no clamping in the kernel)
std::pair<mx::array, mx::array> nax_tq4_levels_int8_rows(const NaxAffine& w, const mx::array& sm) {
    if (!w.tq4 || w.columns % 128) throw std::invalid_argument("nax int8 row levels: TQ4 matrices only");
    const int gpr = w.columns / 128;
    auto scales = mx::astype(mx::view(w.scales, mx::float16), mx::float32);  // [rows/32, GPR, 32] lane order
    auto smg = mx::reshape(mx::max(mx::reshape(sm, {gpr, 128}), 1), {1, gpr, 1});
    auto rowmax = mx::reshape(mx::max(mx::multiply(scales, smg), 1), {w.rows});
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_tq4_levels_rows", {"W", "S", "R", "SM"}, {"Y"}, kLevelsRows, kHeader);
    const int blocks = w.rows / 32, steps = w.columns / 16;
    auto q = kernel({w.words, w.scales, rowmax, sm}, {{w.rows, w.columns}}, {mx::int8}, {blocks * steps * 32, 1, 1}, {256, 1, 1},
                    {{"K", w.columns}, {"STEPS", steps}, {"BLOCKS", blocks}, {"GPR", gpr}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return {q, rowmax};
}
// per-token int8 of x / diag(sm) (sm = ones leaves the numerics of the plain path untouched)
std::pair<mx::array, mx::array> nax_quantize_tokens_int8(const mx::array& input, int row_pad, const mx::array& sm) {
    if (input.ndim() != 2) throw std::invalid_argument("nax int8 token quantize: input must be [m, k]");
    auto x = mx::contiguous(input.dtype() == mx::float16 ? input : mx::astype(input, mx::float16));
    const int m = x.shape(0), k = x.shape(1), mp = ((m + row_pad - 1) / row_pad) * row_pad;
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_quantize_tokens_int8", {"x", "sm"}, {"q", "s"}, R"METAL(
    const uint row = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x;
    const device half* xr = x + size_t(row) * uint(K);
    float amax = 0.0f;
    if (row < uint(M)) for (uint i = tid; i < uint(K); i += 256u) amax = max(amax, abs(float(xr[i]) / sm[i]));
    amax = simd_max(amax);
    threadgroup float red[8];
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (uint g = 1; g < 8u; ++g) amax = max(amax, red[g]);
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (uint i = tid; i < uint(K); i += 256u) q[size_t(row) * uint(K) + i] = row < uint(M) ? int8_t(clamp(rint(float(xr[i]) / sm[i] / scale), -127.0f, 127.0f)) : int8_t(0);
    if (tid == 0) s[row] = row < uint(M) ? scale : 0.0f;
)METAL");
    auto out = kernel({x, sm}, {{mp, k}, {mp}}, {mx::int8, mx::float32}, {mp * 256, 1, 1}, {256, 1, 1}, {{"K", k}, {"M", m}},
                      std::nullopt, false, mx::Device(mx::Device::gpu));
    return {out[0], out[1]};
}
// SmoothQuant-style fold for the fast path: x' = x / s, w' = w * s per input column with s = xmax^a / wmax^(1-a),
// xmax from the chunk itself (no calibration files), wmax cached per matrix. Outlier channels leave the per-token
// input quantization for the per-row weight requant. a <= 0 means off (s = 1). The column max lives in the matrix
// (a cache keyed by array id broke when a freed matrix's id was reused, j641).
static mx::array nax_int8_smooth(const NaxAffine& w, const mx::array& x, float alpha) {
    if (!w.colmax) {
        auto wm = mx::max(mx::abs(mx::astype(nax_affine_dequant(w), mx::float32)), 0);
        mx::eval(wm);
        w.colmax = wm;
    }
    const auto& wmax = *w.colmax;
    auto xm = mx::max(mx::abs(mx::astype(x, mx::float32)), 0);
    auto s = mx::divide(mx::power(xm, mx::array(alpha)), mx::power(wmax, mx::array(1.0f - alpha)));
    auto ok = mx::logical_and(mx::greater(xm, mx::array(0.0f)), mx::greater(wmax, mx::array(0.0f)));
    return mx::where(ok, mx::maximum(s, mx::array(1e-5f)), mx::array(1.0f));
}
const char* kInt8GemmRows = R"METAL(
    const uint tiles_m = uint(MP) / uint(MT), tiles_n = uint(N) / uint(NT);
    const uint l = threadgroup_position_in_grid.x, block = l / (uint(SB) * tiles_m), rem = l - block * uint(SB) * tiles_m;
    const uint width = min(uint(SB), tiles_n - block * uint(SB));
    const uint n0 = (block * uint(SB) + rem % width) * uint(NT), m0 = (rem / width) * uint(MT);
    using TX = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
    auto tx = TX((device int8_t*)(xq + size_t(m0) * uint(K)), dextents<int, 2>(K, MT));
    auto tw = TX((device int8_t*)(wq + size_t(n0) * uint(K)), dextents<int, 2>(K, NT));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(MT, NT, metal::dynamic_extent, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SIMDS>> op;
    auto acc = op.template get_destination_cooperative_tensor<TX, TX, int32_t>();
    constexpr uint CAP = uint(MT) * uint(NT) / (32u * uint(SIMDS));
#pragma unroll
    for (uint i = 0; i < CAP; ++i) if (acc.is_valid_element(i)) acc[i] = 0;
    op.run(tx, tw, acc);
#pragma unroll
    for (uint i = 0; i < CAP; ++i) {
        if (!acc.is_valid_element(i)) continue;
        auto c = acc.get_multidimensional_index(i);
        const uint col = n0 + uint(c[0]), row = m0 + uint(c[1]);
        if (row < uint(M)) {
            const float v = float(acc[i]) * sa[row] * sw[col];
            y[row * uint(N) + col] = ROUND_BF16 ? float(bfloat16_t(v)) : v;
        }
    }
)METAL";
mx::array nax_int8_gemm_rows(const NaxAffine& w, const mx::array& input, bool round_bf16, float smooth_alpha) {
    if (!w.tq4 || w.columns % 128 || w.rows % 128) throw std::invalid_argument("nax int8 gemm: TQ4 matrices (rows % 128) only");
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax int8 gemm: input must be [m, columns]");
    constexpr int MT = 64, NT = 128, SIMDS = 4, SB = 8;
    static auto kernels = [] {
        auto make = [](bool round) {
            return mx::fast::metal_kernel(std::string("tqllm_nax_int8_gemm_rows") + (round ? "_bf" : ""), {"xq", "sa", "wq", "sw"}, {"y"}, kInt8GemmRows,
                                          std::string("#include <metal_tensor>\n") + kHeader);
        };
        return std::pair{make(false), make(true)};
    }();
    auto sm = smooth_alpha > 0 ? nax_int8_smooth(w, input, smooth_alpha) : mx::ones({w.columns}, mx::float32);
    auto [xq, sa] = nax_quantize_tokens_int8(input, MT, sm);
    auto [wq, sw] = nax_tq4_levels_int8_rows(w, sm);
    const int m = input.shape(0), mp = xq.shape(0), n = w.rows, k = w.columns;
    return (round_bf16 ? kernels.second : kernels.first)(
        {xq, sa, wq, sw}, {{m, n}}, {mx::float32}, {(n / NT) * (mp / MT) * 32 * SIMDS, 1, 1}, {32 * SIMDS, 1, 1},
        {{"MT", MT}, {"NT", NT}, {"SIMDS", SIMDS}, {"SB", SB}, {"M", m}, {"MP", mp}, {"N", n}, {"K", k}, {"ROUND_BF16", round_bf16}},
        std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
mx::array nax_int8_gemm(const NaxAffine& w, const mx::array& input, bool round_bf16, bool exact, float smooth_alpha) {
    if (!exact) {
        static const float env_alpha = std::getenv("TQLLM_MLX_INT8_SMOOTH") ? float(std::atof(std::getenv("TQLLM_MLX_INT8_SMOOTH"))) : 0.0f;
        return nax_int8_gemm_rows(w, input, round_bf16, smooth_alpha < 0 ? env_alpha : smooth_alpha);
    }
    if (!w.tq4 || w.group != 128 || w.columns % 128 || w.rows % 128) throw std::invalid_argument("nax int8 gemm: TQ4 matrices (group 128, rows % 128) only");
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax int8 gemm: input must be [m, columns]");
    constexpr int MT = 64, NT = 128, SIMDS = 4, SB = 8;  // j638: 48 TOPS at this tile
    static auto kernels = [] {
        auto make = [](bool round) {
            return mx::fast::metal_kernel(std::string("tqllm_nax_int8_gemm") + (round ? "_bf" : ""), {"xq", "sa", "wq", "sw"}, {"y"}, kInt8Gemm,
                                          std::string("#include <metal_tensor>\n") + kHeader);
        };
        return std::pair{make(false), make(true)};
    }();
    auto [xq, sa] = nax_quantize_rows_int8(input, MT);
    auto wq = nax_tq4_levels_int8(w);
    const int m = input.shape(0), mp = xq.shape(1), n = w.rows, k = w.columns;
    return (round_bf16 ? kernels.second : kernels.first)(
        {xq, sa, wq, w.scales}, {{m, n}}, {mx::float32}, {(n / NT) * (mp / MT) * 32 * SIMDS, 1, 1}, {32 * SIMDS, 1, 1},
        {{"MT", MT}, {"NT", NT}, {"SIMDS", SIMDS}, {"SB", SB}, {"GPR", k / 128}, {"M", m}, {"MP", mp}, {"N", n}, {"ROUND_BF16", round_bf16}},
        std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

NaxAffine nax_affine_concat(const std::vector<const NaxAffine*>& parts) {
    if (parts.empty()) throw std::invalid_argument("nax affine concat: no parts");
    const auto& f = *parts.front();
    std::vector<mx::array> w, h, sc, b;
    int rows = 0;
    for (const auto* p : parts) {
        if (p->columns != f.columns || p->bits != f.bits || p->group != f.group)
            throw std::invalid_argument("nax affine concat: parts differ in columns, bits or group");
        w.push_back(p->words); h.push_back(p->high); sc.push_back(p->scales); b.push_back(p->biases);
        rows += p->rows;
    }
    NaxAffine out{mx::concatenate(w, 0), f.bits == 6 ? mx::concatenate(h, 0) : f.high, mx::concatenate(sc, 0),
                  mx::concatenate(b, 0), rows, f.columns, f.bits, f.group};
    mx::eval(out.words, out.high, out.scales, out.biases);
    return out;
}

NaxAffine nax_affine_select_rows(std::span<const uint8_t> payload, int rows, int columns, int bits, int group,
                                 std::span<const uint32_t> keep) {
    if (payload.size() != nax_affine_bytes(rows, columns, bits, group) || keep.empty() || keep.size() % 128)
        throw std::invalid_argument("nax affine select rows: bad payload or row count");
    const int blocks = rows / 32, steps = columns / 16, gpr = columns / group, n = int(keep.size());
    const uint32_t* words = reinterpret_cast<const uint32_t*>(payload.data());
    const uint32_t* high = words + size_t(blocks) * steps * 32 * 2;
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(payload.data() + size_t(rows) * columns * bits / 8);
    const uint16_t* biases = scales + size_t(blocks) * gpr * 32;
    std::vector<uint8_t> codes(size_t(n) * columns);
    std::vector<uint16_t> s(size_t(n) * gpr), b(s.size());
    for (int out = 0; out < n; ++out) {
        const uint32_t r = keep[out];
        if (r >= uint32_t(rows)) throw std::invalid_argument("nax affine select rows: row out of range");
        const int blk = int(r / 32), w = int(r % 32) / 16, nin = int(r % 16), q = nin >> 2, e = nin & 3;
        for (int c = 0; c < columns; ++c) {
            const int ks = c / 16, kk = c % 16, f = kk & 7, i = (kk >> 3) * 4 + e;
            const int lane = ((f >> 2) << 4) | ((q >> 1) << 3) | (((f >> 1) & 1) << 2) | ((f & 1) << 1) | (q & 1);
            const size_t at = (size_t(blk) * steps + ks) * 32 + lane;
            uint32_t code = (words[at * 2 + w] >> (4 * i)) & 15u;
            if (bits == 6) code |= ((high[at] >> (w * 8 + 2 * (i & 3) + 16 * (i >> 2))) & 3u) << 4;
            codes[size_t(out) * columns + c] = uint8_t(code);
        }
        for (int g = 0; g < gpr; ++g) {
            const size_t from = (size_t(blk) * gpr + g) * 32 + (r % 32);
            s[size_t(out) * gpr + g] = scales[from];
            b[size_t(out) * gpr + g] = biases[from];
        }
    }
    auto packed = nax_affine_pack(codes, s, b, n, columns, bits, group);
    return nax_affine_load(packed, n, columns, bits, group);
}

NaxAffine nax_from_tq4(const Tq4Weights& t) {
    if (!t.integer_levels || t.codes.ndim() != 2 || t.codes.dtype() != mx::uint8 || t.scales.dtype() != mx::float16)
        throw std::invalid_argument("nax tq4: needs integer-level TQ4 weights");
    const int rows = t.codes.shape(0), columns = t.codes.shape(1) * 2;
    auto codes = mx::contiguous(t.codes), scales = mx::contiguous(t.scales);
    mx::eval(codes, scales);
    if (scales.shape(1) != columns / 128) throw std::invalid_argument("nax tq4: expected one scale per 128 columns");
    const uint8_t* packed = codes.data<uint8_t>();
    std::vector<uint8_t> unpacked(size_t(rows) * columns);
    for (size_t i = 0; i < unpacked.size(); ++i) {
        const uint8_t byte = packed[i / 2];
        unpacked[i] = (i % 2 ? byte >> 4 : byte) & 15u;
    }
    const uint16_t* sbits = reinterpret_cast<const uint16_t*>(scales.data<mx::float16_t>());
    std::vector<uint16_t> s(sbits, sbits + scales.size()), b(s.size(), 0);
    auto out = nax_affine_load(nax_affine_pack(unpacked, s, b, rows, columns, 4, 128), rows, columns, 4, 128);
    out.biases = mx::zeros({64}, mx::uint16);
    mx::eval(out.biases);
    out.tq4 = true;
    return out;
}

NaxAffine nax_from_q6(const Q6Weights& q) {
    if (q.blocks.ndim() != 3 || q.blocks.shape(2) != 210 || q.blocks.dtype() != mx::uint8)
        throw std::invalid_argument("nax q6: needs Q6_K blocks");
    const int rows = q.blocks.shape(0), columns = q.blocks.shape(1) * 256, gpr = columns / 16;
    auto blocks = mx::contiguous(q.blocks);
    mx::eval(blocks);
    const uint8_t* b = blocks.data<uint8_t>();
    std::vector<uint8_t> codes(size_t(rows) * columns);
    std::vector<float> scale(size_t(rows) * gpr);
    for (size_t r = 0; r < size_t(rows); ++r)
        for (int sb = 0; sb < columns / 256; ++sb) {
            const uint8_t* p = b + (r * (columns / 256) + sb) * 210;
            uint16_t dbits = uint16_t(p[208] | (p[209] << 8));
            const float d = float(mx::float16_t(*reinterpret_cast<const mx::float16_t*>(&dbits)));
            for (int w = 0; w < 256; ++w) {
                const int h = w / 128, l = w % 32, c = (w % 128) / 32;
                const uint8_t lo = p[h * 64 + (c & 1) * 32 + l], hi = p[128 + h * 32 + l];
                codes[r * columns + sb * 256 + w] = uint8_t((c < 2 ? lo & 15u : lo >> 4) | (((hi >> (2 * c)) & 3u) << 4));
            }
            for (int j = 0; j < 16; ++j) scale[r * gpr + sb * 16 + j] = d * float(int8_t(p[192 + j]));
        }
    std::vector<uint16_t> zero(size_t(rows) * gpr, 0);
    auto out = nax_affine_load(nax_affine_pack(codes, zero, zero, rows, columns, 6, 16), rows, columns, 6, 16);
    // scales in the kernel's [blocks, groups, 32] order, FP32 (two uint16 words each)
    std::vector<float> ordered(scale.size());
    for (int blk = 0; blk < rows / 32; ++blk)
        for (int g = 0; g < gpr; ++g)
            for (int j = 0; j < 32; ++j) ordered[(size_t(blk) * gpr + g) * 32 + j] = scale[size_t(blk * 32 + j) * gpr + g];
    out.scales = mx::array(reinterpret_cast<const uint16_t*>(ordered.data()), {rows / 32, gpr, 64}, mx::uint16);
    out.biases = mx::zeros({64}, mx::uint16);
    mx::eval(out.scales, out.biases);
    out.q6 = true;
    return out;
}

uint64_t affine8_embedding_bytes(uint64_t rows, uint64_t columns) {
    if (rows == 0 || columns % 64) throw std::invalid_argument("affine8 embedding needs columns % 64 == 0");
    return rows * columns + rows * (columns / 64) * 4;
}

Affine8Embedding affine8_embedding_load(std::span<const uint8_t> payload, int rows, int columns) {
    if (payload.size() != affine8_embedding_bytes(rows, columns))
        throw std::invalid_argument("affine8 embedding payload has the wrong size");
    const uint8_t* at = payload.data();
    auto codes = mx::array(at, {rows, columns}, mx::uint8);
    at += size_t(rows) * columns;
    const int g = columns / 64;
    auto scales = mx::array(reinterpret_cast<const mx::bfloat16_t*>(at), {rows, g}, mx::bfloat16);
    at += size_t(rows) * g * 2;
    auto biases = mx::array(reinterpret_cast<const mx::bfloat16_t*>(at), {rows, g}, mx::bfloat16);
    mx::eval(codes, scales, biases);
    return {codes, scales, biases, rows, columns};
}

mx::array affine8_embedding(const Affine8Embedding& e, std::span<const uint32_t> tokens) {
    const int m = int(tokens.size());
    for (auto t : tokens) if (t >= uint32_t(e.rows)) throw std::invalid_argument("token id out of range");
    return affine8_embedding(e, mx::array(tokens.data(), {m}, mx::uint32));
}

mx::array affine8_embedding(const Affine8Embedding& e, const mx::array& ids) {
    const int m = ids.shape(0);
    auto q = mx::reshape(mx::astype(mx::take(e.codes, ids, 0), mx::float32), {m, e.columns / 64, 64});
    auto s = mx::expand_dims(mx::astype(mx::take(e.scales, ids, 0), mx::float32), 2);
    auto b = mx::expand_dims(mx::astype(mx::take(e.biases, ids, 0), mx::float32), 2);
    return mx::reshape(mx::add(mx::multiply(q, s), b), {m, e.columns});
}
}

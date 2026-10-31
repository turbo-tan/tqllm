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
// c += a x b on the matrix unit, 16 x 32 x 16, with the operands and the accumulator in plain registers (the MLX
// steel NAX pattern): a = 16 rows x 16 k (rows fm, fm + 8; k fn .. fn + 3), b = the lane's decoded weight fragment
inline void nax_mma(thread float* c, thread const half* a, thread const half* b) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
    for (short i = 0; i < 8; i++) ca[i] = a[i];
    for (short i = 0; i < 16; i++) {cb[i] = b[i]; cc[i] = c[i];}
    op.run(ca, cb, cc);
    for (short i = 0; i < 16; i++) c[i] = cc[i];
}
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
    const device uint* hp = (const device uint*)H + (BITS == 6 ? ((size_t)blk * (K / 16) + split * (KS / 16)) * 32 + lane : 0);
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

// Verify / decode FFN gate and up in one launch with SwiGLU in the epilogue (16-row kernel, 4-bit affine or TQ4): each
// simdgroup decodes the same 32-row block of the gate and of the up matrix, both matmuls share the input fragment, and
// since both accumulators have the same lane layout every thread holds gate and up of the same (row, column), so it
// writes h = bf16(bf16(silu(bf16(gate))) * bf16(up)) directly: one launch instead of two matmuls and the elementwise ops.
// Split-K partials meet in threadgroup memory in split order (as the matmul kernel), so a row's result does not depend
// on the other rows: one row and a verify batch agree bit for bit.
const char* kSwiglu = R"METAL(
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int BPT = SPLIT >= 4 ? 1 : 4 / SPLIT, KS = K / SPLIT;
    const uint local = sg / SPLIT;
    const int split = int(sg % SPLIT);
    const uint blk = threadgroup_position_in_grid.x * BPT + local;
    threadgroup float part[SPLIT > 1 ? BPT * SPLIT * 32 * 32 : 1];
    const int rows = min(int(R[0]), 16);
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto cg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto cu = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ag = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cg), float>();
    auto au = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cu), float>();
    for (short i = 0; i < 16; i++) {ag[i] = 0.0f; au[i] = 0.0f;}
    const size_t wofs = ((size_t)blk * (K / 16) + split * (KS / 16)) * 32 + lane;
    const size_t sofs = ((size_t)blk * (K / GS) + split * (KS / GS)) * 32;
    const device uint2* wg = (const device uint2*)WG + wofs;
    const device uint2* wu = (const device uint2*)WU + wofs;
    const device half* xs = X + (size_t)fm * K + fn + split * KS;
    for (int g = 0; g < KS / GS; g++) {
        float4 sg0, sg1, su0, su1, bg0 = 0.0f, bg1 = 0.0f, bu0 = 0.0f, bu1 = 0.0f;
        if (TQ4) {
            sg0 = float4(as_type<half4>(*(const device ushort4*)(SG + sofs + g * 32 + fn)));
            sg1 = float4(as_type<half4>(*(const device ushort4*)(SG + sofs + g * 32 + 16 + fn)));
            su0 = float4(as_type<half4>(*(const device ushort4*)(SU + sofs + g * 32 + fn)));
            su1 = float4(as_type<half4>(*(const device ushort4*)(SU + sofs + g * 32 + 16 + fn)));
        } else {
            sg0 = bf16x4(*(const device ushort4*)(SG + sofs + g * 32 + fn));
            sg1 = bf16x4(*(const device ushort4*)(SG + sofs + g * 32 + 16 + fn));
            su0 = bf16x4(*(const device ushort4*)(SU + sofs + g * 32 + fn));
            su1 = bf16x4(*(const device ushort4*)(SU + sofs + g * 32 + 16 + fn));
            bg0 = bf16x4(*(const device ushort4*)(BG + sofs + g * 32 + fn));
            bg1 = bf16x4(*(const device ushort4*)(BG + sofs + g * 32 + 16 + fn));
            bu0 = bf16x4(*(const device ushort4*)(BU + sofs + g * 32 + fn));
            bu1 = bf16x4(*(const device ushort4*)(BU + sofs + g * 32 + 16 + fn));
        }
        for (short t = 0; t < GS / 16; t++) {
            const int ks = g * (GS / 16) + t;
            half4 a0 = fm < rows ? *(const device half4*)(xs + ks * 16) : half4(0.0h);
            half4 a1 = fm + 8 < rows ? *(const device half4*)(xs + 8 * K + ks * 16) : half4(0.0h);
            for (short i = 0; i < 4; i++) {ca[i] = a0[i]; ca[4 + i] = a1[i];}
            const uint2 w0 = wg[ks * 32], w1 = wu[ks * 32];
            if (TQ4) {
                for (short i = 0; i < 8; i++) {
                    cg[i] = half(float(TQ4_LEVEL[(w0.x >> (4 * i)) & 15u]) * sg0[i & 3]);
                    cg[8 + i] = half(float(TQ4_LEVEL[(w0.y >> (4 * i)) & 15u]) * sg1[i & 3]);
                    cu[i] = half(float(TQ4_LEVEL[(w1.x >> (4 * i)) & 15u]) * su0[i & 3]);
                    cu[8 + i] = half(float(TQ4_LEVEL[(w1.y >> (4 * i)) & 15u]) * su1[i & 3]);
                }
            } else {
                for (short p = 0; p < 4; p++) {
                    const uint l0 = (w0.x >> (4 * p)) & 0x000F000Fu, l1 = (w0.y >> (4 * p)) & 0x000F000Fu;
                    const uint m0 = (w1.x >> (4 * p)) & 0x000F000Fu, m1 = (w1.y >> (4 * p)) & 0x000F000Fu;
                    const float2 q0 = float2(as_type<half2>(l0 | 0x64006400u)) - 1024.0f;
                    const float2 q1 = float2(as_type<half2>(l1 | 0x64006400u)) - 1024.0f;
                    const float2 r0 = float2(as_type<half2>(m0 | 0x64006400u)) - 1024.0f;
                    const float2 r1 = float2(as_type<half2>(m1 | 0x64006400u)) - 1024.0f;
                    const half2 v0 = half2(fma(q0, float2(sg0[p]), float2(bg0[p])));
                    const half2 v1 = half2(fma(q1, float2(sg1[p]), float2(bg1[p])));
                    const half2 u0 = half2(fma(r0, float2(su0[p]), float2(bu0[p])));
                    const half2 u1 = half2(fma(r1, float2(su1[p]), float2(bu1[p])));
                    cg[p] = v0.x; cg[p + 4] = v0.y; cg[8 + p] = v1.x; cg[12 + p] = v1.y;
                    cu[p] = u0.x; cu[p + 4] = u0.y; cu[8 + p] = u1.x; cu[12 + p] = u1.y;
                }
            }
            op.run(ca, cg, ag);
            op.run(ca, cu, au);
        }
    }
    if (SPLIT > 1) {
        threadgroup float* mine = part + ((local * SPLIT + split) * 32 + lane) * 32;
        for (short i = 0; i < 16; i++) {mine[i] = ag[i]; mine[16 + i] = au[i];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split != 0) return;
        for (int p = 1; p < SPLIT; p++) {
            const threadgroup float* other = part + ((local * SPLIT + p) * 32 + lane) * 32;
            for (short i = 0; i < 16; i++) {ag[i] = ag[i] + other[i]; au[i] = au[i] + other[16 + i];}
        }
    }
    const uint col0 = blk * 32 + fn;
    for (short i = 0; i < 16; i++) {
        const short r = fm + ((i & 7) >> 2) * 8;
        if (r >= rows) continue;
        const uint c = col0 + (i >> 3) * 16 + (i & 3);
        const float gate = float(bfloat16_t(float(ag[i]))), up = float(bfloat16_t(float(au[i])));
        const float act = float(bfloat16_t(gate / (1.0f + metal::exp(-gate))));
        Y[(size_t)r * N + c] = half(float(bfloat16_t(act * up)));  // BF16 value as the FP16 matmul input of ffn_down
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

// Prefill GEMM on the stored codes (no FP16 weight copy), staged: per 64-input step all SGS x 32 threads dequantize the
// threadgroup's TN columns once into a double-buffered threadgroup stage [k][column], one barrier, then each simdgroup
// runs one RS x TN x 64 matmul2d of its RS input rows (read from device) on the stage while the threads load the next
// step's codes and scales. The tile design follows Splash's gguf_prefill_tile (Apache-2.0, see NOTICE). Epilogue EP:
// 0 plain, 1 silu(Aux) x acc (Aux = the gate output: the up projection fuses SwiGLU), 2 Aux + acc (residual).
// FP16 [m, N] output.
const char* kPrefill = R"METAL(
    // TB 0: stage [k][column], item = one lane's uint2 (8 rows x 2 inputs). TB 1: stage [column][k] (the right operand
    // transposed, K contiguous per column), item = 8 lanes' words of one row quad and input half (4 rows x 8 inputs).
    // PF 1: the next step's codes and scales load while the step's matmul runs.
    constexpr int KS = 64, KQ = KS / 16, THREADS = SGS * 32, STEPS = K / KS;
    constexpr int ITEMS = TB ? (TN / 32) * KQ * 16 : (TN / 32) * KQ * 32, IPT = (ITEMS + THREADS - 1) / THREADS;
    constexpr int E0 = TB ? KS : TN, E1 = TB ? TN : KS;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32 + thread_index_in_simdgroup;
    // RF 1: row tiles fastest in the grid (consecutive threadgroups share one column tile's weights, read from DRAM
    // about once and reused from cache; Splash's order), RF 0: column tiles fastest (weights re-read per row tile)
    const uint tile_r = RF ? threadgroup_position_in_grid.x : threadgroup_position_in_grid.y;
    const uint tile_c = RF ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x;
    const int row0 = int(tile_r) * (SGS * RS) + int(sg) * RS;
    // FULL: m is a multiple of SGS * RS, so every tile holds RS rows; the A tensor takes RS as a static extent and the
    // matmul and the store drop the per-row bounds check (the right operand's TN always divides N).
    const int rows = FULL ? RS : clamp(int(R[0]) - row0, 0, RS);
    const uint blk0 = tile_c * (TN / 32);
    threadgroup half stage[2 * KS * TN];
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(RS, TN, KS, false, TB != 0, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<1>> op;
    auto a = metal::tensor((device half*)X + (size_t)(rows > 0 ? row0 : 0) * K, metal::dextents<int, 2>{K, FULL ? RS : max(rows, 1)},
                           metal::array<int, 2>{1, K});
    metal::tensor<threadgroup half, metal::dextents<int, 2>, metal::tensor_inline> bt0(stage, metal::dextents<int, 2>{E0, E1},
                                                                                   metal::array<int, 2>{1, E0});
    metal::tensor<threadgroup half, metal::dextents<int, 2>, metal::tensor_inline> bt1(stage + KS * TN, metal::dextents<int, 2>{E0, E1},
                                                                                   metal::array<int, 2>{1, E0});
    auto b0 = bt0.template slice<E0, E1>(0, 0), b1 = bt1.template slice<E0, E1>(0, 0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(a.template slice<KS, RS>(0, 0)), decltype(b0), float>();
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    // TQ4 signed levels in a threadgroup table (Splash's codebook stage, kernels/common/gguf_staged.h): a table read
    // beats the shift/select chain when every step decodes K / 16 groups of TN columns.
    threadgroup half lvl[TQ4 ? 16 : 1];
    if (TQ4 && tid < 16) lvl[tid] = TQ4_LEVEL[tid];
    if (TQ4) threadgroup_barrier(mem_flags::mem_threadgroup);
    const device uint* W32 = (const device uint*)W;
    uint wq[IPT][TB ? 8 : 2];
    ushort4 sq0[IPT], sq1[IPT], bq0[IPT], bq1[IPT];
#define TQ_STAGE_LOAD(STEP)                                                                                       \
    for (short it = 0; it < IPT; it++) {                                                                          \
        const uint u = tid + it * THREADS;                                                                        \
        if (u >= uint(ITEMS)) break;                                                                              \
        if (TB) {                                                                                                 \
            const uint wsel = (u >> 1) & 1u, fn4 = (u >> 2) & 3u, bl = (u >> 4) / KQ;                             \
            const uint kstep = uint(STEP) * KQ + (u >> 4) % KQ;                                                   \
            const device uint* wp = W32 + ((size_t)(blk0 + bl) * (K / 16) + kstep) * 64 + wsel;                   \
            for (uint fm = 0; fm < 8u; fm++)                                                                      \
                wq[it][fm] = wp[(((fm & 4u) << 2) | ((fn4 & 2u) << 2) | ((fm & 3u) << 1) | (fn4 & 1u)) * 2];      \
            const size_t sb = ((size_t)(blk0 + bl) * (K / GS) + kstep * 16 / GS) * 32 + wsel * 16 + fn4 * 4;     \
            sq0[it] = *(const device ushort4*)(S + sb);                                                           \
            if (!TQ4) bq0[it] = *(const device ushort4*)(Bz + sb);                                                \
        } else {                                                                                                  \
            const uint l = u & 31u, bl = (u >> 5) / KQ, kstep = uint(STEP) * KQ + (u >> 5) % KQ;                  \
            const uint fn4 = ((l >> 2) & 2u) | (l & 1u);                                                          \
            const device uint* wp = W32 + (((size_t)(blk0 + bl) * (K / 16) + kstep) * 32 + l) * 2;                \
            wq[it][0] = wp[0]; wq[it][1] = wp[1];                                                                 \
            const size_t sb = ((size_t)(blk0 + bl) * (K / GS) + kstep * 16 / GS) * 32 + fn4 * 4;                 \
            sq0[it] = *(const device ushort4*)(S + sb); sq1[it] = *(const device ushort4*)(S + sb + 16);          \
            if (!TQ4) {bq0[it] = *(const device ushort4*)(Bz + sb); bq1[it] = *(const device ushort4*)(Bz + sb + 16);} \
        }                                                                                                         \
    }
    if (PF) {TQ_STAGE_LOAD(0)}
    for (int st = 0; st < STEPS; st++) {
        const int buf = st & 1;
        if (!PF) {TQ_STAGE_LOAD(st)}
        for (short it = 0; it < IPT; it++) {
            const uint u = tid + it * THREADS;
            if (u >= uint(ITEMS)) break;
            if (TB) {
                const uint kh = u & 1u, wsel = (u >> 1) & 1u, fn4 = (u >> 2) & 3u, kq = (u >> 4) % KQ, bl = (u >> 4) / KQ;
                const float4 s = TQ4 ? float4(as_type<half4>(sq0[it])) : bf16x4(sq0[it]);
                const float4 c = TQ4 ? float4(0.0f) : bf16x4(bq0[it]);
                threadgroup half* dst = stage + buf * (KS * TN) + (bl * 32 + wsel * 16 + fn4 * 4) * KS + kq * 16 + kh * 8;
                for (short p = 0; p < 4; p++) {
                    half v[8];
                    for (short fm = 0; fm < 8; fm++) {
                        const uint q = (wq[it][fm] >> (4u * (p + 4u * kh))) & 15u;
                        if (TQ4) {
                            v[fm] = half(float(lvl[q]) * s[p]);
                        } else {
                            v[fm] = half(fma(float(q), s[p], c[p]));
                        }
                    }
                    *(threadgroup half4*)(dst + p * KS) = half4(v[0], v[1], v[2], v[3]);
                    *(threadgroup half4*)(dst + p * KS + 4) = half4(v[4], v[5], v[6], v[7]);
                }
                continue;
            }
            const uint l = u & 31u, bl = (u >> 5) / KQ, kq = (u >> 5) % KQ;
            const uint fm = ((l >> 2) & 4u) | ((l >> 1) & 3u), fn4 = ((l >> 2) & 2u) | (l & 1u);
            const uint2 w = uint2(wq[it][0], wq[it][1]);
            half4 xl, xh, yl, yh;   // rows fn4 * 4 + p (x) and + 16 (y), inputs fm (l) and fm + 8 (h)
            if (TQ4) {
                const float4 s0 = float4(as_type<half4>(sq0[it])), s1 = float4(as_type<half4>(sq1[it]));
                for (short p = 0; p < 4; p++) {
                    const uint q[4] = {(w.x >> (4 * p)) & 15u, (w.x >> (4 * p + 16)) & 15u, (w.y >> (4 * p)) & 15u, (w.y >> (4 * p + 16)) & 15u};
                    xl[p] = half(float(lvl[q[0]]) * s0[p]); xh[p] = half(float(lvl[q[1]]) * s0[p]);
                    yl[p] = half(float(lvl[q[2]]) * s1[p]); yh[p] = half(float(lvl[q[3]]) * s1[p]);
                }
            } else {
                const float4 s0 = bf16x4(sq0[it]), s1 = bf16x4(sq1[it]), c0 = bf16x4(bq0[it]), c1 = bf16x4(bq1[it]);
                for (short p = 0; p < 4; p++) {
                    const uint l0 = (w.x >> (4 * p)) & 0x000F000Fu, l1 = (w.y >> (4 * p)) & 0x000F000Fu;
                    const float2 q0 = float2(as_type<half2>(l0 | 0x64006400u)) - 1024.0f;
                    const float2 q1 = float2(as_type<half2>(l1 | 0x64006400u)) - 1024.0f;
                    const half2 v0 = half2(fma(q0, float2(s0[p]), float2(c0[p])));
                    const half2 v1 = half2(fma(q1, float2(s1[p]), float2(c1[p])));
                    xl[p] = v0.x; xh[p] = v0.y; yl[p] = v1.x; yh[p] = v1.y;
                }
            }
            threadgroup half* dst = stage + buf * (KS * TN) + (kq * 16 + fm) * TN + bl * 32 + fn4 * 4;
            *(threadgroup half4*)dst = xl; *(threadgroup half4*)(dst + 16) = yl;
            *(threadgroup half4*)(dst + 8 * TN) = xh; *(threadgroup half4*)(dst + 8 * TN + 16) = yh;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && st + 1 < STEPS) {TQ_STAGE_LOAD(st + 1)}
        if (FULL || rows > 0) {
            auto as = a.template slice<KS, RS>(st * KS, 0);
            if (buf) op.run(as, b1, acc); else op.run(as, b0, acc);
        }
    }
#undef TQ_STAGE_LOAD
    if (!FULL && rows <= 0) return;
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const auto idx = acc.get_multidimensional_index(i);
        if (!FULL && int(idx[1]) >= rows) continue;
        const size_t o = (size_t)(row0 + int(idx[1])) * N + blk0 * 32 + uint(idx[0]);
        float v = acc[i];
        if (EP == 1) {const float g = float(Aux[o]); v *= g / (1.0f + metal::exp(-g));}
        else if (EP == 2) v += float(Aux[o]);
        Y[o] = half(v);
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

NaxAffine nax_affine_from_quantized(const mx::array& codes, const mx::array& scales, const mx::array& biases, int group) {
    const int rows = codes.ndim() == 2 ? codes.shape(0) : 0, columns = codes.ndim() == 2 ? codes.shape(1) * 8 : 0;
    if (codes.dtype() != mx::uint32 || scales.dtype() != mx::bfloat16 || biases.dtype() != mx::bfloat16 ||
        scales.shape() != mx::Shape{rows, columns / std::max(group, 1)} || biases.shape() != scales.shape())
        throw std::invalid_argument("nax affine from quantized: needs 4-bit codes and BF16 scales / biases");
    check_shape(rows, columns, 4, group);
    const int steps = columns / 16, gpr = columns / group;
    // lane -> (fn4, fm): the lane's right-operand rows fn4 * 4 + (i & 3) and columns fm + (i >> 2) * 8 of a 32 x 16 tile
    std::vector<uint32_t> perm(32);
    for (int lane = 0; lane < 32; ++lane) perm[lane] = uint32_t((lane_fn(lane) / 4) * 8 + lane_fm(lane));
    const auto lanes = mx::array(perm.data(), {32}, mx::uint32);
    const auto shifts8 = mx::array({0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u}, {8}, mx::uint32);
    // nibble i = ilo + 4 ihi of a word sits at bit 4 i
    const auto shifts = mx::reshape(mx::array({0u, 16u, 4u, 20u, 8u, 24u, 12u, 28u}, {8}, mx::uint32), {4, 1, 2, 1});
    std::vector<mx::array> parts;
    for (int r0 = 0; r0 < rows; r0 += 16384) {
        const int n = std::min(16384, rows - r0), g = n / 32;
        auto q = mx::slice(codes, {r0, 0}, {r0 + n, columns / 8});
        auto c = mx::reshape(mx::bitwise_and(mx::right_shift(mx::expand_dims(q, -1), shifts8), mx::array(15u)), {n, columns});
        // r = blk * 32 + w * 16 + fn4 * 4 + ilo, c = ks * 16 + ihi * 8 + fm
        auto t = mx::reshape(c, {g, 2, 4, 4, steps, 2, 8});
        t = mx::sum(mx::left_shift(t, shifts), {3, 5});                   // [g, w, fn4, ks, fm]
        t = mx::reshape(mx::transpose(t, {0, 3, 2, 4, 1}), {g, steps, 32, 2});  // [g, ks, fn4 * 8 + fm, w]
        parts.push_back(mx::take(t, lanes, 2));
        mx::eval(parts.back());
    }
    auto words = mx::contiguous(mx::concatenate(parts, 0));
    auto lay = [&](const mx::array& a) {
        return mx::contiguous(mx::transpose(mx::reshape(mx::view(a, mx::uint16), {rows / 32, 32, gpr}), {0, 2, 1}));
    };
    NaxAffine out{words, mx::zeros({64}, mx::uint32), lay(scales), lay(biases), rows, columns, 4, group};
    mx::eval(out.words, out.high, out.scales, out.biases);
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
    if (w.split > 0) return w.split;
    constexpr int cap = 8, target = 512;  // at most 8-way split-K, aiming for 512 simdgroups per matrix
    int split = 1;
    while (split < cap && (w.rows / 32) * split < target && (w.columns / w.group) % (split * 2) == 0) split *= 2;
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
    const int m = input.shape(0), mr = m <= 16 ? 16 : 32;
    const int split = nax_affine_splits(w);
    auto& kernel = mr == 16 ? kernel16 : kernel32;
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

mx::array nax_affine_swiglu(const NaxAffine& gate, const NaxAffine& up, const mlx::core::array& input) {
    if (input.ndim() != 2 || input.shape(1) != gate.columns || input.shape(0) < 1 || input.shape(0) > 16)
        throw std::invalid_argument("nax affine swiglu: input must be [m <= 16, columns]");
    if (gate.bits != 4 || up.bits != 4 || gate.q6 || up.q6 || gate.tq4 != up.tq4 || gate.rows != up.rows ||
        gate.columns != up.columns || gate.group != up.group)
        throw std::invalid_argument("nax affine swiglu: gate and up must be matching 4-bit matrices");
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_swiglu16", {"WG", "SG", "BG", "WU", "SU", "BU", "X", "R"}, {"Y"},
                                                kSwiglu, kHeader);
    const int m = input.shape(0), split = nax_affine_splits(gate);
    auto x = mx::contiguous(mx::astype(input, mx::float16));
    const uint32_t m32 = uint32_t(m);
    auto count = mx::array(&m32, {1}, mx::uint32);
    const int bpt = split >= 4 ? 1 : 4 / split, threads = 32 * bpt * split;
    return kernel({gate.words, gate.scales, gate.biases, up.words, up.scales, up.biases, x, count}, {{m, gate.rows}},
                  {mx::float16}, {gate.rows / 32 / bpt * threads, 1, 1}, {threads, 1, 1},
                  {{"K", gate.columns}, {"N", gate.rows}, {"GS", gate.group}, {"SPLIT", split}, {"TQ4", gate.tq4}},
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

mx::array nax_affine_prefill(const NaxAffine& w, const mx::array& input, int ep, const std::optional<mx::array>& aux, int rs,
                             int tn, int sgs, int tb, int pf, int rf) {
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax affine prefill: input must be [m, columns]");
    if (w.bits != 4 || w.q6 || w.group % 64 != 0 || w.columns % 64 != 0)
        throw std::invalid_argument("nax affine prefill: 4-bit affine or TQ4, group and columns multiples of 64");
    if ((rs != 16 && rs != 32 && rs != 64) || (tn != 32 && tn != 64 && tn != 128) || (sgs != 2 && sgs != 4 && sgs != 8) ||
        w.rows % tn != 0)
        throw std::invalid_argument("nax affine prefill: tile RS 16/32/64, TN 32/64/128 dividing rows, SGS 2/4/8");
    if ((tb != 0 && tb != 1) || (pf != 0 && pf != 1)) throw std::invalid_argument("nax affine prefill: tb and pf are 0 or 1");
    if (ep < 0 || ep > 2 || (ep != 0) != aux.has_value())
        throw std::invalid_argument("nax affine prefill: epilogue 0 plain, 1 SwiGLU, 2 residual (1 and 2 take aux)");
    const int m = input.shape(0);
    // rows are independent: a partial last row tile is padded with zero rows so every tile takes the unchecked path
    // (j911: 1023 rows 8.7 ms vs 1536 rows 9.8 ms with the bounds-checked tiles)
    const int tile = sgs * rs, mp = (m + tile - 1) / tile * tile;
    auto padded = [&](const mx::array& v) {
        auto h = v.dtype() == mx::float16 ? v : mx::astype(v, mx::float16);
        return mx::contiguous(mp == m ? h : mx::pad(h, {{0, mp - m}, {0, 0}}));
    };
    auto x = padded(input);
    auto a = x;
    if (aux) {
        if (aux->ndim() != 2 || aux->shape(0) != m || aux->shape(1) != w.rows)
            throw std::invalid_argument("nax affine prefill: aux must be [m, rows]");
        a = padded(*aux);
    }
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_affine_prefill", {"W", "S", "Bz", "X", "Aux", "R"}, {"Y"}, kPrefill, kHeader);
    const uint32_t m32 = uint32_t(mp);
    auto count = mx::array(&m32, {1}, mx::uint32);
    const int threads = 32 * sgs, tx = w.rows / tn, ty = mp / tile;
    auto y = kernel({w.words, w.scales, w.biases, x, a, count}, {{mp, w.rows}}, {mx::float16}, rf ? std::tuple<int, int, int>{ty * threads, tx, 1} : std::tuple<int, int, int>{tx * threads, ty, 1}, {threads, 1, 1},
                    {{"K", w.columns}, {"N", w.rows}, {"GS", w.group}, {"TQ4", w.tq4}, {"RS", rs}, {"TN", tn}, {"SGS", sgs}, {"EP", ep}, {"TB", tb}, {"PF", pf}, {"FULL", 1}, {"RF", rf}},
                    std::nullopt, false, mx::Device(mx::Device::gpu))[0];
    return mp == m ? y : mx::slice(y, {0, 0}, {m, w.rows});
}

MlxQuantized nax_affine_to_mlx(const NaxAffine& w) {
    if (w.bits != 4 || w.tq4 || w.q6 || (w.group != 32 && w.group != 64 && w.group != 128))
        throw std::invalid_argument("nax affine to mlx: 4-bit affine, group 32 / 64 / 128 only");
    // one thread per MLX word (row r, columns 8j .. 8j + 7); column c of row r sits in NAX lane (fm = c % 8, fn4 = (r % 16) / 4)
    // of word w = (r % 32) / 16, nibble (r % 4) + 4 ((c % 16) / 8): the inverse of nax_affine_pack's order
    static auto kernel = mx::fast::metal_kernel("tqllm_nax_affine_to_mlx", {"W", "S", "Bz"}, {"Q", "S16", "B16"}, R"METAL(
    const uint t = thread_position_in_grid.x;
    const uint words = uint(K) / 8u;
    if (t >= uint(N) * words) return;
    const uint r = t / words, j = t % words;
    const uint blk = r / 32u, rr = r % 32u, wsel = rr / 16u, fn4 = (rr % 16u) / 4u, ilo = rr % 4u;
    const uint ks = j / 2u, ihi = j % 2u;
    const device uint* base = W + ((size_t)blk * uint(STEPS) + ks) * 64u + wsel;
    uint q = 0;
    for (uint fm = 0; fm < 8u; ++fm) {
        const uint lane = ((fm & 4u) << 2) | ((fn4 & 2u) << 2) | ((fm & 3u) << 1) | (fn4 & 1u);
        q |= ((base[lane * 2u] >> (4u * (ilo + 4u * ihi))) & 15u) << (4u * fm);
    }
    Q[t] = q;
    const uint per = uint(GS) / 8u;
    if (j % per == 0u) {
        const uint g = j / per;
        const size_t from = ((size_t)blk * uint(GPR) + g) * 32u + rr, to = (size_t)r * uint(GPR) + g;
        S16[to] = half(as_type<float>(uint(S[from]) << 16));
        B16[to] = half(as_type<float>(uint(Bz[from]) << 16));
    }
)METAL");
    const int gpr = w.columns / w.group, total = w.rows * (w.columns / 8);
    auto out = kernel({w.words, w.scales, w.biases}, {{w.rows, w.columns / 8}, {w.rows, gpr}, {w.rows, gpr}},
                      {mx::uint32, mx::float16, mx::float16}, {total, 1, 1}, {256, 1, 1},
                      {{"K", w.columns}, {"N", w.rows}, {"STEPS", w.columns / 16}, {"GPR", gpr}, {"GS", w.group}},
                      std::nullopt, false, mx::Device(mx::Device::gpu));
    return {out[0], out[1], out[2], w.group};
}

mx::array nax_affine_qmm(const MlxQuantized& q, const mx::array& input) {
    auto x = input.dtype() == mx::float16 ? input : mx::astype(input, mx::float16);
    return mx::quantized_matmul(x, q.codes, q.scales, q.biases, true, q.group, 4, "affine");
}

mx::array nax_affine_gemm(const NaxAffine& w, const mx::array& input, bool round_bf16) {
    if (input.ndim() != 2 || input.shape(1) != w.columns || input.shape(0) < 1)
        throw std::invalid_argument("nax affine gemm: input must be [m, columns]");
    return nax_affine_gemm_dequantized(nax_affine_dequant(w), input, round_bf16);
}

mx::array nax_affine_gemm_dequantized(const mx::array& weights, const mx::array& input, bool round_bf16) {
    if (input.ndim() != 2 || weights.ndim() != 2 || input.shape(1) != weights.shape(1) || input.shape(0) < 1)
        throw std::invalid_argument("nax affine gemm: input must be [m, columns]");
    auto x = input.dtype() == mx::float16 ? input : mx::astype(input, mx::float16);
    auto y = mx::matmul(x, mx::transpose(weights));
    // FP16 product -> BF16 rounding -> FP32 as one fused elementwise pass
    static auto to_bf = mx::compile([](const std::vector<mx::array>& a) {
        return std::vector<mx::array>{mx::astype(mx::astype(mx::astype(a[0], mx::float32), mx::bfloat16), mx::float32)};
    }, true);
    return round_bf16 ? to_bf({y})[0] : mx::astype(y, mx::float32);
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

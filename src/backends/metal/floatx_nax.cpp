#include "mlx_floatx.hpp"
#include <stdexcept>
#include <string>

// NVFP4 routed experts for a few rows on the M5 matrix unit (speculative verify blocks): FloatxExperts::pack_nax()
// stores each expert's codes in the order of the 16 x 32 x 16 matrix op's weight fragment (one uint2 per lane and
// 16-wide K step) and its scales by column; rows() runs one simdgroup per 32-output block of each run of rows that
// share an expert, so an expert's weights are read once for every row routed to it. Decoded weights (E2M1 x E4M3)
// and BF16 activations are exact in FP16; only the accumulation order differs from gather_qmm.
namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
std::string e4m3_table() {
    std::string out = "constant half E4M3[256] = {";
    for (int b = 0; b < 256; ++b) {
        const int ex = (b >> 3) & 15, man = b & 7;
        double v = ex ? (1.0 + man / 8.0) * double(1 << ex) / 128.0 : man / 8.0 / 64.0;
        if (v > 448.0) v = 448.0;  // 0x7f / 0xff are NaN in E4M3; never valid scales
        out += std::string(b ? ", " : "") + (b & 128 ? "-" : "") + std::to_string(v) + "h";
    }
    return out + "};\n";
}
const std::string& header() {
    static const std::string h = std::string(R"METAL(
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
constant half E2M1[16] = {0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h, -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h};
)METAL") + e4m3_table();
    return h;
}
const char* kSource = R"METAL(
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint r0 = threadgroup_position_in_grid.z;
const uint e = ids[r0];
if (r0 > 0 && ids[r0 - 1] == e) return;  // the run's first row computes the whole run (<= 16 rows)
uint count = 1;
while (count < 16u && r0 + count < uint(R) && ids[r0 + count] == e) ++count;
const uint blk = threadgroup_position_in_grid.y * SG + sg;
if (blk * 32u >= uint(O)) return;
const short qid = lane >> 2;
const short fm = (qid & 4) | ((lane >> 1) & 3);
const short fn = ((qid & 2) | (lane & 1)) * 4;
constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, false, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
auto cb = op.template get_right_input_cooperative_tensor<half, half, float>();
auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
for (short i = 0; i < 16; i++) cc[i] = 0.0f;
// A: rows fm and fm + 8 of the run, inputs fn .. fn + 3 of each 16-wide step
const bool has0 = uint(fm) < count, has1 = uint(fm) + 8u < count;
const device bfloat16_t* xa = x + size_t(has0 ? xrow[r0 + fm] : 0u) * K + fn;
const device bfloat16_t* xb = x + size_t(has1 ? xrow[r0 + fm + 8] : 0u) * K + fn;
// B: the lane's packed fragment (codes [E, O/32, K/16, 32] uint2) and the block's scales ([E, O/32, K/16, 32] bytes)
const device uint2* wp = (const device uint2*)codes + (size_t(e) * (O / 32) + blk) * (K / 16) * 32 + lane;
const device uchar* sp = scales + (size_t(e) * (O / 32) + blk) * (K / 16) * 32;
for (int ks = 0; ks < K / 16; ks++) {
    const uint2 w = wp[ks * 32];
    const uchar4 s0 = *(const device uchar4*)(sp + ks * 32 + fn), s1 = *(const device uchar4*)(sp + ks * 32 + 16 + fn);
    const half4 a0 = has0 ? half4(float4(*(const device vec<bfloat16_t, 4>*)(xa + ks * 16))) : half4(0.0h);
    const half4 a1 = has1 ? half4(float4(*(const device vec<bfloat16_t, 4>*)(xb + ks * 16))) : half4(0.0h);
    for (short i = 0; i < 4; i++) { ca[i] = a0[i]; ca[4 + i] = a1[i]; }
    half h0[4], h1[4];
    for (short i = 0; i < 4; i++) { h0[i] = E4M3[s0[i]]; h1[i] = E4M3[s1[i]]; }
    for (short i = 0; i < 8; i++) {
        cb[i] = E2M1[(w.x >> (4 * i)) & 15u] * h0[i & 3];
        cb[8 + i] = E2M1[(w.y >> (4 * i)) & 15u] * h1[i & 3];
    }
    op.run(ca, cb, cc);
}
const float g = global[e] * (1.0f / 2688.0f);  // the expert's global scale over E4M3 max x E2M1 max
for (short i = 0; i < 16; i++) {
    const uint r = uint(fm + ((i & 7) >> 2) * 8);
    const uint col = blk * 32u + uint(fn + (i >> 3) * 16 + (i & 3));
    if (r < count) y[size_t(r0 + r) * O + col] = bfloat16_t(float(cc[i]) * g);
}
)METAL";
// lane of the fragment element at (fm, fn / 4): MLX steel NAX layout, fm = (qid & 4) | ((lane >> 1) & 3),
// fn / 4 = (qid & 2) | (lane & 1), qid = lane >> 2
int lane_fm(int lane) { return ((lane >> 2) & 4) | ((lane >> 1) & 3); }
int lane_fq(int lane) { return ((lane >> 2) & 2) | (lane & 1); }
}  // namespace

void FloatxExperts::pack_nax() {
    if (bits != 4) throw std::invalid_argument("matrix-unit experts are NVFP4 only");
    const int E = codes.shape(0), O = codes.shape(1), K = codes.shape(2) * 8;
    if (O % 32 || K % 16) throw std::invalid_argument("matrix-unit experts need O % 32 == 0, K % 16 == 0");
    const int NB = O / 32, KS = K / 16;
    std::vector<int32_t> perm(32);
    for (int l = 0; l < 32; ++l) perm[size_t(l)] = lane_fm(l) * 4 + lane_fq(l);
    const auto lanes = mx::array(perm.data(), {32}, mx::int32);
    std::vector<uint32_t> shift(8);
    for (int i = 0; i < 8; ++i) shift[size_t(i)] = uint32_t(4 * i);
    const auto shifts = mx::array(shift.data(), {8}, mx::uint32);
    const auto bytes = mx::view(codes, mx::uint8);  // [E, O, K / 2]
    std::vector<mx::array> packed;
    for (int e0 = 0; e0 < E; e0 += 16) {  // 16 experts at a time bounds the unpacked temporary
        const int e1 = std::min(E, e0 + 16), n = e1 - e0;
        auto b = mx::slice(bytes, {e0, 0, 0}, {e1, O, K / 2});
        auto nib = mx::reshape(mx::stack({mx::bitwise_and(b, mx::array(uint8_t(15))), mx::right_shift(b, mx::array(uint8_t(4)))}, -1),
                               {n, NB, 2, 4, 4, KS, 2, 8});  // o = blk*32 + h*16 + fq*4 + c, k = ks*16 + kh*8 + fm
        nib = mx::transpose(nib, {0, 1, 5, 7, 3, 2, 6, 4});  // [n, NB, KS, fm, fq, h, kh, c]
        nib = mx::take(mx::reshape(nib, {n, NB, KS, 32, 2, 8}), lanes, 3);
        auto words = mx::sum(mx::left_shift(mx::astype(nib, mx::uint32), shifts), -1);  // nibble i = kh * 4 + c
        packed.push_back(mx::astype(words, mx::uint32));
        mx::eval(packed.back());
    }
    auto pc = mx::concatenate(packed, 0);  // [E, NB, KS, 32, 2]
    auto ps = mx::transpose(mx::reshape(scales, {E, NB, 32, KS}), {0, 1, 3, 2});
    ps = mx::contiguous(ps);
    mx::eval({pc, ps});
    nax_codes = pc; nax_scales = ps;
}

mx::array FloatxExperts::rows(const mx::array& x, const mx::array& xrow, const mx::array& ids) const {
    if (!nax_codes || !global) throw std::logic_error("FloatxExperts::rows needs pack_nax()");
    static auto kernel = mx::fast::metal_kernel("tqllm_nvfp4_expert_rows_nax", {"x", "xrow", "ids", "codes", "scales", "global"}, {"y"},
                                                kSource, header());
    constexpr int SG = 4;
    const int E = codes.shape(0), O = codes.shape(1), K = codes.shape(2) * 8, R = ids.shape(0), blocks = O / 32;
    (void)E;
    if (x.ndim() != 2 || x.shape(1) != K || x.dtype() != mx::bfloat16) throw std::invalid_argument("FloatxExperts::rows input");
    return kernel({x, mx::astype(xrow, mx::uint32), mx::astype(ids, mx::uint32), *nax_codes, *nax_scales, *global}, {{R, O}}, {mx::bfloat16},
                  {32 * SG, (blocks + SG - 1) / SG, R}, {32 * SG, 1, 1}, {{"K", K}, {"O", O}, {"R", R}, {"SG", SG}},
                  std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

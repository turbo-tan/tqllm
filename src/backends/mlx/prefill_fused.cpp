#include "prefill_fused.hpp"
#include <mlx/compile.h>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
namespace {
mx::array bf_raw(const mx::array& a) {return mx::astype(mx::astype(a, mx::bfloat16), mx::float32);}
}

std::pair<mx::array, mx::array> rms_norm_bf16_pair_reference(const mx::array& x, const mx::array& w, float eps) {
    static auto round = mx::compile([](const std::vector<mx::array>& a) {return std::vector<mx::array>{bf_raw(a[0])};}, true);
    auto n = round({mx::fast::rms_norm(x, w, eps)})[0];
    return {n, mx::astype(n, mx::float16)};
}

std::pair<mx::array, mx::array> rms_norm_bf16_pair(const mx::array& x, const mx::array& w, float eps) {
    if (x.ndim() != 2 || x.dtype() != mx::float32 || w.ndim() != 1 || w.shape(0) != x.shape(1) || w.dtype() != mx::float32 ||
        x.shape(1) <= 4096)
        throw std::invalid_argument("rms_norm_bf16_pair: FP32 [rows, d > 4096] and FP32 weight [d]");
    // MLX rms_looped (kernels/rms_norm.metal) with T = float at 1024 threads, then the BF16 rounding and the FP16 copy
    static auto kernel = mx::fast::metal_kernel("tqllm_rms_norm_bf16_pair", {"x", "w", "eps"}, {"out", "out16"}, R"METAL(
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x, lsize = threads_per_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup, simd_group_id = simdgroup_index_in_threadgroup;
    const uint axis_size = uint(D);
    const device float* xr = x + gid * size_t(axis_size) + lid * N_READS;
    const device float* wr = w + lid * N_READS;
    float acc = 0;
    for (uint r = 0; r < axis_size; r += lsize * N_READS) {
        if (r + lid * N_READS + N_READS <= axis_size) {
            for (int i = 0; i < N_READS; i++) {float xi = xr[i + r]; acc += xi * xi;}
        } else {
            for (int i = 0; i < N_READS; i++)
                if ((r + lid * N_READS + i) < axis_size) {float xi = xr[i + r]; acc += xi * xi;}
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t o0 = gid * size_t(axis_size) + lid * N_READS;
    for (uint r = 0; r < axis_size; r += lsize * N_READS)
        for (int i = 0; i < N_READS; i++)
            if ((r + lid * N_READS + i) < axis_size) {
                const float v = float(bfloat16_t(wr[i + r] * static_cast<float>(xr[r + i] * local_inv_mean[0])));
                out[o0 + r + i] = v;
                out16[o0 + r + i] = half(v);
            }
)METAL");
    const int rows = x.shape(0), d = x.shape(1);
    auto o = kernel({mx::contiguous(x), mx::contiguous(w), mx::array(eps)}, {x.shape(), x.shape()}, {mx::float32, mx::float16},
                    {rows * 1024, 1, 1}, {1024, 1, 1}, {{"D", d}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return {o[0], o[1]};
}

std::tuple<mx::array, mx::array, mx::array> rms_norm_bf16_pair_residual_reference(const mx::array& x, const mx::array& r, const mx::array& w, float eps) {
    static auto add = mx::compile([](const std::vector<mx::array>& a) {return std::vector<mx::array>{bf_raw(mx::add(a[0], a[1]))};}, true);
    auto s = add({x, r})[0];
    auto n = rms_norm_bf16_pair_reference(s, w, eps);
    return {s, n.first, n.second};
}

std::tuple<mx::array, mx::array, mx::array> rms_norm_bf16_pair_residual(const mx::array& x, const mx::array& r, const mx::array& w, float eps) {
    if (x.ndim() != 2 || x.dtype() != mx::float32 || r.shape() != x.shape() || r.dtype() != mx::float32 || w.ndim() != 1 ||
        w.shape(0) != x.shape(1) || w.dtype() != mx::float32 || x.shape(1) <= 4096)
        throw std::invalid_argument("rms_norm_bf16_pair_residual: FP32 [rows, d > 4096] x and r, FP32 weight [d]");
    // s = bf(x + r) computed twice per element (identical bits), the reduction and the output as rms_norm_bf16_pair
    static auto kernel = mx::fast::metal_kernel("tqllm_rms_norm_bf16_pair_res", {"x", "r", "w", "eps"}, {"sum", "out", "out16"}, R"METAL(
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x, lsize = threads_per_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup, simd_group_id = simdgroup_index_in_threadgroup;
    const uint axis_size = uint(D);
    const size_t o0 = gid * size_t(axis_size) + lid * N_READS;
    const device float* xr = x + o0;
    const device float* rr = r + o0;
    const device float* wr = w + lid * N_READS;
    float acc = 0;
    for (uint r2 = 0; r2 < axis_size; r2 += lsize * N_READS) {
        for (int i = 0; i < N_READS; i++)
            if ((r2 + lid * N_READS + i) < axis_size) {
                const float si = float(bfloat16_t(xr[r2 + i] + rr[r2 + i]));
                sum[o0 + r2 + i] = si;
                acc += si * si;
            }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r2 = 0; r2 < axis_size; r2 += lsize * N_READS)
        for (int i = 0; i < N_READS; i++)
            if ((r2 + lid * N_READS + i) < axis_size) {
                const float si = float(bfloat16_t(xr[r2 + i] + rr[r2 + i]));
                const float v = float(bfloat16_t(wr[i + r2] * static_cast<float>(si * local_inv_mean[0])));
                out[o0 + r2 + i] = v;
                out16[o0 + r2 + i] = half(v);
            }
)METAL");
    const int rows = x.shape(0), d = x.shape(1);
    auto o = kernel({mx::contiguous(x), mx::contiguous(r), mx::contiguous(w), mx::array(eps)}, {x.shape(), x.shape(), x.shape()},
                    {mx::float32, mx::float32, mx::float16}, {rows * 1024, 1, 1}, {1024, 1, 1}, {{"D", d}}, std::nullopt, false,
                    mx::Device(mx::Device::gpu));
    return {o[0], o[1], o[2]};
}

mx::array swiglu_pack16_reference(const mx::array& gate_up) {
    const int rows = gate_up.shape(0), f = gate_up.shape(1) / 2;
    static auto swiglu16 = mx::compile([](const std::vector<mx::array>& a) {
        auto gate = bf_raw(mx::astype(a[0], mx::float32)), up = bf_raw(mx::astype(a[1], mx::float32));
        return std::vector<mx::array>{mx::astype(bf_raw(mx::multiply(bf_raw(mx::multiply(gate, mx::sigmoid(gate))), up)), mx::float16)};
    }, true);
    return swiglu16({mx::slice(gate_up, {0, 0}, {rows, f}), mx::slice(gate_up, {0, f}, {rows, 2 * f})})[0];
}

mx::array swiglu_pack16(const mx::array& gate_up) {
    if (gate_up.ndim() != 2 || gate_up.dtype() != mx::float16 || gate_up.shape(1) % 8)
        throw std::invalid_argument("swiglu_pack16: FP16 [rows, 2F], F a multiple of 4");
    const int rows = gate_up.shape(0), f = gate_up.shape(1) / 2;
    // the compiled chain of swiglu_pack16_reference; sigmoid as MLX's Sigmoid (kernels/unary_ops.h)
    static auto kernel = mx::fast::metal_kernel("tqllm_swiglu_pack16", {"gu"}, {"out"}, R"METAL(
    const uint t = thread_position_in_grid.x;
    const uint row = t / uint(F / 4), c = (t % uint(F / 4)) * 4u;
    if (row >= uint(ROWS)) return;
    const half4 a = *(const device half4*)(gu + (size_t)row * (2 * F) + c);
    const half4 b = *(const device half4*)(gu + (size_t)row * (2 * F) + F + c);
    half4 h;
    for (short i = 0; i < 4; i++) {
        const float gate = float(bfloat16_t(float(a[i]))), up = float(bfloat16_t(float(b[i])));
        const float y = 1 / (1 + metal::precise::exp(metal::abs(gate)));
        const float s = (gate < 0) ? y : 1 - y;
        h[i] = half(float(bfloat16_t(float(bfloat16_t(gate * s)) * up)));
    }
    *(device half4*)(out + (size_t)row * F + c) = h;
)METAL");
    const int threads = rows * (f / 4);
    return kernel({mx::contiguous(gate_up)}, {{rows, f}}, {mx::float16}, {(threads + 255) / 256 * 256, 1, 1}, {256, 1, 1},
                  {{"F", f}, {"ROWS", rows}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

namespace {
// round_product: the FP16 product rounded to BF16 values in FP32 (nax_affine_gemm(..., true))
mx::array round_product16(const mx::array& p) {
    static auto fused = mx::compile([](const std::vector<mx::array>& a) {
        return std::vector<mx::array>{bf_raw(mx::astype(a[0], mx::float32))};
    }, true);
    return fused({p})[0];
}
// norm: bf(rms_norm(x, w, eps)), the attn_q_norm / attn_k_norm path
mx::array norm_bf(const mx::array& x, const mx::array& w, float eps) {
    static auto round = mx::compile([](const std::vector<mx::array>& a) {return std::vector<mx::array>{bf_raw(a[0])};}, true);
    return round({mx::fast::rms_norm(x, w, eps)})[0];
}
mx::array bf(const mx::array& a) {
    static auto fused = mx::compile([](const std::vector<mx::array>& a) {return std::vector<mx::array>{bf_raw(a[0])};}, true);
    return fused({a})[0];
}
}  // namespace

AttnQkNormRope attn_qk_norm_rope_reference(const mx::array& q_prod, const mx::array& k_prod, const mx::array& v_prod,
                                           const mx::array& q_norm_w, const mx::array& k_norm_w, int heads, int kvheads,
                                           int hd, int rot, float theta, float eps, int offset) {
    const int m = q_prod.shape(0);
    auto qg = mx::reshape(round_product16(q_prod), {m, heads, 2 * hd});
    auto q = mx::slice(qg, {0, 0, 0}, {m, heads, hd}), gate = mx::slice(qg, {0, 0, hd}, {m, heads, 2 * hd});
    q = mx::transpose(mx::reshape(norm_bf(q, q_norm_w, eps), {1, m, heads, hd}), {0, 2, 1, 3});
    auto k = mx::reshape(round_product16(k_prod), {1, m, kvheads, hd});
    k = mx::transpose(norm_bf(k, k_norm_w, eps), {0, 2, 1, 3});
    auto v = mx::transpose(mx::reshape(round_product16(v_prod), {1, m, kvheads, hd}), {0, 2, 1, 3});
    q = bf(mx::fast::rope(q, rot, false, theta, 1.0f, mx::array(offset)));
    k = bf(mx::fast::rope(k, rot, false, theta, 1.0f, mx::array(offset)));
    return {mx::contiguous(q), mx::contiguous(k), mx::contiguous(v), mx::contiguous(gate)};
}

AttnQkNormRope attn_qk_norm_rope(const mx::array& q_prod, const mx::array& k_prod, const mx::array& v_prod,
                                 const mx::array& q_norm_w, const mx::array& k_norm_w, int heads, int kvheads, int hd,
                                 int rot, float theta, float eps, int offset) {
    const int m = q_prod.shape(0);
    if (q_prod.ndim() != 2 || q_prod.dtype() != mx::float16 || q_prod.shape(1) != heads * 2 * hd ||
        k_prod.ndim() != 2 || k_prod.dtype() != mx::float16 || k_prod.shape() != mx::Shape{m, kvheads * hd} ||
        v_prod.ndim() != 2 || v_prod.dtype() != mx::float16 || v_prod.shape() != mx::Shape{m, kvheads * hd} ||
        q_norm_w.shape() != mx::Shape{hd} || q_norm_w.dtype() != mx::float32 ||
        k_norm_w.shape() != mx::Shape{hd} || k_norm_w.dtype() != mx::float32 ||
        hd % 4 || rot <= 0 || rot % 2 || rot > hd)
        throw std::invalid_argument("attn_qk_norm_rope: FP16 products [m, heads*2*hd] / [m, kvheads*hd], FP32 weight [hd]");
    const float base = std::log2(theta);
    // MLX rms_single_row (hd <= 4096): one row per 64-thread threadgroup (2 simdgroups), N_READS 4,
    // metal::precise::rsqrt; then bf(norm), the non-traditional partial-rotary RoPE of rope.metal (forward true,
    // metal::exp2/fast::cos/sin), and the final BF16 rounding. gate is round_product only (no norm, no rope).
    static auto qk_kernel = mx::fast::metal_kernel("tqllm_attn_qk_norm_rope", {"qprod", "kprod", "qw", "kw", "offset", "eps", "base"},
                                                   {"q", "k", "gate"}, R"METAL(
    constexpr int N_READS = 4;
    const uint HD = uint(HD_), ROT = uint(ROT_), HALF = ROT / 2u;
    const uint nqrows = uint(M) * uint(HEADS), row = threadgroup_position_in_grid.x;
    const bool is_k = row >= nqrows;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup, simd_group_id = simdgroup_index_in_threadgroup;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    // locate the input row: q|gate pack for q rows, k pack for k rows
    uint tok, head;
    const device half* src;
    const device float* wr;
    if (!is_k) {
        tok = row / uint(HEADS); head = row % uint(HEADS);
        src = qprod + (size_t)tok * (uint(HEADS) * 2u * HD) + head * (2u * HD);  // q half of the head
        wr = qw;
    } else {
        const uint r = row - nqrows;
        tok = r / uint(KVHEADS); head = r % uint(KVHEADS);
        src = kprod + (size_t)tok * (uint(KVHEADS) * HD) + head * HD;
        wr = kw;
    }
    // round_product then RMS over HD (single_row order): 64 threads, 4 contiguous reads each
    float thread_x[N_READS];
    float acc = 0;
    for (int i = 0; i < N_READS; i++) {
        const uint c = lid * N_READS + i;
        float xi = float(bfloat16_t(float(src[c])));
        thread_x[i] = xi;
        acc += xi * xi;
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) local_inv_mean[0] = metal::precise::rsqrt(acc / float(HD) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_mean = local_inv_mean[0];
    // bf(norm): BF16 rounding of w * (x * inv_mean), kept per thread for the rope pairing below
    float bn[N_READS];
    for (int i = 0; i < N_READS; i++)
        bn[i] = float(bfloat16_t(wr[lid * N_READS + i] * (thread_x[i] * inv_mean)));
    // write to a threadgroup copy so this thread can read its pair partner (j and j + HALF) for RoPE
    threadgroup float normed[HD_MAX];
    for (int i = 0; i < N_READS; i++) normed[lid * N_READS + i] = bn[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int64_t off = offset;
    // destination: q is [1, HEADS, M, HD], k is [1, KVHEADS, M, HD], both contiguous
    device float* dst = (!is_k ? q : k) + (size_t)head * (uint(M) * HD) + (size_t)tok * HD;
    for (int i = 0; i < N_READS; i++) {
        const uint j = lid * N_READS + i;
        float outv;
        if (j < HALF) {
            const float d = float(j) / float(HALF);
            const float inv_freq = metal::exp2(-d * base);
            const float L = float(off + int64_t(tok));
            const float th = L * inv_freq;
            const float ct = metal::fast::cos(th), st = metal::fast::sin(th);
            const float x1 = normed[j], x2 = normed[j + HALF];
            outv = x1 * ct - x2 * st;          // forward == true
        } else if (j < ROT) {
            const uint jp = j - HALF;
            const float d = float(jp) / float(HALF);
            const float inv_freq = metal::exp2(-d * base);
            const float L = float(off + int64_t(tok));
            const float th = L * inv_freq;
            const float ct = metal::fast::cos(th), st = metal::fast::sin(th);
            const float x1 = normed[jp], x2 = normed[j];
            outv = x1 * st + x2 * ct;          // forward == true
        } else {
            outv = normed[j];                  // dims [ROT, HD) copied through RoPE
        }
        dst[j] = float(bfloat16_t(outv));      // final bf()
    }
    // gate (q rows only): round_product of the gate half, no norm or rope, laid out [M, HEADS, HD]
    if (!is_k) {
        const device half* gsrc = qprod + (size_t)tok * (uint(HEADS) * 2u * HD) + head * (2u * HD) + HD;
        device float* gdst = gate + ((size_t)tok * uint(HEADS) + head) * HD;
        for (int i = 0; i < N_READS; i++) {
            const uint c = lid * N_READS + i;
            gdst[c] = float(bfloat16_t(float(gsrc[c])));
        }
    }
)METAL");
    const int nqrows = m * heads, nkrows = m * kvheads;
    const int tg = ((hd + 4 - 1) / 4 + 31) / 32 * 32;  // ceil(hd/N_READS) rounded up to a simd multiple (hd=256 -> 64)
    auto out = qk_kernel({mx::contiguous(q_prod), mx::contiguous(k_prod), mx::contiguous(q_norm_w),
                          mx::contiguous(k_norm_w), mx::array(offset), mx::array(eps), mx::array(base)},
                         {{1, heads, m, hd}, {1, kvheads, m, hd}, {m, heads, hd}},
                         {mx::float32, mx::float32, mx::float32},
                         {(nqrows + nkrows) * tg, 1, 1}, {tg, 1, 1},
                         {{"M", m}, {"HEADS", heads}, {"KVHEADS", kvheads}, {"HD_", hd}, {"ROT_", rot}, {"HD_MAX", hd}},
                         std::nullopt, false, mx::Device(mx::Device::gpu));
    // v: round_product + the same reshape / transpose as the chain (a view-only op, bit identical)
    auto v = mx::transpose(mx::reshape(round_product16(v_prod), {1, m, kvheads, hd}), {0, 2, 1, 3});
    return {out[0], out[1], mx::contiguous(v), out[2]};
}
}  // namespace tqllm::mlx_backend

// DFlash2 far-context retrieval (dflash_retrieve.hpp): (1) planted far chunks are selected per kv head, in ascending
// order, and a higher-scoring chunk inside the window (not wholly below F) is not; (2) retrieved keys equal the raw keys
// rotated directly to F - R + i, values copied; (3) when retrieval picks the chunks right before the window,
// [retrieved | window] equals the full cache from F - R bit for bit and attention over it equals windowed attention;
// (4) a selection issued ahead on a second GPU stream equals the inline one; (5) per-thread side streams.
#include "dflash_retrieve.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

namespace mx = mlx::core;
using tqllm::mlx_backend::dflash_retrieve;
namespace {
int failures = 0;
void check(bool ok, const char* what, double v = 0) {
    std::printf("%-78s %.6f %s\n", what, v, ok ? "PASS" : "FAIL");
    failures += !ok;
}
constexpr int KVH = 8, HEADS = 32, HD = 128, GROUP = HEADS / KVH;
constexpr float THETA = 1e7f;
// NeoX halves, float: x rotated to position p
void rope_cpu(float* x, double p) {
    for (int d = 0; d < HD / 2; ++d) {
        const double ang = p / std::pow(double(THETA), double(2 * d) / HD);
        const double c = std::cos(ang), s = std::sin(ang), a = x[d], b = x[d + HD / 2];
        x[d] = float(a * c - b * s);
        x[d + HD / 2] = float(b * c + a * s);
    }
}
std::vector<float> to_host(const mx::array& a) {
    auto f = mx::astype(a, mx::float32);
    mx::eval(f);
    return std::vector<float>(f.data<float>(), f.data<float>() + f.size());
}
}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int cap = 4096, pos = 3008, nq = 16, recent = 1024, F = pos - recent;
    // raw keys k0 / values, keys stored rotated at their absolute positions (as the drafter's cache holds them)
    std::vector<float> k0(size_t(KVH) * cap * HD), v0(k0.size()), q0(size_t(HEADS) * nq * HD);
    for (auto& x : k0) x = 0.1f * nd(rng);
    for (auto& x : v0) x = nd(rng);
    for (auto& x : q0) x = nd(rng);
    // (1) plant: kv head h scores highest on far chunks {3 + h, 40 + 2h, 100 - h}: raw keys there = 4 x the group-mean
    // anchor query rotated back from its position (so the rotated dot product is large)
    const std::vector<std::vector<int>> planted = [] {
        std::vector<std::vector<int>> p(KVH);
        for (int h = 0; h < KVH; ++h) p[h] = {3 + h, 40 + 2 * h, 100 - h};
        return p;
    }();
    std::vector<float> qrot(q0);  // queries rotated at pos + row
    for (int hq = 0; hq < HEADS; ++hq) for (int t = 0; t < nq; ++t) rope_cpu(&qrot[(size_t(hq) * nq + t) * HD], pos + t);
    for (int h = 0; h < KVH; ++h)
        for (int c : planted[h]) for (int t = 0; t < 16; ++t) {
            const int key = c * 16 + t;
            float m[HD] = {};
            for (int g = 0; g < GROUP; ++g) for (int d = 0; d < HD; ++d) m[d] += qrot[(size_t(h * GROUP + g) * nq) * HD + d] / GROUP;
            rope_cpu(m, -double(key));  // raw = rotated anchor direction at the key's own position
            for (int d = 0; d < HD; ++d) k0[(size_t(h) * cap + key) * HD + d] = 4.0f * m[d];
        }
    // chunks starting at F and inside the window score higher still: never selected
    for (int h = 0; h < KVH; ++h) for (int key : {F, F + 40}) for (int t = 0; t < 16; ++t) {
        float m[HD] = {};
        for (int g = 0; g < GROUP; ++g) for (int d = 0; d < HD; ++d) m[d] += qrot[(size_t(h * GROUP + g) * nq) * HD + d] / GROUP;
        rope_cpu(m, -double(key + t));
        for (int d = 0; d < HD; ++d) k0[(size_t(h) * cap + key + t) * HD + d] = 16.0f * m[d];
    }
    std::vector<float> krot(k0);
    for (int h = 0; h < KVH; ++h) for (int key = 0; key < cap; ++key) rope_cpu(&krot[(size_t(h) * cap + key) * HD], key);
    auto kfull = mx::astype(mx::array(krot.data(), {1, KVH, cap, HD}), mx::bfloat16);
    auto vfull = mx::astype(mx::array(v0.data(), {1, KVH, cap, HD}), mx::bfloat16);
    auto q = mx::astype(mx::array(qrot.data(), {1, HEADS, nq, HD}), mx::bfloat16);
    {
        auto r = dflash_retrieve(q, kfull, vfull, mx::array(F, mx::int32), cap / 16, 3, THETA);
        check(r.retrieved == 48, "R = 3 chunks x 16", r.retrieved);
        bool ok = r.retrieved == 48;
        if (ok) {
            auto sel = mx::astype(r.chunks, mx::int32);
            mx::eval(sel);
            for (int h = 0; h < KVH; ++h) for (int j = 0; j < 3; ++j) ok &= sel.data<int32_t>()[h * 3 + j] == planted[h][j];
        }
        check(ok, "(1) planted far chunks selected per kv head, ascending; chunks at / past F never");
        // (2) retrieved rows = raw keys rotated straight to F - R + i, values copied
        const int R = r.retrieved, rows = R;
        check(r.k.shape() == mx::Shape{1, KVH, R, HD} && r.v.shape() == r.k.shape(), "rows [1, kvh, R, hd]", rows);
        const auto ks = to_host(r.k), vs = to_host(r.v), kf = to_host(kfull), vf = to_host(vfull);
        auto sel = mx::astype(r.chunks, mx::int32);
        mx::eval(sel);
        double kerr = 0, kmax = 0, verr = 0;
        for (int h = 0; h < KVH; ++h) for (int i = 0; i < rows; ++i) {
            const int src = sel.data<int32_t>()[h * 3 + i / 16] * 16 + i % 16;
            float want[HD];
            for (int d = 0; d < HD; ++d) want[d] = k0[(size_t(h) * cap + src) * HD + d];
            rope_cpu(want, F - R + i);
            for (int d = 0; d < HD; ++d) {
                const size_t at = (size_t(h) * rows + i) * HD + d, from = (size_t(h) * cap + src) * HD + d;
                kerr = std::max(kerr, double(std::fabs(ks[at] - want[d]))); kmax = std::max(kmax, double(std::fabs(want[d])));
                verr = std::max(verr, double(std::fabs(vs[at] - vf[from])));
            }
        }
        check(kerr <= 2e-2 * kmax, "(2) retrieved keys = raw keys rotated to F - R + i (rel to max |k|)", kerr / kmax);
        check(verr == 0.0, "(2) values gathered bit for bit", verr);
    }
    {
        // (3) plant the chunks right before the recent part: the scratch is the full cache's window [F - R, pos + nq)
        std::vector<float> k3(k0);
        for (int h = 0; h < KVH; ++h) for (int c : {F / 16 - 2, F / 16 - 1}) for (int t = 0; t < 16; ++t) {
            const int key = c * 16 + t;
            float m[HD] = {};
            for (int g = 0; g < GROUP; ++g) for (int d = 0; d < HD; ++d) m[d] += qrot[(size_t(h * GROUP + g) * nq) * HD + d] / GROUP;
            rope_cpu(m, -double(key));
            for (int d = 0; d < HD; ++d) k3[(size_t(h) * cap + key) * HD + d] = 8.0f * m[d];
        }
        for (int h = 0; h < KVH; ++h) for (int key = 0; key < cap; ++key) rope_cpu(&k3[(size_t(h) * cap + key) * HD], key);
        auto kf3 = mx::astype(mx::array(k3.data(), {1, KVH, cap, HD}), mx::bfloat16);
        // pos 3008, recent 1024: F = 1984 = 124 chunks, so the two chunks below F end exactly at F
        auto r = dflash_retrieve(q, kf3, vfull, mx::array(F, mx::int32), cap / 16, 2, THETA);
        const int R = r.retrieved, lo = F - R, rows = R + recent + nq;
        auto sk = mx::concatenate({r.k, mx::slice(kf3, {0, 0, F, 0}, {1, KVH, pos + nq, HD})}, 2);
        auto sv = mx::concatenate({r.v, mx::slice(vfull, {0, 0, F, 0}, {1, KVH, pos + nq, HD})}, 2);
        auto window_k = mx::slice(kf3, {0, 0, lo, 0}, {1, KVH, pos + nq, HD}), window_v = mx::slice(vfull, {0, 0, lo, 0}, {1, KVH, pos + nq, HD});
        const double dk = mx::max(mx::abs(mx::subtract(mx::astype(sk, mx::float32), mx::astype(window_k, mx::float32)))).item<float>();
        check(R == 32 && dk == 0.0, "(3) adjacent chunks: [retrieved | window] = the full cache from F - R bit for bit", dk);
        const float scale = 1.0f / std::sqrt(float(HD));
        auto a = mx::fast::scaled_dot_product_attention(q, sk, sv, scale);
        auto b = mx::fast::scaled_dot_product_attention(q, window_k, window_v, scale);
        const double da = mx::max(mx::abs(mx::subtract(mx::astype(a, mx::float32), mx::astype(b, mx::float32)))).item<float>();
        check(rows == pos + nq - lo && da == 0.0, "(3) attention over the scratch = windowed attention", da);
    }
    {
        // (4) look-ahead: the selection issued on a second GPU stream (overlapping other work) equals the inline one, and the
        // gather on the default stream consumes it (MLX orders the streams by the data dependency)
        using tqllm::mlx_backend::dflash_retrieve_gather;
        using tqllm::mlx_backend::dflash_retrieve_select;
        auto side = mx::new_stream(mx::Device::gpu);
        mx::array sel_side = [&] {
            mx::StreamContext ctx(side);
            return dflash_retrieve_select(q, kfull, mx::array(F, mx::int32), cap / 16, 3);
        }();
        auto busy = mx::matmul(mx::ones({2048, 2048}), mx::ones({2048, 2048}));   // default-stream work to overlap
        auto g = dflash_retrieve_gather(sel_side, kfull, vfull, mx::array(F, mx::int32), THETA);
        auto inline_r = dflash_retrieve(q, kfull, vfull, mx::array(F, mx::int32), cap / 16, 3, THETA);
        mx::eval({g.k, g.v, inline_r.k, inline_r.v, busy});
        const double ds = mx::max(mx::abs(mx::subtract(sel_side, inline_r.chunks))).item<int32_t>();
        const double dk = mx::max(mx::abs(mx::subtract(mx::astype(g.k, mx::float32), mx::astype(inline_r.k, mx::float32)))).item<float>();
        check(ds == 0 && dk == 0.0, "(4) side-stream selection + gather = inline retrieval bit for bit", dk);
    }
    {
        // (5) server requests run on other threads: a stream made by one thread is unknown to another (the j1096 failure),
        // retrieval_stream() gives each thread its own and the selection there equals the inline one
        using tqllm::mlx_backend::dflash_retrieve_select;
        using tqllm::mlx_backend::retrieval_stream;
        auto main_stream = retrieval_stream();
        bool foreign_fails = false, own_ok = false;
        std::thread([&] {
            try {mx::StreamContext ctx(main_stream); auto x = dflash_retrieve_select(q, kfull, mx::array(F, mx::int32), cap / 16, 3); mx::eval(x);}
            catch (const std::exception&) {foreign_fails = true;}
            mx::array sel = [&] {mx::StreamContext ctx(retrieval_stream()); return dflash_retrieve_select(q, kfull, mx::array(F, mx::int32), cap / 16, 3);}();
            auto ref = dflash_retrieve_select(q, kfull, mx::array(F, mx::int32), cap / 16, 3);
            own_ok = mx::max(mx::abs(mx::subtract(sel, ref))).item<int32_t>() == 0;
        }).join();
        check(foreign_fails, "(5) another thread's stream is refused (MLX streams are per thread)");
        check(own_ok, "(5) retrieval_stream() per thread: selection equals the inline one");
    }
    {
        // (6) selection from chunk key sums: the planted chunks (all 16 keys aligned with the anchor) are found, chunks at /
        // past F never
        using tqllm::mlx_backend::dflash_retrieve_select_sums;
        auto ksum = mx::sum(mx::reshape(mx::astype(kfull, mx::float32), {1, KVH, cap / 16, 16, HD}), 3);
        auto sel = mx::astype(dflash_retrieve_select_sums(q, ksum, mx::array(F, mx::int32), cap / 16, 3), mx::int32);
        mx::eval(sel);
        bool ok = true;
        for (int h = 0; h < KVH; ++h) for (int j = 0; j < 3; ++j) ok &= sel.data<int32_t>()[h * 3 + j] == planted[h][j];
        check(ok, "(6) chunk-sum selection finds the planted chunks, never at / past F");
    }
    {
        // (7) int8 far caches: the gather dequantizes codes x per-row scale to BF16 rows and then equals the BF16 gather of
        // those rows bit for bit; the quantization itself stays within one step (max |x| / 127) of the BF16 value
        using tqllm::mlx_backend::dflash_far_dequantize;
        using tqllm::mlx_backend::dflash_far_quantize;
        using tqllm::mlx_backend::dflash_retrieve_gather;
        using tqllm::mlx_backend::dflash_retrieve_select;
        const auto qk = dflash_far_quantize(kfull), qv = dflash_far_quantize(vfull);
        auto k16 = mx::astype(dflash_far_dequantize(qk.codes, qk.scales), mx::bfloat16);
        auto v16 = mx::astype(dflash_far_dequantize(qv.codes, qv.scales), mx::bfloat16);
        const double kq = mx::max(mx::divide(mx::abs(mx::subtract(mx::astype(k16, mx::float32), mx::astype(kfull, mx::float32))),
                                             mx::max(mx::abs(mx::astype(kfull, mx::float32)), -1, true))).item<float>();
        check(qk.codes.dtype() == mx::int8 && qk.scales.shape() == mx::Shape{1, KVH, cap} && kq <= 1.0 / 127 + 1e-3,
              "(7) int8 codes [1, kvh, cap, hd] + FP16 scales [1, kvh, cap]; error <= max |k| / 127", kq);
        auto sel = dflash_retrieve_select(q, k16, mx::array(F, mx::int32), cap / 16, 3);
        auto a = dflash_retrieve_gather(sel, qk.codes, qv.codes, mx::array(F, mx::int32), THETA, qk.scales, qv.scales);
        auto b = dflash_retrieve_gather(sel, k16, v16, mx::array(F, mx::int32), THETA);
        const double dk = mx::max(mx::abs(mx::subtract(mx::astype(a.k, mx::float32), mx::astype(b.k, mx::float32)))).item<float>();
        const double dv = mx::max(mx::abs(mx::subtract(mx::astype(a.v, mx::float32), mx::astype(b.v, mx::float32)))).item<float>();
        check(a.k.dtype() == mx::bfloat16 && dk == 0.0 && dv == 0.0, "(7) int8 gather = BF16 gather of the dequantized rows bit for bit", dk + dv);
    }
    std::printf(failures ? "DFLASH_RETRIEVE_FAIL\n" : "DFLASH_RETRIEVE_PASS\n");
    return failures ? 1 : 0;
}

// Trellis KV on Metal (kv_tcq.hpp): (1) the GPU encoder's records decode (host reference) to the same error as the host
// encoder's, and most records are bit-identical; (2) attention over the codes equals SDPA over the host-decoded K / V
// (original basis) within FP16-operand error, causal per row; (3) the n-row call equals n single-row calls bit for bit;
// (4) timing vs the BF16 decode-rows path (v11) at 8K / 32K / 64K.
#include "attention_decode.hpp"
#include "kv_tcq.hpp"
#include <mlx/mlx.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
namespace {
int failures = 0;
void check(bool ok, const char* what, double v) {std::printf("%-74s %.6f %s\n", what, v, ok ? "PASS" : "FAIL"); failures += !ok;}
std::vector<float> rows_with_outliers(int n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f); std::vector<float> v(size_t(n) * 256);
    for (size_t i = 0; i < v.size(); ++i) v[i] = nd(rng) * (i % 37 == 0 ? 6.f : 1.f);
    return v;
}
// host decode of codes [kvh, cap, words] -> [1, kvh, cap, 256] FP32 original basis
mx::array host_decode(const mx::array& codes, int kb) {
    auto c = mx::contiguous(codes); mx::eval(c);
    const int kvh = c.shape(0), cap = c.shape(1), w = c.shape(2);
    std::vector<float> out(size_t(kvh) * cap * 256);
    const uint32_t* p = c.data<uint32_t>();
    for (int h = 0; h < kvh; ++h) for (int k = 0; k < cap; ++k) tcq::decode_row(p + (size_t(h) * cap + k) * w, kb, &out[(size_t(h) * cap + k) * 256], true);
    return mx::array(out.data(), {1, kvh, cap, 256}, mx::float32);
}
}  // namespace

int main() {
    std::mt19937 rng(5);
    {
        const int n = 64, kvh = 4;
        for (int kb : {3, 2}) {
            auto host = rows_with_outliers(n * kvh, rng);
            auto gpu = tcq_encode(mx::array(host.data(), {n, kvh, 256}, mx::float32), kb); mx::eval(gpu);
            const int w = 2 * tcq::words_per_group(kb);
            double eg = 0, eh = 0, ref = 0; int same = 0;
            for (int r = 0; r < n * kvh; ++r) {
                std::vector<uint32_t> rec(w); tcq::encode_row(&host[size_t(r) * 256], kb, rec.data());
                float dg[256], dh[256];
                tcq::decode_row(gpu.data<uint32_t>() + size_t(r) * w, kb, dg, true); tcq::decode_row(rec.data(), kb, dh, true);
                bool eq = true; for (int i = 0; i < w; ++i) eq &= rec[i] == gpu.data<uint32_t>()[size_t(r) * w + i]; same += eq;
                for (int i = 0; i < 256; ++i) {const double x = host[size_t(r) * 256 + i]; eg += (dg[i] - x) * (dg[i] - x); eh += (dh[i] - x) * (dh[i] - x); ref += x * x;}
            }
            check(eg <= eh * 1.02 + 1e-12, kb == 3 ? "(1) K: GPU encoder error <= host encoder error x 1.02" : "(1) V: GPU encoder error <= host encoder error x 1.02", eg / ref);
            check(same >= n * kvh * 9 / 10, kb == 3 ? "(1) K: >= 90 % of GPU records bit-identical to the host's" : "(1) V: >= 90 % of GPU records bit-identical to the host's", double(same) / (n * kvh));
        }
    }
    const int heads = 24, kvh = 4, dim = 256;
    const float scale = 1.0f / 16.0f;
    for (int length : {64, 1025, 4096, 32768}) {
        const int cap = (length + 1023) / 1024 * 1024 + 1024;
        auto kraw = mx::random::normal({cap, kvh, dim}, mx::float32), vraw = mx::random::normal({cap, kvh, dim}, mx::float32);
        auto kcodes = mx::transpose(tcq_encode(kraw, 3), {1, 0, 2}), vcodes = mx::transpose(tcq_encode(vraw, 2), {1, 0, 2});
        kcodes = mx::contiguous(kcodes); vcodes = mx::contiguous(vcodes); mx::eval(kcodes, vcodes);
        auto kd = host_decode(kcodes, 3), vd = host_decode(vcodes, 2);
        for (int nq : {1, 8, 16}) {
            auto q = mx::random::normal({1, heads, nq, dim}, mx::float32);
            auto got = decode_rows_attention_tcq(q, kcodes, vcodes, scale, length); mx::eval(got);
            double err = 0, ref = 0; bool rows_exact = true;
            for (int t = 0; t < nq; ++t) {
                const int seen = length - nq + 1 + t;
                auto qt = mx::slice(q, {0, 0, t, 0}, {1, heads, t + 1, dim});
                auto want = mx::fast::scaled_dot_product_attention(qt, mx::slice(kd, {0, 0, 0, 0}, {1, kvh, seen, dim}), mx::slice(vd, {0, 0, 0, 0}, {1, kvh, seen, dim}), scale);
                auto mine = mx::slice(got, {0, 0, t, 0}, {1, heads, t + 1, dim});
                err += mx::sum(mx::square(mx::subtract(mine, want))).item<float>(); ref += mx::sum(mx::square(want)).item<float>();
                if (nq > 1) {
                    auto single = decode_rows_attention_tcq(qt, kcodes, vcodes, scale, seen);
                    rows_exact &= mx::array_equal(single, mine).item<bool>();
                }
            }
            char what[128];
            std::snprintf(what, sizeof what, "(2) length %d n %d: attention over codes = SDPA over decoded K / V (rel)", length, nq);
            check(std::sqrt(err / ref) < 2e-2, what, std::sqrt(err / ref));
            if (nq > 1) {std::snprintf(what, sizeof what, "(3) length %d n %d: n-row call = single-row calls bit for bit", length, nq); check(rows_exact, what, 0);}
        }
    }
    // (4) timing, 16 calls per eval, against the BF16 path the engine runs today
    for (int length : {8192, 32768, 65536}) {
        const int cap = length + 1024;
        auto kraw = mx::random::normal({cap, kvh, dim}, mx::float32), vraw = mx::random::normal({cap, kvh, dim}, mx::float32);
        auto kcodes = mx::contiguous(mx::transpose(tcq_encode(kraw, 3), {1, 0, 2})), vcodes = mx::contiguous(mx::transpose(tcq_encode(vraw, 2), {1, 0, 2}));
        auto kb = mx::contiguous(mx::astype(mx::reshape(mx::transpose(kraw, {1, 0, 2}), {1, kvh, cap, dim}), mx::bfloat16));
        auto vb = mx::contiguous(mx::astype(mx::reshape(mx::transpose(vraw, {1, 0, 2}), {1, kvh, cap, dim}), mx::bfloat16));
        mx::eval(kcodes, vcodes, kb, vb);
        for (int nq : {8, 16}) {
            auto q = mx::random::normal({1, heads, nq, dim}, mx::float32); mx::eval(q);
            auto bench = [&](auto fn) {
                for (int w = 0; w < 2; ++w) mx::eval(fn());
                const auto t0 = std::chrono::steady_clock::now(); std::vector<mx::array> o; for (int r = 0; r < 16; ++r) o.push_back(fn()); mx::eval(o);
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 16;
            };
            const double tt = bench([&] {return decode_rows_attention_tcq(q, kcodes, vcodes, scale, length);});
            if (nq == 8) {
                double td[4];
                for (int dg : {1, 2, 3, 4}) td[dg - 1] = bench([&] {return decode_rows_attention_tcq(q, kcodes, vcodes, scale, length, dg);});
                std::printf("KV_TCQ_DIAG length=%d n=8 full %.1f | no K decode %.1f | no V decode %.1f | no decode %.1f | no rotations %.1f us\n",
                            length, tt, td[0], td[1], td[2], td[3]);
            }
            const double tb = bench([&] {return decode_rows_attention(q, kb, vb, scale, length);});
            std::printf("KV_TCQ_BENCH length=%d n=%d tcq %.1f us  bf16 v11 %.1f us  (KV bytes %.0f vs %.0f MB per layer)\n", length, nq, tt, tb,
                        double(length) * kvh * 176 / 1e6, double(length) * kvh * 1024 / 1e6);
        }
    }
    std::printf(failures ? "KV_TCQ_FAIL\n" : "KV_TCQ_PASS\n");
    return failures ? 1 : 0;
}

// Host reference of the trellis KV codec (tcq_codec.hpp): rotation is orthonormal, K (3-bit, 512 states) / V (2-bit,
// 256 states) reconstruction error on Gaussian rows with outliers, dot products in the rotated basis equal those in the
// original basis, record sizes 104 / 72 B.
#include "tcq_codec.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace tcq = tqllm::mlx_backend::tcq;
int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what, double v) {std::printf("%-72s %.5f %s\n", what, v, ok ? "PASS" : "FAIL"); failures += !ok;};
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    {
        float x[128], y[128]; for (int i = 0; i < 128; ++i) x[i] = y[i] = nd(rng);
        tcq::rotate_group(y); tcq::unrotate_group(y); double e = 0; for (int i = 0; i < 128; ++i) e = std::max(e, double(std::fabs(x[i] - y[i])));
        check(e < 1e-5, "rotate then unrotate = identity (max abs)", e);
    }
    check(tcq::words_per_group(3) * 2 * 4 == 104 && tcq::words_per_group(2) * 2 * 4 == 72, "record bytes K 104 / V 72", 176);
    for (int kb : {3, 2}) {
        double err = 0, ref = 0, dot_err = 0, dot_ref = 0;
        for (int r = 0; r < 200; ++r) {
            float row[256], q[256], dec[256], decr[256], qr[256];
            for (int i = 0; i < 256; ++i) {row[i] = nd(rng) * (i % 37 == 0 ? 6.f : 1.f); q[i] = qr[i] = nd(rng);}  // outlier columns
            std::vector<uint32_t> rec(2 * tcq::words_per_group(kb));
            tcq::encode_row(row, kb, rec.data()); tcq::decode_row(rec.data(), kb, dec, true); tcq::decode_row(rec.data(), kb, decr, false);
            tcq::rotate_group(qr); tcq::rotate_group(qr + 128);
            double d0 = 0, d1 = 0;
            for (int i = 0; i < 256; ++i) {err += double(dec[i] - row[i]) * (dec[i] - row[i]); ref += double(row[i]) * row[i]; d0 += double(q[i]) * dec[i]; d1 += double(qr[i]) * decr[i];}
            dot_err = std::max(dot_err, std::fabs(d0 - d1)); dot_ref = std::max(dot_ref, std::fabs(d0));
        }
        const double rel = err / ref;
        if (kb == 3) check(rel < 0.06, "K 3-bit trellis: relative squared error", rel);
        else check(rel < 0.16, "V 2-bit trellis: relative squared error", rel);
        check(dot_err <= 1e-4 * dot_ref, kb == 3 ? "K: <Rq, decoded-rotated k> = <q, decoded k>" : "V: rotated decode is an orthonormal image", dot_err / dot_ref);
    }
    std::printf(failures ? "TCQ_CODEC_FAIL\n" : "TCQ_CODEC_PASS\n");
    return failures ? 1 : 0;
}

#pragma once
#include "dequant.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <future>
#include <thread>
#include <numeric>

namespace tqllm {
inline int gsq_q2_code(float ratio) {
    // Exactly clamp(round(ratio), -1, 2), including ties away from zero.
    return ratio <= -0.5f ? -1 : ratio < 0.5f ? 0 : ratio < 1.5f ? 1 : 2;
}
// Pilot encoder for the existing plain GSQ_Q2_0 layout: fp16 scale + 64 two-bit codes.
// Requantizing shipped weights is lossy; this is not a BF16-source quality claim.
inline std::vector<char> encode_gsq_q2(const float * x, size_t n) {
    if (!x || !n || n % 64) throw std::invalid_argument("cold Q2: invalid size");
    std::vector<char> out(n / 64 * 18, 0);
    for (size_t b = 0; b < n / 64; ++b) {
        const float * v = x + b * 64; float peak = 0;
        for (int i = 0; i < 64; ++i) {
            if (!std::isfinite(v[i])) throw std::invalid_argument("cold Q2: nonfinite input");
            peak = std::max(peak, std::abs(v[i]));
        }
        double best_error = INFINITY; float best_scale = 0;
        for (int sign : {-1, 1}) {
            float s = float(sign) * peak / 2;
            for (int it = 0; it < 5 && s != 0; ++it) {
                double xy = 0, yy = 0;
                for (int i = 0; i < 64; ++i) {
                    const int q = gsq_q2_code(v[i] / s); xy += double(v[i]) * q; yy += q * q;
                }
                if (yy) s = float(xy / yy);
            }
            s = half_to_float(float_to_half(s));
            if (!std::isfinite(s)) continue;
            double error = 0;
            for (int i = 0; i < 64; ++i) {
                const int q = s ? gsq_q2_code(v[i] / s) : 0;
                const double d = double(v[i]) - double(s) * q; error += d * d;
            }
            if (error < best_error) { best_error = error; best_scale = s; }
        }
        if (!std::isfinite(best_error)) throw std::invalid_argument("cold Q2: scale overflow");
        char * dst = out.data() + b * 18; const uint16_t hs = float_to_half(best_scale); std::memcpy(dst, &hs, 2);
        for (int i = 0; i < 64; ++i) {
            const int q = best_scale ? gsq_q2_code(v[i] / best_scale) : 0;
            dst[2 + i / 4] = char(uint8_t(dst[2 + i / 4]) | uint8_t((q + 1) << (2 * (i % 4))));
        }
    }
    return out;
}
inline std::vector<char> encode_gsq_q2_parallel(const float * x, size_t n, unsigned workers) {
    if (!x || !n || n % 64 || !workers) throw std::invalid_argument("cold Q2: invalid parallel input");
    workers = std::min<unsigned>(workers, unsigned(std::min<size_t>(n / 64, 32)));
    if (workers == 1) return encode_gsq_q2(x, n);
    std::vector<char> out(n / 64 * 18);
    std::vector<std::future<void>> pending;
    for (unsigned worker = 0; worker < workers; ++worker) {
        const size_t b0 = (n / 64) * worker / workers, b1 = (n / 64) * (worker + 1) / workers;
        pending.push_back(std::async(std::launch::async, [&, b0, b1] {
            const auto chunk = encode_gsq_q2(x + b0 * 64, (b1 - b0) * 64);
            std::memcpy(out.data() + b0 * 18, chunk.data(), chunk.size());
        }));
    }
    for (auto & job : pending) job.get();
    return out;
}
inline std::vector<char> requantize_gsq_q2_parallel(uint32_t type, const char * src, size_t n,
                                                   size_t source_block, size_t source_bytes, unsigned workers) {
    if (!src || !n || !source_block || !source_bytes || !workers || n % 64 || n % source_block)
        throw std::invalid_argument("cold Q2: invalid source shape");
    const size_t unit = std::lcm(source_block, size_t(64));
    const size_t groups = n / unit;
    workers = std::min<unsigned>(workers,unsigned(std::min<size_t>(groups,32)));
    if (!workers) throw std::invalid_argument("cold Q2: no complete source groups");
    std::vector<char> out(n / 64 * 18);
    std::vector<std::future<void>> pending;
    for (unsigned worker=0;worker<workers;++worker) {
        const size_t i0=groups*worker/workers*unit, i1=groups*(worker+1)/workers*unit;
        pending.push_back(std::async(std::launch::async,[&,i0,i1] {
            std::vector<float> x(i1-i0);
            dequantize_host(type,src+i0/source_block*source_bytes,x.data(),int64_t(x.size()));
            const auto q=encode_gsq_q2(x.data(),x.size());
            std::memcpy(out.data()+i0/64*18,q.data(),q.size());
        }));
    }
    for(auto & job:pending)job.get();
    return out;
}
} // namespace tqllm

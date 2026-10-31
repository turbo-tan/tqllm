#pragma once
// KL(reference || candidate) per row of two logit blocks [rows, vocab] (FP32 logits, softmax taken here in double),
// accumulated over a stream of blocks: mean / max KL, top-1 agreement, and the mean log-probability gap of a target
// token when one is given. The KL gate of a candidate file against a reference dump of the same tokens.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace tqllm {
struct KlRows {
    double kl_sum = 0, kl_max = 0, nll_gap_sum = 0, nll_ref_sum = 0, nll_cand_sum = 0;
    size_t rows = 0, top1_same = 0, targets = 0;
    // terms of reference tokens with log p <= min_log_p are left out of the KL sum (llama.cpp / tqkld use -16)
    double min_log_p = -INFINITY;
    // targets_in: next-token ids per row (nullptr: no log-probability gap)
    void add(const float* reference, const float* candidate, size_t count, size_t vocab, const uint32_t* targets_in = nullptr) {
        if (vocab == 0) throw std::invalid_argument("KL rows need a vocabulary");
        for (size_t r = 0; r < count; ++r) {
            const float* a = reference + r * vocab;
            const float* b = candidate + r * vocab;
            const size_t ia = size_t(std::max_element(a, a + vocab) - a), ib = size_t(std::max_element(b, b + vocab) - b);
            const double ma = a[ia], mb = b[ib];
            double sa = 0, sb = 0;
            for (size_t j = 0; j < vocab; ++j) {sa += std::exp(double(a[j]) - ma); sb += std::exp(double(b[j]) - mb);}
            const double za = ma + std::log(sa), zb = mb + std::log(sb);
            double kl = 0;
            for (size_t j = 0; j < vocab; ++j) {
                const double la = double(a[j]) - za;
                if (la > min_log_p) kl += std::exp(la) * (la - (double(b[j]) - zb));
            }
            if (!std::isfinite(kl)) throw std::runtime_error("invalid KL (non-finite logits)");
            if (!std::isfinite(min_log_p)) {
                if (kl < -1e-9) throw std::runtime_error("invalid KL (negative)");
                kl = std::max(kl, 0.0);
            }
            kl_sum += kl; kl_max = std::max(kl_max, kl); top1_same += ia == ib; ++rows;
            if (targets_in) {
                const uint32_t t = targets_in[r];
                if (t >= vocab) throw std::invalid_argument("target token outside the vocabulary");
                nll_gap_sum += (zb - double(b[t])) - (za - double(a[t])); nll_ref_sum += za - double(a[t]); nll_cand_sum += zb - double(b[t]); ++targets;
            }
        }
    }
    double mean() const {return rows ? kl_sum / double(rows) : 0.0;}
};
}  // namespace tqllm

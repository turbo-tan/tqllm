// KlRows: identical logits give KL 0 and full top-1 agreement; a shift by a constant (same softmax) gives 0; a known
// two-token case matches the closed form; logits that differ in the argmax lower top-1; non-finite logits are refused.
#include "kl_rows.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {if (!ok) {std::printf("FAIL %s\n", what); ++failures;}};
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 0.5f, 0.0f, -1.0f, 4.0f, 1.0f};  // 2 rows x 4
    std::vector<float> shifted(a.size());
    for (size_t i = 0; i < a.size(); ++i) shifted[i] = a[i] + 7.0f;
    tqllm::KlRows same;
    same.add(a.data(), a.data(), 2, 4);
    same.add(a.data(), shifted.data(), 2, 4);
    check(same.rows == 4 && same.kl_max < 1e-12 && same.top1_same == 4, "identical / shifted logits: KL 0, top-1 4/4");
    // two tokens: p = softmax(0, 0) = (0.5, 0.5), q = softmax(0, ln 3) = (0.25, 0.75): KL = 0.5 ln 2 + 0.5 ln(2/3)
    std::vector<float> p = {0.0f, 0.0f}, q = {0.0f, float(std::log(3.0))};
    tqllm::KlRows two;
    const uint32_t target = 1;
    two.add(p.data(), q.data(), 1, 2, &target);
    const double expected = 0.5 * std::log(2.0) + 0.5 * std::log(2.0 / 3.0);
    check(std::abs(two.mean() - expected) < 1e-6, "two-token KL equals the closed form");
    check(std::abs(two.nll_gap_sum - (std::log(4.0 / 3.0) - std::log(2.0))) < 1e-6 && two.targets == 1, "target log-probability gap");
    std::vector<float> c = {3.0f, 2.0f, 1.0f, 0.5f};
    tqllm::KlRows flip;
    flip.add(a.data(), c.data(), 1, 4);
    check(flip.top1_same == 0 && flip.mean() > 0, "argmax change lowers top-1");
    // a cutoff drops the reference terms at or below it: tokens with p = e^-20 contribute nothing at -16
    std::vector<float> r3 = {0.0f, -20.0f, -20.0f}, c3 = {0.0f, 0.0f, -20.0f};
    tqllm::KlRows full, cut;
    cut.min_log_p = -16;
    full.add(r3.data(), c3.data(), 1, 3);
    cut.add(r3.data(), c3.data(), 1, 3);
    const double z = std::log(1.0 + 2.0 * std::exp(-20.0)), zc = std::log(2.0 + std::exp(-20.0));
    check(std::abs(cut.mean() - std::exp(-z) * ((0.0 - z) - (0.0 - zc))) < 1e-9, "cutoff keeps only terms above -16");
    double expected_full = 0;
    for (int j = 0; j < 3; ++j) {
        const double la = double(r3[j]) - z, lb = double(c3[j]) - zc;
        expected_full += std::exp(la) * (la - lb);
    }
    check(std::abs(full.mean() - expected_full) < 1e-9 && std::abs(full.mean() - cut.mean()) > 1e-10, "full-vocab KL keeps the tail terms");
    bool refused = false;
    std::vector<float> bad = {NAN, 0.0f, 0.0f, 0.0f};
    try {tqllm::KlRows k; k.add(a.data(), bad.data(), 1, 4);} catch (const std::runtime_error&) {refused = true;}
    check(refused, "non-finite logits refused");
    std::puts(failures ? "KL_ROWS_FAIL" : "KL_ROWS_PASS");
    return failures ? 1 : 0;
}

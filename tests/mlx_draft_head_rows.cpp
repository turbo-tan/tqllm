// Fragment proposal head (draft_head4_matmul): a 16-row call (one pass over the head) gives the same bits per row as
// two 8-row calls, a 13-row call as an 8-row + a 5-row call, and every result stays within FP16-input rounding of
// MLX's quantized_matmul on the same 4-bit weights.
#include "draft_head.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main() {
    setenv("TQLLM_MLX_DRAFT_HEAD4", "1", 1);  // the fragment layout (the default is the matrix-unit layout)
    int failures = 0;
    auto check = [&](bool ok, const char* what) {if (!ok) {std::printf("FAIL %s\n", what); ++failures;}};
    const int n = 2048, k = 1024;
    mx::random::seed(5);
    auto weights = mx::astype(mx::multiply(mx::random::normal({n, k}), mx::array(0.05f)), mx::bfloat16);
    auto head = draft_head4(Matrix{weights});
    check(head.frag_codes.has_value(), "fragment layout packed");
    auto x16 = mx::random::normal({16, k});
    auto bits_equal = [](const mx::array& a, const mx::array& b) {
        auto ca = mx::contiguous(a), cb = mx::contiguous(b);
        mx::eval(ca, cb);
        return ca.shape() == cb.shape() && std::memcmp(ca.data<float>(), cb.data<float>(), ca.nbytes()) == 0;
    };
    auto full = draft_head4_matmul(head, x16);
    auto lo = draft_head4_matmul(head, mx::slice(x16, {0, 0}, {8, k})), hi = draft_head4_matmul(head, mx::slice(x16, {8, 0}, {16, k}));
    check(bits_equal(full, mx::concatenate({lo, hi}, 0)), "16 rows == two 8-row calls, bitwise");
    auto x13 = mx::slice(x16, {0, 0}, {13, k});
    auto t13 = draft_head4_matmul(head, x13);
    auto t5 = draft_head4_matmul(head, mx::slice(x16, {8, 0}, {13, k}));
    check(bits_equal(t13, mx::concatenate({lo, t5}, 0)), "13 rows == 8 + 5 rows, bitwise");
    auto ref = mx::quantized_matmul(x16, head.codes, head.scales, head.biases, true, 64, 4, "affine");
    auto rel = mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(full, ref))), mx::sum(mx::square(ref))));
    mx::eval(rel);
    check(rel.item<float>() < 2e-3f, "close to quantized_matmul (FP16 input)");
    std::printf("DRAFT_HEAD_ROWS rel_vs_quantized_matmul %.2e\n", rel.item<float>());
    std::puts(failures ? "DRAFT_HEAD_ROWS_FAIL" : "DRAFT_HEAD_ROWS_PASS");
    return failures ? 1 : 0;
}

// A dense matrix of a TQ4 model (TQ_L: BF16 ssm_alpha / ssm_beta) is stored in the TQ4 rotated basis, S = W R^T per
// 128-column block (tq4_rotate of W's rows); its product must be x W^T, so the input is rotated too (j945 / C15as: the
// plain input gave mean KL 0.53 vs the CUDA engine). Old behaviour (plain input) is checked to differ, so the test fails
// on the code before the fix.
#include "weights.hpp"
#include <mlx/mlx.h>
#include <mlx/random.h>
#include <cstdio>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main() {
    const int rows = 48, cols = 640;
    auto W = mx::random::normal({rows, cols}), x = mx::random::normal({16, cols});
    auto S = tq4_rotate(W);  // stored dense matrix, rotated basis
    mx::eval(W, x, S);
    const Matrix stored = S;
    auto want = mx::matmul(x, mx::transpose(W));
    auto rel = [&](const mx::array& y) {
        auto r = mx::divide(mx::max(mx::abs(mx::subtract(y, want))), mx::max(mx::abs(want)));
        mx::eval(r); return double(r.item<float>());
    };
    const double fixed = rel(linear(stored, x, true)), plain = rel(linear(stored, x, false));
    const double untouched = [&] {  // non-dense matrices ignore the flag: a TQ4 matrix rotates its input itself
        auto r = mx::max(mx::abs(mx::subtract(basis_input(Matrix(S), x, false), x))); mx::eval(r); return double(r.item<float>());
    }();
    std::printf("TQ4_BASIS dense rotated-basis matrix: rel err with basis input %.2e, with plain input %.2e (must be large), flag off is identity %.1e\n",
                fixed, plain, untouched);
    const bool ok = fixed < 1e-5 && plain > 0.1 && untouched == 0;
    std::printf(ok ? "MLX_TQ4_BASIS_PASS\n" : "MLX_TQ4_BASIS_FAIL\n");
    return ok ? 0 : 1;
}

// K-quant / IQ matrices on the MLX backend (Q2_K, Q3_K, Q4_K, Q5_K, IQ3_S, IQ4_XS, IQ4_NL: what unsloth UD GGUFs mix in).
// 1. GPU dequant == gguf-py's reference values (tests/data/kquant_golden.bin, tools/gen_kquant_golden.py) and == the
//    host decoder dequantize_host, bit for bit; 2. load_matrix accepts each type (before: "unsupported MLX linear weight
//    type"); 3. linear() at 1 / 4 / 16 rows (block-decoding kernel) and 40 rows (dequant + FP32 GEMM) vs FP32 matmul on
//    the dequantized matrix.
#include "weights.hpp"
#include "dequant.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;

int main() {
    std::ifstream in(KQ_GOLDEN, std::ios::binary);
    if (!in) {std::printf("MLX_KQUANT_FAIL cannot open %s\n", KQ_GOLDEN); return 1;}
    std::mt19937 rng(7);
    bool ok = true;
    int types = 0;
    uint32_t h[4];
    while (in.read(reinterpret_cast<char*>(h), 16)) {
        const uint32_t type = h[0], blocks = h[1], bytes = h[2], elems = h[3];
        std::vector<uint8_t> raw(size_t(blocks) * bytes);
        std::vector<float> want(size_t(blocks) * elems);
        in.read(reinterpret_cast<char*>(raw.data()), raw.size());
        in.read(reinterpret_cast<char*>(want.data()), want.size() * 4);
        // the golden blocks as a matrix of `rows` rows
        const int columns = elems == 256 ? 1024 : 256, rows = int(want.size()) / columns;
        tqllm::GgufTensor tensor;
        tensor.type = type;
        tensor.ne = {uint64_t(columns), uint64_t(rows)};
        Matrix matrix = mx::array(0.0f);
        try {matrix = load_matrix(tensor, {raw.data(), raw.size()});}
        catch (const std::exception& e) {std::printf("KQUANT type %u load_matrix FAILED: %s\n", type, e.what()); ok = false; continue;}
        const auto* w = std::get_if<KQuantWeights>(&matrix);
        if (!w) {std::printf("KQUANT type %u loaded as another matrix kind\n", type); ok = false; continue;}
        auto dense = kquant_dequant(*w);
        mx::eval(dense);
        std::vector<float> host(want.size());
        tqllm::dequantize_host(type, raw.data(), host.data(), int64_t(host.size()));
        size_t vs_ref = 0, vs_host = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            vs_ref += std::memcmp(dense.data<float>() + i, &want[i], 4) != 0;
            vs_host += std::memcmp(dense.data<float>() + i, &host[i], 4) != 0;
        }
        double worst_mm = 0;
        for (int m : {1, 4, 16, 40}) {
            std::vector<float> x(size_t(m) * columns);
            std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
            for (auto& v : x) v = xd(rng);
            auto X = mx::array(x.data(), {m, columns}, mx::float32);
            auto got = linear(matrix, X), ref = mx::matmul(X, mx::transpose(dense));
            auto rel = mx::divide(mx::max(mx::abs(mx::subtract(got, ref))), mx::max(mx::abs(ref)));
            mx::eval(rel);
            worst_mm = std::max(worst_mm, double(rel.item<float>()));
        }
        std::printf("KQUANT type %u rows=%d cols=%d dequant vs gguf-py %zu / %zu differ, vs host %zu differ, linear rel_err %.2e\n",
                    type, rows, columns, vs_ref, want.size(), vs_host, worst_mm);
        ok = ok && vs_ref == 0 && vs_host == 0 && worst_mm < 1e-5;
        ++types;
    }
    ok = ok && types == 7;
    std::printf(ok ? "MLX_KQUANT_PASS\n" : "MLX_KQUANT_FAIL\n");
    return ok ? 0 : 1;
}

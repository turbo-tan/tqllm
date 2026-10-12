#include "attention_rows.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace mx = mlx::core;
mx::array fixture(int heads, int rows, int dim, int seed) {
    std::vector<float> data(size_t(heads) * rows * dim);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = std::sin(float((i * 17 + size_t(seed)) % 1009) * .031f);
    return mx::astype(mx::astype(mx::array(data.data(), {1, heads, rows, dim}), mx::bfloat16), mx::float32);
}
mx::array bf(const mx::array& x) {
    return mx::astype(mx::astype(x, mx::bfloat16), mx::float32);
}
int main(int argc, char** argv) {
    try {
        const bool original = argc == 2 && std::string(argv[1]) == "--original";
        const bool views = argc == 2 && std::string(argv[1]) == "--views";
        if (argc > 2 || (argc == 2 && !original && !views)) throw std::invalid_argument("unknown test option");
        size_t changed = 0;
        for (int past : (views?std::vector<int>{0,17,257,1023,1024,2048,4097}:std::vector<int>{0,17,257})) for (int n : {1, 2, 4, 8}) {
            const int dim = 256;
            auto q = fixture(8, n, dim, 23), k = fixture(2, past + n, dim, 41), v = fixture(2, past + n, dim, 79);
            const float scale = 1.f / std::sqrt(float(dim));
            std::vector<mx::array> rows;
            for (int t = 0; t < n; ++t) {
                auto qt = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {1, 8, t + 1, dim}));
                auto kt = mx::contiguous(mx::slice(k, {0, 0, 0, 0}, {1, 2, past + t + 1, dim}));
                auto vt = mx::contiguous(mx::slice(v, {0, 0, 0, 0}, {1, 2, past + t + 1, dim}));
                rows.push_back(mx::fast::scaled_dot_product_attention(qt, kt, vt, scale, ""));
            }
            auto raw_ref=mx::contiguous(mx::concatenate(rows,2));
            auto raw_out=mx::contiguous(tqllm::mlx_backend::causal_attention_rows(q,k,v,scale,!original,views));
            mx::eval(raw_ref,raw_out);
            if(views)for(size_t i=0;i<raw_ref.size();++i)
                if(raw_ref.data<float>()[i]!=raw_out.data<float>()[i])
                    throw std::runtime_error("KV views change FP32 attention before BF16 rounding");
            auto ref = bf(raw_ref);
            auto out = bf(raw_out);
            mx::eval(ref, out);
            size_t different = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                if (!std::isfinite(ref.data<float>()[i]) || !std::isfinite(out.data<float>()[i]))
                    throw std::runtime_error("nonfinite attention result");
                different += ref.data<float>()[i] != out.data<float>()[i];
            }
            changed += different;
            std::cout << "ATTENTION_ROW_CHECK past=" << past << " width=" << n << " changed=" << different << '\n';
            if(views && n==8)for(int arm:{0,1,1,0}) {
                std::vector<double> times;
                for(int run=-1;run<7;++run) {
                    auto begin=std::chrono::steady_clock::now();
                    mx::eval(tqllm::mlx_backend::causal_attention_rows(q,k,v,scale,true,arm==1));
                    if(run>=0)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
                }
                std::sort(times.begin(),times.end());
                std::cout<<"ATTENTION_VIEWS_BENCH past="<<past<<" width="<<n<<" arm="<<arm
                         <<" median_ms="<<times[times.size()/2]<<std::endl;
            }
        }
        if (changed) throw std::runtime_error("batched attention differs from contiguous sequential queries");
        std::cout << "ATTENTION_ROWS_PASS\n";
    } catch (const std::exception& e) { std::cerr << "ATTENTION_ROWS_ERROR " << e.what() << '\n'; return 1; }
}

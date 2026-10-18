#include "draft_head.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;
using namespace tqllm::mlx_backend;
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: bench-mlx-draft-head model.gguf");
        auto file = tqllm::gguf_open(argv[1]);
        const int k = file.tensors.at("output.weight").ne[0];
        auto head = draft_head4(load_matrix(file, "output.weight"));
        for (int m : {1, 3, 5, 7, 8, 16}) {
            std::vector<float> values(size_t(m)*k);
            for (size_t j = 0; j < values.size(); ++j) values[j] = .1f*std::sin(float(j)*.037f);
            mx::array input(values.data(), {m,k});
            unsetenv("TQLLM_MLX_DRAFT_HEAD_UNPAD");
            auto reference = draft_head4_matmul(head,input); mx::eval(reference);
            setenv("TQLLM_MLX_DRAFT_HEAD_UNPAD","1",1);
            auto candidate = draft_head4_matmul(head,input); mx::eval(candidate);
            const float error = mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(candidate,reference))),
                mx::maximum(mx::sum(mx::square(reference)),mx::array(1e-20f)))).item<float>();
            if (!std::isfinite(error) || error > 2e-5f)
                throw std::runtime_error("unpadding changed proposal projection beyond FP32 gate");
            const int top1_same = mx::sum(mx::astype(mx::equal(mx::argmax(candidate,-1),mx::argmax(reference,-1)),mx::int32)).item<int>();
            std::vector<double> padded,native;
            for (int repeat = 0; repeat < 5; ++repeat) for (int arm = 0; arm < 2; ++arm) {
                if (arm) setenv("TQLLM_MLX_DRAFT_HEAD_UNPAD","1",1);
                else unsetenv("TQLLM_MLX_DRAFT_HEAD_UNPAD");
                auto start = std::chrono::steady_clock::now(); mx::eval(draft_head4_matmul(head,input));
                (arm?native:padded).push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            }
            std::sort(padded.begin(),padded.end()); std::sort(native.begin(),native.end());
            std::cout << "DRAFT_HEAD_BENCH m=" << m << " padded_ms=" << padded[2] << " native_ms=" << native[2]
                      << " rel_l2=" << error << " top1_same=" << top1_same << std::endl;
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

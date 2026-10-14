#include "gsq_activation.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static std::vector<__half> expected(const std::vector<float> & x, int block) {
    std::vector<__half> y(x.size());
    for (size_t base = 0; base < x.size(); base += block) {
        float peak = 0.f;
        for (int i = 0; i < block; ++i)
            if (std::abs(x[base + i]) > std::abs(peak)) peak = x[base + i];
        float scale = 0.f, inverse = 0.f;
        if (peak) {
            if (block == 256) inverse = -127.f / peak, scale = 1.f / inverse;
            else {
                const float raw = std::abs(peak) / 127.f;
                inverse = 1.f / raw;
                scale = __half2float(__float2half_rn(raw));
            }
        }
        for (int i = 0; i < block; ++i)
            y[base + i] = __float2half_rn(std::min(127.f, std::round(inverse * x[base + i])) * scale);
    }
    return y;
}
static void test(int block) {
    const int count = block * 4;
    std::vector<float> x(count);
    for (int i = 0; i < block; ++i) {
        x[i] = 0.f;
        x[block + i] = std::sin(float(i) * 0.23f) * 5.f;
        x[2*block + i] = std::cos(float(i) * 0.17f) * 6.f;
        x[3*block + i] = float((i * 37) % 131 - 65) * .073f;
    }
    x[block + 3] = -12.f;
    x[2*block + 5] = 14.f;
    x[3*block] = 127.f;
    x[3*block + 7] = .5f;
    x[3*block + 8] = -.5f;
    float * dx = nullptr; __half * dy = nullptr;
    check(cudaMalloc(&dx, x.size()*sizeof(float)));
    check(cudaMalloc(&dy, x.size()*sizeof(__half)));
    try {
        check(cudaMemcpy(dx, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice));
        if (block == 256) tqllm_cuda::gsq_q8k_round_h(dx,dy,count,nullptr);
        else tqllm_cuda::gsq_q8_round_h(dx,dy,count,nullptr);
        std::vector<__half> actual(count);
        check(cudaMemcpy(actual.data(),dy,actual.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        const auto oracle = expected(x,block);
        for (int i = 0; i < count; ++i) {
            const float a=__half2float(actual[i]), b=__half2float(oracle[i]);
            if (std::abs(a-b)>std::max(.0005f,std::abs(b)*.001f)) throw std::runtime_error("GSQ activation differs from Q8 oracle");
        }
        bool rejected = false;
        try {
            if (block == 256) tqllm_cuda::gsq_q8k_round_h(dx,dy,count-1,nullptr);
            else tqllm_cuda::gsq_q8_round_h(dx,dy,count-1,nullptr);
        } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("partial Q8 block accepted");
    } catch (...) { cudaFree(dx); cudaFree(dy); throw; }
    cudaFree(dx); cudaFree(dy);
}
int main() { test(256); test(32); }

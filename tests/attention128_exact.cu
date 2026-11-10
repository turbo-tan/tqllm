#include "qwen2_kernels.cuh"
#include "qwen35_kernels.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

template<class T> T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess ||
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("attention test allocation failed");
    return p;
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    using bf16 = __nv_bfloat16;
    constexpr int H = 32, KH = 8, D = 128, N = 16, CTX = 1024;
    std::mt19937 rng(191);
    std::normal_distribution<float> normal;
    auto values = [&](size_t n) {
        std::vector<bf16> v(n);
        for (auto& x : v) x = __float2bfloat16_rn(normal(rng));
        return v;
    };
    auto *q = upload(values(N * H * D)), *k = upload(values(CTX * KH * D)), *v = upload(values(CTX * KH * D));
    auto *y = upload(std::vector<bf16>(N * H * D));
    auto *part = upload(std::vector<float>(tqllm_cuda::attention_gen_partial_floats(H, CTX, N)));
    auto *pos = upload(std::vector<int>{0});
    int failures = 0;
    for (int start : {3, 127, 513}) {
        std::vector<bf16> reference(N * H * D), got(reference.size());
        for (int z = 0; z < N; ++z) {
            const int p = start + z;
            cudaMemcpy(pos, &p, sizeof(p), cudaMemcpyHostToDevice);
            tqllm_cuda::attention_gen128(q + z * H * D, k, v, y + z * H * D, part, pos, 1, CTX, H, KH,
                                         1.f / std::sqrt(float(D)), true, 0, nullptr);
        }
        cudaMemcpy(reference.data(), y, reference.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
        for (int n : {2, 8, 16}) {
            for (int z = 0; z < N; z += n) {
                const int p = start + z;
                cudaMemcpy(pos, &p, sizeof(p), cudaMemcpyHostToDevice);
                tqllm_cuda::attention_gen128(q + z * H * D, k, v, y + z * H * D, part, pos, n, CTX, H, KH,
                                             1.f / std::sqrt(float(D)), true, 0, nullptr);
            }
            if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) return 1;
            cudaMemcpy(got.data(), y, got.size() * sizeof(bf16), cudaMemcpyDeviceToHost);
            int differences = 0;
            for (size_t i = 0; i < got.size(); ++i)
                differences += std::memcmp(&got[i], &reference[i], sizeof(bf16)) != 0;
            std::printf("causal attention128 pos=%d batch=%d: %d BF16 differences\n", start, n, differences);
            failures += differences != 0;
        }
    }
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(y); cudaFree(part); cudaFree(pos);
    return failures ? 1 : 0;
}

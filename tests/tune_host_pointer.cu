// Load-time tuning must not launch GPU kernels on host memory. repack_expert (the MTP experts at start-up) loads each expert
// through an Upload that keeps the bytes on the host; IQ4_XS::load then called u4h_family_tune on that host pointer -> without
// HMM (e.g. the 3090 on driver 595 at .77) an illegal access that killed tqserve at start-up. Old code: the context dies here
// on such a machine; new code: tuning is skipped for non-device pointers and the context stays healthy.
#include "quant_kernels.cuh"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
int main() {
    const int rows = 640, k = 2560;
    std::vector<unsigned char> w(size_t(rows) * k);          // >= the u4h code bytes for these shapes
    std::vector<unsigned char> aux(size_t(rows) * (k / 32) * 8);  // >= the scale / min pairs
    tqllm_cuda::u4h_family_tune(13, w.data(), aux.data(), rows, k, nullptr);
    const cudaError_t e = cudaDeviceSynchronize(); const cudaError_t l = cudaGetLastError();
    if (e != cudaSuccess || l != cudaSuccess) { std::printf("FAIL: tuning on a host pointer broke the context: %s\n", cudaGetErrorString(e != cudaSuccess ? e : l)); return 1; }
    void * d = nullptr; if (cudaMalloc(&d, 16) != cudaSuccess) { std::printf("FAIL: context unusable after host-pointer tuning\n"); return 1; }
    cudaFree(d); std::printf("tuning on a host pointer is skipped, context healthy PASS\n"); return 0;
}

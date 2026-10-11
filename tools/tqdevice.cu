#include <cuda_runtime.h>
#include <iostream>
int main() {
    int n = 0;
    auto e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { std::cerr << cudaGetErrorString(e) << '\n'; return 1; }
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        e = cudaGetDeviceProperties(&p, i);
        if (e != cudaSuccess) { std::cerr << cudaGetErrorString(e) << '\n'; return 1; }
        std::cout << p.name << " cc=" << p.major << '.' << p.minor << " memory_bytes=" << p.totalGlobalMem << '\n';
    }
    return n ? 0 : 1;
}

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// The vendored MMQ headers call these helpers without linking the ggml runtime.
void ggml_cuda_error(const char* statement, const char* function, const char* file, int line, const char* message) {
    std::fprintf(stderr, "%s:%d: %s: %s: %s\n", file, line, function, statement, message);
    std::abort();
}

int ggml_cuda_get_device() {
    int device = 0;
    const auto result = cudaGetDevice(&device);
    if (result != cudaSuccess)
        ggml_cuda_error("cudaGetDevice", __func__, __FILE__, __LINE__, cudaGetErrorString(result));
    return device;
}

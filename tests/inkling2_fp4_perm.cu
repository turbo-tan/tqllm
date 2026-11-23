// Diagnostic: one non-zero weight code at q and one non-zero activation at p; print the p that lights up row 0.
#include "inkling2_grouped.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace tqllm_cuda::inkling2;

static void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int main() {
    const int D = 64, I = 64, A = 1, E = 1;
    const size_t wb = size_t(D) * I / 2, ab = size_t(D) * (I / 16 + 4);
    uint8_t * w = nullptr, * aux = nullptr, * qcodes = nullptr, * qscales = nullptr;
    float * qrow = nullptr;
    __nv_bfloat16 * x = nullptr, * y = nullptr;
    int * ids = nullptr, * order = nullptr, * counts = nullptr, * offsets = nullptr, * tile_off = nullptr;
    ck(cudaMalloc(&w, wb), "w"); ck(cudaMalloc(&aux, ab), "aux");
    ck(cudaMalloc(&qcodes, size_t(A) * I / 2), "qcodes"); ck(cudaMalloc(&qscales, size_t(A) * I / 16), "qscales");
    ck(cudaMalloc(&qrow, size_t(A) * 4), "qrow"); ck(cudaMalloc(&x, size_t(A) * I * 2), "x"); ck(cudaMalloc(&y, size_t(A) * D * 2), "y");
    ck(cudaMalloc(&ids, size_t(A) * 4), "ids"); ck(cudaMalloc(&order, size_t(A) * 4), "order");
    ck(cudaMalloc(&counts, size_t(E) * 4), "counts"); ck(cudaMalloc(&offsets, size_t(E) * 4), "offsets");
    ck(cudaMalloc(&tile_off, size_t(E + 1) * 4), "tile off");

    std::vector<uint8_t> ha(ab, 0), hw(wb, 0);
    for (int r = 0; r < D; ++r) {
        uint8_t * ar = ha.data() + size_t(r) * (I / 16 + 4);
        std::memset(ar, 0x38, I / 16);
        const float one = 1.0f;
        std::memcpy(ar + I / 16, &one, 4);
    }
    ck(cudaMemcpy(aux, ha.data(), ha.size(), cudaMemcpyHostToDevice), "aux");
    const int zero = 0;
    ck(cudaMemcpy(ids, &zero, 4, cudaMemcpyHostToDevice), "ids");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    group_assignments(ids, A, E, order, counts, offsets, tile_off, s);
    ExpertMat m; m.w = w; m.aux = aux; m.stride = wb; m.aux_stride = ab; m.fmt = 2;

    for (int q = 0; q < I; ++q) {
        std::fill(hw.begin(), hw.end(), 0);
        hw[q / 2] = uint8_t(2u << (4 * (q & 1)));
        ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "w");
        int hit = -1;
        for (int p = 0; p < I; ++p) {
            std::vector<__nv_bfloat16> hx(I, __float2bfloat16(0.0f));
            hx[p] = __float2bfloat16(1.0f);
            ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "x");
            quantize_act_nvfp4(x, A, I, qcodes, qscales, qrow, s);
            grouped_down_fp4(m, D, I, qcodes, qscales, qrow, A, E, order, counts, offsets, tile_off, y, s);
            ck(cudaStreamSynchronize(s), "down");
            __nv_bfloat16 got;
            ck(cudaMemcpy(&got, y, 2, cudaMemcpyDeviceToHost), "read y");
            if (std::fabs(__bfloat162float(got)) > 0.5f) { hit = p; break; }
        }
        std::printf("q %2d -> p %2d\n", q, hit);
    }
    return 0;
}

// Prefill GEMM paths at Qwen3.8-27B shapes: the format's own GEMM (what tqLLM runs) against FP16 weights + a cuBLAS
// FP16 GEMM with FP32 accumulation (the rate gpu-unlock doubles on GeForce), with and without the per-call dequant.
// Random weight bytes (timing only). Prints tok/s-equivalent TFLOPS per path; no ctest entry.
//   prefill-fp16-bench [TYPE ...]   GGUF type ids, default 300 (TQ4) 12 (Q4_K)
#include "weight_format.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

using namespace tqllm;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); std::exit(2); } }

double time_ms(const std::function<void()> & f, cudaStream_t s) {
    for (int w = 0; w < 2; ++w) f();
    cudaEvent_t a, b; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    std::vector<float> t;
    for (int r = 0; r < 5; ++r) {
        ck(cudaEventRecord(a, s)); f(); ck(cudaEventRecord(b, s)); ck(cudaEventSynchronize(b));
        float ms = 0; ck(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[2];
}

size_t raw_bytes(uint32_t type, const WeightFormat & f, int rows, int cols) {
    if (const size_t rb = f.row_bytes(cols)) return size_t(rows) * (rb + f.aux_row_bytes(cols));
    if (type == 12) return size_t(rows) * (cols / 256) * 144;  // Q4_K
    if (type == 14) return size_t(rows) * (cols / 256) * 210;  // Q6_K
    if (type == 8) return size_t(rows) * (cols / 32) * 34;     // Q8_0
    std::fprintf(stderr, "no raw size for type %u\n", type); std::exit(2);
}
}  // namespace

int main(int argc, char ** argv) {
    std::vector<uint32_t> types;
    for (int i = 1; i < argc; ++i) types.push_back(uint32_t(std::atoi(argv[i])));
    if (types.empty()) types = {300, 12};
    struct Shape { const char * name; int rows, cols; } shapes[] = {{"gate/up 5120->17408", 17408, 5120}, {"down 17408->5120", 5120, 17408},
                                                                     {"q 5120->12288", 12288, 5120}};
    cudaStream_t s; ck(cudaStreamCreate(&s));
    cublasHandle_t h; cublasCreate(&h); cublasSetStream(h, s);
    std::mt19937 rng(1);
    for (uint32_t type : types) {
        const WeightFormat * f = weight_format(type);
        if (!f) { std::printf("type %u: no format\n", type); continue; }
        for (const Shape & sh : shapes) {
            std::vector<char> raw(raw_bytes(type, *f, sh.rows, sh.cols));
            for (auto & c : raw) c = char(rng() & 0x3f);  // small exponents: no inf / NaN scales
            std::vector<void *> keep;
            const Mat m = f->load(raw, sh.rows, sh.cols, [&](const void * p, size_t b) -> const void * {
                void * d; ck(cudaMalloc(&d, b)); ck(cudaMemcpy(d, p, b, cudaMemcpyHostToDevice)); keep.push_back(d); return d; }, s);
            const WeightFormat * g = m.fmt;  // a loader (Q4_K) hands back the runtime format (U4H)
            __half * wh; ck(cudaMalloc(&wh, size_t(sh.rows) * sh.cols * 2));
            for (int n : {512, 2048, 4096}) {
                __half * x; __nv_bfloat16 * y; float * yf;
                ck(cudaMalloc(&x, size_t(n) * sh.cols * 2)); ck(cudaMemset(x, 0, size_t(n) * sh.cols * 2));
                ck(cudaMalloc(&y, size_t(n) * sh.rows * 2)); ck(cudaMalloc(&yf, size_t(n) * sh.rows * 4));
                const double flop = 2.0 * sh.rows * sh.cols * n;
                auto tf = [&](double ms) { return flop / (ms / 1e3) / 1e12; };
                double native = -1;
                if (g->has_gemm()) native = time_ms([&] { g->gemm(m, x, y, n, s); }, s);
                const float one = 1, zero = 0;
                auto blas = [&] {
                    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.rows, n, sh.cols, &one, wh, CUDA_R_16F, sh.cols, x, CUDA_R_16F, sh.cols,
                                 &zero, yf, CUDA_R_32F, sh.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                };
                const double dq = time_ms([&] { g->dequant_h(m, wh, s); }, s);
                const double fp16 = time_ms(blas, s);
                std::printf("%-6s %-20s n=%4d  native %s  fp16-cublas %6.1f TF (dequant %.3f ms = %6.1f TF incl.)\n", g->name(), sh.name, n,
                            native > 0 ? (std::to_string(tf(native)).substr(0, 5) + " TF").c_str() : "  n/a  ", tf(fp16), dq, tf(fp16 + dq));
                std::fflush(stdout);
                cudaFree(x); cudaFree(y); cudaFree(yf);
            }
            cudaFree(wh); for (void * p : keep) cudaFree(p);
        }
    }
    return 0;
}

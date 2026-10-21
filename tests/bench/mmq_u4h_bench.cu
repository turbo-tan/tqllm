// ggml MMQ (vendored, MIT) on tqLLM's IQ3_XXS bank layout vs the stock GGUF layout: same data, same routing.
// Outputs must be bit-identical; time = mul_mat_q kernel only (quantize + routing helpers outside the timed loop).
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

template <ggml_type type>
static void run(ggml_backend_cuda_context & ctx, const mmq_args & a, cudaStream_t s) {
    if (a.nrows_x % 128 == 0) mul_mat_q_switch_J<type, false>(ctx, a, s);
    else mul_mat_q_switch_J<type, true>(ctx, a, s);
}

int main(int argc, char ** argv) {
    const int E = 512, ROWS = 640, K = 2560, T = argc > 1 ? std::atoi(argv[1]) : 8192, TOPK = 10, NB = K / 256;
    std::mt19937 rng(1);
    // GGUF blocks: d (fp16) + qs[96]
    const size_t nblk = size_t(E) * ROWS * NB;
    std::vector<block_iq3_xxs> w(nblk);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & b : w) {
        b.d = __float2half(0.002f + 0.001f * std::fabs(nd(rng)));
        for (int i = 0; i < 96; ++i) b.qs[i] = uint8_t(byte(rng));
    }
    // tqLLM layout: per (row, 32-block) 12 bytes = qs[8 j .. 8 j + 7] + aux32 j; record per (row, 256) = d + 2 pad bytes
    const int nb32 = K / 32, R = E * ROWS;
    std::vector<uint8_t> codes(size_t(R) * nb32 * 12), codes_t(codes.size()), rec(size_t(R) * NB * 4, 0);
    for (int r = 0; r < R; ++r)
        for (int sb = 0; sb < NB; ++sb) {
            const block_iq3_xxs & b = w[size_t(r) * NB + sb];
            std::memcpy(&rec[(size_t(r) * NB + sb) * 4], &b.d, 2);
            for (int j = 0; j < 8; ++j) {
                const int s = 8 * sb + j;
                uint8_t blk[12];
                std::memcpy(blk, b.qs + 8 * j, 8);
                std::memcpy(blk + 8, b.qs + 64 + 4 * j, 4);
                std::memcpy(&codes[(size_t(r) * nb32 + s) * 12], blk, 12);
                const size_t to = (((size_t(r >> 4) * (nb32 >> 2) + (s >> 2)) * 16 + (r & 15)) * 4 + (s & 3)) * 12;
                std::memcpy(&codes_t[to], blk, 12);
            }
        }
    // routing: Zipf-ish expert popularity, TOPK distinct experts per token
    std::vector<double> pop(E); for (int e = 0; e < E; ++e) pop[e] = 1.0 / std::pow(e + 1.0, 0.6);
    std::shuffle(pop.begin(), pop.end(), rng); std::discrete_distribution<int> pick(pop.begin(), pop.end());
    std::vector<std::vector<int>> per(E);  // assignment index t * TOPK + j per expert
    for (int t = 0; t < T; ++t) {
        std::vector<int> ch; while (int(ch.size()) < TOPK) { int e = pick(rng); if (std::find(ch.begin(), ch.end(), e) == ch.end()) ch.push_back(e); }
        for (int j = 0; j < TOPK; ++j) per[ch[j]].push_back(t * TOPK + j);
    }
    std::vector<int32_t> ids_src1, ids_dst, bounds(E + 1, 0);
    for (int e = 0; e < E; ++e) {
        bounds[e] = int(ids_dst.size());
        for (int a : per[e]) { ids_src1.push_back(a / TOPK); ids_dst.push_back(a); }
    }
    bounds[E] = int(ids_dst.size());
    const int NA = T * TOPK;
    std::vector<float> x(size_t(T) * K); for (auto & v : x) v = 0.3f * nd(rng);

    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_cuda_context & ctx = *new ggml_backend_cuda_context(0);
    cudaStream_t st = cudaStreamPerThread;
    char * dw, * dc, * dct; uint8_t * dr; float * dx, * dy; int32_t * dsrc, * ddst, * dbnd; char * dq;
    CK(cudaMalloc(&dw, nblk * sizeof(block_iq3_xxs))); CK(cudaMalloc(&dc, codes.size())); CK(cudaMalloc(&dct, codes.size()));
    CK(cudaMalloc(&dr, rec.size())); CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, size_t(NA) * ROWS * 4));
    CK(cudaMalloc(&dsrc, NA * 4)); CK(cudaMalloc(&ddst, NA * 4)); CK(cudaMalloc(&dbnd, (E + 1) * 4));
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t qbytes = size_t(NA) * K_pad * sizeof(block_q8_1) / QK8_1 + 128 * sizeof(block_q8_1_mmq);
    CK(cudaMalloc(&dq, qbytes)); CK(cudaMemset(dq, 0, qbytes));
    CK(cudaMemcpy(dw, w.data(), nblk * sizeof(block_iq3_xxs), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, codes.data(), codes.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dct, codes_t.data(), codes.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dr, rec.data(), rec.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsrc, ids_src1.data(), NA * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddst, ids_dst.data(), NA * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbnd, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice));
    quantize_mmq_q8_1_cuda(dx, dsrc, dq, GGML_TYPE_IQ3_XXS, K, K, int64_t(K) * T, int64_t(K) * T, K_pad, NA, 1, 1, st);
    CK(cudaGetLastError());

    const int64_t s12 = int64_t(1) * K_pad * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    mmq_args a = {nullptr, GGML_TYPE_IQ3_XXS, (const int *) dq, ddst, dbnd, dy, nullptr,
                  K, ROWS, NA, NB, NA, ROWS,
                  E, E, int64_t(ROWS) * NB, s12, ROWS,
                  1, 1, 0, 0, 0,
                  T, T};
    std::vector<float> ref(size_t(NA) * ROWS), out(ref.size());
    const char * names[3] = {"stock GGUF", "u4h untiled", "u4h tiled"};
    for (int m = 0; m < 3; ++m) {
        tq_u4h_desc d = {m ? dr : nullptr, m ? NB : 0, m == 2};
        CK(cudaMemcpyToSymbol(tq_u4h, &d, sizeof d));
        a.x = m == 0 ? dw : m == 1 ? dc : dct;
        CK(cudaMemsetAsync(dy, 0, size_t(NA) * ROWS * 4, st));
        run<GGML_TYPE_IQ3_XXS>(ctx, a, st);
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        CK(cudaMemcpy(m ? out.data() : ref.data(), dy, ref.size() * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0; double sum = 0;
        if (m) for (size_t i = 0; i < ref.size(); ++i) diff += std::memcmp(&ref[i], &out[i], 4) != 0;
        for (size_t i = 0; i < ref.size(); i += 997) sum += std::fabs(m ? out[i] : ref[i]);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 20;
        cudaEventRecord(e0, st);
        for (int it = 0; it < iters; ++it) run<GGML_TYPE_IQ3_XXS>(ctx, a, st);
        cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        std::printf("%-12s T%d: %7.3f ms/projection  |sum| %.4g  bit-diff %zu / %zu\n", names[m], T, ms / iters, sum, diff, ref.size());
    }
    std::fflush(stdout);
    std::_Exit(0);
}

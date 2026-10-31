// Two-box NCCL latency: rank 0 / rank 1 time all-reduces of a MiniMax hidden vector (6144 floats) the way a
// tensor-parallel decode step issues them (stream-ordered, back to back).  tp-bench RANK PEER PORT [floats] [iters]
#include "tp_comm.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
int main(int argc, char ** argv) try {
    if (argc < 4) { std::fprintf(stderr, "usage: tp-bench RANK PEER PORT [floats] [iters]\n"); return 2; }
    const int rank = std::atoi(argv[1]); const size_t n = argc > 4 ? size_t(std::atol(argv[4])) : 6144; const int it = argc > 5 ? std::atoi(argv[5]) : 2000;
    tqllm::TpComm tp(rank, 2, argv[2], std::atoi(argv[3]));
    float * x = nullptr; cudaMalloc(&x, n * 4); cudaStream_t s; cudaStreamCreate(&s);
    for (int i = 0; i < 100; ++i) tp.all_reduce_sum(x, n, s);
    cudaStreamSynchronize(s);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) tp.all_reduce_sum(x, n, s);
    cudaStreamSynchronize(s);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::printf("rank %d: all-reduce %zu floats: %.1f us each\n", rank, n, us);
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "tp-bench: %s\n", e.what()); return 1; }

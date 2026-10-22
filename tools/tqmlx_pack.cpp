// Converts a TQ3/TQ4 GGUF into the tq_mlx kernel-ready layout.
#include "tq_mlx.hpp"
#include <chrono>
#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: tqmlx-pack <source.gguf> <destination.gguf>\n");
        return 2;
    }
    try {
        const auto start = std::chrono::steady_clock::now();
        const auto s = tqllm::mlx_backend::tq_mlx_convert(argv[1], argv[2]);
        std::printf("tq_mlx: %d tiled, %d mma8, %d copied, %.2f GB in, %.2f GB out, %.1f s\n", s.tiled, s.mma8, s.copied,
                    s.bytes_in / 1e9, s.bytes_out / 1e9,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tqmlx-pack: %s\n", e.what());
        return 1;
    }
}

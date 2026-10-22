// Image embeddings from an mmproj GGUF: [int32 nx, ny, n_tok, n_embd] + float[n_tok][n_embd] (the layout of
// llama.cpp mtmd_get_output_embd, for parity checks).
// usage: tqvision MMPROJ.gguf IMAGE OUT.bin [--image-min-tokens N] [--image-max-tokens N]
#include "vision.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MMPROJ.gguf IMAGE OUT.bin [--image-min-tokens N] [--image-max-tokens N]\n", argv[0]);
        return 1;
    }
    int mn = 8, mx = 4096;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--image-min-tokens")) mn = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--image-max-tokens")) mx = std::atoi(argv[i + 1]);
    }
    try {
        cudaStream_t s = cudaStreamPerThread;  // ordered with the synchronous copies (stream 0 is per thread)
        auto enc = tqllm::VisionEncoder::load(argv[1], s);
        enc->set_token_limits(mn, mx);
        std::ifstream in(argv[2], std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        if (!tqllm::decode_image(bytes.data(), bytes.size(), rgb, w, h)) throw std::runtime_error("cannot decode image");
        const auto t0 = std::chrono::steady_clock::now();
        const auto im = enc->preprocess(rgb.data(), w, h);
        int nx = 0, ny = 0;
        const float * e = enc->encode(im, nx, ny);
        const int nt = nx * ny, d = enc->embd();
        std::vector<float> out(size_t(nt) * d);
        cudaMemcpyAsync(out.data(), e, out.size() * 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA error");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const int32_t hdr[4] = {nx, ny, nt, d};
        FILE * f = std::fopen(argv[3], "wb");
        std::fwrite(hdr, 4, 4, f);
        std::fwrite(out.data(), 4, out.size(), f);
        std::fclose(f);
        std::printf("image %dx%d -> %dx%d input, %dx%d tokens, %.1f ms\n", w, h, im.w, im.h, nx, ny, ms);
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "tqvision: %s\n", ex.what());
        return 1;
    }
    return 0;
}

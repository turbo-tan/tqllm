#include "host_matrix.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <unistd.h>

using namespace tqllm;
namespace {
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
void compare(const std::vector<float> & a, const std::vector<float> & b, double & error) {
    check(a.size() == b.size(), "packed output shape");
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite packed output");
        error = std::max(error, double(std::abs(a[i] - b[i])));
        check(std::abs(a[i] - b[i]) <= 2e-5f * std::max(1.f, std::abs(b[i])), "packed dot differs from decoded reference");
    }
}
void fixtures() {
    std::string pattern = (std::filesystem::temp_directory_path() / "tqandroid-packed-XXXXXX").string();
    std::vector<char> filename(pattern.begin(), pattern.end()); filename.push_back(0);
    int fd = mkstemp(filename.data()); check(fd >= 0, "packed test temporary file"); close(fd);
    struct Cleanup { const char * path; ~Cleanup() { unlink(path); } } cleanup{filename.data()};
    for (uint32_t type : {GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0}) {
        for (int cols : {256, 768, 3072}) {
            uint64_t block, bytes; check(gguf_block_layout(type, block, bytes), "block layout");
            std::vector<unsigned char> data(size_t(cols) / block * bytes * 7);
            uint32_t rng = 1234567;
            for (auto & v : data) { rng = rng * 1664525u + 1013904223u; v = uint8_t(rng >> 24); }
            for (size_t k = 0; k < data.size(); k += bytes) {
                const size_t scale = k + (type == GGML_Q6_K ? 208 : 0);
                data[scale] = uint8_t(k); data[scale + 1] = 0x24;
                if (type == GGML_Q4_K || type == GGML_Q5_K) { data[k + 2] = uint8_t(k >> 3); data[k + 3] = 0x20; }
            }
            { std::ofstream f(filename.data(), std::ios::binary); f.write(reinterpret_cast<const char *>(data.data()), data.size()); }
            MappedFile mapping(filename.data()); GgufTensor t;
            t.type = type; t.ne = {uint64_t(cols), 7}; t.bytes = data.size();
            HostMatrix matrix(t, mapping, cols, 7);
            std::vector<float> input(cols);
            for (int i = 0; i < cols; ++i) input[i] = float(i % 31 - 15) * .03125f;
            double error = 0.;
            for (int threads : {1, 4}) {
                HostExecutor executor(threads);
                compare(matrix.multiply(input, executor), matrix.multiply(input, executor, false), error);
                std::fill(input.begin(), input.end(), 0.f);
                compare(matrix.multiply(input, executor), std::vector<float>(7, 0.f), error);
                for (int i = 0; i < cols; ++i) input[i] = float(i % 31 - 15) * .03125f;
            }
            std::cout << "packed_oracle type=" << type << " cols=" << cols << " max_abs=" << error << '\n';
        }
    }
}
void real_shapes(const std::string & path) {
    auto file = gguf_open(path); MappedFile mapping(path); HostExecutor executor(4);
    std::map<uint32_t, std::string> selected;
    for (const auto & [name, t] : file.tensors)
        if (t.ne.size() == 2 && name != "token_embd.weight" && t.ne[1] > 1024 &&
            (t.type == GGML_Q4_K || t.type == GGML_Q5_K || t.type == GGML_Q6_K || t.type == GGML_Q8_0))
            selected.try_emplace(t.type, name);
    for (const auto & [type, name] : selected) {
        const auto & t = file.tensors.at(name);
        HostMatrix matrix(t, mapping, int(t.ne[0]), int(t.ne[1]));
        std::vector<float> input(matrix.cols);
        for (int i = 0; i < matrix.cols; ++i) input[i] = float(i % 31 - 15) * .03125f;
        double error = 0.;
        compare(matrix.multiply(input, executor), matrix.multiply(input, executor, false), error);
        double total[2] = {};
        float checksum = 0.f;
        for (int round = 0; round < 6; ++round) {
            for (int j = 0; j < 2; ++j) {
                const int packed = (round + j) % 2;
                auto start = std::chrono::steady_clock::now();
                auto y = matrix.multiply(input, executor, bool(packed));
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                if (round) total[packed] += ms;
                checksum += y[0];
            }
        }
        std::cout << "packed_bench name=" << name << " type=" << type << " rows=" << matrix.rows
                  << " cols=" << matrix.cols << " reference_ms=" << total[0] / 5.
                  << " packed_ms=" << total[1] / 5. << " speedup=" << total[0] / total[1]
                  << " max_abs=" << error << " checksum=" << checksum << '\n';
    }
}
}
int main(int argc, char ** argv) {
    try {
        fixtures();
        if (argc == 2) real_shapes(argv[1]);
        std::cout << "android_packed: PASS\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}

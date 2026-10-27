#include "host_matrix.hpp"
#include "host_attention_cache.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace tqllm;
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        HostExecutor executor(4);
        std::vector<int> visits(37);
        executor.rows(37, [&](int a, int b) { for (int i = a; i < b; ++i) ++visits[i]; });
        for (int v : visits) check(v == 1, "row partition lost/duplicated work");
        bool caught = false;
        try { executor.rows(1, [](int a, int b) { if (a != b) throw std::runtime_error("worker"); }); }
        catch (const std::runtime_error &) { caught = true; }
        check(caught, "worker error not propagated");
        executor.rows(0, [](int, int) {});
        HostAttentionCache kv(3, 2);
        kv.write(2, {1.f, 2.f}, {3.f, 4.f});
        check(kv.key(2, 1)[0] == 2.f && kv.value(2, 0)[0] == 3.f, "KV addressing");
        caught = false; try { kv.write(3, {1.f, 2.f}, {3.f, 4.f}); } catch (const std::out_of_range &) { caught = true; }
        check(caught, "KV overflow accepted");
        const std::string pattern = (std::filesystem::temp_directory_path() / "tqandroid-matrix-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
        int fd = mkstemp(name.data()); check(fd >= 0, "temporary file"); close(fd);
        struct Cleanup { const char * name; ~Cleanup() { unlink(name); } } cleanup{name.data()};
        // Two Q4_K rows, each with different scale/min metadata and nibble patterns.
        std::vector<unsigned char> bytes(288);
        for (int r = 0; r < 2; ++r) {
            auto * b = bytes.data() + r * 144;
            b[1] = 0x3c; b[3] = 0x38; // d=1, dmin=0.5
            for (int j = 0; j < 4; ++j) { b[4 + j] = 2 + r; b[8 + j] = 1; b[12 + j] = 0x12; }
            for (int j = 0; j < 128; ++j) b[16 + j] = uint8_t(j + r);
        }
        { std::ofstream f(name.data(), std::ios::binary); f.write(reinterpret_cast<char *>(bytes.data()), bytes.size()); }
        MappedFile mapping(name.data());
        GgufTensor tensor; tensor.type = GGML_Q4_K; tensor.ne = {256, 2}; tensor.bytes = bytes.size();
        HostMatrix matrix(tensor, mapping, 256, 2);
        std::vector<float> x(256);
        for (int i = 0; i < 256; ++i) x[i] = float(i % 7 - 3) * .125f;
        auto result = matrix.multiply(x, executor);
        for (int r = 0; r < 2; ++r) {
            double expected = 0.;
            for (int i = 0; i < 256; ++i) {
                const int group = i / 32, j = i % 32;
                const int packed = bytes[size_t(r) * 144 + 16 + (group / 2) * 32 + j];
                const int code = group % 2 ? packed >> 4 : packed & 15;
                const int scale = group < 4 ? 2 + r : 2;
                expected += (float(scale * code) - .5f) * x[i];
            }
            check(std::abs(result[r] - expected) < .001, "Q4_K native dot vs independent block formula");
        }
        caught = false; try { mapping.at(287, 2); } catch (const std::out_of_range &) { caught = true; }
        check(caught, "mapping out of bounds accepted");
        tensor.type = GGML_TQ4;
        caught = false; try { HostMatrix unsupported(tensor, mapping, 256, 2); } catch (const std::invalid_argument &) { caught = true; }
        check(caught, "unsupported nonlinear format accepted");
        std::cout << "android_native: PASS (worker recovery, KV bounds, packed Q4_K, mapping, format rejection)\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}

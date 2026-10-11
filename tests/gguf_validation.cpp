#include "gguf.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
int failures = 0;
struct Fixture {
    uint32_t alignment = 32, alignment_type = 4, type = 0;
    bool explicit_alignment = false;
    std::vector<uint64_t> dims{1};
    uint64_t offset = 0;
    template <class T> static void put(std::string & out, T v) { out.append(reinterpret_cast<const char *>(&v), sizeof(v)); }
    static void str(std::string & out, const std::string & s) { put<uint64_t>(out, s.size()); out += s; }
    void write(const std::string & path) const {
        std::string out = "GGUF";
        put<uint32_t>(out, 3); put<uint64_t>(out, 1); put<uint64_t>(out, explicit_alignment ? 1 : 0);
        if (explicit_alignment) {
            str(out, "general.alignment"); put<uint32_t>(out, alignment_type);
            if (alignment_type == 8) str(out, "32"); else put<uint32_t>(out, alignment);
        }
        str(out, "weight"); put<uint32_t>(out, dims.size());
        for (auto d : dims) put<uint64_t>(out, d);
        put<uint32_t>(out, type); put<uint64_t>(out, offset);
        const auto pad = alignment && !(alignment & (alignment - 1)) && alignment <= 64 ? alignment : 32;
        out.resize((out.size() + pad - 1) / pad * pad, '\0');
        out.resize(out.size() + 128, '\0');
        std::ofstream(path, std::ios::binary).write(out.data(), out.size());
    }
};
void check(const Fixture & fixture, bool accept, const std::string & name, const std::string & path) {
    fixture.write(path);
    bool accepted = false;
    try {
        auto f = tqllm::gguf_open(path);
        auto data = f.read(f.tensors.at("weight"));
        accepted = true;
    } catch (const std::exception &) {}
    if (accepted != accept) { std::cerr << "FAIL: " << name << "\n"; ++failures; }
}
}
int main() {
    const auto path = (std::filesystem::temp_directory_path() / ("tqllm_gguf_validation_" + std::to_string(getpid()))).string();
    Fixture f;
    check(f, true, "default alignment", path);
    f.explicit_alignment = true; f.alignment = 64;
    check(f, true, "explicit power-of-two alignment", path);
    f.alignment = 3; check(f, false, "non-power-of-two alignment", path);
    f.alignment = 32; f.alignment_type = 8; check(f, false, "alignment must be uint32", path);
    f = Fixture{}; f.dims = {}; check(f, false, "zero dimensions", path);
    f.dims = {0}; check(f, false, "zero extent", path);
    f.dims = {1, 1, 1, 1, 1}; check(f, false, "too many dimensions", path);
    f.dims = {uint64_t(1) << 32, uint64_t(1) << 32}; check(f, false, "dimension product overflow", path);
    f.dims = {(uint64_t(1) << 62) + 1}; check(f, false, "tensor bytes overflow", path);
    f = Fixture{}; f.offset = std::numeric_limits<uint64_t>::max(); check(f, false, "offset addition overflow", path);
    f.offset = std::numeric_limits<uint64_t>::max() - 128; f.dims = {16};
    check(f, false, "offset plus bytes overflow", path);
    f = Fixture{}; f.dims = {64}; check(f, false, "truncated payload", path);
    f.dims = {256}; f.type = 13; check(f, false, "truncated Q5_K payload", path);
    f = Fixture{}; f.explicit_alignment = true; f.alignment = 0;
    check(f, false, "zero alignment", path);
    std::filesystem::remove(path);
    if (failures) return 1;
    std::cout << "gguf_validation: 14 cases passed\n";
}

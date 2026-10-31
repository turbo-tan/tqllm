// Split GGUFs (<prefix>-0000k-of-0000N.gguf) open as one file from any shard; tensors read from their own shard.
#include "gguf.hpp"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace tqllm;
namespace fs = std::filesystem;
static int failures = 0;
static void check(bool ok, const char * what) { if (!ok) { std::printf("FAIL: %s\n", what); ++failures; } }
template<class T> static void put(std::vector<char> & b, T v) {
    const auto p = reinterpret_cast<const char *>(&v); b.insert(b.end(), p, p + sizeof(v));
}
static void text(std::vector<char> & b, const std::string & s) { put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }
static void kv_u32(std::vector<char> & b, const std::string & k, uint32_t v) { text(b, k); put<uint32_t>(b, 4); put<uint32_t>(b, v); }
static void kv_u16(std::vector<char> & b, const std::string & k, uint16_t v) { text(b, k); put<uint32_t>(b, 2); put<uint16_t>(b, v); }

// One shard: split.no / split.count / split.tensors.count (first shard only, plus an arch key), and F32 tensors
// named t<k> holding [k*10 + i, i = 0..3].
static void write_shard(const fs::path & path, int no, int count, int total, const std::vector<int> & ids) {
    std::vector<char> h = {'G', 'G', 'U', 'F'};
    put<uint32_t>(h, 3); put<uint64_t>(h, ids.size()); put<uint64_t>(h, no == 0 ? 4 : 2);
    if (no == 0) { text(h, "general.architecture"); put<uint32_t>(h, 8); text(h, "test"); }
    kv_u16(h, "split.no", uint16_t(no)); kv_u16(h, "split.count", uint16_t(count));
    if (no == 0) kv_u32(h, "split.tensors.count", uint32_t(total));
    for (size_t j = 0; j < ids.size(); ++j) {
        text(h, "t" + std::to_string(ids[j])); put<uint32_t>(h, 1); put<uint64_t>(h, 4);
        put<uint32_t>(h, 0); put<uint64_t>(h, j * 32);
    }
    h.resize((h.size() + 31) / 32 * 32);
    for (int id : ids) { for (int i = 0; i < 4; ++i) put<float>(h, float(id * 10 + i)); h.resize(h.size() + 16); }
    std::ofstream(path, std::ios::binary).write(h.data(), std::streamsize(h.size()));
}

static bool throws(const std::string & path) {
    try { gguf_open(path); } catch (const std::exception &) { return true; }
    return false;
}

int main() {
    const auto dir = fs::temp_directory_path() / ("tqllm-split-" + std::to_string(getpid()));
    fs::create_directories(dir);
    struct Cleanup { fs::path d; ~Cleanup() { fs::remove_all(d); } } cleanup{dir};
    auto name = [&](int k) { return (dir / ("m.gguf-0000" + std::to_string(k) + "-of-00003.gguf")).string(); };
    write_shard(name(1), 0, 3, 5, {0, 1});
    write_shard(name(2), 1, 3, 5, {2, 3});
    write_shard(name(3), 2, 3, 5, {4});

    for (int open_from : {1, 2, 3}) {
        const GgufFile f = gguf_open(name(open_from));
        check(f.tensors.size() == 5, "all five tensors from three shards");
        check(f.str("general.architecture") == "test", "metadata comes from the first shard");
        check(f.path == name(1), "path is the first shard");
        for (int id = 0; id < 5; ++id) {
            const auto & t = f.tensors.at("t" + std::to_string(id));
            check(t.file == name(id < 2 ? 1 : id < 4 ? 2 : 3), "tensor records its shard");
            const auto raw = f.read(t);
            float v[4]; std::memcpy(v, raw.data(), sizeof v);
            check(v[0] == id * 10 && v[3] == id * 10 + 3, "tensor bytes read from its own shard");
        }
    }
    const GgufFile one = gguf_open_shard(name(2));
    check(one.tensors.size() == 2 && one.tensors.count("t2") && one.tensors.count("t3"), "gguf_open_shard keeps one file");

    fs::rename(name(3), name(3) + ".gone");
    check(throws(name(1)), "a missing shard is refused");
    fs::rename(name(3) + ".gone", name(3));

    write_shard(name(3), 2, 3, 5, {3});  // t3 again
    check(throws(name(1)), "a tensor in two shards is refused");
    write_shard(name(3), 2, 3, 6, {4});
    write_shard(name(1), 0, 3, 6, {0, 1});
    check(throws(name(1)), "a tensor count that differs from split.tensors.count is refused");
    write_shard(name(1), 0, 3, 5, {0, 1});
    write_shard(name(3), 1, 3, 5, {4});  // split.no 1 under the name of shard 3
    check(throws(name(1)), "a split.no that does not match the file name is refused");

    const auto odd = (dir / "odd.gguf").string();
    write_shard(odd, 0, 3, 5, {0, 1});
    check(throws(odd), "a split GGUF without the -NNNNN-of-NNNNN.gguf name is refused");

    std::printf("%s\n", failures ? "gguf-split: FAILED" : "gguf-split: PASS");
    return failures ? 1 : 0;
}

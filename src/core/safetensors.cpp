#include "safetensors.hpp"
#include "nlohmann/json.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tqllm {
namespace {

std::string slurp(const std::filesystem::path & p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + p.string());
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void index_shard(const std::filesystem::path & file, std::map<std::string, SafeTensor> & out) {
    std::ifstream in(file, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open shard " + file.string());
    uint64_t n = 0;
    in.read(reinterpret_cast<char *>(&n), 8);
    if (!in || n == 0 || n > (1ull << 30)) throw std::runtime_error("bad safetensors header: " + file.string());
    std::string header(n, '\0');
    in.read(header.data(), std::streamsize(n));
    if (!in) throw std::runtime_error("truncated safetensors header: " + file.string());
    in.seekg(0, std::ios::end);
    const uint64_t size = uint64_t(in.tellg()), base = 8 + n;

    const Json root = Parser(header).parse();
    for (const auto & [name, t] : root.obj) {
        if (name == "__metadata__") continue;
        const Json *dt = t.get("dtype"), *shape = t.get("shape"), *off = t.get("data_offsets");
        if (!dt || !shape || !off || off->arr.size() != 2) throw std::runtime_error("bad tensor entry: " + name);
        SafeTensor st;
        st.file = file.string();
        st.dtype = dt->str;
        for (const auto & d : shape->arr) st.shape.push_back(uint64_t(d.num));
        st.begin = base + uint64_t(off->arr[0].num);
        st.bytes = uint64_t(off->arr[1].num) - uint64_t(off->arr[0].num);
        if (st.begin + st.bytes > size) throw std::runtime_error("tensor out of shard bounds: " + name);
        if (!out.emplace(name, std::move(st)).second) throw std::runtime_error("duplicate tensor: " + name);
    }
}

} // namespace

std::map<std::string, SafeTensor> safetensors_index(const std::filesystem::path & dir) {
    std::map<std::string, SafeTensor> out;
    const auto index = dir / "model.safetensors.index.json";
    if (std::filesystem::exists(index)) {
        const Json root = Parser(slurp(index)).parse();
        const Json * map = root.get("weight_map");
        if (!map) throw std::runtime_error("index without weight_map: " + index.string());
        std::set<std::string> shards;
        for (const auto & [name, file] : map->obj) shards.insert(file.str);
        for (const auto & s : shards) index_shard(dir / s, out);
    } else {
        index_shard(dir / "model.safetensors", out);
    }
    return out;
}

std::vector<char> safetensors_read(const SafeTensor & t) {
    std::ifstream in(t.file, std::ios::binary);
    std::vector<char> buf(t.bytes);
    in.seekg(std::streamoff(t.begin));
    in.read(buf.data(), std::streamsize(t.bytes));
    if (!in) throw std::runtime_error("failed reading tensor from " + t.file);
    return buf;
}

} // namespace tqllm

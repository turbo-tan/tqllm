#include "gguf_mapped.hpp"
#include "dequant.hpp"
#include "hf_folder.hpp"
#include <map>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>
#include "platform/os_file.hpp"

namespace tqllm {
struct MappedGguf::Mapping {
    os::MappedFile file;
    const char * data;
    size_t size;
    const std::string & path;
    explicit Mapping(const std::string & p) : file(p), data(file.data()), size(file.size()), path(file.path()) {}
};
// A Hugging Face folder (src/core/hf): tensors stored as is are views of their shard; tensors built on read (MLX
// 4-bit -> Q4_1, BF16 dequants) are built once and kept; routed experts are built per expert and owned by the caller;
// the per-layer token embedding is gathered by row from the shards.
struct MappedGguf::Folder {
    std::mutex lock;
    std::map<std::string, std::unique_ptr<Mapping>> shards;
    std::map<std::string, std::vector<char>> built;
    const Mapping & shard(const std::string & f) {
        std::lock_guard<std::mutex> hold(lock);
        auto & m = shards[f]; if (!m) m = std::make_unique<Mapping>(f);
        return *m;
    }
    GgufBytes bytes(const std::string & name, const GgufTensor & t, const GgufFile & file) {
        if (t.built()) {
            if (name.empty()) throw std::runtime_error("hf: a tensor built on read has no stored range");
            std::lock_guard<std::mutex> hold(lock);
            auto it = built.find(name);
            if (it == built.end()) it = built.emplace(name, file.read(t)).first;
            return {it->second.data(), it->second.size()};
        }
        const Mapping & m = shard(t.file);
        if (t.offset > m.size || t.bytes > m.size - t.offset) throw std::runtime_error("hf: tensor " + name + " exceeds its shard");
        return {m.data + t.offset, size_t(t.bytes)};
    }
};

MappedGguf::MappedGguf(const std::string & path, GgufDialect dialect) {
    if (hf_folder(path)) { folder_ = std::make_unique<Folder>(); file_ = hf_open(path); return; }
    mapping_ = std::make_unique<Mapping>(path); file_ = gguf_open_shard(path, dialect);
}
MappedGguf::MappedGguf(const std::string & path, Raw) : mapping_(std::make_unique<Mapping>(path)) { file_.path = path; }
std::unique_ptr<MappedGguf> MappedGguf::raw(const std::string & path) {
    return std::unique_ptr<MappedGguf>(new MappedGguf(path, Raw{}));
}
MappedGguf::~MappedGguf() = default;
void MappedGguf::release_tensor(const std::string & name) const {
    if (folder_) return;  // a folder's shards stay mapped; built tensors stay built
    const auto v = tensor(name);
    mapping_->file.drop(size_t(v.data - mapping_->data), v.size);
}
size_t MappedGguf::warm_tensor(const std::string & name, int threads) const {
    if (folder_) {
        const auto & t = file_.tensors.at(name);
        if (t.warm) return t.warm(threads);
        if (t.built()) return 0;
        const GgufBytes v = folder_->bytes(name, t, file_);  // a stored range: touch its pages
        size_t sum = 0; const size_t page = os::page_size();
        for (size_t i = 0; i < v.size; i += page) sum += size_t(uint8_t(v.data[i]));
        (void) sum; return v.size;
    }
    const auto v = tensor(name); const uint64_t base = uint64_t(v.data - mapping_->data);
    const os::ReadFile file(mapping_->path);
    if (!file.ok()) return 0;
    constexpr size_t step = size_t(1) << 24; std::atomic<size_t> next{0}, done{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, threads); ++t) pool.emplace_back([&] {
        std::vector<char> buf(step);
        for (size_t at; (at = next.fetch_add(step)) < v.size;) {
            const int64_t r = file.read_at(buf.data(), std::min(step, v.size - at), base + at);
            if (r > 0) done += size_t(r);
        }
    });
    for (auto & t : pool) t.join();
    return done;
}
void MappedGguf::release_pages() const {
    if (folder_) return;
    mapping_->file.drop(0, mapping_->size);
}
bool MappedGguf::discard_pages() const noexcept { return folder_ ? true : discard_pages({mapping_->data,mapping_->size}); }
bool MappedGguf::discard_pages(GgufBytes bytes) const noexcept {
    if (folder_) return true;
    const auto base=reinterpret_cast<uintptr_t>(mapping_->data), at=reinterpret_cast<uintptr_t>(bytes.data);
    if(at<base || at-base>mapping_->size || bytes.size>mapping_->size-(at-base))return false;
    // shared boundary pages are kept (drop rounds inward): another tensor's view may be in use concurrently
    return mapping_->file.drop(at-base, bytes.size);
}
GgufBytes MappedGguf::checked(const GgufTensor & t) const {
    if (folder_) return folder_->bytes(std::string(), t, file_);  // a stored range (named tensors go through tensor())
    if (t.offset > mapping_->size || t.bytes > mapping_->size - t.offset)
        throw std::runtime_error("GGUF mapped view exceeds file");
    return {mapping_->data + t.offset, size_t(t.bytes)};
}
GgufBytes MappedGguf::tensor(const std::string & name) const {
    if (folder_) return folder_->bytes(name, file_.tensors.at(name), file_);
    return checked(file_.tensors.at(name));
}
GgufBytes MappedGguf::expert(const std::string & name, uint64_t index) const {
    if (folder_ && file_.tensors.at(name).make_slice) throw std::runtime_error("hf: " + name + " is built per expert (expert_owned)");
    return checked(gguf_expert_slice(file_.tensors.at(name), index, name));
}
std::pair<GgufBytes, std::shared_ptr<const void>> MappedGguf::expert_owned(const std::string & name, uint64_t index) const {
    const auto & t = file_.tensors.at(name);
    if (folder_ && t.make_slice) {
        auto buf = std::make_shared<std::vector<char>>(t.make_slice(index));
        if (buf->size() != t.bytes / t.ne.back()) throw std::runtime_error("hf: " + name + ": expert slice size");
        return {{buf->data(), buf->size()}, buf};
    }
    return {expert(name, index), nullptr};
}
std::vector<float> MappedGguf::gather_rows(const std::string & name, const std::vector<uint64_t> & rows) const {
    const auto & t = file_.tensors.at(name);
    if (t.gather) {
        if (t.ne.size() != 2) throw std::runtime_error("GGUF row gather: not a 2-D tensor");
        for (auto row : rows) if (row >= t.ne[1]) throw std::runtime_error("GGUF row gather: row out of range");
        std::vector<float> result(rows.size() * size_t(t.ne[0]));
        t.gather(rows, result.data());
        return result;
    }
    uint64_t block, bytes;
    if (t.ne.size() != 2 || !gguf_block_layout(t.type, block, bytes) || !t.ne[0] || t.ne[0] % block)
        throw std::runtime_error("GGUF row gather: not a row-aligned 2-D tensor");
    if (t.ne[0] > uint64_t(std::numeric_limits<int64_t>::max()) ||
        t.ne[0] > std::numeric_limits<size_t>::max() / sizeof(float) ||
        rows.size() > std::numeric_limits<size_t>::max() / sizeof(float) / t.ne[0])
        throw std::runtime_error("GGUF row gather: output size overflow");
    for (auto row : rows) if (row >= t.ne[1]) throw std::runtime_error("GGUF row gather: row out of range");
    const auto data = folder_ ? tensor(name) : checked(t);  // a folder's built tensor (token embedding): built once
    const size_t stride = size_t(t.ne[0] / block * bytes);
    std::vector<float> result(rows.size() * size_t(t.ne[0]));
    const auto part = [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i)
            dequantize_host(t.type, data.data + rows[i] * stride, result.data() + i * t.ne[0], int64_t(t.ne[0]));
    };
    // prefill chunks gather many scattered rows (per-layer embeddings: 16 per token) from the mapped file: split them;
    // a several-sequence decode step (np 8: ~512 per-layer rows) too, in parts of at least 64 rows (each row dequantises
    // the same way in any part, so the result does not change)
    const size_t cores = std::max(1u, std::thread::hardware_concurrency());
    const size_t threads = rows.size() >= 4096 ? std::min<size_t>(16, cores) : std::min<size_t>({8, cores, rows.size() / 64});
    if (threads <= 1) { part(0, rows.size()); return result; }
    std::vector<std::thread> pool;
    const size_t step = (rows.size() + threads - 1) / threads;
    for (size_t i0 = 0; i0 < rows.size(); i0 += step) pool.emplace_back(part, i0, std::min(rows.size(), i0 + step));
    for (auto & th : pool) th.join();
    return result;
}
} // namespace tqllm

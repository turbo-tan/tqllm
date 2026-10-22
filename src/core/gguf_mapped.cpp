#include "gguf_mapped.hpp"
#include "dequant.hpp"
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tqllm {
struct MappedGguf::Mapping {
    const char * data = nullptr;
    size_t size = 0;
    std::string path;
    explicit Mapping(const std::string & p) : path(p) {
#ifndef _WIN32
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("GGUF map open: " + std::string(std::strerror(errno)));
        struct stat info{};
        if (fstat(fd, &info) || info.st_size <= 0 || uint64_t(info.st_size) > std::numeric_limits<size_t>::max()) {
            close(fd); throw std::runtime_error("GGUF map: invalid file size");
        }
        size = size_t(info.st_size);
        void * base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        const int error = errno;
        close(fd);
        if (base == MAP_FAILED) throw std::runtime_error("GGUF mmap: " + std::string(std::strerror(error)));
        data = static_cast<const char *>(base);
#else
        throw std::runtime_error("GGUF mapping: use Linux/WSL or macOS; native Windows mapping is not implemented");
#endif
    }
    ~Mapping() {
#ifndef _WIN32
        if (data) munmap(const_cast<char *>(data), size);
#endif
    }
};
MappedGguf::MappedGguf(const std::string & path, GgufDialect dialect)
    : mapping_(std::make_unique<Mapping>(path)), file_(gguf_open(path, dialect)) {}
MappedGguf::~MappedGguf() = default;
void MappedGguf::release_pages() const {
#ifndef _WIN32
    madvise(const_cast<char *>(mapping_->data), mapping_->size, MADV_DONTNEED);
    const int fd = open(mapping_->path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
#endif
}
GgufBytes MappedGguf::checked(const GgufTensor & t) const {
    if (t.offset > mapping_->size || t.bytes > mapping_->size - t.offset)
        throw std::runtime_error("GGUF mapped view exceeds file");
    return {mapping_->data + t.offset, size_t(t.bytes)};
}
GgufBytes MappedGguf::tensor(const std::string & name) const { return checked(file_.tensors.at(name)); }
GgufBytes MappedGguf::expert(const std::string & name, uint64_t index) const {
    return checked(gguf_expert_slice(file_.tensors.at(name), index, name));
}
std::vector<float> MappedGguf::gather_rows(const std::string & name, const std::vector<uint64_t> & rows) const {
    const auto & t = file_.tensors.at(name);
    uint64_t block, bytes;
    if (t.ne.size() != 2 || !gguf_block_layout(t.type, block, bytes) || !t.ne[0] || t.ne[0] % block)
        throw std::runtime_error("GGUF row gather: not a row-aligned 2-D tensor");
    if (t.ne[0] > uint64_t(std::numeric_limits<int64_t>::max()) ||
        t.ne[0] > std::numeric_limits<size_t>::max() / sizeof(float) ||
        rows.size() > std::numeric_limits<size_t>::max() / sizeof(float) / t.ne[0])
        throw std::runtime_error("GGUF row gather: output size overflow");
    for (auto row : rows) if (row >= t.ne[1]) throw std::runtime_error("GGUF row gather: row out of range");
    const auto data = checked(t);
    const size_t stride = size_t(t.ne[0] / block * bytes);
    std::vector<float> result(rows.size() * size_t(t.ne[0]));
    const auto part = [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i)
            dequantize_host(t.type, data.data + rows[i] * stride, result.data() + i * t.ne[0], int64_t(t.ne[0]));
    };
    // prefill chunks gather many scattered rows (per-layer embeddings: 16 per token) from the mapped file: split them
    const size_t threads = rows.size() >= 4096 ? std::min<size_t>(16, std::max(1u, std::thread::hardware_concurrency())) : 1;
    if (threads <= 1) { part(0, rows.size()); return result; }
    std::vector<std::thread> pool;
    const size_t step = (rows.size() + threads - 1) / threads;
    for (size_t i0 = 0; i0 < rows.size(); i0 += step) pool.emplace_back(part, i0, std::min(rows.size(), i0 + step));
    for (auto & th : pool) th.join();
    return result;
}
} // namespace tqllm

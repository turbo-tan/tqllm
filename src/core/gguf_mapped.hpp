#pragma once
#include "gguf.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tqllm {
struct GgufBytes { const char * data = nullptr; size_t size = 0; };

// Views remain valid until their owning mapping is destroyed. The tensor bytes are immutable.
class MappedGguf {
    struct Mapping;
    std::unique_ptr<Mapping> mapping_;
    GgufFile file_;
    GgufBytes checked(const GgufTensor &) const;
public:
    explicit MappedGguf(const std::string & path, GgufDialect dialect = GgufDialect::Native);
    ~MappedGguf();
    MappedGguf(const MappedGguf &) = delete;
    MappedGguf & operator=(const MappedGguf &) = delete;
    const GgufFile & file() const { return file_; }
    GgufBytes tensor(const std::string & name) const;
    GgufBytes expert(const std::string & name, uint64_t index) const;
    // Drop this file's pages from memory; views stay valid and fault back in on access.
    void release_pages() const;
    // Drop one tensor's pages (whole pages inside it); its view stays valid and faults back in on access.
    void release_tensor(const std::string & name) const;
    // Read one tensor once so its pages are cached; returns the bytes read.
    size_t warm_tensor(const std::string & name, int threads) const;
    // Gather only requested rows into a small F32 buffer, preserving order and duplicates.
    std::vector<float> gather_rows(const std::string & name, const std::vector<uint64_t> & rows) const;
};
} // namespace tqllm

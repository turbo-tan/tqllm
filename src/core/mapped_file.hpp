#pragma once
#include <cstddef>
#include <string>

namespace tqllm {
class MappedFile {
public:
    explicit MappedFile(const std::string & path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    const unsigned char * at(size_t offset, size_t bytes) const;
private:
    void * data_ = nullptr;
    size_t size_ = 0;
};
}

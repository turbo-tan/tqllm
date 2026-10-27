#include "mapped_file.hpp"
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace tqllm {
MappedFile::MappedFile(const std::string & path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open model: " + path);
    struct stat st{};
    if (fstat(fd, &st) || st.st_size <= 0 || uint64_t(st.st_size) > SIZE_MAX) {
        close(fd); throw std::runtime_error("invalid model file size");
    }
    size_ = size_t(st.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    int error = errno;
    close(fd);
    if (data_ == MAP_FAILED) { data_ = nullptr; throw std::runtime_error(std::strerror(error)); }
}
MappedFile::~MappedFile() { if (data_) munmap(data_, size_); }
const unsigned char * MappedFile::at(size_t offset, size_t bytes) const {
    if (offset > size_ || bytes > size_ - offset) throw std::out_of_range("model mapping bounds");
    return static_cast<const unsigned char *>(data_) + offset;
}
}

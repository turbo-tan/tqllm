#include "os_file.hpp"

#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <crt_externs.h>
#endif

namespace tqllm::os {

size_t page_size() {
    const long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? size_t(p) : 4096;
}
size_t physical_ram_bytes() { return size_t(sysconf(_SC_PHYS_PAGES)) * page_size(); }

struct MappedFile::Impl { int fd = -1; };

MappedFile::MappedFile(const std::string & path) : path_(path), impl_(std::make_unique<Impl>()) {
    impl_->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl_->fd < 0) throw std::runtime_error("map open " + path + ": " + std::string(std::strerror(errno)));
    struct stat info{};
    if (fstat(impl_->fd, &info) || info.st_size <= 0 || uint64_t(info.st_size) > std::numeric_limits<size_t>::max()) {
        close(impl_->fd);
        throw std::runtime_error("map " + path + ": invalid file size");
    }
    size_ = size_t(info.st_size);
    void * base = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, impl_->fd, 0);
    const int error = errno;
    if (base == MAP_FAILED) {
        close(impl_->fd);
        throw std::runtime_error("mmap " + path + ": " + std::string(std::strerror(error)));
    }
    data_ = static_cast<const char *>(base);
}
MappedFile::~MappedFile() {
    if (data_) munmap(const_cast<char *>(data_), size_);
    if (impl_ && impl_->fd >= 0) close(impl_->fd);
}
bool MappedFile::drop(size_t offset, size_t bytes) const noexcept {
    if (offset > size_ || bytes > size_ - offset) return false;
    if (!bytes) return true;
    const size_t page = page_size();
    const size_t start = (offset + page - 1) / page * page, stop = (offset + bytes) / page * page;
    if (stop <= start) return true;
    if (madvise(const_cast<char *>(data_) + start, stop - start, MADV_DONTNEED)) return false;
#if defined(__linux__)
    return posix_fadvise(impl_->fd, off_t(start), off_t(stop - start), POSIX_FADV_DONTNEED) == 0;
#else
    return true;  // macOS: no posix_fadvise; madvise above already drops the mapped pages
#endif
}
void MappedFile::advise_random() const noexcept { madvise(const_cast<char *>(data_), size_, MADV_RANDOM); }

struct ReadFile::Impl { int fd = -1; };
ReadFile::ReadFile(const std::string & path) : impl_(std::make_unique<Impl>()) {
    impl_->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
}
ReadFile::~ReadFile() { if (impl_->fd >= 0) close(impl_->fd); }
bool ReadFile::ok() const { return impl_->fd >= 0; }
int64_t ReadFile::read_at(void * dst, size_t bytes, uint64_t offset) const {
    return int64_t(pread(impl_->fd, dst, bytes, off_t(offset)));
}
bool ReadFile::drop_cache(uint64_t offset, uint64_t bytes) const noexcept {
#ifdef POSIX_FADV_DONTNEED
    return impl_->fd >= 0 && posix_fadvise(impl_->fd, off_t(offset), off_t(bytes), POSIX_FADV_DONTNEED) == 0;
#else
    (void) offset; (void) bytes; return false;
#endif
}

void * load_library(const std::string & path) { return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL); }
void * library_symbol(void * library, const char * name) { return dlsym(library, name); }
void close_library(void * library) { if (library) dlclose(library); }
std::string library_error() { const char * e = dlerror(); return e ? e : "unknown error"; }

int process_id() { return int(getpid()); }
char ** environment() {
#if defined(__APPLE__)
    return *_NSGetEnviron();
#else
    return environ;
#endif
}

}  // namespace tqllm::os

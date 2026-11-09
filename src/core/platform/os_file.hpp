#pragma once
// Operating-system file and library primitives, one implementation per OS: os_posix.cpp (Linux, macOS) and
// os_win32.cpp (native Windows). Code outside src/core/platform never includes <sys/mman.h>, <unistd.h>, <fcntl.h>,
// <dlfcn.h> or <windows.h> for these.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tqllm::os {

size_t page_size();
size_t physical_ram_bytes();

// A whole file mapped read-only (copy-on-write private mapping on POSIX, a read-only view on Windows).
class MappedFile {
public:
    explicit MappedFile(const std::string & path);  // throws std::runtime_error with the OS reason
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    const char * data() const { return data_; }
    size_t size() const { return size_; }
    const std::string & path() const { return path_; }
    // Ask the OS to drop the cached pages of [offset, offset + bytes) (page-aligned inward). Advisory: false when
    // the OS refused or cannot do it; the mapping stays valid either way and reads fault the pages back in.
    bool drop(size_t offset, size_t bytes) const noexcept;
    // Random-access hint for the whole mapping (no-op where unsupported).
    void advise_random() const noexcept;

private:
    std::string path_;
    const char * data_ = nullptr;
    size_t size_ = 0;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A file opened read-only for positioned reads (pread / overlapped ReadFile); safe from several threads at once.
class ReadFile {
public:
    explicit ReadFile(const std::string & path);  // check ok()
    ~ReadFile();
    ReadFile(const ReadFile &) = delete;
    ReadFile & operator=(const ReadFile &) = delete;
    bool ok() const;
    // bytes read (short at end of file), or -1 on error
    int64_t read_at(void * dst, size_t bytes, uint64_t offset) const;
    // drop the OS page cache for [offset, offset + bytes) (0 bytes: the whole file); false where unsupported
    bool drop_cache(uint64_t offset, uint64_t bytes) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Shared libraries (plugins, optional runtimes).
void * load_library(const std::string & path);  // nullptr on failure, reason in library_error()
void * library_symbol(void * library, const char * name);
void close_library(void * library);
std::string library_error();

int process_id();
// the process environment as NAME=value strings (environ / _NSGetEnviron / _environ)
char ** environment();

}  // namespace tqllm::os

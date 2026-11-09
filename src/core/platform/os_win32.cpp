#include "os_file.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <process.h>
#include <stdlib.h>
#include <stdexcept>

namespace tqllm::os {
namespace {
std::wstring wide(const std::string & s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
    return w;
}
std::string message(DWORD code) {
    char * text = nullptr;
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                   nullptr, code, 0, reinterpret_cast<char *>(&text), 0, nullptr);
    std::string r = n ? std::string(text, n) : "error " + std::to_string(code);
    if (text) LocalFree(text);
    while (!r.empty() && (r.back() == '\n' || r.back() == '\r' || r.back() == ' ')) r.pop_back();
    return r;
}
HANDLE open_read(const std::string & path) {
    return CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
}
DWORD g_library_error = 0;
}  // namespace

size_t page_size() {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return size_t(info.dwPageSize);
}
size_t physical_ram_bytes() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? size_t(m.ullTotalPhys) : 0;
}

struct MappedFile::Impl { HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr; };

MappedFile::MappedFile(const std::string & path) : path_(path), impl_(std::make_unique<Impl>()) {
    impl_->file = open_read(path);
    if (impl_->file == INVALID_HANDLE_VALUE) throw std::runtime_error("map open " + path + ": " + message(GetLastError()));
    LARGE_INTEGER bytes{};
    if (!GetFileSizeEx(impl_->file, &bytes) || bytes.QuadPart <= 0) {
        CloseHandle(impl_->file);
        throw std::runtime_error("map " + path + ": invalid file size");
    }
    size_ = size_t(bytes.QuadPart);
    impl_->mapping = CreateFileMappingW(impl_->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!impl_->mapping) {
        const DWORD e = GetLastError();
        CloseHandle(impl_->file);
        throw std::runtime_error("map " + path + ": " + message(e));
    }
    data_ = static_cast<const char *>(MapViewOfFile(impl_->mapping, FILE_MAP_READ, 0, 0, 0));
    if (!data_) {
        const DWORD e = GetLastError();
        CloseHandle(impl_->mapping);
        CloseHandle(impl_->file);
        throw std::runtime_error("map view " + path + ": " + message(e));
    }
}
MappedFile::~MappedFile() {
    if (data_) UnmapViewOfFile(data_);
    if (impl_ && impl_->mapping) CloseHandle(impl_->mapping);
    if (impl_ && impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
}
bool MappedFile::drop(size_t offset, size_t bytes) const noexcept {
    if (offset > size_ || bytes > size_ - offset) return false;
    if (!bytes) return true;
    const size_t page = page_size();
    const size_t start = (offset + page - 1) / page * page, stop = (offset + bytes) / page * page;
    if (stop <= start) return true;
    // a read-only file view: unlocking drops the pages from this process's working set (the standby list keeps them
    // cached until the memory is needed, like Linux's page cache after MADV_DONTNEED)
    VirtualUnlock(const_cast<char *>(data_) + start, stop - start);
    return true;
}
void MappedFile::advise_random() const noexcept {}

struct ReadFile::Impl { HANDLE file = INVALID_HANDLE_VALUE; };
ReadFile::ReadFile(const std::string & path) : impl_(std::make_unique<Impl>()) { impl_->file = open_read(path); }
ReadFile::~ReadFile() { if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file); }
bool ReadFile::ok() const { return impl_->file != INVALID_HANDLE_VALUE; }
int64_t ReadFile::read_at(void * dst, size_t bytes, uint64_t offset) const {
    char * out = static_cast<char *>(dst);
    int64_t total = 0;
    while (bytes) {  // ReadFile takes a DWORD count: read in chunks; OVERLAPPED carries the offset (thread-safe)
        const DWORD want = DWORD(bytes > (size_t(1) << 30) ? (size_t(1) << 30) : bytes);
        OVERLAPPED at{};
        at.Offset = DWORD(offset & 0xffffffffu);
        at.OffsetHigh = DWORD(offset >> 32);
        DWORD got = 0;
        if (!::ReadFile(impl_->file, out, want, &got, &at)) return GetLastError() == ERROR_HANDLE_EOF ? total : -1;
        if (!got) break;
        total += got; out += got; bytes -= got; offset += got;
    }
    return total;
}
bool ReadFile::drop_cache(uint64_t, uint64_t) const noexcept { return false; }

void * load_library(const std::string & path) {
    HMODULE h = LoadLibraryW(wide(path).c_str());
    g_library_error = h ? 0 : GetLastError();
    return reinterpret_cast<void *>(h);
}
void * library_symbol(void * library, const char * name) {
    FARPROC f = GetProcAddress(reinterpret_cast<HMODULE>(library), name);
    g_library_error = f ? 0 : GetLastError();
    return reinterpret_cast<void *>(f);
}
void close_library(void * library) { if (library) FreeLibrary(reinterpret_cast<HMODULE>(library)); }
std::string library_error() { return g_library_error ? message(g_library_error) : "unknown error"; }

int process_id() { return _getpid(); }
char ** environment() { return _environ; }

}  // namespace tqllm::os

#pragma once
// Start-up memory guard for GPUs that share memory with the CPU (GB10, Jetson-class): a model loaded while another model
// process holds most of that memory exhausts it, and a process blocked in the GPU driver cannot be killed, so the box
// hangs until the watchdog panics (.44, 11-08). tqserve refuses to load instead. Pure functions: no CUDA, testable.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace tqllm {

// Bytes of the files a model path loads: a file (a split GGUF counts all its shards), a folder (every regular file), or
// a Flash-Next .tqfn manifest (the gguf= / mtp= / experts= files it names, relative to the manifest).
inline uint64_t model_files_bytes(const std::string & path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (path.empty()) return 0;
    const fs::path p(path);
    if (fs::is_directory(p, ec)) {
        uint64_t sum = 0;
        for (const auto & e : fs::directory_iterator(p, ec))
            if (e.is_regular_file(ec)) sum += e.file_size(ec);
        return sum;
    }
    if (p.extension() == ".tqfn") {
        std::ifstream in(p);
        uint64_t sum = 0;
        for (std::string line; std::getline(in, line);) {
            const size_t eq = line.find('=');
            if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
            const std::string key = line.substr(0, eq);
            if (key != "gguf" && key != "mtp" && key != "experts") continue;
            const fs::path f = p.parent_path() / line.substr(eq + 1);
            if (fs::is_regular_file(f, ec)) sum += fs::file_size(f, ec);
        }
        return sum;
    }
    // split GGUF: NAME-00001-of-0000N.gguf -> every NAME-*-of-0000N.gguf beside it
    const std::string name = p.filename().string();
    const size_t of = name.rfind("-of-");
    if (p.extension() == ".gguf" && of != std::string::npos && of >= 6 && name[of - 6] == '-') {
        const std::string prefix = name.substr(0, of - 5), suffix = name.substr(of);
        uint64_t sum = 0;
        for (const auto & e : fs::directory_iterator(p.has_parent_path() ? p.parent_path() : fs::path("."), ec)) {
            const std::string n = e.path().filename().string();
            if (n.size() == name.size() && n.compare(0, prefix.size(), prefix) == 0 &&
                n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0 && e.is_regular_file(ec))
                sum += e.file_size(ec);
        }
        if (sum) return sum;
    }
    return fs::is_regular_file(p, ec) ? uint64_t(fs::file_size(p, ec)) : 0;
}

// Empty when the load fits; otherwise the reason to refuse. `available`: memory the OS can hand out (MemAvailable,
// page cache included); `need`: the model's files. A load that fits only without the margin is allowed (the engine's
// own fit trims buffers), but only `available >= need` is a real chance of loading at all.
inline std::string unified_memory_refusal(uint64_t available, uint64_t need) {
    if (!need || available >= need) return {};
    const auto gb = [](uint64_t b) { return std::to_string((b + (uint64_t(1) << 29)) >> 30); };
    return "not enough free memory on this unified-memory GPU: the model's files are " + gb(need) + " GB but only " +
           gb(available) + " GB is available. Another model process is probably running (one model per GB10): stop it "
           "first. Loading anyway would exhaust the shared memory and hang the machine.";
}

}  // namespace tqllm

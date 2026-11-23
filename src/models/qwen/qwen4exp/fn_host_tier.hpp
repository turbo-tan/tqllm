#pragma once
// Flash-Next's host tier loader (experts read from the GGUF on demand): one expert's three slices read with pread (their
// pages dropped from the page cache after the read: the tier holds the bytes, so the cache keeps room for the embedding
// rows and dense weights), repacked into the banks' row layout (tiled like the pinned pack).
#include "fn_checkpoint.hpp"
#include "host_expert_tier.hpp"
#include "packed_experts.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace tqllm::flashnext {

class TierLoader {
public:
    TierLoader(const Checkpoint & cp, bool tile) : cp_(cp), tile_(tile) {}
    ~TierLoader() { for (auto & [p, fd] : fds_) ::close(fd); }
    TierLoader(const TierLoader &) = delete;
    TierLoader & operator=(const TierLoader &) = delete;
    // one expert's bank-layout bytes (the six parts back to back), as repack_expert makes them
    std::array<std::vector<char>, 6> parts(int layer, int expert) {
        static const std::array<const char *, 3> suffix{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        std::array<ExpertPayload, 3> payload; std::array<std::vector<char>, 3> raw;
        const auto mapped = cp_.expert_payload(layer, expert);  // types and, for a folder, the bytes
        for (int p = 0; p < 3; ++p) {
            const std::string name = "blk." + std::to_string(layer) + "." + suffix[size_t(p)];
            std::string path; uint64_t offset = 0; size_t bytes = 0;
            payload[size_t(p)] = mapped[size_t(p)];
            if (!cp_.file(name).expert_range(name, uint64_t(expert), path, offset, bytes)) continue;  // read through the mapping
            raw[size_t(p)].resize(bytes);
            const int fd = open_file(path);
            for (size_t done = 0; done < bytes;) {
                const ssize_t r = ::pread(fd, raw[size_t(p)].data() + done, bytes - done, off_t(offset + done));
                if (r <= 0) throw std::runtime_error("host tier: short read of " + path);
                done += size_t(r);
            }
            ::posix_fadvise(fd, off_t(offset), off_t(bytes), POSIX_FADV_DONTNEED);
            payload[size_t(p)].bytes = {raw[size_t(p)].data(), bytes};
        }
        return repack_expert(payload, cp_.config.hidden, cp_.config.expert_ff, tile_);
    }
    void load(int layer, int expert, char * dst) {
        size_t at = 0;
        for (const auto & part : parts(layer, expert)) { std::memcpy(dst + at, part.data(), part.size()); at += part.size(); }
    }

private:
    int open_file(const std::string & path) {
        std::lock_guard<std::mutex> hold(lock_);
        auto it = fds_.find(path);
        if (it != fds_.end()) return it->second;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("host tier: cannot open " + path);
        fds_.emplace(path, fd); return fd;
    }
    const Checkpoint & cp_;
    bool tile_;
    std::mutex lock_;
    std::map<std::string, int> fds_;
};

}  // namespace tqllm::flashnext

#pragma once
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <map>
#include <memory>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include "xxh64.hpp"

namespace tqllm {
// Immutable Q2 records; identity binds the encoder revision, source files and exact precision map.
class ColdExpertStore {
    std::filesystem::path root_;
    size_t piece_;
public:
    struct Stats { uint64_t loads=0, bytes=0, read_ns=0, checksum_ns=0, mapped_ns=0; };
    struct Piece {
        const char * p;size_t n;
        const char * data() const {return p;}
        size_t size() const {return n;}
    };
    struct View {std::array<Piece,3> parts;std::shared_ptr<const void> owner;};
    Stats stats() const { std::lock_guard<std::mutex> lock(stats_mutex_); return stats_; }
    void profile(bool on) { profile_ = on; }
private:
    bool profile_ = false, fast_ = false;
    template<class P> static uint64_t fast_hash(const std::array<P,3> & parts) {
        uint64_t h=0;
        for (const auto & p : parts) h=xxh64(p.data(),p.size(),h);
        return h;
    }
    uint64_t fast_expected(int layer,int expert,uint64_t original) const {
        const auto p=std::filesystem::path(path(layer,expert).string()+".xxh64");
        if (std::filesystem::file_size(p)!=48) throw std::runtime_error("cold store: fast checksum size mismatch");
        std::array<uint64_t,6> h{};std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char *>(h.data()),48);
        if (!f || h[0]!=0x3148363458484354ull || h[1]!=uint64_t(layer) || h[2]!=uint64_t(expert) || h[3]!=piece_ || h[4]!=original)
            throw std::runtime_error("cold store: fast checksum identity mismatch");
        return h[5];
    }

    struct MappedRecord {std::shared_ptr<const void> owner;dev_t dev;ino_t ino;int64_t mtime;};
    // a record rewritten in place keeps its inode; a MAP_PRIVATE mapping need not show the new bytes (macOS does not)
    static int64_t mtime_ns(const struct stat & s) {
#ifdef __APPLE__
        return int64_t(s.st_mtimespec.tv_sec)*1000000000+s.st_mtimespec.tv_nsec;
#else
        return int64_t(s.st_mtim.tv_sec)*1000000000+s.st_mtim.tv_nsec;
#endif
    }
    mutable std::map<std::pair<int,int>,MappedRecord> mapped_;
    mutable std::mutex mapped_mutex_;
    mutable Stats stats_;
    mutable std::mutex stats_mutex_;

    static std::string read(const std::filesystem::path & p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) throw std::runtime_error("cold store: cannot read " + p.string());
        return {std::istreambuf_iterator<char>(f), {}};
    }
    template<class P> static uint64_t checksum(const std::array<P,3> & parts) {
        uint64_t h = 14695981039346656037ull;
        for (const auto & p : parts) for(size_t i=0;i<p.size();++i) h = (h ^ static_cast<unsigned char>(p.data()[i])) * 1099511628211ull;
        return h;
    }
public:
    void fast_verification(bool enabled) {
        if (enabled && read(root_/"xxh64-complete")!="XXH64_1\n") throw std::runtime_error("cold store: fast verification incomplete");
        fast_=enabled;
    }
    bool contains_fast(int layer,int expert) const { return std::filesystem::exists(path(layer,expert).string()+".xxh64"); }
    void prepare_fast(int layer,int expert) {
        if (fast_) throw std::runtime_error("cold store: sidecar generation requires original FNV verification");
        const auto parts=load(layer,expert);
        const std::array<uint64_t,6> h{0x3148363458484354ull,uint64_t(layer),uint64_t(expert),piece_,checksum(parts),fast_hash(parts)};
        const auto final=std::filesystem::path(path(layer,expert).string()+".xxh64"), tmp=std::filesystem::path(final.string()+".partial");
        std::ofstream f(tmp,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char *>(h.data()),48);f.close();
        if (!f) throw std::runtime_error("cold store: fast checksum write failed");
        std::filesystem::rename(tmp,final);
    }
    void finish_fast() {
        std::ofstream f(root_/"xxh64-complete.partial");f<<"XXH64_1\n";f.close();
        if (!f) throw std::runtime_error("cold store: fast marker write failed");
        std::filesystem::rename(root_/"xxh64-complete.partial",root_/"xxh64-complete");
    }
    static std::string identity(const std::vector<std::string> & sources, const std::vector<int32_t> & map, size_t piece) {
        std::string s = "GSQ_Q2_0_ENCODER_1\npiece=" + std::to_string(piece) + "\n";
        for (const auto & p : sources) {
            const auto full = std::filesystem::canonical(p);
            s += full.string() + "\n" + std::to_string(std::filesystem::file_size(full)) + "\n" +
                 std::to_string(static_cast<long long>(std::filesystem::last_write_time(full).time_since_epoch().count())) + "\n";
        }
        s.append(reinterpret_cast<const char *>(map.data()), map.size()*sizeof(int32_t));
        return s;
    }
    ColdExpertStore(const std::string & root, size_t piece, const std::string & identity, bool create = false)
        : root_(root), piece_(piece) {
        if (!piece || piece % 18 || piece>(SIZE_MAX-32)/3) throw std::invalid_argument("cold store: invalid piece size");
        if (create) {
            std::filesystem::create_directories(root_);
            if (!std::filesystem::exists(root_/"identity")) {
                // Refuse to bind orphan records to a new identity.
                if (!std::filesystem::is_empty(root_)) throw std::runtime_error("cold store: missing identity in nonempty directory");
                std::ofstream f(root_/"identity", std::ios::binary); f.write(identity.data(), identity.size());
                f.close(); if (!f) throw std::runtime_error("cold store: identity write failed");
            }
        }
        if (read(root_/"identity") != identity) throw std::runtime_error("cold store: source/map identity mismatch");
        if (!create && read(root_/"complete") != "COMPLETE\n") throw std::runtime_error("cold store: incomplete conversion");
    }
    std::filesystem::path path(int layer, int expert) const {
        if (layer < 0 || expert < 0) throw std::invalid_argument("cold store: invalid expert");
        return root_/(std::to_string(layer)+"-"+std::to_string(expert)+".q2");
    }
    bool contains(int layer, int expert) const { return std::filesystem::exists(path(layer,expert)); }
    std::array<std::vector<char>,3> load(int layer, int expert) const {
        const auto start = profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto p = path(layer,expert);
        if (std::filesystem::file_size(p) != 32 + 3*piece_) throw std::runtime_error("cold store: record size mismatch");
        std::ifstream f(p,std::ios::binary); std::array<uint64_t,4> h{};
        f.read(reinterpret_cast<char *>(h.data()),32);
        if (h[0] != 0x315132444c4f4354ull || h[1] != uint64_t(layer) || h[2] != uint64_t(expert))
            throw std::runtime_error("cold store: record identity mismatch");
        std::array<std::vector<char>,3> parts;
        for (auto & v : parts) { v.resize(piece_); f.read(v.data(),v.size()); }
        const auto read_end = profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!f || (fast_ ? fast_expected(layer,expert,h[3]) != fast_hash(parts) : h[3] != checksum(parts))) throw std::runtime_error("cold store: record checksum mismatch");
        if (profile_) {
            const auto checksum_end=std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.loads; stats_.bytes += 3*piece_;
            stats_.read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(read_end-start).count();
            stats_.checksum_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(checksum_end-read_end).count();
        }
        return parts;
    }
    View load_mapped(int layer,int expert) const {
        const auto start=profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto p=path(layer,expert);const size_t bytes=32+3*piece_;
        struct stat info{};
        if(::stat(p.c_str(),&info) || uint64_t(info.st_size)!=bytes)throw std::runtime_error("cold store: mapped record size mismatch");
        std::shared_ptr<const void> owner;
        {
            std::lock_guard<std::mutex> lock(mapped_mutex_);
            auto it=mapped_.find({layer,expert});
            if(it!=mapped_.end() && it->second.dev==info.st_dev && it->second.ino==info.st_ino && it->second.mtime==mtime_ns(info))owner=it->second.owner;
            else {
                const int fd=::open(p.c_str(),O_RDONLY|O_CLOEXEC);
                if(fd<0)throw std::runtime_error("cold store: mapped record open failed");
                struct stat actual{};
                if(fstat(fd,&actual) || uint64_t(actual.st_size)!=bytes) {::close(fd);throw std::runtime_error("cold store: mapped record changed");}
                void * data=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);::close(fd);
                if(data==MAP_FAILED)throw std::runtime_error("cold store: mmap failed");
                owner=std::shared_ptr<const void>(data,[bytes](const void * v){munmap(const_cast<void *>(v),bytes);});
                mapped_[{layer,expert}]={owner,actual.st_dev,actual.st_ino,mtime_ns(actual)};
            }
        }
        const auto * data=static_cast<const char *>(owner.get());std::array<uint64_t,4> h{};
        std::memcpy(h.data(),data,32);
        if(h[0]!=0x315132444c4f4354ull || h[1]!=uint64_t(layer) || h[2]!=uint64_t(expert))throw std::runtime_error("cold store: mapped record identity mismatch");
        View view{{Piece{data+32,piece_},Piece{data+32+piece_,piece_},Piece{data+32+2*piece_,piece_}},owner};
        if(fast_ ? fast_expected(layer,expert,h[3])!=fast_hash(view.parts) : h[3]!=checksum(view.parts))throw std::runtime_error("cold store: mapped record checksum mismatch");
        if(profile_) {
            const auto end=std::chrono::steady_clock::now();std::lock_guard<std::mutex> lock(stats_mutex_);
            ++stats_.loads;stats_.bytes+=3*piece_;
            // Faults and verification both occur while hashing mapped bytes.
            stats_.mapped_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        }
        return view;
    }
    void save(int layer, int expert, const std::array<std::vector<char>,3> & parts) const {
        for (const auto & v : parts) if (v.size()!=piece_) throw std::invalid_argument("cold store: projection size mismatch");
        const auto final = path(layer,expert), tmp = std::filesystem::path(final.string()+".partial");
        const std::array<uint64_t,4> h{0x315132444c4f4354ull,uint64_t(layer),uint64_t(expert),checksum(parts)};
        std::ofstream f(tmp,std::ios::binary|std::ios::trunc); f.write(reinterpret_cast<const char *>(h.data()),32);
        for (const auto & v : parts) f.write(v.data(),v.size());
        f.close(); if (!f) throw std::runtime_error("cold store: record write failed");
        std::filesystem::rename(tmp,final);
    }
    void finish() const {
        std::ofstream f(root_/"complete.partial"); f << "COMPLETE\n"; f.close();
        if (!f) throw std::runtime_error("cold store: completion write failed");
        std::filesystem::rename(root_/"complete.partial",root_/"complete");
    }
};
} // namespace tqllm

#pragma once
// Packed expert file (`TQFNEXP1`): the header, then every expert of every layer as one contiguous block in the decode
// cache's row layout (gate data / gate aux / up data / up aux / down data / down aux), so a MixedExpertCache miss is a
// copy instead of a loader repack. Written from any model's expert payloads (Flash-Next, GLM-5.3-Flash); `TQFNEXPT` is the
// same file with the code parts permuted into the grouped kernels' lane-contiguous tiles (values unchanged).
#include "quant_kernels.cuh"
#include "tq3_expert_cache.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {

namespace packed_detail {
inline void need(bool ok, const std::string & why) { if (!ok) throw std::runtime_error("packed experts: " + why); }
}  // namespace packed_detail

struct PackedExperts {
    int layers = 0, experts = 0; std::vector<std::array<uint64_t, 6>> sizes; std::vector<uint64_t> base, block;
    const char * map = nullptr; size_t bytes = 0; bool tiled = false;
    const char * at(int layer, int id) const { return map + base[layer] + uint64_t(id) * block[layer]; }
};

// Maps the whole file read-only; the caller unmaps `map` (as the engines do: they may replace it with a pinned copy).
inline PackedExperts open_packed(const std::string & path) {
    PackedExperts pe;
    FILE * f = std::fopen(path.c_str(), "rb");
    packed_detail::need(f != nullptr, "cannot open " + path);
    char magic[8];
    int32_t dims[2];
    packed_detail::need(std::fread(magic, 1, 8, f) == 8 &&
                            (std::memcmp(magic, "TQFNEXP1", 8) == 0 || std::memcmp(magic, "TQFNEXPT", 8) == 0) &&
                            std::fread(dims, 4, 2, f) == 2, "not a TQFNEXP1 / TQFNEXPT file");
    pe.tiled = std::memcmp(magic, "TQFNEXPT", 8) == 0;
    pe.layers = dims[0];
    pe.experts = dims[1];
    pe.sizes.resize(size_t(pe.layers));
    for (auto & z : pe.sizes) packed_detail::need(std::fread(z.data(), 8, 6, f) == 6, "packed header truncated");
    uint64_t off = 16 + uint64_t(pe.layers) * 48;
    for (int l = 0; l < pe.layers; ++l) {
        uint64_t b = 0;
        for (auto v : pe.sizes[size_t(l)]) b += v;
        pe.base.push_back(off);
        pe.block.push_back(b);
        off += b * uint64_t(pe.experts);
    }
    std::fseek(f, 0, SEEK_END);
    pe.bytes = size_t(std::ftell(f));
    std::fclose(f);
    packed_detail::need(pe.bytes == off, "packed file size differs from its header");
    const int fd = ::open(path.c_str(), O_RDONLY);
    packed_detail::need(fd >= 0, "cannot open " + path);
    void * m = ::mmap(nullptr, pe.bytes, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    packed_detail::need(m != MAP_FAILED, "cannot map " + path);
    pe.map = static_cast<const char *>(m);
    return pe;
}

// u4h kernel id of each projection's device format (what the grouped kernels see), from one expert's payload
inline std::array<int, 3> expert_u4h_ids(const std::array<ExpertPayload, 3> & payload, int hidden, int ff) {
    std::array<int, 3> ids{};
    for (int p = 0; p < 3; ++p) {
        const int cols = p == 2 ? ff : hidden, rows = p == 2 ? hidden : ff;
        std::vector<std::vector<char>> keep_bytes;
        Upload keep = [&](const void * data, size_t bytes) {
            keep_bytes.emplace_back(static_cast<const char *>(data), static_cast<const char *>(data) + bytes);
            return keep_bytes.back().data();
        };
        ids[p] = weight_format(payload[p].type)
                     ->load(std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols, keep,
                            nullptr)
                     .fmt->u4h_id();
    }
    return ids;
}

// What one projection's loaded device format can do, reported by repack_expert: the grouped-kernel id, the widest matvec
// it takes, and whether an expert bank can hold its row layout at all (row-sliceable, plain basis, grouped kernel).
struct ExpertFormat {
    int u4h_id = -1;
    int max_matvec_cols = 0;
    bool bank_row_layout = false;
};

// One expert in the device layout (data, aux per projection), as the packed file and an MTP prepack store it.
// `tile`: permute the code parts into the grouped kernels' lane-contiguous tiles (TQFNEXPT).
inline std::array<std::vector<char>, 6> repack_expert(const std::array<ExpertPayload, 3> & payload, int hidden, int ff, bool tile,
                                                      std::array<ExpertFormat, 3> * fmt = nullptr) {
    std::array<std::vector<char>, 6> parts;
    for (int p = 0; p < 3; ++p) {
        const int cols = p == 2 ? ff : hidden, rows = p == 2 ? hidden : ff;
        std::vector<std::vector<char>> pieces;
        std::vector<const void *> where;
        pieces.reserve(4);
        Upload keep = [&](const void * data, size_t bytes) {
            pieces.emplace_back(static_cast<const char *>(data), static_cast<const char *>(data) + bytes);
            where.push_back(pieces.back().data());
            return pieces.back().data();
        };
        const auto m = weight_format(payload[p].type)
                           ->load(std::vector<char>(payload[p].bytes.data, payload[p].bytes.data + payload[p].bytes.size), rows, cols,
                                  keep, nullptr);
        for (size_t k = 0; k < pieces.size(); ++k) {
            if (where[k] == m.p) parts[size_t(2 * p)] = std::move(pieces[k]);
            else if (where[k] == m.aux) parts[size_t(2 * p + 1)] = std::move(pieces[k]);
        }
        packed_detail::need(!parts[size_t(2 * p)].empty(), "loader produced no row data");
        if (fmt) {
            (*fmt)[p].u4h_id = m.fmt->u4h_id();
            (*fmt)[p].max_matvec_cols = m.fmt->max_matvec_cols();
            (*fmt)[p].bank_row_layout = m.fmt->has_expert_matvec() && m.fmt->basis() == Basis::Plain && m.fmt->row_bytes(cols);
        }
        const int bpb = tile ? tqllm_cuda::tiled_block_bytes(m.fmt->u4h_id()) : 0;
        if (bpb && rows % 16 == 0 && (cols / 32) % 4 == 0 && parts[size_t(2 * p)].size() == size_t(rows) * (cols / 32) * bpb) {
            std::vector<char> t(parts[size_t(2 * p)].size());
            tqllm_cuda::tile_expert_rows(parts[size_t(2 * p)].data(), t.data(), rows, cols, bpb);
            parts[size_t(2 * p)] = std::move(t);
        }
    }
    return parts;
}

// Writes the file: `repack(layer, expert)` returns that expert's six parts (the bytes the cache banks hold). Every expert
// of a layer must have the same layout, which the header records once per layer.
inline void write_packed_experts(const std::string & path, int layers, int experts, bool tiled,
                                 const std::function<std::array<std::vector<char>, 6>(int, int)> & repack) {
    packed_detail::need(layers > 0 && experts > 0, "no layers or no experts");
    FILE * f = std::fopen(path.c_str(), "wb");
    packed_detail::need(f != nullptr, "cannot create " + path);
    std::fwrite(tiled ? "TQFNEXPT" : "TQFNEXP1", 1, 8, f);
    const int32_t dims[2] = {layers, experts};
    std::fwrite(dims, 4, 2, f);
    std::vector<std::array<uint64_t, 6>> sizes;
    sizes.resize(size_t(layers));
    for (int l = 0; l < layers; ++l) {
        const auto parts = repack(l, 0);
        for (int k = 0; k < 6; ++k) sizes[size_t(l)][size_t(k)] = parts[size_t(k)].size();
        std::fwrite(sizes[size_t(l)].data(), 8, 6, f);
    }
    for (int l = 0; l < layers; ++l) {
        for (int e = 0; e < experts; ++e) {
            const auto parts = repack(l, e);
            for (int k = 0; k < 6; ++k) {
                packed_detail::need(parts[size_t(k)].size() == sizes[size_t(l)][size_t(k)], "expert layout varies within a layer");
                std::fwrite(parts[size_t(k)].data(), 1, parts[size_t(k)].size(), f);
            }
        }
        std::printf("packed layer %d\n", l);
        std::fflush(stdout);
    }
    packed_detail::need(std::fclose(f) == 0, "write failed: " + path);
}

}  // namespace tqllm
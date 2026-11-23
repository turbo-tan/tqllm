#pragma once
// --split layer: the ranks agree on who runs which layers before any of them loads. Every rank reports its free device
// bytes; rank 0 picks the bounds (--split-at as given, else bounds_by_budget over each rank's free bytes less what every
// rank holds whatever its layers) and sends them to all, so the ranks cannot disagree.
#include "layer_split.hpp"
#include "tp_comm.hpp"

#include <algorithm>
#include <cstdio>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

namespace tqllm {
// A rank's host RAM limit (Flash-Next on a discrete card): the most layer cost it may take, what its box has available and
// what one layer would need there (for the refusal message)
struct HostRoom { bool limited = false; uint64_t cap = 0, available = 0, need_one = 0; };
// `cost`: each layer's device bytes; `fixed`: bytes every rank holds besides its layers; `free_b`: this rank's free bytes
inline std::vector<LayerRange> agree_split(TpComm & tp, const std::vector<uint64_t> & cost, uint64_t fixed, uint64_t free_b,
                                           const std::vector<uint32_t> & manual, const HostRoom & room = {}) {
    const int world = tp.world();
    const uint32_t layers = uint32_t(cost.size());
    struct Report { uint64_t free_b, layers, cap_b, avail_b, need_b, limited; char host[48]; };
    Report mine{free_b, layers, room.limited ? room.cap : 0, room.available, room.need_one, room.limited ? 1u : 0u, {}};
#ifndef _WIN32  // Windows builds have no multi-GPU split (TpComm stub): the host name stays empty
    gethostname(mine.host, sizeof mine.host - 1);
#endif
    std::vector<Report> all(static_cast<size_t>(world));
    tp.ctl_all_gather(&mine, all.data(), sizeof mine);
    std::vector<uint32_t> bounds(size_t(world - 1), 0);
    if (tp.rank() == 0) {
        for (int r = 0; r < world; ++r)
            if (all[size_t(r)].layers != layers)
                throw std::runtime_error("--split layer: rank " + std::to_string(r) + " loads a model of " + std::to_string(all[size_t(r)].layers) +
                                         " layers, rank 0 one of " + std::to_string(layers));
        std::vector<uint64_t> budget, cap, raw_cap;
        std::vector<bool> limited;
        for (const auto & r : all) {
            budget.push_back(r.free_b > fixed ? r.free_b - fixed : 1);
            cap.push_back(r.limited ? std::max<uint64_t>(r.cap_b, 1) : 0);
            raw_cap.push_back(r.cap_b); limited.push_back(r.limited != 0);
        }
        const RoomCheck rc = room_check(limited, raw_cap, *std::min_element(cost.begin(), cost.end()));
        if (!rc.refused.empty()) {
            std::string why = "--split layer: not enough host RAM to keep the 20 GB reserve";
            for (int r : rc.refused)
                why += "; rank " + std::to_string(r) + " (" + all[size_t(r)].host + "): " + std::to_string(all[size_t(r)].avail_b >> 30) +
                       " GB available, one layer there needs " + std::to_string((all[size_t(r)].need_b + (uint64_t(1) << 30) - 1) >> 30) + " GB beside it";
            if (manual.empty()) throw std::runtime_error(why + " (leave the box out, or use a smaller --context)");
            std::fprintf(stderr, "tqllm: WARN %s; --split-at given: loading anyway\n", why.c_str());
        }
        if (rc.move_head && manual.empty())
            std::fprintf(stderr, "tqllm: --split layer: the last rank (%s) has the least host RAM room and also holds the head's prompt buffers: "
                                 "start the box with the most RAM as the last rank (--tp-rank %d)\n", all[size_t(world - 1)].host, world - 1);
        bounds = manual.empty() ? bounds_by_budget(cost, budget, cap) : manual;
        split_ranges(layers, world, bounds);  // validates --split-at
        tp.ctl_send(bounds.data(), bounds.size() * sizeof(uint32_t));
    } else {
        tp.ctl_recv(bounds.data(), bounds.size() * sizeof(uint32_t));
    }
    const auto ranges = split_ranges(layers, world, bounds);
    if (tp.rank() == 0)
        for (int r = 0; r < world; ++r) {
            uint64_t c = 0;
            for (uint32_t l = ranges[size_t(r)].lo; l < ranges[size_t(r)].hi; ++l) c += cost[l];
            char capped[64] = "";
            if (all[size_t(r)].cap_b) std::snprintf(capped, sizeof capped, ", RAM holds %.2f GB of them", double(all[size_t(r)].cap_b) / 1e9);
            std::fprintf(stderr, "tqllm: --split layer rank %d: %.2f GB free, layers %u..%u (%.2f GB of layers%s + %.2f GB on every rank)%s\n", r,
                         double(all[size_t(r)].free_b) / 1e9, ranges[size_t(r)].lo, ranges[size_t(r)].hi - 1, double(c) / 1e9, capped,
                         double(fixed) / 1e9, manual.empty() ? "" : " (--split-at)");
        }
    return ranges;
}
}  // namespace tqllm

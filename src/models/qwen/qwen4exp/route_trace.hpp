#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace tqllm::flashnext {
struct RouteTraceClose { void operator()(FILE * f) const { std::fclose(f); } };
using RouteTraceFile = std::unique_ptr<FILE, RouteTraceClose>;

inline bool routing_trace_enabled() {
    static const bool on = [] {
        const char * a = std::getenv("TQFN_ROUTE_DUMP"), * b = std::getenv("TQFN_PRED_DUMP");
        return (a && *a) || (b && *b);
    }();
    return on;
}

// Legacy little-endian dump framing consumed by cachesim.py: layer, rows,
// optional prediction count, then rows x top-10 original expert ids.
inline void write_route_trace(FILE * out, bool predictions, int layer, int rows, int topk,
                              const int * ids, size_t count, const int * next = nullptr, size_t next_count = 0) {
    if (!out) return;
    static_assert(sizeof(int) == sizeof(int32_t), "route trace requires 32-bit expert ids");
    if (layer < 0 || rows <= 0 || topk != 10 || rows > std::numeric_limits<int32_t>::max() / topk ||
        count != size_t(rows) * size_t(topk) || !ids ||
        (predictions && next_count && (next_count != count || !next)))
        throw std::invalid_argument("route trace: invalid top-10 record shape");
    const int32_t header[3] = {layer, rows, int32_t(predictions ? next_count : 0)};
    const size_t fields = predictions ? 3 : 2;
    if (std::fwrite(header, sizeof(int32_t), fields, out) != fields ||
        std::fwrite(ids, sizeof(int), count, out) != count ||
        (predictions && next_count && std::fwrite(next, sizeof(int), next_count, out) != next_count) ||
        std::fflush(out) != 0)
        throw std::runtime_error("route trace: write failed");
}

inline void dump_routing(int layer, int rows, int topk, const int * ids, size_t count,
                         const int * next = nullptr, size_t next_count = 0) {
    if (!routing_trace_enabled()) return;
    static const auto files = [] {
        const char * route = std::getenv("TQFN_ROUTE_DUMP"), * pred = std::getenv("TQFN_PRED_DUMP");
        if (route && pred && *route && std::string(route) == pred)
            throw std::invalid_argument("route trace: route and prediction paths must differ");
        const auto open = [](const char * path) {
            RouteTraceFile f(path && *path ? std::fopen(path, "ab") : nullptr);
            if (path && *path && !f) throw std::runtime_error("route trace: cannot open " + std::string(path));
            return f;
        };
        return std::make_pair(open(route), open(pred));
    }();
    if (!files.first && !files.second) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    write_route_trace(files.first.get(), false, layer, rows, topk, ids, count);
    write_route_trace(files.second.get(), true, layer, rows, topk, ids, count, next, next_count);
}
} // namespace tqllm::flashnext

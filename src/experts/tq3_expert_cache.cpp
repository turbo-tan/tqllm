#include "tq3_expert_cache.hpp"
#include "quant_kernels.cuh"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace tqllm {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("expert cache: ") + cudaGetErrorString(status));
}
}
struct Tq3ExpertCache::State {
    struct Slot { int layer = -1, expert = -1; uint64_t age = 0; };
    int layers, experts, hidden, ff, capacity;
    cudaStream_t stream;
    Source source;
    std::array<char *, 3> device{};
    std::array<size_t, 3> stride{};
    std::vector<Slot> slots;
    std::vector<uint64_t> frequency;
    uint64_t clock = 0;
    ExpertCacheStats counters;
    const WeightFormat * format;
    State(int l, int e, int h, int f, int c, size_t budget, Source s, cudaStream_t cs)
        : layers(l), experts(e), hidden(h), ff(f), capacity(c), stream(cs), source(std::move(s)),
          format(weight_format(GGML_TQ3_4S)) {
        if (l <= 0 || e <= 0 || h <= 0 || f <= 0 || c <= 0 || h % 32 || f % 32 || !source ||
            int64_t(c) * std::max(h, f) > std::numeric_limits<int>::max() ||
            uint64_t(l) * e > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
            throw std::invalid_argument("expert cache: invalid dimensions or source");
        const uint64_t bytes = uint64_t(h) * f / 2;
        if (bytes > std::numeric_limits<size_t>::max() / 3 / c || bytes * 3 * c > budget)
            throw std::invalid_argument("expert cache: slots exceed device budget");
        if (!format || !format->has_expert_matvec()) throw std::runtime_error("expert cache: TQ3 grouped format missing");
        stride.fill(size_t(bytes)); slots.resize(c); frequency.resize(size_t(l) * e);
        try {
            for (auto & pointer : device) check(cudaMalloc(&pointer, size_t(bytes) * c));
        } catch (...) { for (auto pointer : device) if (pointer) cudaFree(pointer); throw; }
        counters.device_bytes = size_t(bytes) * 3 * c;
    }
    ~State() {
        cudaStreamSynchronize(stream);
        for (auto pointer : device) cudaFree(pointer);
    }
    uint64_t count(const Slot & slot) const {
        return slot.layer < 0 ? 0 : frequency[size_t(slot.layer) * experts + slot.expert];
    }
};
Tq3ExpertCache::Tq3ExpertCache(int l, int e, int h, int f, int c, size_t budget, Source source, cudaStream_t stream)
    : state_(std::make_unique<State>(l, e, h, f, c, budget, std::move(source), stream)) {}
Tq3ExpertCache::~Tq3ExpertCache() = default;
std::vector<int> Tq3ExpertCache::prepare(int layer, const std::vector<int> & ids) {
    auto & s = *state_;
    if (layer < 0 || layer >= s.layers) throw std::invalid_argument("expert cache: layer out of range");
    std::set<int> needed;
    for (int id : ids) {
        if (id < 0 || id >= s.experts) throw std::invalid_argument("expert cache: expert out of range");
        needed.insert(id);
    }
    if (needed.size() > size_t(s.capacity)) throw std::invalid_argument("expert cache: active experts exceed slots");
    std::map<int, int> placements;
    std::map<int, std::array<GgufBytes, 3>> incoming;
    for (int slot = 0; slot < s.capacity; ++slot)
        if (s.slots[slot].layer == layer && needed.count(s.slots[slot].expert))
            placements.emplace(s.slots[slot].expert, slot);
    for (int id : needed) if (!placements.count(id)) {
        auto slices = s.source(layer, id);
        for (int projection = 0; projection < 3; ++projection)
            if (!slices[projection].data || slices[projection].size != s.stride[projection])
                throw std::invalid_argument("expert cache: raw TQ3 slice size differs");
        incoming.emplace(id, slices);
    }
    // Stream ordering protects the prior projections before a slot is overwritten.
    for (const auto & [id, slices] : incoming) {
        int target = -1;
        for (int slot = 0; slot < s.capacity; ++slot) {
            const auto & candidate = s.slots[slot];
            if (candidate.layer == layer && needed.count(candidate.expert)) continue;
            if (target < 0 || candidate.layer < 0 || s.count(candidate) < s.count(s.slots[target]) ||
                (s.count(candidate) == s.count(s.slots[target]) && candidate.age < s.slots[target].age)) target = slot;
            if (candidate.layer < 0) break;
        }
        if (target < 0) throw std::logic_error("expert cache: no unpinned slot");
        auto & slot = s.slots[target];
        if (slot.layer >= 0) ++s.counters.evictions;
        // Invalidate before mutation: failed CUDA work cannot leave a plausible stale hit.
        slot.layer = -1; slot.expert = -1;
        for (int projection = 0; projection < 3; ++projection) {
            auto * destination = s.device[projection] + size_t(target) * s.stride[projection];
            check(cudaMemcpyAsync(destination, slices[projection].data, slices[projection].size,
                                  cudaMemcpyHostToDevice, s.stream));
            tqllm_cuda::tq3_repack(destination, int64_t(s.stride[projection] / 16), s.stream);
            check(cudaGetLastError());
        }
        check(cudaStreamSynchronize(s.stream));
        slot = {layer, id, ++s.clock}; placements.emplace(id, target);
        ++s.counters.misses; s.counters.transfer_bytes += s.stride[0] + s.stride[1] + s.stride[2];
    }
    s.counters.hits += needed.size() - incoming.size();
    std::vector<int> result; result.reserve(ids.size());
    for (int id : ids) {
        result.push_back(placements.at(id));
        auto & frequency = s.frequency[size_t(layer) * s.experts + id];
        if (frequency != std::numeric_limits<uint64_t>::max()) ++frequency;
        s.slots[placements.at(id)].age = ++s.clock;
    }
    return result;
}
Mat Tq3ExpertCache::gate() const {
    const auto & s = *state_; return {s.device[0], GGML_TQ3_4S, s.capacity * s.ff, s.hidden, nullptr, s.format};
}
Mat Tq3ExpertCache::up() const {
    const auto & s = *state_; return {s.device[1], GGML_TQ3_4S, s.capacity * s.ff, s.hidden, nullptr, s.format};
}
Mat Tq3ExpertCache::down() const {
    const auto & s = *state_; return {s.device[2], GGML_TQ3_4S, s.capacity * s.hidden, s.ff, nullptr, s.format};
}
ExpertCacheStats Tq3ExpertCache::stats() const { return state_->counters; }
void Tq3ExpertCache::clear() {
    auto & s = *state_; check(cudaStreamSynchronize(s.stream));
    std::fill(s.slots.begin(), s.slots.end(), State::Slot{});
    std::fill(s.frequency.begin(), s.frequency.end(), 0);
    const auto bytes = s.counters.device_bytes;
    s.counters = {}; s.counters.device_bytes = bytes; s.clock = 0;
}
} // namespace tqllm

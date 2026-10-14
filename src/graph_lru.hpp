#pragma once
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <stdexcept>
#include <utility>

namespace tqllm {
// Instantiated graphs keyed by (family, key), least recently used evicted first. Keys that name a slot set grow with
// the parallel slots (2^np subsets x rows), and each graph holds ~12 MB of device memory: unbounded maps ran a 3090 at
// np 8 out of memory mid-run ("graph instantiate: out of memory"). A miss evicts while the cache is at `cap` or
// `low_memory()` says so; a make() that still runs out of memory empties the cache and is retried once.
// `quiesce` runs before anything is destroyed (the evicted graph may still be queued on the stream).
template <class Exec>
class GraphLru {
  public:
    using Key = std::pair<int, uint64_t>;
    explicit GraphLru(size_t cap = 256) : cap_(cap ? cap : 1) {}
    void set_cap(size_t cap) { cap_ = cap ? cap : 1; }
    size_t size() const { return map_.size(); }
    size_t evictions() const { return evictions_; }

    Exec find(int family, uint64_t key) {
        const auto it = map_.find({family, key});
        if (it == map_.end()) return Exec{};
        order_.splice(order_.begin(), order_, it->second.second);
        return it->second.first;
    }
    template <class Make, class LowMemory, class Quiesce, class Destroy>
    Exec insert(int family, uint64_t key, Make make, LowMemory low_memory, Quiesce quiesce, Destroy destroy) {
        bool synced = false;
        while (!map_.empty() && (map_.size() >= cap_ || low_memory())) {
            if (!synced) quiesce(), synced = true;
            evict_one(destroy);
        }
        Exec e{};
        try {
            e = make();
        } catch (const std::runtime_error & ex) {
            if (map_.empty() || !std::strstr(ex.what(), "out of memory")) throw;
            if (!synced) quiesce();
            clear(destroy);
            e = make();
        }
        order_.push_front({family, key});
        map_.emplace(Key{family, key}, std::make_pair(e, order_.begin()));
        return e;
    }
    template <class Destroy>
    void clear(Destroy destroy) {
        for (auto & [k, v] : map_) destroy(v.first);
        evictions_ += map_.size();
        map_.clear(), order_.clear();
    }

  private:
    template <class Destroy>
    void evict_one(Destroy destroy) {
        const Key k = order_.back();
        order_.pop_back();
        const auto it = map_.find(k);
        destroy(it->second.first);
        map_.erase(it);
        ++evictions_;
    }
    size_t cap_;
    size_t evictions_ = 0;
    std::list<Key> order_;
    std::map<Key, std::pair<Exec, typename std::list<Key>::iterator>> map_;
};
}  // namespace tqllm

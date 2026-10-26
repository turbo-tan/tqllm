#pragma once
#include <array>
#include <list>
#include <map>
#include <memory>
#include <future>
#include <utility>
#include <vector>

namespace tqllm {
class ColdExpertMemo {
public:
    using Key = std::pair<int,int>;
    using Payload = std::array<std::vector<char>,3>;
private:
    struct Entry { std::shared_ptr<Payload> payload; std::list<Key>::iterator age; size_t bytes; };
    size_t capacity_, bytes_ = 0;
    std::list<Key> recent_;
    std::map<Key, Entry> entries_;
public:
    explicit ColdExpertMemo(size_t capacity) : capacity_(capacity) {}
    size_t bytes() const { return bytes_; }
    void set_capacity(size_t capacity) {
        capacity_=capacity;
        while(bytes_>capacity_) {
            auto victim=entries_.find(recent_.front());
            bytes_-=victim->second.bytes;
            entries_.erase(victim);recent_.pop_front();
        }
    }
    template<class Pool,class Load> std::vector<std::shared_ptr<Payload>> get_batch(const std::vector<Key> & keys, Pool & pool, Load load) {
        std::vector<std::shared_ptr<Payload>> result(keys.size());
        std::vector<std::pair<size_t,std::future<Payload>>> pending;
        for(size_t i=0;i<keys.size();++i) {
            auto it=entries_.find(keys[i]);
            if(it!=entries_.end()) {
                recent_.splice(recent_.end(),recent_,it->second.age);result[i]=it->second.payload;
            } else {
                const auto key=keys[i];
                pending.emplace_back(i,pool.submit([load,key]{return load(key);}));
            }
        }
        for(auto & job:pending) {
            auto raw=job.second.get();
            result[job.first]=get(keys[job.first],[&]{return std::move(raw);});
        }
        return result;
    }
    template<class Load> std::shared_ptr<Payload> get(Key key, Load load) {
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            recent_.splice(recent_.end(), recent_, it->second.age);
            return it->second.payload;
        }
        auto payload = std::make_shared<Payload>(load());
        size_t bytes = 0; for (const auto & p : *payload) bytes += p.size();
        if (bytes > capacity_) return payload;
        while (bytes_ > capacity_ - bytes) {
            auto victim = entries_.find(recent_.front());
            bytes_ -= victim->second.bytes;
            entries_.erase(victim); recent_.pop_front();
        }
        recent_.push_back(key);
        try { entries_.emplace(key, Entry{payload, std::prev(recent_.end()), bytes}); }
        catch (...) { recent_.pop_back(); throw; }
        bytes_ += bytes;
        return payload;
    }
};
} // namespace tqllm

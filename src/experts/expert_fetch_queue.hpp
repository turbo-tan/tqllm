#pragma once
#include "expert_read_pool.hpp"
#include <condition_variable>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace tqllm {
template<class Payload> class ExpertFetchQueue {
public:
    using Key=std::pair<int,int>;
    struct Stats {uint64_t prefetched=0,deduplicated=0,promoted=0,used=0,expired=0,rejected=0,demand_reads=0,completed=0;};
private:
    struct Entry {
        Key key;std::packaged_task<Payload()> task;std::shared_future<Payload> result;
        bool active=false,done=false,demand=false;
        Entry(Key k,std::function<Payload()> f):key(k),task(std::move(f)),result(task.get_future().share()){}
    };
    ExpertReadPool & pool_;
    std::function<Payload(Key)> load_;
    size_t limit_,demands_=0;
    std::mutex mutex_;std::condition_variable ready_;bool stop_=false;
    std::map<Key,std::shared_ptr<Entry>> pending_,inflight_;
    std::list<std::shared_ptr<Entry>> speculative_;
    Stats stats_;
    std::thread worker_;
    std::shared_ptr<Entry> make(Key key) {
        auto load=load_;
        return std::make_shared<Entry>(key,[load,key]{return load(key);});
    }
    void execute(const std::shared_ptr<Entry> & e) {
        e->task();
        {std::lock_guard<std::mutex> lock(mutex_);e->done=true;if(e->demand){--demands_;inflight_.erase(e->key);}++stats_.completed;}
        ready_.notify_all();
    }
    void run() {
        for(;;) {
            std::shared_ptr<Entry> e;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,[this]{return stop_ || (!demands_ && !speculative_.empty());});
                if(stop_)return;
                e=speculative_.front();speculative_.pop_front();e->active=true;
            }
            execute(e);
        }
    }
public:
    ExpertFetchQueue(ExpertReadPool & pool,size_t limit,std::function<Payload(Key)> load)
        :pool_(pool),load_(std::move(load)),limit_(limit) {
        if(!limit || !load_)throw std::invalid_argument("expert fetch queue: invalid limit or loader");
        worker_=std::thread([this]{run();});
    }
    ~ExpertFetchQueue() {
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;speculative_.clear();pending_.clear();}
        ready_.notify_all();if(worker_.joinable())worker_.join();
        std::unique_lock<std::mutex> lock(mutex_);ready_.wait(lock,[this]{return !demands_;});
    }
    Stats stats() {std::lock_guard<std::mutex> lock(mutex_);return stats_;}
    void prefetch(int layer,const std::vector<int> & ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(stop_)return;
        // Only the current and next layer survive; speculative work never grows across a whole pass.
        for(auto it=pending_.begin();it!=pending_.end();) {
            if(it->first.first==layer || it->first.first==layer-1){++it;continue;}
            auto e=it->second;speculative_.remove(e);it=pending_.erase(it);++stats_.expired;
        }
        for(int id:ids) {
            Key key{layer,id};
            if(pending_.count(key) || inflight_.count(key)){++stats_.deduplicated;continue;}
            if(pending_.size()>=limit_){++stats_.rejected;continue;}
            auto e=make(key);speculative_.push_back(e);
            try{pending_.emplace(key,e);}catch(...){speculative_.pop_back();throw;}
            ++stats_.prefetched;
        }
        ready_.notify_all();
    }
    std::vector<std::shared_future<Payload>> demand(const std::vector<Key> & keys) {
        std::vector<std::shared_future<Payload>> out;out.reserve(keys.size());
        std::vector<std::shared_ptr<Entry>> start;start.reserve(keys.size());
        auto dispatch=[&] {
            for(auto & e:start)try{(void)pool_.submit([this,e]{execute(e);});}catch(...){execute(e);}
        };
        std::map<Key,std::shared_future<Payload>> seen;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if(stop_)throw std::runtime_error("expert fetch queue is closed");
            for(Key key:keys) {
                if(auto dup=seen.find(key);dup!=seen.end()){out.push_back(dup->second);++stats_.deduplicated;continue;}
                if(auto active=inflight_.find(key);active!=inflight_.end()) {
                    seen.emplace(key,active->second->result);out.push_back(active->second->result);++stats_.deduplicated;continue;
                }
                auto predicted=pending_.find(key);
                auto e=predicted==pending_.end() ? make(key) : predicted->second;
                // Allocate bookkeeping before changing ownership or marking a task runnable.
                seen.emplace(key,e->result);
                if(!e->done)inflight_.emplace(key,e);
                if(predicted!=pending_.end()) {
                    pending_.erase(predicted);++stats_.used;
                    if(!e->active){speculative_.remove(e);++stats_.promoted;}
                } else ++stats_.demand_reads;
                if(!e->done){e->demand=true;++demands_;}
                if(!e->active){e->active=true;start.push_back(e);}
                out.push_back(e->result);
            }
        } catch(...) {dispatch();throw;}
        // Futures are joined by the caller, never by a worker waiting on this same pool.
        dispatch();return out;
    }
};
} // namespace tqllm

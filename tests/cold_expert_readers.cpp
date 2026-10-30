#include "cold_expert_memo.hpp"
#include "expert_read_pool.hpp"
#include "cold_expert_store.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <iostream>
using namespace tqllm;
int main() {
    std::promise<int> blocked,ready;
    std::vector<std::shared_future<int>> pending{blocked.get_future().share(),ready.get_future().share()};
    ready.set_value(17);
    assert(next_ready_expert(pending)==1 && pending[1].get()==17);
    pending[1]={};blocked.set_value(9);
    assert(next_ready_expert(pending)==0 && pending[0].get()==9);pending[0]={};
    bool empty_failed=false;try{next_ready_expert(pending);}catch(const std::logic_error &){empty_failed=true;}
    assert(empty_failed);
    pending={std::async(std::launch::deferred,[]{return 23;}).share()};
    assert(next_ready_expert(pending)==0 && pending[0].get()==23);

    ExpertReadPool pool(2);ColdExpertMemo memo(6);
    std::atomic<int> active{0},peak{0},loads{0};
    auto load=[&](ColdExpertMemo::Key key) {
        int now=++active;int old=peak.load();while(old<now&&!peak.compare_exchange_weak(old,now)){}
        ++loads;std::this_thread::sleep_for(std::chrono::milliseconds(key.second==0?20:2));
        ColdExpertMemo::Payload out;for(auto & v:out)v.assign(2,char(key.second));--active;return out;
    };
    auto got=memo.get_batch(std::vector<ColdExpertMemo::Key>{{0,0},{0,1},{0,2}},pool,load);
    assert(peak>=2 && loads==3 && memo.bytes()==6);
    for(int i=0;i<3;++i)for(const auto & v:*got[i])assert(v==std::vector<char>(2,char(i)));
    auto retained=got[0];got.clear();assert((*retained)[0][0]==0);
    auto hit=memo.get_batch(std::vector<ColdExpertMemo::Key>{{0,2}},pool,load);assert(loads==3);
    bool threw=false;
    try {auto f=pool.submit([]()->int{throw std::runtime_error("read failure");});f.get();}catch(const std::runtime_error &){threw=true;}
    assert(threw);assert(pool.submit([]{return 7;}).get()==7);
    const auto root=std::filesystem::temp_directory_path()/("cold-reader-test-"+std::to_string(getpid()));
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::filesystem::remove_all(p);}} cleanup{root};
    ColdExpertStore store(root.string(),18,"parallel-store",true);
    ColdExpertMemo::Payload raw;for(auto & v:raw)v.assign(18,42);
    store.save(0,0,raw);store.profile(true);
    std::vector<std::future<ColdExpertMemo::Payload>> reads;
    for(int i=0;i<16;++i)reads.push_back(pool.submit([&]{return store.load(0,0);}));
    for(auto & f:reads)assert(f.get()==raw);
    assert(store.stats().loads==16 && store.stats().bytes==16*54);
    std::vector<std::future<ColdExpertStore::View>> maps;
    for(int i=0;i<16;++i)maps.push_back(pool.submit([&]{return store.load_mapped(0,0);}));
    const void * first=nullptr;
    for(auto & f:maps) {
        auto view=f.get();
        if(!first)first=view.owner.get();
        assert(first==view.owner.get());
        for(int p=0;p<3;++p)assert(std::vector<char>(view.parts[p].data(),view.parts[p].data()+view.parts[p].size())==raw[p]);
    }
    assert(store.stats().loads==32 && store.stats().mapped_ns>0);
    std::cout<<"cold_expert_readers: PASS (parallel reads, ordered payloads, bounded memo, retained ownership, exception recovery)\n";
}

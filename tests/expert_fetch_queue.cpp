#include "expert_fetch_queue.hpp"
#include "expert_prediction.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <map>
using namespace tqllm;
int main() {
    ExpertReadPool pool(2);
    std::promise<void> entered,release,demand_entered,release_demand,spec_entered;
    auto demand_ready=demand_entered.get_future();auto demand_unblock=release_demand.get_future().share();auto spec_ready=spec_entered.get_future();
    auto ready=entered.get_future();auto unblock=release.get_future().share();
    std::mutex mutex;std::map<int,int> calls;
    ExpertFetchQueue<int> queue(pool,4,[&](auto key) {
        {std::lock_guard<std::mutex> lock(mutex);++calls[key.second];}
        if(key.second==1){entered.set_value();unblock.wait();}
        if(key.second==20){demand_entered.set_value();demand_unblock.wait();}
        if(key.second==21)spec_entered.set_value();
        if(key.second<0)throw std::runtime_error("injected read failure");
        return key.second*10;
    });
    queue.prefetch(1,{1,2,2,3});
    assert(ready.wait_for(std::chrono::seconds(1))==std::future_status::ready);
    auto demand=queue.demand({{1,3},{1,3},{1,4}});
    for(auto & f:demand)assert(f.wait_for(std::chrono::seconds(1))==std::future_status::ready);
    assert(demand[0].get()==30 && demand[1].get()==30 && demand[2].get()==40);
    auto joined=queue.demand({{1,1}}); // joins the in-flight prefetch, without using up a demand worker
    auto joined_again=queue.demand({{1,1}});
    queue.prefetch(2,{5,6,7,8,9});
    auto stats=queue.stats();assert(stats.promoted==1 && stats.deduplicated>=2 && stats.rejected>0);
    queue.prefetch(4,{}); // expires old queued work before it can read anything
    release.set_value();assert(joined[0].get()==10 && joined_again[0].get()==10);
    auto failed=queue.demand({{4,-1}});bool rejects=false;
    try{failed[0].get();}catch(const std::runtime_error &){rejects=true;}assert(rejects);
    assert(queue.demand({{4,10}})[0].get()==100);
    {std::lock_guard<std::mutex> lock(mutex);assert(calls[1]==1 && calls[2]==0 && calls[3]==1 && calls[4]==1 && calls[5]==0);}
    assert(queue.stats().expired>0);
    auto blocked=queue.demand({{5,20}});assert(demand_ready.wait_for(std::chrono::seconds(1))==std::future_status::ready);
    queue.prefetch(5,{21});assert(spec_ready.wait_for(std::chrono::milliseconds(50))==std::future_status::timeout);
    release_demand.set_value();assert(blocked[0].get()==200);
    assert(spec_ready.wait_for(std::chrono::seconds(1))==std::future_status::ready);
    assert(queue.demand({{5,21}})[0].get()==210);
    auto ranked=rank_expert_predictions(std::vector<int>{1,2,3,2,4,5,2,5,3},3,2,[](int id){return id!=2;});
    assert((ranked==std::vector<int>{5,3}));
    assert(rank_expert_predictions(std::vector<int>{1,2,3},3,0,[](int){return true;}).empty());
    std::cout<<"expert_fetch_queue: PASS (demand priority, promotion, in-flight dedup, bound, expiry, error recovery)\n";
}

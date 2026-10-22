#pragma once
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>
namespace tqllm {
template<class Eligible> std::vector<int> rank_expert_predictions(const std::vector<int> & ids,int topk,int limit,Eligible eligible) {
    if(topk<=0 || limit<0 || ids.size()%size_t(topk))throw std::invalid_argument("expert prediction shape differs");
    struct Score {int id,votes=0,rank=0;};std::map<int,Score> scores;
    for(size_t a=0;a<ids.size();++a) {
        const int id=ids[a];if(!eligible(id))continue;
        auto & s=scores[id];s.id=id;++s.votes;s.rank+=topk-int(a%size_t(topk));
    }
    std::vector<Score> ranked;for(auto & [id,s]:scores)ranked.push_back(s);
    std::sort(ranked.begin(),ranked.end(),[](const Score & a,const Score & b){
        return a.votes!=b.votes ? a.votes>b.votes : a.rank!=b.rank ? a.rank>b.rank : a.id<b.id;
    });
    std::vector<int> out;for(const auto & s:ranked){if(out.size()>=size_t(limit))break;out.push_back(s.id);}return out;
}
} // namespace tqllm

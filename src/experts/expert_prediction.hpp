#pragma once
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>
namespace tqllm {
// Guess pipeline (tqueue.hpp): how many style guesses the host-to-device copy queue keeps on the link at once, and how
// many experts the VRAM park holds off the cache slots. The engine's small-card decode reads misses from here.
//
// Measured (.188, RTX 3090, 16 GB VRAM hog, 25.3K prompt, 192 tokens, one request text, decode medians of 4 after a
// warm request): a layer call routes to ~4.7 experts the banks do not hold, 3.9 MB each, and the layer's own kernels
// (the hit experts' GEMM, 0.19 ms) cover only part of the 0.55 ms those copies take, so the GPU waits on the link with
// nothing to run. Fewer bytes cannot be had (the banks hold 107 of 512 experts: rounds 1-3), so the lever is to move a
// guessed miss onto the link one layer early: a correct guess waits in the park and its demand miss becomes a
// device-to-device scatter instead of a copy. That is the only lookahead the router allows, since layer l's routing is
// known only after layer l-1's MoE output exists (fn_layer_forward.cpp, round 2).
//   depth 2 (before): 89.5 tok/s, 49.6 ms a step, 136.0 MB of demand + 36.2 MB of guesses per token, 3,365 guesses
//                     copied per decode, 2,314 pickups
//   depth 4:          94.0 tok/s, 47.0 ms a step, 125.2 + 61.1 MB per token, 5,690 copies, 3,793 pickups
//   depth 6:          93.8 tok/s, 47.1 ms a step, 119.1 + 74.4 MB per token, 6,944 copies, 4,521 pickups
//                     (the same speed for 13 MB a token more, i.e. the knee sits at 4-6)
//   depth 10:         still 121.9 MB of demand per token against 102.3 of guesses, i.e. the deeper guesses are copied
//                     and thrown away
// One layer call's demand misses (the LEDGER's misses / (steps x layers), 4.7 at 25.3K): the pipeline has to cover that
// many, and tests/tqueue.cpp states the rule with this number.
constexpr int kLayerDemandMisses = 5;
// The park stays at the 16 experts it had: it is taken out of the expert banks at start-up, and one bank slot less moves
// the prefill plan (room = free + banks) and with it the 25.3K tokens (rounds 2 and 4), so growing it would not be exact.
// The rule the measurements state: one layer's demand misses are on the link at once, so the layer that counts on them
// never waits for a guess to be queued. 5 is one layer's misses (4.7) rounded up, between the two measured-equal depths
// 4 and 6; the old value was 2, half a layer, and a demand batch of layer l then waited behind guesses of layer l-1.
constexpr int kGuessLinkDepth = 5;
constexpr int kGuessParkSpaces = 16;

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

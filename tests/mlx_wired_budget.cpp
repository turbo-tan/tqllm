#include "wired_budget.hpp"
#include <iostream>
int main() {
    using tqllm::mlx_backend::wired_bytes;
    size_t recommended=size_t(40)*1024*1024*1024;
    if(wired_bytes("0",recommended)!=0 || wired_bytes("28672",recommended)!=size_t(28)*1024*1024*1024)
        throw std::runtime_error("wired budget conversion mismatch");
    for(auto text:{"","-1","+1","1.5"," 28","28 ","40961","18446744073709551615","18446744073709551616"}) {
        bool rejected=false;try{wired_bytes(text,recommended);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid wired budget accepted");
    }
    if(wired_bytes("40960",recommended)!=recommended)throw std::runtime_error("recommended boundary rejected");
    using tqllm::mlx_backend::request_wired_bytes;
    const size_t gib=size_t(1)<<30;
    if(request_wired_bytes(26*gib,gib,recommended)!=27*gib || request_wired_bytes(40*gib,gib,recommended)!=recommended)
        throw std::runtime_error("request residency is not active + headroom capped at the recommendation");
    using tqllm::mlx_backend::keep_state_after_request;
    {   // thinking budget: phase 1 keeps its state, only phase 2 reuses it; tail reuse (--prompt-cache / --exact off) as before
        using tqllm::mlx_backend::keep_state_for; using tqllm::mlx_backend::may_reuse_held;
        const bool ok = keep_state_for(false,1,true) && !keep_state_for(false,1,false) && !keep_state_for(false,0,true) &&
            !keep_state_for(false,2,true) && keep_state_for(true,0,true) &&
            may_reuse_held(false,2,true) && !may_reuse_held(false,2,false) && !may_reuse_held(false,0,true) &&
            !may_reuse_held(false,1,true) && may_reuse_held(true,0,false);
        if(!ok) {std::printf("FAIL thinking-budget continuation rule\n"); return 1;}
    }
    if(keep_state_after_request(false,true) || keep_state_after_request(true,false) || !keep_state_after_request(true,true))
        throw std::runtime_error("an idle server keeps KV / drafter state no later request can reuse");
    std::cout<<"mlx_wired_budget: PASS\n";
}

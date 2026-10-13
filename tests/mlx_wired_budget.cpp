#include "../src/mlx/wired_budget.hpp"
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
    std::cout<<"mlx_wired_budget: PASS\n";
}

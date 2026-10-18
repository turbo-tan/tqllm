#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mx=mlx::core;
using tqllm::mlx_backend::Qwen35;
mx::array rows(const mx::array& x,int a,int b) {return mx::slice(x,{a,0},{b,x.shape(1)});}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("usage: test-mlx-mtp model.gguf");
        mx::set_cache_limit(512*1024*1024);
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);
        auto prompt=tokenizer.encode("Explain why rainbows form, including refraction and reflection.");
        Qwen35 engine(argv[1],256);engine.gdn_mode(Qwen35::GdnMode::scan);engine.enable_mtp();
        auto prefill=engine.forward(prompt);auto hidden=engine.hidden_states();
        engine.draft_forward({prompt.data()+1,prompt.size()-1},rows(hidden,0,int(prompt.size())-1),1,false);
        const auto snapshot=engine.draft_checkpoint();
        const uint32_t token=mx::argmax(prefill).item<uint32_t>();
        const auto previous=rows(hidden,int(prompt.size())-1,int(prompt.size()));
        auto full=engine.draft_forward({&token,1},previous,int(prompt.size()));
        const std::vector<float> expected(full.data<float>(),full.data<float>()+full.size());
        const uint32_t next=mx::argmax(full).item<uint32_t>();
        auto next_hidden=engine.draft_hidden_states();
        auto continuation=engine.draft_forward({&next,1},next_hidden,int(prompt.size())+1);
        std::vector<float> expected_next(continuation.data<float>(),continuation.data<float>()+continuation.size());
        engine.draft_restore(snapshot);engine.draft_inject({&token,1},previous,int(prompt.size()));
        auto injected=engine.draft_forward({&next,1},next_hidden,int(prompt.size())+1);
        if(!std::equal(expected_next.begin(),expected_next.end(),injected.data<float>()))
            throw std::runtime_error("MTP injection continuation differs from full forward");
        engine.draft_restore(snapshot);
        auto replay=engine.draft_forward({&token,1},previous,int(prompt.size()));
        if(!std::equal(expected.begin(),expected.end(),replay.data<float>()))throw std::runtime_error("MTP snapshot replay changed logits");
        engine.draft_restore(snapshot);engine.draft_shortlist(prefill,8192);
        auto subset=engine.draft_forward({&token,1},previous,int(prompt.size()));
        auto mapping=engine.draft_ids();const auto* ids=mapping.data<uint32_t>();const auto* values=subset.data<float>();
        for(size_t j=0;j<subset.size();++j)
            if(ids[j]>=expected.size() || values[j]!=expected[ids[j]])throw std::runtime_error("shortlist row differs from full head");
        engine.reset();bool rejected=false;
        try {engine.draft_restore(snapshot);}catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected)throw std::runtime_error("MTP snapshot survived reset");
        std::cout<<"MTP_PASS replay_bit_exact=1 injection_continuation_bit_exact=1 shortlist_rows_bit_exact=8192 stale_reset_rejected=1"<<std::endl;
    } catch(const std::exception& e) {std::cerr<<"MTP_ERROR "<<e.what()<<'\n';return 1;}
}

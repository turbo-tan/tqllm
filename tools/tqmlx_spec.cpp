#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using tqllm::mlx_backend::Qwen35;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
mx::array rows(const mx::array& x,int begin,int end) {return mx::slice(x,{begin,0},{end,x.shape(1)});}
int main(int argc,char** argv) {
    try {
        if(argc<3 || argc>8)throw std::invalid_argument("usage: tqmlx-spec model.gguf prompt [draft=3] [count=128] [reps=5] [shortlist=8192] [warmup=1]");
        const int draft=argc>3?std::stoi(argv[3]):3,count=argc>4?std::stoi(argv[4]):128;
        const int reps=argc>5?std::stoi(argv[5]):5,shortlist=argc>6?std::stoi(argv[6]):8192,warmup=argc>7?std::stoi(argv[7]):1;
        if(draft<1 || draft>7 || count<2 || reps<1 || shortlist<0 || warmup<0)
            throw std::invalid_argument("invalid speculative benchmark parameters");
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);auto prompt=tokenizer.encode(argv[2]);
        if(prompt.size()<2)throw std::invalid_argument("MTP benchmark needs two prompt tokens");
        mx::set_cache_limit(512*1024*1024);
        Qwen35 engine(argv[1],int(prompt.size())+count+16);
        engine.gdn_mode(Qwen35::GdnMode::scan);engine.enable_mtp();engine.verify_tiles(true);
        const bool head4=std::getenv("TQLLM_MLX_DRAFT_HEAD4")!=nullptr;
        if(head4)engine.enable_draft_head4();
        std::vector<uint32_t> reference,first_spec;
        for(int mode=0;mode<2;++mode)for(int run=-warmup;run<reps;++run) {
            engine.reset();auto begin=Clock::now();
            auto logits=engine.forward(prompt);
            if(mode) {
                if(shortlist && !head4)engine.draft_shortlist(logits,shortlist);
                engine.draft_inject({prompt.data()+1,prompt.size()-1},rows(engine.hidden_states(),0,int(prompt.size())-1),1);
            }
            const double prefill=seconds(begin);
            uint32_t pending=mx::argmax(logits).item<uint32_t>();
            std::vector<uint32_t> output{pending};
            int steps=0,drafted=0,accepted_total=0;double draft_ms=0,verify_ms=0,commit_ms=0;
            const auto decode_begin=Clock::now();
            while(output.size()<size_t(count) && pending!=tokenizer.eos_id() && output.back()!=tokenizer.eos_id()) {
                if(!mode) {logits=engine.forward({&pending,1});pending=mx::argmax(logits).item<uint32_t>();output.push_back(pending);continue;}
                const int position=engine.position(),width=std::min(draft,int(count-output.size()));
                const auto previous=rows(engine.hidden_states(),engine.hidden_states().shape(0)-1,engine.hidden_states().shape(0));
                const auto draft_start=engine.draft_checkpoint();
                begin=Clock::now();
                std::vector<uint32_t> proposed;
                for(int j=0;j<width;++j) {
                    const uint32_t input=j?proposed.back():pending;
                    const auto h=j?engine.draft_hidden_states():previous;
                    auto d=engine.draft_forward({&input,1},h,position+j);
                    const uint32_t index=mx::argmax(d).item<uint32_t>();
                    proposed.push_back(engine.draft_ids().data<uint32_t>()[index]);
                }
                draft_ms+=seconds(begin)*1000;
                std::vector<uint32_t> block{pending};block.insert(block.end(),proposed.begin(),proposed.end());
                Qwen35::Verification ticket;begin=Clock::now();
                auto verified=engine.verify(block,ticket);auto choices=mx::argmax(verified,-1);mx::eval(choices);
                const auto* ids=choices.data<uint32_t>();
                auto hidden=engine.hidden_states();
                verify_ms+=seconds(begin)*1000;
                int accepted=0;
                while(accepted<width && proposed[accepted]==ids[accepted])++accepted;
                ++steps;drafted+=width;accepted_total+=accepted;
                for(int j=0;j<accepted && output.size()<size_t(count);++j) {
                    output.push_back(proposed[j]);if(output.back()==tokenizer.eos_id())break;
                }
                if(output.size()<size_t(count) && output.back()!=tokenizer.eos_id())output.push_back(ids[accepted]);
                pending=ids[accepted];begin=Clock::now();
                engine.accept_prefix(ticket,accepted+1);
                engine.draft_restore(draft_start);
                auto pairs=accepted?mx::concatenate({previous,rows(hidden,0,accepted)},0):previous;
                engine.draft_inject({block.data(),size_t(accepted+1)},pairs,position);
                commit_ms+=seconds(begin)*1000;
            }
            const double decode=seconds(decode_begin);
            if(run<0)continue;
            auto& check=mode?first_spec:reference;
            if(check.empty())check=output;
            if(check!=output)throw std::runtime_error("greedy output changed across reset repetitions");
            size_t equal=0;while(equal<std::min(output.size(),reference.size()) && output[equal]==reference[equal])++equal;
            std::cout<<"{\"kind\":\"mtp_generation\",\"mode\":\""<<(mode?"spec":"ar")<<"\",\"run\":"<<run
                     <<",\"draft\":"<<draft<<",\"proposal_head_bits\":"<<(head4?4:6)<<",\"shortlist\":"<<(head4?0:shortlist)<<",\"prompt_tokens\":"<<prompt.size()
                     <<",\"output_tokens\":"<<output.size()<<",\"decode_tok_s\":"<<(output.size()-1)/decode
                     <<",\"ttft_ms\":"<<prefill*1000<<",\"steps\":"<<steps<<",\"drafted\":"<<drafted<<",\"accepted\":"<<accepted_total
                     <<",\"draft_ms\":"<<draft_ms<<",\"verify_ms\":"<<verify_ms<<",\"commit_ms\":"<<commit_ms
                     <<",\"ar_equal_prefix\":"<<equal<<",\"active_bytes\":"<<mx::get_active_memory()<<",\"token_ids\":[";
            for(size_t j=0;j<output.size();++j)std::cout<<(j?",":"")<<output[j];
            std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT mode="<<(mode?"spec":"ar")<<' '<<tokenizer.decode(output)<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<"SPEC_ERROR "<<e.what()<<'\n';return 1;}
}

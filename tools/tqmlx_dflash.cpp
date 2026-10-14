#include "wired_limit.hpp"
#include "dflash.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <span>
#include <vector>
namespace mx=mlx::core;
using tqllm::mlx_backend::Qwen35;
using tqllm::mlx_backend::DFlash2;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
mx::array rows(const mx::array& x,int a,int b) {return mx::slice(x,{a,0},{b,x.shape(1)});}
int main(int argc,char** argv) {
    try {
        if(argc<4 || argc>9)throw std::invalid_argument("usage: tqmlx-dflash model.gguf draft.gguf prompt [width=7] [count=128] [reps=5] [shortlist=32768] [warmup=1]");
        const int width=argc>4?std::stoi(argv[4]):7,count=argc>5?std::stoi(argv[5]):128,reps=argc>6?std::stoi(argv[6]):5;
        const int shortlist=argc>7?std::stoi(argv[7]):32768,warmup=argc>8?std::stoi(argv[8]):1;
        if(width<1 || width>7 || count<2 || reps<1 || shortlist<16 || warmup<0)throw std::invalid_argument("invalid DFlash benchmark parameters");
        int prefill_chunk=0;
        if(const char* mode=std::getenv("TQLLM_MLX_PREFILL_CHUNK")) {
            prefill_chunk=std::stoi(mode);
            if(prefill_chunk<128)throw std::invalid_argument("TQLLM_MLX_PREFILL_CHUNK must be >=128");
        }
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);auto prompt=tokenizer.encode(argv[3]);
        if(prompt.empty())throw std::invalid_argument("empty prompt");
        mx::set_cache_limit(std::getenv("TQLLM_MLX_CACHE_LIMIT_MIB")?size_t(std::stoull(std::getenv("TQLLM_MLX_CACHE_LIMIT_MIB")))*1024*1024:size_t(512)*1024*1024);
        tqllm::mlx_backend::ScopedWiredLimit residency;
        Qwen35 engine(argv[1],int(prompt.size())+count+16);engine.gdn_mode(Qwen35::GdnMode::scan);engine.verify_tiles(true);
        if(std::getenv("TQLLM_MLX_PREFILL_GEMM"))engine.prefill_gemm(true);
        DFlash2 drafter(argv[2],engine);
        const bool head4=std::getenv("TQLLM_MLX_DRAFT_HEAD4")!=nullptr;
        if(head4)engine.enable_draft_head4();
        std::vector<uint32_t> reference,first_spec;
        for(int mode=0;mode<2;++mode)for(int run=-warmup;run<reps;++run) {
            engine.reset();drafter.reset();auto begin=Clock::now();mx::array logits=mx::array(0.0f);
            if(prefill_chunk>0 && int(prompt.size())>prefill_chunk) {
                for(size_t off=0;off<prompt.size();off+=size_t(prefill_chunk)) {
                    const int n=std::min(prefill_chunk,int(prompt.size()-off));
                    logits=engine.forward(std::span<const uint32_t>(prompt.data()+off,size_t(n)));
                    if(mode)drafter.inject(engine.feature_states(),int(off));
                }
            } else {
                logits=engine.forward(prompt);
            }
            if(mode) {if(!head4)engine.draft_shortlist(logits,shortlist);if(!(prefill_chunk>0 && int(prompt.size())>prefill_chunk))drafter.inject(engine.feature_states(),0);}
            const double prefill=seconds(begin);uint32_t pending=mx::argmax(logits).item<uint32_t>();
            std::vector<uint32_t> output{pending};int steps=0,drafted=0,accepted_total=0;
            double draft_ms=0,verify_ms=0,commit_ms=0,build_ms=0;const auto start=Clock::now();
            while(output.size()<size_t(count) && output.back()!=tokenizer.eos_id()) {
                if(!mode) {logits=engine.forward({&pending,1});pending=mx::argmax(logits).item<uint32_t>();output.push_back(pending);continue;}
                const int pos=engine.position(),want=std::min(width,int(count-output.size()));
                begin=Clock::now();auto proposed=drafter.draft(pending,pos,want);draft_ms+=seconds(begin)*1000;
                std::vector<uint32_t> block{pending};block.insert(block.end(),proposed.begin(),proposed.end());
                Qwen35::Verification ticket;begin=Clock::now();const bool direct_ids=std::getenv("TQLLM_MLX_TARGET_HEAD_IDS")&&std::getenv("TQLLM_MLX_TARGET_HEAD_SHORTLIST");
                auto choices=direct_ids?engine.verify_choices(block,ticket):mx::argmax(engine.verify(block,ticket),-1);
                mx::eval(choices);const auto* ids=choices.data<uint32_t>();
                auto features=engine.feature_states();verify_ms+=seconds(begin)*1000;build_ms+=engine.timings().first;
                int accepted=0;while(accepted<want && proposed[accepted]==ids[accepted])++accepted;
                ++steps;drafted+=want;accepted_total+=accepted;
                for(int j=0;j<accepted && output.size()<size_t(count);++j) {
                    output.push_back(proposed[j]);if(output.back()==tokenizer.eos_id())break;
                }
                if(output.size()<size_t(count) && output.back()!=tokenizer.eos_id())output.push_back(ids[accepted]);
                pending=ids[accepted];begin=Clock::now();engine.accept_prefix(ticket,accepted+1);
                drafter.inject(rows(features,0,accepted+1),pos);commit_ms+=seconds(begin)*1000;
            }
            const double decode=seconds(start);if(run<0)continue;
            auto& repeat=mode?first_spec:reference;if(repeat.empty())repeat=output;
            if(repeat!=output)throw std::runtime_error("DFlash reset changed greedy output");
            size_t equal=0;while(equal<std::min(output.size(),reference.size()) && output[equal]==reference[equal])++equal;
            std::cout<<"{\"kind\":\"dflash_generation\",\"mode\":\""<<(mode?"spec":"ar")<<"\",\"run\":"<<run
                     <<",\"width\":"<<width<<",\"proposal_head_bits\":"<<(head4?4:6)<<",\"shortlist\":"<<(head4?0:shortlist)<<",\"prompt_tokens\":"<<prompt.size()
                     <<",\"output_tokens\":"<<output.size()<<",\"decode_tok_s\":"<<(output.size()-1)/decode
                     <<",\"ttft_ms\":"<<prefill*1000<<",\"steps\":"<<steps<<",\"drafted\":"<<drafted<<",\"accepted\":"<<accepted_total
                     <<",\"draft_ms\":"<<draft_ms<<",\"verify_ms\":"<<verify_ms<<",\"commit_ms\":"<<commit_ms<<",\"verify_build_ms\":"<<build_ms
                     <<",\"ar_equal_prefix\":"<<equal<<",\"active_bytes\":"<<mx::get_active_memory()<<",\"token_ids\":[";
            for(size_t j=0;j<output.size();++j)std::cout<<(j?",":"")<<output[j];std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT mode="<<(mode?"spec":"ar")<<' '<<tokenizer.decode(output)<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<"DFLASH_ERROR "<<e.what()<<'\n';return 1;}
}

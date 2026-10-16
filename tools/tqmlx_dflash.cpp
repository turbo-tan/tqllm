#include "wired_limit.hpp"
#include "dflash.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <optional>
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
        const bool rank_stats=std::getenv("TQLLM_MLX_DFLASH_RANKSTATS")!=nullptr;
        std::vector<long> rank_hist(17,0),chosen_hist(17,0),rejection_at(8,0);long rejections=0;
        // TQLLM_MLX_DFLASH_SPEC_ONLY=1: speculative runs only (ar_equal_prefix then compares with nothing);
        // TQLLM_MLX_DFLASH_WARM_TOKENS=n: warmup runs stop after n tokens (Metal compiles are not billed to decode)
        const bool spec_only=std::getenv("TQLLM_MLX_DFLASH_SPEC_ONLY")!=nullptr;
        // TQLLM_MLX_DFLASH_PIPELINE=1: draft and verify chained on the GPU (needs TQLLM_MLX_DEVICE_SELECTOR)
        const bool pipeline=std::getenv("TQLLM_MLX_DFLASH_PIPELINE")!=nullptr && std::getenv("TQLLM_MLX_DEVICE_SELECTOR")!=nullptr;
        const int warm_tokens=std::getenv("TQLLM_MLX_DFLASH_WARM_TOKENS")?std::max(2,std::atoi(std::getenv("TQLLM_MLX_DFLASH_WARM_TOKENS"))):count;
        for(int mode=spec_only?1:0;mode<2;++mode)for(int run=-warmup;run<reps;++run) {
            const int limit=run<0?std::min(count,warm_tokens):count;
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
            double draft_ms=0,verify_ms=0,commit_ms=0,build_ms=0,step_ms=0,accept_ms=0;const auto start=Clock::now();
            while(output.size()<size_t(limit) && output.back()!=tokenizer.eos_id()) {
                if(!mode) {logits=engine.forward({&pending,1});pending=mx::argmax(logits).item<uint32_t>();output.push_back(pending);continue;}
                const int pos=engine.position(),want=std::min(width,int(limit-output.size()));
                const auto step_begin=Clock::now();
                Qwen35::Verification ticket;const bool direct_ids=std::getenv("TQLLM_MLX_TARGET_HEAD_IDS")&&std::getenv("TQLLM_MLX_TARGET_HEAD_SHORTLIST");
                std::vector<uint32_t> proposed;std::optional<mx::array> choices;
                if(pipeline) {
                    // draft -> verify as one graph: the draft ids never leave the GPU, one host sync per cycle
                    begin=Clock::now();
                    auto drafted=drafter.draft_device(pending,pos,want);
                    const uint32_t anchor[1]={pending};
                    auto block_ids=mx::concatenate({mx::array(anchor,{1},mx::uint32),drafted},0);
                    draft_ms+=seconds(begin)*1000;begin=Clock::now();
                    choices=direct_ids?engine.verify_device(block_ids,ticket,true):mx::argmax(engine.verify_device(block_ids,ticket),-1);
                    mx::eval(*choices,drafted);
                    proposed.assign(drafted.data<uint32_t>(),drafted.data<uint32_t>()+want);
                    for(auto id:proposed)if(id>=uint32_t(engine.vocabulary()))throw std::runtime_error("draft token outside vocabulary");
                } else {
                    begin=Clock::now();proposed=drafter.draft(pending,pos,want);draft_ms+=seconds(begin)*1000;
                    std::vector<uint32_t> block{pending};block.insert(block.end(),proposed.begin(),proposed.end());
                    begin=Clock::now();
                    choices=direct_ids?engine.verify_choices(block,ticket):mx::argmax(engine.verify(block,ticket),-1);
                    mx::eval(*choices);
                }
                const auto* ids=choices->data<uint32_t>();
                auto features=engine.feature_states();verify_ms+=seconds(begin)*1000;build_ms+=engine.timings().first;
                int accepted=0;while(accepted<want && proposed[accepted]==ids[accepted])++accepted;
                if(rank_stats && mode && run>=0 && accepted<want) {
                    const auto& c=drafter.candidates();int rank=16;
                    for(int j=0;j<16 && accepted<int(c.size());++j)if(c[accepted][j]==ids[accepted]){rank=j;break;}
                    int chosen=16;for(int j=0;j<16 && accepted<int(c.size());++j)if(c[accepted][j]==proposed[accepted]){chosen=j;break;}
                    ++rank_hist[rank];++chosen_hist[chosen];++rejections;rejection_at[accepted]++;
                }
                ++steps;drafted+=want;accepted_total+=accepted;
                for(int j=0;j<accepted && output.size()<size_t(limit);++j) {
                    output.push_back(proposed[j]);if(output.back()==tokenizer.eos_id())break;
                }
                if(output.size()<size_t(limit) && output.back()!=tokenizer.eos_id())output.push_back(ids[accepted]);
                pending=ids[accepted];begin=Clock::now();engine.accept_prefix(ticket,accepted+1);
                if(std::getenv("TQLLM_MLX_SPLIT_COMMIT_TIMING")){mx::synchronize();}accept_ms+=seconds(begin)*1000;
                drafter.inject(rows(features,0,accepted+1),pos);commit_ms+=seconds(begin)*1000;step_ms+=seconds(step_begin)*1000;
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
                     <<",\"step_ms\":"<<step_ms<<",\"accept_ms\":"<<accept_ms<<",\"decode_ms\":"<<decode*1000
                     <<",\"ar_equal_prefix\":"<<equal<<",\"active_bytes\":"<<mx::get_active_memory()<<",\"token_ids\":[";
            for(size_t j=0;j<output.size();++j)std::cout<<(j?",":"")<<output[j];std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT mode="<<(mode?"spec":"ar")<<' '<<tokenizer.decode(output)<<'\n';
        }
        if(rank_stats) {
            std::cout<<"{\"kind\":\"dflash_rank_stats\",\"rejections\":"<<rejections<<",\"target_rank\":[";
            for(int j=0;j<17;++j)std::cout<<(j?",":"")<<rank_hist[j];
            std::cout<<"],\"chosen_rank\":[";for(int j=0;j<17;++j)std::cout<<(j?",":"")<<chosen_hist[j];
            std::cout<<"],\"rejection_position\":[";for(int j=0;j<8;++j)std::cout<<(j?",":"")<<rejection_at[j];
            std::cout<<"]}"<<std::endl;
        }
    } catch(const std::exception& e) {std::cerr<<"DFLASH_ERROR "<<e.what()<<'\n';return 1;}
}

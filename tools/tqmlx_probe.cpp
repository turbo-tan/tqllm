#include "wired_limit.hpp"
#include "qwen35.hpp"
#include "mlx/memory.h"
#include "tqllm/tokenizer.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mx=mlx::core;
static void memory(const char* phase) {
    std::cerr << "MLX_MEMORY phase=" << phase << " active=" << mx::get_active_memory()
              << " cache=" << mx::get_cache_memory() << " peak=" << mx::get_peak_memory() << '\n';
}
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
int main(int argc,char** argv) {
    try {
        std::string model,prompt,tokens_path,logits_path;
        int count=32,repeat=1,warmup=0,chunk=64,context=32768;
        int verify_repeats=5,verify_width=0;
        bool ignore_eos=false,all_logits=false,profiling=false,prefill_gemm=false,clear_prefill_cache=false,verify_probe=false,verify_tiles=false,tq3_vector_load=false,stage_profiling=false,verify_prefix_probe=false,draft_head4=false;
        size_t cache_limit=0;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() {if(++i==argc)throw std::invalid_argument("missing argument value");return std::string(argv[i]);};
            if(arg=="--model") model=value();
            else if(arg=="--prompt") prompt=value();
            else if(arg=="--tokens-file") tokens_path=value();
            else if(arg=="--logits") logits_path=value();
            else if(arg=="--generate") count=std::stoi(value());
            else if(arg=="--repeat") repeat=std::stoi(value());
            else if(arg=="--warmup") warmup=std::stoi(value());
            else if(arg=="--prefill-chunk") chunk=std::stoi(value());
            else if(arg=="--context") context=std::stoi(value());
            else if(arg=="--ignore-eos") ignore_eos=true;
            else if(arg=="--all-logits") all_logits=true;
            else if(arg=="--profile") profiling=true;
            else if(arg=="--verify-probe") verify_probe=true;
            else if(arg=="--verify-prefix-probe") {verify_probe=true;verify_prefix_probe=true;}
            else if(arg=="--verify-repeats") verify_repeats=std::stoi(value());
            else if(arg=="--verify-width") verify_width=std::stoi(value());
            else if(arg=="--verify-tiles") verify_tiles=true;
            else if(arg=="--tq3-vector-load") tq3_vector_load=true;
            else if(arg=="--stage-profile") stage_profiling=true;
            else if(arg=="--draft-head4") draft_head4=true;
            else if(arg=="--prefill-gemm") prefill_gemm=true;
            else if(arg=="--clear-cache-after-prefill") clear_prefill_cache=true;
            else if(arg=="--cache-limit-mib") cache_limit=std::stoull(value())*1024*1024;
            else throw std::invalid_argument("unknown argument: "+arg);
        }
        if(model.empty()||count<1||repeat<1||warmup<0||chunk<1||context<1||verify_repeats<1||
           (verify_width!=0 && verify_width!=1 && verify_width!=2 && verify_width!=4 && verify_width!=8))
            throw std::invalid_argument("usage: tqmlx-probe --model file.gguf --prompt text [--generate N --repeat N --warmup N]");
        tqllm::Tokenizer tokenizer;tokenizer.load(model);
        std::vector<uint32_t> input;
        if(tokens_path.empty()) input=tokenizer.encode(prompt);
        else {std::ifstream in(tokens_path);uint32_t token;while(in>>token)input.push_back(token);if(!in.eof())throw std::runtime_error("invalid tokens file");}
        if(input.empty()||input.size()+size_t(count)>size_t(context))throw std::invalid_argument("prompt or output exceeds context");
        if(cache_limit)mx::set_cache_limit(cache_limit);
        tqllm::mlx_backend::ScopedWiredLimit residency;
        std::map<std::string,double> prefill_stages;
        auto started=Clock::now();tqllm::mlx_backend::Qwen35 engine(model,context);
        if(verify_prefix_probe)engine.gdn_mode(tqllm::mlx_backend::Qwen35::GdnMode::scan);
        engine.profile(profiling);
        engine.prefill_gemm(prefill_gemm);
        engine.verify_tiles(verify_tiles);
        engine.tq3_vector_load(tq3_vector_load);
        engine.profile_stages(stage_profiling);
        if(draft_head4)engine.enable_draft_head4();
        std::cerr<<"MLX_MODEL_LOADED seconds="<<elapsed(started)<<" vocabulary="<<engine.vocabulary()<<'\n';
        memory("loaded");
        if(verify_probe) {
            std::optional<mx::array> last;
            for(size_t p=0;p<input.size();p+=size_t(chunk))
                last=engine.forward({input.data()+p,std::min(size_t(chunk),input.size()-p)});
            const auto checkpoint=engine.checkpoint();
            const int start_pos=engine.position();
            uint32_t token=mx::argmax(*last).item<uint32_t>();
            std::vector<uint32_t> chain;
            std::vector<std::vector<float>> reference;
            for(int i=0;i<8;++i) {
                chain.push_back(token);
                auto y=engine.forward({&token,1});
                const float* p=y.data<float>();reference.emplace_back(p,p+y.size());
                token=mx::argmax(y).item<uint32_t>();
            }
            for(int width:{1,2,4,8}) {
                if(verify_width && width!=verify_width)continue;
                std::vector<double> milliseconds,bodies,heads;
                double max_diff=0,mean_kld=0;
                int same=0;
                for(int run=-1;run<verify_repeats;++run) {
                    engine.restore(checkpoint);
                    if(engine.position()!=start_pos)throw std::runtime_error("snapshot restored wrong position");
                    const auto begin=Clock::now();
                    tqllm::mlx_backend::Qwen35::Verification ticket;
                    auto y=verify_prefix_probe?engine.verify({chain.data(),size_t(width)},ticket):
                        engine.forward({chain.data(),size_t(width)},true);
                    const float* values=y.data<float>();
                    const double ms=elapsed(begin)*1000;
                    if(run>=0) {milliseconds.push_back(ms);const auto [body,head]=engine.timings();bodies.push_back(body);heads.push_back(head);}
                    if(run==0) {
                        if(stage_profiling) {
                            std::cerr<<"MLX_STAGE_PROFILE width="<<width<<" sync_per_stage=1";
                            for(const auto& [name,ms]:engine.stage_timings())std::cerr<<' '<<name<<"_ms="<<ms;
                            std::cerr<<'\n';
                        }
                        for(int row=0;row<width;++row) {
                            const auto& ref=reference[row];const float* got=values+size_t(row)*ref.size();
                            const double a=*std::max_element(ref.begin(),ref.end()),b=*std::max_element(got,got+ref.size());
                            double sa=0,sb=0,kld=0;
                            for(size_t j=0;j<ref.size();++j) {sa+=std::exp(double(ref[j])-a);sb+=std::exp(double(got[j])-b);max_diff=std::max(max_diff,std::abs(double(ref[j])-got[j]));}
                            for(size_t j=0;j<ref.size();++j) kld+=std::exp(double(ref[j])-a)/sa*((double(ref[j])-a-std::log(sa))-(double(got[j])-b-std::log(sb)));
                            mean_kld+=kld/width;
                            same+=std::max_element(ref.begin(),ref.end())-ref.begin()==std::max_element(got,got+ref.size())-got;
                        }
                    }
                }
                std::sort(milliseconds.begin(),milliseconds.end());std::sort(bodies.begin(),bodies.end());std::sort(heads.begin(),heads.end());const double ms=milliseconds[milliseconds.size()/2];
                std::cout<<"{\"kind\":\"verify_cost_only\",\"width\":"<<width<<",\"median_ms\":"<<ms
                         <<",\"body_ms\":"<<bodies[bodies.size()/2]<<",\"head_ms\":"<<heads[heads.size()/2]<<",\"profile_sync\":"<<profiling
                         <<",\"captures_prefix\":"<<verify_prefix_probe
                         <<",\"ideal_tokens_per_s\":"<<1000*width/ms<<",\"mean_kld_vs_sequential\":"<<mean_kld
                         <<",\"top1_same\":"<<same<<",\"max_logit_difference\":"<<max_diff<<"}"<<std::endl;
                if(width==1 && (max_diff!=0 || same!=1))throw std::runtime_error("snapshot replay changed sequential logits");
            }
            engine.reset();bool rejected=false;
            try {engine.restore(checkpoint);}catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected)throw std::runtime_error("snapshot survived reset");
            std::cerr<<"MLX_SNAPSHOT_REPLAY_PASS stale_reset_rejected=1\n";
            return 0;
        }
        std::vector<uint32_t> baseline;
        for(int run=-warmup;run<repeat;++run) {
            engine.reset();started=Clock::now();std::optional<mx::array> logits;
            for(size_t p=0;p<input.size();p+=size_t(chunk)) {
                const size_t n=std::min(size_t(chunk),input.size()-p);
                auto y=engine.forward({input.data()+p,n},all_logits);
                if(stage_profiling)for(const auto& [name,ms]:engine.stage_timings())prefill_stages[name]+=ms;
                if(all_logits && !logits_path.empty()) {
                    std::ofstream out(logits_path,(p?std::ios::app:std::ios::trunc)|std::ios::binary);
                    out.write(reinterpret_cast<const char*>(y.data<float>()),std::streamsize(y.size()*sizeof(float)));
                    if(!out)throw std::runtime_error("cannot write logits file");
                }
                logits=mx::reshape(mx::slice(y,{y.shape(0)-1,0},{y.shape(0),y.shape(1)}),{y.shape(1)});
            }
            const double prefill=elapsed(started);
            if(stage_profiling) {
                std::cerr<<"MLX_PREFILL_STAGES tokens="<<input.size()<<" sync_per_stage=1";
                for(const auto& [name,ms]:prefill_stages)std::cerr<<' '<<name<<"_ms="<<ms;
                std::cerr<<'\n';prefill_stages.clear();
            }
            memory("prefilled");
            if(clear_prefill_cache) {mx::clear_cache();memory("cache_cleared");}
            uint32_t next=mx::argmax(*logits).item<uint32_t>();
            std::vector<uint32_t> output={next};
            const auto first=Clock::now();
            for(int i=1;i<count && (ignore_eos || next!=tokenizer.eos_id());++i) {
                logits=engine.forward({&next,1});next=mx::argmax(*logits).item<uint32_t>();output.push_back(next);
                if(profiling) {
                    const auto [body,head]=engine.timings();
                    std::cerr<<"MLX_PROFILE body_ms="<<body<<" head_ms="<<head<<'\n';
                }
            }
            const double decode=elapsed(first);
            memory("decoded");
            if(run<0)continue;
            if(baseline.empty())baseline=output;
            if(output!=baseline)throw std::runtime_error("reset changed greedy output across repeats");
            std::cout<<"{\"kind\":\"native_mlx\",\"run\":"<<run<<",\"prompt_tokens\":"<<input.size()
                     <<",\"output_tokens\":"<<output.size()<<",\"prefill_tok_s\":"<<input.size()/prefill
                     <<",\"ttft_ms\":"<<prefill*1000<<",\"decode_tok_s\":"<<(output.size()-1)/decode<<",\"token_ids\":[";
            for(size_t i=0;i<output.size();++i)std::cout<<(i?",":"")<<output[i];
            std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT "<<tokenizer.decode(output)<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<"MLX_PROBE_ERROR "<<e.what()<<'\n';return 1;}
}

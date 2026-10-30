#include "wired_limit.hpp"
#include "qwen35.hpp"
#include "kl_rows.hpp"
#include <map>
#include "mlx/memory.h"
#include <cstdint>
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
        std::string model,prompt,tokens_path,logits_path,kl_reference;
        int count=32,repeat=1,warmup=0,chunk=64,context=32768;
        int verify_repeats=5,verify_width=0;
        bool ignore_eos=false,all_logits=false,profiling=false,clear_prefill_cache=false,verify_probe=false,stage_profiling=false,verify_prefix_probe=false,draft_head4=false;
        size_t cache_limit=0;int kl_ctx=0;double kl_cutoff=-INFINITY;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() {if(++i==argc)throw std::invalid_argument("missing argument value");return std::string(argv[i]);};
            if(arg=="--model") model=value();
            else if(arg=="--prompt") prompt=value();
            else if(arg=="--tokens-file") tokens_path=value();
            else if(arg=="--logits") logits_path=value();
            // --kl-reference FILE (with --all-logits): KL(reference || this run) per prompt position against FP32 logits
            // [rows, vocab] dumped by --logits from a reference run of the same tokens (e.g. the BF16 model)
            else if(arg=="--kl-reference") kl_reference=value();
            // --kl-ctx N with --kl-reference: llama.cpp --kl-divergence scoring - the tokens split into N-token chunks,
            // each evaluated from an empty cache, positions N/2 .. N-2 scored (N/2 - 1 reference rows per chunk)
            else if(arg=="--kl-ctx") kl_ctx=std::stoi(value());
            else if(arg=="--kl-cutoff") kl_cutoff=std::stod(value());  // e.g. -16: tqkld / llama.cpp KL (log p > cutoff)
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
            else if(arg=="--stage-profile") stage_profiling=true;
            else if(arg=="--draft-head4") draft_head4=true;
            else if(arg=="--clear-cache-after-prefill") clear_prefill_cache=true;
            else if(arg=="--cache-limit-mib") cache_limit=std::stoull(value())*1024*1024;
            else throw std::invalid_argument("unknown argument: "+arg);
        }
        if(model.empty()||count<1||repeat<1||warmup<0||chunk<1||context<1||verify_repeats<1||
           (verify_width!=0 && verify_width!=1 && verify_width!=2 && verify_width!=4 && verify_width!=8 && verify_width!=16))
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
        engine.profile(profiling);
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
            const int chain_len=verify_width==16?16:8;
            for(int i=0;i<chain_len;++i) {
                chain.push_back(token);
                auto y=engine.forward({&token,1});
                const float* p=y.data<float>();reference.emplace_back(p,p+y.size());
                token=mx::argmax(y).item<uint32_t>();
            }
            for(int width:{1,2,4,8,16}) {
                if((verify_width && width!=verify_width) || (width==16 && verify_width!=16))continue;
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
        if(kl_ctx>0) {
            if(kl_reference.empty() || kl_ctx<4 || kl_ctx%2 || size_t(kl_ctx)>size_t(context))
                throw std::invalid_argument("--kl-ctx needs --kl-reference and an even chunk no longer than --context");
            std::ifstream ref_in(kl_reference,std::ios::binary);
            if(!ref_in)throw std::runtime_error("cannot open the KL reference");
            const size_t chunks=input.size()/size_t(kl_ctx),first=size_t(kl_ctx/2),scored=size_t(kl_ctx/2-1);
            tqllm::KlRows kl;kl.min_log_p=kl_cutoff;kl.min_log_p=kl_cutoff;
            const auto t0=Clock::now();
            for(size_t c=0;c<chunks;++c) {
                engine.reset();
                const uint32_t* ids=input.data()+c*size_t(kl_ctx);
                std::vector<mx::array> parts;
                for(int at=0;at<kl_ctx;at+=chunk)
                    parts.push_back(engine.forward({ids+at,size_t(std::min(chunk,kl_ctx-at))},true));
                auto y=mx::astype(mx::concatenate(parts,0),mx::float32);mx::eval(y);
                const size_t vocab=size_t(y.shape(1));
                std::vector<float> ref(scored*vocab);
                ref_in.read(reinterpret_cast<char*>(ref.data()),std::streamsize(ref.size()*sizeof(float)));
                if(!ref_in)throw std::runtime_error("KL reference shorter than chunks x scored rows");
                std::vector<uint32_t> targets(ids+first+1,ids+first+1+scored);
                kl.add(ref.data(),y.data<float>()+first*vocab,scored,vocab,targets.data());
            }
            std::cout<<"{\"kind\":\"kl_reference\",\"chunks\":"<<chunks<<",\"ctx\":"<<kl_ctx<<",\"rows\":"<<kl.rows<<",\"mean_kl\":"<<kl.mean()
                     <<",\"max_kl\":"<<kl.kl_max<<",\"top1_same\":"<<double(kl.top1_same)/double(std::max<size_t>(kl.rows,1))
                     <<",\"ln_ppl_ratio\":"<<(kl.targets?kl.nll_gap_sum/double(kl.targets):0.0)<<",\"cutoff\":"<<kl_cutoff<<",\"seconds\":"<<elapsed(t0)<<"}"<<std::endl;
            return 0;
        }
        std::vector<uint32_t> baseline;
        for(int run=-warmup;run<repeat;++run) {
            engine.reset();started=Clock::now();std::optional<mx::array> logits;
            if(!kl_reference.empty() && !all_logits)throw std::invalid_argument("--kl-reference needs --all-logits");
            std::ifstream kl_ref;if(!kl_reference.empty())kl_ref.open(kl_reference,std::ios::binary);
            if(!kl_reference.empty() && !kl_ref)throw std::runtime_error("cannot open the KL reference");
            tqllm::KlRows kl;kl.min_log_p=kl_cutoff;
            std::map<size_t,std::pair<double,size_t>> slice_kl;std::map<size_t,size_t> slice_bad;size_t slice_width=0;
            for(size_t p=0;p<input.size();p+=size_t(chunk)) {
                const size_t n=std::min(size_t(chunk),input.size()-p);
                auto y=engine.forward({input.data()+p,n},all_logits);
                if(stage_profiling)for(const auto& [name,ms]:engine.stage_timings())prefill_stages[name]+=ms;
                if(all_logits && kl_ref.is_open()) {  // an unopened ifstream tests true: --logits alone must not read a reference
                    const size_t vocab=size_t(y.shape(1)),rows_here=size_t(y.shape(0));
                    std::vector<float> ref(rows_here*vocab);
                    kl_ref.read(reinterpret_cast<char*>(ref.data()),std::streamsize(ref.size()*sizeof(float)));
                    if(!kl_ref)throw std::runtime_error("KL reference shorter than this run's logits");
                    auto yf=mx::astype(y,mx::float32);mx::eval(yf);
                    std::vector<uint32_t> targets;
                    for(size_t r=0;r<rows_here && p+r+1<input.size();++r)targets.push_back(input[p+r+1]);
                    kl.add(ref.data(),yf.data<float>(),targets.size(),vocab,targets.data());
                    if(targets.size()<rows_here)kl.add(ref.data()+targets.size()*vocab,yf.data<float>()+targets.size()*vocab,rows_here-targets.size(),vocab);
                    // TQLLM_KL_SLICE=N: mean KL per N-position slice of the prompt (printed after the run), to see whether the
                    // error sits in one slice (a join / padded tail) or follows tokens
                    if(const char* ks=std::getenv("TQLLM_KL_SLICE")) {
                        const size_t width=std::max<size_t>(1,size_t(std::atoi(ks)));
                        for(size_t r=0;r<rows_here;++r){
                            const float* a=ref.data()+r*vocab;const float* b=yf.data<float>()+r*vocab;
                            const double ma=*std::max_element(a,a+vocab),mb=*std::max_element(b,b+vocab);
                            double sa=0,sb=0;for(size_t j=0;j<vocab;++j){sa+=std::exp(double(a[j])-ma);sb+=std::exp(double(b[j])-mb);}
                            const double za=ma+std::log(sa),zb=mb+std::log(sb);double klr=0;
                            for(size_t j=0;j<vocab;++j){const double la=double(a[j])-za;klr+=std::exp(la)*(la-(double(b[j])-zb));}
                            auto& cell=slice_kl[(p+r)/width];cell.first+=klr;cell.second+=1;
                            if(klr>1.0)++slice_bad[(p+r)/width];
                        }
                        slice_width=width;
                    }
                    // TQLLM_KL_ROWS=K: print the K worst rows of this chunk by KL (absolute position), to find bad rows
                    if(const char* kr=std::getenv("TQLLM_KL_ROWS")) {
                        const int topk=std::max(1,std::atoi(kr));
                        std::vector<std::pair<double,size_t>> rowkl;
                        for(size_t r=0;r<rows_here;++r){
                            const float* a=ref.data()+r*vocab;const float* b=yf.data<float>()+r*vocab;
                            const double ma=*std::max_element(a,a+vocab),mb=*std::max_element(b,b+vocab);
                            double sa=0,sb=0;for(size_t j=0;j<vocab;++j){sa+=std::exp(double(a[j])-ma);sb+=std::exp(double(b[j])-mb);}
                            const double za=ma+std::log(sa),zb=mb+std::log(sb);double klr=0;
                            for(size_t j=0;j<vocab;++j){const double la=double(a[j])-za;klr+=std::exp(la)*(la-(double(b[j])-zb));}
                            rowkl.push_back({klr,p+r});
                        }
                        std::partial_sort(rowkl.begin(),rowkl.begin()+std::min<size_t>(topk,rowkl.size()),rowkl.end(),
                                          [](auto&x,auto&y){return x.first>y.first;});
                        for(int k=0;k<topk && k<int(rowkl.size());++k)
                            std::fprintf(stderr,"KL_ROW pos=%zu kl=%.4f\n",rowkl[k].second,rowkl[k].first);
                    }
                }
                if(all_logits && !logits_path.empty()) {
                    std::ofstream out(logits_path,(p?std::ios::app:std::ios::trunc)|std::ios::binary);
                    out.write(reinterpret_cast<const char*>(y.data<float>()),std::streamsize(y.size()*sizeof(float)));
                    if(!out)throw std::runtime_error("cannot write logits file");
                }
                logits=mx::reshape(mx::slice(y,{y.shape(0)-1,0},{y.shape(0),y.shape(1)}),{y.shape(1)});
            }
            const double prefill=elapsed(started);
            for(const auto& [i,c]:slice_kl)
                std::fprintf(stderr,"KL_SLICE positions %zu-%zu rows %zu mean_kl %.4f rows_kl_gt_1 %zu\n",i*slice_width,i*slice_width+c.second-1,c.second,c.first/double(c.second),slice_bad[i]);
            if(kl_ref)std::cout<<"{\"kind\":\"kl_reference\",\"rows\":"<<kl.rows<<",\"mean_kl\":"<<kl.mean()<<",\"max_kl\":"<<kl.kl_max
                               <<",\"top1_same\":"<<double(kl.top1_same)/double(std::max<size_t>(kl.rows,1))
                               <<",\"ln_ppl_ratio\":"<<(kl.targets?kl.nll_gap_sum/double(kl.targets):0.0)
                               <<",\"ref_nll\":"<<(kl.targets?kl.nll_ref_sum/double(kl.targets):0.0)<<",\"cand_nll\":"<<(kl.targets?kl.nll_cand_sum/double(kl.targets):0.0)<<"}"<<std::endl;
            // FNV-1a over the last prefill row's logits: equal hashes = bitwise equal prefill (chunking / layer-major A/B)
            uint64_t logits_hash=1469598103934665603ull;
            {auto v=mx::astype(*logits,mx::float32);mx::eval(v);const auto* b=reinterpret_cast<const unsigned char*>(v.data<float>());
             for(size_t i=0;i<v.size()*sizeof(float);++i){logits_hash^=b[i];logits_hash*=1099511628211ull;}}
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
                     <<",\"ttft_ms\":"<<prefill*1000<<",\"decode_tok_s\":"<<(output.size()-1)/decode<<",\"prefill_logits_fnv\":\""<<std::hex<<logits_hash<<std::dec<<"\",\"token_ids\":[";
            for(size_t i=0;i<output.size();++i)std::cout<<(i?",":"")<<output[i];
            std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT "<<tokenizer.decode(output)<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<"MLX_PROBE_ERROR "<<e.what()<<'\n';return 1;}
}

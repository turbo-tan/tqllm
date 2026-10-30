#include "wired_limit.hpp"
#include "dflash.hpp"
#include "dflash_rank_stats.hpp"
#include "spec_pipeline.hpp"
#include "width_policy.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <optional>
#include <span>
#include <string>
#include <vector>
namespace mx=mlx::core;
using tqllm::mlx_backend::Qwen35;
using tqllm::mlx_backend::DFlash2;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
// host CPU time of this thread (MLX encodes command buffers on the calling thread), ms
double cpu_ms() {timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
mx::array rows(const mx::array& x,int a,int b) {return mx::slice(x,{a,0},{b,x.shape(1)});}
int main(int argc,char** argv) {
    try {
        if(argc<4 || argc>9)throw std::invalid_argument("usage: tqmlx-dflash model.gguf draft.gguf prompt [width=7, 0 = per-step 7 / 15 (WidthPolicy)] [count=128] [reps=5] [shortlist=32768] [warmup=1]");
        const int width_arg=argc>4?std::stoi(argv[4]):7,width=width_arg==0?15:width_arg,count=argc>5?std::stoi(argv[5]):128,reps=argc>6?std::stoi(argv[6]):5;
        const int shortlist=argc>7?std::stoi(argv[7]):32768,warmup=argc>8?std::stoi(argv[8]):1;
        if(width<1 || width>15 || count<2 || reps<1 || shortlist<16 || warmup<0)throw std::invalid_argument("invalid DFlash benchmark parameters");
        int prefill_chunk=8192;  // the server's chunk (model_mlx.cpp): layer-major in 2048-row slices; a whole-prompt prefill at 24K needs a 53 GB score buffer (j729)
        if(const char* mode=std::getenv("TQLLM_MLX_PREFILL_CHUNK")) {
            prefill_chunk=std::stoi(mode);
            if(prefill_chunk<128)throw std::invalid_argument("TQLLM_MLX_PREFILL_CHUNK must be >=128");
        }
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);auto prompt=tokenizer.encode(argv[3]);
        if(prompt.empty())throw std::invalid_argument("empty prompt");
        mx::set_cache_limit(std::getenv("TQLLM_MLX_CACHE_LIMIT_MIB")?size_t(std::stoull(std::getenv("TQLLM_MLX_CACHE_LIMIT_MIB")))*1024*1024:size_t(512)*1024*1024);
        tqllm::mlx_backend::ScopedWiredLimit residency;
        Qwen35 engine(argv[1],int(prompt.size())+count+16);
        DFlash2 drafter(argv[2],engine);
        const bool head4=std::getenv("TQLLM_MLX_DRAFT_HEAD4") && std::string(std::getenv("TQLLM_MLX_DRAFT_HEAD4"))!="0";
        if(head4)engine.enable_draft_head4();
        // TQLLM_MLX_DRAFT_BENCH=n: after the prefill, n synchronous draft_device calls of `width` rows at the same
        // position; prints the median ms (drafter GPU + host) and exits
        if(const char* bench=std::getenv("TQLLM_MLX_DRAFT_BENCH")) {
            mx::array logits=mx::array(0.0f);
            for(size_t off=0;off<prompt.size();off+=size_t(prefill_chunk)) {
                const int n=std::min(prefill_chunk,int(prompt.size()-off));
                logits=engine.forward(std::span<const uint32_t>(prompt.data()+off,size_t(n)));
                drafter.inject(engine.feature_states(),int(off));
            }
            if(!head4)engine.draft_shortlist(logits,shortlist);
            const uint32_t pending=mx::argmax(logits).item<uint32_t>();
            std::vector<double> ms;
            // TQLLM_MLX_DRAFT_STAGES: time draft_device_at (the draft-ahead draft) per stage instead
            const bool stage_mode=std::getenv("TQLLM_MLX_DRAFT_STAGES")!=nullptr;
            for(int r=-3;r<std::max(1,std::atoi(bench));++r) {
                const auto t0=Clock::now();
                auto d=stage_mode?drafter.draft_device_at(mx::array(pending),mx::array(engine.position(),mx::int32),width):
                    drafter.draft_device(pending,engine.position(),width);
                mx::eval(d);
                if(r>=0)ms.push_back(seconds(t0)*1000);
            }
            std::sort(ms.begin(),ms.end());
            std::cout<<"{\"kind\":\"draft_bench\",\"prompt_tokens\":"<<prompt.size()<<",\"width\":"<<width<<",\"median_ms\":"<<ms[ms.size()/2]
                     <<",\"min_ms\":"<<ms.front();
            if(stage_mode) {const auto st=drafter.stage_ms_per_call();
                std::cout<<",\"layers_ms\":"<<st[0]<<",\"head_ms\":"<<st[1]<<",\"top16_ms\":"<<st[2]<<",\"selector_ms\":"<<st[3];}
            std::cout<<"}"<<std::endl;
            return 0;
        }
        std::vector<uint32_t> reference,first_spec;
        const bool rank_stats=std::getenv("TQLLM_MLX_DFLASH_RANKSTATS")!=nullptr;
        tqllm::DraftRankStats rank_table(width);
        // TQLLM_MLX_DFLASH_SPEC_ONLY=1: speculative runs only (ar_equal_prefix then compares with nothing);
        // TQLLM_MLX_DFLASH_WARM_TOKENS=n: warmup runs stop after n tokens (Metal compiles are not billed to decode)
        const bool spec_only=std::getenv("TQLLM_MLX_DFLASH_SPEC_ONLY")!=nullptr;
        // TQLLM_MLX_DRAFT_AHEAD=1: verify(k) and draft(k+1) chained on the GPU from device values (spec_pipeline.hpp)
        const bool draft_ahead=std::getenv("TQLLM_MLX_DRAFT_AHEAD") && std::string(std::getenv("TQLLM_MLX_DRAFT_AHEAD"))!="0";
        const int warm_tokens=std::getenv("TQLLM_MLX_DFLASH_WARM_TOKENS")?std::max(2,std::atoi(std::getenv("TQLLM_MLX_DFLASH_WARM_TOKENS"))):count;
        tqllm::mlx_backend::WidthPolicy policy;  // width 0: the per-step verify width, its time lines kept across runs
        for(int mode=spec_only?1:0;mode<2;++mode)for(int run=-warmup;run<reps;++run) {
            const int limit=run<0?std::min(count,warm_tokens):count;
            engine.reset();drafter.reset();engine.reserve_kv(int(prompt.size())+count+16);
            engine.kv_format(engine.choose_kv_format(int(prompt.size())+count+16,prefill_chunk>0?prefill_chunk:int(prompt.size())));auto begin=Clock::now();mx::array logits=mx::array(0.0f);
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
            double draft_ms=0,verify_ms=0,commit_ms=0,build_ms=0,step_ms=0,accept_ms=0,submit_ms=0,wait_ms=0,draft_cpu=0,verify_cpu=0;const auto start=Clock::now();
            std::optional<tqllm::mlx_backend::DraftAhead> ahead;
            if(mode && draft_ahead)ahead.emplace(engine,drafter,width);
            policy.begin(int(prompt.size()));
            std::vector<int> trace_width,trace_accepted;std::vector<double> trace_ms;  // per step: verify width, acceptance at that width, step ms
            while(output.size()<size_t(limit) && output.back()!=tokenizer.eos_id()) {
                if(!mode) {logits=engine.forward({&pending,1});pending=mx::argmax(logits).item<uint32_t>();output.push_back(pending);continue;}
                const int verify=width_arg==0?policy.choose(engine.position()):width;
                const int pos=engine.position(),want=std::min(verify,int(limit-output.size()));
                const auto step_begin=Clock::now();
                if(ahead) {
                    if(!ahead->active())ahead->start(pending);
                    begin=Clock::now();const auto st=ahead->step(verify);verify_ms+=seconds(begin)*1000;
                    build_ms+=st.build_ms;submit_ms+=st.submit_ms;wait_ms+=st.wait_ms;accept_ms+=st.post_ms;
                    const int accepted=std::min(st.accepted,want);
                    ++steps;drafted+=want;accepted_total+=accepted;
                    for(int j=0;j<accepted && output.size()<size_t(limit);++j) {
                        output.push_back(st.proposed[size_t(j)]);if(output.back()==tokenizer.eos_id())break;
                    }
                    if(output.size()<size_t(limit) && output.back()!=tokenizer.eos_id())output.push_back(st.choices[size_t(accepted)]);
                    pending=st.choices[size_t(accepted)];step_ms+=seconds(step_begin)*1000;
                    if(want==verify)policy.record(verify,st.accepted,pos,seconds(step_begin)*1000);
                    trace_width.push_back(want);trace_accepted.push_back(st.accepted);trace_ms.push_back(seconds(step_begin)*1000);
                    continue;
                }
                Qwen35::Verification ticket;
                std::vector<uint32_t> proposed;std::optional<mx::array> choices;
                {
                    // draft -> verify as one graph: the draft ids never leave the GPU, one host sync per cycle
                    begin=Clock::now();double c0=cpu_ms();
                    auto drafted=drafter.draft_device(pending,pos,want);
                    // submit the draft now; the verify graph is built while the GPU runs it
                    {const auto t0=Clock::now();mx::async_eval(drafted);submit_ms+=seconds(t0)*1000;}
                    const uint32_t anchor[1]={pending};
                    auto block_ids=mx::concatenate({mx::array(anchor,{1},mx::uint32),drafted},0);
                    draft_ms+=seconds(begin)*1000;begin=Clock::now();{const double c1=cpu_ms();draft_cpu+=c1-c0;c0=c1;}
                    choices=mx::argmax(engine.verify_device(block_ids,ticket),-1);
                    {const auto t0=Clock::now();mx::eval(*choices,drafted);wait_ms+=seconds(t0)*1000;}
                    verify_cpu+=cpu_ms()-c0;
                    proposed.assign(drafted.data<uint32_t>(),drafted.data<uint32_t>()+want);
                    for(auto id:proposed)if(id>=uint32_t(engine.vocabulary()))throw std::runtime_error("draft token outside vocabulary");
                }
                const auto* ids=choices->data<uint32_t>();
                auto features=engine.feature_states();verify_ms+=seconds(begin)*1000;build_ms+=engine.timings().first;
                int accepted=0;while(accepted<want && proposed[accepted]==ids[accepted])++accepted;
                if(rank_stats && mode && run>=0 && accepted<want) {
                    rank_table.record(drafter.candidates(),accepted,ids[accepted],proposed[accepted]);
                }
                ++steps;drafted+=want;accepted_total+=accepted;
                for(int j=0;j<accepted && output.size()<size_t(limit);++j) {
                    output.push_back(proposed[j]);if(output.back()==tokenizer.eos_id())break;
                }
                if(output.size()<size_t(limit) && output.back()!=tokenizer.eos_id())output.push_back(ids[accepted]);
                pending=ids[accepted];begin=Clock::now();engine.accept_prefix(ticket,accepted+1);
                accept_ms+=seconds(begin)*1000;
                drafter.inject(rows(features,0,accepted+1),pos);commit_ms+=seconds(begin)*1000;step_ms+=seconds(step_begin)*1000;
                if(want==verify)policy.record(verify,accepted,pos,seconds(step_begin)*1000);
                trace_width.push_back(want);trace_accepted.push_back(accepted);trace_ms.push_back(seconds(step_begin)*1000);
            }
            const double decode=seconds(start);if(run<0)continue;
            auto& repeat=mode?first_spec:reference;if(repeat.empty())repeat=output;
            if(repeat!=output)throw std::runtime_error("DFlash reset changed greedy output");
            size_t equal=0;while(equal<std::min(output.size(),reference.size()) && output[equal]==reference[equal])++equal;
            std::cout<<"{\"kind\":\"dflash_generation\",\"mode\":\""<<(mode?"spec":"ar")<<"\",\"run\":"<<run<<",\"narrow_steps\":"<<policy.narrow_steps()<<",\"wide_steps\":"<<policy.wide_steps()
                     <<",\"width\":"<<width<<",\"proposal_head_bits\":"<<(head4?4:6)<<",\"shortlist\":"<<(head4?0:shortlist)<<",\"prompt_tokens\":"<<prompt.size()
                     <<",\"output_tokens\":"<<output.size()<<",\"decode_tok_s\":"<<(output.size()-1)/decode
                     <<",\"ttft_ms\":"<<prefill*1000<<",\"steps\":"<<steps<<",\"drafted\":"<<drafted<<",\"accepted\":"<<accepted_total
                     <<",\"draft_ms\":"<<draft_ms<<",\"verify_ms\":"<<verify_ms<<",\"commit_ms\":"<<commit_ms<<",\"verify_build_ms\":"<<build_ms
                     <<",\"step_ms\":"<<step_ms<<",\"accept_ms\":"<<accept_ms<<",\"submit_ms\":"<<submit_ms<<",\"wait_ms\":"<<wait_ms<<",\"draft_cpu_ms\":"<<draft_cpu<<",\"verify_cpu_ms\":"<<verify_cpu<<",\"decode_ms\":"<<decode*1000
                     <<",\"ar_equal_prefix\":"<<equal<<",\"active_bytes\":"<<mx::get_active_memory();
            std::cout<<",\"trace_width\":[";for(size_t j=0;j<trace_width.size();++j)std::cout<<(j?",":"")<<trace_width[j];
            std::cout<<"],\"trace_accepted\":[";for(size_t j=0;j<trace_accepted.size();++j)std::cout<<(j?",":"")<<trace_accepted[j];
            std::cout<<"],\"trace_ms\":[";for(size_t j=0;j<trace_ms.size();++j)std::cout<<(j?",":"")<<trace_ms[j];
            std::cout<<"],\"token_ids\":[";
            for(size_t j=0;j<output.size();++j)std::cout<<(j?",":"")<<output[j];std::cout<<"]}"<<std::endl;
            std::cerr<<"GENERATED_TEXT mode="<<(mode?"spec":"ar")<<' '<<tokenizer.decode(output)<<'\n';
        }
        if(rank_stats) {
            rank_table.write_json(std::cout);std::cout<<std::endl;
        }
    } catch(const std::exception& e) {std::cerr<<"DFLASH_ERROR "<<e.what()<<'\n';return 1;}
}

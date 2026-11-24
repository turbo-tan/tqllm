#include "inkling2_mlx.hpp"
#include "inkling2_turns.hpp"
#include "dequant.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using tqllm::mlx_backend::Inkling2Mlx;
using tqllm::mlx_backend::inkling2_prefill_chunks;
namespace {
template<class T> std::vector<T> load(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if (!f || f.tellg()<0 || size_t(f.tellg())%sizeof(T)) throw std::runtime_error("invalid input "+path);
    std::vector<T> v(size_t(f.tellg())/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),std::streamsize(v.size()*sizeof(T)));
    if (!f) throw std::runtime_error("short read "+path);
    return v;
}
double seconds(std::chrono::steady_clock::time_point start) {return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
uint32_t best(const mx::array& x) {return mx::argmax(x,-1).item<uint32_t>();}
mx::array prefill(Inkling2Mlx& m,std::span<const uint32_t> ids,int chunk) {
    std::optional<mx::array> logits;
    for (size_t p=0;p<ids.size();p+=size_t(chunk)) logits=m.forward(ids.subspan(p,std::min(size_t(chunk),ids.size()-p)));
    if (!logits) throw std::runtime_error("empty prompt");
    return *logits;
}
// Greedy tokens after `first` (the prefill's argmax): `count` of them, plain (pipelined greedy_step) or DSpark
// speculative (depth > 0 fixed drafts per step, 0 the model's adaptive depth, as the engine). steps: speculative steps taken.
std::vector<uint32_t> plain_tokens(Inkling2Mlx& m,uint32_t first,int count) {
    std::vector<uint32_t> out;
    auto pending=m.greedy_step(mx::array(&first,{1},mx::uint32));
    for(int i=0;i<count;++i) {
        std::optional<mx::array> following;
        if(i+1<count) following=m.greedy_step(pending);
        out.push_back(pending.item<uint32_t>());
        if(following) pending=*following;
    }
    return out;
}
std::vector<uint32_t> spec_tokens(Inkling2Mlx& m,uint32_t first,int count,int depth,int& steps,int& accepted) {
    std::vector<uint32_t> out;uint32_t anchor=first;steps=accepted=0;
    while(int(out.size())<count) {
        auto s=m.speculate(anchor,depth);++steps;accepted+=s.accepted;
        for(auto t:s.tokens) if(int(out.size())<count) out.push_back(t);
        anchor=s.tokens.back();
    }
    return out;
}
int full_gate(const std::string& model,const std::string& dir,int chunk,bool exact,int bits) {
    auto packed=load<uint32_t>(dir+"/texts.bin");
    if(packed.empty()) throw std::runtime_error("empty multi-text input");
    std::vector<std::vector<uint32_t>> texts;size_t offset=1,total=0,maxn=0;
    for(uint32_t i=0;i<packed[0];++i) {
        if(offset>=packed.size()) throw std::runtime_error("truncated multi-text header");
        const auto n=packed[offset++];if(!n || n>packed.size()-offset) throw std::runtime_error("invalid multi-text length");
        texts.emplace_back(packed.begin()+offset,packed.begin()+offset+n);offset+=n;total+=n;maxn=std::max(maxn,size_t(n));
    }
    if(offset!=packed.size()) throw std::runtime_error("trailing multi-text data");
    auto top=load<uint32_t>(dir+"/top1_bf16.i32");if(top.size()!=total) throw std::runtime_error("multi-text top1 count");
    Inkling2Mlx m(model,int(maxn)+64,exact,bits);const int V=m.vocabulary();
    std::ifstream file(dir+"/lp_bf16.f16",std::ios::binary|std::ios::ate);
    if(!file || size_t(file.tellg())!=total*size_t(V)*2) throw std::runtime_error("full-vocabulary reference dimensions");file.seekg(0);
    std::vector<uint16_t> half(V);std::vector<float> ref(V);size_t seen=0,agree=0,nll_n=0;double kl_all=0,nll=0;
    for(size_t text=0;text<texts.size();++text) {
        m.reset();const auto& ids=texts[text];double kl=0;size_t same=0;
        for(size_t p=0;p<ids.size();p+=size_t(chunk)) {
            const auto n=std::min(size_t(chunk),ids.size()-p);auto logits=m.forward(std::span<const uint32_t>(ids).subspan(p,n),true);
            mx::eval(logits);const float* data=logits.data<float>();
            for(size_t row=0;row<n;++row) {
                file.read(reinterpret_cast<char*>(half.data()),std::streamsize(V)*2);if(!file) throw std::runtime_error("short full-vocabulary reference");
                const float* x=data+row*V;double mq=-INFINITY,mr=-INFINITY;uint32_t am=0;
                for(int v=0;v<V;++v) {if(!std::isfinite(x[v])) throw std::runtime_error("non-finite native logits");if(x[v]>mq){mq=x[v];am=uint32_t(v);}ref[v]=tqllm::half_to_float(half[v]);mr=std::max(mr,double(ref[v]));}
                double zq=0,zr=0;for(int v=0;v<V;++v){zq+=std::exp(double(x[v])-mq);zr+=std::exp(double(ref[v])-mr);}
                const double lq=mq+std::log(zq),lr=mr+std::log(zr);
                for(int v=0;v<V;++v){const double r=double(ref[v])-lr;if(r> -80)kl+=std::exp(r)*(r-(double(x[v])-lq));}
                same+=am==top[seen++];if(p+row+1<ids.size()){nll+=lq-x[ids[p+row+1]];++nll_n;}
            }
        }
        kl_all+=kl;agree+=same;
        std::printf("MT text %zu positions %zu full KLD %.7f top1 %.4f%%\n",text,ids.size(),kl/ids.size(),100.*same/ids.size());std::fflush(stdout);
    }
    const bool pass=seen>=2000 && kl_all/seen<=.01 && double(agree)/seen>=.98;
    std::printf("MT_ALL positions %zu full-vocabulary KLD %.7f top1 %.4f%% PPL %.5f %s\n",seen,kl_all/seen,100.*agree/seen,std::exp(nll/nll_n),pass?"PASS":"FAIL");
    return pass?0:1;
}
}
int main(int argc,char** argv) {
    try {
        if (argc<3) {std::fprintf(stderr,"usage: inkling2-mlx-check MODEL GOLDEN_BIN [--chunk N] [--decode N] [--bench DEPTH] [--greedy-only]\n");return 2;}
        std::string drafter;  // --draft PATH: a DSpark drafter GGUF or a DFlash2 drafter folder instead of the checkpoint's mtp.safetensors
        int chunk=64,decode=64,bench=0,spec=-1,runs=3;bool greedy=false,flips=false,onestep=false,mt=false,profile=false,exact=true,blocks=false;int bits=8,rowcheck=0,turns=0;Inkling2Mlx::DraftPrecision draft_precision;
        for (int i=3;i<argc;++i) {
            std::string a=argv[i];
            if (a=="--greedy-only") greedy=true;
            else if(a=="--flips") flips=true;
            else if(a=="--profile") profile=true;
            else if(a=="--blocks") blocks=true;  // with --bench: wall time of one forward of 1..8 tokens (the verify block's cost)
            else if(a=="--onestep") onestep=true;
            // with --bench: a block of N tokens run as a verify block against N single-token forwards from the same
            // prefill, visit by visit (spec == plain needs row j of the block bitwise equal to the j-th single step)
            else if(i+1<argc && a=="--rowcheck") rowcheck=std::stoi(argv[++i]);
            // turn reuse: a conversation of N tokens, then a follow-up (an answer message + a user message): the follow-up
            // resumed from the kept state must give a cold prefill's bits (first logits, 64 greedy tokens); prints both TTFTs
            else if(i+1<argc && a=="--turns") turns=std::stoi(argv[++i]);
            else if(a=="--draft-bf16-proj") draft_precision.bf16_projections=true;   // drafter A/B (see DraftPrecision)
            else if(a=="--draft-bf16-head") draft_precision.bf16_head=true;
            else if(i+1<argc && a=="--mt") {if(std::string(argv[++i])!="bf16") throw std::runtime_error("only pinned BF16 reference staged");mt=true;}
            else if (i+1<argc && a=="--chunk") chunk=std::stoi(argv[++i]);
            else if (i+1<argc && a=="--decode") decode=std::stoi(argv[++i]);
            else if (i+1<argc && a=="--bench") bench=std::stoi(argv[++i]);
            else if (i+1<argc && a=="--spec") spec=std::stoi(argv[++i]);  // DSpark drafts per step (0: adaptive)
            else if (i+1<argc && a=="--draft") drafter=argv[++i];
            else if (i+1<argc && a=="--runs") runs=std::stoi(argv[++i]);  // --bench repetitions (default 3)
            else if (i+1<argc && a=="--exact") {const std::string v=argv[++i];if(v!="on" && v!="off") throw std::runtime_error("--exact on|off");exact=v=="on";}
            else if (i+1<argc && a=="--dense-bits") {bits=std::stoi(argv[++i]);if(bits!=8 && bits!=4) throw std::runtime_error("--dense-bits 8|4");}
            else throw std::runtime_error("unknown option "+a);
        }
        if (chunk<=0 || decode<=0 || bench<0 || runs<=0) throw std::runtime_error("invalid timing configuration");
        mx::set_default_device(mx::Device::gpu);
        mx::set_cache_limit(size_t(512)*1024*1024);
        const std::string model=argv[1],dir=argv[2];
        if(mt) return full_gate(model,dir,chunk,exact,bits);
        auto ids=load<uint32_t>(dir+"/tf_ids.i32");
        Inkling2Mlx m(model,std::max(bench+decode+16,int(ids.size())+1024),exact,bits);
        if(spec>=0) m.enable_draft(drafter.empty()?model:drafter,draft_precision);
        std::printf("MODEL_READY positions %zu vocab %d\n",ids.size(),m.vocabulary());std::fflush(stdout);
        if(onestep) {
            if(ids.size()<64) throw std::runtime_error("visit trace needs first64 prompt ids");
            auto reference=load<float>(dir+"/visits_first64.f32");std::vector<mx::array> trace;
            m.forward(std::span<const uint32_t>(ids).first(64),false,&trace);
            const size_t width=size_t(64)*m.hidden_size();
            if(reference.size()!=trace.size()*width) throw std::runtime_error("visit trace reference count "+std::to_string(reference.size()/width)+" versus ours "+std::to_string(trace.size()));
            for(size_t v=0;v<trace.size();++v) {
                mx::eval(trace[v]);double err=0,mag=0;const float* got=trace[v].data<float>();
                for(size_t i=0;i<width;++i) {double a=got[i]-reference[v*width+i];err+=a*a;mag+=double(reference[v*width+i])*reference[v*width+i];}
                std::printf("VISIT_INPUT %zu relative_rms %.8f\n",v,std::sqrt(err/(mag+1e-30)));
            }
            return 0;
        }
        if (bench) {
            std::vector<uint32_t> prompt;while(int(prompt.size())<bench) prompt.insert(prompt.end(),ids.begin(),ids.end());prompt.resize(bench);
            if(turns>0) {
                // A: N tokens of golden text after a user role token; B: A + a model message + a user message
                std::vector<uint32_t> A{200000u};
                while(int(A.size())<turns) A.insert(A.end(),ids.begin(),ids.end());
                A.resize(size_t(turns));
                std::vector<uint32_t> B(A);
                B.push_back(200001u);B.insert(B.end(),ids.begin(),ids.begin()+std::min<size_t>(200,ids.size()));
                B.push_back(200000u);B.insert(B.end(),ids.begin()+std::min<size_t>(300,ids.size()/2),ids.begin()+std::min<size_t>(420,ids.size()));
                auto prefill_chunks=[&](std::span<const uint32_t> prompt,size_t from) {
                    mx::array lg=mx::array(0.0f);
                    for(auto [p,end]:inkling2_prefill_chunks(prompt,from,4096)) lg=m.forward(prompt.subspan(p,end-p));
                    mx::eval(lg);return lg;
                };
                auto decode=[&](const mx::array& lg) {
                    std::vector<uint32_t> out{best(lg)};
                    while(out.size()<64) {auto l=m.forward(std::span<const uint32_t>(&out.back(),1));out.push_back(best(l));}
                    return out;
                };
                m.reset();auto t0=std::chrono::steady_clock::now();
                auto cold=prefill_chunks(B,0);const double cold_ms=seconds(t0)*1000;
                auto cold_tokens=decode(cold);
                m.reset();prefill_chunks(A,0);auto kept=m.keep();
                decode(m.forward(std::span<const uint32_t>(&A.back(),1)));   // the answer that turn generated (moves the state on)
                t0=std::chrono::steady_clock::now();m.restore(kept);
                auto warm=prefill_chunks(B,A.size());const double warm_ms=seconds(t0)*1000;
                auto warm_tokens=decode(warm);
                auto a=mx::contiguous(cold),b=mx::contiguous(warm);mx::eval(a,b);
                const bool same_logits=a.nbytes()==b.nbytes() && std::memcmp(a.data<void>(),b.data<void>(),a.nbytes())==0;
                const bool same=same_logits && cold_tokens==warm_tokens;
                std::printf("TURNS conversation %d follow-up %zu: cold prefill %.1f ms, resumed %.1f ms (%zu new tokens); first logits %s, 64 tokens %s: %s\n",
                            turns,B.size(),cold_ms,warm_ms,B.size()-A.size(),same_logits?"same":"DIFFER",cold_tokens==warm_tokens?"same":"DIFFER",same?"PASS":"FAIL");
                return same?0:1;
            }
            if(rowcheck>0) {
                auto run=[&](bool block,bool verify,std::vector<uint32_t>& tokens) {
                    m.reset();auto lg=prefill(m,prompt,chunk);uint32_t token=best(lg);
                    std::vector<std::vector<mx::array>> traces;
                    if(!block) {
                        tokens.clear();
                        for(int j=0;j<rowcheck;++j) {
                            tokens.push_back(token);std::vector<mx::array> t;
                            lg=m.forward(std::span<const uint32_t>(&token,1),false,&t);token=best(lg);
                            for(auto& a:t) {a=mx::contiguous(a);mx::eval(a);}
                            traces.push_back(std::move(t));
                        }
                    } else {
                        std::vector<mx::array> t;
                        lg=m.forward(std::span<const uint32_t>(tokens),true,&t,nullptr,verify);
                        for(auto& a:t) {a=mx::contiguous(a);mx::eval(a);}
                        traces.push_back(std::move(t));
                    }
                    return traces;
                };
                std::vector<uint32_t> tokens;
                const auto single=run(false,false,tokens);
                for(bool verify:{true,false}) {
                    const auto block=run(true,verify,tokens)[0];
                    if(block.size()!=single[0].size()) throw std::runtime_error("rowcheck trace length");
                    for(int j=0;j<rowcheck;++j) {
                        int first=-1;double worst=0;
                        for(size_t v=0;v<block.size();++v) {
                            auto row=mx::contiguous(mx::slice(block[v],{j,0},{j+1,block[v].shape(1)}));mx::eval(row);
                            const float* a=row.data<float>();const float* b=single[size_t(j)][v].data<float>();
                            const int D=row.shape(1);double err=0,mag=0;bool same=true;
                            for(int c=0;c<D;++c) {same=same && a[c]==b[c];err+=double(a[c]-b[c])*(a[c]-b[c]);mag+=double(b[c])*b[c];}
                            if(!same && first<0) first=int(v);
                            worst=std::max(worst,std::sqrt(err/(mag+1e-30)));
                        }
                        std::printf("ROWCHECK %s row %d first_diff_visit %d of %zu max_rel %.3g\n",verify?"verify":"chunk",j,first,block.size(),worst);
                    }
                    std::fflush(stdout);
                }
                return 0;
            }
            if(blocks) {
                m.warm_up();   // as the server does at load: no first-use compiles in the timed rows
                auto lg=prefill(m,prompt,chunk);uint32_t token=best(lg);
                for(int i=0;i<4;++i) {lg=m.forward(std::span<const uint32_t>(&token,1));token=best(lg);}
                double one=0;
                for(int n:{1,2,3,4,6,8}) {
                    std::vector<uint32_t> block(size_t(n),token);double ms=0;const int reps=8;
                    for(int r=0;r<reps;++r) {const auto t0=std::chrono::steady_clock::now();lg=m.forward(std::span<const uint32_t>(block),false);mx::eval(lg);ms+=seconds(t0)*1000;}
                    ms/=reps;if(n==1) one=ms;
                    std::printf("BLOCK tokens %d forward_ms %.3f vs_one %.2f\n",n,ms,ms/one);std::fflush(stdout);
                }
                for(int n:{1,8}) {  // where an n-token forward's time goes (synchronised per phase: instrumented, not throughput)
                    std::vector<uint32_t> block(size_t(n),token);std::array<double,5> ph{};const int reps=4;
                    for(int r=0;r<reps;++r) {lg=m.forward(std::span<const uint32_t>(block),false,nullptr,&ph);mx::eval(lg);}
                    std::printf("BLOCK_PHASES tokens %d input %.3f attention_conv %.3f mlp_input %.3f experts_conv %.3f residual_state %.3f\n",
                                n,ph[0]/reps,ph[1]/reps,ph[2]/reps,ph[3]/reps,ph[4]/reps);std::fflush(stdout);
                }
                return 0;
            }
            if(profile) {
                auto lg=prefill(m,prompt,chunk);uint32_t token=best(lg);
                for(int i=0;i<8;++i) {lg=m.forward(std::span<const uint32_t>(&token,1));token=best(lg);}
                std::array<double,5> phases{};auto start=std::chrono::steady_clock::now();
                for(int i=0;i<8;++i) {lg=m.forward(std::span<const uint32_t>(&token,1),false,nullptr,&phases);token=best(lg);}
                std::printf("PROFILE synchronized wall_ms/token %.3f input %.3f attention_conv %.3f mlp_input %.3f experts_conv %.3f residual_state %.3f (instrumented, not throughput)\n",seconds(start)*1000/8,phases[0]/8,phases[1]/8,phases[2]/8,phases[3]/8,phases[4]/8);
                return 0;
            }
            for (int run=0;run<runs;++run) {
                m.reset();const auto start=std::chrono::steady_clock::now();auto lg=prefill(m,prompt,chunk);mx::eval(lg);const auto td=std::chrono::steady_clock::now();  // prefill done
                uint64_t hash=1469598103934665603ull;int steps=0,accepted=0;
                const auto tokens=spec>=0?spec_tokens(m,best(lg),decode,spec,steps,accepted):plain_tokens(m,best(lg),decode);
                for(auto tok:tokens) hash=(hash^tok)*1099511628211ull;
                if(spec>=0) std::printf("SPEC steps %d accepted %d tokens/step %.3f\n",steps,accepted,double(decode)/steps);
                std::printf("BENCH run %d context %d chunk %d prefill %.3f tok/s decode %.3f tok/s tokens %d hash %016llx\n",run,bench,chunk,
                    bench/std::chrono::duration<double>(td-start).count(),decode/seconds(td),decode,(unsigned long long)hash);std::fflush(stdout);
                std::printf("MEMORY active_bytes %zu peak_bytes %zu cache_bytes %zu\n",mx::get_active_memory(),mx::get_peak_memory(),mx::get_cache_memory());
            }
            return 0;
        }
        if (spec>=0 && !greedy && !flips) {
            // Spec gate: per golden prompt, `decode` tokens plain and speculative from the same prefill. The verify
            // path computes blocks (matmul) where plain decode computes one row (matvec), so a near-tie can flip:
            // report the agreed prefix against plain and against the reference greedy tokens.
            std::ifstream f(dir+"/greedy.txt");if(!f) throw std::runtime_error("greedy inputs missing");
            std::string line;int prompts=0,identical=0,total_steps=0,total_tokens=0;
            double spec_seconds=0,plain_seconds=0;
            while(std::getline(f,line)) {
                std::istringstream in(line);std::string name;int n=0,k=0;in>>name>>n;
                std::vector<uint32_t> prompt(size_t(std::max(n,0)));for(auto& v:prompt) in>>v;
                in>>k;std::vector<uint32_t> expected(size_t(std::max(k,0)));for(auto& v:expected) in>>v;
                if(n<=0 || k<=0 || !in) throw std::runtime_error("invalid greedy row");
                m.reset();auto first=best(prefill(m,prompt,chunk));auto t0=std::chrono::steady_clock::now();
                auto plain=plain_tokens(m,first,decode);plain_seconds+=seconds(t0);
                m.reset();first=best(prefill(m,prompt,chunk));int steps=0,accepted=0;t0=std::chrono::steady_clock::now();
                auto fast=spec_tokens(m,first,decode,spec,steps,accepted);spec_seconds+=seconds(t0);
                auto prefix=[&](const std::vector<uint32_t>& a,const std::vector<uint32_t>& b,size_t skip) {
                    size_t i=0;while(i<a.size() && i+skip<b.size() && a[i]==b[i+skip]) ++i;return int(i);
                };
                const int with_plain=prefix(fast,plain,0);
                // expected[0] is the prefill's own token (first); the decoded tokens follow it
                const int plain_ref=expected[0]==first?prefix(plain,expected,1):-1,spec_ref=expected[0]==first?prefix(fast,expected,1):-1;
                identical+=with_plain==decode;++prompts;total_steps+=steps;total_tokens+=decode;
                std::printf("SPEC %s spec==plain prefix %d/%d plain==ref %d spec==ref %d steps %d accepted %d tokens/step %.3f\n",
                            name.c_str(),with_plain,decode,plain_ref,spec_ref,steps,accepted,double(decode)/steps);std::fflush(stdout);
            }
            std::printf("SPEC_ALL identical %d/%d tokens/step %.3f decode plain %.2f tok/s spec %.2f tok/s\n",identical,prompts,
                        double(total_tokens)/total_steps,total_tokens/plain_seconds,total_tokens/spec_seconds);
            // the release gate: greedy speculative output is the plain greedy output on every prompt
            std::printf("SPEC_GATE %s\n",identical==prompts?"PASS":"FAIL");
            return identical==prompts?0:1;
        }
        if (greedy || flips) {
            std::ifstream f(dir+"/greedy.txt");if(!f) throw std::runtime_error("greedy inputs missing");
            std::string line;int passed=0,count=0,total_flips=0,total_positions=0;
            while(std::getline(f,line)) {
                std::istringstream in(line);std::string name;int n=0,k=0;in>>name>>n;
                if(n<=0) throw std::runtime_error("invalid greedy prompt");std::vector<uint32_t> prompt(n);for(auto& v:prompt) in>>v;
                in>>k;if(k<=0) throw std::runtime_error("invalid greedy reference");std::vector<uint32_t> expected(k);for(auto& v:expected) in>>v;
                if(!in) throw std::runtime_error("truncated greedy row");m.reset();auto lg=prefill(m,prompt,chunk);
                bool equal=true;int nflip=0;
                for(int t=0;t<std::min(64,k);++t) {
                    auto got=best(lg);++total_positions;
                    if(got!=expected[t]) {
                        equal=false;++nflip;
                        const float* data=lg.data<float>();
                        std::printf("%s %s mismatch %d ours %u ref %u native_margin %.6f\n",flips?"FLIPS":"GREEDY",name.c_str(),t,got,expected[t],data[got]-data[expected[t]]);
                        if(!flips) break;
                    }
                    uint32_t token=flips?expected[t]:got;
                    lg=m.forward(std::span<const uint32_t>(&token,1));
                }
                total_flips+=nflip;
                passed+=equal;++count;std::printf("GREEDY %s %s\n",name.c_str(),equal?"PASS":"FAIL");std::fflush(stdout);
            }
            if(flips) {
                // The pinned HF F32-vs-BF16 floor is 1/320; the lead's floor+3 limit is four.
                const bool pass=count==5 && total_positions==320 && total_flips<=4;
                std::printf("FLIPS total %d/%d limit 4 %s\n",total_flips,total_positions,pass?"PASS":"FAIL");
                return pass?0:1;
            }
            std::printf("GREEDY total %d/%d\n",passed,count);return count==5 && passed>=4?0:1;
        }
        auto top=load<uint32_t>(dir+"/tf_top_ids.i32");auto lp=load<float>(dir+"/tf_top_lp.f32");
        if(ids.empty() || top.size()!=lp.size() || top.size()%ids.size()) throw std::runtime_error("teacher reference shape");
        const size_t K=top.size()/ids.size();double kl=0;size_t agree=0,count=0;
        for(size_t p=0;p<ids.size();p+=size_t(chunk)) {
            const auto n=std::min(size_t(chunk),ids.size()-p);auto lg=m.forward(std::span<const uint32_t>(ids).subspan(p,n),true);mx::eval(lg);
            const float* data=lg.data<float>();
            for(size_t t=0;t<n;++t) {
                const auto* x=data+t*m.vocabulary();float maximum=*std::max_element(x,x+m.vocabulary());double z=0;
                for(int v=0;v<m.vocabulary();++v) z+=std::exp(double(x[v])-maximum);
                const double logz=maximum+std::log(z);size_t off=(p+t)*K;uint32_t am=uint32_t(std::max_element(x,x+m.vocabulary())-x);agree+=am==top[off];
                for(size_t j=0;j<K;++j) {if(top[off+j]>=uint32_t(m.vocabulary())) throw std::runtime_error("reference token outside vocabulary");kl+=std::exp(lp[off+j])*(lp[off+j]-(double(x[top[off+j]])-logz));}
                ++count;
            }
            std::printf("TF positions %zu/%zu top-K KL contribution %.7f top1 %.4f%%\n",count,ids.size(),kl/count,100.*agree/count);std::fflush(stdout);
        }
        std::puts("Top-K contribution is a diagnostic; full-vocabulary KLD gate remains separate.");
        return count>=2000 && kl/count<=.01 && double(agree)/count>=.98?0:1;
    } catch(const std::exception& e) {std::fprintf(stderr,"inkling2 MLX error: %s\n",e.what());return 2;}
}

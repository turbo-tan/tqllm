#include <mlx/mlx.h>
#include "gguf.hpp"
#include "gdn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <bit>

namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;
static void check(const mx::array& input,const std::vector<double>& expected,const char* name) {
    auto got=mx::contiguous(input);
    mx::eval(got);
    if(got.size()!=expected.size())throw std::runtime_error("GDN output shape mismatch");
    double diff=0,norm=0,maxerr=0;
    for(size_t i=0;i<expected.size();++i) {
        const double delta=got.data<float>()[i]-expected[i];
        diff+=delta*delta;norm+=expected[i]*expected[i];maxerr=std::max(maxerr,std::abs(delta));
    }
    const double rel=std::sqrt(diff/std::max(norm,1e-20));
    std::cout<<"GDN_CPU_CHECK buffer="<<name<<" rel_l2="<<rel<<" max_abs="<<maxerr<<'\n';
    if(rel>2e-5 && maxerr>1e-5) {
        for(size_t i=0;i<std::min(expected.size(),size_t(12));++i)
            std::cerr<<"GDN_SAMPLE buffer="<<name<<" at="<<i<<" got="<<got.data<float>()[i]<<" cpu="<<expected[i]<<'\n';
        throw std::runtime_error("GDN CPU recurrence mismatch");
    }
}
int main(int argc,char** argv) {
    try {
        int heads=48,key_heads=16,dk=128,dv=128;
        if(argc==2) {
            const auto file=tqllm::gguf_open(argv[1]);
            heads=int(file.num("qwen35.ssm.time_step_rank"));
            key_heads=int(file.num("qwen35.ssm.group_count"));
            dk=int(file.num("qwen35.ssm.state_size"));
            dv=int(file.num("qwen35.ssm.inner_size"))/heads;
        } else if(argc!=1)throw std::invalid_argument("usage: test-mlx-gdn [model.gguf]");
        if(heads<1 || key_heads<1 || heads%key_heads || dk<32 || dk%32 || dv<1)throw std::invalid_argument("invalid GDN dimensions");
        const bool scan=true;
        for(int invalid=0;invalid<5;++invalid) {
            bool rejected=false;
            try {
                if(invalid==0)tqllm::mlx_backend::gdn_convolve_native(mx::zeros({3,32}),mx::zeros({32,4}));
                if(invalid==1)tqllm::mlx_backend::gdn_convolve_native(mx::zeros({4,32}),mx::zeros({4,32}));
                if(invalid==2)tqllm::mlx_backend::gdn_convolve_native(mx::zeros({4,32},mx::bfloat16),mx::zeros({32,4}));
                if(invalid==3)tqllm::mlx_backend::gdn_gates_native(mx::zeros({8,48}),mx::zeros({8,47}),mx::zeros({48}),mx::zeros({48}));
                if(invalid==4)tqllm::mlx_backend::gdn_gates_native(mx::zeros({0,48}),mx::zeros({0,48}),mx::zeros({48}),mx::zeros({48}));
            } catch(const std::invalid_argument&) {rejected=true;}
            if(!rejected)throw std::runtime_error("invalid native GDN preparation accepted");
        }
        for(int count:{1,3,5,8}) {
            const int width=2*key_heads*dk+heads*dv;
            std::vector<float> joined(size_t(count+3)*width),weights(size_t(width)*4);
            std::vector<float> alpha(size_t(count)*heads),bias(heads),rate(heads);
            for(size_t i=0;i<joined.size();++i)joined[i]=std::sin(float(i)*.031f)*2.0f;
            for(size_t i=0;i<weights.size();++i)weights[i]=std::cos(float(i)*.023f)*.3f;
            for(size_t i=0;i<alpha.size();++i)alpha[i]=std::sin(float(i)*.071f)*100.0f;
            for(int i=0;i<heads;++i) {bias[i]=std::cos(float(i)*.11f);rate[i]=-.01f*(i+1);}
            mx::array aj(joined.data(),{count+3,width}),aw(weights.data(),{width,4});
            mx::array aa(alpha.data(),{count,heads}),abias(bias.data(),{heads}),ar(rate.data(),{heads});
            auto reference=tqllm::mlx_backend::gdn_convolve(aj,aw);
            mx::eval(reference);
            std::vector<double> expected(reference.data<float>(),reference.data<float>()+reference.size());
            auto aq=mx::reshape(mx::slice(reference,{0,0},{count,key_heads*dk}),{1,count,key_heads,dk});
            auto normalized=tqllm::mlx_backend::gdn_normalize(aq,1.0f/std::sqrt(float(dk)));
            auto decay=tqllm::mlx_backend::gdn_decay(aa,abias,ar);
            mx::eval(normalized,decay);
            std::vector<double> nq(normalized.data<float>(),normalized.data<float>()+normalized.size());
            std::vector<double> dg(decay.data<float>(),decay.data<float>()+decay.size());
            auto fused=tqllm::mlx_backend::gdn_convolve(aj,aw,true);
            check(fused,expected,"compiled_convolution");
            auto fq=mx::reshape(mx::slice(fused,{0,0},{count,key_heads*dk}),{1,count,key_heads,dk});
            check(tqllm::mlx_backend::gdn_normalize(fq,1.0f/std::sqrt(float(dk)),true),nq,"compiled_normalization");
            check(tqllm::mlx_backend::gdn_decay(aa,abias,ar,true),dg,"compiled_decay_extremes");
            auto native_cv=tqllm::mlx_backend::gdn_convolve_native(aj,aw);
            check(native_cv,expected,"native_convolution_vs_original");
            auto native_gates=tqllm::mlx_backend::gdn_gates_native(aa,aa,abias,ar);
            check(native_gates.first,dg,"native_decay_vs_original");
            std::vector<double> cpu_decay(alpha.size()),cpu_beta(alpha.size()),cpu_cv(size_t(count)*width);
            auto round_bf=[](float x) {
                auto bits=std::bit_cast<uint32_t>(x);
                bits+=0x7fffu+((bits>>16)&1u);
                return std::bit_cast<float>(bits&0xffff0000u);
            };
            for(int token=0;token<count;++token)for(int col=0;col<width;++col) {
                float sum=0;
                for(int tap=0;tap<4;++tap)
                    sum=sum+joined[size_t(token+tap)*width+col]*weights[size_t(col)*4+tap];
                float rounded=round_bf(sum);
                cpu_cv[size_t(token)*width+col]=round_bf(rounded*(1.0f/(1.0f+std::exp(-rounded))));
            }
            for(size_t i=0;i<alpha.size();++i) {
                const double dt=double(alpha[i])+bias[i%heads];
                cpu_decay[i]=std::exp((std::max(dt,0.0)+std::log1p(std::exp(-std::abs(dt))))*rate[i%heads]);
                cpu_beta[i]=1.0/(1.0+std::exp(-double(alpha[i])));
            }
            check(native_cv,cpu_cv,"native_convolution_cpu");
            check(native_gates.first,cpu_decay,"native_decay_cpu");
            check(native_gates.second,cpu_beta,"native_beta_cpu");
            for(int mode:{0,1,2}) {
                std::vector<double> times;
                for(int run=-1;run<5;++run) {
                    auto start=Clock::now();
                    auto cv=mode==2?tqllm::mlx_backend::gdn_convolve_native(aj,aw):
                        tqllm::mlx_backend::gdn_convolve(aj,aw,mode==1);
                    auto qv=mx::reshape(mx::slice(cv,{0,0},{count,key_heads*dk}),{1,count,key_heads,dk});
                    auto nv=tqllm::mlx_backend::gdn_normalize(qv,1.0f/std::sqrt(float(dk)),mode==1);
                    auto gates=mode==2?tqllm::mlx_backend::gdn_gates_native(aa,aa,abias,ar):
                        std::pair{tqllm::mlx_backend::gdn_decay(aa,abias,ar,mode==1),mx::sigmoid(aa)};
                    mx::eval(cv,nv,gates.first,gates.second);
                    if(run>=0)times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
                }
                std::sort(times.begin(),times.end());
                std::cout<<"GDN_PREP_BENCH count="<<count<<" mode="<<mode<<" median_ms="<<times[2]<<std::endl;
            }
        }
        for(int count:{1,3,5,8,16,64}) {
            std::vector<float> q(size_t(count)*key_heads*dk),k(q.size()),v(size_t(count)*heads*dv);
            std::vector<float> g(size_t(count)*heads),beta(g.size()),initial(size_t(heads)*dv*dk);
            for(size_t i=0;i<q.size();++i) {
                q[i]=std::sin(float(i)*.031f)/float(dk);
                k[i]=std::cos(float(i)*.029f)/std::sqrt(float(dk));
            }
            for(size_t i=0;i<v.size();++i)v[i]=std::sin(float(i)*.037f)*.1f;
            for(size_t i=0;i<g.size();++i) {g[i]=.8f+.19f*std::abs(std::cos(float(i)*.13f));beta[i]=.3f+.2f*std::sin(float(i)*.17f);}
            for(size_t i=0;i<initial.size();++i)initial[i]=.01f*std::sin(float(i)*.019f);
            std::vector<double> state(initial.begin(),initial.end()),output(v.size()),prefix_states;
            for(int t=0;t<count;++t)for(int h=0;h<heads;++h)for(int row=0;row<dv;++row) {
                const size_t at=(size_t(h)*dv+row)*dk,kh=(size_t(t)*key_heads+h%key_heads)*dk;
                double memory=0;
                for(int j=0;j<dk;++j) {state[at+j]*=g[t*heads+h];memory+=state[at+j]*k[kh+j];}
                const double delta=(v[(size_t(t)*heads+h)*dv+row]-memory)*beta[t*heads+h];
                double y=0;
                for(int j=0;j<dk;++j) {state[at+j]+=k[kh+j]*delta;y+=state[at+j]*q[kh+j];}
                output[(size_t(t)*heads+h)*dv+row]=y;
                if(scan && count<=8 && h==heads-1 && row==dv-1)
                    prefix_states.insert(prefix_states.end(),state.begin(),state.end());
            }
            mx::array aq(q.data(),{1,count,key_heads,dk}),ak(k.data(),{1,count,key_heads,dk});
            mx::array av(v.data(),{1,count,heads,dv}),ag(g.data(),{1,count,heads}),ab(beta.data(),{1,count,heads});
            mx::array as(initial.data(),{1,heads,dv,dk});
            mx::eval(aq,ak,av,ag,ab,as);
            std::vector<double> times;
            for(int run=-1;run<5;++run) {
                const auto start=Clock::now();
                auto scanned=scan?std::optional<tqllm::mlx_backend::GdnScan>(
                    tqllm::mlx_backend::gdn_scan(aq,ak,av,ag,ab,as,count<=8)):std::nullopt;
                auto update=std::vector<mx::array>{scanned->output,scanned->state};mx::eval(update);
                const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if(run==0) {
                    check(update[0],output,"output");check(update[1],state,"state");
                    if(scanned && scanned->snapshots)check(*scanned->snapshots,prefix_states,"prefix_states");
                    if(scanned && count<=8) {
                        auto compact=tqllm::mlx_backend::gdn_scan(aq,ak,av,ag,ab,as,true,true);
                        mx::eval(compact.output,compact.state,*compact.deltas);
                        check(compact.output,output,"compact_output_cpu");
                        check(compact.state,state,"compact_final_cpu");
                        if(compact.snapshots || !mx::all(mx::equal(compact.output,scanned->output)).item<bool>() ||
                           !mx::all(mx::equal(compact.state,scanned->state)).item<bool>())
                            throw std::runtime_error("compact tape changed original scan output/final state");
                        for(int prefix=1;prefix<=count;++prefix) {
                            auto restored=tqllm::mlx_backend::gdn_replay_prefix(ak,ag,*compact.deltas,as,prefix);
                            auto expected=mx::slice(*scanned->snapshots,{prefix-1,0,0,0},{prefix,heads,dv,dk});
                            if(!mx::all(mx::equal(restored,expected)).item<bool>())
                                throw std::runtime_error("compact replay differs from independently saved prefix state");
                            const size_t n=initial.size();
                            check(restored,std::vector<double>(prefix_states.begin()+size_t(prefix-1)*n,
                                prefix_states.begin()+size_t(prefix)*n),"compact_prefix_cpu");
                        }
                        auto empty=tqllm::mlx_backend::gdn_scan(aq,ak,av,ag,ab,std::nullopt,true);
                        auto compact_empty=tqllm::mlx_backend::gdn_scan(aq,ak,av,ag,ab,std::nullopt,true,true);
                        for(int prefix=1;prefix<=count;++prefix) {
                            auto restored=tqllm::mlx_backend::gdn_replay_prefix(ak,ag,*compact_empty.deltas,std::nullopt,prefix);
                            if(!mx::all(mx::equal(restored,mx::slice(*empty.snapshots,{prefix-1,0,0,0},{prefix,heads,dv,dk}))).item<bool>())
                                throw std::runtime_error("compact empty-initial replay differs from original");
                        }
                        for(int bad:{0,count+1}) {
                            bool rejected=false;
                            try {tqllm::mlx_backend::gdn_replay_prefix(ak,ag,*compact.deltas,as,bad);}
                            catch(const std::invalid_argument&) {rejected=true;}
                            if(!rejected)throw std::runtime_error("compact replay accepted invalid prefix count");
                        }
                        std::vector<double> compact_times,replay_times;
                        for(int run=-1;run<5;++run) {
                            auto begin=Clock::now();auto candidate=tqllm::mlx_backend::gdn_scan(aq,ak,av,ag,ab,as,true,true);
                            mx::eval(candidate.output,candidate.state,*candidate.deltas);
                            const double scan_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
                            begin=Clock::now();mx::eval(tqllm::mlx_backend::gdn_replay_prefix(ak,ag,*candidate.deltas,as,std::max(1,count-1)));
                            if(run>=0) {compact_times.push_back(scan_ms);replay_times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());}
                        }
                        std::sort(compact_times.begin(),compact_times.end());std::sort(replay_times.begin(),replay_times.end());
                        std::cout<<"GDN_COMPACT_PASS count="<<count<<" compact_scan_ms="<<compact_times[2]
                            <<" partial_replay_ms="<<replay_times[2]<<" full_tape_bytes="<<scanned->snapshots->nbytes()
                            <<" compact_delta_bytes="<<compact.deltas->nbytes()<<" final_state_bytes="<<compact.state.nbytes()<<std::endl;
                    }
                }
                if(run>=0)times.push_back(ms);
            }
            std::sort(times.begin(),times.end());
            std::cout<<"GDN_BENCH count="<<count<<" heads="<<heads<<" dk="<<dk<<" dv="<<dv<<" median_ms="<<times[2]<<std::endl;
        }
        std::cout<<"GDN_CPU_PASS\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

#include <mlx/mlx.h>
#include "../src/gguf.hpp"
#include "gdn.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdlib>

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
        const bool native=std::getenv("TQLLM_MLX_NATIVE_GDN")!=nullptr;
        const bool scan=std::getenv("TQLLM_MLX_GDN_SCAN")!=nullptr;
        auto order=tqllm::mlx_backend::gdn_head_order(key_heads,heads);
        for(const auto& [kh,vh]:std::vector<std::pair<int,int>>{{16,16},{16,32},{16,48},{16,64},{24,24},{32,32}}) {
            auto layout=tqllm::mlx_backend::gdn_head_order(kh,vh);
            mx::eval(layout.to_native,layout.from_native);
            for(int i=0;i<vh;++i)
                if(layout.from_native.data<uint32_t>()[layout.to_native.data<uint32_t>()[i]]!=uint32_t(i) ||
                   layout.to_native.data<uint32_t>()[i]%uint32_t(kh)!=uint32_t(i/(vh/kh)))
                    throw std::runtime_error("GDN head order is not an inverse permutation");
        }
        std::vector<uint32_t> indices(heads);
        for(int h=0;h<heads;++h)indices[h]=uint32_t(h%key_heads);
        mx::array key_indices(indices.data(),{heads},mx::uint32);
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
            for(bool compiled:{false,true}) {
                std::vector<double> times;
                for(int run=-1;run<5;++run) {
                    auto start=Clock::now();
                    auto cv=tqllm::mlx_backend::gdn_convolve(aj,aw,compiled);
                    auto qv=mx::reshape(mx::slice(cv,{0,0},{count,key_heads*dk}),{1,count,key_heads,dk});
                    auto nv=tqllm::mlx_backend::gdn_normalize(qv,1.0f/std::sqrt(float(dk)),compiled);
                    auto dv=tqllm::mlx_backend::gdn_decay(aa,abias,ar,compiled);
                    mx::eval(cv,nv,dv);
                    if(run>=0)times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
                }
                std::sort(times.begin(),times.end());
                std::cout<<"GDN_PREP_BENCH count="<<count<<" compiled="<<compiled<<" median_ms="<<times[2]<<std::endl;
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
            if(native && !scan)as=mx::take(as,order.to_native,1);
            else if(!scan) {aq=mx::take(aq,key_indices,2);ak=mx::take(ak,key_indices,2);}
            mx::eval(aq,ak,av,ag,ab,as);
            std::vector<double> times;
            for(int run=-1;run<5;++run) {
                const auto start=Clock::now();
                auto scanned=scan?std::optional<tqllm::mlx_backend::GdnScan>(
                    tqllm::mlx_backend::gdn_scan(aq,ak,av,ag,ab,as,count<=8)):std::nullopt;
                auto update=scanned?std::vector<mx::array>{scanned->output,scanned->state}:native?tqllm::mlx_backend::gdn_native(aq,ak,av,ag,ab,as,order):
                    mx::fast::gated_delta_update(aq,ak,av,ag,ab,as);mx::eval(update);
                const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if(run==0) {
                    check(update[0],output,"output");check(native && !scan?mx::take(update[1],order.from_native,1):update[1],state,"state");
                    if(scanned && scanned->snapshots)check(*scanned->snapshots,prefix_states,"prefix_states");
                }
                if(run>=0)times.push_back(ms);
            }
            std::sort(times.begin(),times.end());
            std::cout<<"GDN_BENCH count="<<count<<" heads="<<heads<<" dk="<<dk<<" dv="<<dv<<" median_ms="<<times[2]<<std::endl;
        }
        std::cout<<"GDN_CPU_PASS\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

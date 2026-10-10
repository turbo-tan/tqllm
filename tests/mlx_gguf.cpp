#include "weights.hpp"
#include "prefill.hpp"
#include "../src/dequant.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

namespace mx = mlx::core;
using namespace tqllm;
using namespace tqllm::mlx_backend;
static uint32_t word(const uint8_t* p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
static void dequant(uint32_t type, const std::vector<uint8_t>& bytes, std::vector<float>& w) {
    if(type!=GGML_TQ3 && type!=GGML_TQ4 && type!=GGML_Q8_0) {
        dequantize_host(type, bytes.data(),w.data(),int64_t(w.size()));return;
    }
    if(type==GGML_Q8_0) {
        for(size_t b=0;b<w.size()/32;++b) {
            const auto* p=bytes.data()+b*34;
            const float d=half_to_float(uint16_t(p[0])|(uint16_t(p[1])<<8));
            for(int j=0;j<32;++j) w[b*32+j]=float(std::bit_cast<int8_t>(p[2+j]))*d;
        }
        return;
    }
    constexpr int lv4[]={6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};
    constexpr int lv3[]={14,45,79,127,-14,-45,-79,-127};
    for(size_t b=0;b<w.size()/128;++b) {
        const auto* p=bytes.data()+b*(type==GGML_TQ3?50:66);
        const float d=half_to_float(uint16_t(p[0])|(uint16_t(p[1])<<8));
        for(int j=0;j<128;++j) {
            if(type==GGML_TQ4) w[b*128+j]=d*lv4[(p[2+j/2]>>(4*(j%2)))&15];
            else {
                const auto* group=p+2+(j/32)*12; const int t=j%32;
                const uint32_t ranks=word(group+(t/16)*4),signs=word(group+8);
                const int rank=(ranks>>(4*(t%8)+2*((t%16)/8)))&3;
                const int sign=(signs>>(4*(t%8)+t/8))&1;
                w[b*128+j]=d*lv3[rank+4*sign];
            }
        }
    }
}
static std::vector<float> rotate(std::vector<float> x) {
    constexpr uint32_t signs[]={0xE46A0359,0x42F85949,0xA4C63BBB,0x49F250A9};
    for(size_t b=0;b<x.size();b+=128) {
        for(int j=0;j<128;++j) x[b+j]*=(signs[j/32]>>(j%32))&1?-1.0f:1.0f;
        for(int width=1;width<128;width*=2)
            for(int j=0;j<128;j+=2*width) for(int a=0;a<width;++a) {
                const float lo=x[b+j+a],hi=x[b+j+a+width];
                x[b+j+a]=lo+hi;x[b+j+a+width]=lo-hi;
            }
        for(int j=0;j<128;++j) x[b+j]/=std::sqrt(128.0f);
    }
    return x;
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) {std::cerr<<"usage: test-mlx-gguf model.gguf [--prefill]\n";return 2;}
    const bool prefill=argc==3 && std::string(argv[2])=="--prefill";
    if(argc==3 && !prefill) return 2;
    auto file=gguf_open(argv[1]); std::set<uint32_t> covered;
    for(const auto& [name,original]:file.tensors) {
        if(original.ne.size()!=2 || covered.count(original.type)) continue;
        if(original.type!=GGML_Q4_K && original.type!=GGML_F32 && original.type!=GGML_BF16 &&
           original.type!=GGML_TQ3 && original.type!=GGML_TQ4 && original.type!=GGML_Q6_K && original.type!=GGML_Q8_0)
            throw std::runtime_error("unsupported model tensor type");
        auto t=original;
        const int k=int(t.ne[0]),r=int(std::min<uint64_t>(17,t.ne[1]));
        t.bytes=t.bytes/t.ne[1]*uint64_t(r);t.ne[1]=uint64_t(r);
        auto data=file.read(t);
        std::vector<uint8_t> payload(data.begin(),data.end());
        std::vector<float> w(size_t(r)*k);dequant(t.type,payload,w);
        if(t.type==GGML_Q4_K) {
            auto embedding=load_q4_embedding(payload,r,k);
            std::vector<uint32_t> ids={0,uint32_t(r-1),uint32_t(r/2),0};
            auto y=q4_embedding(embedding,ids);mx::eval(y);
            float error=0;
            for(size_t a=0;a<ids.size();++a) for(int j=0;j<k;++j)
                error=std::max(error,std::abs(y.data<float>()[a*k+j]-w[ids[a]*k+j]));
            if(error>1e-5) throw std::runtime_error("real GGUF embedding mismatch");
            std::cout<<"GGUF_EMBED_PASS type="<<t.type<<" max_abs="<<error<<" tensor="<<name<<'\n';
        } else {
            auto matrix=load_matrix(t,payload);
            if(prefill) {
                auto staged=stage_bf16(matrix);
                if(staged.dtype()!=mx::bfloat16)throw std::runtime_error("prefill staging dtype mismatch");
                auto staged_fp32=mx::astype(staged,mx::float32);mx::eval(staged_fp32);
                for(size_t i=0;i<w.size();++i) {
                    const uint32_t bits=std::bit_cast<uint32_t>(w[i]);
                    const float rounded=std::bit_cast<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
                    if(staged_fp32.data<float>()[i]!=rounded)
                        throw std::runtime_error("prefill BF16 weights differ from CPU rounding: "+name);
                }
                std::cout<<"GGUF_STAGE_PASS type="<<t.type<<" tensor="<<name<<'\n';
            }
            for(int m:{1,8,64}) {
                std::vector<float> input(size_t(m)*k);
                for(size_t i=0;i<input.size();++i) input[i]=std::sin(float(i)*0.037f)*0.1f;
                auto refinput=(t.type==GGML_TQ3||t.type==GGML_TQ4)?rotate(input):input;
                auto y=linear(matrix,mx::array(input.data(),{m,k}));mx::eval(y);
                double diff=0,total=0;float maxerr=0;
                for(int a=0;a<m;++a) for(int row=0;row<r;++row) {
                    double expected=0;
                    for(int j=0;j<k;++j) expected+=double(w[row*k+j])*refinput[a*k+j];
                    const double delta=y.data<float>()[a*r+row]-expected;
                    diff+=delta*delta;total+=expected*expected;maxerr=std::max(maxerr,float(std::abs(delta)));
                }
                const double rel=std::sqrt(diff/std::max(total,1e-20));
                if(rel>2e-5 && maxerr>1e-5) {
                    std::cerr<<"GGUF_LINEAR_FAIL type="<<t.type<<" m="<<m<<" rel_l2="<<rel<<" max_abs="<<maxerr<<'\n';
                    throw std::runtime_error("real GGUF linear mismatch: "+name);
                }
                std::cout<<"GGUF_LINEAR_PASS type="<<t.type<<" m="<<m<<" rel_l2="<<rel<<" max_abs="<<maxerr<<" tensor="<<name<<'\n';
            }
        }
        covered.insert(t.type);
    }
    std::set<uint32_t> required;
    for(const auto& [name,t]:file.tensors) if(t.ne.size()==2) required.insert(t.type);
    if(covered!=required) throw std::runtime_error("incomplete GGUF matrix type coverage");
    std::cout<<"GGUF_TYPES_PASS "<<covered.size()<<'\n';
}

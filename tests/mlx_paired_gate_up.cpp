#include "tq3_tiled.hpp"
#include "weights.hpp"
#include "dequant.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace mx = mlx::core;
using namespace tqllm;
using namespace tqllm::mlx_backend;
#include "mlx_tiled_fixture.hpp"
static TqTiledWeights tile(const std::vector<uint8_t>& raw,int r,int k,int bits) {
    return bits==3?tq3_tile(load_tq3_blocks(raw,r,k)):tq4_tile(load_tq4_blocks(raw,r,k));
}
static std::vector<float> oracle(const std::vector<uint8_t>& raw,int r,int k,int bits,
        const std::vector<float>& x,int m,bool fp32) {
    constexpr int l3[]={14,45,79,127,-14,-45,-79,-127};
    constexpr int l4[]={6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};
    std::vector<float> y(size_t(m)*r);
    for(int col=0;col<m;++col)for(int row=0;row<r;++row) {
        double sum=0;
        for(int j=0;j<k;++j) {
            auto p=raw.data()+(size_t(row)*(k/128)+j/128)*(bits==3?50:66);
            float scale=half_to_float(uint16_t(p[0])|(uint16_t(p[1])<<8));int level;
            if(bits==3) {
                int t=j%32;auto b=p+2+((j%128)/32)*12;
                uint32_t code=((word(b+(t/16)*4)>>(4*(t%8)+2*((t%16)/8)))&3u)+4*((word(b+8)>>(4*(t%8)+t/8))&1u);
                level=l3[code];
            } else level=l4[(p[2+(j%128)/2]>>(4*(j%2)))&15];
            float w=scale*float(level),v=x[col*k+j];
            if(!fp32){w=float(_Float16(w));v=float(_Float16(v));}
            sum+=double(w)*v;
        }
        y[col*r+row]=float(sum);
    }
    return y;
}
static void exact(const mx::array& a,const mx::array& b) {
    mx::eval(a,b);
    if(a.shape()!=b.shape() || std::memcmp(a.data<float>(),b.data<float>(),a.size()*sizeof(float)))
        throw std::runtime_error("paired projection differs from separate raw FP32 output");
}
template<class F> static double timing(F f) {
    auto warm=f();mx::eval(warm.first,warm.second);std::vector<double> ms;
    for(int i=0;i<7;++i) {
        auto start=std::chrono::steady_clock::now();auto y=f();mx::eval(y.first,y.second);
        ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(ms.begin(),ms.end());return ms[3];
}
int main(int argc,char** argv) {
    try {
        for(int bits:{3,4})for(int r:{17,33}) {
            const int k=512;auto graw=fixture(r,k,bits),uraw=graw;
            for(size_t b=0;b<uraw.size()/size_t(bits==3?50:66);++b)uraw[b*(bits==3?50:66)+1]^=0x80;
            auto g=tile(graw,r,k,bits),u=tile(uraw,r,k,bits);check_layout(g,graw);check_layout(u,uraw);
            auto values=inputs(8,k);
            for(bool fp32:{true,false})for(int split:{1,2,4}) {
                auto x=mx::array(values.data(),{8,k});auto pair=tq_paired_matmul(g,u,x,fp32,split);
                exact(pair.first,tq_tiled_matmul(g,x,split,true,32,fp32));
                exact(pair.second,tq_tiled_matmul(u,x,split,true,32,fp32));
                auto bf=[](const mx::array& a){return mx::astype(mx::astype(a,mx::bfloat16),mx::float32);};
                auto bg=bf(pair.first),bu=bf(pair.second);
                auto ff=bf(mx::multiply(bf(mx::multiply(bg,mx::sigmoid(bg))),bu));
                exact(tq_paired_swiglu(g,u,x,fp32,split),ff);
                for(auto which:{0,1}) {
                    auto ref=oracle(which?uraw:graw,r,k,bits,values,8,fp32);
                    double e=rel_l2(which?pair.second:pair.first,mx::array(ref.data(),{8,r}));
                    if(!std::isfinite(e)||e>2e-5)throw std::runtime_error("paired projection fails original CPU oracle");
                }
                for(int m:{1,3,5}) {
                    auto y=tq_paired_matmul(g,u,mx::array(values.data(),{m,k}),fp32,split);
                    exact(y.first,mx::slice(pair.first,{0,0},{m,r}));exact(y.second,mx::slice(pair.second,{0,0},{m,r}));
                }
                std::cout<<"PAIRED_CPU_PASS bits="<<bits<<" rows="<<r<<" fp32="<<fp32<<" split="<<split<<std::endl;
            }
            bool rejected=false;auto bad=u;bad.columns+=128;
            try{tq_paired_matmul(g,bad,mx::array(values.data(),{8,k}),true);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("paired shape mismatch accepted");
        }
        if(argc==2) {
            auto file=gguf_open(argv[1]);
            for(const auto& p:{"blk.0.","blk.16.","blk.32."}) {
                auto gmat=load_matrix(file,std::string(p)+"ffn_gate.weight"),umat=load_matrix(file,std::string(p)+"ffn_up.weight");
                auto prepare=[](const Matrix& m){if(auto v=std::get_if<Tq3Weights>(&m))return tq3_tile(*v);return tq4_tile(std::get<Tq4Weights>(m));};
                auto g=prepare(gmat),u=prepare(umat);auto values=inputs(8,g.columns);
                for(int width:{1,8})for(bool fp32:{true,false}) {
                    auto x=mx::array(values.data(),{width,g.columns});mx::eval(x,g.planes,g.scales,u.planes,u.scales);
                    auto old=[&]{return std::pair{tq_tiled_matmul(g,x,4,true,32,fp32),tq_tiled_matmul(u,x,4,true,32,fp32)};};
                    auto paired=[&]{return tq_paired_matmul(g,u,x,fp32);};
                    auto a=old(),b=paired();exact(a.first,b.first);exact(a.second,b.second);
                    for(int arm:{0,1,1,0})std::cout<<"PAIRED_BENCH layer="<<p<<" bits="<<g.bits<<" width="<<width<<" fp32="<<fp32<<" arm="<<arm<<" ms="<<(arm?timing(paired):timing(old))<<std::endl;
                }
            }
        } else if(argc!=1)throw std::invalid_argument("usage: test-mlx-paired-gate-up [model.gguf]");
        std::cout<<"PAIRED_GATE_UP_COMPLETE"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<"PAIRED_ERROR "<<e.what()<<std::endl;return 1;}
}

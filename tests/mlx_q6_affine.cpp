#include "q6.hpp"
#include "weights.hpp"
#include "../src/dequant.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mx = mlx::core;
using namespace tqllm;
using namespace tqllm::mlx_backend;
static std::vector<float> input(int m,int k) {
    std::vector<float> x(size_t(m)*k);
    for(size_t i=0;i<x.size();++i)x[i]=0.1f*std::sin(float(i)*0.037f);
    return x;
}
static float error(const mx::array& a,const mx::array& b) {
    return mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(a,b))),
        mx::maximum(mx::sum(mx::square(b)),mx::array(1e-20f)))).item<float>();
}
template<class F> static double time_ms(F f) {
    mx::eval(f());std::vector<double> times;
    for(int i=0;i<7;++i) {
        auto start=std::chrono::steady_clock::now();mx::eval(f());
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(times.begin(),times.end());return times[3];
}
static void fixture() {
    const int r=17,k=512;
    std::vector<uint8_t> bytes(size_t(r)*(k/256)*210);
    const uint16_t scales[]={0x0000,0x0001,0x0400,0x3555,0xb555,0x3c00};
    for(size_t b=0;b<bytes.size()/210;++b) {
        auto p=bytes.data()+b*210;
        for(int j=0;j<192;++j)p[j]=uint8_t(j*37+b*13);
        for(int j=0;j<16;++j)p[192+j]=uint8_t(int(j)*17-128);
        auto scale=scales[b%6];p[208]=uint8_t(scale);p[209]=uint8_t(scale>>8);
    }
    std::vector<float> expected(size_t(r)*k);
    dequantize_host(GGML_Q6_K,bytes.data(),expected.data(),expected.size());
    auto w=load_q6_blocks(bytes,r,k);auto cache=q6_affine_split(w,6);
    std::vector<mx::array> parts;
    for(int part=0;part<2;++part)parts.push_back(mx::dequantize(cache.codes[part],cache.scales[part],cache.biases[part],32,6,"affine"));
    auto decoded=mx::add(parts[0],parts[1]);mx::eval(decoded);
    for(size_t i=0;i<expected.size();++i)
        if(decoded.data<float>()[i]!=expected[i])throw std::runtime_error("native6 stored-weight mismatch");
    for(int m:{1,3,5,8}) {
        auto x=input(m,k);std::vector<float> ref(size_t(m)*r);
        for(int a=0;a<m;++a)for(int row=0;row<r;++row) {
            double sum=0;for(int j=0;j<k;++j)sum+=double(expected[row*k+j])*x[a*k+j];
            ref[a*r+row]=float(sum);
        }
        float e=error(q6_affine_matmul(cache,mx::array(x.data(),{m,k})),mx::array(ref.data(),{m,r}));
        if(!std::isfinite(e)||e>2e-5f)throw std::runtime_error("native6 projection mismatch");
        std::cout<<"Q6_NATIVE6_CPU_PASS width="<<m<<" rel_l2="<<e<<std::endl;
    }
    bool rejected=false;try{q6_affine_split(w,4);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("unsupported cache bits accepted");
}
int main(int argc,char** argv) {
    try {
        fixture();
        if(argc==2) {
            auto file=gguf_open(argv[1]);auto matrix=load_matrix(file,"output.weight");
            const auto& w=std::get<Q6Weights>(matrix);int r=w.blocks.shape(0),k=w.blocks.shape(1)*256;
            auto eight=q6_affine_split(w,8),six=q6_affine_split(w,6);
            size_t bytes8=0,bytes6=0;
            for(const auto& a:eight.codes)bytes8+=a.nbytes();for(const auto& a:six.codes)bytes6+=a.nbytes();
            std::cout<<"Q6_NATIVE6_PAYLOAD bytes8="<<bytes8<<" bytes6="<<bytes6<<std::endl;
            for(int m:{1,3,5,8}) {
                auto x=input(m,k);auto a=mx::array(x.data(),{m,k});mx::eval(a);
                auto ref=q6_matmul(w,a),candidate=q6_affine_matmul(six,a);float e=error(candidate,ref);
                if(!std::isfinite(e)||e>2e-5f)throw std::runtime_error("full head native6 drift");
                auto ids=mx::argmax(candidate,1),refids=mx::argmax(ref,1);mx::eval(ids,refids);
                int same=0;for(int j=0;j<m;++j)same+=ids.data<uint32_t>()[j]==refids.data<uint32_t>()[j];
                for(int arm:{0,1,1,0})std::cout<<"Q6_NATIVE6_BENCH width="<<m<<" arm="<<arm<<" ms="
                    <<time_ms([&]{return q6_affine_matmul(arm?six:eight,a);})<<" original_rel_l2="<<e
                    <<" argmax_same="<<same<<" rows="<<m<<std::endl;
            }
        } else if(argc!=1)throw std::invalid_argument("usage: test-mlx-q6-affine [model.gguf]");
        std::cout<<"Q6_NATIVE6_SCREEN_COMPLETE"<<std::endl;return 0;
    } catch(const std::exception& e){std::cerr<<"Q6_NATIVE6_ERROR "<<e.what()<<std::endl;return 1;}
}

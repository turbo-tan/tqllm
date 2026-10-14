#include "dflash.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mx=mlx::core;
using tqllm::mlx_backend::dflash_select;
using tqllm::mlx_backend::dflash_top16_sharded;
namespace {
float bf(float x) {
    auto b=std::bit_cast<uint32_t>(x);
    return std::bit_cast<float>((b+0x7fffu+((b>>16)&1u))&0xffff0000u);
}
std::vector<uint32_t> legacy(const mx::array& mapped,const mx::array& unary,
    const mx::array& gate,const mx::array& successors,const mx::array& predecessors,
    uint32_t anchor,int count) {
    mx::eval(mapped);
    std::vector<uint32_t> result;
    for(int step=1;step<=count;++step) {
        auto previous=mx::astype(mx::take(predecessors,mx::array(anchor),0),mx::float32);
        auto g=mx::slice(gate,{step,0},{step+1,256});
        auto s=mx::reshape(mx::slice(successors,{step,0,0},{step+1,16,256}),{16,256});
        auto u=mx::reshape(mx::slice(unary,{step,0},{step+1,16}),{16});
        auto scores=mx::add(mx::sum(mx::multiply(s,mx::multiply(previous,g)),-1),u);
        auto best=mx::argmax(scores).item<uint32_t>();
        anchor=mapped.data<uint32_t>()[step*16+best];result.push_back(anchor);
    }
    return result;
}

void check_top16() {
    constexpr int rows=8,vocab=257;
    std::vector<float> logits(rows*vocab);
    for(int r=0;r<rows;++r)for(int c=0;c<vocab;++c) {
        float value=std::sin(float((r+3)*(c+11))*0.017f)+0.001f*float((c*13+r*7)%29);
        logits[r*vocab+c]=value;
    }
    logits[2*vocab+17]=3.0f;logits[2*vocab+29]=3.0f;
    auto ids=dflash_top16_sharded(mx::array(logits.data(),{rows,vocab}));mx::eval(ids);
    const auto* got=ids.data<uint32_t>();
    for(int r=0;r<rows;++r) {
        std::vector<uint32_t> order(vocab);
        for(int c=0;c<vocab;++c)order[c]=uint32_t(c);
        std::stable_sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b) {
            float va=logits[r*vocab+a],vb=logits[r*vocab+b];
            return va>vb || (va==vb && a<b);
        });
        for(int k=0;k<16;++k)
            if(got[r*16+k]!=order[k])throw std::runtime_error("sharded top16 differs from CPU order");
    }
    std::cout<<"DFLASH_TOP16_PASS rows="<<rows<<" vocab="<<vocab<<'\n';
}
}
int main() {
    try {
        check_top16();
        constexpr int vocab=37;
        std::vector<uint32_t> ids(8*16);
        std::vector<float> unary(8*16),gate(8*256),successors(8*16*256),pred(vocab*256);
        int passed=0;
        for(int seed=0;seed<6;++seed) {
            for(size_t j=0;j<ids.size();++j) {
                ids[j]=uint32_t((j*17+seed*11)%vocab);
                unary[j]=seed?std::sin(float(j+seed)*0.311f):0.0f;
            }
            for(size_t j=0;j<gate.size();++j)gate[j]=seed?bf(std::sin(float(j+seed)*0.019f)):0.0f;
            for(size_t j=0;j<pred.size();++j)pred[j]=bf(std::cos(float(j+seed)*0.041f));
            for(size_t j=0;j<successors.size();++j)successors[j]=bf(std::sin(float(j+seed)*0.023f));
            auto im=mx::array(ids.data(),{8,16},mx::uint32),u=mx::array(unary.data(),{8,16});
            auto g=mx::array(gate.data(),{8,256}),s=mx::array(successors.data(),{8,16,256});
            auto p=mx::astype(mx::array(pred.data(),{vocab,256}),mx::bfloat16);
            mx::eval(im,u,g,s,p);
            for(uint32_t anchor:{0u,3u,uint32_t(vocab-1)})for(int count=1;count<=7;++count) {
                auto got=dflash_select(im,u,g,s,p,anchor,count);mx::eval(got);
                auto expected=legacy(im,u,g,s,p,anchor,count);
                uint32_t previous=anchor;
                for(int step=1;step<=count;++step) {
                    double maximum=-INFINITY;int winner=0;
                    for(int c=0;c<16;++c) {
                        double sum=0;
                        for(int k=0;k<256;++k) {
                            float weighted=pred[previous*256+k]*gate[step*256+k];
                            float product=successors[(step*16+c)*256+k]*weighted;
                            sum+=double(product);
                        }
                        double score=sum+unary[step*16+c];
                        if(score>maximum){maximum=score;winner=c;}
                    }
                    previous=ids[step*16+winner];
                    if(got.data<uint32_t>()[step-1]!=previous || expected[step-1]!=previous)
                        throw std::runtime_error("device selector differs from CPU or legacy chain");
                }
                ++passed;
            }
            if(seed==5) {
                auto measure=[&](bool device) {
                    for(int j=0;j<3;++j) {
                        if(device){auto y=dflash_select(im,u,g,s,p,3,7);mx::eval(y);}
                        else legacy(im,u,g,s,p,3,7);
                    }
                    auto start=std::chrono::steady_clock::now();
                    for(int j=0;j<30;++j) {
                        if(device){auto y=dflash_select(im,u,g,s,p,3,7);mx::eval(y);}
                        else legacy(im,u,g,s,p,3,7);
                    }
                    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/30.0;
                };
                std::cout<<"SELECTOR_MICRO legacy_ms="<<measure(false)<<" device_ms="<<measure(true)<<'\n';
                int rejected=0;
                for(int count:{0,8})try{dflash_select(im,u,g,s,p,0,count);}catch(const std::invalid_argument&){++rejected;}
                try{dflash_select(im,u,g,s,p,vocab,7);}catch(const std::invalid_argument&){++rejected;}
                try{dflash_select(mx::astype(im,mx::int32),u,g,s,p,0,7);}catch(const std::invalid_argument&){++rejected;}
                try{dflash_select(im,u,g,s,mx::astype(p,mx::float32),0,7);}catch(const std::invalid_argument&){++rejected;}
                if(rejected!=5)throw std::runtime_error("selector invalid dimensions accepted");
                auto invalid=mx::full({8,16},uint32_t(vocab),mx::uint32);
                auto y=dflash_select(invalid,u,g,s,p,0,7);mx::eval(y);
                for(int j=0;j<7;++j)if(y.data<uint32_t>()[j]!=uint32_t(vocab))
                    throw std::runtime_error("invalid vocabulary guard failed");
            }
        }
        std::cout<<"DFLASH_SELECTOR_PASS cases="<<passed<<" cpu+legacy+ties+invalid\n";
    }catch(const std::exception& e){std::cerr<<"DFLASH_SELECTOR_ERROR "<<e.what()<<'\n';return 1;}
}

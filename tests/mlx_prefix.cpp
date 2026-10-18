#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using tqllm::mlx_backend::Qwen35;
template<class F> void rejects(F operation) {
    try {operation();}catch(const std::invalid_argument&) {return;}
    throw std::runtime_error("invalid verification operation was accepted");
}
int main(int argc,char** argv) {
    try {
        const bool sequential = argc==3 && std::string(argv[2])=="--sequential-reference";
        if(argc!=2 && !sequential)throw std::invalid_argument("usage: test-mlx-prefix model.gguf [--sequential-reference]");
        mx::set_cache_limit(512*1024*1024);
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);
        auto prompt=tokenizer.encode("Explain why rainbows form, including refraction and reflection.");
        auto tokens=tokenizer.encode("The light bends through water droplets before it reflects and emerges.");
        if(tokens.size()<9)throw std::runtime_error("prefix test token fixture is too short");
        Qwen35 engine(argv[1],256);
        if(sequential) {
            const auto file=tqllm::gguf_open(argv[1]);
            const int layers=int(file.num("qwen35.block_count"))-int(file.num("qwen35.nextn_predict_layers",0));
            std::vector<int> selected(layers);std::iota(selected.begin(),selected.end(),0);
            engine.feature_layers(selected);
        }
        engine.gdn_mode(Qwen35::GdnMode::scan);engine.verify_tiles(true);
        engine.forward(prompt);
        const auto start=engine.checkpoint();const int position=engine.position(),vocab=engine.vocabulary();
        Qwen35::Verification ticket;
        rejects([&]{engine.accept_prefix(ticket,0);});
        for(int accepted=0;accepted<=8;++accepted) {
            engine.restore(start);
            if(sequential) {
                for(int t=0;t<accepted;++t)engine.forward({tokens.data()+t,1},true);
            } else if(accepted)engine.forward({tokens.data(),size_t(accepted)},true);
            auto reference=engine.forward({&tokens[8],1});
            std::vector<float> a(reference.data<float>(),reference.data<float>()+vocab);
            std::vector<float> feature_reference;
            if(sequential) {
                auto features=engine.feature_states();mx::eval(features);
                feature_reference.assign(features.data<float>(),features.data<float>()+features.size());
            }
            engine.restore(start);
            auto begin=std::chrono::steady_clock::now();
            engine.verify({tokens.data(),8},ticket);
            const double verify_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            rejects([&]{engine.accept_prefix(ticket,9);});
            begin=std::chrono::steady_clock::now();
            engine.accept_prefix(ticket,accepted);
            const double commit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            if(engine.position()!=position+accepted)throw std::runtime_error("wrong accepted position");
            rejects([&]{engine.accept_prefix(ticket,accepted);});
            auto candidate=engine.forward({&tokens[8],1});const auto* b=candidate.data<float>();
            const auto ia=std::max_element(a.begin(),a.end())-a.begin(),ib=std::max_element(b,b+vocab)-b;
            const double ma=a[ia],mb=b[ib];double sa=0,sb=0,max_error=0,kld=0;
            for(int j=0;j<vocab;++j) {sa+=std::exp(double(a[j])-ma);sb+=std::exp(double(b[j])-mb);max_error=std::max(max_error,std::abs(double(a[j])-b[j]));}
            const double za=ma+std::log(sa),zb=mb+std::log(sb);
            for(int j=0;j<vocab;++j)kld+=std::exp(double(a[j])-za)*(double(a[j])-za-double(b[j])+zb);
            std::cout<<"PREFIX_CHECK accepted="<<accepted<<" kld="<<kld<<" top1_same="<<(ia==ib)
                     <<" max_abs="<<max_error<<" verify_ms="<<verify_ms<<" commit_ms="<<commit_ms
                     <<" active_bytes="<<mx::get_active_memory()<<std::endl;
            if(!std::isfinite(kld) || kld>0.002 || ((sequential || accepted==0 || accepted==8) && max_error!=0)) {
                if(sequential) {
                    auto features=engine.feature_states();mx::eval(features);
                    const int hidden=int(tqllm::gguf_open(argv[1]).num("qwen35.embedding_length"));
                    for(size_t i=0;i<feature_reference.size();++i) if(feature_reference[i]!=features.data<float>()[i]) {
                        std::cerr<<"PREFIX_FIRST_DIFFERENCE accepted="<<accepted<<" layer_input="<<i/size_t(hidden)<<'\n';break;
                    }
                }
                throw std::runtime_error("prefix state continuation failed accuracy gate");
            }
        }
        engine.restore(start);engine.verify({tokens.data(),8},ticket);engine.forward({&tokens[8],1});
        rejects([&]{engine.accept_prefix(ticket,1);});
        engine.restore(start);engine.verify({tokens.data(),8},ticket);engine.restore(start);
        rejects([&]{engine.accept_prefix(ticket,1);});
        engine.verify({tokens.data(),8},ticket);engine.reset();
        rejects([&]{engine.accept_prefix(ticket,1);});
        engine.gdn_mode(Qwen35::GdnMode::baseline);
        rejects([&]{engine.verify({tokens.data(),8},ticket);});
        std::cout<<"PREFIX_PASS accepted=0..8 stale_forward=1 stale_restore=1 stale_reset=1 duplicate=1 invalid_count=1"<<std::endl;
    } catch(const std::exception& e) {std::cerr<<"PREFIX_ERROR "<<e.what()<<'\n';return 1;}
}

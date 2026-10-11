#include "qwen35.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using tqllm::mlx_backend::Qwen35;
int main(int argc,char** argv) {
    try {
        if(argc<3 || argc>5)throw std::invalid_argument("usage: tqmlx-kld model.gguf corpus.txt [chunks] [--verify]");
        const int chunks=argc>=4?std::stoi(argv[3]):2,rows=512,batch=8;
        const bool verification=argc==5 && std::string(argv[4])=="--verify";
        if(argc==5 && !verification)throw std::invalid_argument("unknown accuracy mode");
        if(chunks<1 || chunks>100)throw std::invalid_argument("chunks must be 1..100");
        std::ifstream in(argv[2]);if(!in)throw std::runtime_error("cannot open corpus");
        std::string text{std::istreambuf_iterator<char>(in),{}};
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);auto tokens=tokenizer.encode(text);
        if(tokens.size()<size_t(chunks)*rows+1)throw std::invalid_argument("corpus too short");
        mx::set_cache_limit(512*1024*1024);
        Qwen35 engine(argv[1],rows+1);engine.verify_tiles(true);
        const int vocab=engine.vocabulary();
        std::vector<double> chunk_kld,chunk_ppl;
        int total_same=0;
        for(int chunk=0;chunk<chunks;++chunk) {
            const auto* input=tokens.data()+size_t(chunk)*rows;
            engine.reset();engine.gdn_mode(verification?Qwen35::GdnMode::scan:Qwen35::GdnMode::baseline);
            engine.verify_tiles(!verification);
            std::vector<float> reference(size_t(rows)*vocab);
            const int reference_batch=verification?1:batch;
            for(int off=0;off<rows;off+=reference_batch) {
                auto y=engine.forward({input+off,size_t(reference_batch)},true);
                std::copy(y.data<float>(),y.data<float>()+y.size(),reference.begin()+size_t(off)*vocab);
            }
            engine.reset();engine.gdn_mode(Qwen35::GdnMode::scan);engine.verify_tiles(true);
            double kl_sum=0,ppl_sum=0;int same=0;
            for(int off=0;off<rows;off+=batch) {
                auto y=engine.forward({input+off,batch},true);const auto* values=y.data<float>();
                for(int r=0;r<batch;++r) {
                    const auto* a=reference.data()+size_t(off+r)*vocab;
                    const auto* b=values+size_t(r)*vocab;
                    const auto ia=std::max_element(a,a+vocab)-a,ib=std::max_element(b,b+vocab)-b;
                    const double ma=a[ia],mb=b[ib];double sa=0,sb=0;
                    for(int j=0;j<vocab;++j) {sa+=std::exp(double(a[j])-ma);sb+=std::exp(double(b[j])-mb);}
                    const double za=ma+std::log(sa),zb=mb+std::log(sb);
                    double kl=0;
                    for(int j=0;j<vocab;++j)kl+=std::exp(double(a[j])-za)*(double(a[j])-za-double(b[j])+zb);
                    const auto target=input[off+r+1];
                    if(target>=uint32_t(vocab))throw std::runtime_error("target token outside vocabulary");
                    const double delta=(zb-b[target])-(za-a[target]);
                    if(!std::isfinite(kl) || !std::isfinite(delta) || kl < -1e-6)
                        throw std::runtime_error("invalid accuracy statistic");
                    kl_sum+=kl;ppl_sum+=delta;same+=ia==ib;
                }
            }
            total_same+=same;chunk_kld.push_back(kl_sum/rows);chunk_ppl.push_back(ppl_sum/rows);
            std::cout<<"{\"kind\":\""<<(verification?"same_file_verify_screen":"same_file_gdn_screen")<<"\",\"chunk\":"<<chunk<<",\"rows\":"<<rows
                     <<",\"reference_batch\":"<<reference_batch
                     <<",\"batch\":"<<batch<<",\"mean_kld\":"<<kl_sum/rows<<",\"ln_ppl_ratio\":"<<ppl_sum/rows
                     <<",\"top1_same\":"<<same<<",\"active_bytes\":"<<mx::get_active_memory()<<"}"<<std::endl;
        }
        auto mean=[](const auto& x){return std::accumulate(x.begin(),x.end(),0.0)/x.size();};
        auto se=[&](const auto& x) {if(x.size()<2)return 0.0;double ss=0,m=mean(x);for(double a:x)ss+=(a-m)*(a-m);return std::sqrt(ss/(x.size()*(x.size()-1)));};
        std::cout<<"{\"kind\":\""<<(verification?"same_file_verify_summary":"same_file_gdn_summary")<<"\",\"chunks\":"<<chunks<<",\"rows_per_chunk\":"<<rows
                 <<",\"mean_kld\":"<<mean(chunk_kld)<<",\"kld_chunk_se\":"<<se(chunk_kld)
                 <<",\"ln_ppl_ratio\":"<<mean(chunk_ppl)<<",\"ppl_chunk_se\":"<<se(chunk_ppl)
                 <<",\"top1_same\":"<<total_same<<"}"<<std::endl;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

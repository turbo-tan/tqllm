#include "qwen35.hpp"
#include "mlx_kld_window.hpp"
#include "mlx_kld_settings.hpp"
#include "tqllm/tokenizer.hpp"
#include <mlx/memory.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
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
        if(argc<3 || argc>6)throw std::invalid_argument("usage: tqmlx-kld model.gguf corpus.txt [chunks] [--verify|--kernel-verify] [--start-chunk=N]");
        const int chunks=argc>=4?tqllm::mlx_backend::kld_integer(argv[3]):2,rows=512,batch=8;
        const bool kernel_verification=argc>=5 && std::string(argv[4])=="--kernel-verify";
        const bool verification=kernel_verification || (argc>=5 && std::string(argv[4])=="--verify");
        const auto setting=[](const char* name)->std::optional<std::string> {
            const auto* value=std::getenv(name);
            return value?std::optional<std::string>(value):std::nullopt;
        };
        const auto subgroup=setting("TQLLM_MLX_TQ3_SUBGROUP"),columns=setting("TQLLM_MLX_MATMUL_COLS");
        const auto tiled_tq3=setting("TQLLM_MLX_TILED_TQ3_FP16");
        const auto tiled_tq3_fp32=setting("TQLLM_MLX_TILED_TQ3_FP32");
        const auto tiled_tq4_fp32=setting("TQLLM_MLX_TILED_TQ4_FP32");
        const auto tiled_mixer_fp32=setting("TQLLM_MLX_TILED_MIXER_FP32");
        const auto tiled_tq4_fp16=setting("TQLLM_MLX_TILED_TQ4_FP16");
        const auto tiled_mixer_fp16=setting("TQLLM_MLX_TILED_MIXER_FP16");
        const auto paired=setting("TQLLM_MLX_PAIRED_GATE_UP");
        tqllm::mlx_backend::KldKernelSettings kernel_settings;
        if(kernel_verification && (!subgroup || *subgroup!="8"))
            throw std::invalid_argument("kernel verification requires the subgroup8 candidate");
        if(argc>=5 && !verification)throw std::invalid_argument("unknown accuracy mode");
        int first = 0;
        if (argc == 6) {
            std::string_view offset(argv[5]);
            if (!offset.starts_with("--start-chunk=")) throw std::invalid_argument("unknown accuracy offset");
            first = tqllm::mlx_backend::kld_integer(offset.substr(14));
        }
        std::ifstream in(argv[2]);if(!in)throw std::runtime_error("cannot open corpus");
        std::string text{std::istreambuf_iterator<char>(in),{}};
        tqllm::Tokenizer tokenizer;tokenizer.load(argv[1]);auto tokens=tokenizer.encode(text);
        const size_t begin = tqllm::mlx_backend::kld_window_begin(first,chunks,tokens.size());
        mx::set_cache_limit(512*1024*1024);
        Qwen35 engine(argv[1],rows+1);engine.verify_tiles(true);
        const int vocab=engine.vocabulary();
        std::vector<double> chunk_kld,chunk_ppl;
        int total_same=0;
        for(int chunk=0;chunk<chunks;++chunk) {
            const auto* input=tokens.data()+begin+size_t(chunk)*rows;
            if(kernel_verification) {
                kernel_settings.clear();
            }
            engine.reset();engine.gdn_mode(verification?Qwen35::GdnMode::scan:Qwen35::GdnMode::baseline);
            engine.verify_tiles(!verification);
            std::vector<float> reference(size_t(rows)*vocab);
            const int reference_batch=verification?1:batch;
            for(int off=0;off<rows;off+=reference_batch) {
                auto y=engine.forward({input+off,size_t(reference_batch)},true);
                std::copy(y.data<float>(),y.data<float>()+y.size(),reference.begin()+size_t(off)*vocab);
            }
            if(kernel_verification) {
                kernel_settings.restore();
            }
            engine.reset();engine.gdn_mode(Qwen35::GdnMode::scan);engine.verify_tiles(true);
            double kl_sum=0,ppl_sum=0;int same=0;
            for(int off=0;off<rows;off+=batch) {
                Qwen35::Verification ticket;
                auto y=kernel_verification?engine.verify({input+off,batch},ticket):engine.forward({input+off,batch},true);
                const auto* values=y.data<float>();
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
                if(kernel_verification)engine.accept_prefix(ticket,batch);
            }
            total_same+=same;chunk_kld.push_back(kl_sum/rows);chunk_ppl.push_back(ppl_sum/rows);
            std::cout<<"{\"kind\":\""<<(kernel_verification?"same_file_kernel_verify_screen":verification?"same_file_verify_screen":"same_file_gdn_screen")<<"\",\"chunk\":"<<(first+chunk)<<",\"rows\":"<<rows
                     <<",\"reference_batch\":"<<reference_batch
                     <<",\"candidate_fp16_tq3\":"<<bool(tiled_tq3)<<",\"candidate_fp32_tq3\":"<<bool(tiled_tq3_fp32)<<",\"candidate_fp32_tq4\":"<<bool(tiled_tq4_fp32)<<",\"candidate_fp32_mixer\":"<<bool(tiled_mixer_fp32)<<",\"candidate_fp16_tq4\":"<<bool(tiled_tq4_fp16)<<",\"candidate_fp16_mixer\":"<<bool(tiled_mixer_fp16)<<",\"candidate_paired_gate_up\":"<<bool(paired)<<",\"captures_prefix\":"<<kernel_verification
                     <<",\"batch\":"<<batch<<",\"mean_kld\":"<<kl_sum/rows<<",\"ln_ppl_ratio\":"<<ppl_sum/rows
                     <<",\"top1_same\":"<<same<<",\"active_bytes\":"<<mx::get_active_memory()<<"}"<<std::endl;
        }
        auto mean=[](const auto& x){return std::accumulate(x.begin(),x.end(),0.0)/x.size();};
        auto se=[&](const auto& x) {if(x.size()<2)return 0.0;double ss=0,m=mean(x);for(double a:x)ss+=(a-m)*(a-m);return std::sqrt(ss/(x.size()*(x.size()-1)));};
        std::cout<<"{\"kind\":\""<<(kernel_verification?"same_file_kernel_verify_summary":verification?"same_file_verify_summary":"same_file_gdn_summary")<<"\",\"chunks\":"<<chunks<<",\"first_chunk\":"<<first<<",\"rows_per_chunk\":"<<rows
                 <<",\"candidate_fp16_tq3\":"<<bool(tiled_tq3)<<",\"candidate_fp32_tq3\":"<<bool(tiled_tq3_fp32)<<",\"candidate_fp32_tq4\":"<<bool(tiled_tq4_fp32)<<",\"candidate_fp32_mixer\":"<<bool(tiled_mixer_fp32)<<",\"candidate_fp16_tq4\":"<<bool(tiled_tq4_fp16)<<",\"candidate_fp16_mixer\":"<<bool(tiled_mixer_fp16)<<",\"candidate_paired_gate_up\":"<<bool(paired)<<",\"mean_kld\":"<<mean(chunk_kld)<<",\"kld_chunk_se\":"<<se(chunk_kld)
                 <<",\"ln_ppl_ratio\":"<<mean(chunk_ppl)<<",\"ppl_chunk_se\":"<<se(chunk_ppl)
                 <<",\"top1_same\":"<<total_same<<"}"<<std::endl;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

#include "dense_rows.hpp"
#include "weights.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace mx=mlx::core;
using namespace tqllm::mlx_backend;

static void check(const mx::array& w,const mx::array& x,bool bench) {
    std::vector<mx::array> rows;
    for(int i=0;i<x.shape(0);++i)
        rows.push_back(mx::matmul(mx::contiguous(mx::slice(x,{i,0},{i+1,x.shape(1)})),mx::transpose(w)));
    auto reference=mx::contiguous(mx::concatenate(rows,0));
    auto got=mx::contiguous(dense_decode_rows(w,dense_row_views(x)));
    mx::eval(reference,got);
    for(size_t i=0;i<reference.size();++i)
        if(!std::isfinite(got.data<float>()[i]) || got.data<float>()[i]!=reference.data<float>()[i])
            throw std::runtime_error("batched singleton dense projection changes FP32 decode arithmetic");
    std::cout<<"DENSE_ROWS_EXACT rows="<<w.shape(0)<<" k="<<w.shape(1)<<" width="<<x.shape(0)<<std::endl;
    if(!bench)return;
    for(int arm:{0,1,1,0}) {
        std::vector<double> times;
        for(int run=-1;run<7;++run) {
            auto begin=std::chrono::steady_clock::now();
            if(arm)mx::eval(dense_decode_rows(w,dense_row_views(x)));
            else {
                std::vector<mx::array> values;
                for(int i=0;i<x.shape(0);++i)
                    values.push_back(mx::matmul(mx::contiguous(mx::slice(x,{i,0},{i+1,x.shape(1)})),mx::transpose(w)));
                mx::eval(mx::concatenate(values,0));
            }
            if(run>=0)times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
        }
        std::sort(times.begin(),times.end());
        std::cout<<"DENSE_ROWS_BENCH rows="<<w.shape(0)<<" k="<<w.shape(1)<<" width="<<x.shape(0)
                 <<" arm="<<arm<<" elapsed_ms="<<times[times.size()/2]<<std::endl;
    }
}

int main(int argc,char** argv) {
    try {
        if(argc>2)throw std::invalid_argument("usage: test-mlx-dense-rows [model.gguf]");
        for(int k:{128,512,5120})for(int r:{17,48,128})for(int m:{1,3,5,8}) {
            std::vector<float> weights(size_t(r)*k),input(size_t(m)*k);
            for(size_t i=0;i<weights.size();++i)weights[i]=.1f*std::sin(float(i)*.023f);
            for(size_t i=0;i<input.size();++i)input[i]=.1f*std::cos(float(i)*.071f);
            check(mx::array(weights.data(),{r,k}),mx::array(input.data(),{m,k}),false);
        }
        for(int bad:{0,1,2}) {
            bool caught=false;
            try {dense_row_views(mx::zeros({bad==0?17:1,128},bad==1?mx::float16:mx::float32));
                if(bad==2)dense_decode_rows(mx::zeros({48,256}),mx::zeros({1,1,128}));}
            catch(const std::invalid_argument&) {caught=true;}
            if(!caught)throw std::runtime_error("invalid singleton dense batch accepted");
        }
        if(argc==2) {
            auto file=tqllm::gguf_open(argv[1]);
            for(const auto* name:{"blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight"}) {
                auto matrix=load_matrix(file,name);
                auto w=mx::astype(std::get<mx::array>(matrix),mx::float32);
                const int k=w.shape(1);
                std::vector<float> input(size_t(8)*k);
                for(size_t i=0;i<input.size();++i)input[i]=.1f*std::sin(float(i)*.037f);
                check(w,mx::array(input.data(),{8,k}),true);
            }
        }
        std::cout<<"DENSE_ROWS_PASS"<<std::endl;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

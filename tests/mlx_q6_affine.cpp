#include "q6.hpp"
#include "weights.hpp"
#include "dequant.hpp"
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
    for(auto& a:parts)a=mx::reshape(mx::slice(mx::reshape(a,{r,k/32,2,16}),{0,0,0,0},{r,k/32,1,16}),{r,k/32,16});
    auto decoded=mx::reshape(mx::concatenate(parts,2),{r,k});mx::eval(decoded);
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
        const std::vector<uint32_t> ids={0,uint32_t(r-1),uint32_t(r/2),uint32_t((m+3)%r)};
        std::vector<uint32_t> tiled_ids(size_t(m)*ids.size());
        for(int row=0;row<m;++row)std::copy(ids.begin(),ids.end(),tiled_ids.begin()+size_t(row)*ids.size());
        auto selected=q6_rows_matmul(w,mx::array(x.data(),{m,k}),
            mx::array(tiled_ids.data(),{m,int(ids.size())},mx::uint32));
        mx::eval(selected);
        for(int row=0;row<m;++row)for(size_t j=0;j<ids.size();++j) {
            float want=ref[size_t(row)*r+ids[j]],got=selected.data<float>()[size_t(row)*ids.size()+j];
            if(std::abs(want-got)>2e-3f)throw std::runtime_error("Q6 row projection mismatch");
        }
        auto sparse=sparse_logits(mx::array(tiled_ids.data(),{m,int(ids.size())},mx::uint32),selected,r);
        auto sparse_ids=mx::argmax(sparse,1);
        auto best_ids=mx::take_along_axis(mx::array(tiled_ids.data(),{m,int(ids.size())},mx::uint32),
            mx::expand_dims(mx::argmax(selected,1),1),1);
        best_ids=mx::reshape(best_ids,{m});
        std::vector<float> probe={0.0f,4.0f,2.0f,3.0f,9.0f,1.0f,8.0f,7.0f,6.0f,5.0f,10.0f,11.0f};
        auto shard_ids=shard_topk_ids(mx::array(probe.data(),{1,int(probe.size())}),4);
        mx::eval(sparse_ids,best_ids,shard_ids);
        bool saw11=false,saw10=false;
        for(int j=0;j<shard_ids.shape(1);++j) {
            saw11|=shard_ids.data<uint32_t>()[j]==11;
            saw10|=shard_ids.data<uint32_t>()[j]==10;
        }
        if(!saw11||!saw10)throw std::runtime_error("shard top-k missed shard winners");
        for(int row=0;row<m;++row) {
            uint32_t best=ids[0];float bestv=ref[size_t(row)*r+best];
            for(uint32_t id:ids)if(ref[size_t(row)*r+id]>bestv){best=id;bestv=ref[size_t(row)*r+id];}
            if(sparse_ids.data<uint32_t>()[row]!=best)throw std::runtime_error("sparse logits argmax mismatch");
            if(best_ids.data<uint32_t>()[row]!=best)throw std::runtime_error("selected ids argmax mismatch");
        }
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

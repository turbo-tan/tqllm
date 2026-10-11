#include "draft_head.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mx=mlx::core;
using namespace tqllm::mlx_backend;
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("usage: test-mlx-draft-head model.gguf");
        auto file=tqllm::gguf_open(argv[1]);auto tensor=file.tensors.at("output.weight");
        const int rows=17,k=int(tensor.ne[0]);tensor.bytes=tensor.bytes/tensor.ne[1]*rows;tensor.ne[1]=rows;
        auto raw=file.read(tensor);auto original=load_matrix(tensor,{reinterpret_cast<const uint8_t*>(raw.data()),raw.size()});
        auto head=draft_head4(original);auto scales=mx::astype(head.scales,mx::float32),biases=mx::astype(head.biases,mx::float32);
        auto decoded=mx::astype(mx::dequantize(head.codes,head.scales,head.biases,64,4,"affine"),mx::float32);
        mx::eval(scales,biases,decoded);const auto* codes=head.codes.data<uint32_t>();
        auto weight=[&](int r,int j) {
            const int code=int((codes[r*(k/8)+j/8]>>(4*(j%8)))&15u);
            return scales.data<float>()[r*(k/64)+j/64]*code+biases.data<float>()[r*(k/64)+j/64];
        };
        for(int r=0;r<rows;++r)for(int j=0;j<k;++j)
            if(decoded.data<float>()[r*k+j]!=weight(r,j))
                throw std::runtime_error("proposal head affine dequantization differs from CPU");
        for(int m:{1,8}) {
            std::vector<float> x(size_t(m)*k);for(size_t j=0;j<x.size();++j)x[j]=std::sin(float(j)*.037f)*.1f;
            auto y=draft_head4_matmul(head,mx::array(x.data(),{m,k}));mx::eval(y);double difference=0,total=0;
            for(int t=0;t<m;++t)for(int r=0;r<rows;++r) {
                double expected=0;for(int j=0;j<k;++j)
                    expected+=double(weight(r,j))*x[t*k+j];
                const double error=y.data<float>()[t*rows+r]-expected;
                difference+=error*error;total+=expected*expected;
            }
            const double relative=std::sqrt(difference/std::max(total,1e-20));
            std::cout<<"DRAFT_HEAD_CPU_CHECK m="<<m<<" rel_l2="<<relative<<std::endl;
            if(!std::isfinite(relative) || relative>1e-3)throw std::runtime_error("proposal head projection differs from CPU");
        }
        std::cout<<"DRAFT_HEAD_PASS proposal_only=1 packed_bits=4 group=64"<<std::endl;
    } catch(const std::exception& e) {std::cerr<<"DRAFT_HEAD_ERROR "<<e.what()<<'\n';return 1;}
}

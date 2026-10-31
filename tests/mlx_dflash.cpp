#include "dflash.hpp"
#include <mlx/random.h>
#include <bit>
#include <optional>
#include <utility>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
// the reference keeps the kernel's order with no fused multiply-add (Apple clang contracts a += b * c by default)
#pragma clang fp contract(off)
namespace mx=mlx::core;
float bf(float x) {auto b=std::bit_cast<uint32_t>(x);return std::bit_cast<float>((b+0x7fffu+((b>>16)&1u))&0xffff0000u);}
int main() {
    try {
        // the drafter's shape (16 x 5120, group 16) and a small one; side 0 / 1, with and without the residual
        for(auto [m,h]:std::vector<std::pair<int,int>>{{8,32},{16,5120}}) {
        const int group=16,g=h/group;
        std::vector<float> input(m*h),dynamic(m*4*g),base(4*h),residual(m*h);
        for(size_t j=0;j<input.size();++j)input[j]=bf(std::sin(float(j)*0.17f));
        for(size_t j=0;j<dynamic.size();++j)dynamic[j]=bf(std::cos(float(j)*0.07f));
        for(size_t j=0;j<base.size();++j)base[j]=std::sin(float(j)*0.01f);
        for(size_t j=0;j<residual.size();++j)residual[j]=bf(std::cos(float(j)*0.013f)*3.0f);
        for(int side=0;side<2;++side)for(bool with_residual:{false,true}) {
            std::optional<mx::array> res;
            if(with_residual)res=mx::array(residual.data(),{m,h});
            auto got=tqllm::mlx_backend::dflash_conv(mx::array(input.data(),{m,h}),mx::array(dynamic.data(),{m,4*g}),mx::array(base.data(),{2,2,h}),group,side,res);
            mx::eval(got);double maximum=0;
            for(int t=0;t<m;++t)for(int c=0;c<h;++c) {
                float result=0;for(int tap=0;tap<2;++tap)if(t>=tap) {
                    const float w=dynamic[t*4*g+(2*side+tap)*g+c/group]+base[(2*side+tap)*h+c];
                    const float product=w*input[(t-tap)*h+c];
                    result=result+product;
                }
                float expected=bf(result);
                if(with_residual)expected=bf(residual[t*h+c]+expected);
                maximum=std::max(maximum,double(std::abs(got.data<float>()[t*h+c]-expected)));
            }
            if(maximum!=0) {
                std::cerr<<"m="<<m<<" h="<<h<<" side="<<side<<" residual="<<with_residual<<" max_abs="<<maximum<<'\n';
                throw std::runtime_error("DFlash convolution differs from independent CPU recurrence");
            }
            std::cout<<"DFLASH_CONV_PASS m="<<m<<" h="<<h<<" side="<<side<<" residual="<<with_residual<<" max_abs="<<maximum<<std::endl;
        }
        }
        {
            // ring attention in BF16: the drafter's q / K / V hold BF16 values in FP32 arrays (bf() rounding), so BF16 inputs
            // are lossless; the output (rounded to BF16 by the caller) may move by an ulp from the FP32 kernel's order
            const int heads=32,kvh=8,hd=128,nq=16,keys=2064;
            auto rnd=[&](const mx::Shape& shape,float s) {return mx::astype(mx::astype(mx::multiply(mx::random::normal(shape,mx::float32),mx::array(s)),mx::bfloat16),mx::float32);};
            auto q=rnd({1,heads,nq,hd},1.0f),k=rnd({1,kvh,keys,hd},1.0f),v=rnd({1,kvh,keys,hd},1.0f);
            const float scale=1.0f/std::sqrt(float(hd));
            auto a=mx::astype(mx::astype(mx::fast::scaled_dot_product_attention(q,k,v,scale),mx::bfloat16),mx::float32);
            auto b=mx::astype(mx::fast::scaled_dot_product_attention(mx::astype(q,mx::bfloat16),mx::astype(k,mx::bfloat16),mx::astype(v,mx::bfloat16),scale),mx::float32);
            const float diff=mx::max(mx::abs(mx::subtract(a,b))).item<float>(),peak=mx::max(mx::abs(a)).item<float>();
            if(!(diff<=peak*2e-2f))throw std::runtime_error("DFlash BF16 ring attention differs from FP32 beyond 2 % of the peak");
            std::cout<<"DFLASH_ATTN_BF16_PASS max_abs="<<diff<<" peak="<<peak<<std::endl;
        }
    } catch(const std::exception& e) {std::cerr<<"DFLASH_CONV_ERROR "<<e.what()<<'\n';return 1;}
}

#include "dflash.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace mx=mlx::core;
float bf(float x) {auto b=std::bit_cast<uint32_t>(x);return std::bit_cast<float>((b+0x7fffu+((b>>16)&1u))&0xffff0000u);}
int main() {
    try {
        constexpr int m=8,h=32,group=16,g=h/group;
        std::vector<float> input(m*h),dynamic(m*4*g),base(4*h);
        for(size_t j=0;j<input.size();++j)input[j]=bf(std::sin(float(j)*0.17f));
        for(size_t j=0;j<dynamic.size();++j)dynamic[j]=bf(std::cos(float(j)*0.07f));
        for(size_t j=0;j<base.size();++j)base[j]=std::sin(float(j)*0.01f);
        for(int side=0;side<2;++side) {
            auto got=tqllm::mlx_backend::dflash_conv(mx::array(input.data(),{m,h}),mx::array(dynamic.data(),{m,4*g}),mx::array(base.data(),{2,2,h}),group,side);
            mx::eval(got);double maximum=0;
            for(int t=0;t<m;++t)for(int c=0;c<h;++c) {
                float result=0;for(int tap=0;tap<2;++tap)if(t>=tap)
                    result+=(dynamic[t*4*g+(2*side+tap)*g+c/group]+base[(2*side+tap)*h+c])*input[(t-tap)*h+c];
                maximum=std::max(maximum,double(std::abs(got.data<float>()[t*h+c]-bf(result))));
            }
            if(maximum!=0)throw std::runtime_error("DFlash convolution differs from independent CPU recurrence");
            std::cout<<"DFLASH_CONV_PASS side="<<side<<" max_abs="<<maximum<<std::endl;
        }
    } catch(const std::exception& e) {std::cerr<<"DFLASH_CONV_ERROR "<<e.what()<<'\n';return 1;}
}

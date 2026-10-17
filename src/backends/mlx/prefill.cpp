#include "prefill.hpp"
#include <stdexcept>
#include <type_traits>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
mx::array dense(const Tq4Weights& w) {
    const int r=w.codes.shape(0),k=w.codes.shape(1)*2;
    static auto kernel=mx::fast::metal_kernel("tqllm_tq4_stage",{"codes","scales"},{"y"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K) return;
uint row=i/K,col=i%K,code=(codes[row*(K/2)+col/2]>>(4*(col%2)))&15u;
float s=float(scales[row*(K/128)+col/128]);
y[i]=bfloat16_t(s*(INTEGER_LEVELS?LVI[code]:LV[code]));
)METAL","constant float LVI[16]={6.0f,18.0f,31.0f,44.0f,58.0f,75.0f,96.0f,127.0f,-6.0f,-18.0f,-31.0f,-44.0f,-58.0f,-75.0f,-96.0f,-127.0f}; constant float LV[16]={0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};");
    return kernel({w.codes,w.scales},{{r,k}},{mx::bfloat16},{r*k,1,1},{256,1,1},
                  {{"R",r},{"K",k},{"INTEGER_LEVELS",w.integer_levels}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
mx::array dense(const Tq3Weights& w) {
    const int r=w.planes.shape(0),k=(w.interleaved?w.planes.shape(1):w.planes.shape(2))*32;
    static auto kernel=mx::fast::metal_kernel("tqllm_tq3_stage",{"planes","scales"},{"y"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K) return;
uint row=i/K,col=i%K,b=col/32,j=col%32;
uint a=INTERLEAVED?planes[(row*(K/32)+b)*4+j/16]:planes[(row*3+j/16)*(K/32)+b];
uint signs=INTERLEAVED?planes[(row*(K/32)+b)*4+2]:planes[(row*3+2)*(K/32)+b];
uint rank=(a>>(4*(j%8)+2*((j%16)/8)))&3u;
uint sign=(signs>>(4*(j%8)+j/8))&1u;
y[i]=bfloat16_t(float(scales[row*(K/128)+col/128])*LV[rank|(sign<<2)]);
)METAL","constant float LV[8]={14.0f,45.0f,79.0f,127.0f,-14.0f,-45.0f,-79.0f,-127.0f};");
    return kernel({w.planes,w.scales},{{r,k}},{mx::bfloat16},{r*k,1,1},{256,1,1},
                  {{"R",r},{"K",k},{"INTERLEAVED",w.interleaved}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
mx::array dense(const Q6Weights& w) {
    const int r=w.blocks.shape(0),k=w.blocks.shape(1)*256;
    static auto kernel=mx::fast::metal_kernel("tqllm_q6_stage",{"blocks"},{"y"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K) return;
uint row=i/K,col=i%K,b=col/256,j=col%256,h=j/128,c=(j%128)/32,l=j%32;
const device uchar* p=blocks+(row*(K/256)+b)*210;
float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8))));
uint lo=p[h*64+(c&1u)*32+l],hi=p[128+h*32+l];
int q=int((c<2?lo&15u:lo>>4)|(((hi>>(2*c))&3u)<<4))-32;
y[i]=bfloat16_t((float(q)*d)*float(as_type<char>(p[192+h*8+c*2+l/16])));
)METAL");
    return kernel({w.blocks},{{r,k}},{mx::bfloat16},{r*k,1,1},{256,1,1},
                  {{"R",r},{"K",k}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
mx::array dense(const Q8Weights& w) {
    const int r=w.blocks.shape(0),k=w.blocks.shape(1)*32;
    static auto kernel=mx::fast::metal_kernel("tqllm_q8_stage",{"blocks"},{"y"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K) return;
uint row=i/K,col=i%K;
const device uchar* p=blocks+(row*(K/32)+col/32)*34;
float d=float(as_type<half>(ushort(uint(p[0])|(uint(p[1])<<8))));
y[i]=bfloat16_t(float(as_type<char>(p[2+col%32]))*d);
)METAL");
    return kernel({w.blocks},{{r,k}},{mx::bfloat16},{r*k,1,1},{256,1,1},
                  {{"R",r},{"K",k}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
}
mx::array stage_bf16(const Matrix& weights) {
    return std::visit([](const auto& w)->mx::array {
        using T=std::decay_t<decltype(w)>;
        if constexpr(std::is_same_v<T,mx::array>)return mx::astype(w,mx::bfloat16);
        else return dense(w);
    },weights);
}
mx::array prefill_linear(const Matrix& weights,const mx::array& input) {
    if(input.ndim()!=2||input.dtype()!=mx::float32||input.shape(0)<1)
        throw std::invalid_argument("prefill requires a nonempty FP32 matrix");
    return std::visit([&](const auto& w)->mx::array {
        using T=std::decay_t<decltype(w)>;
        mx::array x=input;
        if constexpr(std::is_same_v<T,Tq3Weights>||std::is_same_v<T,Tq4Weights>)x=tq4_rotate(input);
        mx::array expanded=stage_bf16(weights);
        if(expanded.shape(1)!=input.shape(1))throw std::invalid_argument("incompatible prefill matrix dimensions");
        auto y=mx::astype(mx::matmul(mx::astype(x,mx::bfloat16),mx::transpose(expanded)),mx::float32);
        // Complete one projection before another can allocate its dense scratch.
        mx::eval(y);
        return y;
    },weights);
}
}

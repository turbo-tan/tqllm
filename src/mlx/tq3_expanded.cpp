#include "tq3.hpp"
#include "tuning.hpp"
#include <stdexcept>
#include <cstdlib>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
Tq3Expanded tq3_expand(const Tq3Weights& weights) {
    if(weights.interleaved || weights.planes.ndim()!=3 || weights.planes.shape(1)!=3)
        throw std::invalid_argument("TQ3 expansion requires planar weights");
    const int r=weights.planes.shape(0),k=weights.planes.shape(2)*32;
    static auto expand=mx::fast::metal_kernel("tqllm_tq3_expand",{"planes"},{"values"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K)return;
uint row=i/K,col=i%K,b=col/32u,j=col%32u;
uint rank=(planes[(row*3u+j/16u)*(K/32u)+b]>>(4u*(j%8u)+2u*((j%16u)/8u)))&3u;
uint sign=(planes[(row*3u+2u)*(K/32u)+b]>>(4u*(j%8u)+j/8u))&1u;
int level=int((0x7f4f2d0eu>>(rank*8u))&255u);
values[i]=char(sign?-level:level);
)METAL");
    auto values=expand({weights.planes},{{r,k}},{mx::int8},{r*k,1,1},{256,1,1},{{"R",r},{"K",k}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
    mx::eval(values);
    return {values,weights.scales};
}
mx::array tq3_expanded_matmul(const Tq3Expanded& weights,const mx::array& x) {
    if(weights.values.ndim()!=2 || weights.values.dtype()!=mx::int8 || x.ndim()!=2 || x.dtype()!=mx::float32)
        throw std::invalid_argument("invalid expanded TQ3 types");
    const int r=weights.values.shape(0),k=weights.values.shape(1),m=x.shape(0);
    if(k%128 || x.shape(1)!=k || weights.scales.shape()!=mx::Shape{r,k/128})
        throw std::invalid_argument("expanded TQ3 shape mismatch");
    static auto multiply=mx::fast::metal_kernel("tqllm_tq3_expanded",{"values","scales","x"},{"y"},R"METAL(
uint lane=thread_position_in_threadgroup.x;
uint row=threadgroup_position_in_grid.y*threads_per_threadgroup.y+thread_position_in_threadgroup.y;
uint col=threadgroup_position_in_grid.z*MCOLS;
if(row>=R)return;
float acc[MCOLS];
for(uint m=0;m<MCOLS;++m)acc[m]=0.0f;
for (uint b = lane; b < K/32; b += 32) {
 float s=float(scales[row*(K/128)+b/4]);
 float4 weights[8];
 const device char4* packed=(const device char4*)(values+row*K+b*32);
 for(uint j=0;j<8;++j)weights[j]=float4(packed[j])*s;
 for (uint m=0;m<MCOLS;++m) {
  if (col+m>=M) break;
  const device float4* input=(const device float4*)(x+(col+m)*K+b*32);
  float partial=0.0f;
  for (uint j=0;j<8;++j) partial+=dot(weights[j],input[j]);
  acc[m]+=partial;
 }
}
for (uint m = 0; m < MCOLS; ++m) {
 if (col+m >= M) break;
 float value = simd_sum(acc[m]);
 if (lane == 0) y[(col+m)*R+row] = value;
}
)METAL");
    const int cols=m<=1?1:m<=2?2:4,simds=matmul_simds();
    return multiply({weights.values,weights.scales,x},{{m,r}},{mx::float32},{32,(r+simds-1)/simds*simds,(m+cols-1)/cols},{32,simds,1},
                    {{"R",r},{"K",k},{"M",m},{"MCOLS",cols}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
Tq3Affine8 tq3_affine8(const Tq3Weights& weights) {
    auto expanded=tq3_expand(weights);
    auto bytes=mx::astype(mx::add(mx::astype(expanded.values,mx::int16),mx::array(128,mx::int16)),mx::uint8);
    auto codes=mx::view(mx::contiguous(bytes),mx::uint32);
    auto scales=mx::astype(weights.scales,mx::float32);
    auto biases=mx::multiply(scales,mx::array(-128.0f));
    auto half_scales=mx::astype(scales,mx::float16),half_biases=mx::astype(biases,mx::float16);
    mx::eval(codes,scales,biases,half_scales,half_biases);
    return {codes,scales,biases,half_scales,half_biases};
}
Tq3Affine8 tq3_affine8_join(const Tq3Affine8& gate,const Tq3Affine8& up) {
    if(gate.codes.ndim()!=2 || gate.codes.shape()!=up.codes.shape() ||
       gate.scales.shape()!=up.scales.shape() || gate.biases.shape()!=up.biases.shape() ||
       gate.codes.dtype()!=up.codes.dtype() || gate.scales.dtype()!=up.scales.dtype())
        throw std::invalid_argument("affine gate/up shape mismatch");
    Tq3Affine8 joined{mx::concatenate({gate.codes,up.codes},0),
        mx::concatenate({gate.scales,up.scales},0),mx::concatenate({gate.biases,up.biases},0),
        mx::concatenate({gate.half_scales,up.half_scales},0),mx::concatenate({gate.half_biases,up.half_biases},0)};
    mx::eval(joined.codes,joined.scales,joined.biases,joined.half_scales,joined.half_biases);
    return joined;
}
mx::array tq3_affine8_matmul(const Tq3Affine8& weights,const mx::array& x) {
    const bool half=std::getenv("TQLLM_MLX_AFFINE_HALF")!=nullptr && x.shape(0)<=8;
    auto input=half?mx::astype(x,mx::float16):x;
    const auto& scales=half?weights.half_scales:weights.scales;
    const auto& biases=half?weights.half_biases:weights.biases;
    if(std::getenv("TQLLM_MLX_AFFINE_PAD") && x.shape(0)<32) {
        auto padded=mx::concatenate({input,mx::zeros({32-x.shape(0),x.shape(1)},input.dtype())},0);
        auto result=mx::quantized_matmul(padded,weights.codes,scales,biases,true,128,8,"affine");
        return mx::astype(mx::slice(result,{0,0},{x.shape(0),weights.codes.shape(0)}),mx::float32);
    }
    return mx::astype(mx::quantized_matmul(input,weights.codes,scales,biases,true,128,8,"affine"),mx::float32);
}

}

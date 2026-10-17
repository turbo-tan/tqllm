#include "verify.hpp"
#include <stdexcept>
#include <type_traits>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
mx::array tiled(const mx::array& packed,const mx::array& scales,const mx::array& x,int rows,int k,int type,bool integer=true,bool interleaved=false) {
    const int m=x.shape(0);
    static auto kernel=mx::fast::metal_kernel("tqllm_verify_tiled",{"packed","scales","x"},{"y"},R"METAL(
uint tid=thread_position_in_threadgroup.x;
uint rbase=threadgroup_position_in_grid.y*8u,mBase=threadgroup_position_in_grid.z*8u;
threadgroup float A[256],B[256],C[64];
simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.0f);
const device uchar* bytes=(const device uchar*)packed;
const device uint* words=(const device uint*)packed;
for(uint base=0;base<K;base+=32u) {
 for(uint i=tid;i<256u;i+=32u) {
  uint ar=i/32u,ac=i%32u;
  A[i]=mBase+ar<M?x[(mBase+ar)*K+base+ac]:0.0f;
  uint col=base+i/8u,row=rbase+i%8u;
  float value=0.0f;
  if(row<R) {
   if(TYPE==300) {
    uint code=(bytes[row*(K/2u)+col/2u]>>(4u*(col%2u)))&15u;
    value=float(scales[row*(K/128u)+col/128u])*(INTEGER_LEVELS?L4I[code]:L4[code]);
   } else if(TYPE==301) {
    uint block=col/32u,j=col%32u;
    uint a=INTERLEAVED?words[(row*(K/32u)+block)*4u+j/16u]:words[(row*3u+j/16u)*(K/32u)+block];
    uint signs=INTERLEAVED?words[(row*(K/32u)+block)*4u+2u]:words[(row*3u+2u)*(K/32u)+block];
    uint rank=(a>>(4u*(j%8u)+2u*((j%16u)/8u)))&3u;
    uint sign=(signs>>(4u*(j%8u)+j/8u))&1u;
    value=float(scales[row*(K/128u)+col/128u])*L3[rank|(sign<<2u)];
   } else {
    const device uchar* p=bytes+(row*(K/256u)+col/256u)*210u;
    uint j=col%256u,h=j/128u,c=(j%128u)/32u,l=j%32u;
    uint lo=p[h*64u+(c&1u)*32u+l],hi=p[128u+h*32u+l];
    int q=int((c<2u?lo&15u:lo>>4u)|(((hi>>(2u*c))&3u)<<4u))-32;
    float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8u))));
    value=float(q)*(d*float(as_type<char>(p[192u+h*8u+c*2u+l/16u])));
   }
  }
  B[i]=value;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint offset=0;offset<32u;offset+=8u) {
  simdgroup_float8x8 a,b;
  simdgroup_load(a,A+offset,32u);
  simdgroup_load(b,B+offset*8u,8u);
  simdgroup_multiply_accumulate(acc,a,b,acc);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(acc,C,8u);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<64u;i+=32u) {
 uint row=rbase+i%8u,m=mBase+i/8u;
 if(row<R && m<M)y[m*R+row]=C[i];
}
)METAL",R"METAL(
#include <metal_simdgroup_matrix>
constant float L3[8]={14.0f,45.0f,79.0f,127.0f,-14.0f,-45.0f,-79.0f,-127.0f};
constant float L4I[16]={6.0f,18.0f,31.0f,44.0f,58.0f,75.0f,96.0f,127.0f,-6.0f,-18.0f,-31.0f,-44.0f,-58.0f,-75.0f,-96.0f,-127.0f};
constant float L4[16]={0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};
)METAL");
    return kernel({packed,scales,x},{{m,rows}},{mx::float32},{32,(rows+7)/8,(m+7)/8},{32,1,1},
                  {{"R",rows},{"K",k},{"M",m},{"TYPE",type},{"INTEGER_LEVELS",integer},{"INTERLEAVED",interleaved}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
}
mx::array verify_tiled(const Matrix& weights,const mx::array& input) {
    if(input.ndim()!=2||input.dtype()!=mx::float32||input.shape(0)<=0)throw std::invalid_argument("verify tile requires FP32 matrix input");
    return std::visit([&](const auto& w)->mx::array {
        using T=std::decay_t<decltype(w)>;
        if constexpr(std::is_same_v<T,Tq3Weights>) {
            if((w.interleaved?w.planes.shape(1):w.planes.shape(2))*32!=input.shape(1))throw std::invalid_argument("verify TQ3 shape mismatch");
            return tiled(w.planes,w.scales,tq4_rotate(input),w.planes.shape(0),input.shape(1),301,true,w.interleaved);
        } else if constexpr(std::is_same_v<T,Tq4Weights>) {
            if(w.codes.shape(1)*2!=input.shape(1))throw std::invalid_argument("verify TQ4 shape mismatch");
            return tiled(w.codes,w.scales,tq4_rotate(input),w.codes.shape(0),input.shape(1),300,w.integer_levels);
        } else if constexpr(std::is_same_v<T,Q6Weights>) {
            if(w.blocks.shape(1)*256!=input.shape(1))throw std::invalid_argument("verify Q6 shape mismatch");
            return tiled(w.blocks,mx::zeros({1},mx::float16),input,w.blocks.shape(0),input.shape(1),14);
        } else return linear(weights,input);
    },weights);
}
}

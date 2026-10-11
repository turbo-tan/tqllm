#include "tensorops.hpp"
#include <stdexcept>
#include <type_traits>
#include <cstdlib>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
mx::array tiled(const mx::array& packed,const mx::array& scales,const mx::array& x,int rows,int k,int type,bool integer=true,bool interleaved=false,bool fp16=false) {
    const int m=x.shape(0);
    auto option=[](const char* name,int fallback) {
        const char* value=std::getenv(name);return value?std::stoi(value):fallback;
    };
    const bool memory=std::getenv("TQLLM_MLX_TENSOR_MEMORY")!=nullptr;
    const int nt=memory?32:option("TQLLM_MLX_TENSOR_N",32),bk=memory?32:option("TQLLM_MLX_TENSOR_K",16);
    const int split=option("TQLLM_MLX_TENSOR_SPLIT",1);
    const bool transpose=option("TQLLM_MLX_TENSOR_TRANSPOSE",1)!=0;
    if((nt!=16 && nt!=32) || (bk!=16 && bk!=32) || (nt!=32 && bk!=32))
        throw std::invalid_argument("invalid TensorOps tile");
    if((split!=1 && split!=2 && split!=4 && split!=8 && split!=16 && split!=32) || k%(split*bk))
        throw std::invalid_argument("invalid TensorOps split reduction");
    static auto kernel=mx::fast::metal_kernel("tqllm_verify_tensorops",{"packed","scales","x"},{"y"},R"METAL(
uint tid=thread_position_in_threadgroup.x;
uint rbase=threadgroup_position_in_grid.y*uint(NT),mBase=(threadgroup_position_in_grid.z/uint(SPLIT))*16u;
uint part=threadgroup_position_in_grid.z%uint(SPLIT);
threadgroup float C[16*NT];
constexpr auto descriptor=mpp::tensor_ops::matmul2d_descriptor(16,NT,BK,false,TRANSPOSE,false,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<descriptor,metal::execution_simdgroups<1>> op;
tensor<threadgroup float,extents<int,NT,16>,tensor_inline> tc(C,extents<int,NT,16>());
auto ta=op.template get_left_input_cooperative_tensor<Operand,Operand,float>();
auto tb=op.template get_right_input_cooperative_tensor<Operand,Operand,float>();
auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
const device uchar* bytes=(const device uchar*)packed;
const device uint* words=(const device uint*)packed;
for(uint base=part*(K/SPLIT);base<(part+1u)*(K/SPLIT);base+=uint(BK)) {
 for(uint i=0;i<ta.get_capacity();++i) {
  if(!ta.is_valid_element(i))continue;
  auto coord=ta.get_multidimensional_index(i);
  uint ar=uint(coord[1]),ac=uint(coord[0]);
  ta[i]=mBase+ar<M?x[(mBase+ar)*K+base+ac]:0.0f;
 }
 for(uint i=0;i<tb.get_capacity();++i) {
  if(!tb.is_valid_element(i))continue;
  auto coord=tb.get_multidimensional_index(i);
  uint col=base+uint(coord[TRANSPOSE?0:1]),row=rbase+uint(coord[TRANSPOSE?1:0]);
  float value=0.0f;
  if(row<R) {
   if(TYPE==302) {
    value=float(as_type<char>(bytes[row*K+col]))*float(scales[row*(K/128u)+col/128u]);
   } else if(TYPE==300) {
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
  tb[i]=value;
 }
 op.run(ta,tb,acc);
}
acc.store(tc);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<16u*uint(NT);i+=32u) {
 uint row=rbase+i%uint(NT),m=mBase+i/uint(NT);
 if(row<R && m<M)y[(part*M+m)*R+row]=C[i];
}
)METAL",R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
constant float L3[8]={14.0f,45.0f,79.0f,127.0f,-14.0f,-45.0f,-79.0f,-127.0f};
constant float L4I[16]={6.0f,18.0f,31.0f,44.0f,58.0f,75.0f,96.0f,127.0f,-6.0f,-18.0f,-31.0f,-44.0f,-58.0f,-75.0f,-96.0f,-127.0f};
constant float L4[16]={0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};
)METAL");
    auto partials=kernel({packed,scales,x},{{split,m,rows}},{mx::float32},{32,(rows+nt-1)/nt,(m+15)/16*split},{32,1,1},
                  {{"SPLIT",split},{"NT",nt},{"BK",bk},{"TRANSPOSE",transpose},{"Operand",fp16?mx::float16:mx::float32},{"R",rows},{"K",k},{"M",m},{"TYPE",type},{"INTEGER_LEVELS",integer},{"INTERLEAVED",interleaved}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
    return split==1?mx::reshape(partials,{m,rows}):mx::sum(partials,0);
}

mx::array packed_tq3_memory(const Tq3Weights& w,const mx::array& x,bool fp16) {
    auto option=[](const char* name,int fallback) {
        const char* value=std::getenv(name);return value?std::stoi(value):fallback;
    };
    const int m=x.shape(0),k=x.shape(1),rows=w.planes.shape(0);
    const int nt=option("TQLLM_MLX_TENSOR_N",64),bk=option("TQLLM_MLX_TENSOR_K",128);
    const int simds=option("TQLLM_MLX_TENSOR_SIMDS",4),split=option("TQLLM_MLX_TENSOR_SPLIT",1);
    if((nt!=32 && nt!=64) || (bk!=32 && bk!=64 && bk!=128) ||
       (simds!=1 && simds!=2 && simds!=4) ||
       (split!=1 && split!=2 && split!=4 && split!=8) || k%(split*bk))
        throw std::invalid_argument("invalid packed TQ3 memory tile");
    const int scratch=(16*bk+nt*bk)*(fp16?2:4)+16*nt*4;
    if(scratch>32768)throw std::invalid_argument("packed TQ3 tile exceeds threadgroup memory budget");
    static auto kernel=mx::fast::metal_kernel("tqllm_tq3_tensor_memory",{"packed","scales","x"},{"y"},R"METAL(
uint tid=thread_position_in_threadgroup.x;
uint row0=threadgroup_position_in_grid.y*uint(NT);
uint col0=(threadgroup_position_in_grid.z/uint(SPLIT))*16u;
uint part=threadgroup_position_in_grid.z%uint(SPLIT);
threadgroup Operand A[16*BK],B[NT*BK];
threadgroup float C[16*NT];
tensor<threadgroup Operand,extents<int,BK,16>,tensor_inline> ta(A,extents<int,BK,16>());
tensor<threadgroup Operand,extents<int,BK,NT>,tensor_inline> tb(B,extents<int,BK,NT>());
tensor<threadgroup float,extents<int,NT,16>,tensor_inline> tc(C,extents<int,NT,16>());
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(16,NT,BK,false,true,false,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroups<SIMDS>> op;
auto acc=op.template get_destination_cooperative_tensor<decltype(ta),decltype(tb),float>();
for(uint i=0;i<acc.get_capacity();++i)if(acc.is_valid_element(i))acc[i]=0.0f;
for(uint base=part*(K/SPLIT);base<(part+1u)*(K/SPLIT);base+=uint(BK)) {
 for(uint i=tid;i<16u*uint(BK);i+=32u*uint(SIMDS)) {
  uint col=base+i%uint(BK),m=col0+i/uint(BK);
  A[i]=Operand(m<M?x[m*K+col]:0.0f);
 }
 if(WIDE_UNPACK) {
 for(uint i=tid;i<uint(NT)*uint(BK)/4u;i+=32u*uint(SIMDS)) {
  uint row=row0+i/(uint(BK)/4u),slot=i%(uint(BK)/4u);
  uint block=base/32u+slot/8u,j=slot%8u;
  uint dest=(row-row0)*uint(BK)+(slot/8u)*32u+j;
  float scale=0.0f;uint a0=0u,a1=0u,signs=0u;
  if(row<R) {
   a0=INTERLEAVED?packed[(row*(K/32u)+block)*4u]:packed[(row*3u)*(K/32u)+block];
   a1=INTERLEAVED?packed[(row*(K/32u)+block)*4u+1u]:packed[(row*3u+1u)*(K/32u)+block];
   signs=INTERLEAVED?packed[(row*(K/32u)+block)*4u+2u]:packed[(row*3u+2u)*(K/32u)+block];
   scale=float(scales[row*(K/128u)+block/4u]);
  }
  a0>>=4u*j;a1>>=4u*j;signs>>=4u*j;
  B[dest]=Operand(scale*L3[(a0&3u)|((signs&1u)<<2u)]);
  B[dest+8u]=Operand(scale*L3[((a0>>2u)&3u)|(((signs>>1u)&1u)<<2u)]);
  B[dest+16u]=Operand(scale*L3[(a1&3u)|(((signs>>2u)&1u)<<2u)]);
  B[dest+24u]=Operand(scale*L3[((a1>>2u)&3u)|(((signs>>3u)&1u)<<2u)]);
 }
 } else {
 for(uint i=tid;i<uint(NT)*uint(BK);i+=32u*uint(SIMDS)) {
  uint row=row0+i/uint(BK),col=base+i%uint(BK);
  float value=0.0f;
  if(row<R) {
   uint block=col/32u,j=col%32u;
   uint a=INTERLEAVED?packed[(row*(K/32u)+block)*4u+j/16u]:packed[(row*3u+j/16u)*(K/32u)+block];
   uint signs=INTERLEAVED?packed[(row*(K/32u)+block)*4u+2u]:packed[(row*3u+2u)*(K/32u)+block];
   uint rank=(a>>(4u*(j%8u)+2u*((j%16u)/8u)))&3u;
   uint sign=(signs>>(4u*(j%8u)+j/8u))&1u;
   value=float(scales[row*(K/128u)+col/128u])*L3[rank|(sign<<2u)];
  }
  B[i]=Operand(value);
 }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 op.run(ta,tb,acc);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
acc.store(tc);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<16u*uint(NT);i+=32u*uint(SIMDS)) {
 uint row=row0+i%uint(NT),m=col0+i/uint(NT);
 if(row<R && m<M)y[(part*M+m)*R+row]=C[i];
}
)METAL",R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
constant float L3[8]={14.0f,45.0f,79.0f,127.0f,-14.0f,-45.0f,-79.0f,-127.0f};
)METAL");
    auto result=kernel({w.planes,w.scales,tq4_rotate(x)},{{split,m,rows}},{mx::float32},
        {32*simds,(rows+nt-1)/nt,(m+15)/16*split},{32*simds,1,1},
        {{"NT",nt},{"BK",bk},{"SIMDS",simds},{"SPLIT",split},{"WIDE_UNPACK",option("TQLLM_MLX_TENSOR_WIDE",0)!=0},{"Operand",fp16?mx::float16:mx::float32},
         {"R",rows},{"K",k},{"M",m},{"INTERLEAVED",w.interleaved}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
    return split==1?mx::reshape(result,{m,rows}):mx::sum(result,0);
}
}
mx::array verify_expanded_tensorops(const Tq3Expanded& weights,const mx::array& input,bool fp16) {
    if(weights.values.ndim()!=2 || weights.values.dtype()!=mx::int8 || input.ndim()!=2 ||
       input.dtype()!=mx::float32 || input.shape(1)!=weights.values.shape(1))
        throw std::invalid_argument("invalid expanded TensorOps input");
    return tiled(weights.values,weights.scales,tq4_rotate(input),weights.values.shape(0),input.shape(1),302,true,false,fp16);
}
mx::array verify_tensorops(const Matrix& weights,const mx::array& input,bool fp16) {
    if(input.ndim()!=2||input.dtype()!=mx::float32||input.shape(0)<=0)throw std::invalid_argument("verify tile requires FP32 matrix input");
    return std::visit([&](const auto& w)->mx::array {
        using T=std::decay_t<decltype(w)>;
        if constexpr(std::is_same_v<T,Tq3Weights>) {
            if((w.interleaved?w.planes.shape(1):w.planes.shape(2))*32!=input.shape(1))throw std::invalid_argument("verify TQ3 shape mismatch");
            if(std::getenv("TQLLM_MLX_TENSOR_MEMORY"))return packed_tq3_memory(w,input,fp16);
            return tiled(w.planes,w.scales,tq4_rotate(input),w.planes.shape(0),input.shape(1),301,true,w.interleaved,fp16);
        } else if constexpr(std::is_same_v<T,Tq4Weights>) {
            if(w.codes.shape(1)*2!=input.shape(1))throw std::invalid_argument("verify TQ4 shape mismatch");
            return tiled(w.codes,w.scales,tq4_rotate(input),w.codes.shape(0),input.shape(1),300,w.integer_levels,false,fp16);
        } else if constexpr(std::is_same_v<T,Q6Weights>) {
            if(w.blocks.shape(1)*256!=input.shape(1))throw std::invalid_argument("verify Q6 shape mismatch");
            return tiled(w.blocks,mx::zeros({1},mx::float16),input,w.blocks.shape(0),input.shape(1),14,true,false,fp16);
        } else return linear(weights,input);
    },weights);
}
}

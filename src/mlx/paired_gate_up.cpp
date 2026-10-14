#include "tq3_tiled.hpp"
#include "tq4.hpp"
#include <limits>
#include <mlx/compile.h>
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
std::pair<mx::array,mx::array> tq_paired_matmul(const TqTiledWeights& gate,
        const TqTiledWeights& up,const mx::array& input,bool fp32,int splits) {
    for(const auto* w:{&gate,&up}) {
        if(w->rows<=0 || w->rows>std::numeric_limits<int>::max()-31 || w->columns<=0 ||
           (w->bits!=3 && w->bits!=4) || (w->bits==3 && !w->integer_levels) ||
           w->columns%128 || w->planes.dtype()!=mx::uint32 || w->scales.dtype()!=mx::float16 ||
           w->planes.shape()!=mx::Shape{(w->rows+31)/32,w->columns/32,w->bits,32} ||
           w->scales.shape()!=mx::Shape{(w->rows+31)/32,w->columns/128,32})
            throw std::invalid_argument("invalid paired tiled weights");
    }
    if(gate.rows!=up.rows || gate.columns!=up.columns || gate.bits!=up.bits ||
       gate.integer_levels!=up.integer_levels || input.ndim()!=2 || input.dtype()!=mx::float32 ||
       input.shape(0)<1 || input.shape(0)>8 || input.shape(1)!=gate.columns ||
       (splits!=1 && splits!=2 && splits!=4) || gate.columns%(128*splits))
        throw std::invalid_argument("incompatible paired tiled projection");
    const int m=input.shape(0);
    auto x=m<8?mx::concatenate({input,mx::zeros({8-m,gate.columns},mx::float32)},0):input;
    x=mx::contiguous(fp32?x:mx::astype(x,mx::float16));
    static auto kernel=mx::fast::metal_kernel("tqllm_paired_gate_up",
        {"gplanes","gscales","uplanes","uscales","x"},{"gout","uout"},R"METAL(
uint lane=thread_position_in_threadgroup.x;
uint tile=threadgroup_position_in_grid.y,part=threadgroup_position_in_grid.z;
threadgroup Operand storage[2*32*32];
tensor<device Operand,dextents<int,2>,tensor_inline> inputs((device Operand*)x,dextents<int,2>{int(K),8},array<int,2>{1,int(K)});
tensor<threadgroup Operand,extents<int,32,32>,tensor_inline> gs(storage,extents<int,32,32>());
tensor<threadgroup Operand,extents<int,32,32>,tensor_inline> us(storage+32*32,extents<int,32,32>());
constexpr auto desc=mpp::tensor_ops::matmul2d_descriptor(8,32,32,false,false,false,mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
mpp::tensor_ops::matmul2d<desc,metal::execution_simdgroups<1>> op;
auto initial=inputs.template slice<32,8>(0,0);
auto ga=op.template get_destination_cooperative_tensor<decltype(initial),decltype(gs),float>();
auto ua=op.template get_destination_cooperative_tensor<decltype(initial),decltype(us),float>();
for(uint i=0;i<ga.get_capacity();++i)if(ga.is_valid_element(i)) {ga[i]=0.0f;ua[i]=0.0f;}
uint begin=part*(K/32u/SPLITS),end=(part+1u)*(K/32u/SPLITS);
ulong origin=ulong(tile)*(K/32u)*(32u*BITS)+lane;
ulong meta=ulong(tile)*(K/128u)*32u+lane;
uint gl=gplanes[origin+ulong(begin)*32u*BITS],gh=gplanes[origin+ulong(begin)*32u*BITS+32u];
uint gn=gplanes[origin+ulong(begin)*32u*BITS+64u],gf=0u;
uint ul=uplanes[origin+ulong(begin)*32u*BITS],uh=uplanes[origin+ulong(begin)*32u*BITS+32u];
uint un=uplanes[origin+ulong(begin)*32u*BITS+64u],uf=0u;
if(BITS==4) {gf=gplanes[origin+ulong(begin)*32u*BITS+96u];uf=uplanes[origin+ulong(begin)*32u*BITS+96u];}
float gd=float(gscales[meta+ulong(begin/4u)*32u]),ud=float(uscales[meta+ulong(begin/4u)*32u]);
for(uint b=begin;b<end;++b) {
    #pragma clang loop unroll(full)
    for(uint j=0;j<32u;++j) {
        float gv,uv;
        if(BITS==3) {
            uint gcode=((j<16u?gl:gh)>>(4u*(j%8u)+2u*((j%16u)/8u)))&3u;
            uint ucode=((j<16u?ul:uh)>>(4u*(j%8u)+2u*((j%16u)/8u)))&3u;
            int glevel=int((0x7f4f2d0eu>>(8u*gcode))&255u),ulevel=int((0x7f4f2d0eu>>(8u*ucode))&255u);
            gv=float(((gn>>(4u*(j%8u)+j/8u))&1u)?-glevel:glevel);
            uv=float(((un>>(4u*(j%8u)+j/8u))&1u)?-ulevel:ulevel);
        } else {
            uint gc=((j<8u?gl:j<16u?gh:j<24u?gn:gf)>>(4u*(j%8u)))&15u;
            uint uc=((j<8u?ul:j<16u?uh:j<24u?un:uf)>>(4u*(j%8u)))&15u;
            gv=INTEGER_LEVELS?L4I[gc]:L4F[gc];uv=INTEGER_LEVELS?L4I[uc]:L4F[uc];
        }
        storage[j*32u+lane]=Operand(gd*gv);storage[32u*32u+j*32u+lane]=Operand(ud*uv);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if(b+1u<end) {
        ulong at=origin+ulong(b+1u)*32u*BITS;
        gl=gplanes[at];gh=gplanes[at+32u];gn=gplanes[at+64u];
        ul=uplanes[at];uh=uplanes[at+32u];un=uplanes[at+64u];
        if(BITS==4){gf=gplanes[at+96u];uf=uplanes[at+96u];}
        if((b+1u)%4u==0u){gd=float(gscales[meta+ulong((b+1u)/4u)*32u]);ud=float(uscales[meta+ulong((b+1u)/4u)*32u]);}
    }
    auto activation=inputs.template slice<32,8>(b*32u,0);
    op.run(activation,gs,ga);op.run(activation,us,ua);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint i=0;i<ga.get_capacity();++i) {
    if(!ga.is_valid_element(i))continue;
    auto pos=ga.get_multidimensional_index(i);uint row=tile*32u+uint(pos[0]),col=uint(pos[1]);
    if(row<R && col<M) {ulong at=(ulong(part)*M+col)*R+row;gout[at]=ga[i];uout[at]=ua[i];}
}
)METAL",R"METAL(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#pragma clang fp reassociate(off)
constant float L4I[16]={6.0f,18.0f,31.0f,44.0f,58.0f,75.0f,96.0f,127.0f,-6.0f,-18.0f,-31.0f,-44.0f,-58.0f,-75.0f,-96.0f,-127.0f};
constant float L4F[16]={0.129099213f,0.387297638f,0.667012598f,0.946727559f,1.247959055f,1.613740157f,2.065587402f,2.732600000f,-0.129099213f,-0.387297638f,-0.667012598f,-0.946727559f,-1.247959055f,-1.613740157f,-2.065587402f,-2.732600000f};
)METAL");
    auto outputs=kernel({gate.planes,gate.scales,up.planes,up.scales,x},
        {{splits,m,gate.rows},{splits,m,gate.rows}},{mx::float32,mx::float32},
        {32,(gate.rows+31)/32,splits},{32,1,1},{{"R",gate.rows},{"K",gate.columns},{"M",m},
        {"BITS",gate.bits},{"INTEGER_LEVELS",gate.integer_levels},{"SPLITS",splits},
        {"Operand",fp32?mx::float32:mx::float16}},std::nullopt,false,mx::Device(mx::Device::gpu));
    auto reduce=[&](const mx::array& a){return splits==1?mx::reshape(a,{m,gate.rows}):mx::sum(a,0);};
    return {reduce(outputs[0]),reduce(outputs[1])};
}
mx::array tq_paired_swiglu(const TqTiledWeights& gate,const TqTiledWeights& up,
        const mx::array& input,bool fp32,int splits) {
    auto projections=tq_paired_matmul(gate,up,input,fp32,splits);
    static auto epilogue=mx::compile([](const std::vector<mx::array>& a) {
        auto bf=[](const mx::array& x){return mx::astype(mx::astype(x,mx::bfloat16),mx::float32);};
        auto g=bf(a[0]),u=bf(a[1]);
        return std::vector<mx::array>{bf(mx::multiply(bf(mx::multiply(g,mx::sigmoid(g))),u))};
    });
    return epilogue({projections.first,projections.second})[0];
}
mx::array tq_paired_swiglu_rotated(const TqTiledWeights& gate,const TqTiledWeights& up,
        const mx::array& input,bool fp32,int splits) {
    auto projections=tq_paired_matmul(gate,up,input,fp32,splits);
    static auto epilogue=mx::compile([](const std::vector<mx::array>& a) {
        auto bf=[](const mx::array& x){return mx::astype(mx::astype(x,mx::bfloat16),mx::float32);};
        auto g=bf(a[0]),u=bf(a[1]);
        return std::vector<mx::array>{tq4_rotate(bf(mx::multiply(bf(mx::multiply(g,mx::sigmoid(g))),u)))};
    });
    return epilogue({projections.first,projections.second})[0];
}
}

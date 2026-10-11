#include "gdn.hpp"
#include <mlx/compile.h>
#include <stdexcept>
#include <cstdlib>
#include <string_view>

namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
mx::array bf(const mx::array& x) {return mx::astype(mx::astype(x,mx::bfloat16),mx::float32);}
std::vector<mx::array> convolve(const std::vector<mx::array>& args) {
    const auto& joined=args[0];const auto& weights=args[1];
    const int m=joined.shape(0)-3,width=joined.shape(1);
    auto result=mx::zeros({m,width});
    for(int tap=0;tap<4;++tap) {
        auto w=mx::reshape(mx::slice(weights,{0,tap},{width,tap+1}),{1,width});
        result=mx::add(result,mx::multiply(mx::slice(joined,{tap,0},{tap+m,width}),w));
    }
    result=bf(result);
    return {bf(mx::multiply(result,mx::sigmoid(result)))};
}
std::vector<mx::array> normalize(const std::vector<mx::array>& args) {
    auto inverse=mx::rsqrt(mx::add(mx::sum(mx::square(args[0]),-1,true),mx::array(1e-6f)));
    return {mx::multiply(args[0],mx::multiply(inverse,args[1]))};
}
std::vector<mx::array> decay(const std::vector<mx::array>& args) {
    auto dt=mx::add(args[0],args[1]);
    auto sp=mx::add(mx::maximum(dt,mx::array(0.0f)),mx::log1p(mx::exp(mx::negative(mx::abs(dt)))));
    return {mx::exp(mx::multiply(sp,args[2]))};
}
}
mx::array gdn_convolve(const mx::array& joined,const mx::array& weights,bool compiled) {
    if(joined.ndim()!=2 || joined.shape(0)<4 || weights.shape()!=mx::Shape{joined.shape(1),4} ||
       joined.dtype()!=mx::float32 || weights.dtype()!=mx::float32)
        throw std::invalid_argument("invalid GDN convolution dimensions");
    static auto fused=mx::compile(convolve);
    return compiled?fused({joined,weights})[0]:convolve({joined,weights})[0];
}
mx::array gdn_normalize(const mx::array& input,float scale,bool compiled) {
    if(input.ndim()!=4 || input.shape(3)<32 || input.shape(3)%32 || input.dtype()!=mx::float32)
        throw std::invalid_argument("invalid GDN normalization dimensions");
    static auto fused=mx::compile(normalize);
    auto factor=mx::array(scale);
    return compiled?fused({input,factor})[0]:normalize({input,factor})[0];
}
mx::array gdn_decay(const mx::array& alpha,const mx::array& bias,const mx::array& rate,bool compiled) {
    if(alpha.ndim()!=2 || bias.shape()!=mx::Shape{alpha.shape(1)} || rate.shape()!=bias.shape() ||
       alpha.dtype()!=mx::float32 || bias.dtype()!=mx::float32 || rate.dtype()!=mx::float32)
        throw std::invalid_argument("invalid GDN decay dimensions");
    static auto fused=mx::compile(decay);
    return compiled?fused({alpha,bias,rate})[0]:decay({alpha,bias,rate})[0];
}
GdnScan gdn_scan(const mx::array& q,const mx::array& k,const mx::array& v,
    const mx::array& decay,const mx::array& beta,const std::optional<mx::array>& state,bool snapshots) {
    if(q.ndim()!=4 || q.shape(0)!=1 || q.shape(3)!=128 || q.dtype()!=mx::float32 || k.shape()!=q.shape() ||
       k.dtype()!=mx::float32 || v.ndim()!=4 || v.shape(0)!=1 || v.shape(1)!=q.shape(1) ||
       v.shape(3)!=128 || v.dtype()!=mx::float32 || q.shape(2)<1 || v.shape(2)%q.shape(2) ||
       decay.shape()!=mx::Shape{1,v.shape(1),v.shape(2)} || beta.shape()!=decay.shape() ||
       decay.dtype()!=mx::float32 || beta.dtype()!=mx::float32)
        throw std::invalid_argument("unsupported GDN scan dimensions");
    const int m=q.shape(1),kh=q.shape(2),vh=v.shape(2);
    auto initial=state?*state:mx::zeros({1,vh,128,128});
    if(initial.shape()!=mx::Shape{1,vh,128,128} || initial.dtype()!=mx::float32)
        throw std::invalid_argument("invalid GDN scan state");
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_scan",{"q","k","v","decay","beta","initial"},{"out","states"},R"METAL(
uint lane=thread_position_in_threadgroup.x;
uint row=thread_position_in_grid.y,head=thread_position_in_grid.z;
uint base=(head*128u+row)*128u+lane*4u;
float4 s=*((const device float4*)(initial+base));
for(uint t=0;t<M;++t) {
 uint at=(t*KH+head%KH)*128u+lane*4u;
 float4 key=*((const device float4*)(k+at));
 float4 query=*((const device float4*)(q+at));
 s*=decay[t*VH+head];
 float memory=0.0f;
 for(uint j=0;j<4;++j)memory+=s[j]*key[j];
 memory=simd_sum(memory);
 float delta=(v[(t*VH+head)*128u+row]-memory)*beta[t*VH+head];
 float value=0.0f;
 for(uint j=0;j<4;++j) {s[j]=s[j]+key[j]*delta;value+=s[j]*query[j];}
 value=simd_sum(value);
 if(lane==0)out[(t*VH+head)*128u+row]=value;
 if(SNAPSHOTS || t+1==M) {
  uint offset=(SNAPSHOTS?t:0u)*VH*128u*128u+base;
  *((device float4*)(states+offset))=s;
 }
}
)METAL");
    auto result=kernel({mx::contiguous(q),mx::contiguous(k),mx::contiguous(v),mx::contiguous(decay),
        mx::contiguous(beta),mx::contiguous(initial)},{{1,m,vh,128},{snapshots?m:1,vh,128,128}},
        {mx::float32,mx::float32},{32,128,vh},{32,4,1},{{"M",m},{"KH",kh},{"VH",vh},{"SNAPSHOTS",snapshots}},
        std::nullopt,false,mx::Device(mx::Device::gpu));
    auto final=snapshots?mx::slice(result[1],{m-1,0,0,0},{m,vh,128,128}):result[1];
    return {result[0],final,snapshots?std::optional<mx::array>(result[1]):std::nullopt};
}
GdnHeadOrder gdn_head_order(int kh,int vh) {
    if(!((kh==16 && (vh==16 || vh==32 || vh==48 || vh==64)) ||
         (kh==24 && vh==24) || (kh==32 && vh==32)))
        throw std::invalid_argument("unsupported native MLX GDN head layout");
    const int repeats=vh/kh;
    std::vector<uint32_t> to(vh),from(vh);
    for(int i=0;i<vh;++i) {
        to[i]=uint32_t((i%repeats)*kh+i/repeats);
        from[i]=uint32_t((i%kh)*repeats+i/kh);
    }
    return {mx::array(to.data(),{vh},mx::uint32),mx::array(from.data(),{vh},mx::uint32),kh,vh};
}
std::vector<mx::array> gdn_native(const mx::array& q,const mx::array& k,const mx::array& v,
    const mx::array& decay,const mx::array& beta,const std::optional<mx::array>& state,const GdnHeadOrder& order) {
    const char* chunk=std::getenv("GATED_DELTA_CHUNK");
    if(!chunk || std::string_view(chunk)!="1")
        throw std::invalid_argument("native GDN requires GATED_DELTA_CHUNK=1; chunked precision is unvalidated");
    if(q.ndim()!=4 || k.shape()!=q.shape() || q.shape(3)!=128 || v.ndim()!=4 || v.shape(3)!=128 ||
       q.shape(0)!=v.shape(0) || q.shape(1)!=v.shape(1) || q.shape(2)!=order.key_heads ||
       v.shape(2)!=order.value_heads || q.dtype()!=mx::float32 || k.dtype()!=mx::float32 || v.dtype()!=mx::float32 ||
       decay.shape()!=mx::Shape{v.shape(0),v.shape(1),v.shape(2)} || beta.shape()!=decay.shape() ||
       decay.dtype()!=mx::float32 || beta.dtype()!=mx::float32 || order.to_native.size()!=size_t(v.shape(2)) ||
       (state && (state->shape()!=mx::Shape{v.shape(0),v.shape(2),128,128} || state->dtype()!=mx::float32)))
        throw std::invalid_argument("unsupported native MLX GDN dimensions");
    // Native kernels require dense storage after the head permutation.
    auto update=mx::fast::gated_delta_update(mx::contiguous(q),mx::contiguous(k),
        mx::contiguous(mx::take(v,order.to_native,2)),mx::contiguous(mx::take(decay,order.to_native,2)),
        mx::contiguous(mx::take(beta,order.to_native,2)),state);
    return {mx::contiguous(mx::take(update[0],order.from_native,2)),update[1]};
}
}

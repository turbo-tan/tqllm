#include "gdn.hpp"
#include <array>
#include <cmath>
#include <optional>
#include <vector>
#include <mlx/compile.h>
#include <stdexcept>
#include <cstdlib>
#include <string_view>
#include <limits>

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
mx::array gdn_convolve_native(const mx::array& joined,const mx::array& weights) {
    if(joined.ndim()!=2 || joined.shape(0)<4 || joined.shape(1)<1 ||
       uint64_t(joined.shape(0)-3)*uint64_t(joined.shape(1))>uint64_t(std::numeric_limits<int>::max()) ||
       weights.shape()!=mx::Shape{joined.shape(1),4} ||
       joined.dtype()!=mx::float32 || weights.dtype()!=mx::float32)
        throw std::invalid_argument("invalid native GDN convolution dimensions");
    const int m=joined.shape(0)-3,width=joined.shape(1);
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_convolve_native",{"joined","weights"},{"out"},R"METAL(
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
uint i=thread_position_in_grid.x;
if(i>=M*WIDTH)return;
uint token=i/WIDTH,col=i%WIDTH;
float sum=0.0f;
for(uint tap=0;tap<4u;++tap)sum=sum+joined[(token+tap)*WIDTH+col]*weights[col*4u+tap];
float rounded=float(bfloat16_t(sum));
float gate=1.0f/(1.0f+exp(-rounded));
out[i]=float(bfloat16_t(rounded*gate));
)METAL");
    return kernel({mx::contiguous(joined),mx::contiguous(weights)},{{m,width}},{mx::float32},
        {m*width,1,1},{256,1,1},{{"M",m},{"WIDTH",width}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
std::pair<mx::array,mx::array> gdn_gates_native(const mx::array& alpha,const mx::array& beta,
    const mx::array& bias,const mx::array& rate) {
    if(alpha.ndim()!=2 || alpha.shape(0)<1 || alpha.shape(1)<1 ||
       alpha.size()>size_t(std::numeric_limits<int>::max()) ||
       alpha.shape()!=beta.shape() || bias.shape()!=mx::Shape{alpha.shape(1)} ||
       rate.shape()!=bias.shape() || alpha.dtype()!=mx::float32 || beta.dtype()!=mx::float32 ||
       bias.dtype()!=mx::float32 || rate.dtype()!=mx::float32)
        throw std::invalid_argument("invalid native GDN gates dimensions");
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_gates_native",{"alpha","beta","bias","rate"},
        {"decay","gate"},R"METAL(
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
uint i=thread_position_in_grid.x;
if(i>=N)return;
uint head=i%HEADS;
float dt=alpha[i]+bias[head];
float softplus=max(dt,0.0f)+log1p(exp(-abs(dt)));
decay[i]=exp(softplus*rate[head]);
gate[i]=1.0f/(1.0f+exp(-beta[i]));
)METAL");
    auto result=kernel({mx::contiguous(alpha),mx::contiguous(beta),mx::contiguous(bias),mx::contiguous(rate)},
        {alpha.shape(),alpha.shape()},{mx::float32,mx::float32},{int(alpha.size()),1,1},{256,1,1},
        {{"N",int(alpha.size())},{"HEADS",alpha.shape(1)}},std::nullopt,false,mx::Device(mx::Device::gpu));
    return {result[0],result[1]};
}
// conv + silu (as gdn_convolve_native) and the q / k normalization in one launch: one simdgroup per (token, 128-column
// segment); segments [0, KH) are q (scaled by qscale), [KH, 2KH) are k, the rest v (no normalization).
mx::array gdn_pre_fused(const mx::array& joined,const mx::array& weights,int kh,float qscale) {
    if(joined.ndim()!=2 || joined.shape(0)<4 || weights.shape()!=mx::Shape{joined.shape(1),4} || joined.shape(1)%128 ||
       joined.dtype()!=mx::float32 || weights.dtype()!=mx::float32)
        throw std::invalid_argument("invalid fused GDN prep dimensions");
    const int m=joined.shape(0)-3,width=joined.shape(1),segs=width/128;
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_pre_fused",{"joined","weights","qscale"},{"out"},R"METAL(
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
uint lane=thread_index_in_simdgroup;
uint seg=threadgroup_position_in_grid.x*4u+simdgroup_index_in_threadgroup;
uint token=threadgroup_position_in_grid.y;
if(seg>=uint(SEGS))return;
float x[4];
for(uint e=0;e<4u;++e) {
    uint col=seg*128u+lane*4u+e;
    float sum=0.0f;
    for(uint tap=0;tap<4u;++tap)sum=sum+joined[(token+tap)*WIDTH+col]*weights[col*4u+tap];
    float rounded=float(bfloat16_t(sum));
    float gate=1.0f/(1.0f+exp(-rounded));
    x[e]=float(bfloat16_t(rounded*gate));
}
if(seg<2u*uint(KH)) {
    float ss=0.0f;
    for(uint e=0;e<4u;++e)ss+=x[e]*x[e];
    ss=simd_sum(ss);
    float factor=rsqrt(ss+1e-6f)*(seg<uint(KH)?qscale:1.0f);
    for(uint e=0;e<4u;++e)x[e]=x[e]*factor;
}
for(uint e=0;e<4u;++e)out[token*WIDTH+seg*128u+lane*4u+e]=x[e];
)METAL");
    return kernel({mx::contiguous(joined),mx::contiguous(weights),mx::array(qscale)},{{m,width}},{mx::float32},
        {((segs+3)/4)*128,m,1},{128,1,1},{{"WIDTH",width},{"SEGS",segs},{"KH",kh}},
        std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
// gdn_pre_fused on a prefill pack's FP16 product (qkv in its first WIDTH columns, row stride LD), the conv history
// separate: each joined value is history row r < 3 or the product's BF16 rounding (as round_product), read in place, so
// no rounded copy, no concatenate; q, k and v come out as separate dense arrays (the scan reads them without copies).
// Bitwise equal to gdn_pre_fused(concatenate(history, round(product))).
std::array<mx::array,3> gdn_pre_pack(const mx::array& history,const mx::array& product,int width,const mx::array& weights,
                                     int kh,float qscale) {
    const int m=product.shape(0),ld=product.shape(1),segs=width/128;
    if(product.ndim()!=2 || product.dtype()!=mx::float16 || ld<width || width%128 || history.shape()!=mx::Shape{3,width} ||
       history.dtype()!=mx::float32 || weights.shape()!=mx::Shape{width,4} || weights.dtype()!=mx::float32 || 2*kh*128>=width)
        throw std::invalid_argument("invalid packed GDN prep dimensions");
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_pre_pack",{"hist","prod","weights","qscale"},{"q","k","v"},R"METAL(
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
uint lane=thread_index_in_simdgroup;
uint seg=threadgroup_position_in_grid.x*4u+simdgroup_index_in_threadgroup;
uint token=threadgroup_position_in_grid.y;
if(seg>=uint(SEGS))return;
float x[4];
for(uint e=0;e<4u;++e) {
    uint col=seg*128u+lane*4u+e;
    float sum=0.0f;
    for(uint tap=0;tap<4u;++tap) {
        uint r=token+tap;
        float j=r<3u?hist[r*WIDTH+col]:float(bfloat16_t(float(prod[(size_t)(r-3u)*LD+col])));
        sum=sum+j*weights[col*4u+tap];
    }
    float rounded=float(bfloat16_t(sum));
    float gate=1.0f/(1.0f+exp(-rounded));
    x[e]=float(bfloat16_t(rounded*gate));
}
if(seg<2u*uint(KH)) {
    float ss=0.0f;
    for(uint e=0;e<4u;++e)ss+=x[e]*x[e];
    ss=simd_sum(ss);
    float factor=rsqrt(ss+1e-6f)*(seg<uint(KH)?qscale:1.0f);
    for(uint e=0;e<4u;++e)x[e]=x[e]*factor;
}
constexpr uint QW=uint(KH)*128u,VW=uint(WIDTH)-2u*QW;
for(uint e=0;e<4u;++e) {
    uint c=seg*128u+lane*4u+e;
    if(c<QW)q[token*QW+c]=x[e];
    else if(c<2u*QW)k[token*QW+c-QW]=x[e];
    else v[token*VW+c-2u*QW]=x[e];
}
)METAL");
    auto out=kernel({mx::contiguous(history),mx::contiguous(product),mx::contiguous(weights),mx::array(qscale)},
        {{m,kh*128},{m,kh*128},{m,width-2*kh*128}},{mx::float32,mx::float32,mx::float32},
        {((segs+3)/4)*128,m,1},{128,1,1},{{"WIDTH",width},{"SEGS",segs},{"KH",kh},{"LD",ld}},
        std::nullopt,false,mx::Device(mx::Device::gpu));
    return {out[0],out[1],out[2]};
}
// gdn_post_fused with z read in place from a prefill pack's FP16 product (columns offset .. offset + heads x 128, row
// stride LD) and rounded as round_product: bitwise equal to gdn_post_fused(y, reshape(round(product slice)))
mx::array gdn_post_pack(const mx::array& y,const mx::array& product,int offset,const mx::array& weight,float eps) {
    if(y.ndim()!=4 || y.shape(3)!=128 || y.dtype()!=mx::float32 || product.ndim()!=2 || product.dtype()!=mx::float16 ||
       product.shape(0)!=y.shape(1) || offset+y.shape(2)*128>product.shape(1) || weight.shape()!=mx::Shape{128} ||
       weight.dtype()!=mx::float32)
        throw std::invalid_argument("invalid packed GDN output dimensions");
    const int rows=y.shape(1)*y.shape(2);
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_post_pack",{"y","prod","w","eps"},{"out"},R"METAL(
uint lane=thread_index_in_simdgroup;
uint row=threadgroup_position_in_grid.x*4u+simdgroup_index_in_threadgroup;
if(row>=uint(ROWS))return;
float a[4];float ss=0.0f;
for(uint e=0;e<4u;++e) {a[e]=float(bfloat16_t(y[row*128u+lane*4u+e]));ss+=a[e]*a[e];}
ss=simd_sum(ss);
float inv=rsqrt(ss/128.0f+eps);
const size_t zrow=(size_t)(row/uint(HEADS))*LD+OFFSET+(row%uint(HEADS))*128u;
for(uint e=0;e<4u;++e) {
    uint i=row*128u+lane*4u+e;
    float g=float(bfloat16_t(float(prod[zrow+lane*4u+e])));
    out[i]=float(bfloat16_t((a[e]*inv*w[lane*4u+e])*(g*(1.0f/(1.0f+exp(-g))))));
}
)METAL");
    return kernel({mx::contiguous(y),mx::contiguous(product),mx::contiguous(weight),mx::array(eps)},{y.shape()},{mx::float32},
        {((rows+3)/4)*128,1,1},{128,1,1},{{"ROWS",rows},{"HEADS",y.shape(2)},{"LD",product.shape(1)},{"OFFSET",offset}},
        std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
// bf16(y) -> rms_norm(weight, eps) -> * silu(z) -> bf16, one simdgroup per (token, value head)
mx::array gdn_post_fused(const mx::array& y,const mx::array& z,const mx::array& weight,float eps) {
    if(y.ndim()!=4 || y.shape(3)!=128 || z.shape()!=y.shape() || weight.shape()!=mx::Shape{128} ||
       y.dtype()!=mx::float32 || z.dtype()!=mx::float32 || weight.dtype()!=mx::float32)
        throw std::invalid_argument("invalid fused GDN output dimensions");
    const int rows=y.shape(1)*y.shape(2);
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_post_fused",{"y","z","w","eps"},{"out"},R"METAL(
uint lane=thread_index_in_simdgroup;
uint row=threadgroup_position_in_grid.x*4u+simdgroup_index_in_threadgroup;
if(row>=uint(ROWS))return;
float a[4];float ss=0.0f;
for(uint e=0;e<4u;++e) {a[e]=float(bfloat16_t(y[row*128u+lane*4u+e]));ss+=a[e]*a[e];}
ss=simd_sum(ss);
float inv=rsqrt(ss/128.0f+eps);
for(uint e=0;e<4u;++e) {
    uint i=row*128u+lane*4u+e;
    float g=z[i];
    out[i]=float(bfloat16_t((a[e]*inv*w[lane*4u+e])*(g*(1.0f/(1.0f+exp(-g))))));
}
)METAL");
    return kernel({mx::contiguous(y),mx::contiguous(z),mx::contiguous(weight),mx::array(eps)},{y.shape()},{mx::float32},
        {((rows+3)/4)*128,1,1},{128,1,1},{{"ROWS",rows}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
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
// prefill chunks (no snapshots): 8 lanes per state row holding 16 columns each, so each token's two row reductions
// are 3 shuffle steps instead of 5 and a simdgroup carries 4 rows (fewer instructions per row and token; the scan is
// issue-bound). Same recurrence, different FP32 summation tree than the 32-lane scan below, which decode, verify and
// the compact tape keep.
GdnScan gdn_scan8(const mx::array& q,const mx::array& k,const mx::array& v,const mx::array& decay,const mx::array& beta,
    const mx::array& initial) {
    const int m=q.shape(1),kh=q.shape(2),vh=v.shape(2);
    static auto kernel8=mx::fast::metal_kernel("tqllm_gdn_scan8",{"q","k","v","decay","beta","initial"},{"out","states"},R"METAL(
uint tid=thread_position_in_grid.x,head=thread_position_in_grid.z;
uint row=tid>>3,sub=tid&7u;
uint base=(head*128u+row)*128u+sub*16u;
float s[16];
for(uint j=0;j<4;++j) {float4 a=*((const device float4*)(initial+base+4u*j));s[4*j]=a.x;s[4*j+1]=a.y;s[4*j+2]=a.z;s[4*j+3]=a.w;}
for(uint t=0;t<M;++t) {
 uint at=(t*KH+head%KH)*128u+sub*16u;
 float key[16],query[16];
 for(uint j=0;j<4;++j) {
  float4 a=*((const device float4*)(k+at+4u*j)),b=*((const device float4*)(q+at+4u*j));
  key[4*j]=a.x;key[4*j+1]=a.y;key[4*j+2]=a.z;key[4*j+3]=a.w;query[4*j]=b.x;query[4*j+1]=b.y;query[4*j+2]=b.z;query[4*j+3]=b.w;
 }
 float d=decay[t*VH+head];
 float memory=0.0f;
 for(uint j=0;j<16;++j) {s[j]*=d;memory+=s[j]*key[j];}
 memory+=simd_shuffle_xor(memory,1u);memory+=simd_shuffle_xor(memory,2u);memory+=simd_shuffle_xor(memory,4u);
 float delta=(v[(t*VH+head)*128u+row]-memory)*beta[t*VH+head];
 float value=0.0f;
 for(uint j=0;j<16;++j) {s[j]=s[j]+key[j]*delta;value+=s[j]*query[j];}
 value+=simd_shuffle_xor(value,1u);value+=simd_shuffle_xor(value,2u);value+=simd_shuffle_xor(value,4u);
 if(sub==0)out[(t*VH+head)*128u+row]=value;
}
for(uint j=0;j<4;++j)*((device float4*)(states+base+4u*j))=float4(s[4*j],s[4*j+1],s[4*j+2],s[4*j+3]);
)METAL");
    auto r=kernel8({mx::contiguous(q),mx::contiguous(k),mx::contiguous(v),mx::contiguous(decay),mx::contiguous(beta),
            mx::contiguous(initial)},{{1,m,vh,128},{1,vh,128,128}},{mx::float32,mx::float32},{1024,1,vh},{256,1,1},
            {{"M",m},{"KH",kh},{"VH",vh}},std::nullopt,false,mx::Device(mx::Device::gpu));
    return {r[0],r[1],std::nullopt,std::nullopt};
}
// gdn_scan8 with each 16-token block of k, q, the threadgroup's v rows and the gates staged once in threadgroup memory
// (as Splash's prefill scan, prefill/gdn.metal) instead of every simdgroup reading them from device memory per token;
// ROWS state rows per threadgroup. The per-row arithmetic is gdn_scan8's, so the outputs are equal bit for bit.
GdnScan gdn_scan8_staged(const mx::array& q,const mx::array& k,const mx::array& v,const mx::array& decay,const mx::array& beta,
    const mx::array& initial,int rows) {
    const int m=q.shape(1),kh=q.shape(2),vh=v.shape(2);
    if(rows!=16 && rows!=32)throw std::invalid_argument("gdn_scan8_staged: 16 or 32 rows per threadgroup");
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_scan8_staged",{"q","k","v","decay","beta","initial"},{"out","states"},R"METAL(
constexpr uint T8=uint(ROWS)*8u,B=16u;
threadgroup float kb[B*128u],qb[B*128u],vb[B*uint(ROWS)],db[B],bb[B];
uint lt=thread_position_in_threadgroup.x,head=threadgroup_position_in_grid.y;
uint row0=threadgroup_position_in_grid.x*uint(ROWS),lr=lt>>3,row=row0+lr,sub=lt&7u;
uint base=(head*128u+row)*128u+sub*16u;
float s[16];
for(uint j=0;j<4;++j) {float4 a=*((const device float4*)(initial+base+4u*j));s[4*j]=a.x;s[4*j+1]=a.y;s[4*j+2]=a.z;s[4*j+3]=a.w;}
for(uint t0=0;t0<uint(M);t0+=B) {
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint e=lt*4u;e<B*128u;e+=T8*4u) {
  uint tt=t0+e/128u;float4 kv=float4(0.0f),qv=float4(0.0f);
  if(tt<uint(M)) {uint at=(tt*KH+head%KH)*128u+e%128u;kv=*((const device float4*)(k+at));qv=*((const device float4*)(q+at));}
  *((threadgroup float4*)(kb+e))=kv;*((threadgroup float4*)(qb+e))=qv;
 }
 for(uint e=lt;e<B*uint(ROWS);e+=T8) {uint tt=t0+e/uint(ROWS);vb[e]=tt<uint(M)?v[(tt*VH+head)*128u+row0+e%uint(ROWS)]:0.0f;}
 if(lt<B) {uint tt=t0+lt;db[lt]=tt<uint(M)?decay[tt*VH+head]:0.0f;bb[lt]=tt<uint(M)?beta[tt*VH+head]:0.0f;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 const uint nb=min(B,uint(M)-t0);
 for(uint i=0;i<nb;++i) {
  float key[16],query[16];
  for(uint j=0;j<16;++j) {key[j]=kb[i*128u+sub*16u+j];query[j]=qb[i*128u+sub*16u+j];}
  float d=db[i];
  float memory=0.0f;
  for(uint j=0;j<16;++j) {s[j]*=d;memory+=s[j]*key[j];}
  memory+=simd_shuffle_xor(memory,1u);memory+=simd_shuffle_xor(memory,2u);memory+=simd_shuffle_xor(memory,4u);
  float delta=(vb[i*uint(ROWS)+lr]-memory)*bb[i];
  float value=0.0f;
  for(uint j=0;j<16;++j) {s[j]=s[j]+key[j]*delta;value+=s[j]*query[j];}
  value+=simd_shuffle_xor(value,1u);value+=simd_shuffle_xor(value,2u);value+=simd_shuffle_xor(value,4u);
  if(sub==0)out[((t0+i)*VH+head)*128u+row]=value;
 }
}
for(uint j=0;j<4;++j)*((device float4*)(states+base+4u*j))=float4(s[4*j],s[4*j+1],s[4*j+2],s[4*j+3]);
)METAL");
    auto r=kernel({mx::contiguous(q),mx::contiguous(k),mx::contiguous(v),mx::contiguous(decay),mx::contiguous(beta),
            mx::contiguous(initial)},{{1,m,vh,128},{1,vh,128,128}},{mx::float32,mx::float32},{128*8,vh,1},{rows*8,1,1},
            {{"M",m},{"KH",kh},{"VH",vh},{"ROWS",rows}},std::nullopt,false,mx::Device(mx::Device::gpu));
    return {r[0],r[1],std::nullopt,std::nullopt};
}
GdnScan gdn_scan(const mx::array& q,const mx::array& k,const mx::array& v,
    const mx::array& decay,const mx::array& beta,const std::optional<mx::array>& state,bool snapshots,bool compact) {
    if(q.ndim()!=4 || q.shape(0)!=1 || q.shape(3)!=128 || q.dtype()!=mx::float32 || k.shape()!=q.shape() ||
       k.dtype()!=mx::float32 || v.ndim()!=4 || v.shape(0)!=1 || v.shape(1)!=q.shape(1) ||
       v.shape(3)!=128 || v.dtype()!=mx::float32 || q.shape(2)<1 || v.shape(2)%q.shape(2) ||
       decay.shape()!=mx::Shape{1,v.shape(1),v.shape(2)} || beta.shape()!=decay.shape() ||
       decay.dtype()!=mx::float32 || beta.dtype()!=mx::float32)
        throw std::invalid_argument("unsupported GDN scan dimensions");
    const int m=q.shape(1),kh=q.shape(2),vh=v.shape(2);
    if(compact && (!snapshots || m>16))
        throw std::invalid_argument("compact GDN tape requires prefix snapshots and 1-8 tokens");
    auto initial=state?*state:mx::zeros({1,vh,128,128});
    if(initial.shape()!=mx::Shape{1,vh,128,128} || initial.dtype()!=mx::float32)
        throw std::invalid_argument("invalid GDN scan state");
    // TQLLM_MLX_GDN_SCAN_STAGED=0: gdn_scan8 (equal output); =16/32 rows per threadgroup, default 16
    static const int staged=[]{const char* e=std::getenv("TQLLM_MLX_GDN_SCAN_STAGED");return e?std::atoi(e):16;}();
    if(!snapshots && !compact && m>=64)return staged==16||staged==32?gdn_scan8_staged(q,k,v,decay,beta,initial,staged):gdn_scan8(q,k,v,decay,beta,initial);
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_scan",{"q","k","v","decay","beta","initial"},{"out","states","deltas"},R"METAL(
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
 if(COMPACT && lane==0)deltas[(t*VH+head)*128u+row]=delta;
 float value=0.0f;
 for(uint j=0;j<4;++j) {s[j]=s[j]+key[j]*delta;value+=s[j]*query[j];}
 value=simd_sum(value);
 if(lane==0)out[(t*VH+head)*128u+row]=value;
 if((SNAPSHOTS && !COMPACT) || t+1==M) {
  uint offset=((SNAPSHOTS && !COMPACT)?t:0u)*VH*128u*128u+base;
  *((device float4*)(states+offset))=s;
 }
}
)METAL");
    auto result=kernel({mx::contiguous(q),mx::contiguous(k),mx::contiguous(v),mx::contiguous(decay),
        mx::contiguous(beta),mx::contiguous(initial)},{{1,m,vh,128},{snapshots&&!compact?m:1,vh,128,128},compact?mx::Shape{m,vh,128}:mx::Shape{1}},
        {mx::float32,mx::float32,mx::float32},{32,128,vh},{32,4,1},{{"M",m},{"KH",kh},{"VH",vh},{"SNAPSHOTS",snapshots},{"COMPACT",compact}},
        std::nullopt,false,mx::Device(mx::Device::gpu));
    auto final=snapshots&&!compact?mx::slice(result[1],{m-1,0,0,0},{m,vh,128,128}):result[1];
    return {result[0],final,snapshots&&!compact?std::optional<mx::array>(result[1]):std::nullopt,
        compact?std::optional<mx::array>(result[2]):std::nullopt};
}
// One threadgroup of 1024 threads per value head: thread = (state row r = tid / 8, lane sub = tid % 8) holding
// state columns [16 sub, 16 sub + 16) of row r. Prep (conv + silu for q, k, v of every token, gates, q / k norms)
// goes to threadgroup memory once, the token recurrence then needs only 8-lane shuffles, and the gated norm over
// the 128 rows of each token runs after one barrier. Same per-token arithmetic for m = 1 and m = 16 (verify == AR).
GdnStep gdn_step_fused(const mx::array& joined,const mx::array& weights,const mx::array& alpha,const mx::array& beta,
    const mx::array& bias,const mx::array& rate,const mx::array& z,const mx::array& norm_weight,float eps,float query_scale,
    const std::optional<mx::array>& state,int kh,int vh) {
    if(joined.ndim()!=2 || joined.shape(0)<4)throw std::invalid_argument("invalid fused GDN step dimensions");
    const int n=joined.shape(0),width=joined.shape(1);
    return gdn_step_fused(mx::slice(joined,{0,0},{3,width}),mx::slice(joined,{3,0},{n,width}),weights,alpha,beta,bias,rate,z,
                          norm_weight,eps,query_scale,state,kh,vh);
}
GdnStep gdn_step_fused(const mx::array& prev,const mx::array& rows,const mx::array& weights,const mx::array& input,
    const mx::array& ab,const mx::array& bias,const mx::array& rate,const mx::array& z,const mx::array& norm_weight,
    float eps,float query_scale,const std::optional<mx::array>& state,int kh,int vh,int row_stride,int z_stride,int z_offset) {
    const int m=rows.ndim()==2?rows.shape(0):0,width=2*kh*128+vh*128;
    const int rs=row_stride>0?row_stride:width,zs=z_stride>0?z_stride:vh*128;
    if(rows.ndim()!=2 || rows.shape(1)!=rs || rs<width || zs<vh*128 || z_offset<0 || z_offset+vh*128>zs)
        throw std::invalid_argument("invalid fused GDN step packed product layout");
    const bool gates_given=input.ndim()==2 && input.shape(1)==vh;  // alpha / beta [m, vh] already projected
    const bool f32=prev.dtype()==mx::float32 && rows.dtype()==mx::float32 && weights.dtype()==mx::float32 && input.dtype()==mx::float32 &&
        ab.dtype()==mx::float32 && bias.dtype()==mx::float32 && rate.dtype()==mx::float32 && z.dtype()==mx::float32 &&
        norm_weight.dtype()==mx::float32;
    if(!f32 || m<1 || m>16 || kh<1 || vh<kh || vh%kh || width!=2*kh*128+vh*128 || prev.shape()!=mx::Shape{3,width} ||
       weights.shape()!=mx::Shape{width,4} ||
       (gates_given && (input.shape()!=mx::Shape{m,vh} || ab.shape()!=input.shape())) || bias.shape()!=mx::Shape{vh} ||
       rate.shape()!=bias.shape() || z.shape()!=mx::Shape{m,zs} || norm_weight.shape()!=mx::Shape{128})
        throw std::invalid_argument("invalid fused GDN step dimensions");
    auto initial=state?*state:mx::zeros({1,vh,128,128});
    if(initial.shape()!=mx::Shape{1,vh,128,128} || initial.dtype()!=mx::float32)
        throw std::invalid_argument("invalid fused GDN step state");
    static const std::string body=R"METAL(
const uint tid=thread_position_in_threadgroup.x,head=threadgroup_position_in_grid.y;
const uint khead=head%uint(KH),row=tid>>3,sub=tid&7u;
const uint qoff=khead*128u,koff=uint(KH)*128u+khead*128u,voff=2u*uint(KH)*128u+head*128u;
threadgroup float qs[M*128];
threadgroup float ks[M*128];
threadgroup bfloat16_t vs[M*128];  // BF16-rounded values: exact in BF16, so 16 tokens fit 32 KB
threadgroup float os[M*128];
threadgroup float dec[M];
threadgroup float bet[M];
// prep: conv + silu, one (token, channel) per iteration; channels 0-127 q, 128-255 k, 256-383 v. Strict fp here
// only (as gdn_convolve_native / gdn_gates_native); the recurrence below keeps the default contraction of gdn_scan
// and gdn_replay_prefix so a replayed partial accept equals repeated single-token steps bit for bit
{
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
for(uint i=tid;i<uint(M)*384u;i+=1024u) {
    const uint t=i/384u,ch=i%384u;
    const uint col=ch<128u?qoff+ch:ch<256u?koff+(ch-128u):voff+(ch-256u);
    float sum=0.0f;
    for(uint tap=0;tap<4u;++tap) {
        const uint r=t+tap;  // row r of [prev; rows]
        sum=sum+(r<3u?prev[r*uint(WIDTH)+col]:rows[(r-3u)*uint(RS)+col])*weights[col*4u+tap];
    }
    const float rounded=float(bfloat16_t(sum));
    const float gate=1.0f/(1.0f+exp(-rounded));
    const float y=float(bfloat16_t(rounded*gate));
    if(ch<128u)qs[t*128u+ch]=y;else if(ch<256u)ks[t*128u+ch-128u]=y;else vs[t*128u+ch-256u]=bfloat16_t(y);
}
}
// ABF: the alpha / beta projections of this head in-kernel (simdgroup = (token, alpha | beta), 32 lanes x 4 columns per
// 128-column step, fixed order so m = 1 and m = 16 agree bit for bit), rounded to BF16 like the matmul path's bf()
threadgroup float abv[32];
if(ABF) {
    const uint g=tid>>5,lane=tid&31u;
    if(g<2u*uint(M)) {
        const uint t=g>>1,which=g&1u;
        const device float* xr=A+t*uint(HID)+lane*4u;
        const device float* wr=B+(which*uint(VH)+head)*uint(HID)+lane*4u;
        float acc=0.0f;
        for(uint c=0;c<uint(HID);c+=128u) {
            const float4 xv=*((const device float4*)(xr+c)),wv=*((const device float4*)(wr+c));
            acc+=xv.x*wv.x;acc+=xv.y*wv.y;acc+=xv.z*wv.z;acc+=xv.w*wv.w;
        }
        acc=simd_sum(acc);
        if(lane==0)abv[t*2u+which]=float(bfloat16_t(acc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(tid<uint(M)) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const float dt=(ABF?abv[tid*2u]:A[tid*uint(VH)+head])+bias[head];
    const float softplus=max(dt,0.0f)+log1p(exp(-abs(dt)));
    dec[tid]=exp(softplus*rate[head]);
    bet[tid]=1.0f/(1.0f+exp(-(ABF?abv[tid*2u+1u]:B[tid*uint(VH)+head])));
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// q / k normalization: simdgroup g < 2 M handles (token g / 2, q or k), 32 lanes x 4 channels
{
    const uint g=tid>>5,lane=tid&31u;
    if(g<2u*uint(M)) {
        threadgroup float* vec=(g&1u)?ks+(g>>1)*128u:qs+(g>>1)*128u;
        float ss=0.0f;
        for(uint e=0;e<4u;++e) {const float x=vec[lane*4u+e];ss+=x*x;}
        ss=simd_sum(ss);
        const float factor=rsqrt(ss+1e-6f)*((g&1u)?1.0f:qscale);
        for(uint e=0;e<4u;++e)vec[lane*4u+e]=vec[lane*4u+e]*factor;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// recurrence over the tokens; the 8 lanes of a row reduce with xor shuffles inside their simdgroup
float s[16];
{
    const device float* src=initial+(head*128u+row)*128u+sub*16u;
    for(uint j=0;j<16u;++j)s[j]=src[j];
}
for(uint t=0;t<uint(M);++t) {
    const float d=dec[t],b=bet[t];
    const threadgroup float* kt=ks+t*128u+sub*16u;
    const threadgroup float* qt=qs+t*128u+sub*16u;
    float memory=0.0f;
    for(uint j=0;j<16u;++j) {s[j]*=d;memory+=s[j]*kt[j];}
    memory+=simd_shuffle_xor(memory,1u);memory+=simd_shuffle_xor(memory,2u);memory+=simd_shuffle_xor(memory,4u);
    const float delta=(float(vs[t*128u+row])-memory)*b;
    float value=0.0f;
    for(uint j=0;j<16u;++j) {s[j]=s[j]+kt[j]*delta;value+=s[j]*qt[j];}
    value+=simd_shuffle_xor(value,1u);value+=simd_shuffle_xor(value,2u);value+=simd_shuffle_xor(value,4u);
    if(sub==0u) {os[t*128u+row]=value;deltas[(t*uint(VH)+head)*128u+row]=delta;}
}
{
    device float* dst=state+(head*128u+row)*128u+sub*16u;
    for(uint j=0;j<16u;++j)dst[j]=s[j];
}
// tape: normalized keys (key-head threadgroups), decay, conv history (rows M .. M + 2 of joined)
if(head<uint(KH))for(uint i=tid;i<uint(M)*128u;i+=1024u)keys[((i>>7)*uint(KH)+head)*128u+(i&127u)]=ks[i];
if(tid<uint(M))decay[tid*uint(VH)+head]=dec[tid];
for(uint i=tid;i<384u;i+=1024u) {
    const uint r=i>>7,c=i&127u,j=uint(M)+r,dst=r*uint(WIDTH);  // history row r = row M + r of [prev; rows]
    const device float* src=j<3u?prev+j*uint(WIDTH):rows+(j-3u)*uint(RS);
    history[dst+voff+c]=src[voff+c];
    if(head<uint(KH)) {history[dst+qoff+c]=src[qoff+c];history[dst+koff+c]=src[koff+c];}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// gated norm per token: simdgroup t < M, 32 lanes x 4 rows; bf16(o) -> rms_norm(w, eps) -> * silu(z) -> bf16
{
    const uint g=tid>>5,lane=tid&31u;
    if(g<uint(M)) {
        float a[4];float ss=0.0f;
        for(uint e=0;e<4u;++e) {a[e]=float(bfloat16_t(os[g*128u+lane*4u+e]));ss+=a[e]*a[e];}
        ss=simd_sum(ss);
        const float inv=rsqrt(ss/128.0f+eps);
        for(uint e=0;e<4u;++e) {
            const uint r=lane*4u+e,i=(g*uint(VH)+head)*128u+r;
            const float zz=z[g*uint(ZS)+uint(ZOFF)+head*128u+r];
            out[i]=half(float(bfloat16_t((a[e]*inv*w[r])*(zz*(1.0f/(1.0f+exp(-zz)))))));  // BF16 value as the FP16 ssm_out input
        }
    }
}
)METAL";
    static auto kernel=mx::fast::metal_kernel("tqllm_gdn_step_fused",
        {"prev","rows","weights","A","B","bias","rate","z","w","eps","qscale","initial"},
        {"out","state","deltas","keys","decay","history"},body);
    static auto kernel_ab=mx::fast::metal_kernel("tqllm_gdn_step_fused_ab",
        {"prev","rows","weights","A","B","bias","rate","z","w","eps","qscale","initial"},
        {"out","state","deltas","keys","decay","history"},body);
    const bool abf=input.ndim()==2 && input.shape(1)!=vh;  // [m, hidden] input + [2 vh, hidden] weights instead of alpha / beta
    const int hid=abf?input.shape(1):0;
    if(abf && (input.shape(0)!=m || hid<128 || hid%128 || ab.shape()!=mx::Shape{2*vh,hid} || ab.dtype()!=mx::float32 || input.dtype()!=mx::float32))
        throw std::invalid_argument("invalid fused GDN step alpha / beta projection");
    auto result=(abf?kernel_ab:kernel)({mx::contiguous(prev),mx::contiguous(rows),mx::contiguous(weights),mx::contiguous(input),mx::contiguous(ab),
        mx::contiguous(bias),mx::contiguous(rate),mx::contiguous(z),mx::contiguous(norm_weight),mx::array(eps),
        mx::array(query_scale),mx::contiguous(initial)},
        {{m,vh*128},{1,vh,128,128},{m,vh,128},{1,m,kh,128},{1,m,vh},{3,width}},
        {mx::float16,mx::float32,mx::float32,mx::float32,mx::float32,mx::float32},
        {1024,vh,1},{1024,1,1},{{"M",m},{"KH",kh},{"VH",vh},{"WIDTH",width},{"ABF",int(abf)},{"HID",hid},{"RS",rs},{"ZS",zs},{"ZOFF",z_offset}},
        std::nullopt,false,mx::Device(mx::Device::gpu));
    return {result[0],result[1],result[2],result[3],result[4],result[5]};
}
// Chunked gated delta rule for prefill (no per-token snapshots). Same recurrence as gdn_scan
// (s *= a; d = (v - s k) b; s += d k^T; o = s q) over sub-chunks of c tokens, g the cumulative decay inside a sub-chunk:
//   T = (I + diag(b) A)^-1,  A_tj = (g_t / g_j) k_t.k_j (j < t);  Uv = T diag(b) V,  W = T diag(b g) K
//   U_j = Uv_j - W_j S_j^T,  S_{j+1} = g_C S_j + (U_j * g_C / g)^T K_j   (the only sequential part)
//   O = diag(g) Q S_j^T + (Q K^T * G, causal incl. diagonal) U_j
// T by forward substitution (a power-series product loses everything to cancellation on near-identical keys).
mx::array gdn_replay_prefix(const mx::array& keys,const mx::array& decay,const mx::array& deltas,
    const std::optional<mx::array>& initial,int count) {
    if(keys.ndim()!=4 || keys.shape(0)!=1 || keys.shape(2)<1 || keys.shape(3)!=128 || keys.dtype()!=mx::float32 ||
       deltas.ndim()!=3 || deltas.shape(0)!=keys.shape(1) || deltas.shape(1)<1 || deltas.shape(2)!=128 ||
       deltas.dtype()!=mx::float32 || deltas.shape(1)%keys.shape(2) ||
       decay.shape()!=mx::Shape{1,keys.shape(1),deltas.shape(1)} || decay.dtype()!=mx::float32 ||
       count<1 || count>keys.shape(1) || keys.shape(1)>16)
        throw std::invalid_argument("invalid compact GDN prefix tape");
    const int kh=keys.shape(2),vh=deltas.shape(1);
    auto state=initial?*initial:mx::zeros({1,vh,128,128});
    if(state.shape()!=mx::Shape{1,vh,128,128} || state.dtype()!=mx::float32)
        throw std::invalid_argument("invalid compact GDN initial state");
    static auto replay=mx::fast::metal_kernel("tqllm_gdn_replay_prefix",{"keys","decay","deltas","initial"},{"state"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=VH*128u*128u/4u)return;
uint head=i/(128u*32u),row=(i/32u)%128u,col=(i%32u)*4u;
float4 s=((const device float4*)initial)[i];
for(uint t=0;t<COUNT;++t) {
 float4 key=*((const device float4*)(keys+(t*KH+head%KH)*128u+col));
 float delta=deltas[(t*VH+head)*128u+row];
 s*=decay[t*VH+head];
 for(uint j=0;j<4;++j)s[j]=s[j]+key[j]*delta;
}
((device float4*)state)[i]=s;
)METAL");
    return replay({mx::contiguous(keys),mx::contiguous(decay),mx::contiguous(deltas),mx::contiguous(state)},
        {{1,vh,128,128}},{mx::float32},{vh*128*32,1,1},{256,1,1},
        {{"VH",vh},{"KH",kh},{"COUNT",count}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
}

#include "gdn.hpp"
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
GdnScan gdn_scan(const mx::array& q,const mx::array& k,const mx::array& v,
    const mx::array& decay,const mx::array& beta,const std::optional<mx::array>& state,bool snapshots,bool compact) {
    if(q.ndim()!=4 || q.shape(0)!=1 || q.shape(3)!=128 || q.dtype()!=mx::float32 || k.shape()!=q.shape() ||
       k.dtype()!=mx::float32 || v.ndim()!=4 || v.shape(0)!=1 || v.shape(1)!=q.shape(1) ||
       v.shape(3)!=128 || v.dtype()!=mx::float32 || q.shape(2)<1 || v.shape(2)%q.shape(2) ||
       decay.shape()!=mx::Shape{1,v.shape(1),v.shape(2)} || beta.shape()!=decay.shape() ||
       decay.dtype()!=mx::float32 || beta.dtype()!=mx::float32)
        throw std::invalid_argument("unsupported GDN scan dimensions");
    const int m=q.shape(1),kh=q.shape(2),vh=v.shape(2);
    if(compact && (!snapshots || m>8))
        throw std::invalid_argument("compact GDN tape requires prefix snapshots and 1-8 tokens");
    auto initial=state?*state:mx::zeros({1,vh,128,128});
    if(initial.shape()!=mx::Shape{1,vh,128,128} || initial.dtype()!=mx::float32)
        throw std::invalid_argument("invalid GDN scan state");
    static const bool skip_scan=std::getenv("TQLLM_MLX_ABLATE_BLOCKS") && std::string_view(std::getenv("TQLLM_MLX_ABLATE_BLOCKS")).find("gscan")!=std::string_view::npos;
    if(skip_scan && compact)
        return {mx::zeros({1,m,vh,128}),initial,std::nullopt,std::optional<mx::array>(mx::zeros({m,vh,128}))};
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
// Chunked gated delta rule for prefill (no per-token snapshots). Same recurrence as gdn_scan
// (s *= a; d = (v - s k) b; s += d k^T; o = s q) over sub-chunks of c tokens, g the cumulative decay inside a sub-chunk:
//   T = (I + diag(b) A)^-1,  A_tj = (g_t / g_j) k_t.k_j (j < t);  Uv = T diag(b) V,  W = T diag(b g) K
//   U_j = Uv_j - W_j S_j^T,  S_{j+1} = g_C S_j + (U_j * g_C / g)^T K_j   (the only sequential part)
//   O = diag(g) Q S_j^T + (Q K^T * G, causal incl. diagonal) U_j
// T by forward substitution (a power-series product loses everything to cancellation on near-identical keys).
static mx::array unit_lower_inverse(const mx::array& l) {  // [batch, c, c] strictly lower L -> (I + L)^-1
    const int batch=l.shape(0),c=l.shape(1);
    static auto kernel=mx::fast::metal_kernel("tqllm_unit_lower_inverse",{"l"},{"t"},R"METAL(
uint x=thread_position_in_grid.x,bt=thread_position_in_grid.y;
const device float* row=l+size_t(bt)*C*C;
device float* out=t+size_t(bt)*C*C;
float col[C];
for(uint i=0;i<uint(C);++i) {
    float s=i==x?1.0f:0.0f;
    for(uint j=x;j<i;++j)s-=row[i*C+j]*col[j];
    col[i]=s;
    out[i*C+x]=s;
}
)METAL");
    return kernel({mx::contiguous(l)},{{batch,c,c}},{mx::float32},{c,batch,1},{c,1,1},{{"C",c}},
        std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
std::pair<mx::array,mx::array> gdn_chunked(const mx::array& q,const mx::array& k,const mx::array& v,
    const mx::array& decay,const mx::array& beta,const std::optional<mx::array>& state,int chunk) {
    const int m=q.shape(1),kh=q.shape(2),vh=v.shape(2),dk=q.shape(3),dv=v.shape(3);
    if(q.ndim()!=4 || k.shape()!=q.shape() || v.shape(1)!=m || vh%kh || (chunk!=16 && chunk!=32 && chunk!=64))
        throw std::invalid_argument("invalid chunked GDN dimensions");
    const int c=chunk,n=(m+c-1)/c,pad=n*c-m;
    auto heads=[&](const mx::array& x) {  // [1, m, kh, d] -> value heads [vh, m, d] (value head h reads key head h % kh)
        auto idx=mx::remainder(mx::arange(vh,mx::int32),mx::array(kh));
        return mx::transpose(mx::take(mx::squeeze(x,0),idx,1),{1,0,2});
    };
    auto blocks=[&](const mx::array& x,float value) {  // [vh, m, d] -> [vh, n, c, d]
        auto y=pad?mx::pad(x,std::vector<std::pair<int,int>>{{0,0},{0,pad},{0,0}},mx::array(value)):x;
        return mx::reshape(y,{vh,n,c,y.shape(2)});
    };
    auto rows=[&](const mx::array& x,float value) {  // [1, m, vh] -> [vh, n, c]
        auto y=mx::transpose(mx::squeeze(x,0),{1,0});
        if(pad)y=mx::pad(y,std::vector<std::pair<int,int>>{{0,0},{0,pad}},mx::array(value));
        return mx::reshape(y,{vh,n,c});
    };
    auto K=blocks(heads(k),0.0f),Q=blocks(heads(q),0.0f),V=blocks(mx::transpose(mx::squeeze(v,0),{1,0,2}),0.0f);
    auto B=rows(beta,0.0f),A=rows(decay,1.0f);
    auto strict=mx::astype(mx::tril(mx::ones({c,c}),-1),mx::float32),causal=mx::astype(mx::tril(mx::ones({c,c}),0),mx::float32);
    // log decays clamped (decay 0 -> exp(-1e4)); ratios as segment sums over (j, t] so no exp sees a positive
    // argument and masked pairs give 0, not inf * 0
    auto lg=mx::maximum(mx::log(A),mx::array(-1e4f)),gl=mx::cumsum(lg,2);
    auto seg=mx::cumsum(mx::multiply(mx::expand_dims(lg,3),strict),2);                      // sum_{j<s<=t} lg_s
    auto ratio=mx::exp(mx::where(mx::astype(causal,mx::bool_),seg,mx::array(-INFINITY)));    // g_t / g_j, 0 above
    auto Kt=mx::swapaxes(K,2,3);
    auto L=mx::multiply(mx::multiply(mx::matmul(K,Kt),mx::multiply(ratio,strict)),mx::expand_dims(B,3));  // diag(b) A
    auto T=mx::reshape(unit_lower_inverse(mx::reshape(L,{vh*n,c,c})),{vh,n,c,c});
    auto g=mx::exp(gl);
    auto Uv=mx::matmul(T,mx::multiply(mx::expand_dims(B,3),V));
    auto W=mx::matmul(T,mx::multiply(mx::expand_dims(mx::multiply(B,g),3),K));
    auto Aqk=mx::multiply(mx::matmul(Q,Kt),mx::multiply(ratio,causal));
    auto gC=mx::expand_dims(mx::exp(mx::slice(gl,{0,0,c-1},{vh,n,c})),3);                 // [vh, n, 1, 1]
    auto w=mx::expand_dims(mx::squeeze(mx::slice(ratio,{0,0,c-1,0},{vh,n,c,c}),2),3);       // g_C / g_t [vh, n, c, 1]
    auto S=state?mx::reshape(*state,{vh,dv,dk}):mx::zeros({vh,dv,dk});
    std::vector<mx::array> states,us;
    auto at=[&](const mx::array& x,int j) {return mx::squeeze(mx::slice(x,{0,j,0,0},{vh,j+1,x.shape(2),x.shape(3)}),1);};
    for(int j=0;j<n;++j) {
        states.push_back(S);
        auto U=mx::subtract(at(Uv,j),mx::matmul(at(W,j),mx::swapaxes(S,1,2)));                // [vh, c, dv]
        us.push_back(U);
        S=mx::add(mx::multiply(at(gC,j),S),mx::matmul(mx::swapaxes(mx::multiply(U,at(w,j)),1,2),at(K,j)));
    }
    auto Ss=mx::stack(states,1),Us=mx::stack(us,1);                                              // [vh, n, dv, dk], [vh, n, c, dv]
    auto O=mx::add(mx::matmul(mx::multiply(mx::expand_dims(g,3),Q),mx::swapaxes(Ss,2,3)),mx::matmul(Aqk,Us));
    O=mx::reshape(O,{vh,n*c,dv});
    if(pad)O=mx::slice(O,{0,0,0},{vh,m,dv});
    return {mx::expand_dims(mx::transpose(O,{1,0,2}),0),mx::reshape(S,{1,vh,dv,dk})};
}

mx::array gdn_replay_prefix(const mx::array& keys,const mx::array& decay,const mx::array& deltas,
    const std::optional<mx::array>& initial,int count) {
    if(keys.ndim()!=4 || keys.shape(0)!=1 || keys.shape(2)<1 || keys.shape(3)!=128 || keys.dtype()!=mx::float32 ||
       deltas.ndim()!=3 || deltas.shape(0)!=keys.shape(1) || deltas.shape(1)<1 || deltas.shape(2)!=128 ||
       deltas.dtype()!=mx::float32 || deltas.shape(1)%keys.shape(2) ||
       decay.shape()!=mx::Shape{1,keys.shape(1),deltas.shape(1)} || decay.dtype()!=mx::float32 ||
       count<1 || count>keys.shape(1) || keys.shape(1)>8)
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

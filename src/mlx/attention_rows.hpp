#pragma once
#include <mlx/ops.h>
#include <mlx/fast.h>
#include <stdexcept>
#include <vector>

namespace tqllm::mlx_backend {
inline mlx::core::array fused_decode_attention(const mlx::core::array& q,
        const mlx::core::array& k,const mlx::core::array& v,float scale) {
    namespace mx=mlx::core;
    if(q.ndim()!=4 || k.ndim()!=4 || v.shape()!=k.shape() || q.shape(0)!=1 || k.shape(0)!=1 ||
       q.shape(2)<1 || q.shape(2)>8 || k.shape(2)<q.shape(2) || k.shape(2)>=1024 ||
       q.shape(1)<1 || k.shape(1)<1 || q.shape(1)%k.shape(1) ||
       (q.shape(3)!=128 && q.shape(3)!=256) || k.shape(3)!=q.shape(3) ||
       q.dtype()!=mx::float32 || k.dtype()!=mx::float32 || v.dtype()!=mx::float32)
        throw std::invalid_argument("unsupported fused decode attention shape");
    const uint32_t layout[]={uint32_t(k.shape(2)),uint32_t(q.shape(2)),
        uint32_t(q.shape(1)),uint32_t(k.shape(1))};
    static auto kernel=mx::fast::metal_kernel("tqllm_decode_attention_rows",{"q","k","v","layout","scale"},
        {"out"},R"METAL(
uint tid=thread_position_in_threadgroup.x;
uint lane=tid%32u,partition=tid/32u;
uint group=threadgroup_position_in_grid.y;
uint length=layout[0],rows=layout[1],heads=layout[2],kvheads=layout[3];
uint head=group/rows,token=group%rows,kvhead=head/(heads/kvheads);
uint visible=length-rows+token+1u;
constexpr uint elements=DIM/32u;
float query[elements],weighted[elements];
for(uint j=0;j<elements;++j) {
    query[j]=scale[0]*q[group*DIM+lane*elements+j];
    weighted[j]=0.0f;
}
float peak=-3.4028234663852886e38f,mass=0.0f;
for(uint row=partition;row<visible;row+=32u) {
    uint offset=(kvhead*length+row)*DIM+lane*elements;
    float dot=0.0f;
    for(uint j=0;j<elements;++j)dot+=query[j]*k[offset+j];
    dot=simd_sum(dot);
    float next=max(peak,dot);
    float retain=fast::exp(peak-next),weight=fast::exp(dot-next);
    mass=mass*retain+weight;
    for(uint j=0;j<elements;++j)weighted[j]=weighted[j]*retain+weight*v[offset+j];
    peak=next;
}
// Reuse the same storage for partition metadata and the full numerator tile.
threadgroup float workspace[32u*DIM];
if(lane==0u) {workspace[partition]=peak;workspace[32u+partition]=mass;}
threadgroup_barrier(mem_flags::mem_threadgroup);
float common_peak=simd_max(workspace[lane]);
float factor=fast::exp(workspace[lane]-common_peak);
float total=simd_sum(workspace[32u+lane]*factor);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint j=0;j<elements;++j)workspace[(lane*elements+j)*32u+partition]=weighted[j];
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint j=0;j<elements;++j) {
    uint column=partition*elements+j;
    float value=simd_sum(workspace[column*32u+lane]*factor);
    if(lane==0u)out[group*DIM+column]=total==0.0f?value:value/total;
}
)METAL");
    return kernel({mx::contiguous(q),mx::contiguous(k),mx::contiguous(v),
        mx::array(layout,{4},mx::uint32),mx::array(scale)}, {q.shape()},{mx::float32},
        {1024,q.shape(1)*q.shape(2),1},{1024,1,1},{{"DIM",q.shape(3)}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
inline mlx::core::array causal_attention_rows(const mlx::core::array& q,
        const mlx::core::array& k, const mlx::core::array& v, float scale,
        bool decode_order, bool kv_views = false,bool fused_rows = false) {
    namespace mx = mlx::core;
    const int n = q.shape(2), past = k.shape(2) - n;
    if (past < 0 || v.shape(2) != k.shape(2))
        throw std::invalid_argument("attention cache shorter than query block");
    if(fused_rows && (decode_order || n==1) && q.shape(0)==1 && k.shape(2)<1024 &&
       n<=8 && (q.shape(3)==128 || q.shape(3)==256) && q.dtype()==mx::float32)
        return fused_decode_attention(q,k,v,scale);
    if (!decode_order || n == 1)
        return mx::fast::scaled_dot_product_attention(q, k, v, scale, n > 1 ? "causal" : "");
    std::vector<mx::array> output;
    output.reserve(n);
    for (int t = 0; t < n; ++t) {
        auto query = mx::contiguous(mx::slice(q, {0, 0, t, 0}, {q.shape(0), q.shape(1), t + 1, q.shape(3)}));
        auto keys = mx::slice(k, {0, 0, 0, 0}, {k.shape(0), k.shape(1), past + t + 1, k.shape(3)});
        auto values = mx::slice(v, {0, 0, 0, 0}, {v.shape(0), v.shape(1), past + t + 1, v.shape(3)});
        if(!kv_views) {keys=mx::contiguous(keys);values=mx::contiguous(values);}
        output.push_back(mx::fast::scaled_dot_product_attention(query, keys, values, scale, ""));
    }
    return mx::concatenate(output, 2);
}
}

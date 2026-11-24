#include "dflash_mlx.hpp"
#include <stdexcept>
#include <string>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
// One launch per call (was ~12 small ops): y = bf(w0 * x[t] + w1 * x[t - 1]) with w = dynamic (per group) + base (per
// column), the same FP32 products and sum order as the op graph (contraction off); with a residual, bf(residual + y)
mx::array dflash_conv(const mx::array& x,const mx::array& dynamic,const mx::array& base,int group,int side,
                      const std::optional<mx::array>& residual) {
    if(x.ndim()!=2 || x.dtype()!=mx::float32 || group<1 || x.shape(1)%group || side<0 || side>1 ||
       dynamic.shape()!=mx::Shape{x.shape(0),4*x.shape(1)/group} || base.size()!=size_t(4*x.shape(1)) ||
       (residual && residual->shape()!=x.shape()))
        throw std::invalid_argument("invalid DFlash convolution shape");
    static auto kernel=mx::fast::metal_kernel("tqllm_dflash_conv",{"x","dynamic","base","residual"},{"y"},R"METAL(
#pragma clang fp contract(off)
const uint i=thread_position_in_grid.x;
if(i>=uint(M)*uint(H))return;
const uint r=i/uint(H),col=i%uint(H),c0=2u*uint(SIDE),c1=c0+1u,gi=col/uint(GROUP);
const float w0=dynamic[r*4u*uint(G)+c0*uint(G)+gi]+base[c0*uint(H)+col];
const float w1=dynamic[r*4u*uint(G)+c1*uint(G)+gi]+base[c1*uint(H)+col];
const float x1=r>0u?x[i-uint(H)]:0.0f;
float acc=0.0f+w0*x[i];
acc=acc+w1*x1;
float out=float(bfloat16_t(acc));
if(RES)out=float(bfloat16_t(residual[i]+out));
y[i]=out;
)METAL");
    const int m=x.shape(0),h=x.shape(1);
    auto res=residual?mx::contiguous(mx::astype(*residual,mx::float32)):mx::zeros({1},mx::float32);
    return kernel({mx::contiguous(x),mx::contiguous(mx::astype(dynamic,mx::float32)),mx::contiguous(mx::astype(mx::reshape(base,{4*h}),mx::float32)),res},
                  {{m,h}},{mx::float32},{m*h,1,1},{256,1,1},
                  {{"M",m},{"H",h},{"G",h/group},{"GROUP",group},{"SIDE",side},{"RES",bool(residual)}},
                  std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}

// Two passes, one simdgroup each: every 2048-column chunk of a row keeps its own top 16 (per lane a 16-entry list, then
// 16 rounds of simd max), then one simdgroup per row takes the top 16 of the chunk winners. Replaces argpartition over
// [16, 248K] (j776: 2.9 ms of a 13.6 ms draft); the set equals argpartition's whenever the 16th value is not tied.
std::pair<mx::array,mx::array> dflash_top16(const mx::array& logits) {
    if(logits.ndim()!=2 || logits.shape(1)<16)throw std::invalid_argument("invalid DFlash top-16 logits");
    constexpr int chunk=2048;
    const int rows=logits.shape(0),n=logits.shape(1),chunks=(n+chunk-1)/chunk;
    // LOAD(c): value and id of candidate c of this row; list order: larger value first, ties lower id
    auto body=[](const char* load)->std::string {return std::string(R"METAL(
const uint lane=thread_index_in_simdgroup;
const uint part=threadgroup_position_in_grid.x,row=threadgroup_position_in_grid.y;
const uint begin=part*uint(SPAN),end=min(begin+uint(SPAN),uint(N));
float v[16];uint id[16];
for(uint j=0;j<16u;++j){v[j]=-INFINITY;id[j]=0xffffffffu;}
float lo=-INFINITY;uint lo_id=0xffffffffu,lo_at=0;
for(uint c=begin+lane;c<end;c+=32u) {
    )METAL")+load+R"METAL(
    if(x>lo || (x==lo && xi<lo_id)) {
        for(uint j=0;j<16u;++j)if(j==lo_at){v[j]=x;id[j]=xi;}
        lo=v[0];lo_id=id[0];lo_at=0;
        for(uint j=1;j<16u;++j)if(v[j]<lo || (v[j]==lo && id[j]>lo_id)){lo=v[j];lo_id=id[j];lo_at=j;}
    }
}
for(uint k=0;k<16u;++k) {
    float best=v[0];uint best_id=id[0];
    for(uint j=1;j<16u;++j)if(v[j]>best || (v[j]==best && id[j]<best_id)){best=v[j];best_id=id[j];}
    const float top=simd_max(best);
    const uint top_id=simd_min(best==top?best_id:0xffffffffu);
    if(lane==0u){values[(row*uint(PARTS)+part)*16u+k]=top;ids[(row*uint(PARTS)+part)*16u+k]=top_id;}
    for(uint j=0;j<16u;++j)if(id[j]==top_id && v[j]==top){v[j]=-INFINITY;id[j]=0xffffffffu;}
}
)METAL";};
    static auto first=mx::fast::metal_kernel("tqllm_dflash_top16_chunks",{"logits"},{"values","ids"},
        body("const float x=logits[row*uint(N)+c];const uint xi=c;"));
    static auto second=mx::fast::metal_kernel("tqllm_dflash_top16_merge",{"in_values","in_ids"},{"values","ids"},
        body("const float x=in_values[row*uint(N)+c];const uint xi=in_ids[row*uint(N)+c];"));
    const auto gpu=mx::Device(mx::Device::gpu);
    auto part=first({mx::contiguous(mx::astype(logits,mx::float32))},{{rows,chunks*16},{rows,chunks*16}},{mx::float32,mx::uint32},
                    {32*chunks,rows,1},{32,1,1},{{"N",n},{"SPAN",chunk},{"PARTS",chunks}},std::nullopt,false,gpu);
    auto top=second({part[0],part[1]},{{rows,16},{rows,16}},{mx::float32,mx::uint32},
                    {32,rows,1},{32,1,1},{{"N",chunks*16},{"SPAN",chunks*16},{"PARTS",1}},std::nullopt,false,gpu);
    return {top[1],top[0]};
}
mx::array dflash_select(const mx::array& mapped,const mx::array& unary,const mx::array& gate,
    const mx::array& successors,const mx::array& predecessors,uint32_t anchor,int count) {
    if(anchor>=uint32_t(predecessors.shape(0)))throw std::invalid_argument("invalid DFlash selector dimensions");
    return dflash_select(mapped,unary,gate,successors,predecessors,mx::array(anchor),count);
}
mx::array dflash_select(const mx::array& mapped,const mx::array& unary,const mx::array& gate,
    const mx::array& successors,const mx::array& predecessors,const mx::array& anchor_id,int count) {
    const int R=mapped.ndim()==2?mapped.shape(0):0;  // draft block rows (8, or 16: DFlash2 block_rows)
    if((R!=8 && R!=16) || mapped.shape()!=mx::Shape{R,16} || mapped.dtype()!=mx::uint32 ||
       unary.shape()!=mapped.shape() || unary.dtype()!=mx::float32 ||
       gate.shape()!=mx::Shape{R,256} || gate.dtype()!=mx::float32 ||
       successors.shape()!=mx::Shape{R,16,256} || successors.dtype()!=mx::float32 ||
       predecessors.ndim()!=2 || predecessors.shape(1)!=256 || predecessors.dtype()!=mx::bfloat16 ||
       anchor_id.ndim()!=0 || anchor_id.dtype()!=mx::uint32 || count<1 || count>R-1)
        throw std::invalid_argument("invalid DFlash selector dimensions");
    static auto select=mx::fast::metal_kernel("tqllm_dflash_select",
        {"mapped","unary","gate","successors","predecessors","anchor"},{"tokens"},R"METAL(
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
uint lane=thread_position_in_threadgroup.x;
uint previous=anchor;
for(uint step=1;step<=COUNT;++step) {
 float maximum=-INFINITY;
 uint winner=0;
 for(uint candidate=0;candidate<16u;++candidate) {
  float sum=0.0f;
  for(uint block=0;block<2u;++block)for(uint j=0;j<4u;++j) {
   uint col=block*128u+lane*4u+j;
   float weighted=float(predecessors[previous*256u+col])*gate[step*256u+col];
   float product=successors[(step*16u+candidate)*256u+col]*weighted;
   sum=sum+product;
  }
  float score=simd_sum(sum)+unary[step*16u+candidate];
  if(score>maximum) {maximum=score;winner=candidate;}
 }
 previous=mapped[step*16u+winner];
 if(previous>=VOCAB) {
  if(lane==0)for(uint rest=step;rest<=COUNT;++rest)tokens[rest-1u]=VOCAB;
  return;
 }
 if(lane==0)tokens[step-1u]=previous;
}
)METAL");
    return select({mx::contiguous(mapped),mx::contiguous(unary),mx::contiguous(gate),
        mx::contiguous(successors),mx::contiguous(predecessors),anchor_id},
        {{count}},{mx::uint32},{32,1,1},{32,1,1},
        {{"COUNT",count},{"VOCAB",predecessors.shape(0)}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
}  // namespace tqllm::mlx_backend

#include <string>
#include "draft_head.hpp"
#include "prefill.hpp"
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <vector>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
constexpr int NT=2;
const char* kHeader="#include <metal_simdgroup_matrix>\n";
void frag_pack(DraftHead4& head) {
    const int N=head.codes.shape(0),K=head.codes.shape(1)*8,KB=K/64;
    const mx::array shifts8=mx::array({0u,4u,8u,12u,16u,20u,24u,28u},{8},mx::uint32);
    const mx::array lane_shifts=mx::array({0u,8u,16u,24u,4u,12u,20u,28u},{2,4},mx::uint32);
    std::vector<mx::array> ws,ss,bs;
    for(int r0=0;r0<N;r0+=16384) {
        const int rows=std::min(16384,N-r0),G=rows/16;
        auto q=mx::slice(head.codes,{r0,0},{r0+rows,K/8});
        auto codes=mx::reshape(mx::bitwise_and(mx::right_shift(mx::expand_dims(q,-1),shifts8),mx::array(15u)),{rows,K});
        auto c5=mx::reshape(codes,{G,NT,8,KB,8,8});
        std::vector<mx::array> lanes;
        for(int lane=0;lane<32;++lane) {
            const int qid=lane/4,fm=(qid&4)+((lane/2)%4),fn=(qid&2)*2+(lane%2)*2;
            auto x=mx::reshape(mx::slice(c5,{0,0,fn,0,0,fm},{G,NT,fn+2,KB,8,fm+1}),{G,NT,2,KB,2,4});
            x=mx::transpose(x,{0,1,3,4,2,5});
            x=mx::left_shift(x,lane_shifts);
            lanes.push_back(mx::sum(mx::reshape(x,{G,NT,KB,2,8}),-1));
        }
        auto w=mx::transpose(mx::stack(lanes,-2),{0,2,1,3,4});
        auto lay=[&](const mx::array& a) {
            auto part=mx::slice(a,{r0,0},{r0+rows,KB});
            return mx::reshape(mx::transpose(mx::reshape(mx::astype(part,mx::float16),{G,NT,8,KB}),{0,3,1,2}),{-1});
        };
        ws.push_back(mx::reshape(mx::astype(w,mx::uint32),{-1}));
        ss.push_back(lay(head.scales));bs.push_back(lay(head.biases));
        mx::eval(ws.back(),ss.back(),bs.back());
    }
    head.frag_codes=mx::concatenate(ws,0);head.frag_scales=mx::concatenate(ss,0);head.frag_biases=mx::concatenate(bs,0);
    mx::eval(*head.frag_codes,*head.frag_scales,*head.frag_biases);
}
mx::array frag_matmul(const DraftHead4& head,const mx::array& x) {
    const int N=head.codes.shape(0),K=head.codes.shape(1)*8,KB=K/64,m=x.shape(0);
    static auto kernel=mx::fast::metal_kernel("tqllm_head_frag8",{"W","S","Bi","XS","X"},{"out"},R"METAL(
const uint sg=simdgroup_index_in_threadgroup;
const uint lane=thread_index_in_simdgroup;
const uint g=threadgroup_position_in_grid.x*4+sg;
const uint qid=lane>>2;
const uint fm=(qid&4)+((lane>>1)&3);
const uint fn=(qid&2)*2+(lane&1)*2;
const uint KB=K/64;
simdgroup_float8x8 C[2];
for(uint t=0;t<2;++t)C[t]=simdgroup_float8x8(0);
for(uint kb=0;kb<KB;++kb) {
    uint2 w2[2];float2 s2[2],b2[2];
    simdgroup_float8x8 Ct[2];
    const float xs=XS[fm*KB+kb];
    for(uint t=0;t<2;++t) {
        ulong base=((ulong)g*KB+kb)*2+t;
        w2[t]=((const device uint2*)(W+base*64))[lane];
        s2[t]=float2(S[base*8+fn],S[base*8+fn+1]);
        b2[t]=float2(Bi[base*8+fn],Bi[base*8+fn+1]);
        Ct[t]=simdgroup_float8x8(0);
    }
    for(uint kk=0;kk<8;++kk) {
        simdgroup_half8x8 A;
        simdgroup_load(A,X+kb*64+kk*8,K);
        for(uint t=0;t<2;++t) {
            uint word=kk<4?w2[t].x:w2[t].y;
            uint byte=(word>>((kk&3)*8))&255u;
            simdgroup_half8x8 B;
            B.thread_elements()[0]=half(byte&15u);
            B.thread_elements()[1]=half(byte>>4);
            simdgroup_multiply_accumulate(Ct[t],A,B,Ct[t]);
        }
    }
    for(uint t=0;t<2;++t) {
        C[t].thread_elements()[0]+=Ct[t].thread_elements()[0]*s2[t].x+b2[t].x*xs;
        C[t].thread_elements()[1]+=Ct[t].thread_elements()[1]*s2[t].y+b2[t].y*xs;
    }
}
for(uint t=0;t<2;++t)simdgroup_store(C[t],out+(g*2+t)*8,N);
)METAL",kHeader);
    auto xp=mx::astype(x,mx::float16);
    if(m<8)xp=mx::concatenate({xp,mx::zeros({8-m,K},mx::float16)},0);
    auto xs=mx::sum(mx::reshape(mx::astype(xp,mx::float32),{8,KB,64}),-1);
    auto y=kernel({*head.frag_codes,*head.frag_scales,*head.frag_biases,xs,xp},{{8,N}},{mx::float32},
                  {N/(8*NT)*32,1,1},{128,1,1},{{"K",K},{"N",N}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
    return m<8?mx::slice(y,{0,0},{m,N}):y;
}
}
DraftHead4 draft_head4(const Matrix& original) {
    auto staged=mx::astype(stage_bf16(original),mx::float32);
    auto weights=mx::quantize(staged,64,4,"affine");
    mx::eval(weights);
    DraftHead4 head{weights[0],weights[1],weights[2],std::nullopt,std::nullopt,std::nullopt};
    const int rows=head.codes.shape(0),columns=head.codes.shape(1)*8;
    // default on (TQLLM_MLX_HEAD_FRAG=0 opts out): j578 draft 11.6 vs 16.2 ms/step without it
    const char* frag=std::getenv("TQLLM_MLX_HEAD_FRAG");
    if((!frag || std::string(frag)!="0") && rows%(16*4)==0 && columns%64==0)frag_pack(head);
    return head;
}
mx::array draft_head4_matmul(const DraftHead4& head,const mx::array& x) {
    if(x.ndim()!=2 || x.dtype()!=mx::float32 || x.shape(0)<1 || x.shape(1)!=8*head.codes.shape(1))
        throw std::invalid_argument("invalid proposal head input");
    const int m=x.shape(0);auto input=x;
    if(head.frag_codes && m<=8)return frag_matmul(head,x);
    // block-16 drafts: the 8-row fragment kernel twice (rows are independent) instead of the generic quantized matmul,
    // which doubled the draft (j692: 11.1 -> 21.8 ms)
    if(head.frag_codes && m<=16)
        return mx::concatenate({frag_matmul(head,mx::slice(x,{0,0},{8,x.shape(1)})),
                                frag_matmul(head,mx::slice(x,{8,0},{m,x.shape(1)}))},0);
    if(m>1 && m<32 && !std::getenv("TQLLM_MLX_DRAFT_HEAD_UNPAD"))
        input=mx::concatenate({input,mx::zeros({32-m,x.shape(1)},mx::float32)},0);
    auto result=mx::quantized_matmul(input,head.codes,head.scales,head.biases,true,64,4,"affine");
    return mx::astype(mx::slice(result,{0,0},{m,head.codes.shape(0)}),mx::float32);
}
}

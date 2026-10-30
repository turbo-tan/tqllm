#include <sstream>
#include "dflash.hpp"
#include "dflash_retrieve.hpp"
#include <mlx/compile.h>
#include "nax_affine.hpp"
#include "dequant.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <type_traits>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
mx::array bf(const mx::array& x) {return mx::astype(mx::astype(x,mx::bfloat16),mx::float32);}
mx::array rows(const mx::array& x,int a,int b) {
    auto starts=mx::Shape(x.ndim(),0),ends=x.shape();starts[0]=a;ends[0]=b;
    return mx::slice(x,starts,ends);
}
mx::array silu(const mx::array& x) {return mx::multiply(x,mx::sigmoid(x));}
}
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
    const int R=mapped.ndim()==2?mapped.shape(0):0;  // draft block rows (8, or 16 with TQLLM_MLX_DFLASH_BLOCK)
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
struct DFlash2::Impl {
    Qwen35& target;
    GgufFile file;
    int hidden,heads,kvheads,hd,block,native,group,window,rank,vocab,end=0;
    // rows of a draft of `count` tokens: the native block when it fits, else the full block (K / V scratch rows [block - rows, block))
    int draft_rows(int count) const {return native_on && count<native?native:block;}
    uint32_t mask;
    float eps,theta;
    std::vector<int> features;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,NaxAffine> nax;
    std::vector<std::vector<uint32_t>> candidates;
    bool device_only=false;                 // draft_device: leave the chosen ids on the GPU
    std::optional<mx::array> last_chosen;
    std::map<std::string,mx::array> plain;
    // ring (window > 0): K / V rows [0, block) hold the current draft block, rows [block, block + window) the last
    // `window` injected rows in ring order (attention is not causal, so order does not matter). While the window fills,
    // the valid keys are the leading block + filled rows, so attention reads only those; appends write in place
    struct Kv {std::optional<mx::array> k,v;int filled=0,head=0;
        // far-context retrieval (ring drafter, dflash_retrieve.hpp): every injected key / value row at its position, and
        // per 16-row chunk the sum of its keys (the look-ahead selection scores those: 1/16 of a cache scan)
        // kfull / vfull int8 codes with per-row FP16 scales kfs / vfs (dflash_far_quantize)
        std::optional<mx::array> kfull,vfull,kfs,vfs,ksum;};
    bool ring=false;
    static constexpr int kChunk=16,kRetrieve=32;   // retrieved chunks per draft (CUDA D1: 32 lifts 31K acceptance 45 -> 56 %, 64 equal)
    int far_cap=0;                                  // rows of kfull / vfull: whole chunks + one scratch chunk at the end
    // look-ahead: each retrieving draft selects the next draft's chunks per layer from its own anchor queries on a side GPU
    // stream and gathers + rotates their rows once there (onto the window's first positions as of this draft: the next
    // draft is at most a block further, so the keys stay within the trained distances); the next draft only attends them.
    // stream (evaluated at once, beside the verify), so the next draft only gathers; the first retrieving draft selects inline
    std::vector<std::optional<mx::array>> sel_next;   // per layer: rows_k, then rows_v at [L + l]
    bool far_filled=false;   // inject(): the recursive tail call must not refill the far caches
    void drop_selections() {  // pending look-ahead work finishes before its inputs go (reset / teardown)
        std::vector<mx::array> pending;
        for(auto& s:sel_next)if(s)pending.push_back(*s);
        if(!pending.empty())mx::eval(pending);
        sel_next.clear();
    }
    bool far_enabled=true,native_on=false;
    bool far_on() const {return ring && window>0 && far_enabled;}
    // grow the full caches to hold `rows` positions (+ a scratch chunk), x1.5 steps in whole chunks
    void ensure_far(int rows) {
        const int need=(rows+kChunk-1)/kChunk*kChunk+kChunk;
        if(far_cap>=need)return;
        const int cap=std::max(need,(far_cap+far_cap/2+kChunk-1)/kChunk*kChunk);
        for(auto& c:caches) {
            auto nk=mx::zeros({1,kvheads,cap,hd},mx::int8),nv=mx::zeros({1,kvheads,cap,hd},mx::int8);
            auto nks=mx::zeros({1,kvheads,cap},mx::float16),nvs=mx::zeros({1,kvheads,cap},mx::float16);
            auto ns=mx::zeros({1,kvheads,cap/kChunk,hd},mx::float32);
            if(c.kfull) {
                const int old=far_cap-kChunk;   // copy the filled chunks, not the old scratch chunk
                nk=mx::slice_update(nk,mx::slice(*c.kfull,{0,0,0,0},{1,kvheads,old,hd}),{0,0,0,0},{1,kvheads,old,hd});
                nv=mx::slice_update(nv,mx::slice(*c.vfull,{0,0,0,0},{1,kvheads,old,hd}),{0,0,0,0},{1,kvheads,old,hd});
                nks=mx::slice_update(nks,mx::slice(*c.kfs,{0,0,0},{1,kvheads,old}),{0,0,0},{1,kvheads,old});
                nvs=mx::slice_update(nvs,mx::slice(*c.vfs,{0,0,0},{1,kvheads,old}),{0,0,0},{1,kvheads,old});
                ns=mx::slice_update(ns,mx::slice(*c.ksum,{0,0,0,0},{1,kvheads,old/kChunk,hd}),{0,0,0,0},{1,kvheads,old/kChunk,hd});
            }
            c.kfull=nk;c.vfull=nv;c.kfs=nks;c.vfs=nvs;c.ksum=ns;
        }
        far_cap=cap;
    }
    // host-side fill of the full caches for rows [position, position + n) of a layer (prefill inject)
    void far_put_host(Kv& c,const mx::array& k,const mx::array& v,int position) {
        const int n=k.shape(2);
        const auto qk=dflash_far_quantize(k),qv=dflash_far_quantize(v);
        c.kfull=mx::slice_update(*c.kfull,qk.codes,{0,0,position,0},{1,kvheads,position+n,hd});
        c.vfull=mx::slice_update(*c.vfull,qv.codes,{0,0,position,0},{1,kvheads,position+n,hd});
        c.kfs=mx::slice_update(*c.kfs,qk.scales,{0,0,position},{1,kvheads,position+n});
        c.vfs=mx::slice_update(*c.vfs,qv.scales,{0,0,position},{1,kvheads,position+n});
        // chunk sums of the stored (BF16) keys: pad to whole chunks at the position's chunk offset, add in place
        const int c0=position/kChunk,lead=position-c0*kChunk,padded=(lead+n+kChunk-1)/kChunk*kChunk,nch=padded/kChunk;
        auto kf=mx::astype(mx::astype(k,mx::bfloat16),mx::float32);
        if(lead)kf=mx::concatenate({mx::zeros({1,kvheads,lead,hd},mx::float32),kf},2);
        if(padded>lead+n)kf=mx::concatenate({kf,mx::zeros({1,kvheads,padded-lead-n,hd},mx::float32)},2);
        auto sums=mx::sum(mx::reshape(kf,{1,kvheads,nch,kChunk,hd}),3);
        c.ksum=mx::slice_update(*c.ksum,mx::add(mx::slice(*c.ksum,{0,0,c0,0},{1,kvheads,c0+nch,hd}),sums),{0,0,c0,0},{1,kvheads,c0+nch,hd});
    }
    // draft-ahead: the ring's head / fill count live on the GPU once injects carry device counts; the host keeps a bound
    // (settled fill + block rows per inject not yet settled) to size the attention span
    std::optional<mx::array> head_dev,filled_dev;
    int fill_settled=0,fill_unsettled=0;
    double stage_ms[4]={0,0,0,0};int stage_calls=0;  // TQLLM_MLX_DRAFT_STAGES: layers, head, top-16, selector
    void to_device() {
        if(head_dev)return;
        head_dev=mx::array(caches[0].head,mx::int32);filled_dev=mx::array(caches[0].filled,mx::int32);
        fill_settled=caches[0].filled;fill_unsettled=0;
    }
    std::vector<Kv> caches;
    Impl(const std::string& path,Qwen35& engine):target(engine),file(gguf_open(path)) {
        if(file.kv.at("general.architecture").str!="dflash")throw std::invalid_argument("DFlash GGUF required");
        hidden=int(file.num("dflash.embedding_length"));heads=int(file.num("dflash.attention.head_count"));
        kvheads=int(file.num("dflash.attention.head_count_kv"));hd=int(file.num("dflash.attention.key_length"));
        block=int(file.num("dflash.block_size"));
        // TQLLM_MLX_DFLASH_BLOCK=n: draft blocks of up to n rows (block-16 pricing with a block-8 drafter, own block-16 drafters);
        // a draft of fewer than the file's native block rows runs at the native block (j924: its 7 drafts beat the head of a 16-draft)
        if(const char* b=std::getenv("TQLLM_MLX_DFLASH_BLOCK"))block=std::atoi(b);native=std::min(block,int(file.num("dflash.block_size")));group=int(file.num("dflash.conv_group_size"));
        window=int(file.num("dflash.attention.sliding_window"));rank=int(file.num("dflash.selector_rank"));
        mask=uint32_t(file.num("tokenizer.ggml.mask_token_id"));
        eps=float(file.num("dflash.attention.layer_norm_rms_epsilon"));theta=float(file.num("dflash.rope.freq_base"));
        vocab=target.vocabulary();
        for(double id:file.kv.at("dflash.target_layers").arr)features.push_back(int(id));
        const int layers=int(file.num("dflash.block_count"));
        if(hidden!=5120 || hd!=128 || heads!=32 || kvheads!=8 || (block!=8 && block!=16) || group!=16 || rank!=256 ||
           layers!=5 || mask>=uint32_t(vocab) || file.num("dflash.conv_kernel_size")!=2)
            throw std::invalid_argument("unsupported DFlash2 dimensions");
        for(const auto& [name,tensor]:file.tensors) {
            if(name=="selector_predecessor.weight" || name=="selector_successor.weight") {
                if(tensor.ne!=std::vector<uint64_t>{uint64_t(rank),uint64_t(vocab)})
                    throw std::invalid_argument("DFlash selector vocabulary mismatch");
                auto raw=file.read(tensor);std::vector<float> table(size_t(rank)*vocab);
                dequantize_host(tensor.type,raw.data(),table.data(),table.size());
                auto values=mx::astype(mx::array(table.data(),{vocab,rank}),mx::bfloat16);
                mx::eval(values);plain.emplace(name,std::move(values));
            } else if(tensor.type==GGML_NAX_A4_G64 || tensor.type==GGML_NAX_A4_G128 || tensor.type==GGML_NAX_A6_G64) {
                auto raw=file.read(tensor);
                const int bits=tensor.type==GGML_NAX_A6_G64?6:4,group=tensor.type==GGML_NAX_A4_G128?128:64;
                nax.emplace(name,nax_affine_load({reinterpret_cast<const uint8_t*>(raw.data()),raw.size()},
                                                 int(tensor.ne[1]),int(tensor.ne[0]),bits,group));
            } else if(tensor.ne.size()==2 && name.find(".weight")!=std::string::npos &&
                      tensor.ne[0]>128 && tensor.ne[1]>128) {
                auto w=load_matrix(file,name);
                std::visit([](const auto& v) {
                    using T=std::decay_t<decltype(v)>;
                    if constexpr(std::is_same_v<T,mx::array>)mx::eval(v);
                    else if constexpr(std::is_same_v<T,Tq3Weights>)mx::eval(v.planes,v.scales);
                    else if constexpr(std::is_same_v<T,Tq4Weights>)mx::eval(v.codes,v.scales);
                    else mx::eval(v.blocks);
                },w);
                matrices.emplace(name,std::move(w));
            } else {
                auto raw=file.read(tensor);auto value=mx::astype(load_plain_tensor(tensor,
                    {reinterpret_cast<const uint8_t*>(raw.data()),raw.size()}),mx::float32);
                mx::eval(value);plain.emplace(name,std::move(value));
            }
        }
        caches.resize(layers);target.feature_layers(features);
        // TQLLM_MLX_DFLASH_RING=0: concatenate the window every inject and draft (three window copies per step)
        ring=window>0 && (!std::getenv("TQLLM_MLX_DFLASH_RING") || std::string(std::getenv("TQLLM_MLX_DFLASH_RING"))!="0");
    }
    mx::array mat(const std::string& name,const mx::array& x) {
        if(auto found=nax.find(name);found!=nax.end())return nax_affine_matmul(found->second,x,true);
        return bf(linear(matrices.at(name),x));
    }
    mx::array norm(const std::string& name,const mx::array& x) {return bf(mx::fast::rms_norm(x,plain.at(name),eps));}
    // FFN gate, up and SwiGLU: one matrix-unit launch for a draft block (<= 16 rows) of a matching 4-bit pair
    mx::array swiglu(const std::string& p,const mx::array& x) {
        auto g=nax.find(p+"ffn_gate.weight"),u=nax.find(p+"ffn_up.weight");
        if(x.ndim()==2 && x.shape(0)<=16 && g!=nax.end() && u!=nax.end() && g->second.bits==4 && u->second.bits==4 &&
           !g->second.q6 && !u->second.q6 && g->second.tq4==u->second.tq4 &&
           g->second.rows==u->second.rows && g->second.columns==u->second.columns && g->second.group==u->second.group)
            return nax_affine_swiglu(g->second,u->second,x);
        return bf(mx::multiply(bf(silu(mat(p+"ffn_gate.weight",x))),mat(p+"ffn_up.weight",x)));
    }
    mx::array keys(const std::string& p,const mx::array& x,int position) {
        auto k=mx::reshape(mat(p+"attn_k.weight",x),{1,x.shape(0),kvheads,hd});
        k=mx::transpose(norm(p+"attn_k_norm.weight",k),{0,2,1,3});
        return bf(mx::fast::rope(k,hd,false,theta,1.0f,position));
    }
    mx::array values(const std::string& p,const mx::array& x) {
        return mx::transpose(mx::reshape(mat(p+"attn_v.weight",x),{1,x.shape(0),kvheads,hd}),{0,2,1,3});
    }
    // TQLLM_MLX_DFLASH_COMPILE=1: the ring draft's layers as one mx::compile'd graph (traced once per mask state; the
    // position enters as an array, so the trace is reused at every step): fewer host nodes to build and encode per step
    bool compile_draft=std::getenv("TQLLM_MLX_DFLASH_COMPILE") && std::string(std::getenv("TQLLM_MLX_DFLASH_COMPILE"))!="0";
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> compiled_layers[2];   // [0] full block, [1] native block
    // in: x [block, hidden], position (int32 scalar), [mask over the span], then k, v per layer; attention reads the
    // leading `span` rows (block + filled); out: x, then the updated k, v
    std::vector<mx::array> ring_layers(const std::vector<mx::array>& in,int span,bool masked,bool retr=false,int scored=0,bool ahead=false,int rows=0) {
        auto x=in[0];const auto& position=in[1];size_t at=2;
        if(rows<=0)rows=block;
        const int off=block-rows;   // the draft rows sit just below the ring, so the attended span stays contiguous
        std::optional<mx::array> mask;if(masked)mask=in[at++];
        std::vector<mx::array> out{x};
        const int cap=window+block,per=retr?(ahead?4:7):2;   // retr: k, v + (rows_k, rows_v) ahead or (kfull, vfull, kfs, vfs, ksum) inline
        std::vector<mx::array> anchors;   // retr: each layer's anchor query rows, for the next draft's selection
        // the ring holds [position - window, position); retrieved chunks land on the window's first R positions, beside
        // the ring rows there, so every key stays within the distances the drafter was trained on
        const auto far_start=mx::subtract(position,mx::array(window,mx::int32));
        const auto far_to=mx::add(far_start,mx::array(kRetrieve*kChunk,mx::int32));
        for(int l=0;l<int(caches.size());++l) {
            const std::string p="blk."+std::to_string(l)+".";
            auto kc=in[at+per*l],vc=in[at+per*l+1];
            auto residual=x,n=norm(p+"attn_norm.weight",x),dynamic=mat(p+"attn_conv_proj.weight",n);
            auto input=dflash_conv(n,dynamic,plain.at(p+"attn_conv_base"),group,0);
            auto q=mx::reshape(mat(p+"attn_q.weight",input),{1,rows,heads,hd});
            q=mx::transpose(norm(p+"attn_q_norm.weight",q),{0,2,1,3});
            q=bf(mx::fast::rope(q,hd,false,theta,1.0f,position));
            auto k=mx::reshape(mat(p+"attn_k.weight",input),{1,rows,kvheads,hd});
            k=bf(mx::fast::rope(mx::transpose(norm(p+"attn_k_norm.weight",k),{0,2,1,3}),hd,false,theta,1.0f,position));
            auto v=values(p,input);
            kc=mx::slice_update(kc,mx::astype(k,mx::bfloat16),{0,0,off,0},{1,kvheads,block,hd});vc=mx::slice_update(vc,mx::astype(v,mx::bfloat16),{0,0,off,0},{1,kvheads,block,hd});
            const bool whole=off==0 && span>=cap;
            auto ks=whole?kc:mx::slice(kc,{0,0,off,0},{1,kvheads,off+span,hd}),vs=whole?vc:mx::slice(vc,{0,0,off,0},{1,kvheads,off+span,hd});
            std::optional<mx::array> amask=mask;
            if(retr) {
                // inline (the first retrieving draft): selection from the chunk key sums (int8 codes cannot be scanned,
                // j1167), gather with the scales
                const auto r=ahead?DflashRetrieval{in[at+per*l+2],in[at+per*l+3],mx::array(0),kRetrieve*kChunk}:
                                   dflash_retrieve_gather(dflash_retrieve_select_sums(q,in[at+per*l+6],far_start,scored,kRetrieve),
                                                          in[at+per*l+2],in[at+per*l+3],far_to,theta,in[at+per*l+4],in[at+per*l+5]);
                anchors.push_back(mx::slice(q,{0,0,0,0},{1,heads,1,hd}));
                ks=mx::concatenate({r.k,ks},2);vs=mx::concatenate({r.v,vs},2);
                // the retrieved rows are always visible: a mask only while the ring fills (an all-true mask kept the
                // attention off the unmasked kernel at every retrieving draft)
                if(mask)amask=mx::concatenate({mx::full({1,1,1,r.retrieved},true,mx::bool_),*mask},3);
            }
            // BF16 K / V / q (their values are BF16 already: bf()); the FP32 kernel read twice the bytes (j1118)
            const auto q16=mx::astype(q,mx::bfloat16);
            auto attention=amask?mx::fast::scaled_dot_product_attention(q16,ks,vs,1.0f/std::sqrt(float(hd)),"array",*amask):
                mx::fast::scaled_dot_product_attention(q16,ks,vs,1.0f/std::sqrt(float(hd)));
            auto flattened=bf(mx::reshape(mx::transpose(attention,{0,2,1,3}),{rows,heads*hd}));
            x=dflash_conv(mat(p+"attn_output.weight",flattened),dynamic,plain.at(p+"attn_conv_base"),group,1,residual);
            residual=x;n=norm(p+"ffn_norm.weight",x);dynamic=mat(p+"ffn_conv_proj.weight",n);
            input=dflash_conv(n,dynamic,plain.at(p+"ffn_conv_base"),group,0);
            auto ff=mat(p+"ffn_down.weight",swiglu(p,input));
            x=dflash_conv(ff,dynamic,plain.at(p+"ffn_conv_base"),group,1,residual);
            out.push_back(kc);out.push_back(vc);
        }
        out[0]=x;
        out.insert(out.end(),anchors.begin(),anchors.end());
        return out;
    }
    // the ring layers over the leading `span` rows; compiled only in the fixed-shape steady state (full window, no mask)
    mx::array run_ring_layers(const mx::array& x,const mx::array& position,int span,const std::optional<mx::array>& mask,int far_rows=0) {
        std::vector<mx::array> in{x,position};
        if(mask)in.push_back(*mask);
        // retrieval needs kRetrieve whole chunks below the window (far_rows: the host's lower bound of the position);
        // chunks up to the host's upper bound are scored, those not below the window masked on the device
        const bool retr=far_on() && caches[0].kfull && far_rows>=window+kRetrieve*kChunk;
        const int scored=retr?std::min(far_cap/kChunk-1,(far_rows+2*block-window)/kChunk+1):0;
        const bool ahead=retr && !sel_next.empty() && sel_next[0];
        for(size_t l=0;l<caches.size();++l) {
            const auto& c=caches[l];
            in.push_back(*c.k);in.push_back(*c.v);
            if(ahead) {in.push_back(*sel_next[l]);in.push_back(*sel_next[caches.size()+l]);}
            else if(retr) {in.push_back(*c.kfull);in.push_back(*c.vfull);in.push_back(*c.kfs);in.push_back(*c.vfs);in.push_back(*c.ksum);}
        }
        const int rows=x.shape(0),slot=rows==block?0:1;
        const bool fixed=span==window+rows && !mask && !retr;
        if(compile_draft && fixed && !compiled_layers[slot])
            compiled_layers[slot]=mx::compile([this,rows](const std::vector<mx::array>& a) {return ring_layers(a,window+rows,false,false,0,false,rows);});
        auto out=compile_draft && fixed?compiled_layers[slot](in):ring_layers(in,span,mask.has_value(),retr,scored,ahead,rows);
        const int L=int(caches.size());
        for(int l=0;l<L;++l) {caches[l].k=out[1+2*l];caches[l].v=out[2+2*l];}
        if(retr) {
            std::vector<mx::array> rk,rv;
            {   // on the draft's own stream: a second GPU queue cost more in cross-queue waits (j1112: B 168 gaps of 0.1-0.5 ms
                // per 3 s, idle 1.5 -> 3.5 %) than the overlap gave
                const auto far_start=mx::subtract(position,mx::array(window,mx::int32));
                const auto far_to=mx::add(far_start,mx::array(kRetrieve*kChunk,mx::int32));
                for(int l=0;l<L;++l) {
                    const auto sel=dflash_retrieve_select_sums(out[1+2*L+l],*caches[l].ksum,far_start,scored,kRetrieve);
                    const auto r=dflash_retrieve_gather(sel,*caches[l].kfull,*caches[l].vfull,far_to,theta,caches[l].kfs,caches[l].vfs);
                    rk.push_back(r.k);rv.push_back(r.v);
                }
            }
            std::vector<mx::array> next(rk);next.insert(next.end(),rv.begin(),rv.end());
            mx::async_eval(next);
            sel_next.assign(next.begin(),next.end());
        } else sel_next.clear();
        return out[0];
    }
};
DFlash2::DFlash2(const std::string& path,Qwen35& target):impl_(std::make_unique<Impl>(path,target)) {}
DFlash2::~DFlash2() {if(impl_)impl_->drop_selections();}
const std::vector<std::vector<uint32_t>>& DFlash2::candidates() const {return impl_->candidates;}
const std::vector<int>& DFlash2::feature_layers() const {return impl_->features;}
void DFlash2::reset() {
    impl_->end=0;for(auto& c:impl_->caches)c=Impl::Kv{};impl_->far_cap=0;impl_->drop_selections();
    impl_->head_dev.reset();impl_->filled_dev.reset();impl_->fill_settled=impl_->fill_unsettled=0;
}
void DFlash2::inject(const mx::array& features,int position) {
    auto& i=*impl_;const int n=features.shape(0);
    if(i.head_dev)throw std::logic_error("DFlash inject after device injects: the host ring position is stale");
    if(position!=i.end || features.ndim()!=2 || features.shape(1)!=int(i.features.size())*i.hidden || n<1)
        throw std::invalid_argument("invalid DFlash feature injection");
    // far-KV: every row's key / value into the full caches (rows are independent; slabs bound the temporaries)
    if(i.far_on() && !i.far_filled) {
        i.ensure_far(position+n+i.block);
        for(int off=0;off<n;off+=2048) {
            const int m=std::min(2048,n-off);
            auto slab=mx::slice(features,{off,0},{off+m,features.shape(1)});
            auto enc=i.norm("enc.output_norm.weight",i.mat("fc.weight",slab));
            for(int l=0;l<int(i.caches.size());++l) {
                const std::string p="blk."+std::to_string(l)+".";
                i.far_put_host(i.caches[l],i.keys(p,enc,position+off),mx::astype(i.values(p,enc),mx::bfloat16),position+off);
            }
            std::vector<mx::array> live;for(auto& c:i.caches) {live.push_back(*c.kfull);live.push_back(*c.vfull);live.push_back(*c.kfs);live.push_back(*c.vfs);live.push_back(*c.ksum);}
            mx::eval(live);
        }
    }
    // a sliding window keeps only the last `window` rows of an inject this long: encode just those (rows are independent)
    if(i.window>0 && n>i.window) {
        const int skip=n-i.window;
        i.end+=skip;
        i.far_filled=true;   // the full caches already hold every row of this inject
        inject(mx::slice(features,{skip,0},{n,features.shape(1)}),position+skip);
        i.far_filled=false;
        return;
    }
    auto encoded=i.norm("enc.output_norm.weight",i.mat("fc.weight",features));
    std::vector<mx::array> live;
    bool trimmed=false;
    for(int l=0;l<int(i.caches.size());++l) {
        const std::string p="blk."+std::to_string(l)+".";auto& c=i.caches[l];
        auto k=i.keys(p,encoded,position),v=i.values(p,encoded);
        if(i.ring) {
            const int w=i.window,cap=w+i.block,b=i.block;
            k=mx::astype(k,mx::bfloat16);v=mx::astype(v,mx::bfloat16);
            if(!c.k) {c.k=mx::zeros({1,i.kvheads,cap,i.hd},mx::bfloat16);c.v=mx::zeros({1,i.kvheads,cap,i.hd},mx::bfloat16);}
            auto put=[&](int from,int count,int at) {  // ring slot `at` is buffer row b + at
                c.k=mx::slice_update(*c.k,mx::slice(k,{0,0,from,0},{1,i.kvheads,from+count,i.hd}),{0,0,b+at,0},{1,i.kvheads,b+at+count,i.hd});
                c.v=mx::slice_update(*c.v,mx::slice(v,{0,0,from,0},{1,i.kvheads,from+count,i.hd}),{0,0,b+at,0},{1,i.kvheads,b+at+count,i.hd});
            };
            if(n>=w) {put(n-w,w,0);c.head=0;c.filled=w;}
            else {
                const int a=std::min(n,w-c.head);
                put(0,a,c.head);if(n>a)put(a,n-a,0);
                c.head=(c.head+n)%w;c.filled=std::min(w,c.filled+n);
            }
            live.push_back(*c.k);live.push_back(*c.v);
            trimmed=trimmed || c.filled==w;  // full window: evaluate the inject here, as the concatenate path does (j780)
            continue;
        }
        k=mx::astype(k,mx::bfloat16);v=mx::astype(v,mx::bfloat16);
        c.k=c.k?mx::concatenate({*c.k,k},2):k;c.v=c.v?mx::concatenate({*c.v,v},2):v;
        if(i.window>0 && c.k->shape(2)>i.window) {
            const int begin=c.k->shape(2)-i.window;trimmed=true;
            c.k=mx::contiguous(mx::slice(*c.k,{0,0,begin,0},{1,i.kvheads,c.k->shape(2),i.hd}));
            c.v=mx::contiguous(mx::slice(*c.v,{0,0,begin,0},{1,i.kvheads,c.v->shape(2),i.hd}));
        }
        live.push_back(*c.k);live.push_back(*c.v);
    }
    // deferred into the next draft while the window grows (short prompts +0-3 %, j583); once the window is trimmed
    // the deferred copies cost 16 ms/step (16K, j582), so evaluate them here
    if(trimmed)mx::eval(live);
    i.end+=n;
}
mx::array DFlash2::draft_device(uint32_t anchor,int position,int count) {
    impl_->device_only=true;
    try {draft(anchor,position,count);} catch(...) {impl_->device_only=false;throw;}
    impl_->device_only=false;
    return *impl_->last_chosen;
}
void DFlash2::inject_device(const mx::array& features,const mx::array& count,const mx::array& position) {
    auto& i=*impl_;const int n=features.shape(0);
    if(!i.ring || !i.caches[0].k || features.ndim()!=2 || features.shape(1)!=int(i.features.size())*i.hidden || n<1 || n>i.block)
        throw std::invalid_argument("invalid DFlash device injection (needs the ring drafter after a host prefill inject)");
    i.to_device();
    auto encoded=i.norm("enc.output_norm.weight",i.mat("fc.weight",features));
    // row j goes to buffer row block + (head + j) % window while j < count, else to block row j (scratch, rewritten by
    // every draft): one fixed-shape scatter, no readback of the count
    auto j=mx::arange(n,mx::int32);
    auto slot=mx::where(mx::less(j,count),mx::add(mx::remainder(mx::add(*i.head_dev,j),mx::array(i.window,mx::int32)),mx::array(i.block,mx::int32)),j);
    auto index=mx::broadcast_to(mx::reshape(slot,{1,1,n,1}),{1,i.kvheads,n,i.hd});
    // far context: rows j < count at their positions, the rest into the scratch chunk at the end
    std::optional<mx::array> far_index,far_scale,far_chunk;
    if(i.far_on() && i.caches[0].kfull) {
        i.ensure_far(i.target.position()+2*i.block);
        auto pos=mx::where(mx::less(j,count),mx::add(position,j),mx::array(i.far_cap-1,mx::int32));
        far_index=mx::broadcast_to(mx::reshape(pos,{1,1,n,1}),{1,i.kvheads,n,i.hd});
        far_scale=mx::broadcast_to(mx::reshape(pos,{1,1,n}),{1,i.kvheads,n});
        far_chunk=mx::broadcast_to(mx::reshape(mx::floor_divide(pos,mx::array(Impl::kChunk,mx::int32)),{1,1,n,1}),{1,i.kvheads,n,i.hd});
    }
    for(int l=0;l<int(i.caches.size());++l) {
        const std::string p="blk."+std::to_string(l)+".";auto& c=i.caches[l];
        auto k=mx::reshape(i.mat(p+"attn_k.weight",encoded),{1,n,i.kvheads,i.hd});
        k=bf(mx::fast::rope(mx::transpose(i.norm(p+"attn_k_norm.weight",k),{0,2,1,3}),i.hd,false,i.theta,1.0f,position));
        auto v=i.values(p,encoded);
        c.k=mx::put_along_axis(*c.k,index,mx::astype(k,mx::bfloat16),2);c.v=mx::put_along_axis(*c.v,index,mx::astype(v,mx::bfloat16),2);
        if(far_index) {
            const auto qk=dflash_far_quantize(k),qv=dflash_far_quantize(v);
            c.kfull=mx::put_along_axis(*c.kfull,*far_index,qk.codes,2);
            c.vfull=mx::put_along_axis(*c.vfull,*far_index,qv.codes,2);
            c.kfs=mx::put_along_axis(*c.kfs,*far_scale,qk.scales,2);
            c.vfs=mx::put_along_axis(*c.vfs,*far_scale,qv.scales,2);
            c.ksum=mx::scatter_add_axis(*c.ksum,*far_chunk,mx::astype(mx::astype(k,mx::bfloat16),mx::float32),2);
        }
    }
    i.head_dev=mx::remainder(mx::add(*i.head_dev,count),mx::array(i.window,mx::int32));
    i.filled_dev=mx::minimum(mx::add(*i.filled_dev,count),mx::array(i.window,mx::int32));
    ++i.fill_unsettled;
}
void DFlash2::far_context(bool enabled) {impl_->far_enabled=enabled;}
void DFlash2::native_block(bool on) {impl_->native_on=on;}
size_t DFlash2::far_row_bytes() const {
    const auto& i=*impl_;
    if(!i.ring || i.window<=0)return 0;
    return i.caches.size()*size_t(i.kvheads)*(size_t(i.hd)*(1+1)+2*2+size_t(i.hd)*4/Impl::kChunk);   // int8 K, V, FP16 scales, chunk key sums
}
void DFlash2::reserve_far(int rows) {
    auto& i=*impl_;
    if(i.far_on())i.ensure_far(rows+2*i.block);
}
std::vector<double> DFlash2::stage_ms_per_call() const {
    const auto& i=*impl_;std::vector<double> out;
    for(double v:i.stage_ms)out.push_back(i.stage_calls?v/i.stage_calls:0.0);
    return out;
}
void DFlash2::settle_device_inject(int rows) {
    auto& i=*impl_;
    if(i.fill_unsettled<1 || rows<1 || rows>i.block)throw std::invalid_argument("invalid DFlash inject settlement");
    --i.fill_unsettled;i.fill_settled=std::min(i.window,i.fill_settled+rows);
}
mx::array DFlash2::draft_device_at(const mx::array& anchor,const mx::array& position,int count) {
    auto& i=*impl_;
    if(!i.ring || !i.caches[0].k || count<1 || count>=i.block || anchor.dtype()!=mx::uint32 || anchor.ndim()!=0)
        throw std::invalid_argument("invalid DFlash device draft request");
    i.to_device();
    const int rows=i.draft_rows(count);
    auto ids=mx::concatenate({mx::reshape(anchor,{1}),mx::full({rows-1},i.mask,mx::uint32)},0);
    auto x=i.target.embed_device(ids);
    const int bound=std::min(i.window,i.fill_settled+i.block*i.fill_unsettled),span=rows+bound;
    std::optional<mx::array> mask;
    if(bound>i.fill_settled)  // the GPU's fill is somewhere in (settled, bound]: mask the rows past it
        mask=mx::reshape(mx::less(mx::arange(span,mx::int32),mx::add(*i.filled_dev,mx::array(rows,mx::int32))),{1,1,1,span});
    // TQLLM_MLX_DRAFT_STAGES=1 (with TQLLM_MLX_DRAFT_BENCH): evaluate after each stage and add its ms to stage_ms
    static const bool stages=std::getenv("TQLLM_MLX_DRAFT_STAGES")!=nullptr;
    auto tick=std::chrono::steady_clock::now();
    auto lap=[&](int k,const std::vector<mx::array>& a) {
        if(!stages)return;
        mx::eval(a);const auto now=std::chrono::steady_clock::now();
        i.stage_ms[k]+=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;
    };
    if(stages)mx::eval(x,position);tick=std::chrono::steady_clock::now();
    x=i.run_ring_layers(x,position,span,mask,i.far_on()?i.target.position():0);
    lap(0,{x});
    auto normalized=i.norm("output_norm.weight",x);
    auto logits=i.target.shared_draft_head(normalized);
    lap(1,{logits});
    auto top=dflash_top16(logits);auto selected=top.first,unary=top.second;
    auto mapped=mx::take(i.target.draft_ids(),selected,0);
    lap(2,{selected,unary,mapped});
    auto gate=i.mat("selector_hidden.weight",normalized);
    auto successors=mx::astype(mx::take(i.plain.at("selector_successor.weight"),mapped,0),mx::float32);
    auto chosen=dflash_select(mapped,mx::astype(unary,mx::float32),gate,successors,i.plain.at("selector_predecessor.weight"),anchor,count);
    lap(3,{chosen});
    if(stages)++i.stage_calls;
    return chosen;
}
std::vector<uint32_t> DFlash2::draft(uint32_t anchor,int position,int count) {
    auto& i=*impl_;
    if(i.head_dev)throw std::logic_error("DFlash host draft after device injects: use draft_device_at");
    if(position!=i.end || count<1 || count>=i.block || anchor>=uint32_t(i.vocab))
        throw std::invalid_argument("invalid DFlash draft request");
    const int rows=i.draft_rows(count);
    std::vector<uint32_t> tokens(rows,i.mask);tokens[0]=anchor;
    auto x=i.target.embed_tokens(tokens);
    if(i.ring && i.caches[0].k)x=i.run_ring_layers(x,mx::array(position,mx::int32),rows+i.caches[0].filled,std::nullopt);
    else
    for(int l=0;l<int(i.caches.size());++l) {
        const std::string p="blk."+std::to_string(l)+".";
        auto residual=x,n=i.norm(p+"attn_norm.weight",x),dynamic=i.mat(p+"attn_conv_proj.weight",n);
        auto input=dflash_conv(n,dynamic,i.plain.at(p+"attn_conv_base"),i.group,0);
        auto q=mx::reshape(i.mat(p+"attn_q.weight",input),{1,rows,i.heads,i.hd});
        q=mx::transpose(i.norm(p+"attn_q_norm.weight",q),{0,2,1,3});
        q=bf(mx::fast::rope(q,i.hd,false,i.theta,1.0f,position));
        auto k=i.keys(p,input,position),v=i.values(p,input);auto& c=i.caches[l];
        k=mx::astype(k,mx::bfloat16);v=mx::astype(v,mx::bfloat16);
        if(c.k) {k=mx::concatenate({*c.k,k},2);v=mx::concatenate({*c.v,v},2);}
        auto attention=mx::fast::scaled_dot_product_attention(mx::astype(q,mx::bfloat16),k,v,1.0f/std::sqrt(float(i.hd)));
        auto flattened=bf(mx::reshape(mx::transpose(attention,{0,2,1,3}),{rows,i.heads*i.hd}));
        x=dflash_conv(i.mat(p+"attn_output.weight",flattened),dynamic,i.plain.at(p+"attn_conv_base"),i.group,1,residual);
        residual=x;n=i.norm(p+"ffn_norm.weight",x);dynamic=i.mat(p+"ffn_conv_proj.weight",n);
        input=dflash_conv(n,dynamic,i.plain.at(p+"ffn_conv_base"),i.group,0);
        auto ff=i.mat(p+"ffn_down.weight",i.swiglu(p,input));
        x=dflash_conv(ff,dynamic,i.plain.at(p+"ffn_conv_base"),i.group,1,residual);
    }
    auto normalized=i.norm("output_norm.weight",x);
    auto logits=i.target.shared_draft_head(normalized);
    auto top=dflash_top16(logits);auto selected=top.first,unary=top.second;
    auto mapped=mx::take(i.target.draft_ids(),selected,0);
    if(std::getenv("TQLLM_MLX_DFLASH_RANKSTATS")) {
        auto ids=mx::contiguous(mx::astype(mapped,mx::uint32));auto score=mx::contiguous(mx::astype(unary,mx::float32));mx::eval(ids,score);
        i.candidates.assign(count,{});
        for(int n=1;n<=count;++n) {
            std::vector<int> order(16);for(int j=0;j<16;++j)order[j]=j;
            const float* sc=score.data<float>()+n*16;
            std::sort(order.begin(),order.end(),[&](int a,int b){return sc[a]>sc[b];});
            for(int j:order)i.candidates[n-1].push_back(ids.data<uint32_t>()[n*16+j]);
        }
    }
    auto gate=i.mat("selector_hidden.weight",normalized);
    auto successors=mx::astype(mx::take(i.plain.at("selector_successor.weight"),mapped,0),mx::float32);
    auto chosen=dflash_select(mapped,mx::astype(unary,mx::float32),gate,successors,
        i.plain.at("selector_predecessor.weight"),anchor,count);
    if(i.device_only) {i.last_chosen=chosen;return {};}
    mx::eval(chosen);
    std::vector<uint32_t> result(chosen.data<uint32_t>(),chosen.data<uint32_t>()+count);
    if(std::any_of(result.begin(),result.end(),[&](uint32_t id){return id>=uint32_t(i.vocab);}))
        throw std::runtime_error("DFlash selector token outside vocabulary");
    return result;
}
}

#include <sstream>
#include "dflash.hpp"
#include "tq3_tiled.hpp"
#include "mma8.hpp"
#include "nax_affine.hpp"
#include "../dequant.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
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
mx::array dflash_conv(const mx::array& x,const mx::array& dynamic,const mx::array& base,int group,int side) {
    if(x.ndim()!=2 || x.dtype()!=mx::float32 || group<1 || x.shape(1)%group || side<0 || side>1 ||
       dynamic.shape()!=mx::Shape{x.shape(0),4*x.shape(1)/group} || base.size()!=size_t(4*x.shape(1)))
        throw std::invalid_argument("invalid DFlash convolution shape");
    const int m=x.shape(0),h=x.shape(1),g=h/group;auto result=mx::zeros(x.shape());
    auto bases=mx::reshape(base,{4,h});
    for(int tap=0;tap<2;++tap) {
        const int c=2*side+tap;
        auto weights=mx::slice(dynamic,{0,c*g},{m,(c+1)*g});
        weights=mx::add(mx::repeat(weights,group,1),rows(bases,c,c+1));
        auto input=tap?mx::concatenate({mx::zeros({1,h}),rows(x,0,m-1)},0):x;
        result=mx::add(result,mx::multiply(weights,input));
    }
    return bf(result);
}

mx::array dflash_top16_sharded(const mx::array& logits) {
    if(logits.ndim()!=2 || logits.dtype()!=mx::float32 || logits.shape(0)<1 || logits.shape(1)<16)
        throw std::invalid_argument("invalid DFlash logits for sharded top16");
    const int rows=logits.shape(0),vocab=logits.shape(1),shards=8,keep=16;
    static auto shard=mx::fast::metal_kernel("tqllm_dflash_top16_shards", {"logits"}, {"ids","values"}, R"METAL(
uint lane=thread_position_in_threadgroup.x;
uint row=threadgroup_position_in_grid.y;
uint shard=threadgroup_position_in_grid.z;
constexpr uint K=16u;
constexpr uint Threads=256u;
threadgroup float scores[Threads*K];
threadgroup uint tokens[Threads*K];
for(uint r=0;r<K;++r) {scores[lane*K+r]=-INFINITY;tokens[lane*K+r]=0xffffffffu;}
uint span=(uint(VOCAB)+uint(SHARDS)-1u)/uint(SHARDS);
uint begin=shard*span,end=min(begin+span,uint(VOCAB));
for(uint tok=begin+lane;tok<end;tok+=Threads) {
 float value=logits[row*uint(VOCAB)+tok];
 if(value>scores[lane*K+K-1u] || (value==scores[lane*K+K-1u] && tok<tokens[lane*K+K-1u])) {
  uint pos=K-1u;
  while(pos>0u && (value>scores[lane*K+pos-1u] || (value==scores[lane*K+pos-1u] && tok<tokens[lane*K+pos-1u]))) {
   scores[lane*K+pos]=scores[lane*K+pos-1u];
   tokens[lane*K+pos]=tokens[lane*K+pos-1u];
   --pos;
  }
  scores[lane*K+pos]=value;tokens[lane*K+pos]=tok;
 }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane==0u) {
 for(uint rank=0u;rank<K;++rank) {
  float best=-INFINITY;uint best_id=0xffffffffu,best_index=0u;
  for(uint i=0u;i<Threads*K;++i) {
   float value=scores[i];uint tok=tokens[i];
   if(value>best || (value==best && tok<best_id)) {best=value;best_id=tok;best_index=i;}
  }
  uint out=(row*uint(SHARDS)+shard)*K+rank;
  ids[out]=best_id;values[out]=best;scores[best_index]=-INFINITY;tokens[best_index]=0xffffffffu;
 }
}
)METAL");
    static auto merge=mx::fast::metal_kernel("tqllm_dflash_top16_merge", {"ids_in","values_in"}, {"ids"}, R"METAL(
uint lane=thread_position_in_threadgroup.x;
uint row=threadgroup_position_in_grid.y;
constexpr uint K=16u;
constexpr uint Total=128u;
threadgroup float scores[Total];
threadgroup uint tokens[Total];
if(lane<Total) {scores[lane]=values_in[row*Total+lane];tokens[lane]=ids_in[row*Total+lane];}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane==0u) {
 for(uint rank=0u;rank<K;++rank) {
  float best=-INFINITY;uint best_id=0xffffffffu,best_index=0u;
  for(uint i=0u;i<Total;++i) {
   float value=scores[i];uint tok=tokens[i];
   if(value>best || (value==best && tok<best_id)) {best=value;best_id=tok;best_index=i;}
  }
  ids[row*K+rank]=best_id;scores[best_index]=-INFINITY;tokens[best_index]=0xffffffffu;
 }
}
)METAL");
    auto partial=shard({logits}, {{rows,shards*keep},{rows,shards*keep}}, {mx::uint32,mx::float32},
        {256,uint32_t(rows),uint32_t(shards)}, {256,1,1}, {{"VOCAB",vocab},{"SHARDS",shards}}, std::nullopt, false, mx::Device(mx::Device::gpu));
    return merge({partial[0],partial[1]}, {{rows,keep}}, {mx::uint32}, {256,uint32_t(rows),1}, {256,1,1}, {}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

mx::array dflash_select(const mx::array& mapped,const mx::array& unary,const mx::array& gate,
    const mx::array& successors,const mx::array& predecessors,uint32_t anchor,int count) {
    if(mapped.shape()!=mx::Shape{8,16} || mapped.dtype()!=mx::uint32 ||
       unary.shape()!=mapped.shape() || unary.dtype()!=mx::float32 ||
       gate.shape()!=mx::Shape{8,256} || gate.dtype()!=mx::float32 ||
       successors.shape()!=mx::Shape{8,16,256} || successors.dtype()!=mx::float32 ||
       predecessors.ndim()!=2 || predecessors.shape(1)!=256 || predecessors.dtype()!=mx::bfloat16 ||
       anchor>=uint32_t(predecessors.shape(0)) || count<1 || count>7)
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
        mx::contiguous(successors),mx::contiguous(predecessors),mx::array(anchor)},
        {{count}},{mx::uint32},{32,1,1},{32,1,1},
        {{"COUNT",count},{"VOCAB",predecessors.shape(0)}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
struct DFlash2::Impl {
    Qwen35& target;
    GgufFile file;
    int hidden,heads,kvheads,hd,block,group,window,rank,vocab,end=0;
    uint32_t mask;
    float eps,theta;
    std::vector<int> features;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,NaxAffine> nax;
    std::vector<std::vector<uint32_t>> candidates;
    std::map<std::string,TqTiledWeights> tiled_ffn;
    std::map<std::string,Mma8Weights> mma8_attn;
    std::optional<std::pair<mx::array,mx::array>> rotation_half;
    std::optional<std::pair<mx::array,mx::array>> rotation;
    std::map<std::string,mx::array> plain;
    struct Kv {std::optional<mx::array> k,v;};
    std::vector<Kv> caches;
    Impl(const std::string& path,Qwen35& engine):target(engine),file(gguf_open(path)) {
        if(file.kv.at("general.architecture").str!="dflash")throw std::invalid_argument("DFlash GGUF required");
        hidden=int(file.num("dflash.embedding_length"));heads=int(file.num("dflash.attention.head_count"));
        kvheads=int(file.num("dflash.attention.head_count_kv"));hd=int(file.num("dflash.attention.key_length"));
        block=int(file.num("dflash.block_size"));group=int(file.num("dflash.conv_group_size"));
        window=int(file.num("dflash.attention.sliding_window"));rank=int(file.num("dflash.selector_rank"));
        mask=uint32_t(file.num("tokenizer.ggml.mask_token_id"));
        eps=float(file.num("dflash.attention.layer_norm_rms_epsilon"));theta=float(file.num("dflash.rope.freq_base"));
        vocab=target.vocabulary();
        for(double id:file.kv.at("dflash.target_layers").arr)features.push_back(int(id));
        const int layers=int(file.num("dflash.block_count"));
        if(hidden!=5120 || hd!=128 || heads!=32 || kvheads!=8 || block!=8 || group!=16 || rank!=256 ||
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
        if(std::getenv("TQLLM_MLX_TILED_DFLASH_FP32"))
            for(const auto& [name,w]:matrices)
                if(name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight") ||
                   name.ends_with(".ffn_down.weight")) {
                    if(auto packed=std::get_if<Tq3Weights>(&w))tiled_ffn.emplace(name,tq3_tile(*packed));
                    else if(auto packed=std::get_if<Tq4Weights>(&w))tiled_ffn.emplace(name,tq4_tile(*packed));
                    else throw std::invalid_argument("tiled drafter FFN requires TQ3 or TQ4 weights");
                }
        for(auto& [name,w]:tiled_ffn)mx::eval(w.planes,w.scales);
        caches.resize(layers);target.feature_layers(features);
    }
    mx::array mat(const std::string& name,const mx::array& x) {
        if(auto found=nax.find(name);found!=nax.end())return nax_affine_matmul(found->second,x,true);
        static const std::string ablate=std::getenv("TQLLM_MLX_DRAFT_ABLATE")?std::getenv("TQLLM_MLX_DRAFT_ABLATE"):"";
        if(!ablate.empty() && x.shape(0)<=8) {
            std::istringstream in(ablate);std::string tag;
            while(std::getline(in,tag,','))
                if(!tag.empty() && name.find(tag)!=std::string::npos)
                    if(auto it=matrices.find(name);it!=matrices.end()) {
                        int rows=0;
                        std::visit([&](const auto& w){
                            using W=std::decay_t<decltype(w)>;
                            if constexpr(std::is_same_v<W,mx::array>)rows=w.shape(0);
                            else if constexpr(std::is_same_v<W,Tq3Weights>)rows=w.planes.shape(0);
                            else if constexpr(std::is_same_v<W,Tq4Weights>)rows=w.codes.shape(0);
                            else rows=w.blocks.shape(0);
                        },it->second);
                        return mx::zeros({x.shape(0),rows},x.dtype());
                    }
        }
        static const bool mma8=std::getenv("TQLLM_MLX_MMA8_DRAFT")!=nullptr;
        if(mma8 && x.shape(0)<=8 && name.find("attn_")!=std::string::npos && name.find("conv")==std::string::npos) {
            auto found=mma8_attn.find(name);
            if(found==mma8_attn.end())
                if(auto it=matrices.find(name);it!=matrices.end()) {
                    if(auto* t3=std::get_if<Tq3Weights>(&it->second))found=mma8_attn.emplace(name,mma8_pack(*t3)).first;
                    else if(auto* t4=std::get_if<Tq4Weights>(&it->second))found=mma8_attn.emplace(name,mma8_pack(*t4)).first;
                }
            if(found!=mma8_attn.end()) {
                if(!rotation || rotation->first.id()!=x.id())rotation=std::pair{x,tq4_rotate(x)};
                if(!rotation_half || rotation_half->first.id()!=x.id())rotation_half=std::pair{x,mma8_input(rotation->second)};
                return mma8_matmul(found->second,rotation_half->second,x.shape(0),true);
            }
        }
        if(x.shape(0)<=8)if(auto found=tiled_ffn.find(name);found!=tiled_ffn.end()) {
            if(!rotation || rotation->first.id()!=x.id())rotation=std::pair{x,tq4_rotate(x)};
            return bf(tq_tiled_matmul(found->second,rotation->second,4,true,32,true));
        }
        return bf(linear(matrices.at(name),x));
    }
    mx::array norm(const std::string& name,const mx::array& x) {return bf(mx::fast::rms_norm(x,plain.at(name),eps));}
    mx::array keys(const std::string& p,const mx::array& x,int position) {
        auto k=mx::reshape(mat(p+"attn_k.weight",x),{1,x.shape(0),kvheads,hd});
        k=mx::transpose(norm(p+"attn_k_norm.weight",k),{0,2,1,3});
        return bf(mx::fast::rope(k,hd,false,theta,1.0f,position));
    }
    mx::array values(const std::string& p,const mx::array& x) {
        return mx::transpose(mx::reshape(mat(p+"attn_v.weight",x),{1,x.shape(0),kvheads,hd}),{0,2,1,3});
    }
};
DFlash2::DFlash2(const std::string& path,Qwen35& target):impl_(std::make_unique<Impl>(path,target)) {}
DFlash2::~DFlash2()=default;
const std::vector<std::vector<uint32_t>>& DFlash2::candidates() const {return impl_->candidates;}
const std::vector<int>& DFlash2::feature_layers() const {return impl_->features;}
void DFlash2::reset() {impl_->end=0;impl_->rotation.reset();impl_->rotation_half.reset();for(auto& c:impl_->caches)c=Impl::Kv{};}
void DFlash2::inject(const mx::array& features,int position) {
    auto& i=*impl_;const int n=features.shape(0);
    if(position!=i.end || features.ndim()!=2 || features.shape(1)!=int(i.features.size())*i.hidden || n<1)
        throw std::invalid_argument("invalid DFlash feature injection");
    auto encoded=i.norm("enc.output_norm.weight",i.mat("fc.weight",features));
    std::vector<mx::array> live;
    bool trimmed=false;
    for(int l=0;l<int(i.caches.size());++l) {
        const std::string p="blk."+std::to_string(l)+".";auto& c=i.caches[l];
        auto k=i.keys(p,encoded,position),v=i.values(p,encoded);
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
    static const bool eager=std::getenv("TQLLM_MLX_EAGER_COMMIT")!=nullptr,lazy_always=std::getenv("TQLLM_MLX_LAZY_INJECT")!=nullptr;
    if(eager || (trimmed && !lazy_always))mx::eval(live);
    i.end+=n;
}
std::vector<uint32_t> DFlash2::draft(uint32_t anchor,int position,int count) {
    auto& i=*impl_;
    i.rotation.reset();i.rotation_half.reset();
    if(position!=i.end || count<1 || count>=i.block || anchor>=uint32_t(i.vocab))
        throw std::invalid_argument("invalid DFlash draft request");
    std::vector<uint32_t> tokens(i.block,i.mask);tokens[0]=anchor;
    auto x=i.target.embed_tokens(tokens);
    for(int l=0;l<int(i.caches.size());++l) {
        const std::string p="blk."+std::to_string(l)+".";
        auto residual=x,n=i.norm(p+"attn_norm.weight",x),dynamic=i.mat(p+"attn_conv_proj.weight",n);
        auto input=dflash_conv(n,dynamic,i.plain.at(p+"attn_conv_base"),i.group,0);
        auto q=mx::reshape(i.mat(p+"attn_q.weight",input),{1,i.block,i.heads,i.hd});
        q=mx::transpose(i.norm(p+"attn_q_norm.weight",q),{0,2,1,3});
        q=bf(mx::fast::rope(q,i.hd,false,i.theta,1.0f,position));
        auto k=i.keys(p,input,position),v=i.values(p,input);const auto& c=i.caches[l];
        if(c.k) {k=mx::concatenate({*c.k,k},2);v=mx::concatenate({*c.v,v},2);}
        auto attention=mx::fast::scaled_dot_product_attention(q,k,v,1.0f/std::sqrt(float(i.hd)));
        auto flattened=bf(mx::reshape(mx::transpose(attention,{0,2,1,3}),{i.block,i.heads*i.hd}));
        x=bf(mx::add(residual,dflash_conv(i.mat(p+"attn_output.weight",flattened),dynamic,i.plain.at(p+"attn_conv_base"),i.group,1)));
        residual=x;n=i.norm(p+"ffn_norm.weight",x);dynamic=i.mat(p+"ffn_conv_proj.weight",n);
        input=dflash_conv(n,dynamic,i.plain.at(p+"ffn_conv_base"),i.group,0);
        auto gate=i.mat(p+"ffn_gate.weight",input),up=i.mat(p+"ffn_up.weight",input);
        auto ff=i.mat(p+"ffn_down.weight",bf(mx::multiply(bf(silu(gate)),up)));
        x=bf(mx::add(residual,dflash_conv(ff,dynamic,i.plain.at(p+"ffn_conv_base"),i.group,1)));
    }
    auto normalized=i.norm("output_norm.weight",x);
    auto logits=i.target.shared_draft_head(normalized);
    const int classes=logits.shape(1);
    auto selected=std::getenv("TQLLM_MLX_DFLASH_SHARDED_TOP16")?dflash_top16_sharded(logits):[&] {
        auto order=mx::argpartition(logits,classes-16,-1);
        return mx::slice(order,{0,classes-16},{i.block,classes});
    }();
    auto unary=mx::take_along_axis(logits,selected,-1);
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
    if(std::getenv("TQLLM_MLX_DEVICE_SELECTOR")) {
        auto chosen=dflash_select(mapped,mx::astype(unary,mx::float32),gate,successors,
            i.plain.at("selector_predecessor.weight"),anchor,count);
        mx::eval(chosen);
        std::vector<uint32_t> result(chosen.data<uint32_t>(),chosen.data<uint32_t>()+count);
        if(std::any_of(result.begin(),result.end(),[&](uint32_t id){return id>=uint32_t(i.vocab);}))
            throw std::runtime_error("DFlash selector token outside vocabulary");
        return result;
    }
    mx::eval(mapped,unary,gate,successors);
    std::vector<uint32_t> result;
    uint32_t predecessor=anchor;
    for(int n=1;n<=count;++n) {
        auto previous=mx::astype(mx::take(i.plain.at("selector_predecessor.weight"),mx::array(predecessor),0),mx::float32);
        auto next=mx::reshape(rows(successors,n,n+1),{16,i.rank});
        auto scores=mx::add(mx::sum(mx::multiply(next,mx::multiply(previous,rows(gate,n,n+1))),-1),mx::reshape(rows(unary,n,n+1),{16}));
        const uint32_t best=mx::argmax(scores).item<uint32_t>();
        predecessor=mapped.data<uint32_t>()[n*16+best];result.push_back(predecessor);
    }
    return result;
}
}

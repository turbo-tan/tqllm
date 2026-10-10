#include "qwen35.hpp"
#include "prefill.hpp"
#include <cmath>
#include <chrono>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace tqllm::mlx_backend {
namespace mx=mlx::core;
namespace {
mx::array tail_slice(const mx::array& a,int axis,int begin,int end) {
    mx::Shape starts(a.ndim(),0),stops=a.shape();
    starts[axis]=begin;stops[axis]=end;
    return mx::slice(a,starts,stops);
}
mx::array bf(const mx::array& a) {return mx::astype(mx::astype(a,mx::bfloat16),mx::float32);}
mx::array silu(const mx::array& a) {return mx::multiply(a,mx::sigmoid(a));}
struct Cache {
    std::optional<mx::array> conv,state,keys,values;
};
}
struct Qwen35::Impl {
    GgufFile file;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,mx::array> plain;
    std::optional<Q4Embedding> embedding;
    std::vector<Cache> caches;
    uint64_t epoch=0;
    int context,pos=0,layers,hidden,heads,kvheads,hd,rot,kh,vh,dk,dv,conv,interval,vocab;
    float eps,theta;
    mx::array key_indices;
    bool profiling=false;
    bool gemm_prefill=false;
    std::pair<double,double> times={0,0};
    explicit Impl(const std::string& path,int ctx):file(gguf_open(path)),context(ctx),key_indices(mx::array(0)) {
        const auto arch=file.kv.at("general.architecture").str;
        if(arch!="qwen35") throw std::invalid_argument("MLX dense forward requires qwen35 architecture");
        const std::string a="qwen35.";
        auto integer=[&](const std::string& key) {
            const double x=file.num(a+key);
            if(!std::isfinite(x)||x<=0||x>std::numeric_limits<int>::max()||x!=std::floor(x))
                throw std::invalid_argument("invalid qwen35 integer metadata: "+key);
            return int(x);
        };
        layers=integer("block_count")-int(file.num(a+"nextn_predict_layers",0));
        hidden=integer("embedding_length");heads=integer("attention.head_count");
        kvheads=integer("attention.head_count_kv");hd=integer("attention.key_length");
        rot=integer("rope.dimension_count");kh=integer("ssm.group_count");
        vh=integer("ssm.time_step_rank");dk=integer("ssm.state_size");
        dv=integer("ssm.inner_size")/vh;conv=integer("ssm.conv_kernel");
        interval=integer("full_attention_interval");eps=float(file.num(a+"attention.layer_norm_rms_epsilon"));
        theta=float(file.num(a+"rope.freq_base"));
        if(ctx<=0||layers<=0||conv!=4||vh%kh||heads%kvheads||rot>hd||rot%2||dv<=0||dk<32||dk%32)
            throw std::invalid_argument("unsupported qwen35 dimensions");
        if(file.num(a+"rope.scaling.factor",1)!=1)
            throw std::invalid_argument("experimental MLX forward does not implement RoPE scaling");
        const auto& emb=file.tensors.at("token_embd.weight");
        if(emb.type!=GGML_Q4_K||emb.ne.size()!=2||emb.ne[0]!=uint64_t(hidden)||emb.ne[1]>uint64_t(INT32_MAX))
            throw std::invalid_argument("experimental MLX forward requires Q4_K embeddings");
        vocab=int(emb.ne[1]);auto data=file.read(emb);
        embedding=load_q4_embedding({reinterpret_cast<const uint8_t*>(data.data()),data.size()},vocab,hidden);
        mx::eval(embedding->blocks);data.clear();data.shrink_to_fit();
        std::vector<uint32_t> indices(vh);
        for(int h=0;h<vh;++h) indices[h]=uint32_t(h%kh);
        key_indices=mx::array(indices.data(),{vh},mx::uint32);
        auto matrix=[&](const std::string& name) {
            auto w=load_matrix(file,name);
            std::visit([](const auto& v) {
                using T=std::decay_t<decltype(v)>;
                if constexpr(std::is_same_v<T,mx::array>) mx::eval(v);
                else if constexpr(std::is_same_v<T,Tq3Weights>) mx::eval(v.planes,v.scales);
                else if constexpr(std::is_same_v<T,Tq4Weights>) mx::eval(v.codes,v.scales);
                else mx::eval(v.blocks);
            },w);
            matrices.emplace(name,std::move(w));
        };
        auto scalar=[&](const std::string& name) {
            const auto& t=file.tensors.at(name);auto raw=file.read(t);
            auto w=mx::astype(load_plain_tensor(t,{reinterpret_cast<const uint8_t*>(raw.data()),raw.size()}),mx::float32);
            mx::eval(w);plain.emplace(name,std::move(w));
        };
        matrix("output.weight");scalar("output_norm.weight");
        for(int l=0;l<layers;++l) {
            const std::string p="blk."+std::to_string(l)+".";
            for(const auto* name:{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"}) matrix(p+name);
            scalar(p+"attn_norm.weight");scalar(p+"post_attention_norm.weight");
            if((l+1)%interval==0) {
                for(const auto* name:{"attn_q.weight","attn_k.weight","attn_v.weight","attn_output.weight"}) matrix(p+name);
                scalar(p+"attn_q_norm.weight");scalar(p+"attn_k_norm.weight");
            } else {
                for(const auto* name:{"attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight","ssm_out.weight"}) matrix(p+name);
                for(const auto* name:{"ssm_conv1d.weight","ssm_a","ssm_dt.bias","ssm_norm.weight"}) scalar(p+name);
            }
        }
        caches.resize(layers);
    }
    mx::array mat(const std::string& name,const mx::array& x) {
        const auto& w=matrices.at(name);
        return bf(gemm_prefill && x.shape(0)>=8?prefill_linear(w,x):linear(w,x));
    }
    mx::array norm(const mx::array& x,const std::string& name) {return bf(mx::fast::rms_norm(x,plain.at(name),eps));}
    mx::array full_attention(const std::string& p,const mx::array& x,Cache& c) {
        const int m=x.shape(0);
        auto qg=mx::reshape(mat(p+"attn_q.weight",x),{m,heads,2*hd});
        auto q=tail_slice(qg,2,0,hd),gate=tail_slice(qg,2,hd,2*hd);
        q=mx::transpose(mx::reshape(norm(q,p+"attn_q_norm.weight"),{1,m,heads,hd}),{0,2,1,3});
        auto k=mx::reshape(mat(p+"attn_k.weight",x),{1,m,kvheads,hd});
        k=mx::transpose(norm(k,p+"attn_k_norm.weight"),{0,2,1,3});
        auto v=mx::transpose(mx::reshape(mat(p+"attn_v.weight",x),{1,m,kvheads,hd}),{0,2,1,3});
        q=bf(mx::fast::rope(q,rot,false,theta,1.0f,pos));k=bf(mx::fast::rope(k,rot,false,theta,1.0f,pos));
        c.keys=c.keys?mx::concatenate({*c.keys,k},2):k;
        c.values=c.values?mx::concatenate({*c.values,v},2):v;
        auto out=mx::fast::scaled_dot_product_attention(q,*c.keys,*c.values,1.0f/std::sqrt(float(hd)),m>1?"causal":"");
        out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
        out=bf(mx::multiply(bf(out),bf(mx::sigmoid(gate))));
        return mat(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
    }
    mx::array gdn(const std::string& p,const mx::array& x,Cache& c) {
        const int m=x.shape(0),width=2*kh*dk+vh*dv;
        auto qkv=mat(p+"attn_qkv.weight",x);
        auto history=c.conv?*c.conv:mx::zeros({conv-1,width});
        auto joined=mx::concatenate({history,qkv},0);
        auto weights=mx::reshape(plain.at(p+"ssm_conv1d.weight"),{width,conv});
        auto result=mx::zeros({m,width});
        for(int tap=0;tap<conv;++tap) {
            auto w=mx::reshape(tail_slice(weights,1,tap,tap+1),{1,width});
            result=mx::add(result,mx::multiply(tail_slice(joined,0,tap,tap+m),w));
        }
        c.conv=mx::contiguous(tail_slice(joined,0,m,m+conv-1));
        result=bf(silu(bf(result)));
        auto q=mx::reshape(tail_slice(result,1,0,kh*dk),{1,m,kh,dk});
        auto k=mx::reshape(tail_slice(result,1,kh*dk,2*kh*dk),{1,m,kh,dk});
        auto v=mx::reshape(tail_slice(result,1,2*kh*dk,width),{1,m,vh,dv});
        auto normalize=[&](const mx::array& a,float scale) {
            return mx::multiply(a,mx::multiply(mx::rsqrt(mx::add(mx::sum(mx::square(a),-1,true),mx::array(1e-6f))),mx::array(scale)));
        };
        q=mx::take(normalize(q,1.0f/std::sqrt(float(dk))),key_indices,2);
        k=mx::take(normalize(k,1),key_indices,2);
        auto a=mat(p+"ssm_alpha.weight",x),b=mat(p+"ssm_beta.weight",x);
        auto dt=mx::add(a,plain.at(p+"ssm_dt.bias"));
        auto sp=mx::add(mx::maximum(dt,mx::array(0.0f)),mx::log1p(mx::exp(mx::negative(mx::abs(dt)))));
        auto decay=mx::reshape(mx::exp(mx::multiply(sp,plain.at(p+"ssm_a"))),{1,m,vh});
        auto update=mx::fast::gated_delta_update(q,k,v,decay,mx::reshape(mx::sigmoid(b),{1,m,vh}),c.state);
        c.state=update[1];
        auto z=mx::reshape(mat(p+"attn_gate.weight",x),{1,m,vh,dv});
        auto out=bf(mx::multiply(mx::fast::rms_norm(bf(update[0]),plain.at(p+"ssm_norm.weight"),eps),silu(z)));
        return mat(p+"ssm_out.weight",mx::reshape(out,{m,vh*dv}));
    }
    mx::array forward(std::span<const uint32_t> tokens,bool all_logits) {
        auto start=std::chrono::steady_clock::now();
        if(tokens.empty()||tokens.size()>size_t(context-pos)) throw std::invalid_argument("MLX token batch exceeds remaining context");
        const int m=int(tokens.size());auto x=bf(q4_embedding(*embedding,tokens));
        for(int l=0;l<layers;++l) {
            const std::string p="blk."+std::to_string(l)+".";
            auto n=norm(x,p+"attn_norm.weight");
            auto attn=(l+1)%interval==0?full_attention(p,n,caches[l]):gdn(p,n,caches[l]);
            x=bf(mx::add(x,attn));n=norm(x,p+"post_attention_norm.weight");
            auto ff=bf(mx::multiply(bf(silu(mat(p+"ffn_gate.weight",n))),mat(p+"ffn_up.weight",n)));
            x=bf(mx::add(x,mat(p+"ffn_down.weight",ff)));
        }
        x=norm(x,"output_norm.weight");
        if(!all_logits) x=tail_slice(x,0,m-1,m);
        std::vector<mx::array> live={x};
        for(const auto& c:caches) {
            if(c.conv) live.push_back(*c.conv);
            if(c.state) live.push_back(*c.state);
            if(c.keys) live.push_back(*c.keys);
            if(c.values) live.push_back(*c.values);
        }
        if(profiling) mx::eval(live);
        auto body_end=std::chrono::steady_clock::now();
        auto logits=linear(matrices.at("output.weight"),x);
        live.push_back(logits);
        mx::eval(live);
        auto end=std::chrono::steady_clock::now();
        times={std::chrono::duration<double,std::milli>(body_end-start).count(),
               std::chrono::duration<double,std::milli>(end-body_end).count()};
        pos+=m;
        return logits;
    }
};
struct Qwen35::Snapshot::State {
    const void* owner;
    uint64_t epoch;
    int pos;
    std::vector<Cache> caches;
};
Qwen35::Snapshot Qwen35::checkpoint() const {
    Snapshot result;
    result.state_=std::make_shared<Snapshot::State>(Snapshot::State{impl_.get(),impl_->epoch,impl_->pos,impl_->caches});
    return result;
}
void Qwen35::restore(const Snapshot& snapshot) {
    if(!snapshot.state_ || snapshot.state_->owner!=impl_.get() || snapshot.state_->epoch!=impl_->epoch)
        throw std::invalid_argument("MLX snapshot belongs to another engine or reset generation");
    impl_->pos=snapshot.state_->pos;
    impl_->caches=snapshot.state_->caches;
}
Qwen35::Qwen35(const std::string& path,int context):impl_(std::make_unique<Impl>(path,context)) {}
Qwen35::~Qwen35()=default;
mx::array Qwen35::forward(std::span<const uint32_t> tokens,bool all_logits) {return impl_->forward(tokens,all_logits);}
void Qwen35::reset() {++impl_->epoch;impl_->pos=0;for(auto& c:impl_->caches)c=Cache{};}
int Qwen35::position() const {return impl_->pos;}
int Qwen35::vocabulary() const {return impl_->vocab;}
void Qwen35::profile(bool enabled) {impl_->profiling=enabled;}
void Qwen35::prefill_gemm(bool enabled) {impl_->gemm_prefill=enabled;}
std::pair<double,double> Qwen35::timings() const {return impl_->times;}
}

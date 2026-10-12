#include "qwen35.hpp"
#include "prefill.hpp"
#include "verify.hpp"
#include "tuning.hpp"
#include "gdn.hpp"
#include "draft_head.hpp"
#include "attention_rows.hpp"
#include "tq3_tiled.hpp"
#include <mlx/compile.h>
#include <cstdlib>
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
mx::array swiglu(const mx::array& gate,const mx::array& up) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf(mx::multiply(bf(silu(args[0])),args[1]))};
    });
    return fused({gate,up})[0];
}
struct Cache {
    std::optional<mx::array> conv,state,keys,values;
};
struct PrefixTape {
    std::optional<mx::array> conv_inputs,states;
    std::optional<mx::array> keys,decay,deltas;
};
}
struct Qwen35::Impl {
    GgufFile file;
    std::map<std::string,Matrix> matrices;
    std::map<std::string,Tq3Expanded> decoded_ffn;
    std::map<std::string,TqTiledWeights> tiled_projections;
    bool tiled_mixer=std::getenv("TQLLM_MLX_TILED_MIXER_FP32")!=nullptr;
    std::map<std::string,Tq3Affine8> affine_ffn;
    std::map<std::string,Tq3Affine8> affine_gu;
    bool joined_gu=std::getenv("TQLLM_MLX_AFFINE_GU")!=nullptr;
    bool affine_tq4=std::getenv("TQLLM_MLX_AFFINE_TQ4")!=nullptr;
    bool affine_tq4_fp32=std::getenv("TQLLM_MLX_AFFINE_TQ4_FP32")!=nullptr;
    bool affine_mixer=std::getenv("TQLLM_MLX_AFFINE_MIXER")!=nullptr;
    std::optional<std::pair<mx::array,mx::array>> rotation;
    bool reuse_rotation=std::getenv("TQLLM_MLX_REUSE_ROTATION")!=nullptr;
    std::map<std::string,mx::array> plain;
    std::optional<Q4Embedding> embedding;
    std::vector<Cache> caches;
    std::vector<PrefixTape> tapes;
    bool capture_prefix=false;
    bool serial_verify_attention=std::getenv("TQLLM_MLX_SERIAL_VERIFY_ATTENTION")!=nullptr;
    bool attention_kv_views=std::getenv("TQLLM_MLX_ATTENTION_KV_VIEWS")!=nullptr;
    bool serial_verify_dense=std::getenv("TQLLM_MLX_SERIAL_VERIFY_DENSE")!=nullptr;
    std::optional<mx::array> last_hidden;
    std::optional<mx::array> last_features;
    std::vector<int> feature_ids;
    Cache mtp_cache;
    int mtp_pos=1;
    bool mtp_enabled=false;
    std::optional<Q6Weights> mtp_head;
    std::optional<Q6AffineSplit> target_head;
    std::optional<DraftHead4> proposal_head;
    std::optional<mx::array> mtp_ids;
    std::optional<mx::array> mtp_hidden;
    uint64_t epoch=0,revision=0;
    int context,pos=0,layers,hidden,heads,kvheads,hd,rot,kh,vh,dk,dv,conv,interval,vocab;
    float eps,theta;
    mx::array key_indices;
    mx::array global_ids=mx::array(0,mx::uint32);
    std::optional<GdnHeadOrder> native_head_order;
    bool scan_gdn=std::getenv("TQLLM_MLX_GDN_SCAN")!=nullptr;
    bool compact_gdn=std::getenv("TQLLM_MLX_COMPACT_GDN_TAPE")!=nullptr;
    bool compiled_gdn=std::getenv("TQLLM_MLX_COMPILED_GDN")!=nullptr;
    bool native_gdn_prep=std::getenv("TQLLM_MLX_NATIVE_GDN_PREP")!=nullptr;
    bool profiling=false,stage_profiling=false;
    std::map<std::string,double> stages;
    bool gemm_prefill=false,gemm_verify=false;
    bool fused_epilogue=std::getenv("TQLLM_MLX_FUSE_EPILOGUE")!=nullptr;
    std::pair<double,double> times={0,0};
    explicit Impl(const std::string& path,int ctx):file(gguf_open(path)),context(ctx),key_indices(mx::array(0)) {
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ3_FP16"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ3_FP16 must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ3_FP32"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ3_FP32 must be 1");
        if(std::getenv("TQLLM_MLX_TILED_TQ3_FP16") && std::getenv("TQLLM_MLX_TILED_TQ3_FP32"))
            throw std::invalid_argument("select only one tiled TQ3 operand precision");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ4_FP32"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ4_FP32 must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_MIXER_FP32"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_MIXER_FP32 must be 1");
        if(compact_gdn && std::string_view(std::getenv("TQLLM_MLX_COMPACT_GDN_TAPE"))!="1")
            throw std::invalid_argument("TQLLM_MLX_COMPACT_GDN_TAPE must be 1");
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
        vocab=int(emb.ne[1]);global_ids=mx::arange(vocab,mx::uint32);mx::eval(global_ids);
        auto data=file.read(emb);
        embedding=load_q4_embedding({reinterpret_cast<const uint8_t*>(data.data()),data.size()},vocab,hidden);
        mx::eval(embedding->blocks);data.clear();data.shrink_to_fit();
        std::vector<uint32_t> indices(vh);
        for(int h=0;h<vh;++h) indices[h]=uint32_t(h%kh);
        key_indices=mx::array(indices.data(),{vh},mx::uint32);
        if(std::getenv("TQLLM_MLX_NATIVE_GDN"))native_head_order=gdn_head_order(kh,vh);
        auto matrix=[&](const std::string& name) {
            auto w=load_matrix(file,name);
            std::visit([](const auto& v) {
                using T=std::decay_t<decltype(v)>;
                if constexpr(std::is_same_v<T,mx::array>) mx::eval(v);
                else if constexpr(std::is_same_v<T,Tq3Weights>) mx::eval(v.planes,v.scales);
                else if constexpr(std::is_same_v<T,Tq4Weights>) mx::eval(v.codes,v.scales);
                else mx::eval(v.blocks);
            },w);
            if((std::getenv("TQLLM_MLX_AFFINE_FFN") || joined_gu) && name.find(".ffn_")!=std::string::npos)
                if(auto packed=std::get_if<Tq3Weights>(&w))affine_ffn.emplace(name,tq3_affine8(*packed));
            if(affine_tq4 && name.find(".ffn_")!=std::string::npos)
                if(auto packed=std::get_if<Tq4Weights>(&w))affine_ffn.emplace(name,tq4_affine8(*packed));
            if(affine_mixer && (name.find(".attn_")!=std::string::npos || name.ends_with(".ssm_out.weight")))
                if(auto packed=std::get_if<Tq3Weights>(&w))affine_ffn.emplace(name,tq3_affine8(*packed));
            if(std::getenv("TQLLM_MLX_DECODED_FFN") && name.find(".ffn_")!=std::string::npos)
                if(auto packed=std::get_if<Tq3Weights>(&w))decoded_ffn.emplace(name,tq3_expand(*packed));
            if((std::getenv("TQLLM_MLX_TILED_TQ3_FP16") || std::getenv("TQLLM_MLX_TILED_TQ3_FP32")) &&
               name.find(".ffn_")!=std::string::npos)
                if(auto packed=std::get_if<Tq3Weights>(&w)) {
                    auto tiled=tq3_tile(*packed);
                    mx::eval(tiled.planes,tiled.scales);
                    tiled_projections.emplace(name,std::move(tiled));
                }
            if(std::getenv("TQLLM_MLX_TILED_TQ4_FP32") && name.find(".ffn_")!=std::string::npos)
                if(auto packed=std::get_if<Tq4Weights>(&w)) {
                    auto tiled=tq4_tile(*packed);
                    mx::eval(tiled.planes,tiled.scales);
                    tiled_projections.emplace(name,std::move(tiled));
                }
            if(tiled_mixer && (name.ends_with(".attn_qkv.weight") ||
               name.ends_with(".attn_gate.weight") || name.ends_with(".ssm_out.weight"))) {
                if(auto packed=std::get_if<Tq3Weights>(&w)) {
                    auto tiled=tq3_tile(*packed);mx::eval(tiled.planes,tiled.scales);
                    tiled_projections.emplace(name,std::move(tiled));
                }else if(auto packed=std::get_if<Tq4Weights>(&w)) {
                    auto tiled=tq4_tile(*packed);mx::eval(tiled.planes,tiled.scales);
                    tiled_projections.emplace(name,std::move(tiled));
                }
            }
            matrices.emplace(name,std::move(w));
        };
        auto scalar=[&](const std::string& name) {
            const auto& t=file.tensors.at(name);auto raw=file.read(t);
            auto w=mx::astype(load_plain_tensor(t,{reinterpret_cast<const uint8_t*>(raw.data()),raw.size()}),mx::float32);
            mx::eval(w);plain.emplace(name,std::move(w));
        };
        matrix("output.weight");scalar("output_norm.weight");
        if (std::getenv("TQLLM_MLX_Q6_AFFINE")) {
            const auto* head = std::get_if<Q6Weights>(&matrices.at("output.weight"));
            if (!head) throw std::invalid_argument("Q6 affine head requires Q6_K output weights");
            target_head = q6_affine_split(*head);
        }
        for(int l=0;l<layers;++l) {
            const std::string p="blk."+std::to_string(l)+".";
            for(const auto* name:{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"}) matrix(p+name);
            if(joined_gu) {
                auto gate=affine_ffn.find(p+"ffn_gate.weight"),up=affine_ffn.find(p+"ffn_up.weight");
                if(gate!=affine_ffn.end() && up!=affine_ffn.end()) {
                    affine_gu.emplace(p,tq3_affine8_join(gate->second,up->second));
                    affine_ffn.erase(gate);affine_ffn.erase(up);
                }
            }
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
        tapes.resize(layers);
    }
    mx::array mat(const std::string& name,const mx::array& x) {
        const auto& w=matrices.at(name);
        if(capture_prefix && serial_verify_dense && x.shape(0)>1 && std::holds_alternative<mx::array>(w)) {
            std::vector<mx::array> rows;
            rows.reserve(x.shape(0));
            for(int t=0;t<x.shape(0);++t)
                rows.push_back(bf(linear(w,mx::contiguous(tail_slice(x,0,t,t+1)))));
            return mx::concatenate(rows,0);
        }
        auto rotated=[&]() {
            if(reuse_rotation && rotation && rotation->first.id()==x.id())return rotation->second;
            auto value=tq4_rotate(x);
            if(reuse_rotation)rotation=std::pair{x,value};
            return value;
        };
        if(x.shape(0)<=8)
            if(auto found=tiled_projections.find(name);found!=tiled_projections.end()) {
                const bool mixer_projection=name.find(".ffn_")==std::string::npos;
                const bool fp32=mixer_projection?std::getenv("TQLLM_MLX_TILED_MIXER_FP32")!=nullptr:
                    std::getenv(found->second.bits==3?"TQLLM_MLX_TILED_TQ3_FP32":"TQLLM_MLX_TILED_TQ4_FP32")!=nullptr;
                const bool active=fp32 || (!mixer_projection && found->second.bits==3 &&
                    std::getenv("TQLLM_MLX_TILED_TQ3_FP16"));
                if(active)return bf(tq_tiled_matmul(found->second,rotated(),4,true,32,fp32));
            }
        if(x.shape(0)>1 && !(gemm_prefill && x.shape(0)>=8))
            if(auto found=affine_ffn.find(name);found!=affine_ffn.end()) {
                const bool fp32=(affine_tq4_fp32 && std::holds_alternative<Tq4Weights>(w)) ||
                    (affine_mixer && name.find(".ffn_")==std::string::npos);
                return bf(fp32?affine8_matmul(found->second,rotated(),false):tq3_affine8_matmul(found->second,rotated()));
            }
        if(x.shape(0)>1 && !(gemm_prefill && x.shape(0)>=8))
            if(auto found=decoded_ffn.find(name);found!=decoded_ffn.end())
                return bf(tq3_expanded_matmul(found->second,rotated()));
        if(reuse_rotation && !(gemm_prefill && x.shape(0)>=8)) {
            if(auto packed=std::get_if<Tq3Weights>(&w))return bf(tq3_matmul(*packed,rotated()));
            if(auto packed=std::get_if<Tq4Weights>(&w))
                return bf(tq4_matmul(*packed,rotated(),1,matmul_columns(x.shape(0)),matmul_simds()));
        }
        return bf(gemm_prefill && x.shape(0)>=8?prefill_linear(w,x):linear(w,x));
    }
    mx::array norm(const mx::array& x,const std::string& name) {return bf(mx::fast::rms_norm(x,plain.at(name),eps));}
    std::pair<mx::array,mx::array> gate_up(const std::string& p,const mx::array& x) {
        if(x.shape(0)>1 && !(gemm_prefill && x.shape(0)>=8))
            if(auto found=affine_gu.find(p);found!=affine_gu.end()) {
                const bool fp32=affine_tq4_fp32 && std::holds_alternative<Tq4Weights>(matrices.at(p+"ffn_gate.weight"));
                auto gu=fp32?affine8_matmul(found->second,tq4_rotate(x),false):tq3_affine8_matmul(found->second,tq4_rotate(x));
                const int rows=gu.shape(1)/2;
                return {bf(tail_slice(gu,1,0,rows)),bf(tail_slice(gu,1,rows,2*rows))};
            }
        return {mat(p+"ffn_gate.weight",x),mat(p+"ffn_up.weight",x)};
    }
    mx::array full_attention(const std::string& p,const mx::array& x,Cache& c,int offset) {
        const int m=x.shape(0);
        auto qg=mx::reshape(mat(p+"attn_q.weight",x),{m,heads,2*hd});
        auto q=tail_slice(qg,2,0,hd),gate=tail_slice(qg,2,hd,2*hd);
        q=mx::transpose(mx::reshape(norm(q,p+"attn_q_norm.weight"),{1,m,heads,hd}),{0,2,1,3});
        auto k=mx::reshape(mat(p+"attn_k.weight",x),{1,m,kvheads,hd});
        k=mx::transpose(norm(k,p+"attn_k_norm.weight"),{0,2,1,3});
        auto v=mx::transpose(mx::reshape(mat(p+"attn_v.weight",x),{1,m,kvheads,hd}),{0,2,1,3});
        q=bf(mx::fast::rope(q,rot,false,theta,1.0f,offset));k=bf(mx::fast::rope(k,rot,false,theta,1.0f,offset));
        c.keys=c.keys?mx::concatenate({*c.keys,k},2):k;
        c.values=c.values?mx::concatenate({*c.values,v},2):v;
        auto out=causal_attention_rows(q,*c.keys,*c.values,1.0f/std::sqrt(float(hd)),serial_verify_attention&&capture_prefix,attention_kv_views);
        out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
        out=bf(mx::multiply(bf(out),bf(mx::sigmoid(gate))));
        return mat(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
    }
    mx::array gdn(const std::string& p,const mx::array& x,Cache& c,PrefixTape& tape) {
        const int m=x.shape(0),width=2*kh*dk+vh*dv;
        auto qkv=mat(p+"attn_qkv.weight",x);
        auto history=c.conv?*c.conv:mx::zeros({conv-1,width});
        auto joined=mx::concatenate({history,qkv},0);
        if(capture_prefix)tape.conv_inputs=joined;
        auto weights=mx::reshape(plain.at(p+"ssm_conv1d.weight"),{width,conv});
        auto result=native_gdn_prep?gdn_convolve_native(joined,weights):gdn_convolve(joined,weights,compiled_gdn);
        c.conv=mx::contiguous(tail_slice(joined,0,m,m+conv-1));
        auto q=mx::reshape(tail_slice(result,1,0,kh*dk),{1,m,kh,dk});
        auto k=mx::reshape(tail_slice(result,1,kh*dk,2*kh*dk),{1,m,kh,dk});
        auto v=mx::reshape(tail_slice(result,1,2*kh*dk,width),{1,m,vh,dv});
        q=gdn_normalize(q,1.0f/std::sqrt(float(dk)),compiled_gdn);
        k=gdn_normalize(k,1,compiled_gdn);
        if(!native_head_order && !scan_gdn) {q=mx::take(q,key_indices,2);k=mx::take(k,key_indices,2);}
        auto a=mat(p+"ssm_alpha.weight",x),b=mat(p+"ssm_beta.weight",x);
        auto gates=native_gdn_prep?gdn_gates_native(a,b,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a")):
            std::pair{gdn_decay(a,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a"),compiled_gdn),mx::sigmoid(b)};
        auto decay=mx::reshape(gates.first,{1,m,vh}),beta=mx::reshape(gates.second,{1,m,vh});
        auto scanned=scan_gdn?std::optional<GdnScan>(gdn_scan(q,k,v,decay,beta,c.state,capture_prefix,capture_prefix&&compact_gdn)):std::nullopt;
        if(capture_prefix) {
            tape.states=scanned->snapshots;
            if(compact_gdn) {tape.keys=k;tape.decay=decay;tape.deltas=scanned->deltas;}
        }
        auto update=scanned?std::vector<mx::array>{scanned->output,scanned->state}:native_head_order?gdn_native(q,k,v,decay,beta,c.state,*native_head_order):
            mx::fast::gated_delta_update(q,k,v,decay,beta,c.state);
        c.state=update[1];
        auto z=mx::reshape(mat(p+"attn_gate.weight",x),{1,m,vh,dv});
        auto out=bf(mx::multiply(mx::fast::rms_norm(bf(update[0]),plain.at(p+"ssm_norm.weight"),eps),silu(z)));
        return mat(p+"ssm_out.weight",mx::reshape(out,{m,vh*dv}));
    }
    mx::array forward(std::span<const uint32_t> tokens,bool all_logits) {
        auto start=std::chrono::steady_clock::now();
        if(tokens.empty()||tokens.size()>size_t(context-pos)) throw std::invalid_argument("MLX token batch exceeds remaining context");
        stages.clear();rotation.reset();
        auto tick=std::chrono::steady_clock::now();
        auto record=[&](const char* name,const std::vector<mx::array>& arrays) {
            if(!stage_profiling)return;
            mx::eval(arrays);
            auto now=std::chrono::steady_clock::now();
            stages[name]+=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;
        };
        const int m=int(tokens.size());auto x=bf(q4_embedding(*embedding,tokens));
        std::vector<mx::array> features;
        if(stage_profiling)record("embedding",{x});
        for(int l=0;l<layers;++l) {
            if(std::binary_search(feature_ids.begin(),feature_ids.end(),l))features.push_back(x);
            const std::string p="blk."+std::to_string(l)+".";
            auto n=norm(x,p+"attn_norm.weight");
            auto attn=(l+1)%interval==0?full_attention(p,n,caches[l],pos):gdn(p,n,caches[l],tapes[l]);
            if(stage_profiling) {
                std::vector<mx::array> values={attn};const auto& c=caches[l];
                for(const auto* a:{&c.conv,&c.state,&c.keys,&c.values})if(*a)values.push_back(**a);
                record((l+1)%interval==0?"attention":"gdn_mixer",values);
            }
            x=bf(mx::add(x,attn));n=norm(x,p+"post_attention_norm.weight");
            auto [gate,up]=gate_up(p,n);
            auto ff=fused_epilogue?swiglu(gate,up):bf(mx::multiply(bf(silu(gate)),up));
            if(stage_profiling)record("ffn_gate_up",{ff});
            x=bf(mx::add(x,mat(p+"ffn_down.weight",ff)));
            if(stage_profiling)record("ffn_down",{x});
        }
        last_hidden=x;
        if(!features.empty())last_features=mx::concatenate(features,1);
        x=norm(x,"output_norm.weight");
        if(!all_logits) x=tail_slice(x,0,m-1,m);
        std::vector<mx::array> live={x};
        if(last_features)live.push_back(*last_features);
        for(const auto& c:caches) {
            if(c.conv) live.push_back(*c.conv);
            if(c.state) live.push_back(*c.state);
            if(c.keys) live.push_back(*c.keys);
            if(c.values) live.push_back(*c.values);
        }
        if(capture_prefix)for(const auto& t:tapes)
            for(const auto* a:{&t.states,&t.keys,&t.decay,&t.deltas})if(*a)live.push_back(**a);
        if(profiling) mx::eval(live);
        auto body_end=std::chrono::steady_clock::now();
        auto logits=target_head && x.shape(0)<=5?q6_affine_matmul(*target_head,x):gemm_verify && x.shape(0)>1 && x.shape(0)<=8?verify_tiled(matrices.at("output.weight"),x):linear(matrices.at("output.weight"),x);
        live.push_back(logits);
        mx::eval(live);
        auto end=std::chrono::steady_clock::now();
        times={std::chrono::duration<double,std::milli>(body_end-start).count(),
               std::chrono::duration<double,std::milli>(end-body_end).count()};
        pos+=m;++revision;rotation.reset();
        return logits;
    }
};
struct Qwen35::Snapshot::State {
    const void* owner;
    uint64_t epoch;
    int pos;
    std::vector<Cache> caches;
    std::optional<mx::array> hidden;
    std::optional<mx::array> features;
};
Qwen35::Snapshot Qwen35::checkpoint() const {
    Snapshot result;
    result.state_=std::make_shared<Snapshot::State>(Snapshot::State{impl_.get(),impl_->epoch,impl_->pos,impl_->caches,impl_->last_hidden,impl_->last_features});
    return result;
}
void Qwen35::restore(const Snapshot& snapshot) {
    if(!snapshot.state_ || snapshot.state_->owner!=impl_.get() || snapshot.state_->epoch!=impl_->epoch)
        throw std::invalid_argument("MLX snapshot belongs to another engine or reset generation");
    impl_->pos=snapshot.state_->pos;
    impl_->caches=snapshot.state_->caches;
    impl_->last_hidden=snapshot.state_->hidden;
    impl_->last_features=snapshot.state_->features;
    ++impl_->revision;
}
struct Qwen35::Verification::State {
    const void* owner;
    uint64_t epoch,revision;
    int start,count;
    std::vector<Cache> before,after;
    std::vector<PrefixTape> tapes;
    std::optional<mx::array> hidden_before,hidden_after;
    std::optional<mx::array> features_before,features_after;
};
mx::array Qwen35::verify(std::span<const uint32_t> tokens,Verification& batch) {
    if(!impl_->scan_gdn || impl_->native_head_order || tokens.empty() || tokens.size()>8)
        throw std::invalid_argument("MLX prefix verification requires scan GDN and 1-8 tokens");
    batch.state_.reset();
    const auto before=checkpoint();
    impl_->capture_prefix=true;
    for(auto& tape:impl_->tapes)tape=PrefixTape{};
    try {
        auto logits=impl_->forward(tokens,true);
        batch.state_=std::make_shared<Verification::State>(Verification::State{
            impl_.get(),impl_->epoch,impl_->revision,before.state_->pos,int(tokens.size()),
            before.state_->caches,impl_->caches,impl_->tapes,before.state_->hidden,impl_->last_hidden,
            before.state_->features,impl_->last_features});
        impl_->capture_prefix=false;
        for(auto& tape:impl_->tapes)tape=PrefixTape{};
        return logits;
    } catch(...) {
        impl_->capture_prefix=false;
        for(auto& tape:impl_->tapes)tape=PrefixTape{};
        restore(before);
        throw;
    }
}
void Qwen35::accept_prefix(const Verification& batch,size_t count) {
    const auto* s=batch.state_.get();
    if(!s || s->owner!=impl_.get() || s->epoch!=impl_->epoch || s->revision!=impl_->revision ||
       count>size_t(s->count) || impl_->pos!=s->start+s->count)
        throw std::invalid_argument("MLX prefix verification is stale or count is invalid");
    auto caches=count?s->after:s->before;
    if(count && count<size_t(s->count))for(int l=0;l<impl_->layers;++l) {
        auto& c=caches[l];const auto& tape=s->tapes[l];
        if((l+1)%impl_->interval==0) {
            c.keys=mx::contiguous(tail_slice(*c.keys,2,0,s->start+int(count)));
            c.values=mx::contiguous(tail_slice(*c.values,2,0,s->start+int(count)));
        } else {
            c.conv=mx::contiguous(tail_slice(*tape.conv_inputs,0,int(count),int(count)+impl_->conv-1));
            c.state=tape.deltas?gdn_replay_prefix(*tape.keys,*tape.decay,*tape.deltas,s->before[l].state,int(count)):
                mx::contiguous(tail_slice(*tape.states,0,int(count)-1,int(count)));
        }
    }
    std::vector<mx::array> live;
    for(const auto& c:caches)for(const auto* a:{&c.conv,&c.state,&c.keys,&c.values})if(*a)live.push_back(**a);
    mx::eval(live);
    impl_->caches=std::move(caches);impl_->pos=s->start+int(count);++impl_->revision;
    impl_->last_hidden=count?std::optional<mx::array>(tail_slice(*s->hidden_after,0,int(count)-1,int(count))):s->hidden_before;
    impl_->last_features=count && s->features_after?std::optional<mx::array>(tail_slice(*s->features_after,0,0,int(count))):s->features_before;
}
void Qwen35::feature_layers(const std::vector<int>& layers) {
    if(impl_->pos!=0 || !std::is_sorted(layers.begin(),layers.end()) ||
       std::adjacent_find(layers.begin(),layers.end())!=layers.end() ||
       (!layers.empty() && (layers.front()<0 || layers.back()>=impl_->layers)))
        throw std::invalid_argument("feature layers must be unique, ordered and configured before prefill");
    impl_->feature_ids=layers;impl_->last_features.reset();
}
mx::array Qwen35::feature_states() const {
    if(!impl_->last_features)throw std::invalid_argument("no target features captured");
    return *impl_->last_features;
}
mx::array Qwen35::embed_tokens(std::span<const uint32_t> tokens) const {return bf(q4_embedding(*impl_->embedding,tokens));}
mx::array Qwen35::shared_draft_head(const mx::array& x) const {
    if(x.ndim()!=2 || x.shape(1)!=impl_->hidden || x.dtype()!=mx::float32)
        throw std::invalid_argument("invalid shared draft-head input");
    if(impl_->proposal_head)return draft_head4_matmul(*impl_->proposal_head,x);
    return impl_->mtp_head?q6_matmul(*impl_->mtp_head,x):linear(impl_->matrices.at("output.weight"),x);
}
void Qwen35::enable_draft_head4() {
    if(impl_->pos!=0)throw std::invalid_argument("proposal head must be configured before prefill");
    if(!impl_->proposal_head)impl_->proposal_head=draft_head4(impl_->matrices.at("output.weight"));
}
mx::array Qwen35::hidden_states() const {
    if(!impl_->last_hidden)throw std::invalid_argument("no MLX target hidden states");
    return *impl_->last_hidden;
}
void Qwen35::enable_mtp() {
    if(impl_->pos!=0)throw std::invalid_argument("MTP must be loaded before starting a session");
    if(impl_->mtp_enabled)return;
    const std::string p="blk."+std::to_string(impl_->layers)+".";
    if(!impl_->file.tensors.contains(p+"nextn.eh_proj.weight"))throw std::invalid_argument("model has no MTP head");
    for(const auto* suffix:{"nextn.eh_proj.weight","attn_q.weight","attn_k.weight","attn_v.weight",
         "attn_output.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"}) {
        auto w=load_matrix(impl_->file,p+suffix);
        std::visit([](const auto& v) {
            using T=std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T,mx::array>)mx::eval(v);
            else if constexpr(std::is_same_v<T,Tq3Weights>)mx::eval(v.planes,v.scales);
            else if constexpr(std::is_same_v<T,Tq4Weights>)mx::eval(v.codes,v.scales);
            else mx::eval(v.blocks);
        },w);
        impl_->matrices.emplace(p+suffix,std::move(w));
    }
    for(const auto* suffix:{"nextn.enorm.weight","nextn.hnorm.weight","nextn.shared_head_norm.weight",
         "attn_norm.weight","post_attention_norm.weight","attn_q_norm.weight","attn_k_norm.weight"}) {
        const auto& t=impl_->file.tensors.at(p+suffix);auto raw=impl_->file.read(t);
        auto value=mx::astype(load_plain_tensor(t,{reinterpret_cast<const uint8_t*>(raw.data()),raw.size()}),mx::float32);
        mx::eval(value);impl_->plain.emplace(p+suffix,std::move(value));
    }
    impl_->mtp_enabled=true;
}
void Qwen35::draft_shortlist(const mx::array& logits,int count) {
    if(count<1 || count>impl_->vocab || logits.shape()!=mx::Shape{1,impl_->vocab})
        throw std::invalid_argument("invalid MTP shortlist");
    const auto* head=std::get_if<Q6Weights>(&impl_->matrices.at("output.weight"));
    if(!head)throw std::invalid_argument("MTP shortlist requires Q6_K target head");
    auto ordered=mx::reshape(mx::argsort(logits,-1),{impl_->vocab});
    impl_->mtp_ids=mx::contiguous(tail_slice(ordered,0,impl_->vocab-count,impl_->vocab));
    impl_->mtp_head=Q6Weights{mx::contiguous(mx::take(head->blocks,*impl_->mtp_ids,0))};
    mx::eval(*impl_->mtp_ids,impl_->mtp_head->blocks);
}
mx::array Qwen35::draft_ids() const {
    if(impl_->proposal_head)return impl_->global_ids;
    if(impl_->mtp_ids)return *impl_->mtp_ids;
    return impl_->global_ids;
}
struct Qwen35::DraftSnapshot::State {
    const void* owner;
    uint64_t epoch;
    int pos;
    Cache cache;
    std::optional<mx::array> hidden;
};
Qwen35::DraftSnapshot Qwen35::draft_checkpoint() const {
    if(!impl_->mtp_enabled)throw std::invalid_argument("MTP is not loaded");
    DraftSnapshot result;
    result.state_=std::make_shared<DraftSnapshot::State>(DraftSnapshot::State{
        impl_.get(),impl_->epoch,impl_->mtp_pos,impl_->mtp_cache,impl_->mtp_hidden});
    return result;
}
void Qwen35::draft_restore(const DraftSnapshot& snapshot) {
    if(!snapshot.state_ || snapshot.state_->owner!=impl_.get() || snapshot.state_->epoch!=impl_->epoch)
        throw std::invalid_argument("MTP snapshot belongs to another engine or reset generation");
    impl_->mtp_pos=snapshot.state_->pos;impl_->mtp_cache=snapshot.state_->cache;
    impl_->mtp_hidden=snapshot.state_->hidden;
}
mx::array Qwen35::draft_hidden_states() const {
    if(!impl_->mtp_hidden)throw std::invalid_argument("no MTP hidden states");
    return *impl_->mtp_hidden;
}
void Qwen35::draft_inject(std::span<const uint32_t> tokens,const mx::array& hidden,int position) {
    auto& i=*impl_;const int n=int(tokens.size());
    if(!i.mtp_enabled || n<1 || position!=i.mtp_pos || n>i.context-position ||
       hidden.shape()!=mx::Shape{n,i.hidden} || hidden.dtype()!=mx::float32)
        throw std::invalid_argument("invalid MTP injection batch or position");
    const std::string p="blk."+std::to_string(i.layers)+".";
    auto embedded=bf(q4_embedding(*i.embedding,tokens));
    auto joined=mx::concatenate({i.norm(embedded,p+"nextn.enorm.weight"),i.norm(hidden,p+"nextn.hnorm.weight")},1);
    auto x=i.norm(i.mat(p+"nextn.eh_proj.weight",joined),p+"attn_norm.weight");
    auto k=mx::reshape(i.mat(p+"attn_k.weight",x),{1,n,i.kvheads,i.hd});
    k=mx::transpose(i.norm(k,p+"attn_k_norm.weight"),{0,2,1,3});
    k=bf(mx::fast::rope(k,i.rot,false,i.theta,1.0f,position));
    auto v=mx::transpose(mx::reshape(i.mat(p+"attn_v.weight",x),{1,n,i.kvheads,i.hd}),{0,2,1,3});
    i.mtp_cache.keys=i.mtp_cache.keys?mx::concatenate({*i.mtp_cache.keys,k},2):k;
    i.mtp_cache.values=i.mtp_cache.values?mx::concatenate({*i.mtp_cache.values,v},2):v;
    mx::eval(*i.mtp_cache.keys,*i.mtp_cache.values);i.mtp_pos+=n;i.mtp_hidden.reset();i.rotation.reset();
}
mx::array Qwen35::draft_forward(std::span<const uint32_t> tokens,const mx::array& hidden,int position,bool logits) {
    auto& i=*impl_;const int n=int(tokens.size());
    if(!i.mtp_enabled || n<1 || position!=i.mtp_pos || n>i.context-position ||
       hidden.shape()!=mx::Shape{n,i.hidden} || hidden.dtype()!=mx::float32)
        throw std::invalid_argument("invalid MTP token/hidden batch or position");
    const std::string p="blk."+std::to_string(i.layers)+".";
    auto embedded=bf(q4_embedding(*i.embedding,tokens));
    auto joined=mx::concatenate({i.norm(embedded,p+"nextn.enorm.weight"),i.norm(hidden,p+"nextn.hnorm.weight")},1);
    auto x=i.mat(p+"nextn.eh_proj.weight",joined);
    x=bf(mx::add(x,i.full_attention(p,i.norm(x,p+"attn_norm.weight"),i.mtp_cache,position)));
    auto normed=i.norm(x,p+"post_attention_norm.weight");
    auto [gate,up]=i.gate_up(p,normed);
    x=bf(mx::add(x,i.mat(p+"ffn_down.weight",bf(mx::multiply(bf(silu(gate)),up)))));
    auto output=x;
    i.mtp_hidden=tail_slice(x,0,n-1,n);
    if(logits) {
        auto normalized=i.norm(tail_slice(x,0,n-1,n),p+"nextn.shared_head_norm.weight");
        output=shared_draft_head(normalized);
    }
    mx::eval(output,*i.mtp_cache.keys,*i.mtp_cache.values);
    i.mtp_pos+=n;i.rotation.reset();
    return output;
}
Qwen35::Qwen35(const std::string& path,int context):impl_(std::make_unique<Impl>(path,context)) {}
Qwen35::~Qwen35()=default;
void Qwen35::gdn_mode(GdnMode mode) {
    if(impl_->pos!=0)throw std::invalid_argument("cannot change GDN layout during a session");
    reset();
    impl_->native_head_order.reset();impl_->scan_gdn=false;
    if(mode!=GdnMode::baseline && (impl_->dk!=128 || impl_->dv!=128))
        throw std::invalid_argument("unsupported optimized GDN dimensions");
    if(mode==GdnMode::native)impl_->native_head_order=gdn_head_order(impl_->kh,impl_->vh);
    if(mode==GdnMode::scan)impl_->scan_gdn=true;
}
mx::array Qwen35::forward(std::span<const uint32_t> tokens,bool all_logits) {return impl_->forward(tokens,all_logits);}
void Qwen35::reset() {
    ++impl_->epoch;++impl_->revision;impl_->pos=0;for(auto& c:impl_->caches)c=Cache{};
    impl_->last_hidden.reset();impl_->mtp_cache=Cache{};impl_->mtp_pos=1;
    impl_->mtp_hidden.reset();
    impl_->last_features.reset();
}
int Qwen35::position() const {return impl_->pos;}
int Qwen35::vocabulary() const {return impl_->vocab;}
void Qwen35::profile(bool enabled) {impl_->profiling=enabled;}
void Qwen35::profile_stages(bool enabled) {impl_->stage_profiling=enabled;}
const std::map<std::string,double>& Qwen35::stage_timings() const {return impl_->stages;}
void Qwen35::tq3_vector_load(bool enabled) {
    if(!enabled)return;
    if(impl_->pos!=0)throw std::invalid_argument("cannot change TQ3 layout during a session");
    for(auto& [name,w]:impl_->matrices)
        if(auto* q=std::get_if<Tq3Weights>(&w))*q=tq3_interleave(*q);
}
void Qwen35::verify_tiles(bool enabled) {impl_->gemm_verify=enabled;}
void Qwen35::prefill_gemm(bool enabled) {impl_->gemm_prefill=enabled;}
std::pair<double,double> Qwen35::timings() const {return impl_->times;}
}

#include "qwen35.hpp"
#include <cstdio>
#include <sstream>
#include "dense_rows.hpp"
#include "prefill.hpp"
#include "verify.hpp"
#include "tuning.hpp"
#include "gdn.hpp"
#include "draft_head.hpp"
#include "attention_rows.hpp"
#include "tq3_tiled.hpp"
#include "mma8.hpp"
#include "tq_mlx.hpp"
#include "nax_affine.hpp"
#include <cstring>
#include <mlx/compile.h>
#include <mlx/graph_utils.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <limits>
#include <map>
#include <numeric>
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
// matrix-unit defaults (j576): on unless the variable is set to 0
bool env_default_on(const char* name) {
    const char* v=std::getenv(name);
    return !v || std::string(v)!="0";
}
bool ablate_small(const char* tag) {
    static const std::string value=std::getenv("TQLLM_MLX_ABLATE_SMALL")?std::getenv("TQLLM_MLX_ABLATE_SMALL"):"";
    return !value.empty() && value.find(tag)!=std::string::npos;
}
mx::array bf_raw(const mx::array& a) {
    static const bool skip=ablate_small("bf");
    return skip?a:mx::astype(mx::astype(a,mx::bfloat16),mx::float32);
}
mx::array silu(const mx::array& a) {return mx::multiply(a,mx::sigmoid(a));}
bool fuse_ew() {static const bool on=std::getenv("TQLLM_MLX_FUSE_EW")!=nullptr;return on;}
mx::array bf(const mx::array& a) {
    if(!fuse_ew())return bf_raw(a);
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {return std::vector<mx::array>{bf_raw(args[0])};},true);
    return fused({a})[0];
}
mx::array add_bf(const mx::array& a,const mx::array& b) {
    if(!fuse_ew())return bf_raw(mx::add(a,b));
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {return std::vector<mx::array>{bf_raw(mx::add(args[0],args[1]))};},true);
    return fused({a,b})[0];
}
mx::array gate_mul_bf(const mx::array& out,const mx::array& gate) {
    if(!fuse_ew())return bf_raw(mx::multiply(bf_raw(out),bf_raw(mx::sigmoid(gate))));
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf_raw(mx::multiply(bf_raw(args[0]),bf_raw(mx::sigmoid(args[1]))))};
    },true);
    return fused({out,gate})[0];
}
mx::array silu_mul_bf(const mx::array& normed,const mx::array& z) {
    if(!fuse_ew())return bf_raw(mx::multiply(normed,silu(z)));
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf_raw(mx::multiply(args[0],silu(args[1])))};
    },true);
    return fused({normed,z})[0];
}
mx::array swiglu(const mx::array& gate,const mx::array& up) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf_raw(mx::multiply(bf_raw(silu(args[0])),args[1]))};
    });
    return fused({gate,up})[0];
}
mx::array swiglu_rotated(const mx::array& gate,const mx::array& up) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{tq4_rotate(bf_raw(mx::multiply(bf_raw(silu(args[0])),args[1])))};
    });
    return fused({gate,up})[0];
}
struct Cache {
    std::optional<mx::array> conv,state,keys,values;  // keys/values: MTP cache only
    int kv_len=0;  // full-attention positions filled in the layer's KV buffer (Impl::kv_keys / kv_values)
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
    std::map<std::string,Mma8Weights> mma8_projections;
    bool tqmlx=is_tq_mlx_file(file);
    bool mma8=tqmlx||std::getenv("TQLLM_MLX_MMA8")!=nullptr;
    bool mma8_untiled_only=std::getenv("TQLLM_MLX_MMA8_UNTILED_ONLY")!=nullptr;
    std::string ablate_blocks=std::getenv("TQLLM_MLX_ABLATE_BLOCKS")?std::getenv("TQLLM_MLX_ABLATE_BLOCKS"):"";
    std::vector<std::string> ablate=[]{
        std::vector<std::string> out;
        if(const char* v=std::getenv("TQLLM_MLX_ABLATE")) {
            std::string s(v),item;
            std::istringstream in(s);
            while(std::getline(in,item,','))if(!item.empty())out.push_back(item);
        }
        return out;
    }();
    bool tiled_mixer=std::getenv("TQLLM_MLX_TILED_MIXER_FP32")!=nullptr ||
        std::getenv("TQLLM_MLX_TILED_MIXER_FP16")!=nullptr;
    std::map<std::string,Tq3Affine8> affine_ffn;
    std::map<std::string,Tq3Affine8> affine_gu;
    bool joined_gu=std::getenv("TQLLM_MLX_AFFINE_GU")!=nullptr;
    bool affine_tq4=std::getenv("TQLLM_MLX_AFFINE_TQ4")!=nullptr;
    bool affine_tq4_fp32=std::getenv("TQLLM_MLX_AFFINE_TQ4_FP32")!=nullptr;
    bool affine_mixer=std::getenv("TQLLM_MLX_AFFINE_MIXER")!=nullptr;
    std::optional<std::pair<mx::array,mx::array>> rotation;
    std::optional<std::pair<mx::array,mx::array>> rotation_half;
    bool reuse_rotation=std::getenv("TQLLM_MLX_REUSE_ROTATION")!=nullptr;
    std::map<std::string,mx::array> plain;
    std::optional<Q4Embedding> embedding;
    std::optional<Affine8Embedding> embedding8;
    std::map<std::string,NaxAffine> nax;
    std::map<std::string,NaxAffine> nax_tq4;  // TQ4 matrices repacked for the matrix unit (rotated inputs)
    std::optional<std::pair<mx::array,mx::array>> nax_rotated_input;
    std::vector<uint8_t> nax_head_payload;   // host copy of the NAX head, for the drafter's shortlist head
    std::optional<NaxAffine> draft_nax_head;
    // projections sharing an input run as one NAX launch; members read column views of the shared result
    struct NaxGroup {NaxAffine weights;std::vector<std::pair<int,int>> parts;std::optional<std::pair<mx::array,mx::array>> last;};
    std::map<std::string,NaxGroup> nax_groups;
    std::optional<std::pair<mx::array,mx::array>> nax_input;  // last matmul input and its FP16 copy (same-input projections)
    std::optional<std::pair<mx::array,mx::array>> gemm_input,gemm_rotated_input;  // prefill GEMM inputs in FP16
    std::map<std::string,std::pair<std::string,int>> nax_member;
    std::vector<Cache> caches;
    // full-attention KV buffers, written in place; never snapshotted (a snapshot keeps only Cache::kv_len)
    std::vector<std::optional<mx::array>> kv_keys,kv_values;
    std::vector<PrefixTape> tapes;
    bool capture_prefix=false;
    bool serial_verify_attention=std::getenv("TQLLM_MLX_SERIAL_VERIFY_ATTENTION")!=nullptr;
    bool attention_kv_views=std::getenv("TQLLM_MLX_ATTENTION_KV_VIEWS")!=nullptr;
    bool fused_attention_rows=std::getenv("TQLLM_MLX_FUSED_ATTENTION_ROWS")!=nullptr;
    bool serial_verify_dense=std::getenv("TQLLM_MLX_SERIAL_VERIFY_DENSE")!=nullptr;
    bool batch_decode_dense=std::getenv("TQLLM_MLX_BATCH_DECODE_DENSE")!=nullptr;
    std::optional<std::pair<mx::array,mx::array>> dense_input;
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
    // default on since j592 (verify -2.6 ms with both, exact); =0 opts out
    bool compiled_gdn=env_default_on("TQLLM_MLX_COMPILED_GDN");
    bool native_gdn_prep=env_default_on("TQLLM_MLX_NATIVE_GDN_PREP");
    bool profiling=false,stage_profiling=false;
    std::map<std::string,double> stages;
    std::chrono::steady_clock::time_point stage_tick;
    // with profile_stages: evaluate the arrays and charge the time since the last mark to this stage
    void stage(const char* name,const std::vector<mx::array>& arrays) {
        if(!stage_profiling)return;
        mx::eval(arrays);
        auto now=std::chrono::steady_clock::now();
        stages[name]+=std::chrono::duration<double,std::milli>(now-stage_tick).count();stage_tick=now;
    }
    bool gemm_prefill=false,gemm_verify=false;
    int target_head_shortlist=0;
    bool return_head_ids=false;
    bool paired_gate_up=std::getenv("TQLLM_MLX_PAIRED_GATE_UP")!=nullptr;
    bool paired_swiglu_rotation=std::getenv("TQLLM_MLX_PAIRED_SWIGLU_ROTATION")!=nullptr;
    // prefill GEMMs of TQ4 matrices on the int8 matrix-unit path (exact levels, per-row int8 input); opt-in until gated
    // =1 fast (per-row weights, per-token input), =2 exact group scales (slow)
    int prefill_int8=std::getenv("TQLLM_MLX_PREFILL_INT8")?std::atoi(std::getenv("TQLLM_MLX_PREFILL_INT8")):0;
    // which GEMMs take the int8 path: 1 = projections through mat(), 2 = FFN gate/up, 4 = FFN down. Default gate/up only:
    // j642 wikitext KL vs exact 0.0038 / 0.0008 (floor 0.0022 / 0.0004) for +8-18 % prefill; the projections alone cost
    // 0.0151 / 0.0018 and down 0.0047 / 0.0008
    int prefill_int8_mask=std::getenv("TQLLM_MLX_PREFILL_INT8_MASK")?std::atoi(std::getenv("TQLLM_MLX_PREFILL_INT8_MASK")):2;
    bool fused_epilogue=std::getenv("TQLLM_MLX_FUSE_EPILOGUE")!=nullptr;
    bool fused_swiglu_rotation=std::getenv("TQLLM_MLX_FUSE_SWIGLU_ROTATION")!=nullptr;
    bool lazy_verify_commit=std::getenv("TQLLM_MLX_LAZY_VERIFY_COMMIT")!=nullptr;
    bool drop_tiled_originals=std::getenv("TQLLM_MLX_DROP_TILED_ORIGINALS")!=nullptr;
    std::pair<double,double> times={0,0};
    explicit Impl(const std::string& path,int ctx):file(gguf_open(path)),context(ctx),key_indices(mx::array(0)) {
        if(const char* mode=std::getenv("TQLLM_MLX_PAIRED_GATE_UP"))
            if(std::string_view(mode)!="1")throw std::invalid_argument("TQLLM_MLX_PAIRED_GATE_UP must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_PAIRED_SWIGLU_ROTATION"))
            if(std::string_view(mode)!="1")throw std::invalid_argument("TQLLM_MLX_PAIRED_SWIGLU_ROTATION must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ3_FP16"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ3_FP16 must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ3_FP32"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ3_FP32 must be 1");
        if(std::getenv("TQLLM_MLX_TILED_TQ3_FP16") && std::getenv("TQLLM_MLX_TILED_TQ3_FP32"))
            throw std::invalid_argument("select only one tiled TQ3 operand precision");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ4_FP16"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ4_FP16 must be 1");
        if(std::getenv("TQLLM_MLX_TILED_TQ4_FP16") && std::getenv("TQLLM_MLX_TILED_TQ4_FP32"))
            throw std::invalid_argument("select only one tiled TQ4 operand precision");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_TQ4_FP32"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_TQ4_FP32 must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_MIXER_FP16"))
            if(std::string_view(mode)!="1")throw std::invalid_argument("TQLLM_MLX_TILED_MIXER_FP16 must be 1");
        if(std::getenv("TQLLM_MLX_TILED_MIXER_FP16") && std::getenv("TQLLM_MLX_TILED_MIXER_FP32"))
            throw std::invalid_argument("select only one tiled mixer operand precision");
        if(const char* mode=std::getenv("TQLLM_MLX_TILED_MIXER_FP32"))
            if(std::string_view(mode)!="1")
                throw std::invalid_argument("TQLLM_MLX_TILED_MIXER_FP32 must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_LAZY_VERIFY_COMMIT"))
            if(std::string_view(mode)!="1")throw std::invalid_argument("TQLLM_MLX_LAZY_VERIFY_COMMIT must be 1");
        if(const char* mode=std::getenv("TQLLM_MLX_TARGET_HEAD_SHORTLIST")) {
            target_head_shortlist=std::stoi(mode);
            if(target_head_shortlist<16 || target_head_shortlist>2048)
                throw std::invalid_argument("TQLLM_MLX_TARGET_HEAD_SHORTLIST must be 16..2048");
        }
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
        if((emb.type!=GGML_Q4_K&&emb.type!=GGML_AFF8_G64)||emb.ne.size()!=2||emb.ne[0]!=uint64_t(hidden)||emb.ne[1]>uint64_t(INT32_MAX))
            throw std::invalid_argument("experimental MLX forward requires Q4_K or 8-bit affine embeddings");
        vocab=int(emb.ne[1]);global_ids=mx::arange(vocab,mx::uint32);mx::eval(global_ids);
        auto data=file.read(emb);
        if(emb.type==GGML_AFF8_G64)embedding8=affine8_embedding_load({reinterpret_cast<const uint8_t*>(data.data()),data.size()},vocab,hidden);
        else {
            embedding=load_q4_embedding({reinterpret_cast<const uint8_t*>(data.data()),data.size()},vocab,hidden);
            mx::eval(embedding->blocks);
        }
        data.clear();data.shrink_to_fit();
        std::vector<uint32_t> indices(vh);
        for(int h=0;h<vh;++h) indices[h]=uint32_t(h%kh);
        key_indices=mx::array(indices.data(),{vh},mx::uint32);
        if(std::getenv("TQLLM_MLX_NATIVE_GDN"))native_head_order=gdn_head_order(kh,vh);
        auto matrix=[&](const std::string& name) {
            if(const auto& stored=file.tensors.at(name);stored.type==GGML_NAX_A4_G128||stored.type==GGML_NAX_A6_G64||stored.type==GGML_NAX_A4_G64) {
                auto raw=file.read(stored);
                const int bits=stored.type==GGML_NAX_A6_G64?6:4,group=stored.type==GGML_NAX_A4_G128?128:64;
                nax.emplace(name,nax_affine_load({reinterpret_cast<const uint8_t*>(raw.data()),raw.size()},
                                                 int(stored.ne[1]),int(stored.ne[0]),bits,group));
                if(name=="output.weight")nax_head_payload.assign(reinterpret_cast<const uint8_t*>(raw.data()),reinterpret_cast<const uint8_t*>(raw.data())+raw.size());
                return;
            }
            if(const auto& stored=file.tensors.at(name);is_tq_mlx_type(stored.type)) {
                auto raw=file.read(stored);
                auto packed=tq_mlx_load(stored,{reinterpret_cast<const uint8_t*>(raw.data()),raw.size()});
                if(auto* tiled=std::get_if<TqTiledWeights>(&packed))tiled_projections.emplace(name,std::move(*tiled));
                else mma8_projections.emplace(name,std::move(std::get<Mma8Weights>(packed)));
                return;
            }
            auto w=load_matrix(file,name);
            std::visit([](const auto& v) {
                using T=std::decay_t<decltype(v)>;
                if constexpr(std::is_same_v<T,mx::array>) mx::eval(v);
                else if constexpr(std::is_same_v<T,Tq3Weights>) mx::eval(v.planes,v.scales);
                else if constexpr(std::is_same_v<T,Tq4Weights>) mx::eval(v.codes,v.scales);
                else mx::eval(v.blocks);
            },w);
            if(env_default_on("TQLLM_MLX_NAX_TQ4"))
                if(auto packed=std::get_if<Tq4Weights>(&w);packed && packed->integer_levels && packed->codes.shape(0)%128==0) {
                    nax_tq4.emplace(name,nax_from_tq4(*packed));
                    return;  // the repacked copy replaces the original
                }
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
            if((std::getenv("TQLLM_MLX_TILED_TQ4_FP16") || std::getenv("TQLLM_MLX_TILED_TQ4_FP32")) && name.find(".ffn_")!=std::string::npos)
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
            if(mma8 && !(mma8_untiled_only && tiled_projections.contains(name))) {
                if(auto packed=std::get_if<Tq4Weights>(&w)) {
                    if(packed->integer_levels && packed->codes.shape(1)%64==0) {
                        mma8_projections.emplace(name,mma8_pack(*packed));
                        tiled_projections.erase(name);
                    }
                }else if(auto packed=std::get_if<Tq3Weights>(&w)) {
                    if(!packed->interleaved && name.find("_norm")==std::string::npos) {
                        mma8_projections.emplace(name,mma8_pack(*packed));
                        tiled_projections.erase(name);
                    }
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
        if(file.tensors.contains("output_draft.weight") && std::getenv("TQLLM_MLX_NAX_DRAFT_HEAD4"))matrix("output_draft.weight");
        const auto head_matrix = matrices.find("output.weight");
        const auto* q6_head = head_matrix == matrices.end() ? nullptr : std::get_if<Q6Weights>(&head_matrix->second);
        if (q6_head && env_default_on("TQLLM_MLX_NAX_Q6_HEAD") && !target_head_shortlist) {
            nax.emplace("output.weight", nax_from_q6(*q6_head));
        } else if (std::getenv("TQLLM_MLX_Q6_AFFINE")) {
            const auto* head = std::get_if<Q6Weights>(&matrices.at("output.weight"));
            if (!head) throw std::invalid_argument("Q6 affine head requires Q6_K output weights");
            target_head = q6_affine_split(*head);
        }
        if(target_head_shortlist) {
            const auto* head=std::get_if<Q6Weights>(&matrices.at("output.weight"));
            if(!head)throw std::invalid_argument("target-head shortlist requires Q6_K output weights");
            proposal_head=draft_head4(matrices.at("output.weight"));
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
        const std::string fuse=std::getenv("TQLLM_MLX_NAX_FUSE")?std::getenv("TQLLM_MLX_NAX_FUSE"):"";
        std::vector<std::vector<std::string>> fused_sets;
        if(fuse.find("ffn")!=std::string::npos)fused_sets.push_back({"ffn_gate.weight","ffn_up.weight"});
        if(fuse.find("mixer")!=std::string::npos)fused_sets.push_back({"attn_qkv.weight","attn_gate.weight"});
        if(fuse.find("attn")!=std::string::npos)fused_sets.push_back({"attn_q.weight","attn_k.weight","attn_v.weight"});
        if(!fused_sets.empty())
            for(int l=0;l<layers;++l) {
                const std::string p="blk."+std::to_string(l)+".";
                for(const auto& members:fused_sets) {
                    std::vector<const NaxAffine*> parts;
                    for(const auto& m:members)if(auto f=nax.find(p+m);f!=nax.end())parts.push_back(&f->second);
                    if(parts.size()!=members.size())continue;
                    bool same=true;for(const auto* q:parts)same=same&&q->columns==parts[0]->columns&&q->bits==parts[0]->bits&&q->group==parts[0]->group;
                    if(!same)continue;
                    NaxGroup g{nax_affine_concat(parts),{},std::nullopt};
                    int off=0;
                    for(size_t j=0;j<members.size();++j) {
                        const int r=nax.at(p+members[j]).rows;g.parts.push_back({off,r});off+=r;
                        nax_member[p+members[j]]={p+members[0]+"+",int(j)};
                    }
                    for(const auto& m:members)nax.erase(p+m);
                    nax_groups.emplace(p+members[0]+"+",std::move(g));
                }
            }
        caches.resize(layers);
        kv_keys.resize(layers);kv_values.resize(layers);
        tapes.resize(layers);
        if(mma8) {
            mma8_clear_pack_cache();
            size_t bytes=0;for(const auto& [n,m]:mma8_projections)bytes+=m.words.nbytes()+m.scales.nbytes();
            std::fprintf(stderr,"mma8: %zu of %zu matrices packed, %.2f GB\n",mma8_projections.size(),matrices.size(),bytes/1e9);
        }
    }
    bool tiled_active(const std::string& name,const TqTiledWeights& weights,bool& fp32) const {
        if(tqmlx){fp32=false;return true;}
        const bool mixer_projection=name.find(".ffn_")==std::string::npos;
        fp32=mixer_projection?std::getenv("TQLLM_MLX_TILED_MIXER_FP32")!=nullptr:
            std::getenv(weights.bits==3?"TQLLM_MLX_TILED_TQ3_FP32":"TQLLM_MLX_TILED_TQ4_FP32")!=nullptr;
        return fp32 || (mixer_projection && std::getenv("TQLLM_MLX_TILED_MIXER_FP16")) ||
            (!mixer_projection && std::getenv(weights.bits==3?"TQLLM_MLX_TILED_TQ3_FP16":"TQLLM_MLX_TILED_TQ4_FP16"));
    }
    void drop_tiled_original_matrices() {
        if(!drop_tiled_originals)return;
        for(const auto& [name,weights]:tiled_projections) {
            bool fp32=false;
            if(tiled_active(name,weights,fp32))matrices.erase(name);
        }
    }
    std::optional<mx::array> device_ids;  // set by verify_device: the batch's ids stay on the GPU (draft -> verify, no readback)
    mx::array embed(std::span<const uint32_t> tokens) const {
        if(device_ids)return embedding8?affine8_embedding(*embedding8,*device_ids):q4_embedding(*embedding,*device_ids);
        return embedding8?affine8_embedding(*embedding8,tokens):q4_embedding(*embedding,tokens);
    }
    mx::array mat(const std::string& name,const mx::array& x,bool input_rotated=false) {
        auto rotated=[&]() {
            if(input_rotated)return x;
            if(reuse_rotation && rotation && rotation->first.id()==x.id())return rotation->second;
            auto value=tq4_rotate(x);
            if(reuse_rotation)rotation=std::pair{x,value};
            return value;
        };
        if(!ablate.empty() && x.shape(0)<=8)
            for(const auto& raw_tag:ablate) {
                const bool keep_input=raw_tag.rfind("r:",0)==0;
                const std::string tag=keep_input?raw_tag.substr(2):raw_tag;
                if(name.find(tag)!=std::string::npos) {
                    int rows=0;
                    if(auto f=nax.find(name);f!=nax.end())rows=f->second.rows;
                    else if(auto f=nax_tq4.find(name);f!=nax_tq4.end())rows=f->second.rows;
                    else if(auto f=nax_member.find(name);f!=nax_member.end())rows=nax_groups.at(f->second.first).parts[f->second.second].second;
                    else if(auto it=matrices.find(name);it!=matrices.end())
                        std::visit([&](const auto& w){
                            if constexpr(std::is_same_v<std::decay_t<decltype(w)>,mx::array>)rows=w.shape(0);
                            else if constexpr(std::is_same_v<std::decay_t<decltype(w)>,Tq3Weights>)rows=w.planes.shape(0);
                            else if constexpr(std::is_same_v<std::decay_t<decltype(w)>,Tq4Weights>)rows=w.codes.shape(0);
                            else rows=w.blocks.shape(0);
                        },it->second);
                    if(rows) {
                        auto zero=mx::zeros({x.shape(0),rows},x.dtype());
                        if(keep_input)zero=mx::add(zero,mx::astype(mx::multiply(mx::sum(rotated()),mx::array(0.0f)),x.dtype()));
                        return zero;
                    }
                }
            }
        // prefill chunks: dequantize once and run one GEMM instead of re-decoding the matrix per 32-row tile
        static const int gemm_rows=std::getenv("TQLLM_MLX_NAX_PREFILL_GEMM")?std::atoi(std::getenv("TQLLM_MLX_NAX_PREFILL_GEMM")):64;  // default on with the GEMM FFN; =0 off
        if(gemm_rows>0 && x.shape(0)>=gemm_rows) {
            // one FP16 copy per distinct input (projections of one input share it); TQ4 inputs rotate in one launch
            auto half_input=[&](bool tq4) {
                auto& slot=tq4?gemm_rotated_input:gemm_input;
                if(!slot || slot->first.id()!=x.id())slot=std::pair{x,input_rotated && tq4?mx::astype(x,mx::float16):tq4?tq4_rotate_half(x):mx::astype(x,mx::float16)};
                return slot->second;
            };
            if(auto found=nax_tq4.find(name);found!=nax_tq4.end())
                return prefill_int8&&(prefill_int8_mask&1)?nax_int8_gemm(found->second,half_input(true),true,prefill_int8==2):nax_affine_gemm(found->second,half_input(true),true);
            if(auto member=nax_member.find(name);member!=nax_member.end()) {
                auto& g=nax_groups.at(member->second.first);
                if(!g.last || g.last->first.id()!=x.id())g.last=std::pair{x,nax_affine_gemm(g.weights,x,true)};
                const auto [off,rows]=g.parts[member->second.second];
                return mx::slice(g.last->second,{0,off},{x.shape(0),off+rows});
            }
            if(auto found=nax.find(name);found!=nax.end())return nax_affine_gemm(found->second,half_input(false),true);
        }
        if(auto found=nax_tq4.find(name);found!=nax_tq4.end()) {
            static const bool fused_rotation=std::getenv("TQLLM_MLX_TQ4_ROTATE_UNFUSED")==nullptr;
            if(fused_rotation && !input_rotated) {
                if(!nax_rotated_input || nax_rotated_input->first.id()!=x.id())nax_rotated_input=std::pair{x,tq4_rotate_half(x)};
                return nax_affine_matmul(found->second,nax_rotated_input->second,true);
            }
            auto xr=rotated();
            if(!nax_rotated_input || nax_rotated_input->first.id()!=xr.id())
                nax_rotated_input=std::pair{xr,mx::contiguous(mx::astype(xr,mx::float16))};
            return nax_affine_matmul(found->second,nax_rotated_input->second,true);
        }
        if(auto member=nax_member.find(name);member!=nax_member.end()) {
            auto& g=nax_groups.at(member->second.first);
            if(!g.last || g.last->first.id()!=x.id())g.last=std::pair{x,nax_affine_matmul(g.weights,x,true)};
            const auto [off,rows]=g.parts[member->second.second];
            return mx::slice(g.last->second,{0,off},{x.shape(0),off+rows});
        }
        if(auto found=nax.find(name);found!=nax.end()) {
            if(!std::getenv("TQLLM_MLX_NAX_NO_INPUT_CACHE") && (!nax_input || nax_input->first.id()!=x.id()))
                nax_input=std::pair{x,mx::contiguous(mx::astype(x,mx::float16))};
            auto y=nax_affine_matmul(found->second,std::getenv("TQLLM_MLX_NAX_NO_INPUT_CACHE")?x:nax_input->second,true);
            if(std::getenv("TQLLM_MLX_NAX_ROW_CHECK") && x.shape(0)>1) {
                auto full=mx::contiguous(y);mx::eval(full,x);
                const auto xf=mx::astype(x,mx::float32);mx::eval(xf);
                for(int r=0;r<x.shape(0);++r) {
                    auto one=mx::contiguous(nax_affine_matmul(found->second,mx::contiguous(tail_slice(x,0,r,r+1)),true));mx::eval(one);
                    if(std::memcmp(one.data<float>(),full.data<float>()+size_t(r)*one.size(),one.nbytes())) {
                        double peak=0;const float* xp=xf.data<float>();for(size_t j=0;j<xf.size();++j)peak=std::max(peak,double(std::abs(xp[j])));
                        std::fprintf(stderr,"NAX_ROW_MISMATCH %s row=%d m=%d split=%d dtype=%d contiguous=%d max_abs_x=%g\n",name.c_str(),r,x.shape(0),
                                     nax_affine_splits(found->second),int(x.dtype().val()),int(x.flags().row_contiguous),peak);
                        break;
                    }
                }
            }
            return y;
        }
        if(mma8 && x.shape(0)<=8)
            if(auto found=mma8_projections.find(name);found!=mma8_projections.end()) {
                if(!rotation_half || !rotation || rotation_half->first.id()!=x.id())
                    rotation_half=std::pair{x,mma8_input(rotated())};
                return mma8_matmul(found->second,rotation_half->second,x.shape(0),true);
            }
        if(x.shape(0)<=8)
            if(auto found=tiled_projections.find(name);found!=tiled_projections.end()) {
                bool fp32=false;
                if(tiled_active(name,found->second,fp32))return bf(tq_tiled_matmul(found->second,rotated(),4,true,32,fp32));
            }
        if(tqmlx) {
            if(auto found=tiled_projections.find(name);found!=tiled_projections.end())
                return bf(tq_mlx_prefill(found->second,rotated()));
            if(auto found=mma8_projections.find(name);found!=mma8_projections.end())
                return bf(tq_mlx_prefill(found->second,rotated()));
        }
        auto it=matrices.find(name);
        if(it==matrices.end())throw std::invalid_argument("matrix has no original weights: "+name);
        const auto& w=it->second;
        if(capture_prefix && serial_verify_dense && x.shape(0)>1 && std::holds_alternative<mx::array>(w)) {
            if(batch_decode_dense) {
                if(!dense_input || dense_input->first.id()!=x.id())dense_input=std::pair{x,dense_row_views(x)};
                return bf(dense_decode_rows(std::get<mx::array>(w),dense_input->second));
            }
            std::vector<mx::array> rows;
            rows.reserve(x.shape(0));
            for(int t=0;t<x.shape(0);++t)
                rows.push_back(bf(linear(w,mx::contiguous(tail_slice(x,0,t,t+1)))));
            return mx::concatenate(rows,0);
        }
        if(x.shape(0)>1 && !(gemm_prefill && x.shape(0)>8))
            if(auto found=affine_ffn.find(name);found!=affine_ffn.end()) {
                const bool fp32=(affine_tq4_fp32 && std::holds_alternative<Tq4Weights>(w)) ||
                    (affine_mixer && name.find(".ffn_")==std::string::npos);
                return bf(fp32?affine8_matmul(found->second,rotated(),false):tq3_affine8_matmul(found->second,rotated()));
            }
        if(x.shape(0)>1 && !(gemm_prefill && x.shape(0)>8))
            if(auto found=decoded_ffn.find(name);found!=decoded_ffn.end())
                return bf(tq3_expanded_matmul(found->second,rotated()));
        if((reuse_rotation || input_rotated) && !(gemm_prefill && x.shape(0)>8)) {
            if(auto packed=std::get_if<Tq3Weights>(&w))return bf(tq3_matmul(*packed,rotated()));
            if(auto packed=std::get_if<Tq4Weights>(&w))
                return bf(tq4_matmul(*packed,rotated(),1,matmul_columns(x.shape(0)),matmul_simds()));
        }
        return bf(gemm_prefill && x.shape(0)>8?prefill_linear(w,x):linear(w,x));
    }
    mx::array norm(const mx::array& x,const std::string& name) {
        static const bool skip=ablate_small("norm");
        return skip?x:bf(mx::fast::rms_norm(x,plain.at(name),eps));
    }
    std::optional<mx::array> paired_ff(const std::string& p,const mx::array& x) {
        if(!paired_gate_up || !std::getenv("TQLLM_MLX_PAIRED_GATE_UP") || x.shape(0)>8)return std::nullopt;
        auto g=tiled_projections.find(p+"ffn_gate.weight"),u=tiled_projections.find(p+"ffn_up.weight");
        if(g==tiled_projections.end() || u==tiled_projections.end() ||
           g->second.bits!=u->second.bits || g->second.integer_levels!=u->second.integer_levels ||
           g->second.rows!=u->second.rows || g->second.columns!=u->second.columns)return std::nullopt;
        const bool fp32=std::getenv(g->second.bits==3?"TQLLM_MLX_TILED_TQ3_FP32":"TQLLM_MLX_TILED_TQ4_FP32")!=nullptr;
        if(!fp32 && !std::getenv(g->second.bits==3?"TQLLM_MLX_TILED_TQ3_FP16":"TQLLM_MLX_TILED_TQ4_FP16"))return std::nullopt;
        return paired_swiglu_rotation?tq_paired_swiglu_rotated(g->second,u->second,tq4_rotate(x),fp32):
            tq_paired_swiglu(g->second,u->second,tq4_rotate(x),fp32);
    }
    std::pair<mx::array,mx::array> gate_up(const std::string& p,const mx::array& x) {
        if(x.shape(0)>1 && !(gemm_prefill && x.shape(0)>8))
            if(auto found=affine_gu.find(p);found!=affine_gu.end()) {
                const bool fp32=affine_tq4_fp32 && std::holds_alternative<Tq4Weights>(matrices.at(p+"ffn_gate.weight"));
                auto gu=fp32?affine8_matmul(found->second,tq4_rotate(x),false):tq3_affine8_matmul(found->second,tq4_rotate(x));
                const int rows=gu.shape(1)/2;
                return {bf(tail_slice(gu,1,0,rows)),bf(tail_slice(gu,1,rows,2*rows))};
            }
        return {mat(p+"ffn_gate.weight",x),mat(p+"ffn_up.weight",x)};
    }
    // Prefill FFN on MLX's FP16 GEMM (j601: 31 TFLOPS at 2048 rows vs ~22 for the row kernel): the three matrices are
    // dequantized to FP16 once per chunk, swiglu runs in one compiled kernel with the BF16 roundings of the row path,
    // and its FP16 result feeds the down GEMM directly. TQLLM_MLX_FFN_GEMM=<min rows> (prefill chunks only).
    std::optional<mx::array> ffn_gemm(const std::string& p,const mx::array& n) {
        static const int min_rows=std::getenv("TQLLM_MLX_FFN_GEMM")?std::atoi(std::getenv("TQLLM_MLX_FFN_GEMM")):64;  // j604: +47-50 % prefill, exact; =0 off
        if(min_rows<=0 || n.shape(0)<min_rows)return std::nullopt;
        auto find=[&](const std::string& name)->const NaxAffine* {
            if(auto f=nax_tq4.find(name);f!=nax_tq4.end())return &f->second;
            if(auto f=nax.find(name);f!=nax.end())return &f->second;
            return nullptr;
        };
        const auto *g=find(p+"ffn_gate.weight"),*u=find(p+"ffn_up.weight"),*d=find(p+"ffn_down.weight");
        if(!g || !u || !d || g->q6 || u->q6 || d->q6)return std::nullopt;
        auto x16=g->tq4?tq4_rotate_half(n):mx::astype(n,mx::float16);
        // TQLLM_MLX_DUMP_FFN=dir:layer,layer,...: save this layer's FFN input (FP16, rotated for TQ4) per chunk and
        // its dequantized FP16 weights once, as .npy (Neural Engine split calibration and quality checks)
        static const std::string dump=std::getenv("TQLLM_MLX_DUMP_FFN")?std::getenv("TQLLM_MLX_DUMP_FFN"):"";
        if(!dump.empty()) {
            const auto colon=dump.find(':');const std::string dir=dump.substr(0,colon),layers=","+dump.substr(colon+1)+",";
            const std::string layer=p.substr(4,p.size()-5);
            if(layers.find(","+layer+",")!=std::string::npos) {
                static std::map<std::string,int> chunk;
                const int c=chunk[layer]++;
                mx::save(dir+"/x_l"+layer+"_c"+std::to_string(c)+".npy",x16);
                if(c==0) {
                    mx::save(dir+"/wg_l"+layer+".npy",nax_affine_dequant(*g));
                    mx::save(dir+"/wu_l"+layer+".npy",nax_affine_dequant(*u));
                    mx::save(dir+"/wd_l"+layer+".npy",nax_affine_dequant(*d));
                }
            }
        }
        const bool int8_gu=prefill_int8 && (prefill_int8_mask&2), int8_down=prefill_int8 && (prefill_int8_mask&4);
        auto gv=int8_gu && g->tq4?nax_int8_gemm(*g,x16,false,prefill_int8==2):mx::matmul(x16,mx::transpose(nax_affine_dequant(*g)));
        auto uv=int8_gu && u->tq4?nax_int8_gemm(*u,x16,false,prefill_int8==2):mx::matmul(x16,mx::transpose(nax_affine_dequant(*u)));
        static auto swiglu16=mx::compile([](const std::vector<mx::array>& a) {
            auto gate=bf_raw(mx::astype(a[0],mx::float32)),up=bf_raw(mx::astype(a[1],mx::float32));
            return std::vector<mx::array>{mx::astype(bf_raw(mx::multiply(bf_raw(mx::multiply(gate,mx::sigmoid(gate))),up)),mx::float16)};
        },true);
        auto h=swiglu16({gv,uv})[0];
        if(stage_profiling)stage("ffn_gate_up",{h});
        auto hin=d->tq4?tq4_rotate_half(h):h;
        auto y=int8_down && d->tq4?nax_int8_gemm(*d,hin,false,prefill_int8==2):mx::astype(mx::matmul(hin,mx::transpose(nax_affine_dequant(*d))),mx::float32);
        // TQLLM_MLX_ANE_NOISE=<rel>: add Gaussian noise of norm rel * |y| (Neural Engine slice error budget, debug only)
        static const float ane_noise=std::getenv("TQLLM_MLX_ANE_NOISE")?std::stof(std::getenv("TQLLM_MLX_ANE_NOISE")):0.0f;
        if(ane_noise>0) {
            static int calls=0;
            auto e=mx::random::normal(y.shape(),mx::float32,0.0f,1.0f,mx::random::key(uint64_t(++calls)));
            auto scale=mx::multiply(mx::array(ane_noise),mx::sqrt(mx::divide(mx::sum(mx::square(y)),mx::sum(mx::square(e)))));
            y=mx::add(y,mx::multiply(e,scale));
        }
        return bf_raw(y);
    }
    mx::array full_attention(const std::string& p,const mx::array& x,Cache& c,int offset,int layer) {
        const int m=x.shape(0);
        auto qg=mx::reshape(mat(p+"attn_q.weight",x),{m,heads,2*hd});
        auto q=tail_slice(qg,2,0,hd),gate=tail_slice(qg,2,hd,2*hd);
        q=mx::transpose(mx::reshape(norm(q,p+"attn_q_norm.weight"),{1,m,heads,hd}),{0,2,1,3});
        auto k=mx::reshape(mat(p+"attn_k.weight",x),{1,m,kvheads,hd});
        k=mx::transpose(norm(k,p+"attn_k_norm.weight"),{0,2,1,3});
        auto v=mx::transpose(mx::reshape(mat(p+"attn_v.weight",x),{1,m,kvheads,hd}),{0,2,1,3});
        q=bf(mx::fast::rope(q,rot,false,theta,1.0f,offset));k=bf(mx::fast::rope(k,rot,false,theta,1.0f,offset));
        stage("attn_proj",{q,k,v,gate});
        if(layer<0) {  // MTP cache: appended by concatenation (its inject path does the same)
            c.keys=c.keys?mx::concatenate({*c.keys,k},2):k;
            c.values=c.values?mx::concatenate({*c.values,v},2):v;
            auto out=causal_attention_rows(q,*c.keys,*c.values,1.0f/std::sqrt(float(hd)),serial_verify_attention&&capture_prefix,attention_kv_views,fused_attention_rows);
            out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
            out=gate_mul_bf(out,gate);
            return mat(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
        }
        auto& kb=kv_keys[layer];auto& vb=kv_values[layer];
        const int need=c.kv_len+m;
        // keys and values are already BF16-rounded, so BF16 storage is exact and halves the cache
        static const auto kv_type=std::getenv("TQLLM_MLX_KV_F32")?mx::float32:mx::bfloat16;
        k=mx::astype(k,kv_type);v=mx::astype(v,kv_type);
        if(!kb || kb->shape(2)<need || kb->dtype()!=k.dtype()) {
            const int cap=(need+255)/256*256+256;
            auto nk=mx::zeros({1,kvheads,cap,hd},k.dtype()),nv=mx::zeros({1,kvheads,cap,hd},v.dtype());
            if(kb && c.kv_len>0) {
                nk=mx::slice_update(nk,mx::slice(*kb,{0,0,0,0},{1,kvheads,c.kv_len,hd}),{0,0,0,0},{1,kvheads,c.kv_len,hd});
                nv=mx::slice_update(nv,mx::slice(*vb,{0,0,0,0},{1,kvheads,c.kv_len,hd}),{0,0,0,0},{1,kvheads,c.kv_len,hd});
            }
            kb=nk;vb=nv;
        }
        kb=mx::slice_update(*kb,k,{0,0,c.kv_len,0},{1,kvheads,need,hd});
        vb=mx::slice_update(*vb,v,{0,0,c.kv_len,0},{1,kvheads,need,hd});
        c.kv_len=need;
        auto out=causal_attention_rows(q,*kb,*vb,1.0f/std::sqrt(float(hd)),serial_verify_attention&&capture_prefix,attention_kv_views,fused_attention_rows,need);
        out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
        stage("attn_core",{out,*kb,*vb});
        out=gate_mul_bf(out,gate);
        return mat(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
    }
    mx::array gdn(const std::string& p,const mx::array& x,Cache& c,PrefixTape& tape) {
        const int m=x.shape(0),width=2*kh*dk+vh*dv;
        auto qkv=mat(p+"attn_qkv.weight",x);
        stage("gdn_proj_in",{qkv});
        auto history=c.conv?*c.conv:mx::zeros({conv-1,width});
        auto joined=mx::concatenate({history,qkv},0);
        if(capture_prefix)tape.conv_inputs=joined;
        auto weights=mx::reshape(plain.at(p+"ssm_conv1d.weight"),{width,conv});
        // fused prep/post kernels: prefill chunks by default (TQLLM_MLX_GDN_FUSE_PREFILL=0 off), every batch with TQLLM_MLX_GDN_FUSE=1
        static const bool gdn_fuse_all=std::getenv("TQLLM_MLX_GDN_FUSE")!=nullptr;
        // off by default: j606 neutral at 8K/16K, slower at 2K (362 vs 399 tok/s); =1 enables
        static const bool gdn_fuse_prefill=std::getenv("TQLLM_MLX_GDN_FUSE_PREFILL") && std::string(std::getenv("TQLLM_MLX_GDN_FUSE_PREFILL"))!="0";
        const bool gdn_fuse=gdn_fuse_all || (gdn_fuse_prefill && m>8);
        const bool fused_pre=gdn_fuse && ablate_blocks.find("gnorm")==std::string::npos && dk==128 && dv==128;
        auto result=ablate_blocks.find("gconv")!=std::string::npos?tail_slice(joined,0,conv-1,conv-1+m):
            fused_pre?gdn_pre_fused(joined,weights,kh,1.0f/std::sqrt(float(dk))):
            native_gdn_prep?gdn_convolve_native(joined,weights):gdn_convolve(joined,weights,compiled_gdn);
        c.conv=mx::contiguous(tail_slice(joined,0,m,m+conv-1));
        auto q=mx::reshape(tail_slice(result,1,0,kh*dk),{1,m,kh,dk});
        auto k=mx::reshape(tail_slice(result,1,kh*dk,2*kh*dk),{1,m,kh,dk});
        auto v=mx::reshape(tail_slice(result,1,2*kh*dk,width),{1,m,vh,dv});
        if(!fused_pre && ablate_blocks.find("gnorm")==std::string::npos) {
            q=gdn_normalize(q,1.0f/std::sqrt(float(dk)),compiled_gdn);
            k=gdn_normalize(k,1,compiled_gdn);
        }
        if(!native_head_order && !scan_gdn) {q=mx::take(q,key_indices,2);k=mx::take(k,key_indices,2);}
        auto a=mat(p+"ssm_alpha.weight",x),b=mat(p+"ssm_beta.weight",x);
        auto gates=native_gdn_prep?gdn_gates_native(a,b,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a")):
            std::pair{gdn_decay(a,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a"),compiled_gdn),mx::sigmoid(b)};
        auto decay=mx::reshape(gates.first,{1,m,vh}),beta=mx::reshape(gates.second,{1,m,vh});
        stage("gdn_prep",{q,k,v,decay,beta});
        // prefill chunks without prefix capture: chunked form (matmuls over 64-token sub-chunks) instead of the
        // token-by-token scan; opt-in TQLLM_MLX_GDN_CHUNKED=1 (j621: exact ids but 2.27 vs 1.32 s at 8K, slower than the scan)
        static const bool chunked_gdn=std::getenv("TQLLM_MLX_GDN_CHUNKED") && std::string(std::getenv("TQLLM_MLX_GDN_CHUNKED"))!="0";
        if(chunked_gdn && !capture_prefix && m>=128 && dk==128 && dv==128) {
            auto [o,st]=gdn_chunked(q,k,v,decay,beta,c.state,64);
            c.state=st;
            stage("gdn_scan",{o,st});
            auto z=mx::reshape(mat(p+"attn_gate.weight",x),{1,m,vh,dv});
            auto out=gdn_fuse?gdn_post_fused(o,z,plain.at(p+"ssm_norm.weight"),eps):
                silu_mul_bf(mx::fast::rms_norm(bf(o),plain.at(p+"ssm_norm.weight"),eps),z);
            return mat(p+"ssm_out.weight",mx::reshape(out,{m,vh*dv}));
        }
        auto scanned=scan_gdn?std::optional<GdnScan>(gdn_scan(q,k,v,decay,beta,c.state,capture_prefix,capture_prefix&&compact_gdn)):std::nullopt;
        if(capture_prefix) {
            tape.states=scanned->snapshots;
            if(compact_gdn) {tape.keys=k;tape.decay=decay;tape.deltas=scanned->deltas;}
        }
        auto update=scanned?std::vector<mx::array>{scanned->output,scanned->state}:native_head_order?gdn_native(q,k,v,decay,beta,c.state,*native_head_order):
            mx::fast::gated_delta_update(q,k,v,decay,beta,c.state);
        c.state=update[1];
        stage("gdn_scan",{update[0],update[1]});
        auto z=mx::reshape(mat(p+"attn_gate.weight",x),{1,m,vh,dv});
        auto out=gdn_fuse?gdn_post_fused(update[0],z,plain.at(p+"ssm_norm.weight"),eps):
            silu_mul_bf(mx::fast::rms_norm(bf(update[0]),plain.at(p+"ssm_norm.weight"),eps),z);
        return mat(p+"ssm_out.weight",mx::reshape(out,{m,vh*dv}));
    }
    mx::array shortlist_rows(const mx::array& x, int& vocab) const {
        if(!proposal_head)throw std::invalid_argument("target-head shortlist is not configured");
        const bool skip_matmul=ablate_blocks.find("hm")!=std::string::npos;
        auto approximate=skip_matmul?mx::zeros({x.shape(0),248320},mx::float32):draft_head4_matmul(*proposal_head,x);
        const int m=approximate.shape(0),v=approximate.shape(1),n=std::min(target_head_shortlist,v);
        vocab=v;
        if(ablate_blocks.find("ht")!=std::string::npos)
            return mx::astype(mx::broadcast_to(mx::arange(n),{m,n}),mx::uint32);
        if(!std::getenv("TQLLM_MLX_TARGET_HEAD_HOST"))
            return shard_topk_ids(approximate,n);
        mx::eval(approximate);
        std::vector<uint32_t> ids(size_t(m)*n);
        const float* p=approximate.data<float>();
        std::vector<uint32_t> order(static_cast<size_t>(v));
        for(int row=0;row<m;++row) {
            std::iota(order.begin(),order.end(),0);
            const float* base=p+size_t(row)*v;
            std::partial_sort(order.begin(),order.begin()+n,order.end(),
                [&](uint32_t a,uint32_t b) {return base[a]>base[b];});
            std::copy_n(order.begin(),n,ids.begin()+size_t(row)*n);
        }
        return mx::array(ids.data(),{m,n},mx::uint32);
    }
    mx::array shortlist_ids(const mx::array& x) const {
        const auto* head=std::get_if<Q6Weights>(&matrices.at("output.weight"));
        if(!head)throw std::invalid_argument("target-head shortlist requires Q6_K output weights");
        int vocab=0;
        auto rows=shortlist_rows(x,vocab);
        auto scores=ablate_blocks.find("hq")!=std::string::npos?mx::zeros(rows.shape(),mx::float32):q6_rows_matmul(*head,x,rows);
        auto best=mx::argmax(scores,1);
        auto selected=mx::take_along_axis(rows,mx::expand_dims(best,1),1);
        return mx::reshape(selected,{x.shape(0)});
    }
    mx::array shortlist_head(const mx::array& x) const {
        const auto* head=std::get_if<Q6Weights>(&matrices.at("output.weight"));
        if(!head)throw std::invalid_argument("target-head shortlist requires Q6_K output weights");
        int vocab=0;
        auto rows=shortlist_rows(x,vocab);
        auto scores=q6_rows_matmul(*head,x,rows);
        return sparse_logits(rows,scores,vocab);
    }
    mx::array forward(std::span<const uint32_t> tokens,bool all_logits) {
        auto start=std::chrono::steady_clock::now();
        if(tokens.empty()||tokens.size()>size_t(context-pos)) throw std::invalid_argument("MLX token batch exceeds remaining context");
        stages.clear();rotation.reset();rotation_half.reset();dense_input.reset();
        stage_tick=std::chrono::steady_clock::now();
        auto record=[&](const char* name,const std::vector<mx::array>& arrays) {stage(name,arrays);};
        const int m=int(tokens.size());auto x=bf(embed(tokens));
        std::vector<mx::array> features;
        if(stage_profiling)record("embedding",{x});
        for(int l=0;l<layers;++l) {
            if(std::binary_search(feature_ids.begin(),feature_ids.end(),l))features.push_back(x);
            const std::string p="blk."+std::to_string(l)+".";
            auto n=norm(x,p+"attn_norm.weight");
            const bool is_attn=(l+1)%interval==0;
            auto attn=ablate_blocks.find(is_attn?"attn":"gdn")!=std::string::npos?mx::zeros(x.shape(),x.dtype()):
                is_attn?full_attention(p,n,caches[l],pos,l):gdn(p,n,caches[l],tapes[l]);
            if(stage_profiling) {
                std::vector<mx::array> values={attn};const auto& c=caches[l];
                for(const std::optional<mx::array>* a:std::initializer_list<const std::optional<mx::array>*>{&c.conv,&c.state,&kv_keys[l],&kv_values[l]})if(*a)values.push_back(**a);
                record((l+1)%interval==0?"attention":"gdn_mixer",values);
            }
            x=add_bf(x,attn);n=norm(x,p+"post_attention_norm.weight");
            if(ablate_blocks.find("ffn")==std::string::npos)
                if(auto down=ffn_gemm(p,n)) {
                    x=add_bf(x,*down);
                    if(stage_profiling)record("ffn_down",{x});
                    continue;
                }
            auto paired=paired_ff(p,n);
            bool ff_rotated=paired && paired_swiglu_rotation;
            auto ff=paired?*paired:[&] {
                auto [gate,up]=gate_up(p,n);
                if(fused_swiglu_rotation && x.shape(0)>1) {
                    ff_rotated=true;
                    return swiglu_rotated(gate,up);
                }
                return fused_epilogue?swiglu(gate,up):bf(mx::multiply(bf(silu(gate)),up));
            }();
            if(stage_profiling)record("ffn_gate_up",{ff});
            if(ablate_blocks.find("ffn")==std::string::npos)x=add_bf(x,mat(p+"ffn_down.weight",ff,ff_rotated));
            if(stage_profiling)record("ffn_down",{x});
        }
        last_hidden=x;
        if(!features.empty())last_features=mx::concatenate(features,1);
        x=norm(x,"output_norm.weight");
        if(!all_logits) x=tail_slice(x,0,m-1,m);
        std::vector<mx::array> live={x};
        if(last_features)live.push_back(*last_features);
        if(!(capture_prefix && lazy_verify_commit)) {
            for(const auto& c:caches) {
                if(c.conv) live.push_back(*c.conv);
                if(c.state) live.push_back(*c.state);
            }
            for(const auto* kv:{&kv_keys,&kv_values})for(const auto& a:*kv)if(a)live.push_back(*a);
            if(capture_prefix)for(const auto& t:tapes)
                for(const auto* a:{&t.states,&t.keys,&t.decay,&t.deltas})if(*a)live.push_back(**a);
        }
        if(profiling) mx::eval(live);
        auto body_end=std::chrono::steady_clock::now();
        auto logits=return_head_ids && target_head_shortlist && x.shape(0)<=8?shortlist_ids(x):
            target_head_shortlist && x.shape(0)<=8?shortlist_head(x):
            target_head && x.shape(0)<=8?q6_affine_matmul(*target_head,x):
            nax.contains("output.weight")?nax_affine_matmul(nax.at("output.weight"),x):
            gemm_verify && x.shape(0)>1 && x.shape(0)<=8?verify_tiled(matrices.at("output.weight"),x):
            linear(matrices.at("output.weight"),x);
        live.push_back(logits);
        if(std::getenv("TQLLM_MLX_COUNT_NODES")) {
            std::ostringstream dot;mx::export_to_dot(dot,live);
            size_t nodes=0,pos=0;const std::string text=dot.str();
            while((pos=text.find("->",pos))!=std::string::npos){++nodes;pos+=2;}
            size_t prims=0;for(size_t q=0;(q=text.find("shape=rectangle",q))!=std::string::npos;q+=15)++prims;
            std::fprintf(stderr,"graph primitives m=%d: %zu (edges %zu)\n",m,prims,nodes);
            std::map<std::string,size_t> histogram;
            for(size_t at=0;(at=text.find("label =\"",at))!=std::string::npos;) {
                at+=8;const size_t end=text.find('"',at);
                if(end==std::string::npos)break;
                ++histogram[text.substr(at,end-at)];at=end;
            }
            std::vector<std::pair<size_t,std::string>> ranked;
            for(const auto& [name,count]:histogram)ranked.push_back({count,name});
            std::sort(ranked.rbegin(),ranked.rend());
            for(size_t i=0;i<ranked.size() && i<28;++i)std::fprintf(stderr,"graph node %-28s %zu\n",ranked[i].second.c_str(),ranked[i].first);
        }
        mx::eval(live);
        if(m>=8)drop_tiled_original_matrices();
        auto end=std::chrono::steady_clock::now();
        times={std::chrono::duration<double,std::milli>(body_end-start).count(),
               std::chrono::duration<double,std::milli>(end-body_end).count()};
        pos+=m;++revision;rotation.reset();rotation_half.reset();dense_input.reset();
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
mx::array Qwen35::verify_choices(std::span<const uint32_t> tokens,Verification& batch) {
    if(!impl_->scan_gdn || impl_->native_head_order || tokens.empty() || tokens.size()>8)
        throw std::invalid_argument("MLX prefix verification requires scan GDN and 1-8 tokens");
    batch.state_.reset();
    const auto before=checkpoint();
    impl_->capture_prefix=true;
    for(auto& tape:impl_->tapes)tape=PrefixTape{};
    try {
        impl_->return_head_ids=true;
        auto logits=impl_->forward(tokens,true);
        impl_->return_head_ids=false;
        batch.state_=std::make_shared<Verification::State>(Verification::State{
            impl_.get(),impl_->epoch,impl_->revision,before.state_->pos,int(tokens.size()),
            before.state_->caches,impl_->caches,impl_->tapes,before.state_->hidden,impl_->last_hidden,
            before.state_->features,impl_->last_features});
        impl_->capture_prefix=false;
        for(auto& tape:impl_->tapes)tape=PrefixTape{};
        return logits;
    } catch(...) {
        impl_->return_head_ids=false;
        impl_->capture_prefix=false;
        for(auto& tape:impl_->tapes)tape=PrefixTape{};
        restore(before);
        throw;
    }
}
mx::array Qwen35::verify_device(const mx::array& ids,Verification& batch,bool choices) {
    if(ids.ndim()!=1 || ids.dtype()!=mx::uint32 || ids.shape(0)<1 || ids.shape(0)>8)
        throw std::invalid_argument("device verification needs uint32 ids [1..8]");
    const std::vector<uint32_t> placeholder(size_t(ids.shape(0)),0u);
    impl_->device_ids=ids;
    try {
        auto out=choices?verify_choices(placeholder,batch):verify(placeholder,batch);
        impl_->device_ids.reset();
        return out;
    } catch(...) {impl_->device_ids.reset();throw;}
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
            c.kv_len=s->start+int(count);  // positions past the length are rewritten by the next append
        } else {
            c.conv=mx::contiguous(tail_slice(*tape.conv_inputs,0,int(count),int(count)+impl_->conv-1));
            c.state=tape.deltas?gdn_replay_prefix(*tape.keys,*tape.decay,*tape.deltas,s->before[l].state,int(count)):
                mx::contiguous(tail_slice(*tape.states,0,int(count)-1,int(count)));
        }
    }
    std::vector<mx::array> live;
    for(const auto& c:caches)for(const auto* a:{&c.conv,&c.state})if(*a)live.push_back(**a);
    static const bool lazy=!std::getenv("TQLLM_MLX_EAGER_COMMIT") && !std::getenv("TQLLM_MLX_EAGER_ACCEPT");  // replay runs inside the next forward's graph
    if(!lazy)mx::eval(live);
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
mx::array Qwen35::embed_tokens(std::span<const uint32_t> tokens) const {return bf(impl_->embed(tokens));}
mx::array Qwen35::shared_draft_head(const mx::array& x) const {
    if(x.ndim()!=2 || x.shape(1)!=impl_->hidden || x.dtype()!=mx::float32)
        throw std::invalid_argument("invalid shared draft-head input");
    if(impl_->proposal_head)return draft_head4_matmul(*impl_->proposal_head,x);
    if(impl_->draft_nax_head)return nax_affine_matmul(*impl_->draft_nax_head,x);
    if(auto found=impl_->nax.find("output_draft.weight");found!=impl_->nax.end())return nax_affine_matmul(found->second,x);
    if(auto found=impl_->nax.find("output.weight");found!=impl_->nax.end())return nax_affine_matmul(found->second,x);
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
    if(auto head=impl_->nax.find("output.weight");head!=impl_->nax.end()) {
        // the drafter scores only the target's `count` most likely tokens for this prompt (proposals only; verify uses the full head)
        if(!std::getenv("TQLLM_MLX_NAX_DRAFT_SHORTLIST") || count%128)return;
        auto ordered=mx::reshape(mx::argsort(logits,-1),{impl_->vocab});
        impl_->mtp_ids=mx::contiguous(mx::astype(tail_slice(ordered,0,impl_->vocab-count,impl_->vocab),mx::uint32));mx::eval(*impl_->mtp_ids);
        const auto& h=head->second;
        impl_->draft_nax_head=nax_affine_select_rows(impl_->nax_head_payload,h.rows,h.columns,h.bits,h.group,
            {impl_->mtp_ids->data<uint32_t>(),size_t(count)});
        return;
    }
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
    auto embedded=bf(i.embed(tokens));
    auto joined=mx::concatenate({i.norm(embedded,p+"nextn.enorm.weight"),i.norm(hidden,p+"nextn.hnorm.weight")},1);
    auto x=i.norm(i.mat(p+"nextn.eh_proj.weight",joined),p+"attn_norm.weight");
    auto k=mx::reshape(i.mat(p+"attn_k.weight",x),{1,n,i.kvheads,i.hd});
    k=mx::transpose(i.norm(k,p+"attn_k_norm.weight"),{0,2,1,3});
    k=bf(mx::fast::rope(k,i.rot,false,i.theta,1.0f,position));
    auto v=mx::transpose(mx::reshape(i.mat(p+"attn_v.weight",x),{1,n,i.kvheads,i.hd}),{0,2,1,3});
    i.mtp_cache.keys=i.mtp_cache.keys?mx::concatenate({*i.mtp_cache.keys,k},2):k;
    i.mtp_cache.values=i.mtp_cache.values?mx::concatenate({*i.mtp_cache.values,v},2):v;
    mx::eval(*i.mtp_cache.keys,*i.mtp_cache.values);i.mtp_pos+=n;i.mtp_hidden.reset();i.rotation.reset();i.rotation_half.reset();
}
mx::array Qwen35::draft_forward(std::span<const uint32_t> tokens,const mx::array& hidden,int position,bool logits) {
    auto& i=*impl_;const int n=int(tokens.size());
    if(!i.mtp_enabled || n<1 || position!=i.mtp_pos || n>i.context-position ||
       hidden.shape()!=mx::Shape{n,i.hidden} || hidden.dtype()!=mx::float32)
        throw std::invalid_argument("invalid MTP token/hidden batch or position");
    const std::string p="blk."+std::to_string(i.layers)+".";
    auto embedded=bf(i.embed(tokens));
    auto joined=mx::concatenate({i.norm(embedded,p+"nextn.enorm.weight"),i.norm(hidden,p+"nextn.hnorm.weight")},1);
    auto x=i.mat(p+"nextn.eh_proj.weight",joined);
    x=bf(mx::add(x,i.full_attention(p,i.norm(x,p+"attn_norm.weight"),i.mtp_cache,position,-1)));
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
    i.mtp_pos+=n;i.rotation.reset();i.rotation_half.reset();
    return output;
}
namespace {
// A tq_mlx file selects the adopted kernel set; explicit environment settings still win.
bool apply_tq_mlx_defaults(const std::string& path) {
    if(!is_tq_mlx_file(gguf_open(path)))return false;
    for(const char* name:{"TQLLM_MLX_GDN_SCAN","TQLLM_MLX_SERIAL_VERIFY_ATTENTION","TQLLM_MLX_SERIAL_VERIFY_DENSE",
        "TQLLM_MLX_COMPACT_GDN_TAPE","TQLLM_MLX_REUSE_ROTATION","TQLLM_MLX_BATCH_DECODE_DENSE","TQLLM_MLX_Q6_AFFINE",
        "TQLLM_MLX_ATTN_DECODE_ROWS","TQLLM_MLX_HEAD_FRAG","TQLLM_MLX_MMA8_DRAFT"})
        setenv(name,"1",0);
    return true;
}
}
Qwen35::Qwen35(const std::string& path,int context):impl_((apply_tq_mlx_defaults(path),std::make_unique<Impl>(path,context))) {}
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

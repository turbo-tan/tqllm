#include "debug.hpp"
#include "release_defaults.hpp"
#include "qwen35.hpp"
#include <cstdio>
#include <sstream>
#include "dense_rows.hpp"
#include "gdn.hpp"
#include "prefill_fused.hpp"
#include "ane_ffn.hpp"
#include "draft_head.hpp"
#include "attention_rows.hpp"
#include "kv_capacity.hpp"
#include "kv_tcq.hpp"
#include "nax_affine.hpp"
#include <cstring>
#include <mlx/compile.h>
#include <mlx/graph_utils.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <limits>
#include <iostream>
#include <atomic>
#include <ctime>
#include <future>
#include <map>
#include <set>
#include <numeric>
#include <optional>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
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
mx::array bf_raw(const mx::array& a) {return mx::astype(mx::astype(a,mx::bfloat16),mx::float32);}
mx::array silu(const mx::array& a) {return mx::multiply(a,mx::sigmoid(a));}
// TQLLM_MLX_COUNT_NODES: Metal dispatches per layer stage. Each primitive is counted once at the stage that first reaches
// it; after the eval a view primitive (reshape, slice, transpose, contiguous ...) whose output shares its input's buffer
// launched nothing, every other primitive is one launch
struct DispatchCount {
    struct Node {mx::array out;std::vector<mx::array> inputs;std::string stage,name;};
    std::vector<Node> nodes;std::set<const void*> seen;
    void take(const std::vector<mx::array>& roots,const std::string& stage) {
        std::vector<mx::array> stack(roots.begin(),roots.end());
        while(!stack.empty()) {
            auto a=stack.back();stack.pop_back();
            if(!a.has_primitive() || !seen.insert(&a.primitive()).second)continue;
            nodes.push_back({a,a.inputs(),stage,a.primitive().name()});  // name now: eval detaches the graph
            for(const auto& in:a.inputs())stack.push_back(in);
        }
    }
    void report(int m,int gdn_layers,int attn_layers) const {
        static const std::set<std::string> views={"Reshape","Slice","Transpose","Broadcast","BroadcastAxes","ExpandDims","Squeeze",
                                                   "AsStrided","Contiguous","Flatten","Unflatten","Copy","StopGradient","Depends"};
        std::map<std::string,std::map<std::string,size_t>> by_stage;std::map<std::string,size_t> totals;size_t all=0,deferred=0;
        for(const auto& n:nodes) {
            // not evaluated by this forward (lazy commit: the accept replay runs inside the next step's graph)
            if(!n.out.is_available()) {++deferred;continue;}
            const std::string& name=n.name;
            bool shared=false;
            if(views.count(name))for(const auto& in:n.inputs)if(in.is_available() && in.buffer().ptr()==n.out.buffer().ptr())shared=true;
            if(shared)continue;
            ++by_stage[n.stage][name];++totals[n.stage];++all;
        }
        std::fprintf(stderr,"DISPATCH m=%d total %zu (gdn layers %d, attention layers %d; %zu deferred to the next graph)\n",m,all,gdn_layers,attn_layers,deferred);
        for(const auto& [stage,count]:totals) {
            const int per=stage.rfind("gdn",0)==0?gdn_layers:stage.rfind("attn",0)==0?attn_layers:1;
            std::string line;
            std::vector<std::pair<size_t,std::string>> ranked;
            for(const auto& [name,c]:by_stage.at(stage))ranked.push_back({c,name});
            std::sort(ranked.rbegin(),ranked.rend());
            for(const auto& [c,name]:ranked)line+=" "+name+" "+std::to_string(double(c)/per).substr(0,5);
            std::fprintf(stderr,"DISPATCH %-12s per layer %.1f:%s\n",stage.c_str(),double(count)/per,line.c_str());
        }
    }
};
mx::array bf(const mx::array& a) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {return std::vector<mx::array>{bf_raw(args[0])};},true);
    return fused({a})[0];
}
mx::array add_bf(const mx::array& a,const mx::array& b) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {return std::vector<mx::array>{bf_raw(mx::add(args[0],args[1]))};},true);
    return fused({a,b})[0];
}
// residual + an FP16 projection product: the product's BF16 rounding and the residual add in one pass (add_bf of the
// rounded product, bit for bit)
mx::array add_product_bf(const mx::array& x,const mx::array& product16) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf_raw(mx::add(args[0],bf_raw(mx::astype(args[1],mx::float32))))};
    },true);
    return fused({x,product16})[0];
}
// an FP16 projection product rounded to BF16 values in FP32, as nax_affine_gemm(..., true)
mx::array round_product(const mx::array& product16) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {return std::vector<mx::array>{bf_raw(mx::astype(args[0],mx::float32))};},true);
    return fused({product16})[0];
}
mx::array gate_mul_bf(const mx::array& out,const mx::array& gate) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf_raw(mx::multiply(bf_raw(args[0]),bf_raw(mx::sigmoid(args[1]))))};
    },true);
    return fused({out,gate})[0];
}
mx::array silu_mul_bf(const mx::array& normed,const mx::array& z) {
    static auto fused=mx::compile([](const std::vector<mx::array>& args) {
        return std::vector<mx::array>{bf_raw(mx::multiply(args[0],silu(args[1])))};
    },true);
    return fused({normed,z})[0];
}
struct Cache {
    std::optional<mx::array> conv,state,keys,values;  // keys/values: MTP cache only
    int kv_len=0;  // full-attention positions filled in the layer's KV buffer (Impl::kv_keys / kv_values)
};
struct PrefixTape {
    // conv_inputs: [history; rows] of a scan verify; the fused step keeps the two parts (conv_prev, conv_rows) instead
    std::optional<mx::array> conv_inputs,conv_prev,conv_rows,states;
    std::optional<mx::array> keys,decay,deltas;
};
}
struct Qwen35::Impl {
    GgufFile file;
    std::map<std::string,Matrix> matrices;  // formats without a matrix-unit copy (TQ3, Q6_K, Q8_0, dense), via linear()
    std::map<std::string,mx::array> ab_proj;  // per GDN layer: ssm_alpha | ssm_beta rows [2 vh, hidden] FP32 (gdn_step_fused projects them)
    std::map<std::string,mx::array> plain;
    std::optional<Q4Embedding> embedding;
    std::optional<Affine8Embedding> embedding8;
    std::map<std::string,NaxAffine> nax;
    std::map<std::string,NaxAffine> nax_tq4;  // TQ4 matrices repacked for the matrix unit (rotated inputs)
    std::optional<std::pair<mx::array,mx::array>> nax_rotated_input;
    std::vector<uint8_t> nax_head_payload;   // host copy of the NAX head, for the drafter's shortlist head
    // prefill packs: matrices that read the same input, concatenated by rows at load (one dequant and one GEMM per pack in
    // prefill, as Splash's packed projections); the members become row views of the pack, so no memory is added and
    // decode reads the same bytes. Each GEMM output column depends only on its own weight row: slices of a pack's
    // product equal the members' products bit for bit.
    std::map<std::string,NaxAffine> packs;
    std::map<std::string,std::vector<int>> pack_rows;
    void pack(const std::string& key,const std::vector<std::string>& names) {
        static const bool on=env_default_on("TQLLM_MLX_PREFILL_PACK");
        if(!on)return;
        auto& map=nax_tq4.count(names[0])?nax_tq4:nax;
        std::vector<NaxAffine*> ms;
        for(const auto& n:names) {
            auto it=map.find(n);
            if(it==map.end())return;
            ms.push_back(&it->second);
        }
        for(const auto* w:ms)
            if(w->bits!=4 || w->q6 || w->rows%32 || w->group!=ms[0]->group || w->tq4!=ms[0]->tq4 || w->columns!=ms[0]->columns ||
               w->biases.ndim()!=ms[0]->biases.ndim())return;
        std::vector<mx::array> words,scales,biases;std::vector<int> rows;
        for(const auto* w:ms) {words.push_back(w->words);scales.push_back(w->scales);biases.push_back(w->biases);rows.push_back(w->rows);}
        NaxAffine P=*ms[0];
        P.words=mx::concatenate(words,0);P.scales=mx::concatenate(scales,0);
        if(P.biases.ndim()==3)P.biases=mx::concatenate(biases,0);
        P.rows=0;for(int r:rows)P.rows+=r;
        // decode rows run the pack as one launch: keep the members' largest split-K (j1054: TQ_L-4S u2 +2.6 ms at split 1)
        P.split=0;for(const auto* w:ms)P.split=std::max(P.split,nax_affine_splits(*w));
        mx::eval(P.words,P.scales,P.biases);
        auto view=[](const mx::array& a,int b0,int b1) {
            mx::Shape start(a.ndim(),0),stop=a.shape();start[0]=b0;stop[0]=b1;
            return mx::slice(a,start,stop);
        };
        int r0=0;
        for(auto* w:ms) {
            w->words=view(P.words,r0/32,(r0+w->rows)/32);w->scales=view(P.scales,r0/32,(r0+w->rows)/32);
            if(P.biases.ndim()==3)w->biases=view(P.biases,r0/32,(r0+w->rows)/32);
            mx::eval(w->words,w->scales,w->biases);
            r0+=w->rows;
        }
        packs.emplace(key,std::move(P));pack_rows.emplace(key,std::move(rows));
    }
    // FP16 product of a 4-bit matrix on a prefill chunk. One-pass chunks run the 4-bit staged kernel on the stored codes
    // (j911: no per-chunk FP16 weight copy, 512 rows 15-17 % faster per matrix); layer-major chunks share one dequant
    // across their slices and run the MLX GEMM on it
    mx::array gemm4(const NaxAffine& w,const mx::array& x16) {
        const bool staged_ok=w.bits==4 && !w.q6 && w.rows%64==0 && w.columns%64==0 && w.group%64==0;
        if(staged_ok && !share_dequant)
            return nax_affine_prefill(w,x16,0,std::nullopt,32,64,4,1,1,1);
        return mx::matmul(x16,mx::transpose(dequant(w)));
    }
    // a prefill pack's FP16 product [m, pack rows] (rows >= 64); nullopt without the pack
    std::optional<mx::array> pack_full16(const std::string& key,const mx::array& x) {
        auto it=packs.find(key);
        if(it==packs.end() || x.shape(0)<64)return std::nullopt;
        const auto& w=it->second;
        auto& slot=w.tq4?gemm_rotated_input:gemm_input;
        if(!slot || slot->first.id()!=x.id())slot=std::pair{x,w.tq4?tq4_rotate_half(x):mx::astype(x,mx::float16)};
        return gemm4(w,slot->second);
    }
    // the same product split per member as views; empty without the pack
    std::vector<mx::array> pack_product16(const std::string& key,const mx::array& x) {
        auto full=pack_full16(key,x);
        if(!full)return {};
        const auto& y=*full;
        std::vector<mx::array> parts;int c0=0;
        for(int r:pack_rows.at(key)) {parts.push_back(mx::slice(y,{0,c0},{x.shape(0),c0+r}));c0+=r;}
        return parts;
    }
    // FP16 product of one NAX / TQ4 matrix on a prefill chunk (rows >= 64), as mat() computes it before its rounding
    std::optional<mx::array> product16(const std::string& name,const mx::array& x) {
        if(x.shape(0)<64)return std::nullopt;
        const NaxAffine* w=nullptr;bool tq4=false;
        if(auto f=nax_tq4.find(name);f!=nax_tq4.end()) {w=&f->second;tq4=true;}
        else if(auto f=nax.find(name);f!=nax.end())w=&f->second;
        if(!w || w->q6)return std::nullopt;
        auto& slot=tq4?gemm_rotated_input:gemm_input;
        if(!slot || slot->first.id()!=x.id())slot=std::pair{x,tq4?tq4_rotate_half(x):mx::astype(x,mx::float16)};
        return gemm4(*w,slot->second);
    }
    // TQLLM_MLX_PREFILL_FUSED=0: MLX op chains for the prefill norms and SwiGLU. On, each fused kernel is compared with
    // its chain on the first prefill chunk and used only when bitwise equal (-1 unchecked, 0 fallback, 1 fused)
    int norm_fused=-1,swiglu_fused=-1;
    // TQLLM_MLX_ATTN_QK_FUSED=0: the attention q/k norm + RoPE stage runs the MLX chain. On, one fused kernel is compared
    // with the chain on the first prefill chunk and used only when bitwise equal (-1 unchecked, 0 fallback, 1 fused)
    int attn_qk_fused=-1;
    static bool fused_on() {static const bool on=env_default_on("TQLLM_MLX_PREFILL_FUSED");return on;}
    static bool attn_qk_fused_on() {static const bool on=env_default_on("TQLLM_MLX_ATTN_QK_FUSED");return on;}
    static bool same_bits(const mx::array& a,const mx::array& b) {
        return a.shape()==b.shape() && a.dtype()==b.dtype() && mx::array_equal(a,b).item<bool>();
    }
    // prefill norm: the BF16-rounded FP32 rows, and their FP16 copy seeded as the projections' GEMM input
    // decode / verify rows (<= 16): rms_norm + BF16 round + the FP16 matmul copy in one launch (the prefill pair kernel,
    // row-wise so verify rows equal AR rows), checked bitwise against the MLX chain once; add_norm_rows adds the
    // residual add in front (j1038: ~5 of ~20 dispatches per layer were these chains)
    int rows_norm_fused=-1,rows_add_norm_fused=-1;
    mx::array norm_rows(const mx::array& x,const std::string& name) {
        const auto& w=plain.at(name);
        if(x.ndim()!=2 || x.shape(1)<=4096 || x.dtype()!=mx::float32 || rows_norm_fused==0)return norm(x,name);
        auto pair=rms_norm_bf16_pair(x,w,eps);
        if(rows_norm_fused<0) {
            auto ref=rms_norm_bf16_pair_reference(x,w,eps);
            try {rows_norm_fused=same_bits(pair.first,ref.first) && same_bits(pair.second,ref.second);}
            catch(const std::exception& e) {rows_norm_fused=0;std::fprintf(stderr,"DECODE_FUSED rms_norm failed: %s\n",e.what());}
            std::fprintf(stderr,"DECODE_FUSED rms_norm bitwise=%d\n",rows_norm_fused);
            if(!rows_norm_fused)return norm(x,name);
        }
        nax_input=std::pair{pair.first,pair.second};
        return pair.first;
    }
    std::pair<mx::array,mx::array> add_norm_rows(const mx::array& x,const mx::array& r,const std::string& name) {
        const auto& w=plain.at(name);
        if(x.ndim()!=2 || x.shape(1)<=4096 || x.dtype()!=mx::float32 || r.dtype()!=mx::float32 || rows_add_norm_fused==0) {
            auto s=add_bf(x,r);return {s,norm(s,name)};
        }
        auto got=rms_norm_bf16_pair_residual(x,r,w,eps);
        if(rows_add_norm_fused<0) {
            auto ref=rms_norm_bf16_pair_residual_reference(x,r,w,eps);
            try {rows_add_norm_fused=same_bits(std::get<0>(got),std::get<0>(ref)) && same_bits(std::get<1>(got),std::get<1>(ref)) && same_bits(std::get<2>(got),std::get<2>(ref));}
            catch(const std::exception& e) {rows_add_norm_fused=0;std::fprintf(stderr,"DECODE_FUSED add_norm failed: %s\n",e.what());}
            std::fprintf(stderr,"DECODE_FUSED add_norm bitwise=%d\n",rows_add_norm_fused);
            if(!rows_add_norm_fused) {auto s=add_bf(x,r);return {s,norm(s,name)};}
        }
        nax_input=std::pair{std::get<1>(got),std::get<2>(got)};
        return {std::get<0>(got),std::get<1>(got)};
    }
    mx::array norm_prefill(const mx::array& x,const std::string& name) {
        const auto& w=plain.at(name);
        if(!fused_on() || x.ndim()!=2 || x.shape(1)<=4096 || x.dtype()!=mx::float32 || norm_fused==0)return norm(x,name);
        auto pair=rms_norm_bf16_pair(x,w,eps);
        if(norm_fused<0) {
            auto ref=rms_norm_bf16_pair_reference(x,w,eps);
            try {norm_fused=same_bits(pair.first,ref.first) && same_bits(pair.second,ref.second);}
            catch(const std::exception& e) {norm_fused=0;std::fprintf(stderr,"PREFILL_FUSED rms_norm failed: %s\n",e.what());}
            std::fprintf(stderr,"PREFILL_FUSED rms_norm bitwise=%d\n",norm_fused);
            if(!norm_fused)return norm(x,name);
        }
        gemm_input=std::pair{pair.first,pair.second};
        return pair.first;
    }
    mx::array swiglu_prefill(const mx::array& gate_up) {
        if(fused_on() && swiglu_fused!=0) {
            auto h=swiglu_pack16(gate_up);
            if(swiglu_fused<0) {
                try {swiglu_fused=same_bits(h,swiglu_pack16_reference(gate_up));}
                catch(const std::exception& e) {swiglu_fused=0;std::fprintf(stderr,"PREFILL_FUSED swiglu failed: %s\n",e.what());}
                std::fprintf(stderr,"PREFILL_FUSED swiglu bitwise=%d\n",swiglu_fused);
            }
            if(swiglu_fused)return h;
        }
        return swiglu_pack16_reference(gate_up);
    }
    // the mixer's output projection with the layer's residual add fused (prefill chunks); null residual: mat() as before
    const mx::array* mixer_residual=nullptr;bool mixer_added=false;
    mx::array output_projection(const std::string& name,const mx::array& x) {
        if(mixer_residual)
            if(auto y=product16(name,x)) {mixer_added=true;return add_product_bf(*mixer_residual,*y);}
        return mat(name,x);
    }
    std::optional<NaxAffine> draft_nax_head;
    std::optional<std::pair<mx::array,mx::array>> nax_input;  // last matmul input and its FP16 copy (same-input projections)
    std::optional<std::pair<mx::array,mx::array>> gemm_input,gemm_rotated_input;  // prefill GEMM inputs in FP16
    std::vector<Cache> caches;
    // full-attention KV buffers, written in place; never snapshotted (a snapshot keeps only Cache::kv_len)
    std::vector<std::optional<mx::array>> kv_keys,kv_values;
    // KV format of the current request (kv_tier.hpp, chosen before its prefill): BF16 [1, kvh, cap, hd] in kv_keys /
    // kv_values; int8: K^T blocks [kvh, cap / 1024, hd, 1024] / V [kvh, cap, hd] int8 + FP16 scales [kvh, cap] in
    // kv_kscale / kv_vscale; trellis: K codes [kvh, cap, 26] / V codes [kvh, cap, 18] uint32
    KvTier kv_tier=KvTier::Bf16;
    std::vector<std::optional<mx::array>> kv_kscale,kv_vscale;
    std::vector<PrefixTape> tapes;
    bool capture_prefix=false;
    std::optional<std::pair<mx::array,mx::array>> dense_input;
    std::optional<mx::array> last_hidden;
    std::optional<mx::array> last_features;
    std::vector<int> feature_ids;
    Cache mtp_cache;
    int mtp_pos=1;
    bool mtp_enabled=false;
    std::optional<Q6Weights> mtp_head;
    std::optional<DraftHead4> proposal_head;
    std::optional<mx::array> mtp_ids;
    std::optional<mx::array> mtp_hidden;
    uint64_t epoch=0,revision=0;
    int kv_reserve=0;  // rows the current request will need (reserve_kv): one KV allocation, no regrowth
    int context,pos=0,layers,hidden,heads,kvheads,hd,rot,kh,vh,dk,dv,conv,interval,vocab;
    float eps,theta;
    mx::array key_indices;
    mx::array global_ids=mx::array(0,mx::uint32);
    bool profiling=false,stage_profiling=false;
    bool tq4_basis=false;  // the model's quantized matrices are TQ4 (rotated inputs); see the loader
    std::optional<std::pair<mx::array,mx::array>> dense_rotated_input;
    // layer-major prefill (forward_layer_major): one FP16 dequant per matrix shared by the batch's slices, dropped per layer
    bool share_dequant=false;
    bool defer_eval=false;  // draft-ahead: submit this forward without waiting, the caller chains the next draft behind it
    std::map<const NaxAffine*,mx::array> dequant_cache;
    mx::array dequant(const NaxAffine& w) {
        if(!share_dequant)return nax_affine_dequant(w);
        auto it=dequant_cache.find(&w);
        if(it==dequant_cache.end()) {
            it=dequant_cache.emplace(&w,nax_affine_dequant(w)).first;
            stage("dequant",{it->second});
        }
        return it->second;
    }
    mx::array gemm16(const NaxAffine& w,const mx::array& x,bool round) {
        auto y=gemm4(w,x);
        return round?round_product(y):mx::astype(y,mx::float32);
    }
    std::unique_ptr<AneFfn> ane;bool ane_tried=false;

    std::map<std::string,double> stages;
    std::chrono::steady_clock::time_point stage_tick;
    // with profile_stages: evaluate the arrays and charge the time since the last mark to this stage
    void stage(const char* name,const std::vector<mx::array>& arrays) {
        if(!stage_profiling)return;
        mx::eval(arrays);
        auto now=std::chrono::steady_clock::now();
        stages[name]+=std::chrono::duration<double,std::milli>(now-stage_tick).count();stage_tick=now;
        stages[std::string(name)+"_calls"]+=1;
    }
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
        auto matrix=[&](const std::string& name) {
            if(const auto& stored=file.tensors.at(name);stored.type==GGML_NAX_A4_G128||stored.type==GGML_NAX_A6_G64||stored.type==GGML_NAX_A4_G64) {
                auto raw=file.read(stored);
                const int bits=stored.type==GGML_NAX_A6_G64?6:4,group=stored.type==GGML_NAX_A4_G128?128:64;
                nax.emplace(name,nax_affine_load({reinterpret_cast<const uint8_t*>(raw.data()),raw.size()},
                                                 int(stored.ne[1]),int(stored.ne[0]),bits,group));
                if(name=="output.weight")nax_head_payload.assign(reinterpret_cast<const uint8_t*>(raw.data()),reinterpret_cast<const uint8_t*>(raw.data())+raw.size());
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
            // TQ4 with integer levels runs on the matrix unit; the repacked copy replaces the original
            if(auto packed=std::get_if<Tq4Weights>(&w);packed && packed->integer_levels && packed->codes.shape(0)%128==0) {
                nax_tq4.emplace(name,nax_from_tq4(*packed));
                return;
            }
            matrices.emplace(name,std::move(w));
        };
        auto scalar=[&](const std::string& name) {
            const auto& t=file.tensors.at(name);auto raw=file.read(t);
            auto w=mx::astype(load_plain_tensor(t,{reinterpret_cast<const uint8_t*>(raw.data()),raw.size()}),mx::float32);
            mx::eval(w);plain.emplace(name,std::move(w));
        };
        matrix("output.weight");scalar("output_norm.weight");
        const auto head_matrix = matrices.find("output.weight");
        const auto* q6_head = head_matrix == matrices.end() ? nullptr : std::get_if<Q6Weights>(&head_matrix->second);
        if (q6_head) nax.emplace("output.weight", nax_from_q6(*q6_head));
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
            // TQLLM_MLX_PREFILL_PACK=0: separate prefill GEMMs per matrix
            pack(p+"ffn_gu",{p+"ffn_gate.weight",p+"ffn_up.weight"});
            if((l+1)%interval==0)pack(p+"attn_in",{p+"attn_q.weight",p+"attn_k.weight",p+"attn_v.weight"});
            else pack(p+"gdn_in",{p+"attn_qkv.weight",p+"attn_gate.weight"});
        }
        // a TQ4 model's dense matrices (TQ_L: BF16 ssm_alpha / ssm_beta) are stored in the TQ4 rotated basis and take the
        // rotated activation, as the CUDA engine reads them (j945: plain input = mean KL 0.53 vs CUDA on its own)
        tq4_basis=!nax_tq4.empty() || std::any_of(matrices.begin(),matrices.end(),[](const auto& m) {return std::holds_alternative<Tq4Weights>(m.second);});
        // dense alpha | beta stacked per GDN layer for the fused step's in-kernel projection (stored basis, FP32)
        for(int l=0;l<layers;++l) {
            if((l+1)%interval==0)continue;
            const std::string p="blk."+std::to_string(l)+".";
            auto a=matrices.find(p+"ssm_alpha.weight"),b=matrices.find(p+"ssm_beta.weight");
            if(a==matrices.end() || b==matrices.end() || !std::holds_alternative<mx::array>(a->second) || !std::holds_alternative<mx::array>(b->second))continue;
            auto w=mx::contiguous(mx::astype(mx::concatenate({std::get<mx::array>(a->second),std::get<mx::array>(b->second)},0),mx::float32));
            mx::eval(w);ab_proj.emplace(p,std::move(w));
        }
        caches.resize(layers);
        kv_keys.resize(layers);kv_values.resize(layers);kv_kscale.resize(layers);kv_vscale.resize(layers);
        tapes.resize(layers);
    }
    std::optional<mx::array> device_ids;  // set by verify_device: the batch's ids stay on the GPU (draft -> verify, no readback)
    mx::array embed(std::span<const uint32_t> tokens) const {
        if(device_ids)return embedding8?affine8_embedding(*embedding8,*device_ids):q4_embedding(*embedding,*device_ids);
        return embedding8?affine8_embedding(*embedding8,tokens):q4_embedding(*embedding,tokens);
    }
    // from this many rows a product dequantizes the matrix to FP16 once and runs a GEMM (below: the 4-bit kernels)
    static constexpr int kDequantRows=64;
    // nax_affine_prefill's row tile (rs 32 x sgs 4): split slices are whole tiles (j906: doc8k slices of 1969 rows 635
    // tok/s, 1984 = 31 x 64 638, 2048 = 16 x 128 703)
    static constexpr int kPrefillRowTile=128;
    mx::array mat(const std::string& name,const mx::array& x) {
        // prefill chunks: dequantize once and run one GEMM instead of re-decoding the matrix per 32-row tile
        if(x.shape(0)>=kDequantRows) {
            // one FP16 copy per distinct input (projections of one input share it); TQ4 inputs rotate in one launch
            auto half_input=[&](bool tq4) {
                auto& slot=tq4?gemm_rotated_input:gemm_input;
                if(!slot || slot->first.id()!=x.id())slot=std::pair{x,tq4?tq4_rotate_half(x):mx::astype(x,mx::float16)};
                return slot->second;
            };
            if(auto found=nax_tq4.find(name);found!=nax_tq4.end())return gemm16(found->second,half_input(true),true);
            if(auto found=nax.find(name);found!=nax.end())return gemm16(found->second,half_input(false),true);
        }
        if(auto found=nax_tq4.find(name);found!=nax_tq4.end()) {
            if(!nax_rotated_input || nax_rotated_input->first.id()!=x.id())nax_rotated_input=std::pair{x,tq4_rotate_half(x)};
            return nax_affine_matmul(found->second,nax_rotated_input->second,true);
        }
        if(auto found=nax.find(name);found!=nax.end()) {
            if(!nax_input || nax_input->first.id()!=x.id())nax_input=std::pair{x,mx::contiguous(mx::astype(x,mx::float16))};
            return nax_affine_matmul(found->second,nax_input->second,true);
        }
        auto it=matrices.find(name);
        if(it==matrices.end())throw std::invalid_argument("matrix has no weights: "+name);
        const auto& w=it->second;
        const bool dense=std::holds_alternative<mx::array>(w);
        auto in=x;
        if(dense && tq4_basis) {
            if(!dense_rotated_input || dense_rotated_input->first.id()!=x.id())dense_rotated_input=std::pair{x,basis_input(w,x,true)};
            in=dense_rotated_input->second;
        }
        // verify rows of dense matrices: each row as its own decode row, so a row matches its AR call bit for bit
        if(capture_prefix && x.shape(0)>1 && dense) {
            if(!dense_input || dense_input->first.id()!=in.id())dense_input=std::pair{in,dense_row_views(in)};
            return bf(dense_decode_rows(std::get<mx::array>(w),dense_input->second));
        }
        return bf(linear(w,in));
    }
    mx::array norm(const mx::array& x,const std::string& name) {
        return bf(mx::fast::rms_norm(x,plain.at(name),eps));
    }
    // decode / verify rows (<= 16): gate, up and SwiGLU in one matrix-unit launch (nax_affine_swiglu), same BF16
    // roundings as gate_up + silu path; nullopt when the two matrices are not a matching 4-bit pair
    std::optional<mx::array> swiglu_rows(const std::string& p,const mx::array& x) {
        if(x.shape(0)>16)return std::nullopt;
        for(auto* map:{&nax_tq4,&nax}) {
            auto g=map->find(p+"ffn_gate.weight"),u=map->find(p+"ffn_up.weight");
            if(g==map->end() || u==map->end())continue;
            const auto &G=g->second,&U=u->second;
            if(G.bits!=4 || U.bits!=4 || G.q6 || U.q6 || G.rows!=U.rows || G.columns!=U.columns || G.group!=U.group)return std::nullopt;
            if(map==&nax_tq4) {
                if(!nax_rotated_input || nax_rotated_input->first.id()!=x.id())nax_rotated_input=std::pair{x,tq4_rotate_half(x)};
                return nax_affine_swiglu(G,U,nax_rotated_input->second);
            }
            if(!nax_input || nax_input->first.id()!=x.id())nax_input=std::pair{x,mx::contiguous(mx::astype(x,mx::float16))};
            return nax_affine_swiglu(G,U,nax_input->second);
        }
        return std::nullopt;
    }
    std::pair<mx::array,mx::array> gate_up(const std::string& p,const mx::array& x) {
        return {mat(p+"ffn_gate.weight",x),mat(p+"ffn_up.weight",x)};
    }
    // Prefill FFN on MLX's FP16 GEMM (j601: 31 TFLOPS at 2048 rows vs ~22 for the row kernel): the three matrices are
    // dequantized to FP16 once per chunk, swiglu runs in one compiled kernel with the BF16 roundings of the row path,
    // and its FP16 result feeds the down GEMM directly (prefill chunks of 64+ rows).
    void ensure_ane() {
        static const char* ane_dir=std::getenv("TQLLM_MLX_ANE_FFN");
        if(ane_dir && !ane_tried) {ane_tried=true;const auto t0=std::chrono::steady_clock::now();ane=AneFfn::open(ane_dir,layers);
            std::fprintf(stderr,"ANE_FFN open %.0f ms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());}
    }
    // GPU share of the split FFN (intermediate channels [0, F - C)) for an evaluated, dense fp16 input; fp32 [rows, hidden]
    struct ShareWeights {mx::array wg,wu,wd;};
    // fp16 weights of the GPU share, dequantized once per layer and chunk (both wavefront halves use them); down's column
    // slice is made contiguous so its GEMM does not copy per call
    // the packed layout is [rows/32][columns/16 | columns/group][32 lanes]: leading rows are a prefix of blocks, leading
    // columns a prefix of each block's steps and groups (copied, ~1/7 of the fp16 size), so only the GPU's share is decoded
    static NaxAffine nax_leading(const NaxAffine& w,int rows,int columns) {
        if(rows%32 || columns%std::max(16,w.group) || rows>w.rows || columns>w.columns)throw std::invalid_argument("nax_leading: bad shape");
        auto cut=[&](const mx::array& a,int last) {
            mx::Shape stop=a.shape();stop[0]=rows/32;stop[1]=last;
            return mx::contiguous(mx::slice(a,mx::Shape(a.ndim(),0),stop));
        };
        NaxAffine v=w;
        v.words=cut(w.words,columns/16);
        if(w.bits==6)v.high=cut(w.high,columns/16);
        if(w.scales.ndim()==3)v.scales=cut(w.scales,columns/w.group);
        if(w.biases.ndim()==3)v.biases=cut(w.biases,columns/w.group);
        v.rows=rows;v.columns=columns;
        return v;
    }
    ShareWeights share_weights(const NaxAffine& g,const NaxAffine& u,const NaxAffine& d,int C) {
        const int F=g.rows,G=F-C,hidden=g.columns;
        return {nax_affine_dequant(nax_leading(g,G,hidden)),nax_affine_dequant(nax_leading(u,G,hidden)),
                nax_affine_dequant(nax_leading(d,d.rows,G))};
    }
    mx::array ffn_gpu_share(const ShareWeights& w,bool tq4_down,const mx::array& xc) {
        auto gv=mx::matmul(xc,mx::transpose(w.wg));
        auto uv=mx::matmul(xc,mx::transpose(w.wu));
        static auto swiglu_s=mx::compile([](const std::vector<mx::array>& a) {
            auto gate=bf_raw(mx::astype(a[0],mx::float32)),up=bf_raw(mx::astype(a[1],mx::float32));
            return std::vector<mx::array>{mx::astype(bf_raw(mx::multiply(bf_raw(mx::multiply(gate,mx::sigmoid(gate))),up)),mx::float16)};
        },true);
        auto hs=swiglu_s({gv,uv})[0];
        auto hg=tq4_down?tq4_rotate_half(hs):hs;
        return mx::astype(mx::matmul(hg,mx::transpose(w.wd)),mx::float32);
    }
    // TQLLM_MLX_ANE_WAVE=1: a prefill chunk of 2R rows runs as two halves of R rows, layer by layer; while the GPU runs one
    // half's mixer and its FFN share, the Neural Engine runs the other half's FFN share on a worker thread, so the ANE call
    // no longer waits for the GPU's next mixer (the per-layer split is lockstep, j672). Without an ANE share the halves run
    // exactly as two chunks of R rows. Returns the [2R, hidden] output of the last layer; features as forward().
    mx::array wave_layers(std::span<const uint32_t> tokens,std::vector<mx::array>& features) {
        const int R=int(tokens.size())/2;
        struct Half {mx::array x;int offset;std::vector<mx::array> feats;std::optional<mx::array> y_gpu,y_ane,xc;};
        Half h[2]={{bf(embed(tokens.subspan(0,R))),pos,{},{},{},{}},{bf(embed(tokens.subspan(R,R))),pos+R,{},{},{},{}}};
        std::future<void> job;int job_half=-1;
        static const bool timing=std::getenv("TQLLM_MLX_ANE_TIMING")!=nullptr;
        // TQLLM_MLX_ANE_STUB=1 (timing diagnostic, wrong output): skip the ANE predict, the GPU runs only its share
        static const bool stub=std::getenv("TQLLM_MLX_ANE_STUB")!=nullptr;
        // direct evidence: ANE calls made, time inside the predict call (worker thread), layers that fell back to the GPU
        std::atomic<double> ane_ms{0},ane_cpu_ms{0};int ane_calls=0,gpu_layers=0;long gpu_share_cols=0;
        double t_eval=0,t_join=0,t_build=0;auto tick=std::chrono::steady_clock::now();
        auto lap=[&]() {auto now=std::chrono::steady_clock::now();double d=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;return d;};
        auto join=[&]() {
            if(!job.valid())return;
            if(timing)t_build+=lap();
            job.get();
            if(timing)t_join+=lap();
            auto& j=h[job_half];
            j.x=add_bf(j.x,bf_raw(mx::add(*j.y_gpu,mx::astype(*j.y_ane,mx::float32))));
            j.y_gpu.reset();j.y_ane.reset();j.xc.reset();job_half=-1;
        };
        for(int l=0;l<layers;++l) {
            const std::string p="blk."+std::to_string(l)+".";
            const bool is_attn=(l+1)%interval==0;
            std::optional<ShareWeights> share;
            for(int k=0;k<2;++k) {
                auto& c=h[k];
                if(std::binary_search(feature_ids.begin(),feature_ids.end(),l))c.feats.push_back(c.x);
                auto n=norm(c.x,p+"attn_norm.weight");
                auto attn=is_attn?full_attention(p,n,caches[l],c.offset,l):gdn(p,n,caches[l],tapes[l]);
                c.x=add_bf(c.x,attn);n=norm(c.x,p+"post_attention_norm.weight");
                auto find=[&](const std::string& name)->const NaxAffine* {
                    if(auto f=nax_tq4.find(name);f!=nax_tq4.end())return &f->second;
                    if(auto f=nax.find(name);f!=nax.end())return &f->second;
                    return nullptr;
                };
                const auto *g=find(p+"ffn_gate.weight"),*u=find(p+"ffn_up.weight"),*d=find(p+"ffn_down.weight");
                if(!(ane && ane->has(l) && ane->rows()==R && g && u && d && !g->q6 && !u->q6 && !d->q6 &&
                     g->tq4==u->tq4)) {
                    ++gpu_layers;
                    c.x=add_bf(c.x,*ffn_gemm(p,n));
                    continue;
                }
                auto xc=mx::contiguous(g->tq4?tq4_rotate_half(n):mx::astype(n,mx::float16));
                auto y_ane=mx::contiguous(mx::zeros({R,g->columns},mx::float16));
                if(timing)t_build+=lap();
                mx::eval(xc,y_ane);   // this half's mixer; the other half's ANE share runs meanwhile
                if(timing)t_eval+=lap();
                join();               // the other half's ANE share -> its residual (lazy)
                if(!share)share=share_weights(*g,*u,*d,ane->columns());
                c.y_gpu=ffn_gpu_share(*share,d->tq4,xc);
                mx::async_eval(*c.y_gpu);
                c.xc=xc;c.y_ane=y_ane;
                const void* xp=xc.data<mx::float16_t>();void* yp=y_ane.data<mx::float16_t>();
                const int hidden=g->columns;AneFfn* a=ane.get();
                ++ane_calls;gpu_share_cols+=share->wg.shape(0);
                std::atomic<double>* acc=&ane_ms;
                std::atomic<double>* cpu_acc=&ane_cpu_ms;
                job=std::async(std::launch::async,[a,l,xp,yp,hidden,acc,cpu_acc]{
                    if(stub)return;
                    auto thread_ms=[]{timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;};
                    const auto t0=std::chrono::steady_clock::now();const double c0=thread_ms();a->predict(l,xp,yp,hidden);
                    const double d=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),c=thread_ms()-c0;
                    double cur=acc->load();while(!acc->compare_exchange_weak(cur,cur+d)) {}
                    cur=cpu_acc->load();while(!cpu_acc->compare_exchange_weak(cur,cur+c)) {}
                });job_half=k;
            }
        }
        join();
        if(timing)std::fprintf(stderr,"ANE_WAVE chunk %d rows: host graph build %.0f ms, GPU sync waits %.0f ms, ANE join waits %.0f ms | ANE calls %d (inside predict %.0f ms wall, %.0f ms CPU), GPU-only FFN layers %d, GPU share rows per call %ld of %d\n",
            2*R,t_build,t_eval,t_join,ane_calls,ane_ms.load(),ane_cpu_ms.load(),gpu_layers,ane_calls?gpu_share_cols/ane_calls:0L,layers?int(nax_tq4.count("blk.0.ffn_gate.weight")?nax_tq4.at("blk.0.ffn_gate.weight").rows:nax.count("blk.0.ffn_gate.weight")?nax.at("blk.0.ffn_gate.weight").rows:0):0);
        for(int k=0;k<2;++k)if(!h[k].feats.empty())features.push_back(mx::concatenate(h[k].feats,1));
        if(features.size()==2) {auto both=mx::concatenate({features[0],features[1]},0);features={both};}
        return mx::concatenate({h[0].x,h[1].x},0);
    }
    // residual: the down product's rounding and the residual add in one pass; the result is then the layer output
    std::optional<mx::array> ffn_gemm(const std::string& p,const mx::array& n,const mx::array* residual=nullptr) {
        if(n.shape(0)<kDequantRows)return std::nullopt;  // j604: prefill +47-50 %, exact
        auto find=[&](const std::string& name)->const NaxAffine* {
            if(auto f=nax_tq4.find(name);f!=nax_tq4.end())return &f->second;
            if(auto f=nax.find(name);f!=nax.end())return &f->second;
            return nullptr;
        };
        const auto *g=find(p+"ffn_gate.weight"),*u=find(p+"ffn_up.weight"),*d=find(p+"ffn_down.weight");
        if(!g || !u || !d || g->q6 || u->q6 || d->q6)return std::nullopt;
        auto x16=g->tq4?tq4_rotate_half(n):(gemm_input && gemm_input->first.id()==n.id()?gemm_input->second:mx::astype(n,mx::float16));
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
        // TQLLM_MLX_DUMP_FFN_SLICE=dir:C: per layer, the last C intermediate channels' FP16 weights (gate/up rows, down
        // columns, TQ4 basis) and the first chunk's FFN input, for the Neural Engine model builder (ane_build.py)
        static const std::string slice_dump=std::getenv("TQLLM_MLX_DUMP_FFN_SLICE")?std::getenv("TQLLM_MLX_DUMP_FFN_SLICE"):"";
        if(!slice_dump.empty()) {
            const auto colon=slice_dump.find(':');const std::string dir=slice_dump.substr(0,colon);
            const int C=std::stoi(slice_dump.substr(colon+1)),F=g->rows;
            const std::string layer=p.substr(4,p.size()-5);
            static std::set<std::string> done;
            if(!done.count(layer) && C>0 && C<F) {
                done.insert(layer);
                mx::save(dir+"/x_l"+layer+".npy",x16);
                mx::save(dir+"/wg_l"+layer+".npy",mx::slice(nax_affine_dequant(*g),{F-C,0},{F,g->columns}));
                mx::save(dir+"/wu_l"+layer+".npy",mx::slice(nax_affine_dequant(*u),{F-C,0},{F,u->columns}));
                mx::save(dir+"/wd_l"+layer+".npy",mx::contiguous(mx::slice(nax_affine_dequant(*d),{0,F-C},{d->rows,F})));
            }
        }
        // TQLLM_MLX_ANE_FFN=dir: the last C intermediate channels of each layer with a model in dir run on the Neural
        // Engine (Core ML, ane_ffn.mm) while the GPU computes the rest; chunks of the model's row count only.
        // Opt-in fast prefill: the ANE's arithmetic is not exact (j662/j664). TQLLM_MLX_ANE_SERIAL=1 runs it after the GPU.
        ensure_ane();
        const int layer_index=std::stoi(p.substr(4,p.size()-5));
        if(ane && ane->has(layer_index) && n.shape(0)==ane->rows() && g->tq4 && u->tq4 && d->tq4) {
            const int C=ane->columns(),F=g->rows,G=F-C,hidden=g->columns;
            // TQLLM_MLX_ANE_TIMING=1: per-chunk sums of the input sync, the ANE call and the join wait (ms)
            static const bool timing=std::getenv("TQLLM_MLX_ANE_TIMING")!=nullptr;
            static double t_sync=0,t_pred=0,t_join=0;static int t_layers=0;
            auto tick=std::chrono::steady_clock::now();
            auto lap=[&]() {auto now=std::chrono::steady_clock::now();double d=std::chrono::duration<double,std::milli>(now-tick).count();tick=now;return d;};
            auto xc=mx::contiguous(x16);
            auto y_ane=mx::contiguous(mx::zeros({n.shape(0),hidden},mx::float16));
            mx::eval(xc,y_ane);
            if(timing)t_sync+=lap();
            if(!y_ane.flags().row_contiguous || y_ane.data_size()!=y_ane.size() || !xc.flags().row_contiguous)
                throw std::runtime_error("ANE FFN: buffers are not dense");
            auto gv=mx::matmul(xc,mx::transpose(mx::slice(nax_affine_dequant(*g),{0,0},{G,hidden})));
            auto uv=mx::matmul(xc,mx::transpose(mx::slice(nax_affine_dequant(*u),{0,0},{G,hidden})));
            static auto swiglu_a=mx::compile([](const std::vector<mx::array>& a) {
                auto gate=bf_raw(mx::astype(a[0],mx::float32)),up=bf_raw(mx::astype(a[1],mx::float32));
                return std::vector<mx::array>{mx::astype(bf_raw(mx::multiply(bf_raw(mx::multiply(gate,mx::sigmoid(gate))),up)),mx::float16)};
            },true);
            auto hg=tq4_rotate_half(swiglu_a({gv,uv})[0]);
            auto y_gpu=mx::astype(mx::matmul(hg,mx::transpose(mx::slice(nax_affine_dequant(*d),{0,0},{d->rows,G}))),mx::float32);
            static const bool serial=std::getenv("TQLLM_MLX_ANE_SERIAL")!=nullptr;
            if(serial)mx::eval(y_gpu);else mx::async_eval(y_gpu);
            if(timing)lap();
            ane->predict(layer_index,xc.data<mx::float16_t>(),y_ane.data<mx::float16_t>(),hidden);
            if(timing) {
                t_pred+=lap();mx::eval(y_gpu);t_join+=lap();
                if(++t_layers%64==0) {
                    std::fprintf(stderr,"ANE_TIMING 64 layers: input sync %.1f ms, ANE predict %.1f ms, GPU share wait after predict %.1f ms\n",t_sync,t_pred,t_join);
                    t_sync=t_pred=t_join=0;
                }
            }
            auto y=bf_raw(mx::add(y_gpu,mx::astype(y_ane,mx::float32)));
            return residual?add_bf(*residual,y):y;
        }
        std::optional<mx::array> gv,uv,hp;
        if(auto it=packs.find(p+"ffn_gu");it!=packs.end()) {  // gate | up: equal halves
            auto gu=gemm4(it->second,x16);
            stage("ffn_gu_gemm",{gu});
            hp=swiglu_prefill(gu);
        } else {gv=gemm4(*g,x16);uv=gemm4(*u,x16);}
        static auto swiglu16=mx::compile([](const std::vector<mx::array>& a) {
            auto gate=bf_raw(mx::astype(a[0],mx::float32)),up=bf_raw(mx::astype(a[1],mx::float32));
            return std::vector<mx::array>{mx::astype(bf_raw(mx::multiply(bf_raw(mx::multiply(gate,mx::sigmoid(gate))),up)),mx::float16)};
        },true);
        auto h=hp?*hp:swiglu16({*gv,*uv})[0];
        if(stage_profiling)stage(packs.count(p+"ffn_gu")?"ffn_swiglu":"ffn_gate_up",{h});
        auto hin=d->tq4?tq4_rotate_half(h):h;
        auto y=gemm4(*d,hin);
        return residual?add_product_bf(*residual,y):bf_raw(mx::astype(y,mx::float32));
    }
    mx::array full_attention(const std::string& p,const mx::array& x,Cache& c,int offset,int layer) {
        const int m=x.shape(0);
        auto in=pack_product16(p+"attn_in",x);
        if(!in.empty())stage("attn_in_gemm",{in[0]});
        // fused q/k norm + RoPE: read the packed FP16 products in place, one kernel for q (with gate) and k, v unchanged
        if(!in.empty() && attn_qk_fused_on() && attn_qk_fused!=0) {
            auto f=attn_qk_norm_rope(in[0],in[1],in[2],
                                     plain.at(p+"attn_q_norm.weight"),plain.at(p+"attn_k_norm.weight"),
                                     heads,kvheads,hd,rot,theta,eps,offset);
            if(attn_qk_fused<0) {
                auto ref=attn_qk_norm_rope_reference(in[0],in[1],in[2],
                                                     plain.at(p+"attn_q_norm.weight"),plain.at(p+"attn_k_norm.weight"),
                                                     heads,kvheads,hd,rot,theta,eps,offset);
                try {attn_qk_fused=same_bits(f.q,ref.q) && same_bits(f.k,ref.k) && same_bits(f.v,ref.v) && same_bits(f.gate,ref.gate);}
                catch(const std::exception& e) {attn_qk_fused=0;std::fprintf(stderr,"PREFILL_FUSED attn_qk failed: %s\n",e.what());}
                std::fprintf(stderr,"PREFILL_FUSED attn_qk bitwise=%d\n",attn_qk_fused);
            }
            if(attn_qk_fused) {stage("attn_qk_norm_rope",{f.q,f.k,f.v,f.gate});return attn_core(p,f.q,f.k,f.v,f.gate,c,offset,layer,m);}
        }
        auto proj=[&](int i,const char* name) {return in.empty()?mat(p+name,x):round_product(in[i]);};
        auto qg=mx::reshape(proj(0,"attn_q.weight"),{m,heads,2*hd});
        auto q=tail_slice(qg,2,0,hd),gate=tail_slice(qg,2,hd,2*hd);
        q=mx::transpose(mx::reshape(norm(q,p+"attn_q_norm.weight"),{1,m,heads,hd}),{0,2,1,3});
        auto k=mx::reshape(proj(1,"attn_k.weight"),{1,m,kvheads,hd});
        k=mx::transpose(norm(k,p+"attn_k_norm.weight"),{0,2,1,3});
        auto v=mx::transpose(mx::reshape(proj(2,"attn_v.weight"),{1,m,kvheads,hd}),{0,2,1,3});
        q=bf(mx::fast::rope(q,rot,false,theta,1.0f,offset));k=bf(mx::fast::rope(k,rot,false,theta,1.0f,offset));
        stage(in.empty()?"attn_proj":"attn_qk_norm_rope",{q,k,v,gate});
        return attn_core(p,q,k,v,gate,c,offset,layer,m);
    }
    // attn_core for int8 / trellis KV (kv_tier.hpp): one stored format; decode / verify rows read it in place (v12 /
    // trellis kernels), a prefill chunk decodes this layer's prefix to FP16 for the fused SDPA
    mx::array attn_core_compressed(const std::string& p,const mx::array& q,const mx::array& k_in,const mx::array& v_in,
                                   const mx::array& gate,Cache& c,int layer,int m) {
        constexpr int kBlock=kAttentionSegment;
        auto& kb=kv_keys[layer];auto& vb=kv_values[layer];auto& ks=kv_kscale[layer];auto& vs=kv_vscale[layer];
        const bool int8=kv_tier==KvTier::Int8;
        const int need=c.kv_len+m,pos=c.kv_len;
        const int cur=kb?(int8?vb->shape(1):kb->shape(1)):0;
        if(!kb || cur<need || (int8?kb->dtype()!=mx::int8:kb->dtype()!=mx::uint32)) {
            const int cap=kv_capacity(kb && (int8?kb->dtype()==mx::int8:kb->dtype()==mx::uint32)?cur:0,need,kv_reserve,context,kBlock);
            if(int8) {
                auto nk=mx::zeros({kvheads,cap/kBlock,hd,kBlock},mx::int8),nv=mx::zeros({kvheads,cap,hd},mx::int8);
                auto nks=mx::zeros({kvheads,cap},mx::float16),nvs=mx::zeros({kvheads,cap},mx::float16);
                if(kb && kb->dtype()==mx::int8 && pos>0) {
                    nk=mx::slice_update(nk,*kb,{0,0,0,0},{kvheads,kb->shape(1),hd,kBlock});
                    nv=mx::slice_update(nv,mx::slice(*vb,{0,0,0},{kvheads,pos,hd}),{0,0,0},{kvheads,pos,hd});
                    nks=mx::slice_update(nks,mx::slice(*ks,{0,0},{kvheads,pos}),{0,0},{kvheads,pos});
                    nvs=mx::slice_update(nvs,mx::slice(*vs,{0,0},{kvheads,pos}),{0,0},{kvheads,pos});
                }
                kb=nk;vb=nv;ks=nks;vs=nvs;
            } else {
                auto nk=mx::zeros({kvheads,cap,26},mx::uint32),nv=mx::zeros({kvheads,cap,18},mx::uint32);
                if(kb && kb->dtype()==mx::uint32 && pos>0) {
                    nk=mx::slice_update(nk,mx::slice(*kb,{0,0,0},{kvheads,pos,26}),{0,0,0},{kvheads,pos,26});
                    nv=mx::slice_update(nv,mx::slice(*vb,{0,0,0},{kvheads,pos,18}),{0,0,0},{kvheads,pos,18});
                }
                kb=nk;vb=nv;ks.reset();vs.reset();
            }
        }
        if(int8) {
            auto r=quantize_rows_int8(k_in,v_in);
            vb=mx::slice_update(*vb,r.v,{0,pos,0},{kvheads,need,hd});
            ks=mx::slice_update(*ks,r.kscale,{0,pos},{kvheads,need});
            vs=mx::slice_update(*vs,r.vscale,{0,pos},{kvheads,need});
            for(int at=0;at<m;) {  // K^T columns pos .. need - 1, split at block boundaries
                const int blk=(pos+at)/kBlock,col=(pos+at)%kBlock,take=std::min(m-at,kBlock-col);
                auto part=take==m?r.kt:mx::slice(r.kt,{0,0,at},{kvheads,hd,at+take});
                kb=mx::slice_update(*kb,mx::reshape(part,{kvheads,1,hd,take}),{0,blk,0,col},{kvheads,blk+1,hd,col+take});
                at+=take;
            }
        } else {
            auto rows_of=[&](const mx::array& x) {return mx::transpose(mx::reshape(mx::astype(x,mx::float32),{kvheads,m,hd}),{1,0,2});};  // [m, kvh, hd]
            auto kc=mx::transpose(tcq_encode(rows_of(k_in),3),{1,0,2}),vc=mx::transpose(tcq_encode(rows_of(v_in),2),{1,0,2});
            kb=mx::slice_update(*kb,kc,{0,pos,0},{kvheads,need,26});
            vb=mx::slice_update(*vb,vc,{0,pos,0},{kvheads,need,18});
        }
        c.kv_len=need;
        const float scale=1.0f/std::sqrt(float(hd));
        mx::array out=mx::array(0.0f);
        if(m<=16) {
            // int8 verify: v16 at 32-key rounds (= v12 bit for bit, test-mlx-decode-rows; j1188 64K 1237 vs 1361 us at 8 rows,
            // 2418 vs 2910 at 16), v12 for a single row
            out=int8?(m>1?decode_rows_attention_v16(mx::astype(q,mx::float32),*kb,*vb,*ks,*vs,scale,need,32):
                          decode_rows_attention_v12(mx::astype(q,mx::float32),*kb,*vb,*ks,*vs,scale,need)):
                     decode_rows_attention_tcq(q,*kb,*vb,scale,need);
        } else {
            mx::array keys=mx::array(0.0f),values=mx::array(0.0f);
            if(int8) {
                // FP16 throughout (codes x FP16 scale): no FP32 copy of the prefix
                auto kt=mx::reshape(mx::transpose(*kb,{0,1,3,2}),{kvheads,kb->shape(1)*kBlock,hd});
                keys=mx::multiply(mx::astype(mx::slice(kt,{0,0,0},{kvheads,need,hd}),mx::float16),mx::expand_dims(mx::slice(*ks,{0,0},{kvheads,need}),-1));
                values=mx::multiply(mx::astype(mx::slice(*vb,{0,0,0},{kvheads,need,hd}),mx::float16),mx::expand_dims(mx::slice(*vs,{0,0},{kvheads,need}),-1));
                keys=mx::reshape(keys,{1,kvheads,need,hd});values=mx::reshape(values,{1,kvheads,need,hd});
            } else {keys=tcq_decode(*kb,3,need);values=tcq_decode(*vb,2,need);}
            out=mx::astype(prefill_attention_rows(mx::astype(q,mx::float16),keys,values,scale),q.dtype());
            // the decoded prefix lives for this layer only: without the boundary the lazy chunk graph may hold several
            // layers' FP16 prefixes at once (~1 GB each at 256K)
            mx::eval(out);
            static const bool memlog=tqllm::debug_on("memlog");
            if(memlog)std::fprintf(stderr,"MEMLOG attn layer %d rows %d keys %d active %.2f GB peak %.2f GB\n",layer,m,need,mx::get_active_memory()/1e9,mx::get_peak_memory()/1e9);
        }
        out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
        stage("attn_core",{out,*kb,*vb});
        out=gate_mul_bf(out,gate);
        stage("attn_gate",{out});
        return output_projection(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
    }
    // the shared attention core: KV cache append, causal attention, gating and the output projection
    mx::array attn_core(const std::string& p,const mx::array& q_in,const mx::array& k_in,const mx::array& v_in,
                        const mx::array& gate,Cache& c,int offset,int layer,int m) {
        auto q=q_in,k=k_in,v=v_in;
        if(layer<0) {  // MTP cache: appended by concatenation (its inject path does the same), BF16 (exact: BF16-rounded)
            c.keys=c.keys?mx::concatenate({*c.keys,mx::astype(k,mx::bfloat16)},2):mx::astype(k,mx::bfloat16);
            c.values=c.values?mx::concatenate({*c.values,mx::astype(v,mx::bfloat16)},2):mx::astype(v,mx::bfloat16);
            auto out=causal_attention_rows(q,*c.keys,*c.values,1.0f/std::sqrt(float(hd)));
            out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
            out=gate_mul_bf(out,gate);
            return mat(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
        }
        if(kv_tier!=KvTier::Bf16) return attn_core_compressed(p,q,k,v,gate,c,layer,m);
        auto& kb=kv_keys[layer];auto& vb=kv_values[layer];
        const int need=c.kv_len+m;
        // keys and values are already BF16-rounded, so BF16 storage is exact and halves the cache
        k=mx::astype(k,mx::bfloat16);v=mx::astype(v,mx::bfloat16);
        if(!kb || kb->shape(2)<need || kb->dtype()!=k.dtype()) {
            // one allocation per request when it announced its length, else x1.5 growth (kv_capacity.hpp); whole segments
            const int cap=kv_capacity(kb && kb->dtype()==k.dtype()?kb->shape(2):0,need,kv_reserve,context,kAttentionSegment);
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
        auto out=causal_attention_rows(q,*kb,*vb,1.0f/std::sqrt(float(hd)),need);
        out=mx::reshape(mx::transpose(out,{0,2,1,3}),{m,heads,hd});
        stage("attn_core",{out,*kb,*vb});
        out=gate_mul_bf(out,gate);
        stage("attn_gate",{out});
        return output_projection(p+"attn_output.weight",mx::reshape(out,{m,heads*hd}));
    }
    mx::array gdn(const std::string& p,const mx::array& x,Cache& c,PrefixTape& tape) {
        const int m=x.shape(0),width=2*kh*dk+vh*dv;
        auto weights=mx::reshape(plain.at(p+"ssm_conv1d.weight"),{width,conv});
        // prefill chunks with the qkv | z pack: conv + q/k norms and the gated norm read the FP16 product in place
        if(auto full=pack_full16(p+"gdn_in",x);full && !capture_prefix && conv==4 && dk==128 && dv==128 && m>=conv) {
            stage("gdn_in_gemm",{*full});
            auto history=c.conv?*c.conv:mx::zeros({conv-1,width});
            auto [q2,k2,v2]=gdn_pre_pack(history,*full,width,weights,kh,1.0f/std::sqrt(float(dk)));
            c.conv=mx::contiguous(round_product(mx::slice(*full,{m-(conv-1),0},{m,width})));
            stage("gdn_pre",{q2,k2,v2,*c.conv});
            auto q=mx::reshape(q2,{1,m,kh,dk}),k=mx::reshape(k2,{1,m,kh,dk}),v=mx::reshape(v2,{1,m,vh,dv});
            auto a=mat(p+"ssm_alpha.weight",x),b=mat(p+"ssm_beta.weight",x);
            stage("gdn_ab_proj",{a,b});
            auto gates=gdn_gates_native(a,b,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a"));
            auto decay=mx::reshape(gates.first,{1,m,vh}),beta=mx::reshape(gates.second,{1,m,vh});
            stage("gdn_gates",{decay,beta});
            auto scanned=gdn_scan(q,k,v,decay,beta,c.state,false,false);
            c.state=scanned.state;
            stage("gdn_scan",{scanned.output,scanned.state});
            auto out=gdn_post_pack(scanned.output,*full,width,plain.at(p+"ssm_norm.weight"),eps);
            stage("gdn_post",{out});
            return output_projection(p+"ssm_out.weight",mx::reshape(out,{m,vh*dv}));
        }
        auto in=pack_product16(p+"gdn_in",x);   // qkv | z
        // one launch for conv, gates, q/k norms, the 1-16 token scan, the compact tape and the gated norm (j762: verify
        // -2.2 ms); on by default since j892 (TQLLM_MLX_GDN_STEP=0 off). Decode, verify and the accept replay share its recurrence
        static const bool gdn_step=std::getenv("TQLLM_MLX_GDN_STEP") && std::string(std::getenv("TQLLM_MLX_GDN_STEP"))!="0";
        const bool step_path=gdn_step && m<=16 && conv==4 && dk==128 && dv==128;
        auto qkv=in.empty()?mat(p+"attn_qkv.weight",x):round_product(in[0]);
        stage("gdn_proj_in",{qkv});
        auto history=c.conv?*c.conv:mx::zeros({conv-1,width});
        if(step_path) {
            if(capture_prefix) {tape.conv_prev=history;tape.conv_rows=qkv;}
            auto z=mat(p+"attn_gate.weight",x);
            // alpha / beta projected inside the step kernel from the (TQ4: rotated) input; else the two dense GEMVs
            auto ab=ab_proj.find(p);
            auto gates_in=[&]()->std::pair<mx::array,mx::array> {
                if(ab==ab_proj.end())return {mat(p+"ssm_alpha.weight",x),mat(p+"ssm_beta.weight",x)};
                if(!tq4_basis)return {x,ab->second};
                if(!dense_rotated_input || dense_rotated_input->first.id()!=x.id())dense_rotated_input=std::pair{x,tq4_rotate(x)};
                return {dense_rotated_input->second,ab->second};
            }();
            auto step=gdn_step_fused(history,qkv,weights,gates_in.first,gates_in.second,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a"),z,
                plain.at(p+"ssm_norm.weight"),eps,1.0f/std::sqrt(float(dk)),c.state,kh,vh);
            c.conv=step.history;c.state=step.state;
            if(capture_prefix) {tape.states.reset();tape.keys=step.keys;tape.decay=step.decay;tape.deltas=step.deltas;}
            stage("gdn_scan",{step.out,step.state});
            return mat(p+"ssm_out.weight",step.out);
        }
        auto joined=mx::concatenate({history,qkv},0);
        if(capture_prefix)tape.conv_inputs=joined;
        // fused prep / post kernels on prefill chunks only: verify (16 rows, prefix capture) must run the same ops as AR
        // (1 row) or spec != AR (j743: doc8k x 512 diverged at token 367 when verify's rows took the fused kernels)
        const bool fused=m>16 && !capture_prefix && dk==128 && dv==128;
        auto result=fused?gdn_pre_fused(joined,weights,kh,1.0f/std::sqrt(float(dk))):gdn_convolve_native(joined,weights);
        c.conv=mx::contiguous(tail_slice(joined,0,m,m+conv-1));
        auto q=mx::reshape(tail_slice(result,1,0,kh*dk),{1,m,kh,dk});
        auto k=mx::reshape(tail_slice(result,1,kh*dk,2*kh*dk),{1,m,kh,dk});
        auto v=mx::reshape(tail_slice(result,1,2*kh*dk,width),{1,m,vh,dv});
        if(!fused) {
            q=gdn_normalize(q,1.0f/std::sqrt(float(dk)),true);
            k=gdn_normalize(k,1,true);
        }
        auto a=mat(p+"ssm_alpha.weight",x),b=mat(p+"ssm_beta.weight",x);
        auto gates=gdn_gates_native(a,b,plain.at(p+"ssm_dt.bias"),plain.at(p+"ssm_a"));
        auto decay=mx::reshape(gates.first,{1,m,vh}),beta=mx::reshape(gates.second,{1,m,vh});
        stage("gdn_prep",{q,k,v,decay,beta});
        auto scanned=gdn_scan(q,k,v,decay,beta,c.state,capture_prefix,capture_prefix);
        if(capture_prefix) {tape.states=scanned.snapshots;tape.keys=k;tape.decay=decay;tape.deltas=scanned.deltas;}
        c.state=scanned.state;
        stage("gdn_scan",{scanned.output,scanned.state});
        auto z=mx::reshape(in.empty()?mat(p+"attn_gate.weight",x):round_product(in[1]),{1,m,vh,dv});
        auto out=fused?gdn_post_fused(scanned.output,z,plain.at(p+"ssm_norm.weight"),eps):
            silu_mul_bf(mx::fast::rms_norm(bf(scanned.output),plain.at(p+"ssm_norm.weight"),eps),z);
        return output_projection(p+"ssm_out.weight",mx::reshape(out,{m,vh*dv}));
    }
    mx::array forward(std::span<const uint32_t> tokens,bool all_logits) {
        auto start=std::chrono::steady_clock::now();
        if(tokens.empty()||tokens.size()>size_t(context-pos)) throw std::invalid_argument("MLX token batch exceeds remaining context");
        stages.clear();dense_input.reset();
        if(stage_profiling) {  // the cost of one sync on an empty graph, to subtract per stage call
            auto one=mx::array(1.0f);mx::eval(one);
            auto t0=std::chrono::steady_clock::now();
            for(int i=0;i<50;++i) {one=mx::add(one,mx::array(1.0f));mx::eval(one);}
            stages["sync_empty_us_each"]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/50;
        }
        stage_tick=std::chrono::steady_clock::now();
        auto record=[&](const char* name,const std::vector<mx::array>& arrays) {stage(name,arrays);};
        const int m=int(tokens.size());
        std::vector<mx::array> features;
        static const bool wave=std::getenv("TQLLM_MLX_ANE_WAVE")!=nullptr;
        if(wave)ensure_ane();
        const int wave_rows=wave?(ane?ane->rows():std::atoi(std::getenv("TQLLM_MLX_ANE_WAVE"))):0;
        const bool use_wave=wave && wave_rows>0 && m==2*wave_rows && !capture_prefix && !stage_profiling;
        auto x=use_wave?wave_layers(tokens,features):bf(embed(tokens));
        bool wave_features=use_wave;
        if(stage_profiling)record("embedding",{x});
        std::optional<DispatchCount> dispatches;
        if(std::getenv("TQLLM_MLX_COUNT_NODES")) {dispatches.emplace();dispatches->take({x},"embed");}
        auto cache_roots=[&](int l,std::vector<mx::array> roots) {
            const auto& c=caches[l];
            for(const std::optional<mx::array>* a:std::initializer_list<const std::optional<mx::array>*>{&c.conv,&c.state,&kv_keys[l],&kv_values[l],&kv_kscale[l],&kv_vscale[l]})if(*a)roots.push_back(**a);
            for(const auto* a:{&tapes[l].conv_inputs,&tapes[l].conv_prev,&tapes[l].conv_rows,&tapes[l].states,&tapes[l].keys,&tapes[l].decay,&tapes[l].deltas})if(*a)roots.push_back(**a);
            return roots;
        };
        // one layer on rows starting at `offset`; prefill batches above the slice size run it per slice, layers outer
        auto layer_step=[&](int l,mx::array& x,int offset,std::vector<mx::array>& features) {
            if(std::binary_search(feature_ids.begin(),feature_ids.end(),l))features.push_back(x);
            const std::string p="blk."+std::to_string(l)+".";
            static const bool pack_on0=env_default_on("TQLLM_MLX_PREFILL_PACK");
            const bool rows16=x.shape(0)<=16;
            auto n=(pack_on0 && !capture_prefix && x.shape(0)>=64)?norm_prefill(x,p+"attn_norm.weight"):rows16?norm_rows(x,p+"attn_norm.weight"):norm(x,p+"attn_norm.weight");
            stage("norm_in",{n});
            const bool is_attn=(l+1)%interval==0;
            // prefill chunks: the output projection adds the residual (mixer_added), the down product likewise
            static const bool pack_on=env_default_on("TQLLM_MLX_PREFILL_PACK");
            const bool fuse=pack_on && !capture_prefix && x.shape(0)>=64;
            mixer_residual=fuse?&x:nullptr;mixer_added=false;
            auto attn=is_attn?full_attention(p,n,caches[l],offset,l):gdn(p,n,caches[l],tapes[l]);
            mixer_residual=nullptr;
            if(stage_profiling) {
                std::vector<mx::array> values={attn};const auto& c=caches[l];
                for(const std::optional<mx::array>* a:std::initializer_list<const std::optional<mx::array>*>{&c.conv,&c.state,&kv_keys[l],&kv_values[l],&kv_kscale[l],&kv_vscale[l]})if(*a)values.push_back(**a);
                record(mixer_added?(is_attn?"attn_out_gemm_res":"gdn_out_gemm_res"):((l+1)%interval==0?"attention":"gdn_mixer"),values);
            }
            if(dispatches)dispatches->take(cache_roots(l,{attn}),std::string(is_attn?"attn":"gdn")+"/mixer");
            if(!mixer_added && rows16) {auto an=add_norm_rows(x,attn,p+"post_attention_norm.weight");x=an.first;n=an.second;}
            else {x=mixer_added?attn:add_bf(x,attn);n=fuse?norm_prefill(x,p+"post_attention_norm.weight"):norm(x,p+"post_attention_norm.weight");}
            stage("norm_post",{n});
            if(auto down=ffn_gemm(p,n,fuse?&x:nullptr)) x=fuse?*down:add_bf(x,*down);
            else if(auto fused=swiglu_rows(p,n)) {
                if(stage_profiling)record("ffn_gate_up",{*fused});
                x=add_bf(x,mat(p+"ffn_down.weight",*fused));
            }
            else {
                auto [gate,up]=gate_up(p,n);
                auto ff=bf(mx::multiply(bf(silu(gate)),up));
                if(stage_profiling)record("ffn_gate_up",{ff});
                x=add_bf(x,mat(p+"ffn_down.weight",ff));
            }
            if(stage_profiling)record(fuse?"ffn_down_gemm_res":"ffn_down",{x});
            if(dispatches)dispatches->take({x},std::string(is_attn?"attn":"gdn")+"/ffn");
        };
        // TQLLM_MLX_PREFILL_GROUP=S: a prefill batch of more than S rows runs layer-major in S-row slices, each slice with
        // the ops (shapes, positions, cache order) of its own S-row forward, so the result equals S-row chunks while each
        // layer's weights are dequantized once for the batch (pass a chunk of G x S rows); the layer is evaluated before
        // the next so one layer's FP16 weights are live at a time
        // default 2048 since j769 (bitwise equal to 2048-row chunks; prefill +3-6 %: mlxc doc8k 453 -> 480); =0 off
        static const int group_rows=std::getenv("TQLLM_MLX_PREFILL_GROUP")?std::atoi(std::getenv("TQLLM_MLX_PREFILL_GROUP")):2048;
        const bool layer_major=group_rows>16 && m>group_rows && !use_wave && !capture_prefix && !dispatches;  // stage profiles sync per stage inside it
        if(layer_major) {
            const int slice_rows=group_rows;
            const int slices=(m+slice_rows-1)/slice_rows;
            std::vector<mx::array> xs;std::vector<std::vector<mx::array>> feats(slices);
            for(int s=0;s<slices;++s) {
                const int a=s*slice_rows,b=std::min(m,a+slice_rows);
                xs.push_back(bf(embed(tokens.subspan(size_t(a),size_t(b-a)))));
            }
            share_dequant=true;
            try {
                for(int l=0;l<layers;++l) {
                    for(int s=0;s<slices;++s)layer_step(l,xs[s],pos+s*slice_rows,feats[s]);
                    std::vector<mx::array> done;
                    for(int s=0;s<slices;++s)done.push_back(xs[s]);
                    for(const auto& c:caches)for(const std::optional<mx::array>* a:{&c.conv,&c.state})if(*a)done.push_back(**a);
                    for(const auto* kv:{&kv_keys,&kv_values,&kv_kscale,&kv_vscale})if((*kv)[l])done.push_back(*(*kv)[l]);
                    for(const auto& f:feats)for(const auto& a:f)done.push_back(a);
                    mx::eval(done);
                    dequant_cache.clear();
                    static const bool memlog=tqllm::debug_on("memlog");
                    if(memlog && m!=8192)std::fprintf(stderr,"MEMLOG layer %d m %d slices %d active %.2f GB peak %.2f GB\n",l,m,slices,mx::get_active_memory()/1e9,mx::get_peak_memory()/1e9);
                }
            } catch(...) {share_dequant=false;dequant_cache.clear();throw;}
            share_dequant=false;
            x=mx::concatenate(xs,0);
            if(!feature_ids.empty()) {
                std::vector<mx::array> rows;
                for(auto& f:feats)rows.push_back(mx::concatenate(f,1));
                features={mx::concatenate(rows,0)};
                wave_features=true;  // already joined: features[0] is the [m, layers x hidden] block
            }
        } else
        for(int l=0;l<(use_wave?0:layers);++l)layer_step(l,x,pos,features);
        last_hidden=x;
        if(!features.empty())last_features=wave_features?features[0]:mx::concatenate(features,1);
        x=norm(x,"output_norm.weight");
        if(!all_logits) x=tail_slice(x,0,m-1,m);
        std::vector<mx::array> live={x};
        if(last_features)live.push_back(*last_features);
        for(const auto& c:caches) {
            if(c.conv) live.push_back(*c.conv);
            if(c.state) live.push_back(*c.state);
        }
        for(const auto* kv:{&kv_keys,&kv_values,&kv_kscale,&kv_vscale})for(const auto& a:*kv)if(a)live.push_back(*a);
        if(capture_prefix)for(const auto& t:tapes)
            for(const auto* a:{&t.states,&t.keys,&t.decay,&t.deltas})if(*a)live.push_back(**a);
        if(profiling) mx::eval(live);
        auto body_end=std::chrono::steady_clock::now();
        auto logits=nax.contains("output.weight")?nax_affine_matmul(nax.at("output.weight"),x):linear(matrices.at("output.weight"),x);
        live.push_back(logits);
        if(dispatches)dispatches->take(live,"head+rest");
        if(defer_eval && !dispatches)mx::async_eval(live);else mx::eval(live);
        if(dispatches)dispatches->report(m,layers-layers/interval,layers/interval);
        auto end=std::chrono::steady_clock::now();
        times={std::chrono::duration<double,std::milli>(body_end-start).count(),
               std::chrono::duration<double,std::milli>(end-body_end).count()};
        pos+=m;++revision;dense_input.reset();
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
    if(tokens.empty() || tokens.size()>16)throw std::invalid_argument("MLX prefix verification takes 1-16 tokens");
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
mx::array Qwen35::verify_device(const mx::array& ids,Verification& batch) {
    if(ids.ndim()!=1 || ids.dtype()!=mx::uint32 || ids.shape(0)<1 || ids.shape(0)>16)
        throw std::invalid_argument("device verification needs uint32 ids [1..16]");
    const std::vector<uint32_t> placeholder(size_t(ids.shape(0)),0u);
    impl_->device_ids=ids;
    try {
        auto out=verify(placeholder,batch);
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
            // rows count .. count + 2 of [history; rows]
            const int k=int(count),h=impl_->conv-1;
            c.conv=tape.conv_rows?(k>=h?mx::contiguous(tail_slice(*tape.conv_rows,0,k-h,k)):
                                   mx::concatenate({tail_slice(*tape.conv_prev,0,k,h),tail_slice(*tape.conv_rows,0,0,k)},0)):
                mx::contiguous(tail_slice(*tape.conv_inputs,0,k,k+h));
            c.state=tape.deltas?gdn_replay_prefix(*tape.keys,*tape.decay,*tape.deltas,s->before[l].state,int(count)):
                mx::contiguous(tail_slice(*tape.states,0,int(count)-1,int(count)));
        }
    }
    // the replay runs inside the next forward's graph (lazy, j563: exact)
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
mx::array Qwen35::embed_device(const mx::array& ids) const {
    const auto& i=*impl_;
    return bf(i.embedding8?affine8_embedding(*i.embedding8,ids):q4_embedding(*i.embedding,ids));
}
void Qwen35::defer_eval(bool enabled) {impl_->defer_eval=enabled;}
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
    if(impl_->proposal_head)return;
    // a matrix-unit head (mlxc) is held only as its NAX repack: build the 4-bit proposal head from its FP16 values
    if(auto it=impl_->matrices.find("output.weight");it!=impl_->matrices.end())impl_->proposal_head=draft_head4(it->second);
    else impl_->proposal_head=draft_head4(Matrix{nax_affine_dequant(impl_->nax.at("output.weight"))});
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
    k=mx::astype(k,mx::bfloat16);v=mx::astype(v,mx::bfloat16);
    i.mtp_cache.keys=i.mtp_cache.keys?mx::concatenate({*i.mtp_cache.keys,k},2):k;
    i.mtp_cache.values=i.mtp_cache.values?mx::concatenate({*i.mtp_cache.values,v},2):v;
    mx::eval(*i.mtp_cache.keys,*i.mtp_cache.values);i.mtp_pos+=n;i.mtp_hidden.reset();
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
    i.mtp_pos+=n;
    return output;
}
namespace {
// A tq_mlx file selects the adopted kernel set; explicit environment settings still win.
}
Qwen35::Qwen35(const std::string& path,int context):impl_((apply_release_defaults(),std::make_unique<Impl>(path,context))) {}
Qwen35::~Qwen35()=default;
mx::array Qwen35::forward(std::span<const uint32_t> tokens,bool all_logits) {return impl_->forward(tokens,all_logits);}
void Qwen35::reserve_kv(int rows) {impl_->kv_reserve=std::max(0,rows);}
void Qwen35::kv_format(KvTier tier) {
    if(impl_->pos!=0)throw std::invalid_argument("the KV format is chosen per request, before its prefill");
    if(tier!=impl_->kv_tier)for(auto* kv:{&impl_->kv_keys,&impl_->kv_values,&impl_->kv_kscale,&impl_->kv_vscale})for(auto& a:*kv)a.reset();
    impl_->kv_tier=tier;
}
KvTier Qwen35::kv_format() const {return impl_->kv_tier;}
// the pages the system can hand us without swapping (kv_tier.hpp available_memory; 0 if unknown)
static size_t system_available_memory() {
#ifdef __APPLE__
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t n=HOST_VM_INFO64_COUNT;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,reinterpret_cast<host_info64_t>(&vm),&n)!=KERN_SUCCESS)return 0;
    return available_memory(vm.free_count,vm.inactive_count,vm.speculative_count,vm.purgeable_count,vm.external_page_count)*size_t(vm_kernel_page_size);
#else
    return 0;
#endif
}
KvBudget Qwen35::kv_budget(int prefill_chunk) const {
    KvBudget b;
    // the previous request's KV buffers are reused or freed by this one: not in use for its plan (j1159: the 121K int8 KV
    // pushed the next 248K request to trellis)
    size_t held=0;
    for(const auto* kv:{&impl_->kv_keys,&impl_->kv_values,&impl_->kv_kscale,&impl_->kv_vscale})for(const auto& a:*kv)if(a)held+=a->nbytes();
    const size_t active=mx::get_active_memory();
    b.in_use=active>held?active-held:0;
    b.working_set=usable_working_set(std::get<size_t>(mx::device_info(mx::Device::gpu).at("max_recommended_working_set_size")),
                                     active,system_available_memory());
    // the prefill's own transient: the 4 GiB buffer cache it runs under plus its chunk activations (j1140: peak - active =
    // 5.6 GB at 8192-row chunks of hidden 5120, i.e. ~40 B per chunk value)
    b.transient=(size_t(4096)<<20)+size_t(prefill_chunk)*size_t(impl_->hidden)*40;
    return b;
}
KvTier Qwen35::choose_kv_format(int rows,int prefill_chunk,size_t other_row_bytes) const {
    KvBudget b=kv_budget(prefill_chunk);
    b.in_use+=size_t(rows)*other_row_bytes;  // + what other caches (drafter) will hold for the request
    return choose_kv_tier(rows,attention_layers(),impl_->kvheads,impl_->hd,b);
}
KvPlan Qwen35::plan_kv_format(int prompt_rows,int answer_rows,int prefill_chunk,size_t far_row_bytes) const {
    const int rows=prompt_rows+answer_rows+16;
    const size_t lean=(size_t(512)<<20)+size_t(prefill_chunk)*size_t(impl_->hidden)*40;  // the decode-time cache limit
    return plan_kv(rows,attention_layers(),impl_->kvheads,impl_->hd,kv_budget(prefill_chunk),far_row_bytes,lean,request_kv_first(prompt_rows,answer_rows));
}
int Qwen35::attention_layers() const {return impl_->layers/impl_->interval;}
int Qwen35::kv_heads() const {return impl_->kvheads;}
int Qwen35::q_heads() const {return impl_->heads;}
int Qwen35::head_dim() const {return impl_->hd;}
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
std::pair<double,double> Qwen35::timings() const {return impl_->times;}
}

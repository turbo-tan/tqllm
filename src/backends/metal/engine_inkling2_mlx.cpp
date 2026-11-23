#include "engine_inkling2_mlx.hpp"
#include "inkling2_mlx.hpp"
#include "sampling_topk.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>
#include <random>
#include <stdexcept>

namespace tqllm {
namespace mx=mlx::core;
namespace {
class MlxInkling2Engine final : public Engine {
    using Clock=std::chrono::steady_clock;
public:
    // drafter: "" (none), the checkpoint folder (--mtp: its mtp.safetensors) or a DSpark drafter GGUF (-md FILE.gguf)
    MlxInkling2Engine(const std::string& folder,uint32_t context,const std::string& drafter,bool exact) : model_(folder,int(context),exact) {
        if(drafter==folder && !std::filesystem::exists(std::filesystem::path(folder)/"mtp.safetensors"))
            throw std::runtime_error("inkling2 Metal: --mtp needs mtp.safetensors in "+folder);
        if(!drafter.empty()) model_.enable_draft(drafter);
    }
    bool has_drafter() const override {return model_.drafting();}
    bool native_mtp() const override {return model_.drafting();}
    bool adaptive_depth() const override {return model_.drafting();}  // speculate(token, 0): the model's depth policy
    uint32_t step(uint32_t token,uint32_t pos,bool sample,float* capture) override {
        if(capture) throw std::runtime_error("Inkling2 Metal does not expose CUDA-style captures");
        if(pos==0) {model_.reset();inflight_.reset();}
        if(pos!=uint32_t(model_.position())) throw std::invalid_argument("Inkling2 Metal step position");
        auto lg=model_.forward(std::span<const uint32_t>(&token,1));
        return sample?pick(lg):0;
    }
    bool incremental() const override {return true;}
    void set_ignore_eos(bool value) override {ignore_eos_=value;}
    void set_first_token_hook(std::function<void()> hook) override {first_token_=std::move(hook);}
    void set_sampling(float temperature,float top_p,int top_k,uint64_t seed,float) override {
        temperature_=std::max(temperature,0.f);top_p_=std::clamp(top_p,0.f,1.f);
        if(top_p_==0) top_p_=1;top_k_=top_k;rng_.seed(seed);
    }
    void spec_begin(const std::vector<uint32_t>& prompt,uint32_t count,int depth) override {
        if(prompt.empty() || prompt.size()>=size_t(model_.context())) throw std::invalid_argument("Inkling2 Metal prompt/context");
        model_.reset();inflight_.reset();output_.clear();steps_=drafted_=accepted_=0;done_=false;
        fixed_depth_=depth>0?std::min(depth,7):0;
        count_=std::min(count,uint32_t(model_.context())-uint32_t(prompt.size()));
        std::optional<mx::array> logits;
        auto start=Clock::now();
        for(size_t p=0;p<prompt.size();) {
            const size_t n=std::min(prefill_chunk(p),prompt.size()-p);
            logits=model_.forward(std::span<const uint32_t>(prompt).subspan(p,n));
            p+=n;
        }
        decode_start_=Clock::now();prefill_us_=uint64_t(std::chrono::duration<double,std::micro>(decode_start_-start).count());
        done_=count_==0;
        if(!done_) {emit(pick(*logits));if(first_token_) first_token_();}
    }
    bool spec_advance() override {
        if(done_) return false;
        const uint32_t token=output_.back();
        if(temperature_==0 && model_.drafting()) {
            // DSpark: the request's fixed d, or 0 (the model picks the depth from acceptance and step time)
            const int room=model_.context()-model_.position()-1;
            if(room<=1) {done_=true;return false;}
            const int depth=fixed_depth_?std::min(fixed_depth_,room-1):0;
            auto spec=model_.speculate(token,depth);
            ++steps_;drafted_+=uint64_t(spec.drafted);accepted_+=uint64_t(spec.accepted);
            for(uint32_t t:spec.tokens) {emit(t);if(done_) break;}
            return !done_;
        }
        if(temperature_==0) {
            // Greedy: the next step is queued on the GPU before this one's token is read, so building its graph
            // overlaps the running step. A step queued past EOS is discarded with the request.
            if(!inflight_) inflight_=model_.greedy_step(mx::array(&token,{1},mx::uint32));
            std::optional<mx::array> following;
            if(output_.size()+1<count_) following=model_.greedy_step(*inflight_);
            const uint32_t next=inflight_->item<uint32_t>();
            inflight_=std::move(following);
            ++steps_;emit(next);return !done_;
        }
        auto logits=model_.forward(std::span<const uint32_t>(&token,1));
        ++steps_;emit(pick(logits));return !done_;
    }
    void spec_abort(int slot) override {
        select_slot(slot);done_=true;output_.clear();model_.reset();inflight_.reset();
    }
    const std::vector<uint32_t>& spec_tokens(int slot) const override {
        if(slot!=0) throw std::invalid_argument("Inkling2 Metal has one slot");return output_;
    }
    std::vector<uint32_t> spec_finish(uint64_t* stats) override {
        done_=true;
        if(stats) {stats[0]=steps_;stats[1]=drafted_;stats[2]=accepted_;stats[3]=uint64_t(std::chrono::duration<double,std::micro>(Clock::now()-decode_start_).count());stats[4]=prefill_us_;}
        return output_;
    }
    std::vector<uint32_t> generate_spec(const std::vector<uint32_t>& prompt,uint32_t count,int depth,uint64_t* stats) override {
        spec_begin(prompt,count,depth);while(spec_advance()) {}return spec_finish(stats);
    }
    void synchronize() override {mx::synchronize();}
    uint32_t layers() const override {return uint32_t(model_.layer_count());}
    uint32_t hidden() const override {return uint32_t(model_.hidden_size());}
    uint32_t vocab() const override {return uint32_t(model_.vocabulary());}
    uint32_t eos() const override {return 200006;}
    uint32_t context() const override {return uint32_t(model_.context());}
private:
    uint32_t pick(const mx::array& logits) {
        if(temperature_==0) return mx::argmax(logits,-1).item<uint32_t>();
        const int k=top_k_>0?std::min(top_k_,model_.vocabulary()):64;
        return mlx_backend::sample_rows(logits,temperature_,k,top_p_,rng_()).item<uint32_t>();
    }
    void emit(uint32_t token) {
        output_.push_back(token);
        done_=output_.size()>=count_ || (!ignore_eos_ && token==eos());
    }
    // Prompt chunks of 4096 rows: attention builds no mask (inkling2_prompt_attention), so the width does not depend on
    // the context, and a chunk is one graph, so wider chunks fill the grouped expert tiles: prefill +13 % over 1024 at
    // 8K / 32K on an M5 Pro, peak memory +1.2 GB (8192: +1 % more for +2 GB more).
    static size_t prefill_chunk(size_t) { return 4096; }
    mlx_backend::Inkling2Mlx model_;
    std::vector<uint32_t> output_;
    std::optional<mx::array> inflight_;  // greedy step queued on the GPU, token not read yet
    std::function<void()> first_token_;
    std::mt19937_64 rng_{0};
    Clock::time_point decode_start_;
    uint64_t steps_=0,prefill_us_=0,drafted_=0,accepted_=0;
    int fixed_depth_=0;
    uint32_t count_=0;
    float temperature_=0,top_p_=1;
    int top_k_=0;
    bool done_=true,ignore_eos_=false;
};
}
std::unique_ptr<Engine> make_mlx_inkling2_engine(const std::string& model,const std::string&,uint32_t context,int slots,const EngineOptions& options) {
    if(slots!=1) throw std::invalid_argument("Inkling2 Metal requires --np 1");
    const auto& md=options.draft_model;
    if(!md.empty() && (md.size()<5 || md.compare(md.size()-5,5,".gguf")!=0))
        throw std::invalid_argument("Inkling2 Metal drafts with its DSpark head: --mtp, or -md a DSpark drafter GGUF");
    return std::make_unique<MlxInkling2Engine>(model,context,!md.empty()?md:options.native_mtp?model:std::string(),options.exact);
}
}  // namespace tqllm

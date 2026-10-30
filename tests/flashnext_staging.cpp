#include "models/qwen/qwen4exp/fn_model.hpp"
#include <cstdio>
#include <cstring>
int main() {
    try {
        constexpr int L=48,H=64,F=64,E=4,K=2;
        constexpr size_t B=size_t(H)*F/64*18, block=3*B;
        std::vector<char> raw(E*block);
        for(int e=0;e<E;++e)for(int p=0;p<3;++p)for(size_t i=0;i<B;++i)
            raw[size_t(e)*block+size_t(p)*B+i]=char((e*17+p*23+i)&255);
        MixedExpertCache cache(L+1,E,H,F,K,size_t(1)<<20,[&](int layer,int e) {
            require(layer==L,"only the MTP layer has payloads");
            std::array<ExpertPayload,3> out;
            for(int p=0;p<3;++p)out[p]={GGML_GSQ_Q2_0,{raw.data()+size_t(e)*block+size_t(p)*B,B},{},true};
            return out;
        },nullptr);
        cache.set_prepacked([&](int layer,int e)->const char* { return layer==L ? raw.data()+size_t(e)*block : nullptr; });
        cache.prepare(L,{0});
        MixedExpertCache streamed(L+1,E,H,F,K,size_t(1)<<20,[&](int,int e) {
            std::array<ExpertPayload,3> out;
            for(int p=0;p<3;++p)out[p]={GGML_GSQ_Q2_0,{raw.data()+size_t(e)*block+size_t(p)*B,B},{},true};
            return out;
        },nullptr);
        streamed.set_prepacked([&](int,int e)->const char* { return raw.data()+size_t(e)*block; });
        streamed.prepare(L,{0});
        const size_t retained=streamed.stats().device_bytes;
        const std::vector<char> borrowed(L,1);
        for(int round=0;round<2;++round)for(int layer=0;layer<L;++layer) {
            streamed.prepare(layer,{0,1});
            DecodeModel::release_borrowed_prefill_bank(streamed,layer,borrowed);
            require(streamed.stats().device_bytes==retained,"borrowed prefill banks accumulate across layers");
            require(!streamed.bank_ready(layer) && streamed.has_layout(layer),"released bank lost its reload layout");
        }
        DecodeModel::release_borrowed_prefill_bank(streamed,L,borrowed);
        require(streamed.bank_ready(L),"unborrowed MTP bank was released");
        DecodeModel::PrefillStage st;
        bool rejected=false;
        try { DecodeModel::stage_begin(st,cache,L); }
        catch(const std::runtime_error & e) { rejected=std::strstr(e.what(),"staging is uninitialized")!=nullptr; }
        require(rejected,"zero-width staging was not rejected");
        const size_t need=DecodeModel::stage_required(cache,K);
        require(need>0,"MTP-only layer48 was omitted from staging discovery");
        DecodeModel::stage_alloc(st,need,K);DecodeModel::stage_begin(st,cache,L);
        ck(cudaStreamSynchronize(st.copy));
        require(st.K==K && st.plan==std::vector<int>({1,2,3}),"wrong MTP staging plan");
        size_t off[6];DecodeModel::stage_offsets(cache.bank_layout(L),K,off);
        std::vector<char> got(B);ck(cudaMemcpy(got.data(),st.half[0]+off[0],B,cudaMemcpyDeviceToHost));
        require(std::memcmp(got.data(),raw.data()+block,B)==0,"MTP staging changed projection bytes");
        for(auto hp:st.half)ck(cudaFree(hp));
        for(int i=0;i<2;++i){ck(cudaEventDestroy(st.ready[i]));ck(cudaEventDestroy(st.done[i]));}
        ck(cudaStreamDestroy(st.copy));
        std::puts("flashnext_staging: PASS (zero width rejected, MTP-only layer48 discovered, staged bytes exact, borrowed banks bounded across48layers)");
    } catch(const std::exception & e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}

#include "tq3_expert_cache.hpp"
#include "gguf.hpp"
#include "hot_cold_policy.hpp"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace tqllm;
static void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void require(bool b,const char * s) { if(!b) throw std::runtime_error(s); }
__global__ void delayed_read(const char * p,char * out) {
    const auto start=clock64(); while(clock64()-start<200000) {}
    if(threadIdx.x<32) out[threadIdx.x]=p[threadIdx.x];
}
static std::vector<char> read(const void * p,size_t n,cudaStream_t stream) {
    std::vector<char> out(n); ck(cudaMemcpyAsync(out.data(),p,n,cudaMemcpyDeviceToHost,stream)); ck(cudaStreamSynchronize(stream)); return out;
}
static void run(uint32_t type,bool legacy,bool holes=false) {
    constexpr int L=2,E=4,H=64,F=64,C=2;
    uint64_t block=0,block_bytes=0; require(gguf_block_layout(type,block,block_bytes),"missing format");
    using Raw=std::array<std::vector<char>,3>;
    std::array<std::array<Raw,E>,L> raw;
    for(int l=0;l<L;++l)for(int e=0;e<E;++e)for(int p=0;p<3;++p) {
        auto & r=raw[l][e][p];r.assign(size_t(H)*F/block*block_bytes,char(19+e*17+p*7+l));
        for(size_t at=0;at<r.size();at+=block_bytes) {const auto d=__float2half_rn(float(1+e+l));std::memcpy(r.data()+at,&d,2);}
    }
    auto payload=[&](int l,int e) {std::array<ExpertPayload,3> out;for(int p=0;p<3;++p)out[p]={type,{raw[l][e][p].data(),raw[l][e][p].size()}};return out;};
    cudaStream_t compute=nullptr,other=nullptr;ck(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    {
        int source_calls=0;bool source_allowed=true;
        MixedExpertCache reference(L,E,H,F,E,size_t(16)<<20,payload,compute);
        MixedExpertCache cache(L,E,H,F,C,size_t(16)<<20,[&](int l,int e) {
            require(source_allowed,"forbidden source reload after host priming");++source_calls;return payload(l,e);
        },compute);
        for(int l=0;l<L;++l) {reference.prepare(l,{0,1,2,3});cache.prepare(l,{0});
            const std::vector<int32_t> map(L*E,0);
            if(!holes)cache.prepare(l,complete_hot_residency(map,l,E,cache.bank_state(l).ids,C));
            const auto populated=cache.bank_state(l).ids;require(std::count_if(populated.begin(),populated.end(),[](int x){return x>=0;})==C,"native GPU bank has unfilled slots before CPU complement priming");}
        size_t expert_bytes=0;
        for(int p=0;p<3;++p) {const auto m=cache.bank_mat(0,p);expert_bytes+=(m.fmt->row_bytes(H)+m.fmt->aux_row_bytes(H))*H;}
        const size_t budget=expert_bytes*L*(E-C);
        cache.set_host_victim_budget(budget);cache.set_miss_stream(other);
        require(cache.split(),"host victim miss stream is hidden from callers");
        for(int l=0;l<L;++l) require(cache.prime_host_victims(l,{0,1,2,3})==2,"wrong prime count");
        require(cache.host_victim_stats().bytes==budget,"prime budget differs");
        if(legacy) cache.enable_host_victims(false);
        source_allowed=false;const int primed_calls=source_calls;
        const std::array<std::vector<int>,5> requests={std::vector<int>{2,3},{0,1},{1,2},{3,0},{2,2}};
        char * probe=nullptr;ck(cudaMalloc(reinterpret_cast<void **>(&probe),32));
        for(int round=0;round<8;++round)for(int l=0;l<L;++l)for(const auto & ids:requests) {
            auto live=cache.bank_state(l).ids;live.erase(std::remove(live.begin(),live.end(),-1),live.end());
            cache.prepare(l,live);cache.wait_misses();
            const auto old=cache.bank_mat(l,0);const auto before=read(old.p,32,compute);
            delayed_read<<<1,32,0,compute>>>(static_cast<const char *>(old.p),probe);ck(cudaGetLastError());
            const auto slots=cache.prepare(l,ids);if(cache.split())cache.wait_misses();
            require(read(probe,32,compute)==before,"victim overwrite raced the last reader");
            for(size_t i=0;i<ids.size();++i)for(int p=0;p<3;++p) {
                const auto a=cache.bank_mat(l,p),b=reference.bank_mat(l,p);
                const size_t d=a.fmt->row_bytes(H)*H,u=a.fmt->aux_row_bytes(H)*H;
                require(read(static_cast<const char *>(a.p)+size_t(slots[i])*d,d,compute)==read(static_cast<const char *>(b.p)+size_t(ids[i])*d,d,compute),"native victim data changed");
                if(u)require(read(static_cast<const char *>(a.aux)+size_t(slots[i])*u,u,compute)==read(static_cast<const char *>(b.aux)+size_t(ids[i])*u,u,compute),"native victim auxiliary plane changed");
            }
            require(cache.host_victim_stats().bytes<=budget,"host victim budget exceeded");
        }
        require(source_calls==primed_calls,"host hit reread or repacked the source");
        require(cache.host_victim_stats().hits>0 && cache.host_victim_stats().spills>0,"victim swaps did not execute");
        source_allowed=true;cache.release_bank(0);cache.prepare(0,{0,1});cache.wait_misses();
        cache.set_host_victim_budget(expert_bytes);
        for(int l=0;l<L;++l)cache.prime_host_victims(l,{0,1,2,3});
        cache.prepare(0,{2,3});cache.wait_misses();require(cache.host_victim_stats().bytes<=expert_bytes,"small budget fallback overflowed");
        bool fault=true,failed=false;
        cache.set_prepacked([&](int,int)->const char * {if(fault)throw std::runtime_error("injected post-spill failure");return nullptr;},false,MixedExpertCache::PrepackedLayout::Indexed);
        const auto live=cache.bank_state(0).ids;int absent=0;while(std::find(live.begin(),live.end(),absent)!=live.end())++absent;
        try {cache.prepare(0,{absent});}catch(const std::runtime_error & e){failed=std::strstr(e.what(),"injected post-spill failure")!=nullptr;}
        require(failed,"post-spill failure did not fire");fault=false;
        const auto recovered=cache.prepare(0,{absent});cache.wait_misses();
        const auto cm=cache.gate(),rm=reference.bank_mat(0,0);const size_t stride=cm.fmt->row_bytes(H)*H;
        require(read(static_cast<const char *>(cm.p)+size_t(recovered[0])*stride,stride,compute)==read(static_cast<const char *>(rm.p)+size_t(absent)*stride,stride,compute),"post-spill failure poisoned recovery");
        ck(cudaGetLastError());cache.set_prepacked({});
        cache.set_host_victim_budget(0);require(cache.host_victim_stats().bytes==0,"disabled cache retained host pins");
        cache.prepare(0,{0,1});cache.wait_misses();cache.clear();ck(cudaFree(probe));
        std::printf("hot_victim: format %u PASS (data/aux, exclusive swaps, last-reader ordering, no source reload, bounded fallback/reset, post-spill failure recovery)\n",type);
    }
    ck(cudaStreamDestroy(other));ck(cudaStreamDestroy(compute));
}
int main(int argc,char ** argv) {
    try {run(GGML_GSQ_Q2_0,argc>1 && std::strcmp(argv[1],"--legacy")==0,argc>1 && std::strcmp(argv[1],"--holes")==0);run(GGML_IQ4_NL,false);}
    catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}

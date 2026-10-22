#include "tq3_expert_cache.hpp"
#include "hot_cold_policy.hpp"
#include "gguf.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <atomic>
#include <map>
using namespace tqllm;
static void ck(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
// The precision map has four hot experts; the working set needs only two cache slots per layer.
static void bounded_hot_banks(bool legacy_capacity) {
    constexpr int L=3,E=8,H=64,F=4096;
    constexpr size_t B=size_t(H)*F/64*18;
    std::array<std::array<std::array<std::vector<char>,3>,2>,L> raw;
    for(int l=0;l<L;++l)for(int e=0;e<2;++e)for(int p=0;p<3;++p) {
        raw[l][e][p].resize(B);
        for(size_t i=0;i<B;++i)raw[l][e][p][i]=char((i*7+l*31+e*17+p*23)&255);
    }
    const int cap=legacy_capacity ? 4 : hot_cache_slots(4,2);
    const size_t budget=L*2*3*B+3*B;
    size_t calls=0,first_calls=0;
    MixedExpertCache cache(L,E,H,F,cap,budget,[&](int l,int e) {
        ++calls;std::array<ExpertPayload,3> out;
        for(int p=0;p<3;++p)out[p]={GGML_GSQ_Q2_0,{raw[l][e][p].data(),B},{},true};
        return out;
    },nullptr);
    for(int pass=0;pass<20;++pass) {
        for(int l=0;l<L;++l) {
            const auto slots=cache.prepare(l,{0,1});
            const std::array<Mat,3> mat{cache.gate(),cache.up(),cache.down()};
            for(int e=0;e<2;++e)for(int p=0;p<3;++p) {
                std::vector<char> got(B);
                ck(cudaMemcpy(got.data(),static_cast<const char *>(mat[p].p)+size_t(slots[e])*B,B,cudaMemcpyDeviceToHost));
                assert(got==raw[l][e][p]);
            }
        }
        if(pass==0)first_calls=calls;
        else assert(calls==first_calls); // old full-size banks rebuild on every traversal
    }
    const auto stats=cache.stats();
    assert(stats.misses==L*2 && stats.evictions==0 && stats.hits==L*2*19);
    std::puts("bounded_hot_banks: PASS (all layers resident, no repeated source reads, exact projection bytes)");
}
int main(int argc,char **) {
    bounded_hot_banks(argc>1);

    constexpr int H=64,F=4096,E=4,N=80;
    constexpr size_t B=size_t(H)*F/64*18;
    std::array<std::array<std::vector<char>,3>,E> raw;
    for(int e=0;e<E;++e)for(int p=0;p<3;++p) {
        raw[e][p].resize(B); for(size_t i=0;i<B;++i)raw[e][p][i]=char((i*13+e*19+p*23)&255);
    }
    cudaStream_t compute,copy;ck(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
    {
        MixedExpertCache cache(2,E,H,F,2,size_t(4)<<20,[&](int,int e){
            std::array<ExpertPayload,3> r;
            for(int p=0;p<3;++p)r[p]={GGML_GSQ_Q2_0,{raw[e][p].data(),B},{},true};
            return r;
        },compute);
        cache.set_staged_misses(true); cache.set_miss_stream(copy); assert(cache.split());
        char * out=nullptr;ck(cudaMalloc(reinterpret_cast<void **>(&out),N*3*B));
        for(int i=0;i<N;++i) {
            ck(cudaLaunchHostFunc(compute,[](void *) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); },nullptr));
            const int e=(i/2)%E,layer=i%2;
            const auto ids=cache.prepare(layer,{e});cache.wait_misses();
            const std::array<Mat,3> m{cache.gate(),cache.up(),cache.down()};
            for(int p=0;p<3;++p)ck(cudaMemcpyAsync(out+(i*3+p)*B,static_cast<const char *>(m[p].p)+ids[0]*B,B,cudaMemcpyDeviceToDevice,compute));
        }
        ck(cudaStreamSynchronize(compute));std::vector<char> got(N*3*B);ck(cudaMemcpy(got.data(),out,got.size(),cudaMemcpyDeviceToHost));
        for(int i=0;i<N;++i)for(int p=0;p<3;++p)assert(std::memcmp(got.data()+(i*3+p)*B,raw[(i/2)%E][p].data(),B)==0);
        // A blocked later read must not prevent the first record reaching pinned staging.
        using Payload = std::array<ExpertPayload,3>;
        std::promise<void> consumed; auto first_consumed=consumed.get_future();
        auto owned_payload = [&](int e,bool notify) {
            auto owner=std::shared_ptr<std::array<std::vector<char>,3>>(new std::array<std::vector<char>,3>(raw[e]),
                [&,notify](auto * p) { for(auto & v:*p)std::fill(v.begin(),v.end(),0);delete p;if(notify)consumed.set_value(); });
            Payload r;for(int p=0;p<3;++p)r[p]={GGML_GSQ_Q2_0,{(*owner)[p].data(),B},owner,true};return r;
        };
        auto verify = [&](const std::vector<int> & physical) {
            const std::array<Mat,3> m{cache.gate(),cache.up(),cache.down()};cache.wait_misses();
            for(int e=0;e<2;++e)for(int p=0;p<3;++p)
                ck(cudaMemcpyAsync(out+(e*3+p)*B,static_cast<const char *>(m[p].p)+physical[e]*B,B,cudaMemcpyDeviceToDevice,compute));
            ck(cudaStreamSynchronize(compute));std::vector<char> got(6*B);ck(cudaMemcpy(got.data(),out,got.size(),cudaMemcpyDeviceToHost));
            for(int e=0;e<2;++e)for(int p=0;p<3;++p)assert(std::memcmp(got.data()+(e*3+p)*B,raw[e+2][p].data(),B)==0);
        };
        cache.prepare(0,{0,1});cache.wait_misses();ck(cudaStreamSynchronize(compute));
        std::promise<Payload> first,last;
        std::map<int,std::shared_future<Payload>> pending;
        pending.emplace(2,first.get_future().share());pending.emplace(3,last.get_future().share());
        first.set_value(owned_payload(2,true));first=std::promise<Payload>();
        cache.set_async_batch_source([&](int,const std::vector<int> & ids) {
            std::vector<std::shared_future<Payload>> out;for(int id:ids)out.push_back(std::move(pending.at(id)));return out;
        });
        auto run=std::async(std::launch::async,[&] {ck(cudaSetDevice(0));return cache.prepare(0,{2,3});});
        assert(first_consumed.wait_for(std::chrono::seconds(2))==std::future_status::ready);
        last.set_value(owned_payload(3,false));verify(run.get());
        cache.set_async_batch_source({});cache.prepare(0,{0,1});cache.wait_misses();ck(cudaStreamSynchronize(compute));
        std::promise<Payload> ok,bad;pending.clear();pending.emplace(2,ok.get_future().share());pending.emplace(3,bad.get_future().share());
        ok.set_value(owned_payload(2,false));bad.set_exception(std::make_exception_ptr(std::runtime_error("injected checksum failure")));
        std::atomic<bool> released{false};
        ck(cudaLaunchHostFunc(compute,[](void * p) {std::this_thread::sleep_for(std::chrono::milliseconds(100));*static_cast<std::atomic<bool> *>(p)=true;},&released));
        cache.set_async_batch_source([&](int,const std::vector<int> & ids) {
            std::vector<std::shared_future<Payload>> out;for(int id:ids)out.push_back(std::move(pending.at(id)));return out;
        });
        bool rejected=false;try{cache.prepare(0,{2,3});}catch(const std::runtime_error & e){rejected=std::strstr(e.what(),"injected checksum failure")!=nullptr;}
        assert(rejected && released.load()); // partial DMA drained before rethrow, even with a blocked release event
        cache.set_async_batch_source({});verify(cache.prepare(0,{2,3}));
        ck(cudaFree(out));cache.set_miss_stream(nullptr);
    }
    ck(cudaStreamDestroy(copy));ck(cudaStreamDestroy(compute));
    std::puts("hot_cold_staged: PASS (80 evicting prepares, two banks, streaming before later read, partial-failure recovery, all projection bytes exact)");
}

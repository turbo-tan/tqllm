#include "hot_cold_policy.hpp"
#include "gsq_q2_host.hpp"
#include "gguf.hpp"
#include <cassert>
#include <array>
#include <limits>
#include <iostream>
template<class F> void rejects(F f) { bool rejected = false; try { f(); } catch (const std::exception &) { rejected = true; } assert(rejected); }
int main() {
    for (float boundary : {-0.5f,0.5f,1.5f}) for(float ratio : {std::nextafter(boundary,-INFINITY),boundary,std::nextafter(boundary,INFINITY)})
        assert(tqllm::gsq_q2_code(ratio)==int(std::clamp(std::round(ratio),-1.f,2.f)));
    for(int i=-100000;i<=100000;++i) {
        const float ratio=float(i)*0.000073f;
        assert(tqllm::gsq_q2_code(ratio)==int(std::clamp(std::round(ratio),-1.f,2.f)));
    }
    const std::vector<int32_t> map{0,-1,1,-1, -1,0,-1,1};
    assert(tqllm::validate_hot_map(map,2,4)==2);
    assert(tqllm::hot_cache_slots(154,0)==154);
    for(int n:{64,80,96})assert(tqllm::hot_cache_slots(154,n)==n);
    assert(tqllm::hot_cache_slots(2,1)==1);
    rejects([]{tqllm::hot_cache_slots(154,155);});
    rejects([]{tqllm::hot_cache_slots(154,-1);});
    rejects([]{tqllm::hot_cache_slots(0,0);});
    const std::vector<int> ids{3,0,3,1,2}; std::vector<int> seen(ids.size(),0);
    for (bool hot : {false,true}) for(size_t a=0;a<ids.size();++a)
        if(tqllm::expert_is_hot(map,0,4,ids[a])==hot) ++seen[a];
    for(int n:seen) assert(n==1); // every assignment, including duplicates, belongs to exactly one bank
    rejects([]{tqllm::validate_hot_map({0,0,-1,-1},1,4);});
    rejects([]{tqllm::validate_hot_map({0,2,-1,-1},1,4);});
    rejects([]{tqllm::validate_hot_map({0,-2,-1,-1},1,4);});
    rejects([&]{tqllm::expert_is_hot(map,0,4,4);});
    std::vector<float> x(640), y(640);
    for(size_t i=0;i<x.size();++i) x[i]=float(int(i%4)-1)*0.125f;
    for(int sign:{1,-1}) {
        for(float &v:x)v*=sign;
        auto q=tqllm::encode_gsq_q2(x.data(),x.size()); assert(q.size()==180);
        tqllm::dequantize_host(tqllm::GGML_GSQ_Q2_0,q.data(),y.data(),int64_t(y.size()));
        for(size_t i=0;i<x.size();++i) assert(x[i]==y[i]);
    }
    for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*0.17f)*0.2f;
    auto lossy=tqllm::encode_gsq_q2(x.data(),x.size());
    for(unsigned workers : {1u,2u,8u,32u}) {
        assert(tqllm::encode_gsq_q2_parallel(x.data(),x.size(),workers)==lossy);
        assert(tqllm::requantize_gsq_q2_parallel(tqllm::GGML_F32,reinterpret_cast<const char *>(x.data()),x.size(),1,4,workers)==lossy);
    }
    rejects([&]{tqllm::requantize_gsq_q2_parallel(tqllm::GGML_F32,reinterpret_cast<const char *>(x.data()),x.size(),0,4,8);});
    for(const auto spec : {std::array<int,3>{18,256,98},std::array<int,3>{20,32,18},std::array<int,3>{302,64,18}}) {
        const size_t count=640*size_t(spec[1]); std::vector<char> source(count/spec[1]*spec[2],0);
        const uint16_t one=0x3c00;for(size_t b=0;b<count/spec[1];++b)std::memcpy(source.data()+b*spec[2],&one,2);
        std::vector<float> decoded(count);tqllm::dequantize_host(spec[0],source.data(),decoded.data(),int64_t(count));
        const auto expected=tqllm::encode_gsq_q2(decoded.data(),decoded.size());
        for(unsigned workers:{1u,2u,8u})assert(tqllm::requantize_gsq_q2_parallel(spec[0],source.data(),count,spec[1],spec[2],workers)==expected);
    }

    rejects([&]{tqllm::encode_gsq_q2_parallel(x.data(),x.size(),0);});
    tqllm::dequantize_host(tqllm::GGML_GSQ_Q2_0,lossy.data(),y.data(),int64_t(y.size()));
    double error=0,zero_error=0;for(size_t i=0;i<x.size();++i){error+=(x[i]-y[i])*(x[i]-y[i]);zero_error+=x[i]*x[i];}
    assert(error<zero_error*0.2);
    rejects([&]{tqllm::expert_is_hot(map,0,0,0);});
    std::fill(x.begin(),x.end(),0); auto zero=tqllm::encode_gsq_q2(x.data(),x.size());
    tqllm::dequantize_host(tqllm::GGML_GSQ_Q2_0,zero.data(),y.data(),int64_t(y.size()));
    for(float v:y) assert(v==0);
    rejects([&]{tqllm::encode_gsq_q2(x.data(),63);});
    x[0]=std::numeric_limits<float>::quiet_NaN();rejects([&]{tqllm::encode_gsq_q2(x.data(),x.size());});
    rejects([&]{tqllm::encode_gsq_q2_parallel(x.data(),x.size(),8);});
    std::cout<<"hot_cold_host: PASS\n";
}

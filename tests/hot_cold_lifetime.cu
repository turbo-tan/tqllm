#include "tq3_expert_cache.hpp"
#include "gguf.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>

using namespace tqllm;
static void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
// The fallback lets this reproducer run unchanged against the pre-fix payload API.
template<class T, class = void> struct AttachOwner { static void set(T &, const std::shared_ptr<const void> &) {} };
template<class T> struct AttachOwner<T, std::void_t<decltype(std::declval<T>().owner)>> {
    static void set(T & p, const std::shared_ptr<const void> & owner) { p.owner = owner; }
};
int main() {
    try {
        constexpr int experts = 3, hidden = 64, ff = 64;
        using Raw = std::array<std::vector<char>,3>;
        std::array<Raw,experts> storage;
        for (int e = 0; e < experts; ++e) for (auto & raw : storage[e]) {
            raw.resize(hidden * ff / 64 * 18, char(0xaa));
            const auto scale = __float2half_rn(float(e + 1));
            for (size_t b = 0; b < raw.size() / 18; ++b) std::memcpy(raw.data() + b * 18, &scale, 2);
        }
        int released = 0;
        std::map<int,std::shared_ptr<Raw>> memo;
        auto source = [&](int, int e) {
            memo.clear(); // force eviction while prepare collects the next expert's payload
            auto owner = std::shared_ptr<Raw>(&storage[e], [&](Raw * raw) {
                ++released;
                // Keep backing memory allocated, but poison it at semantic destruction: deterministic, no actual UAF.
                for (auto & bytes : *raw) std::fill(bytes.begin(), bytes.end(), 0);
            });
            memo.emplace(e,owner);
            std::array<ExpertPayload,3> result;
            for (int p = 0; p < 3; ++p) {
                result[p] = {GGML_GSQ_Q2_0, {(*owner)[p].data(),(*owner)[p].size()}};
                AttachOwner<ExpertPayload>::set(result[p],owner);
            }
            return result;
        };
        MixedExpertCache cache(1,experts,hidden,ff,experts,size_t(1)<<20,source,nullptr);
        const auto slots = cache.prepare(0,{0,1,2});
        __half * x = nullptr; int * ids = nullptr; __nv_bfloat16 * y = nullptr;
        check(cudaMalloc(&x,hidden * experts * sizeof(*x))); check(cudaMalloc(&ids,experts * sizeof(*ids)));
        check(cudaMalloc(&y,ff * experts * sizeof(*y)));
        const std::vector<__half> input(hidden * experts,__float2half_rn(1.f));
        check(cudaMemcpy(x,input.data(),input.size()*sizeof(*x),cudaMemcpyHostToDevice));
        check(cudaMemcpy(ids,slots.data(),slots.size()*sizeof(*ids),cudaMemcpyHostToDevice));
        const auto m = cache.gate(); m.fmt->expert_matvec(m,x,y,ff,ids,experts,1,false,nullptr);
        std::vector<__nv_bfloat16> output(ff*experts);
        check(cudaMemcpy(output.data(),y,output.size()*sizeof(*y),cudaMemcpyDeviceToHost));
        cudaFree(x);cudaFree(ids);cudaFree(y);
        for(int e=0;e<experts;++e) for(int row=0;row<ff;++row)
            if(__bfloat162float(output[e*ff+row])!=float(hidden*(e+1)))
                throw std::runtime_error("evicted source payload was released before prepare consumed it");
        if(released<2) throw std::runtime_error("test did not exercise source eviction");
        std::puts("hot_cold_lifetime: PASS (eviction during prepare preserves all projections' payloads)");
        return 0;
    } catch(const std::exception & e) { std::fprintf(stderr,"hot_cold_lifetime: FAIL: %s\n",e.what());return 1; }
}

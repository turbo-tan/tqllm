#include "prefill_observer_cuda.hpp"
#include <atomic>
#include <iostream>
__global__ void mark(int * value,int n) { *value=n; }
static void check(bool ok,const char * why) { if (!ok) throw std::runtime_error(why); }
int main() {
    try {
        cudaStream_t stream; check(cudaStreamCreate(&stream)==cudaSuccess,"stream");
        int * value=nullptr; check(cudaMallocManaged(&value,sizeof(int))==cudaSuccess,"allocation");
        *value=0; std::atomic<int> called{0};
        {
            tqllm::PrefillObservation observation([&](uint32_t done,uint32_t total,uint32_t cached) {
                if (*value==1 && done==256 && total==1024 && cached==128) ++called;
            });
            mark<<<1,1,0,stream>>>(value,1);
            tqllm::prefill_completed(stream,256,1024,128);
        }
        check(!tqllm::prefill_observer,"scope restored before callback completion");
        check(cudaStreamSynchronize(stream)==cudaSuccess,"stream synchronization");
        check(called==1,"callback must run after GPU work with captured observer");
        {
            tqllm::PrefillObservation observation([](auto,auto,auto){throw std::runtime_error("test callback");});
            tqllm::prefill_completed(stream,1,1);
        }
        mark<<<1,1,0,stream>>>(value,2);
        check(cudaStreamSynchronize(stream)==cudaSuccess && *value==2,"observer exception poisoned stream");
        cudaFree(value); cudaStreamDestroy(stream);
        std::cout<<"prefill_observer: PASS\n";
    } catch(const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}

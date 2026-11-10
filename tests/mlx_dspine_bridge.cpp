#include "dflash.hpp"
#include "dspine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace mx=mlx::core;
int main() {
    using namespace tqllm::mlx_backend;
    using namespace tqllm::spec;
    const std::vector<float> raw{1,1,1,1,2,1},read{1,0,0,1},write{1,0},book{1,0,0,1};
    const std::vector<float> packed{0,0,0,0,1,0},bias{0};
    const DSpineConfig cfg{2,2,1,1,3,16,1e-6f};
    const DSpineWeights weights{read.data(),write.data(),book.data(),packed.data(),packed.data()+2,packed.data()+4,bias.data()};
    const auto expected=dspine_write_reference(cfg,weights,raw,{0});
    int failed=0;
    for(const auto device:{mx::Device::cpu,mx::Device::gpu}) {
        mx::set_default_device(mx::Device(device));
        auto result=dspine_bridge(mx::array(raw.data(),{3,2}),mx::array(uint32_t(0)),mx::array(read.data(),{2,2}),
                                  mx::array(write.data(),{2,1}),mx::array(book.data(),{2,2}),
                                  mx::array(packed.data(),{3,2}),mx::array(bias.data(),{1}),cfg.epsilon);
        mx::eval(result.output,result.raw,result.rms,result.packed);
        const auto* actual=result.output.data<float>();
        float error=0;
        for(size_t i=0;i<expected.size();++i) {
            if(!std::isfinite(actual[i]))++failed;
            error=std::max(error,std::abs(actual[i]-expected[i]));
        }
        const bool ok=error<.016f && actual[0]==raw[0] && actual[1]==raw[1];
        std::printf("DSpine MLX %s bridge vs BF16 CPU reference: error %g %s\n",device==mx::Device::cpu?"CPU":"GPU",error,ok?"PASS":"FAIL");
        failed+=!ok;
    }
    return failed?1:0;
}

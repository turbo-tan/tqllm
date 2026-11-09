#include "dspine.hpp"
#include <cstdio>
#include <functional>

using namespace tqllm;
GgufFile fixture() {
    GgufFile f;
    for (const auto& [key,value] : std::vector<std::pair<std::string,double>>{
        {"dspine_version",1},{"draft_block_causal",1},{"embedding_length",32},{"block_count",2},
        {"block_size",4},{"selector_top_k",16},{"target_block_count",4},{"dspine_epsilon",1e-6}})
        f.kv["dflash."+key].num=value;
    f.kv["dflash.target_layers"].arr={1,3};
    auto tensor=[&](const std::string& name,std::vector<uint64_t> dims) {f.tensors[name].ne=std::move(dims);f.tensors[name].type=GGML_BF16;};
    tensor("dspine_codebook.weight",{8,32});tensor("dspine_gate.weight",{8,4});
    tensor("dspine_pred_gate.weight",{8,4});tensor("dspine_prev.weight",{8,4});tensor("dspine_gate.bias",{4});
    for(int i=0;i<2;++i) {const auto p="blk."+std::to_string(i)+".";tensor(p+"dspine_read.weight",{32,8});tensor(p+"dspine_write.weight",{4,32});}
    return f;
}
int main() {
    int fail=0;
    auto check=[&](GgufFile f,bool valid,const char* label) {
        bool accepted=true;try {(void)spec::dspine_config(f,32,32,4);}catch(const std::exception&) {accepted=false;}
        std::printf("%s %s\n",accepted==valid?"PASS":"FAIL",label);fail+=accepted!=valid;
    };
    check(fixture(),true,"valid DSpine bridge schema");
    auto f=fixture();f.tensors.erase("blk.1.dspine_write.weight");check(f,false,"missing last write rejected");
    f=fixture();f.tensors["dspine_gate.weight"].ne={4,8};check(f,false,"transposed gate rejected");
    f=fixture();f.tensors["dspine_codebook.weight"].ne={8,33};check(f,false,"wrong target vocabulary rejected");
    f=fixture();f.tensors["dspine_prev.weight"].type=GGML_Q4_K;check(f,false,"quantized bridge rejected");
    f=fixture();f.tensors["selector_hidden.weight"]={};check(f,false,"mixed DFlash2 selector rejected");
    f=fixture();f.kv["dflash.target_layers"].arr={1,1};check(f,false,"duplicate feature layer rejected");
    f=fixture();f.kv["dflash.dspine_epsilon"].num=0;check(f,false,"invalid epsilon rejected");
    f=fixture();f.kv["dflash.target_block_count"].num=36;check(f,false,"wrong target depth rejected");
    return fail?1:0;
}

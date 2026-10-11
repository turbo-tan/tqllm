#include "draft_head.hpp"
#include "prefill.hpp"
#include <stdexcept>
#include <cstdlib>
namespace tqllm::mlx_backend {
namespace mx=mlx::core;
DraftHead4 draft_head4(const Matrix& original) {
    auto staged=mx::astype(stage_bf16(original),mx::float32);
    auto weights=mx::quantize(staged,64,4,"affine");
    mx::eval(weights);
    return {weights[0],weights[1],weights[2]};
}
mx::array draft_head4_matmul(const DraftHead4& head,const mx::array& x) {
    if(x.ndim()!=2 || x.dtype()!=mx::float32 || x.shape(0)<1 || x.shape(1)!=8*head.codes.shape(1))
        throw std::invalid_argument("invalid proposal head input");
    const int m=x.shape(0);auto input=x;
    if(m>1 && m<32 && !std::getenv("TQLLM_MLX_DRAFT_HEAD_UNPAD"))
        input=mx::concatenate({input,mx::zeros({32-m,x.shape(1)},mx::float32)},0);
    auto result=mx::quantized_matmul(input,head.codes,head.scales,head.biases,true,64,4,"affine");
    return mx::astype(mx::slice(result,{0,0},{m,head.codes.shape(0)}),mx::float32);
}
}

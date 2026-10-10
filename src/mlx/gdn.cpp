#include "gdn.hpp"
#include <stdexcept>
#include <cstdlib>
#include <string_view>

namespace tqllm::mlx_backend {
namespace mx=mlx::core;
GdnHeadOrder gdn_head_order(int kh,int vh) {
    if(!((kh==16 && (vh==16 || vh==32 || vh==48 || vh==64)) ||
         (kh==24 && vh==24) || (kh==32 && vh==32)))
        throw std::invalid_argument("unsupported native MLX GDN head layout");
    const int repeats=vh/kh;
    std::vector<uint32_t> to(vh),from(vh);
    for(int i=0;i<vh;++i) {
        to[i]=uint32_t((i%repeats)*kh+i/repeats);
        from[i]=uint32_t((i%kh)*repeats+i/kh);
    }
    return {mx::array(to.data(),{vh},mx::uint32),mx::array(from.data(),{vh},mx::uint32),kh,vh};
}
std::vector<mx::array> gdn_native(const mx::array& q,const mx::array& k,const mx::array& v,
    const mx::array& decay,const mx::array& beta,const std::optional<mx::array>& state,const GdnHeadOrder& order) {
    const char* chunk=std::getenv("GATED_DELTA_CHUNK");
    if(!chunk || std::string_view(chunk)!="1")
        throw std::invalid_argument("native GDN requires GATED_DELTA_CHUNK=1; chunked precision is unvalidated");
    if(q.ndim()!=4 || k.shape()!=q.shape() || q.shape(3)!=128 || v.ndim()!=4 || v.shape(3)!=128 ||
       q.shape(0)!=v.shape(0) || q.shape(1)!=v.shape(1) || q.shape(2)!=order.key_heads ||
       v.shape(2)!=order.value_heads || q.dtype()!=mx::float32 || k.dtype()!=mx::float32 || v.dtype()!=mx::float32 ||
       decay.shape()!=mx::Shape{v.shape(0),v.shape(1),v.shape(2)} || beta.shape()!=decay.shape() ||
       decay.dtype()!=mx::float32 || beta.dtype()!=mx::float32 || order.to_native.size()!=size_t(v.shape(2)) ||
       (state && (state->shape()!=mx::Shape{v.shape(0),v.shape(2),128,128} || state->dtype()!=mx::float32)))
        throw std::invalid_argument("unsupported native MLX GDN dimensions");
    // Native kernels require dense storage after the head permutation.
    auto update=mx::fast::gated_delta_update(mx::contiguous(q),mx::contiguous(k),
        mx::contiguous(mx::take(v,order.to_native,2)),mx::contiguous(mx::take(decay,order.to_native,2)),
        mx::contiguous(mx::take(beta,order.to_native,2)),state);
    return {mx::contiguous(mx::take(update[0],order.from_native,2)),update[1]};
}
}

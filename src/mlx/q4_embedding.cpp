#include "q4_embedding.hpp"
#include <stdexcept>
#include <limits>
namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Q4Embedding load_q4_embedding(std::span<const uint8_t> blocks, int rows, int columns) {
    if (rows<=0 || columns<=0 || columns%256 ||
        blocks.size()!=uint64_t(rows)*uint64_t(columns/256)*144)
        throw std::invalid_argument("invalid Q4_K embedding payload or dimensions");
    return {mx::array(blocks.data(),{rows,columns/256,144},mx::uint8)};
}
mx::array q4_embedding(const Q4Embedding& w, std::span<const uint32_t> ids) {
    if(w.blocks.ndim()!=3 || w.blocks.shape(2)!=144 || w.blocks.dtype()!=mx::uint8 ||
       w.blocks.shape(0)<=0 || w.blocks.shape(1)<=0 || ids.empty())
        throw std::invalid_argument("invalid Q4_K embedding inputs");
    for(uint32_t id:ids) if(id>=uint32_t(w.blocks.shape(0)))
        throw std::out_of_range("embedding token exceeds vocabulary");
    if(w.blocks.shape(1)>std::numeric_limits<int>::max()/256 ||
       ids.size()>size_t(std::numeric_limits<int>::max()/(w.blocks.shape(1)*256)))
        throw std::invalid_argument("embedding output exceeds Metal grid range");
    const int k=w.blocks.shape(1)*256, m=int(ids.size());
    auto tokens=mx::array(ids.data(),{m},mx::uint32);
    static auto kernel=mx::fast::metal_kernel("tqllm_q4_embedding",{"blocks","tokens"},{"y"},R"METAL(
uint i=thread_position_in_grid.x;
if(i>=M*K) return;
uint row=tokens[i/K],j=(i%K)%256,b=(i%K)/256,sub=j/32;
const device uchar* p=blocks+(row*(K/256)+b)*144;
float d=float(as_type<half>(ushort(uint(p[0])|(uint(p[1])<<8))));
float mn=float(as_type<half>(ushort(uint(p[2])|(uint(p[3])<<8))));
uint s=sub<4?(p[4+sub]&63u):((p[4+sub+4]&15u)|((p[4+sub-4]>>6)<<4));
uint sm=sub<4?(p[4+sub+4]&63u):((p[4+sub+4]>>4)|((p[4+sub]>>6)<<4));
uint q=(p[16+(sub/2)*32+j%32]>>(4*(sub%2)))&15u;
y[i]=(d*float(s))*float(q)-mn*float(sm);
)METAL");
    return kernel({w.blocks,tokens},{{m,k}},{mx::float32},{m*k,1,1},{256,1,1},
                  {{"M",m},{"K",k}},std::nullopt,false,mx::Device(mx::Device::gpu))[0];
}
}

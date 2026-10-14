#include "q6.hpp"
#include <stdexcept>

namespace tqllm::mlx_backend {
namespace mx = mlx::core;
Q6AffineSplit q6_affine_split(const Q6Weights& w, int bits) {
    if (bits != 6 && bits != 8) throw std::invalid_argument("Q6 affine cache bits must be 6 or 8");
    if (w.blocks.ndim() != 3 || w.blocks.shape(2) != 210 || w.blocks.dtype() != mx::uint8 ||
        w.blocks.shape(0) <= 0 || w.blocks.shape(1) <= 0)
        throw std::invalid_argument("invalid Q6 affine expansion");
    const int r = w.blocks.shape(0), k = w.blocks.shape(1) * 256;
    static auto expand = mx::fast::metal_kernel("tqllm_q6_affine_split", {"blocks"}, {"codes", "scales", "biases"}, R"METAL(
uint i=thread_position_in_grid.x;
if(i>=R*K)return;
uint row=i/K,col=i%K,g=col/32u,j=col%32u;
uint src=g*32u+uint(PART)*16u+j%16u;
const device uchar* p=blocks+(row*(K/256u)+src/256u)*210u;
uint within=src%256u,h=within/128u,l=within%32u,c=(within%128u)/32u;
int q=0;
if(j<16u) {
 uint lo=p[h*64u+(c&1u)*32u+l],hi=p[128u+h*32u+l];
 q=int(((c<2u?lo&15u:lo>>4u)|(((hi>>(2u*c))&3u)<<4u)))-32;
}
codes[i]=uchar(q+128);
if(j==0u) {
 float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8u))));
 float s=d*float(as_type<char>(p[192u+h*8u+c*2u+l/16u]));
 scales[row*(K/32u)+g]=s;biases[row*(K/32u)+g]=-128.0f*s;
}
)METAL");
    static auto pack6 = mx::fast::metal_kernel("tqllm_q6_pack6", {"codes"}, {"packed"}, R"METAL(
uint i=thread_position_in_grid.x;
if(i>=N/4u)return;
uint word=0u;
for(uint j=0u;j<4u;++j)word|=(uint(codes[4u*i+j])-96u)<<(6u*j);
packed[3u*i]=uchar(word);
packed[3u*i+1u]=uchar(word>>8u);
packed[3u*i+2u]=uchar(word>>16u);
)METAL");
    Q6AffineSplit result; result.bits = bits;
    for (int part = 0; part < 2; ++part) {
        auto a = expand({w.blocks}, {{r, k}, {r, k / 32}, {r, k / 32}},
            {mx::uint8, mx::float32, mx::float32}, {r * k, 1, 1}, {256, 1, 1},
            {{"R", r}, {"K", k}, {"PART", part}}, std::nullopt, false, mx::Device(mx::Device::gpu));
        auto codes = a[0];
        if (bits == 6) {
            codes = pack6({codes}, {{r, k * 3 / 4}}, {mx::uint8}, {r * k / 4, 1, 1},
                {256, 1, 1}, {{"N", r * k}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
        }
        result.codes.push_back(mx::reshape(mx::view(codes, mx::uint32), {r, k / 32 * bits}));
        result.scales.push_back(a[1]);
        result.biases.push_back(bits == 6 ? mx::multiply(a[1], mx::array(-32.0f)) : a[2]);
        mx::eval(result.codes.back(), result.scales.back(), result.biases.back());
    }
    return result;
}

mx::array q6_affine_matmul(const Q6AffineSplit& w, const mx::array& x) {
    if ((w.bits != 6 && w.bits != 8) || w.codes.size() != 2 || w.scales.size() != 2 || w.biases.size() != 2 ||
        x.ndim() != 2 || x.dtype() != mx::float32 || x.shape(0) <= 0 ||
        x.shape(1) <= 0 || x.shape(1) % 256 || w.codes[0].ndim() != 2)
        throw std::invalid_argument("invalid Q6 affine projection");
    const int m = x.shape(0), k = x.shape(1), r = w.codes[0].shape(0);
    auto pairs = mx::reshape(x, {m, k / 32, 2, 16});
    std::vector<mx::array> products;
    for (int part = 0; part < 2; ++part) {
        if (w.codes[part].shape() != mx::Shape{r, k / 32 * w.bits} || w.codes[part].dtype() != mx::uint32 ||
            w.scales[part].shape() != mx::Shape{r, k / 32} || w.scales[part].dtype() != mx::float32 ||
            w.biases[part].shape() != mx::Shape{r, k / 32} || w.biases[part].dtype() != mx::float32)
            throw std::invalid_argument("Q6 affine weight shape mismatch");
        auto columns = mx::reshape(mx::slice(pairs, {0, 0, part, 0}, {m, k / 32, part + 1, 16}), {m, k / 32, 16});
        auto padded = mx::reshape(mx::concatenate({columns, mx::zeros({m, k / 32, 16}, mx::float32)}, 2), {m, k});
        products.push_back(mx::quantized_matmul(padded, w.codes[part], w.scales[part], w.biases[part], true, 32, w.bits, "affine"));
    }
    return mx::add(products[0], products[1]);
}

Q6Weights load_q6_blocks(std::span<const uint8_t> blocks, int rows, int columns) {
    if (rows <= 0 || columns <= 0 || columns % 256 ||
        blocks.size() != uint64_t(rows) * uint64_t(columns / 256) * 210)
        throw std::invalid_argument("invalid Q6_K block payload or dimensions");
    return {mx::array(blocks.data(), {rows, columns / 256, 210}, mx::uint8)};
}

mx::array q6_matmul(const Q6Weights& w, const mx::array& x) {
    if (w.blocks.ndim() != 3 || w.blocks.shape(2) != 210 || w.blocks.dtype() != mx::uint8 ||
        x.ndim() != 2 || x.dtype() != mx::float32)
        throw std::invalid_argument("invalid Q6_K matrix types");
    const int rows = w.blocks.shape(0), k = w.blocks.shape(1) * 256, m = x.shape(0);
    if (rows <= 0 || k <= 0 || m <= 0 || x.shape(1) != k)
        throw std::invalid_argument("incompatible Q6_K matrix dimensions");
    static auto kernel = mx::fast::metal_kernel("tqllm_q6_matmul", {"blocks", "x"}, {"y"}, R"METAL(
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;
uint col = threadgroup_position_in_grid.z * MCOLS;
if(row>=R) return;
float acc[MCOLS];
for(uint m=0;m<MCOLS;++m)acc[m]=0.0f;
for(uint b=lane;b<K/256;b+=32) {
 const device uchar* p=blocks+(row*(K/256)+b)*210;
 float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8))));
 float part[MCOLS];
 for(uint m=0;m<MCOLS;++m)part[m]=0.0f;
 for(uint j=0;j<256;++j) {
  uint h=j/128,l=j%32,c=(j%128)/32;
  uint lo=p[h*64+(c&1u)*32+l],hi=p[128+h*32+l];
  int q=int(((c<2?lo&15u:lo>>4)|(((hi>>(2*c))&3u)<<4)))-32;
  float s=d*float(as_type<char>(p[192+h*8+c*2+l/16]));
  float value=float(q)*s;
  for(uint m=0;m<MCOLS;++m) {
   if(col+m>=M)break;
   part[m]+=value*x[(col+m)*K+b*256+j];
  }
 }
 for(uint m=0;m<MCOLS;++m)acc[m]+=part[m];
}
for(uint m=0;m<MCOLS;++m) {
 if(col+m>=M) break;
 float value=simd_sum(acc[m]);
 if(lane==0) y[(col+m)*R+row]=value;
}
)METAL");
    const int cols=m<=1?1:m<=2?2:4;
    return kernel({w.blocks, x}, {{m, rows}}, {mx::float32},
                  {32, (rows + 1) / 2 * 2, (m + cols - 1) / cols}, {32, 2, 1},
                  {{"R", rows}, {"K", k}, {"M", m}, {"MCOLS", cols}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

mx::array q6_rows_matmul(const Q6Weights& w, const mx::array& x, const mx::array& ids) {
    if (w.blocks.ndim() != 3 || w.blocks.shape(2) != 210 || w.blocks.dtype() != mx::uint8 ||
        x.ndim() != 2 || x.dtype() != mx::float32 || ids.ndim() != 2 || ids.dtype() != mx::uint32)
        throw std::invalid_argument("invalid Q6_K row projection types");
    const int rows = w.blocks.shape(0), k = w.blocks.shape(1) * 256, m = x.shape(0), n = ids.shape(1);
    if (rows <= 0 || k <= 0 || m <= 0 || n <= 0 || x.shape(1) != k || ids.shape(0) != m)
        throw std::invalid_argument("incompatible Q6_K row projection dimensions");
    static auto kernel = mx::fast::metal_kernel("tqllm_q6_rows_matmul", {"blocks", "x", "ids"}, {"y"}, R"METAL(
uint lane = thread_position_in_threadgroup.x;
uint item = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;
if(item>=M*N) return;
uint col = item / N, idx = item % N, row = ids[item];
if(row>=R) {
 if(lane==0) y[item]=-3.4028234663852886e38f;
 return;
}
float acc = 0.0f;
for(uint b=lane;b<K/256;b+=32) {
 const device uchar* p=blocks+(row*(K/256)+b)*210;
 float d=float(as_type<half>(ushort(uint(p[208])|(uint(p[209])<<8))));
 float part=0.0f;
 for(uint j=0;j<256;++j) {
  uint h=j/128,l=j%32,c=(j%128)/32;
  uint lo=p[h*64+(c&1u)*32+l],hi=p[128+h*32+l];
  int q=int(((c<2?lo&15u:lo>>4)|(((hi>>(2*c))&3u)<<4)))-32;
  float s=d*float(as_type<char>(p[192+h*8+c*2+l/16]));
  part+=float(q)*s*x[col*K+b*256+j];
 }
 acc+=part;
}
float value=simd_sum(acc);
if(lane==0) y[item]=value;
)METAL");
    return kernel({w.blocks, x, ids}, {{m, n}}, {mx::float32},
                  {32, (m * n + 1) / 2 * 2, 1}, {32, 2, 1},
                  {{"R", rows}, {"K", k}, {"M", m}, {"N", n}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}


mx::array shard_topk_ids(const mx::array& logits, int candidates) {
    if (logits.ndim() != 2 || logits.dtype() != mx::float32 || logits.shape(0) <= 0 || logits.shape(1) <= 0 ||
        candidates < 1 || candidates > logits.shape(1))
        throw std::invalid_argument("invalid shard top-k logits");
    const int m = logits.shape(0), v = logits.shape(1);
    const int per = candidates == 1 ? 1 : 2;
    const int shards = (candidates + per - 1) / per;
    static auto kernel = mx::fast::metal_kernel("tqllm_shard_topk_ids", {"logits"}, {"ids"}, R"METAL(
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint shard = threadgroup_position_in_grid.z;
uint span = (uint(V) + uint(SHARDS) - 1u) / uint(SHARDS);
uint begin = shard * span;
uint end = min(begin + span, uint(V));
float best0 = -3.4028234663852886e38f, best1 = -3.4028234663852886e38f;
uint id0 = uint(V), id1 = uint(V);
for(uint i = begin + lane; i < end; i += 256u) {
    float x = logits[row * uint(V) + i];
    if(x > best0 || (x == best0 && i < id0)) {
        best1 = best0; id1 = id0; best0 = x; id0 = i;
    } else if(x > best1 || (x == best1 && i < id1)) {
        best1 = x; id1 = i;
    }
}
threadgroup float scores0[256], scores1[256];
threadgroup uint ids0[256], ids1[256];
scores0[lane] = best0; scores1[lane] = best1; ids0[lane] = id0; ids1[lane] = id1;
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint step = 128u; step > 0u; step >>= 1u) {
    if(lane < step) {
        float other0 = scores0[lane + step], other1 = scores1[lane + step];
        uint oid0 = ids0[lane + step], oid1 = ids1[lane + step];
        float a0 = scores0[lane], a1 = scores1[lane];
        uint aid0 = ids0[lane], aid1 = ids1[lane];
        if(other0 > a0 || (other0 == a0 && oid0 < aid0)) {
            a1 = a0; aid1 = aid0; a0 = other0; aid0 = oid0;
        } else if(other0 > a1 || (other0 == a1 && oid0 < aid1)) {
            a1 = other0; aid1 = oid0;
        }
        if(other1 > a0 || (other1 == a0 && oid1 < aid0)) {
            a1 = a0; aid1 = aid0; a0 = other1; aid0 = oid1;
        } else if(other1 > a1 || (other1 == a1 && oid1 < aid1)) {
            a1 = other1; aid1 = oid1;
        }
        scores0[lane] = a0; scores1[lane] = a1; ids0[lane] = aid0; ids1[lane] = aid1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(lane == 0u) {
    uint base = row * uint(SHARDS * PER) + shard * uint(PER);
    ids[base] = ids0[0];
    if(PER == 2) ids[base + 1u] = ids1[0];
}
)METAL");
    return kernel({logits}, {{m, shards * per}}, {mx::uint32}, {256, m, shards}, {256, 1, 1},
                  {{"V", v}, {"SHARDS", shards}, {"PER", per}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}

mx::array sparse_logits(const mx::array& ids, const mx::array& scores, int vocab) {
    if (ids.ndim() != 2 || ids.dtype() != mx::uint32 || scores.ndim() != 2 || scores.dtype() != mx::float32 ||
        ids.shape() != scores.shape() || vocab <= 0)
        throw std::invalid_argument("invalid sparse logits inputs");
    const int m = ids.shape(0), n = ids.shape(1);
    static auto scatter = mx::fast::metal_kernel("tqllm_sparse_logits_scatter", {"ids", "scores"}, {"logits"}, R"METAL(
uint i=thread_position_in_grid.x;
if(i>=M*V)return;
uint row=i/V,tok=i%V;
float value=-3.4028234663852886e38f;
for(uint j=0;j<N;++j) {
 if(ids[row*N+j]==tok) {
  value=scores[row*N+j];
  break;
 }
}
logits[i]=value;
)METAL");
    return scatter({ids, scores}, {{m, vocab}}, {mx::float32}, {uint32_t(m * vocab), 1, 1}, {256, 1, 1},
                   {{"M", m}, {"N", n}, {"V", vocab}}, std::nullopt, false, mx::Device(mx::Device::gpu))[0];
}
}

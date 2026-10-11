#include "tensorops.hpp"
#include "int8_tiles.hpp"
#include "weights.hpp"
#include "prefill.hpp"
#include "verify.hpp"
#include "../src/dequant.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <optional>
#include <cstdlib>

namespace mx = mlx::core;
using namespace tqllm;
using namespace tqllm::mlx_backend;
static uint32_t word(const uint8_t* p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
static void dequant(uint32_t type, const std::vector<uint8_t>& bytes, std::vector<float>& w) {
    if(type!=GGML_TQ3 && type!=GGML_TQ4 && type!=GGML_Q8_0) {
        dequantize_host(type, bytes.data(),w.data(),int64_t(w.size()));return;
    }
    if(type==GGML_Q8_0) {
        for(size_t b=0;b<w.size()/32;++b) {
            const auto* p=bytes.data()+b*34;
            const float d=half_to_float(uint16_t(p[0])|(uint16_t(p[1])<<8));
            for(int j=0;j<32;++j) w[b*32+j]=float(std::bit_cast<int8_t>(p[2+j]))*d;
        }
        return;
    }
    constexpr int lv4[]={6,18,31,44,58,75,96,127,-6,-18,-31,-44,-58,-75,-96,-127};
    constexpr int lv3[]={14,45,79,127,-14,-45,-79,-127};
    for(size_t b=0;b<w.size()/128;++b) {
        const auto* p=bytes.data()+b*(type==GGML_TQ3?50:66);
        const float d=half_to_float(uint16_t(p[0])|(uint16_t(p[1])<<8));
        for(int j=0;j<128;++j) {
            if(type==GGML_TQ4) w[b*128+j]=d*lv4[(p[2+j/2]>>(4*(j%2)))&15];
            else {
                const auto* group=p+2+(j/32)*12; const int t=j%32;
                const uint32_t ranks=word(group+(t/16)*4),signs=word(group+8);
                const int rank=(ranks>>(4*(t%8)+2*((t%16)/8)))&3;
                const int sign=(signs>>(4*(t%8)+t/8))&1;
                w[b*128+j]=d*lv3[rank+4*sign];
            }
        }
    }
}
static std::vector<float> rotate(std::vector<float> x) {
    constexpr uint32_t signs[]={0xE46A0359,0x42F85949,0xA4C63BBB,0x49F250A9};
    for(size_t b=0;b<x.size();b+=128) {
        for(int j=0;j<128;++j) x[b+j]*=(signs[j/32]>>(j%32))&1?-1.0f:1.0f;
        for(int width=1;width<128;width*=2)
            for(int j=0;j<128;j+=2*width) for(int a=0;a<width;++a) {
                const float lo=x[b+j+a],hi=x[b+j+a+width];
                x[b+j+a]=lo+hi;x[b+j+a+width]=lo-hi;
            }
        for(int j=0;j<128;++j) x[b+j]/=std::sqrt(128.0f);
    }
    return x;
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) {std::cerr<<"usage: test-mlx-gguf model.gguf [--prefill]\n";return 2;}
    const bool affine_half=argc==3 && std::string(argv[2])=="--affine-half";
    const bool affine_tf32=argc==3 && std::string(argv[2])=="--affine-tf32";
    const bool affine_gu=argc==3 && std::string(argv[2])=="--affine-gu";
    const bool expanded_half=argc==3 && std::string(argv[2])=="--expanded-half";
    const bool affine8=argc==3 && std::string(argv[2])=="--affine8";
    const bool expanded=argc==3 && std::string(argv[2])=="--expanded";
    const bool prefill=argc==3 && std::string(argv[2])=="--prefill";
    const bool tensorops_half=argc==3 && std::string(argv[2])=="--tensorops-half";
    const bool tensorops=argc==3 && std::string(argv[2])=="--tensorops";
    const bool int8_tiles=argc==3 && std::string(argv[2])=="--int8-tiles";
    const bool q6_affine=argc==3 && std::string(argv[2])=="--q6-affine";
    const bool affine_tq4=argc==3 && std::string(argv[2])=="--affine-tq4";
    const bool verify_tiles=argc==3 && std::string(argv[2])=="--verify-tiles";
    const bool vector_load=argc==3 && std::string(argv[2])=="--vector-load";
    if(argc==3 && !prefill && !verify_tiles && !vector_load && !tensorops && !tensorops_half && !expanded && !affine8 && !expanded_half && !affine_half && !affine_gu && !affine_tf32 && !int8_tiles && !q6_affine && !affine_tq4) return 2;
    if(affine_tf32 && (!std::getenv("MLX_ENABLE_TF32") || std::string(std::getenv("MLX_ENABLE_TF32"))!="1" ||
       std::getenv("TQLLM_MLX_AFFINE_HALF")))
        throw std::invalid_argument("TF32 screen requires MLX_ENABLE_TF32=1 and FP32 affine inputs");
    auto file=gguf_open(argv[1]); std::set<uint32_t> covered;
    for(const auto& [name,original]:file.tensors) {
        if(original.ne.size()!=2 || covered.count(original.type)) continue;
        if(original.type!=GGML_Q4_K && original.type!=GGML_F32 && original.type!=GGML_BF16 &&
           original.type!=GGML_TQ3 && original.type!=GGML_TQ4 && original.type!=GGML_Q6_K && original.type!=GGML_Q8_0)
            throw std::runtime_error("unsupported model tensor type");
        auto t=original;
        const int k=int(t.ne[0]);int r=int(std::min<uint64_t>(17,t.ne[1]));
        t.bytes=t.bytes/t.ne[1]*uint64_t(r);t.ne[1]=uint64_t(r);
        auto data=file.read(t);
        std::vector<uint8_t> payload(data.begin(),data.end());
        std::vector<float> w(size_t(r)*k);dequant(t.type,payload,w);
        if(t.type==GGML_Q4_K) {
            auto embedding=load_q4_embedding(payload,r,k);
            std::vector<uint32_t> ids={0,uint32_t(r-1),uint32_t(r/2),0};
            auto y=q4_embedding(embedding,ids);mx::eval(y);
            float error=0;
            for(size_t a=0;a<ids.size();++a) for(int j=0;j<k;++j)
                error=std::max(error,std::abs(y.data<float>()[a*k+j]-w[ids[a]*k+j]));
            if(error>1e-5) throw std::runtime_error("real GGUF embedding mismatch");
            std::cout<<"GGUF_EMBED_PASS type="<<t.type<<" max_abs="<<error<<" tensor="<<name<<'\n';
        } else {
            auto matrix=load_matrix(t,payload);
            std::optional<Q6AffineSplit> q6_split;
            if (q6_affine && t.type == GGML_Q6_K) q6_split = q6_affine_split(std::get<Q6Weights>(matrix));
            std::optional<Tq3Affine8> affine;
            if (affine_tq4 && t.type == GGML_TQ4) {
                affine = tq4_affine8(std::get<Tq4Weights>(matrix));
                auto decoded = mx::dequantize(affine->codes, affine->scales, affine->biases, 128, 8, "affine");
                mx::eval(decoded);
                for (size_t j = 0; j < w.size(); ++j)
                    if (decoded.data<float>()[j] != w[j]) throw std::runtime_error("affine8 changed TQ4 weight");
                std::cout << "GGUF_TQ4_AFFINE_WEIGHT_EXACT tensor=" << name << '\n';
            }
            if((affine8 || affine_half || affine_gu || affine_tf32) && t.type==GGML_TQ3) {
                affine=tq3_affine8(std::get<Tq3Weights>(matrix));
                auto decoded=mx::dequantize(affine->codes,affine->scales,affine->biases,128,8,"affine");
                mx::eval(decoded);
                for(size_t i=0;i<w.size();++i)
                    if(decoded.data<float>()[i]!=w[i])throw std::runtime_error("affine8 changed TQ3 weight");
                std::cout<<"GGUF_AFFINE8_WEIGHT_EXACT tensor="<<name<<'\n';
                if(affine_gu) {
                    auto twice=[](const mx::array& a){return mx::multiply(a,mx::array(2.0f,a.dtype()));};
                    Tq3Affine8 up{affine->codes,twice(affine->scales),twice(affine->biases),
                        twice(affine->half_scales),twice(affine->half_biases)};
                    affine=tq3_affine8_join(*affine,up);
                    auto copy=w;
                    for(float value:copy)w.push_back(value*2.0f);
                    r*=2;
                    std::cout<<"GGUF_AFFINE8_JOIN rows="<<r<<'\n';
                }
            }
            std::optional<Tq3Expanded> decoded;
            if((expanded || expanded_half) && t.type==GGML_TQ3)decoded=tq3_expand(std::get<Tq3Weights>(matrix));
            std::optional<Matrix> planar;
            if(vector_load && t.type==GGML_TQ3) {planar=matrix;matrix=tq3_interleave(std::get<Tq3Weights>(matrix));}
            if(prefill) {
                auto staged=stage_bf16(matrix);
                if(staged.dtype()!=mx::bfloat16)throw std::runtime_error("prefill staging dtype mismatch");
                auto staged_fp32=mx::astype(staged,mx::float32);mx::eval(staged_fp32);
                for(size_t i=0;i<w.size();++i) {
                    const uint32_t bits=std::bit_cast<uint32_t>(w[i]);
                    const float rounded=std::bit_cast<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
                    if(staged_fp32.data<float>()[i]!=rounded)
                        throw std::runtime_error("prefill BF16 weights differ from CPU rounding: "+name);
                }
                std::cout<<"GGUF_STAGE_PASS type="<<t.type<<" tensor="<<name<<'\n';
            }
            for(int m:{1,2,3,5,8,16,64}) {
                const bool native_half=affine_half && t.type==GGML_TQ3 && m<=8;
                std::vector<float> input(size_t(m)*k);
                for(size_t i=0;i<input.size();++i) input[i]=std::sin(float(i)*0.037f)*0.1f;
                auto refinput=(t.type==GGML_TQ3||t.type==GGML_TQ4)?rotate(input):input;
                if (int8_tiles && t.type == GGML_TQ3) {
                    auto y = verify_tq3_int8(std::get<Tq3Weights>(matrix), mx::array(input.data(), {m, k}));
                    mx::eval(y);
                    double diff = 0, total = 0;
                    float maxerr = 0;
                    for (int a = 0; a < m; ++a) for (int row = 0; row < r; ++row) {
                        double expected = 0;
                        for (int j = 0; j < k; ++j) expected += double(w[row*k+j]) * refinput[a*k+j];
                        const double delta = y.data<float>()[a*r+row] - expected;
                        diff += delta*delta; total += expected*expected;
                        maxerr = std::max(maxerr, float(std::abs(delta)));
                    }
                    const double rel = std::sqrt(diff / std::max(total, 1e-20));
                    if (!std::isfinite(rel) || (rel > 2e-5 && maxerr > 1e-5))
                        throw std::runtime_error("residual INT8 real GGUF projection failed");
                    std::cout << "GGUF_INT8_TILE_PASS m=" << m << " rel_l2=" << rel << " max_abs=" << maxerr << '\n';
                }
                auto y=q6_split?q6_affine_matmul(*q6_split,mx::array(input.data(),{m,k})):affine?tq3_affine8_matmul(*affine,tq4_rotate(mx::array(input.data(),{m,k}))):decoded && expanded_half?verify_expanded_tensorops(*decoded,mx::array(input.data(),{m,k})):decoded?tq3_expanded_matmul(*decoded,tq4_rotate(mx::array(input.data(),{m,k}))):(tensorops || tensorops_half)?verify_tensorops(matrix,mx::array(input.data(),{m,k}),tensorops_half):(verify_tiles && m>1)?verify_tiled(matrix,mx::array(input.data(),{m,k})):linear(matrix,mx::array(input.data(),{m,k}));mx::eval(y);
                if (std::getenv("TQLLM_MLX_COLUMN_AUDIT") && !affine_gu) {
                    size_t changed = 0, bf16_changed = 0;
                    float maximum = 0;
                    for (int a = 0; a < m; ++a) {
                        auto single = linear(matrix, mx::array(input.data()+size_t(a)*k, {1,k}));
                        mx::eval(single);
                        for (int row = 0; row < r; ++row) {
                            const float batched = y.data<float>()[a*r+row], sequential = single.data<float>()[row];
                            const uint32_t b = std::bit_cast<uint32_t>(batched), q = std::bit_cast<uint32_t>(sequential);
                            auto rounded = [](uint32_t v) { return (v+0x7fffu+((v>>16)&1u))&0xffff0000u; };
                            changed += b != q;
                            bf16_changed += rounded(b) != rounded(q);
                            maximum = std::max(maximum, std::abs(batched-sequential));
                        }
                    }
                    std::cout << "GGUF_COLUMN_AUDIT type=" << t.type << " m=" << m
                              << " changed=" << changed << " bf16_changed=" << bf16_changed
                              << " total=" << size_t(m)*r << " max_abs=" << maximum << " tensor=" << name << '\n';
                }
                if(std::getenv("TQLLM_MLX_TQ3_STREAM") && t.type==GGML_TQ3) {
                    unsetenv("TQLLM_MLX_TQ3_STREAM");
                    auto base=linear(matrix,mx::array(input.data(),{m,k}));mx::eval(base);
                    setenv("TQLLM_MLX_TQ3_STREAM","1",1);
                    for(size_t i=0;i<y.size();++i)
                        if(std::bit_cast<uint32_t>(y.data<float>()[i])!=std::bit_cast<uint32_t>(base.data<float>()[i]))
                            throw std::runtime_error("streaming TQ3 changed output bits");
                    std::cout<<"GGUF_TQ3_STREAM_BITEXACT m="<<m<<'\n';
                }
                if(planar) {
                    auto base=linear(*planar,mx::array(input.data(),{m,k}));mx::eval(base);
                    for(size_t i=0;i<y.size();++i)
                        if(std::bit_cast<uint32_t>(y.data<float>()[i])!=std::bit_cast<uint32_t>(base.data<float>()[i]))
                            throw std::runtime_error("interleaved TQ3 changed planar output bits");
                    std::cout<<"GGUF_TQ3_LAYOUT_BITEXACT m="<<m<<'\n';
                }
                double diff=0,total=0,original_diff=0,original_total=0;float maxerr=0;
                for(int a=0;a<m;++a) for(int row=0;row<r;++row) {
                    double expected=0,original_expected=0;
                    const bool rounded=(tensorops_half && (t.type==GGML_TQ3 || t.type==GGML_TQ4 || t.type==GGML_Q6_K)) || (expanded_half && t.type==GGML_TQ3);
                    for(int j=0;j<k;++j) {
                        const float weight=w[row*k+j],value=refinput[a*k+j];
                        original_expected+=double(weight)*value;
                        // Native QMV dequantises in FP32 even with FP16 inputs.
                        if(native_half)
                            expected+=double(weight)*float((_Float16)value);
                        else expected+=rounded?double(float((_Float16)weight))*float((_Float16)value):double(weight)*value;
                    }
                    if(native_half)expected=float((_Float16)expected);
                    const double original_delta=y.data<float>()[a*r+row]-original_expected;
                    original_diff+=original_delta*original_delta;original_total+=original_expected*original_expected;
                    const double delta=y.data<float>()[a*r+row]-expected;
                    diff+=delta*delta;total+=expected*expected;maxerr=std::max(maxerr,float(std::abs(delta)));
                }
                const double rel=std::sqrt(diff/std::max(total,1e-20));
                if(tensorops_half || expanded_half || affine_half || affine_tf32) {
                    const double drift=std::sqrt(original_diff/std::max(original_total,1e-20));
                    std::cout<<(affine_tf32?"GGUF_TF32_DRIFT type=":"GGUF_FP16_DRIFT type=")<<t.type<<" m="<<m<<" rel_l2="<<drift<<'\n';
                    if(drift>1e-3)throw std::runtime_error("reduced precision projection drift exceeds screen");
                }
                const bool approximate=native_half || affine_tf32;
                if(rel>(approximate?1e-3:2e-5) && maxerr>1e-5) {
                    std::cerr<<"GGUF_LINEAR_FAIL type="<<t.type<<" m="<<m<<" rel_l2="<<rel<<" max_abs="<<maxerr<<'\n';
                    throw std::runtime_error("real GGUF linear mismatch: "+name);
                }
                std::cout<<(affine_tf32?"GGUF_TF32_SCREEN_PASS type=":approximate?"GGUF_FP16_SCREEN_PASS type=":"GGUF_LINEAR_PASS type=")<<t.type<<" m="<<m<<" rel_l2="<<rel<<" max_abs="<<maxerr<<" tensor="<<name<<'\n';
            }
        }
        covered.insert(t.type);
    }
    std::set<uint32_t> required;
    for(const auto& [name,t]:file.tensors) if(t.ne.size()==2) required.insert(t.type);
    if(covered!=required) throw std::runtime_error("incomplete GGUF matrix type coverage");
    std::cout<<"GGUF_TYPES_PASS "<<covered.size()<<'\n';
}

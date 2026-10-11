#include "tq4.hpp"
#include "tq3.hpp"
#include "q6.hpp"
#include "q8.hpp"
#include "q4_embedding.hpp"
#include "weights.hpp"
#include "tensorops.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <bit>
#include <cstdlib>
#include <optional>

static void test_tensor_limits() {
    using namespace tqllm::mlx_backend;
    const std::vector<std::string> names={"TQLLM_MLX_TENSOR_MEMORY","TQLLM_MLX_TENSOR_N",
        "TQLLM_MLX_TENSOR_K","TQLLM_MLX_TENSOR_SIMDS","TQLLM_MLX_TENSOR_SPLIT","TQLLM_MLX_TENSOR_M"};
    std::vector<std::optional<std::string>> saved;
    for(const auto& name:names) {
        const char* value=std::getenv(name.c_str());
        saved.emplace_back(value?std::optional<std::string>(value):std::nullopt);
        unsetenv(name.c_str());
    }
    auto input=mlx::core::zeros({1,256},mlx::core::float32);
    Matrix q4=Tq4Weights{mlx::core::zeros({1,128},mlx::core::uint8),
        mlx::core::zeros({1,2},mlx::core::float16),true};
    for(int width:{64,128}) {
        setenv("TQLLM_MLX_TENSOR_K",std::to_string(width).c_str(),1);
        bool rejected=false;
        try {verify_tensorops(q4,input,true);}
        catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected)throw std::runtime_error("unsupported cooperative reduction tile accepted");
    }
    setenv("TQLLM_MLX_TENSOR_MEMORY","1",1);
    setenv("TQLLM_MLX_TENSOR_N","64",1);
    Matrix q3=Tq3Weights{mlx::core::zeros({1,3,8},mlx::core::uint32),
        mlx::core::zeros({1,2},mlx::core::float16),false};
    bool rejected=false;
    try {verify_tensorops(q3,input,false);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected)throw std::runtime_error("oversized threadgroup scratch accepted");
    for(size_t i=0;i<names.size();++i) {
        if(saved[i])setenv(names[i].c_str(),saved[i]->c_str(),1);
        else unsetenv(names[i].c_str());
    }
    std::cout<<"TENSOR_LIMITS_PASS cooperative_k=64,128_rejected oversized_scratch_rejected=1\n";
}

namespace mx = mlx::core;
using tqllm::mlx_backend::tq4_matmul;

static void test_weight_dispatch() {
    using namespace tqllm::mlx_backend;
    tqllm::GgufTensor t;
    t.type = tqllm::GGML_BF16;
    t.ne = {3, 2};
    std::vector<uint16_t> bits = {0x3f80, 0xc000, 0x3f00, 0x4080, 0x4000, 0xbf80};
    std::span<const uint8_t> raw(reinterpret_cast<const uint8_t*>(bits.data()), bits.size()*2);
    auto weights = load_matrix(t, raw);
    float data[] = {1, 2, 3, -1, 0, 1};
    auto out = linear(weights, mx::array(data, {2, 3}));
    mx::eval(out);
    const float expected[] = {-1.5f, 5.0f, -0.5f, -5.0f};
    for (int i=0;i<4;++i)
        if (out.data<float>()[i] != expected[i])
            throw std::runtime_error("GGUF dense shape or BF16 dispatch failed");
    for (int bad=0;bad<4;++bad) {
        bool rejected=false;
        try {
            auto invalid=t;
            if (bad==0) invalid.ne={3};
            if (bad==1) invalid.type=tqllm::GGML_Q4_K;
            if (bad==2) invalid.ne={0,2};
            load_matrix(invalid,bad==3?raw.subspan(1):raw);
        } catch (const std::invalid_argument&) {rejected=true;}
        if (!rejected) throw std::runtime_error("invalid GGUF matrix accepted");
    }
    // Physical TQ4 scales select the integer codebook and rotate inputs exactly once.
    t.type=tqllm::GGML_TQ4;t.ne={128,1};
    std::vector<uint8_t> packed(66,0);
    packed[0]=0;packed[1]=0x1c; // scale 1/256
    auto matrix=load_matrix(t,packed);
    std::vector<float> x(128,0);x[0]=1;
    auto y=linear(matrix,mx::array(x.data(),{1,128}));
    auto rotated=tq4_rotate(mx::array(x.data(),{1,128}));
    mx::eval(y,rotated);
    double sum=0;
    for(int j=0;j<128;++j) sum+=rotated.data<float>()[j]*6.0/256.0;
    if(std::abs(y.item<float>()-sum)>1e-6)
        throw std::runtime_error("GGUF TQ4 dispatch rotation or codebook units failed");
    // Long FP32 activations expose reduced-precision batched GEMM defaults.
    t.type=tqllm::GGML_BF16;t.ne={5120,2};
    std::vector<uint16_t> bf16(10240);
    std::vector<float> dense(10240),input(8*5120);
    for(size_t i=0;i<bf16.size();++i) {
        bf16[i]=uint16_t(std::bit_cast<uint32_t>(std::sin(float(i)*0.071f)*0.1f)>>16);
        dense[i]=std::bit_cast<float>(uint32_t(bf16[i])<<16);
    }
    for(size_t i=0;i<input.size();++i) input[i]=std::sin(float(i)*0.037f)*0.1f;
    matrix=load_matrix(t,{reinterpret_cast<const uint8_t*>(bf16.data()),bf16.size()*2});
    y=linear(matrix,mx::array(input.data(),{8,5120}));mx::eval(y);
    for(int a=0;a<8;++a) for(int r=0;r<2;++r) {
        double expected=0;
        for(int j=0;j<5120;++j) expected+=double(input[a*5120+j])*dense[r*5120+j];
        if(std::abs(y.data<float>()[a*2+r]-expected)>1e-5)
            throw std::runtime_error("batched dense FP32 precision failed; strict tests require MLX_ENABLE_TF32=0");
    }
}

static void test_q8() {
    constexpr int rows=7,k=256;
    std::vector<uint8_t> raw(rows*(k/32)*34);
    std::vector<float> weights(rows*k);
    for(int r=0;r<rows;++r)
        for(int b=0;b<k/32;++b) {
            auto* p=raw.data()+(r*(k/32)+b)*34;
            p[0]=0; p[1]=(b&1)?0x9c:0x1c; // signed 1/256
            for(int j=0;j<32;++j) {
                const int q=(r*17+b*7+j*11)%256-128;
                p[2+j]=uint8_t(int8_t(q));
                weights[r*k+b*32+j]=q*((b&1)?-1.0f:1.0f)/256.0f;
            }
        }
    auto packed=tqllm::mlx_backend::load_q8_blocks(raw,rows,k);
    for(int m:{1,8,64}) {
        std::vector<float> x(m*k);
        for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*0.05f);
        auto out=tqllm::mlx_backend::q8_matmul(packed,mx::array(x.data(),{m,k}));
        mx::eval(out);
        for(int a=0;a<m;++a) for(int r=0;r<rows;++r) {
            double expected=0;
            for(int j=0;j<k;++j) expected+=double(weights[r*k+j])*x[a*k+j];
            if(std::abs(out.data<float>()[a*rows+r]-expected)>1e-4)
                throw std::runtime_error("Q8_0 signed codes or scale failed");
        }
    }
}

static void test_q4_embedding() {
    constexpr int rows=7,k=512;
    std::vector<uint8_t> raw(rows*(k/256)*144);
    std::vector<float> expected(rows*k);
    for(int r=0;r<rows;++r) for(int b=0;b<k/256;++b) {
        auto* p=raw.data()+(r*(k/256)+b)*144;
        p[0]=0;p[1]=0x14;p[2]=0;p[3]=0x18; // d=1/1024, min=1/512
        int scales[8],mins[8];
        for(int i=0;i<8;++i) {scales[i]=(i*9+r*3+7)%64;mins[i]=(i*11+b*5+3)%64;}
        for(int i=0;i<4;++i) {
            p[4+i]=uint8_t(scales[i]|((scales[i+4]>>4)<<6));
            p[8+i]=uint8_t(mins[i]|((mins[i+4]>>4)<<6));
            p[12+i]=uint8_t((scales[i+4]&15)|((mins[i+4]&15)<<4));
        }
        for(int j=0;j<256;++j) {
            const int sub=j/32,q=(r*7+b*3+j*11)%16;
            p[16+(sub/2)*32+j%32]|=uint8_t(q<<(4*(sub%2)));
            expected[r*k+b*256+j]=scales[sub]*q/1024.0f-mins[sub]/512.0f;
        }
    }
    auto packed=tqllm::mlx_backend::load_q4_embedding(raw,rows,k);
    const std::vector<uint32_t> ids={0,6,3,0};
    auto out=tqllm::mlx_backend::q4_embedding(packed,ids);
    mx::eval(out);
    for(size_t a=0;a<ids.size();++a) for(int j=0;j<k;++j)
        if(std::abs(out.data<float>()[a*k+j]-expected[ids[a]*k+j])>1e-6)
            throw std::runtime_error("Q4_K scale/minimum unpack or lookup failed");
    bool rejected=false;
    const uint32_t invalid=rows;
    try {tqllm::mlx_backend::q4_embedding(packed,{&invalid,1});}
    catch(const std::out_of_range&) {rejected=true;}
    if(!rejected) throw std::runtime_error("embedding out-of-range token accepted");
}

static void test_q6() {
    constexpr int rows = 17, k = 512;
    std::vector<uint8_t> raw(rows*(k/256)*210);
    std::vector<float> weights(rows*k);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < k/256; ++b) {
            auto* p = raw.data()+(r*(k/256)+b)*210;
            p[208] = 0; p[209] = 0x18; // FP16 1/512
            for (int g = 0; g < 16; ++g) p[192+g] = uint8_t(int8_t((g&1)?-g-1:g+1));
            for (int h = 0; h < 2; ++h)
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < 32; ++l) {
                        const int j = h*128+c*32+l;
                        const int q = (r*13+b*7+j*3)%64;
                        p[h*64+(c%2)*32+l] |= uint8_t((q&15) << (c/2*4));
                        p[128+h*32+l] |= uint8_t((q>>4) << (2*c));
                        const int scale = int8_t(p[192+h*8+c*2+l/16]);
                        weights[r*k+b*256+j] = (q-32)*float(scale)/512.0f;
                    }
        }
    auto packed = tqllm::mlx_backend::load_q6_blocks(raw, rows, k);
    for (int m : {1, 8, 64}) {
        std::vector<float> input(m*k);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i)*0.1f);
        auto out = tqllm::mlx_backend::q6_matmul(packed, mx::array(input.data(), {m,k}));
        mx::eval(out);
        for (int a = 0; a < m; ++a)
            for (int r = 0; r < rows; ++r) {
                double expected = 0;
                for (int j = 0; j < k; ++j) expected += double(weights[r*k+j])*input[a*k+j];
                if (std::abs(out.data<float>()[a*rows+r]-expected) > 3e-4)
                    throw std::runtime_error("Q6_K code, signed scale or arithmetic failed");
            }
    }
    raw.pop_back();
    bool rejected = false;
    try { tqllm::mlx_backend::load_q6_blocks(raw, rows, k); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Q6_K truncated payload accepted");
}

static void test_tq3() {
    constexpr int rows = 17, k = 256;
    constexpr int levels[] = {14, 45, 79, 127, -14, -45, -79, -127};
    std::vector<uint8_t> codes(rows * k), blocks(rows * k / 128 * 50);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) codes[r*k+j] = uint8_t((r*17+j*3+(j/32)*5)&7);
    for (int g = 0; g < rows*k/128; ++g) {
        blocks[g*50] = 0;
        blocks[g*50+1] = (g&1) ? 0x30 : 0x2c; // 0.125 or 0.0625
        for (int b = 0; b < 4; ++b) {
            uint32_t planes[3] = {};
            for (int j = 0; j < 32; ++j) {
                const int c = codes[g*128+b*32+j];
                planes[j/16] |= uint32_t(c&3) << ((j%8)*4+((j%16)/8)*2);
                planes[2] |= uint32_t(c>>2) << ((j%8)*4+j/8);
            }
            for (int p = 0; p < 3; ++p)
                for (int byte = 0; byte < 4; ++byte)
                    blocks[g*50+2+(b*3+p)*4+byte] = uint8_t(planes[p] >> (byte*8));
        }
    }
    const auto weights = tqllm::mlx_backend::load_tq3_blocks(blocks, rows, k);
    if(std::getenv("TQLLM_MLX_TENSOR_MEMORY")) {
        for(bool interleaved:{false,true})for(int m:{1,2,3,5,8,17}) {
            std::vector<float> input(size_t(m)*k);
            for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*0.09f);
            auto matrix=interleaved?tqllm::mlx_backend::tq3_interleave(weights):weights;
            auto x=mx::array(input.data(),{m,k});
            auto expected=tqllm::mlx_backend::tq3_matmul(weights,tqllm::mlx_backend::tq4_rotate(x));
            auto got=tqllm::mlx_backend::verify_tensorops(matrix,x,true);
            auto error=mx::sqrt(mx::divide(mx::sum(mx::square(mx::subtract(got,expected))),
                mx::maximum(mx::sum(mx::square(expected)),mx::array(1e-20f))));
            float value=error.item<float>();
            if(!std::isfinite(value) || value>1e-3f)
                throw std::runtime_error("packed TensorOps layout or tail differs from scalar projection");
        }
        std::cout<<"TENSOR_MEMORY_LAYOUT_PASS planar=1 interleaved=1 row_tail=17 column_tail=17\n";
    }
    for (int m : {1, 8, 64}) {
        std::vector<float> x(m*k);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i)*0.09f);
        auto out = tqllm::mlx_backend::tq3_matmul(weights, mx::array(x.data(), {m,k}));
        mx::eval(out);
        for (int a = 0; a < m; ++a)
            for (int r = 0; r < rows; ++r) {
                double expected = 0;
                for (int j = 0; j < k; ++j) {
                    const float s = ((r*k+j)/128)&1 ? 0.125f : 0.0625f;
                    expected += levels[codes[r*k+j]] * double(s) * x[a*k+j];
                }
                if (std::abs(out.data<float>()[a*rows+r]-expected) > 3e-4)
                    throw std::runtime_error("TQ3 plane decoder or scale arithmetic failed");
            }
    }
    blocks.pop_back();
    bool rejected = false;
    try { tqllm::mlx_backend::load_tq3_blocks(blocks, rows, k); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("TQ3 truncated payload accepted");
}

int main() {
    test_weight_dispatch();
    test_q4_embedding();
    test_q8();
    test_q6();
    test_tq3();
    constexpr int rows = 17, k = 256;
    const int levels[] = {6, 18, 31, 44, 58, 75, 96, 127, -6, -18, -31, -44, -58, -75, -96, -127};
    std::vector<uint8_t> codes(rows * k / 2);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(i * 37 + 11);
    std::vector<float> scales(rows * k / 128, 0.125f);
    auto c = mx::array(codes.data(), {rows, k / 2}, mx::uint8);
    auto s = mx::astype(mx::array(scales.data(), {rows, k / 128}), mx::float16);
    {
        std::vector<uint8_t> blocks(rows * (k / 128) * 66);
        for (int b = 0; b < rows * k / 128; ++b) {
            blocks[b * 66] = 0;
            blocks[b * 66 + 1] = 0x30; // FP16 0.125
            std::copy_n(codes.data() + b * 64, 64, blocks.data() + b * 66 + 2);
        }
        auto loaded = tqllm::mlx_backend::load_tq4_blocks(blocks, rows, k);
        auto same_codes = mx::all(mx::equal(loaded.codes, c));
        auto same_scales = mx::all(mx::equal(loaded.scales, s));
        if (!same_codes.item<bool>() || !same_scales.item<bool>()) return 5;
        std::vector<float> input(k);
        for (int j = 0; j < k; ++j) input[j] = std::sin(float(j) * 0.11f);
        auto physical = tq4_matmul(loaded, mx::array(input.data(), {1, k}));
        mx::eval(physical);
        for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int j = 0; j < k; ++j) {
                int code = (codes[r * k / 2 + j / 2] >> ((j & 1) * 4)) & 15;
                expected += double(levels[code]) * 0.125 * input[j];
            }
            if (std::abs(physical.data<float>()[r] - expected) > 1e-4) {
                std::cerr << "GGUF integer-codebook scale convention failed\n";
                return 7;
            }
        }
        blocks.pop_back();
        bool rejected = false;
        try { tqllm::mlx_backend::load_tq4_blocks(blocks, rows, k); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 6;
    }
    int cases = 0;
    {
        std::vector<float> input(2 * k);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.07f);
        auto rotated = tqllm::mlx_backend::tq4_rotate(mx::array(input.data(), {2, k}));
        mx::eval(rotated);
        constexpr uint32_t signs[] = {0xE46A0359, 0x42F85949, 0xA4C63BBB, 0x49F250A9};
        for (int b = 0; b < 4; ++b)
            for (unsigned i = 0; i < 128; ++i) {
                double expected = 0;
                for (unsigned j = 0; j < 128; ++j) {
                    const int sign = 1 - 2 * ((signs[j / 32] >> (j % 32)) & 1);
                    const int h = 1 - 2 * (std::popcount(i & j) & 1);
                    expected += sign * h * input[b * 128 + j];
                }
                if (std::abs(rotated.data<float>()[b * 128 + i] - expected / std::sqrt(128.0)) > 1e-4) {
                    std::cerr << "TQ4 signed rotation failed\n";
                    return 4;
                }
            }
    }
    for (int m : {1, 8, 64}) {
        std::vector<float> x(m * k), expected(m * rows);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.13f);
        for (int a = 0; a < m; ++a)
            for (int r = 0; r < rows; ++r) {
                double sum = 0;
                for (int j = 0; j < k; ++j) {
                    int code = (codes[r * k / 2 + j / 2] >> ((j & 1) * 4)) & 15;
                    sum += double(levels[code]) * (2.7326 / 127.0) * 0.125 * x[a * k + j];
                }
                expected[a * rows + r] = float(sum);
            }
        auto input = mx::array(x.data(), {m, k});
        for (int rps : {1, 2, 4, 8})
            for (int cols : {1, 4, 16})
            for (int simds : {1, 2, 4, 8}) {
                auto out = tq4_matmul(c, s, input, rps, cols, simds);
                mx::eval(out);
                float max_error = 0;
                for (size_t i = 0; i < expected.size(); ++i)
                    max_error = std::max(max_error, std::abs(out.data<float>()[i] - expected[i]));
                if (max_error > 1e-4f) {
                    std::cerr << "TQ4 arithmetic failed: m=" << m << " rps=" << rps << " cols=" << cols << " error=" << max_error << '\n';
                    return 1;
                }
                ++cases;
            }
    }
    auto input = mx::zeros({1, k});
    for (int invalid : {0, 3, 16}) {
        bool rejected = false;
        try { tq4_matmul(c, s, input, invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) return 2;
    }
    bool rejected = false;
    try { tq4_matmul(c, s, mx::zeros({1, k / 2})); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return 3;
    test_tensor_limits();
    std::cout << "MLX_TQ4_PASS " << cases << " arithmetic cases; invalid tiles and shapes rejected\n";
}

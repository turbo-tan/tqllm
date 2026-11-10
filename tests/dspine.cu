#include "dspine.hpp"
#include "dspine_kernels.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <random>

using bf16 = __nv_bfloat16;
template<class T> T* upload(const std::vector<T>& values) {
    T* ptr=nullptr;
    if(cudaMalloc(&ptr,values.size()*sizeof(T))!=cudaSuccess)throw std::runtime_error("cudaMalloc failed");
    if(cudaMemcpy(ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("cudaMemcpy failed");
    return ptr;
}
template<class T> std::vector<T> download(const T* ptr,size_t n) {
    std::vector<T> out(n);
    if(cudaMemcpy(out.data(),ptr,n*sizeof(T),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("cudaMemcpy failed");
    return out;
}
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    constexpr int H=32,C=8,R=8,B=3,N=2*B,V=32;
    std::mt19937 random(42);std::uniform_real_distribution<float> distribution(-.5f,.5f);
    auto values=[&](size_t n) {std::vector<bf16> out(n);for(auto& x:out)x=__float2bfloat16_rn(distribution(random));return out;};
    const auto x=values(N*H),read=values(C*H),write=values(H*R),book=values(V*C),packed=values(3*R*C);
    std::vector<float> bias(R);for(auto& x:bias)x=distribution(random);
    auto floats=[](const std::vector<bf16>& x) {std::vector<float> out;for(auto v:x)out.push_back(float(v));return out;};
    const auto xf=floats(x),rf=floats(read),wf=floats(write),bf=floats(book),pf=floats(packed);
    const tqllm::spec::DSpineConfig config{H,C,R,1,B,16,1e-6f};
    const tqllm::spec::DSpineWeights weights{rf.data(),wf.data(),bf.data(),pf.data(),pf.data()+R*C,pf.data()+2*R*C,bias.data()};
    const auto expected=tqllm::spec::dspine_write_reference(config,weights,xf,{1,2});
    std::vector<uint32_t> tokens(N,31);tokens[0]=1;tokens[B]=2;
    auto *dx=upload(x),*dr=upload(read),*dw=upload(write),*db=upload(book),*dp=upload(packed);
    auto *dn=upload(std::vector<bf16>(N*H)),*dc=upload(std::vector<bf16>(N*C));
    auto *dpacked=upload(std::vector<bf16>(N*3*R)),*dm=upload(std::vector<bf16>(N*R)),*dd=upload(std::vector<bf16>(N*H));
    auto *drms=upload(std::vector<float>(N)),*dbias=upload(bias);auto *dt=upload(tokens);
    cublasHandle_t handle;
    if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS initialization failed");
    auto mm=[&](const bf16* w,const bf16* x,bf16* y,int out,int in,int columns) {
        const float alpha=1,beta=0;
        if(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,columns,in,&alpha,w,CUDA_R_16BF,in,x,CUDA_R_16BF,in,
                        &beta,y,CUDA_R_16BF,out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("bridge GEMM failed");
    };
    tqllm_cuda::dspine_read_norm(dx,dn,drms,H,N,config.epsilon,nullptr);
    mm(dr,dn,dc,C,H,N);tqllm_cuda::dspine_codes(dc,db,dt,C,B,N,config.epsilon,nullptr);
    mm(dp,dc,dpacked,3*R,C,N);tqllm_cuda::dspine_message(dpacked,dbias,dm,R,B,N,nullptr);
    mm(dw,dm,dd,H,R,N);tqllm_cuda::dspine_add(dx,dd,drms,H,B,N,nullptr);
    const auto actual=download(dx,N*H);
    float error=0;for(size_t i=0;i<expected.size();++i)error=std::max(error,std::abs(float(actual[i])-expected[i])/(1+std::abs(expected[i])));
    bool finite=std::all_of(actual.begin(),actual.end(),[](bf16 v){return std::isfinite(float(v));});
    bool anchors=true;for(int row:{0,B})for(int h=0;h<H;++h)anchors&=float(actual[row*H+h])==float(x[row*H+h]);
    std::printf("DSpine bridge vs BF16 CPU reference: error=%g, anchors=%s\n",error,anchors?"unchanged":"FAIL");
    std::vector<uint32_t> candidates(N*16);for(size_t i=0;i<candidates.size();++i)candidates[i]=uint32_t((i+3)%V);
    auto *dids=upload(candidates);auto *dprev=upload(std::vector<bf16>(N*16*C));
    tqllm_cuda::dspine_predecessor_codes(dids,dt,db,dprev,C,B,N,nullptr);
    const auto prev=download(dprev,N*16*C);
    bool predecessors=true;
    for(int row=0;row<N;++row)for(int j=0;j<16;++j)for(int k=0;k<C;++k) {
        const auto token=row%B<=1?tokens[row-row%B]:candidates[(row-1)*16+j];
        predecessors&=float(prev[(row*16+j)*C+k])==float(__float2bfloat16_rn(std::sqrt(float(C))*float(book[token*C+k])));
    }
    std::printf("DSpine exact predecessor gather across two slots: %s\n",predecessors?"PASS":"FAIL");
    auto *dpv=upload(std::vector<bf16>(N*16*2*R)),*dmsg=upload(std::vector<bf16>(N*16*R));
    auto *ddelta=upload(std::vector<bf16>(N*16*H)),*drefined=upload(std::vector<bf16>(N*16*H)),*draw=upload(x);
    auto *dnorm=upload(std::vector<float>(H,1));
    mm(dp+R*C,dprev,dpv,2*R,C,N*16);
    tqllm_cuda::dspine_refine_message(dpacked,dpv,dbias,dmsg,R,N,nullptr);
    mm(dw,dmsg,ddelta,H,R,N*16);
    tqllm_cuda::dspine_refine_norm(draw,ddelta,drms,dnorm,drefined,H,N,config.epsilon,nullptr);
    const auto own=download(dpacked,N*3*R),pv=download(dpv,N*16*2*R),messages=download(dmsg,N*16*R);
    bool message_ok=true;
    for(int row=0;row<N*16;++row)for(int r=0;r<R;++r) {
        const float z=float(own[(row/16)*3*R+r])+float(pv[row*2*R+r])+bias[r];
        const float ref=float(__float2bfloat16_rn(float(pv[row*2*R+R+r])/(1+std::exp(-z))));
        message_ok&=std::abs(float(messages[row*R+r])-ref)<.008f*(1+std::abs(ref));
    }
    std::vector<int8_t> head(V*H);for(size_t i=0;i<head.size();++i)head[i]=int8_t(int(i%23)-11);
    const std::vector<__half> scales(V*(H/16),__float2half(.01f));
    auto *dhead=upload(head);auto *dscales=upload(scales);auto *dscores=upload(std::vector<float>(N*256));
    tqllm_cuda::dspine_score(dhead,dscales,drefined,dids,dscores,H,N,nullptr);
    const auto refined=download(drefined,N*16*H);const auto scores=download(dscores,N*256);
    float score_error=0;
    for(int row=0;row<N;++row)for(int parent=0;parent<16;++parent)for(int child=0;child<16;++child) {
        const auto tok=candidates[row*16+child];float ref=0;
        for(int h=0;h<H;++h)ref+=float(head[tok*H+h])*__half2float(scales[tok*(H/16)+h/16])*float(refined[(row*16+parent)*H+h]);
        const float got=scores[row*256+parent*16+child];finite&=std::isfinite(got);
        score_error=std::max(score_error,std::abs(ref-got));
    }
    const auto deltas=download(ddelta,N*16*H);const auto rms=download(drms,N);
    float norm_error=0;
    for(int row=0;row<N*16;++row) {
        float sum=0;std::vector<float> raw(H);
        for(int h=0;h<H;++h) {raw[h]=float(__float2bfloat16_rn(float(x[(row/16)*H+h])+rms[row/16]*float(deltas[row*H+h])));sum+=raw[h]*raw[h];}
        const float inv=1/std::sqrt(sum/H+config.epsilon);
        for(int h=0;h<H;++h)norm_error=std::max(norm_error,std::abs(float(refined[row*H+h])-float(__float2bfloat16_rn(raw[h]*inv))));
    }
    std::printf("DSpine refinement: message=%s, norm error=%g, score error=%g\n",message_ok?"PASS":"FAIL",norm_error,score_error);
    for(void* p:std::initializer_list<void*>{dx,dr,dw,db,dp,dn,dc,dpacked,dm,dd,drms,dbias,dt,dids,dprev,dpv,dmsg,ddelta,drefined,draw,dnorm,dhead,dscales,dscores})cudaFree(p);
    cublasDestroy(handle);
    return finite && error<.02f && anchors && predecessors && message_ok && norm_error<.02f && score_error<1e-4f?0:1;
}

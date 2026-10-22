#include "qwen2_kernels.cuh"
#include <cstdint>
#include <stdexcept>
#include <cfloat>
#include <cmath>
#include <math_constants.h>

namespace {
__global__ void gather_k(const __nv_bfloat16 * w, __nv_bfloat16 * y, int token, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)y[i]=w[int64_t(token)*n+i];
}
__global__ void rms_k(const __nv_bfloat16 * x,const __nv_bfloat16 * w,__nv_bfloat16 * y,int n,float eps) {
    __shared__ float s[256]; int t=threadIdx.x; float v=0;
    for(int i=t;i<n;i+=blockDim.x){float a=__bfloat162float(x[i]);v+=a*a;} s[t]=v;__syncthreads();
    for(int d=blockDim.x/2;d;d>>=1){if(t<d)s[t]+=s[t+d];__syncthreads();}
    float inv=rsqrtf(s[0]/n+eps);for(int i=t;i<n;i+=blockDim.x)y[i]=__float2bfloat16_rn(__bfloat162float(x[i])*inv*__bfloat162float(w[i]));
}
__global__ void bias_k(const __nv_bfloat16 * x,const __nv_bfloat16 * b,__nv_bfloat16 * y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__float2bfloat16_rn(__bfloat162float(x[i])+__bfloat162float(b[i]));}
__global__ void rope_k(__nv_bfloat16 * x,int heads,int dim,int pos,float theta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int half=dim/2,total=heads*dim;
    if(i>=total)return;
    int d=i%dim;
    if(d>=half)return;
    float a=__bfloat162float(x[i]),b=__bfloat162float(x[i+half]);
    float freq=powf(theta,-float(d)/half);
    float ang=pos*freq,c=cosf(ang),s=sinf(ang);
    x[i]=__float2bfloat16_rn(a*c-b*s);
    x[i+half]=__float2bfloat16_rn(a*s+b*c);
}
__global__ void cache_k(const __nv_bfloat16 * k,const __nv_bfloat16 * v,__nv_bfloat16 * kc,__nv_bfloat16 * vc,int pos,int kvh,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<kvh*dim){
        int64_t dst=int64_t(pos)*kvh*dim+i;
        kc[dst]=k[i];
        vc[dst]=v[i];
    }
}
__global__ void scores_k(const __nv_bfloat16 * q,const __nv_bfloat16 * kc,float * score,int seq,int ctx,int heads,int kvheads,int dim,float scale) {
    int h=blockIdx.x,p=blockIdx.y,t=threadIdx.x;
    if(h>=heads||p>=seq)return;
    int kh=h/(heads/kvheads);
    float v=0;
    for(int d=t;d<dim;d+=blockDim.x)v+=__bfloat162float(q[h*dim+d])*__bfloat162float(kc[(int64_t(p)*kvheads+kh)*dim+d]);
    __shared__ float s[64];
    s[t]=v;
    __syncthreads();
    for(int d=32;d;d>>=1){
        if(t<d)s[t]+=s[t+d];
        __syncthreads();
    }
    if(t==0)score[h*ctx+p]=s[0]*scale;
}
__global__ void softmax_k(float * x,int seq,int ctx,int heads) {
    int h=blockIdx.x,t=threadIdx.x;
    __shared__ float a[256],b[256];
    float m=-CUDART_INF_F;
    for(int i=t;i<seq;i+=blockDim.x)m=fmaxf(m,x[h*ctx+i]);
    a[t]=m;
    __syncthreads();
    for(int d=blockDim.x/2;d;d>>=1){
        if(t<d)a[t]=fmaxf(a[t],a[t+d]);
        __syncthreads();
    }
    float z=0;
    for(int i=t;i<seq;i+=blockDim.x){
        float e=expf(x[h*ctx+i]-a[0]);
        x[h*ctx+i]=e;
        z+=e;
    }
    b[t]=z;
    __syncthreads();
    for(int d=blockDim.x/2;d;d>>=1){
        if(t<d)b[t]+=b[t+d];
        __syncthreads();
    }
    for(int i=t;i<seq;i+=blockDim.x)x[h*ctx+i]/=b[0];
}
__global__ void attend_k(const float * a,const __nv_bfloat16 * vc,__nv_bfloat16 * y,int seq,int ctx,int heads,int kvheads,int dim) {
    int h=blockIdx.x,d=threadIdx.x;
    if(d>=dim)return;
    int kh=h/(heads/kvheads);
    float z=0;
    for(int p=0;p<seq;++p)z+=a[h*ctx+p]*__bfloat162float(vc[(int64_t(p)*kvheads+kh)*dim+d]);
    y[h*dim+d]=__float2bfloat16_rn(z);
}
__global__ void residual_k(__nv_bfloat16 * x,const __nv_bfloat16 * y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__float2bfloat16_rn(__bfloat162float(x[i])+__bfloat162float(y[i]));}
__global__ void swiglu_k(const __nv_bfloat16 * g,const __nv_bfloat16 * u,__nv_bfloat16 * y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        float a=__bfloat162float(g[i]);
        float s=1.f/(1.f+expf(-a));
        y[i]=__float2bfloat16_rn(a*s*__bfloat162float(u[i]));
    }
}
__global__ void argmax_k(const float * x,int n,uint32_t * out) {
    __shared__ float v[256];
    __shared__ uint32_t ix[256];
    int t=threadIdx.x;
    float best=-CUDART_INF_F;
    uint32_t bi=0xffffffffu;
    for(int i=t;i<n;i+=blockDim.x){
        float a=x[i];
        if(a>best||(a==best&&uint32_t(i)<bi)){
            best=a;
            bi=i;
        }
    }
    v[t]=best;
    ix[t]=bi;
    __syncthreads();
    for(int d=blockDim.x/2;d;d>>=1){
        if(t<d&&(v[t+d]>v[t]||(v[t+d]==v[t]&&ix[t+d]<ix[t]))){
            v[t]=v[t+d];
            ix[t]=ix[t+d];
        }
        __syncthreads();
    }
    if(t==0)*out=ix[0];
}
}
namespace tqllm_cuda {
void gather(const __nv_bfloat16*w,__nv_bfloat16*y,int tok,int n,cudaStream_t s){gather_k<<<(n+255)/256,256,0,s>>>(w,y,tok,n);}
void rms(const __nv_bfloat16*x,const __nv_bfloat16*w,__nv_bfloat16*y,int n,float e,cudaStream_t s){rms_k<<<1,256,0,s>>>(x,w,y,n,e);}
void bias(const __nv_bfloat16*x,const __nv_bfloat16*b,__nv_bfloat16*y,int n,cudaStream_t s){bias_k<<<(n+255)/256,256,0,s>>>(x,b,y,n);}
void rope(__nv_bfloat16*x,int h,int d,int p,float t,cudaStream_t s){rope_k<<<(h*d+255)/256,256,0,s>>>(x,h,d,p,t);}
void cache_write(const __nv_bfloat16*k,const __nv_bfloat16*v,__nv_bfloat16*kc,__nv_bfloat16*vc,int p,int h,int d,cudaStream_t s){cache_k<<<(h*d+255)/256,256,0,s>>>(k,v,kc,vc,p,h,d);}
void scores(const __nv_bfloat16*q,const __nv_bfloat16*k,float*x,int seq,int ctx,int h,int kh,int d,float scale,cudaStream_t s){scores_k<<<dim3(h,seq),64,0,s>>>(q,k,x,seq,ctx,h,kh,d,scale);}
void softmax(float*x,int seq,int ctx,int h,cudaStream_t s){softmax_k<<<h,256,0,s>>>(x,seq,ctx,h);}
void attend(const float*a,const __nv_bfloat16*v,__nv_bfloat16*y,int seq,int ctx,int h,int kh,int d,cudaStream_t s){attend_k<<<h,d,0,s>>>(a,v,y,seq,ctx,h,kh,d);}
void residual(__nv_bfloat16*x,const __nv_bfloat16*y,int n,cudaStream_t s){residual_k<<<(n+255)/256,256,0,s>>>(x,y,n);}
void swiglu(const __nv_bfloat16*g,const __nv_bfloat16*u,__nv_bfloat16*y,int n,cudaStream_t s){swiglu_k<<<(n+255)/256,256,0,s>>>(g,u,y,n);}
void argmax(const float*x,int n,uint32_t*y,cudaStream_t s){argmax_k<<<1,256,0,s>>>(x,n,y);}
}

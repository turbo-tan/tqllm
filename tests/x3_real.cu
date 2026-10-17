// tqLLM side of the G2 real-matrix bit-match (diagnostic; not part of ctest - it needs the
// llama-paw fork to produce the other side). Reads $X3_GGUF, dumps one matrix per K present
// (blk.0.ssm_out K3.0, blk.0.attn_gate K3.5): <name>.{trellis,suh,svh,x} and tqLLM's <name>.y_tq_n{1,4}
// (f32, [nt][rows]). The fork side is a small ggml driver that reads the same files and runs
// ggml_paw_x3_mm with GGML_PAW_X3_GEMV=2 (x3v; the fork defaults to the GEMM path), dumping
// <name>.y_lp_n{1,4} (f32, ne={rows,nt} -> element (r,t) at t*rows+r). Result 2026-10-16: sq n=1 and
// x3v n=4 are bit-identical (0/5120 and 0/6144, both K) on the real PAW-27B-X3.1 matrices.
//
// build: nvcc -std=c++17 -arch=sm_86 -Isrc -Isrc/formats -Ikernels/formats/x3 tests/x3_real.cu \
//          build/libtqllm.a -L/usr/local/cuda-12.6/lib64 -lcudart -lcublas -lcublasLt -lcusolver -lcuda -ldl
#include "gguf.hpp"
#include "formats/x3_record.hpp"
#include "x3.cuh"
#include "x3_codecs.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace tqllm;
static bool wfile(const std::string&p,const void*d,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f)return false;fwrite(d,1,n,f);fclose(f);return true;}
int main(int argc,char**argv){
  GgufFile f=gguf_open(argv[1]);
  const char* names[]={"blk.0.ssm_out.weight","blk.0.attn_gate.weight"};
  const int NS=2;
  for(int s=0;s<NS;++s){
    const GgufTensor& t=f.tensors.at(names[s]);
    int cols=t.ne[0], rows=t.ne[1];
    std::vector<char> data=f.read(t);
    X3Record r; std::memcpy(&r,data.data(),sizeof r);
    std::string base=std::string("/tmp/opencode/x3real/")+names[s];
    const char* body=data.data()+sizeof(X3Record);
    size_t tb=(size_t)(rows/16)*(cols/16)*r.words*2;
    wfile(base+".trellis",body,tb);
    wfile(base+".suh",body+tb,(size_t)cols*2);
    wfile(base+".svh",body+tb+(size_t)cols*2,(size_t)rows*2);
    // fixed input [cols][nt], nt=4, f16
    const int nt=4; std::vector<half> x((size_t)cols*nt);
    unsigned st=7; for(auto&v:x){st=st*1664525u+1013904223u; v=__float2half(((int(st>>10)%2001)-1000)/8000.0f);}
    wfile(base+".x",x.data(),x.size()*2);
    // A [nt][cols] f32
    std::vector<float> A((size_t)nt*cols);
    for(int j=0;j<nt;++j) for(int i=0;i<cols;++i) A[(size_t)j*cols+i]=__half2float(x[(size_t)j*cols+i]);
    const x3::X3Codec* c=x3_codec_lookup(r.codec,int(r.words));
    if(!c){printf("no codec\n");return 1;}
    // device upload
    uint16_t* dT; half* dSuh; half* dSvh; float* dA; float* dC; half* dHad; int* dLk;
    cudaMalloc(&dT,tb); cudaMemcpy(dT,body,tb,cudaMemcpyHostToDevice);
    cudaMalloc(&dSuh,(size_t)cols*2); cudaMemcpy(dSuh,body+tb,(size_t)cols*2,cudaMemcpyHostToDevice);
    cudaMalloc(&dSvh,(size_t)rows*2); cudaMemcpy(dSvh,body+tb+(size_t)cols*2,(size_t)rows*2,cudaMemcpyHostToDevice);
    cudaMalloc(&dA,(size_t)nt*cols*4); cudaMemcpy(dA,A.data(),(size_t)nt*cols*4,cudaMemcpyHostToDevice);
    cudaMalloc(&dC,(size_t)nt*rows*4); cudaMalloc(&dHad,(size_t)nt*cols*2);
    size_t lb=c->ws_bytes(cols,rows); dLk=nullptr; if(lb){cudaMalloc(&dLk,lb);cudaMemset(dLk,0,lb);}
    for(int n:{1,4}){
      c->gemv(dA,dT,dC,n,cols,rows,dSuh,dHad,dSvh,dLk,0);
      cudaError_t e=cudaDeviceSynchronize();
      if(e){printf("%s n=%d launch %s\n",names[s],n,cudaGetErrorString(e));return 1;}
      std::vector<float> C((size_t)n*rows); cudaMemcpy(C.data(),dC,(size_t)n*rows*4,cudaMemcpyDeviceToHost);
      char buf[256]; snprintf(buf,sizeof buf,"%s.y_tq_n%d",base.c_str(),n);
      wfile(buf,C.data(),C.size()*4);
    }
    printf("%s: rows=%d cols=%d words=%d ok\n",names[s],rows,cols,r.words);
    cudaFree(dT);cudaFree(dSuh);cudaFree(dSvh);cudaFree(dA);cudaFree(dC);cudaFree(dHad);if(dLk)cudaFree(dLk);
  }
  return 0;
}

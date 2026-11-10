// ane_plan: the split decision and slicing come from the measured per-layer costs for every prompt length.
#include "ane_plan.hpp"
#include <cstdio>
#include <vector>
using namespace tqllm::mlx_backend;
static int fails=0;
static void check(bool ok,const char* what,int m) {if(!ok){std::printf("FAIL %s (m=%d)\n",what,m);++fails;}}
int main() {
    const std::vector<int> sizes={512,768,1024,1280,1536,1792,2048};
    // M5 Pro magnitudes (Kiro 10-21 ablation at 1536 rows, C 13824, unit test at 2048 rows)
    AneCost c;
    c.k1=0.0325;c.d1=3.0;c.u=0.0154;c.u_split=0.0154;c.f=0.0166;c.d=3.0;c.s=0.0055;c.p=0.00085;c.a=0.0207;c.o=1.1;c.r=3.9;
    check(!ane_plan(AneCost{},4096,sizes).split,"no constants: no split",4096);
    check(!ane_plan(c,4096,{}).split,"no ANE functions: no split",4096);
    for(int m:{1,8,64,200}) check(!ane_plan(c,m,sizes).split,"fixed costs dominate a short prompt",m);
    for(int m:{1536,1910,4096,7884,16384,65536}) {
        auto p=ane_plan(c,m,sizes);
        check(p.split,"split pays on a long prompt",m);
        check(p.t_split<p.t_gpu,"split chosen only when predicted faster",m);
    }
    // a 1.9K prompt: overlapping slices beat one (one slice cannot hide its ANE share behind a mixer)
    {auto p=ane_plan(c,1910,sizes);check(p.slices>=2,"1910 tokens: more than one slice",1910);}
    // an ANE that is slower than the whole GPU FFN never splits
    {AneCost slow=c;slow.a=1.0;for(int m:{512,2048,8192,65536})check(!ane_plan(slow,m,sizes).split,"slow ANE: GPU only",m);}
    // every length: slices cover the prompt, each fits the largest function, no empty slice
    for(int m=1;m<=70000;m+=(m<4096?1:97)) {
        auto p=ane_plan(c,m,sizes);
        if(!p.split){check(p.slices==1 && p.rows==0 && !p.prefetch,"GPU plan is one pass",m);continue;}
        check(p.rows<=sizes.back(),"slice fits the largest ANE function",m);
        check(p.rows*p.slices>=m && (p.slices-1)*p.rows<m,"balanced slices cover the prompt",m);
        auto sc=ane_split_ms(c,p.slices,p.rows,ane_fit_rows(sizes,p.rows));
        check(sc.ms==p.t_split && sc.prefetch==p.prefetch,"plan reports its own cost",m);
    }
    // prefetch only when the GPU would wait for the ANE at least as long as the requantize
    {AneCost fast_gpu=c;fast_gpu.u_split=0.002;fast_gpu.s=0.001;
     check(ane_plan(fast_gpu,8192,sizes).prefetch,"ANE-bound slices prefetch",8192);
     AneCost fast_ane=c;fast_ane.a=0.005;
     check(!ane_plan(fast_ane,8192,sizes).prefetch,"GPU-bound slices do not prefetch",8192);}
    // aligned slices: whole tiles but the last, still covering the prompt
    for(int m:{1910,7884,4999,65536}) {
        auto p=ane_plan(c,m,sizes,64,64);
        check(p.split && p.rows%64==0 && p.rows*p.slices>=m && (p.slices-1)*p.rows<m,"aligned balanced slices",m);
    }
    // below dequant_rows the GPU path pays no dequantize: a short prompt with a large d stays on the GPU
    {AneCost big_d=c;big_d.d=21.0;big_d.d1=21.0;check(!ane_plan(big_d,49,sizes,64,64,2048).split,"no dequantize below 64 rows: GPU only",49);
     check(ane_plan(big_d,49,sizes,64,0,2048).split,"charging d at 49 rows flips it (the j904 TQ4_L bug)",49);}
    // one pass up to group rows is timed as the engine runs it (k1, d1), layer-major above (d + m (u + f))
    {check(ane_gpu_ms(c,1000,64,2048)==c.d1+1000*c.k1,"one-pass GPU cost",1000);
     check(ane_gpu_ms(c,5000,64,2048)==c.d+5000*(c.u+c.f),"layer-major GPU cost",5000);
     // j906 TQ4_L 511 rows: GPU 420 tok/s (19.0 ms/layer) beat the split (289): a true one-pass cost keeps it on the GPU
     AneCost t4;t4.k1=0.0296;t4.d1=3.9;t4.u=0.0108;t4.u_split=0.0110;t4.f=0.0188;t4.d=21.0;t4.s=0.00284;t4.g0=5.92;
     t4.p=0.00011;t4.a=0.02106;t4.o=2.69;t4.r=3.64;
     check(!ane_plan(t4,511,sizes,128,64,2048).split,"TQ4_L 511 rows: GPU only",511);}
    std::puts(fails?"ANE_PLAN_FAIL":"ANE_PLAN_PASS");
    return fails?1:0;
}
